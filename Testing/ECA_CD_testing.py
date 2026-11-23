# -*- coding: utf-8 -*-
"""
ECA-CD (Extensive Cancellation Algorithm by Carrier and Doppler) for passive radar DSI / clutter suppression.

Based on:
    - C. Schüpbach, S. Paine, D. O'Hagan, "Efficient Direct Signal Cancellation for FM-based Passive Radar",
      IEEE RadarConf 2020 (EDSCFMPR.pdf), Section III-C.
    - J. Schwark, D. Cristallini, "Advanced interference suppression for DVB-T passive radar"
      (the original ECA-CD paper, reference [9] in the above).

How it works (short version):
    1. Both channels are cut into batches of L samples, exactly like the segmented cross-correlation in
       DSP_functions.passive_processing (L = Fs / seg_s), so the batch grid lines up with the range-Doppler
       processing.
    2. Each batch is FFT'd. In the frequency domain a delay is just a phase ramp, so a single complex weight per
       carrier k (frequency bin) can model the whole multipath channel (every delay at once). That is ECA-C.
    3. ECA-CD adds slow-time Doppler: for each carrier k the clutter subspace over the M batches of one CPI is
           X_k = [ D_{-P} Q_k, ..., Q_k, ..., D_{+P} Q_k ]      (M x (2P+1))
       where Q_k is the k-th carrier of the reference over the M batches and D_p = diag(exp(j 2 pi m p dv / M))
       is a slow-time Doppler shift of p * dv bins (dv = doppler_spacing).
    4. Per carrier, a tiny (2P+1) x (2P+1) least-squares problem is solved: w_k = (X_k^H X_k)^-1 X_k^H S_k,
       and X_k w_k is subtracted from the surveillance carrier. This is vectorised over all carriers.
    5. The synthesised clutter is IFFT'd back to the time domain and subtracted, so the output is a normal 1D complex surveillance
       array that can go straight into DSP.passive_processing, just like the output of
       clutterCancellation.Wiener_SMI_Batch_Process.

Implementation notes:
    - Batch edges (overlap-save): each batch FFT also includes the K = max_delay_samples samples before the batch, and
      only the last L samples of the synthesised clutter are subtracted. Those are exact (no circular wrap-around)
      for clutter delayed by up to K samples. Set max_delay_samples=0 for the plain batches described in the paper.
    - Iterations: the first K samples of each batch window do not fit the circular model, which biases the estimate a
      little for delayed clutter. Re-running the estimate on the residual (iterations=2 or 3) removes most of this.
      In synthetic testing (15 MHz, 1 ms batches, DSI 60 dB over noise plus multipath out to 1500 samples) one pass
      left ~10 dB of residual over the noise floor and three passes left < 2 dB (paper version: ~22 dB,
      Wiener_SMI_Batch_Process with 2048 taps / 0.1 s blocks: ~19 dB, as it cannot follow the Doppler-shifted
      multipath). Target loss for a 37 Hz target was ~0.1 dB.
    - Because one weight per carrier models ANY delay, ECA-CD cancels the whole range profile inside the notch
      (zero Doppler +/- P bins). Targets with near-zero Doppler (e.g. a hovering drone/helicopter body line) will be
      suppressed at every range; rotor/blade micro-Doppler outside the notch is untouched. Keep P small.
    - Each carrier has M observations (batches) and 2P+1 unknowns, so M should be much greater than 2P+1, otherwise
      the filter starts eating targets as well.
    - Carriers with no signal (DVB-T guard bands, outside the occupied bandwidth) are handled by diagonal loading.

This file does NOT modify any existing processing code. Swap it in for the Wiener filter by replacing
    CC.Wiener_SMI_Batch_Process(ref_iq, surv_iq, filter_taps, d_Fs, block_duration, smoothing_weight)
with
    ECA_CD_Batch_Process(ref_iq, surv_iq, d_Fs, seg_s, cpi_time_sec, doppler_bins)
Both return (clean_surv, weights).
"""
import gc
import os
import time

import numpy as np
import scipy.fft as sfft


# =============================================================================
# ECA-CD FUNCTIONS
# =============================================================================

def velocity_to_doppler_bins(max_velocity, wavelength, cpi_time_sec, doppler_spacing=0.5):
    """
    Converts a (bistatic) clutter velocity to cover into the one-sided number of ECA-CD Doppler bins P.

    :param max_velocity: Maximum clutter velocity to cancel in m/s (bistatic range-rate)
    :param wavelength: Carrier wavelength in m
    :param cpi_time_sec: ECA-CD estimation interval in seconds (Doppler resolution = 1 / cpi_time_sec)
    :param doppler_spacing: Spacing between Doppler shifts in Doppler bins (1.0 = one bin of 1/cpi_time_sec Hz)
    :return: P (int), the filter will cancel Doppler shifts -P..P (2P+1 columns per carrier)
    """
    max_doppler_hz = max_velocity / wavelength  # bistatic Doppler = range-rate / wavelength
    bin_hz = doppler_spacing / cpi_time_sec
    return int(np.ceil(max_doppler_hz / bin_hz))


def _get_cpi_segments(x, start, num_batches, L, seg_len):
    """
    Builds the (num_batches x seg_len) matrix of batches without copying the whole capture.
    Batch m covers x[start + m * L : start + m * L + seg_len] (seg_len > L means consecutive batches overlap).
    Samples before the start or past the end of the capture are zero.
    """
    end = start + (num_batches - 1) * L + seg_len
    chunk = np.asarray(x[max(start, 0):min(end, x.shape[0])], dtype=np.complex64)
    pre = max(0, -start)
    post = (end - start) - pre - chunk.shape[0]
    if pre or post:
        chunk = np.concatenate((np.zeros(pre, dtype=np.complex64), chunk, np.zeros(post, dtype=np.complex64)))

    idx = np.arange(num_batches)[:, None] * L + np.arange(seg_len)[None, :]
    return chunk[idx]


def _batch_weights(REF, SURV, D, E, p, diagonal_loading):
    """
    Per carrier least squares, vectorised over all carriers k:
        R_k[p, q] = sum_m |REF[m,k]|^2 exp(j 2 pi m (q - p) dv / M)  -> Hermitian Toeplitz in (q - p)
        r_k[p]    = sum_m conj(D[m,p]) conj(REF[m,k]) SURV[m,k]
        w_k       = R_k^-1 r_k
    Returns W ((2P+1) x Nfft).
    """
    n_dop = D.shape[1]
    A = (REF.real ** 2 + REF.imag ** 2).astype(np.float32)  # M x Nfft
    c_pos = E @ A  # (2P+1) x Nfft, lags 0..2P
    del A
    r = np.conj(D).T @ (np.conj(REF) * SURV)  # n_dop x Nfft

    # Assemble the Toeplitz matrices: R[k, p, q] = c[q - p], with c[-d] = conj(c[d])
    lag = p[None, :] - p[:, None]  # q - p
    R = np.where(lag[None, :, :] >= 0,
                 c_pos.T[:, np.abs(lag)],
                 np.conj(c_pos.T[:, np.abs(lag)]))  # Nfft x n_dop x n_dop

    # Diagonal loading: relative to each carrier's power plus a small floor for empty (guard band) carriers
    c0 = c_pos[0].real
    load = diagonal_loading * c0 + diagonal_loading * np.mean(c0) + np.finfo(np.float32).tiny
    R = R + load[:, None, None] * np.eye(n_dop)[None, :, :]

    W = np.linalg.solve(R.astype(np.complex128), r.T[:, :, None].astype(np.complex128))[:, :, 0].T
    return W.astype(np.complex64)


def ECA_CD_CPI(ref_segs, surv_series, doppler_bins=2, doppler_spacing=0.5, iterations=3, diagonal_loading=1e-6,
               workers=-1):
    """
    Runs ECA-CD on one CPI of batched data.

    :param ref_segs: (M x Nfft) complex reference batches. Batch m is ref[cpi_start + m*L - K : cpi_start + (m+1)*L],
                     i.e. each batch also holds the K samples before it (overlap-save, Nfft = L + K).
    :param surv_series: 1D complex surveillance, K + M * L samples: the K samples before the CPI (already cleaned, or
                        zeros at the start of the capture) followed by the raw surveillance for the CPI.
    :param doppler_bins: P, one-sided number of Doppler shifts to cancel (0 = ECA-C, zero-Doppler only)
    :param doppler_spacing: Spacing of the Doppler shifts in Doppler bins (fractions of 1/CPI)
    :param iterations: Number of estimate-and-subtract passes. Pass 1 is ECA-CD as in the paper. Passes 2+ re-estimate
                       on the residual, which removes the error the batch edges leave on delayed clutter.
    :param diagonal_loading: Relative diagonal loading of each carrier's correlation matrix
    :param workers: scipy.fft worker threads (-1 = all cores)
    :return: clean (M * L,) complex64 cleaned surveillance for this CPI, and
             weights: list of each pass's weights ((2P+1) x Nfft). weights[0] is the per-carrier channel estimate.
    """
    M, Nfft = ref_segs.shape
    L = (surv_series.shape[0] - Nfft) // (M - 1)  # surv_series is K + M * L = (M - 1) * L + Nfft long (M >= 2)
    K = Nfft - L

    REF = sfft.fft(ref_segs, axis=1, workers=workers).astype(np.complex64)

    # Slow-time Doppler steering matrix D[m, p] = exp(j 2 pi m p dv / M), p = -P..P
    m = np.arange(M)
    p = np.arange(-doppler_bins, doppler_bins + 1)
    D = np.exp(1j * 2 * np.pi * np.outer(m, p) * doppler_spacing / M).astype(np.complex64)  # M x n_dop
    d = np.arange(0, 2 * doppler_bins + 1)
    E = np.exp(1j * 2 * np.pi * np.outer(d, m) * doppler_spacing / M).astype(np.complex64)  # (2P+1) x M

    series = np.array(surv_series, dtype=np.complex64, copy=True)
    idx = np.arange(M)[:, None] * L + np.arange(Nfft)[None, :]
    weights = []

    for _ in range(max(1, iterations)):
        SURV = sfft.fft(series[idx], axis=1, workers=workers).astype(np.complex64)
        W = _batch_weights(REF, SURV, D, E, p, diagonal_loading)
        del SURV
        weights.append(W)

        # Synthesise the clutter, X_k w_k = REF[:, k] * (D @ W)[:, k], and go back to the time domain.
        # Overlap-save: the first K samples of each batch are wrapped (circular convolution), the last L are exact
        # for any clutter delay up to K, so only those are subtracted.
        model = sfft.ifft(REF * (D @ W), axis=1, workers=workers)[:, K:].astype(np.complex64)  # M x L
        series[K:] -= model.ravel()
        del model

    return series[K:], weights


def ECA_CD_Batch_Process(ref_ch, surv_ch, fs, seg_s, cpi_time_sec=0.5, doppler_bins=2, doppler_spacing=0.5,
                         max_delay_samples=2048, iterations=3, diagonal_loading=1e-6, store_weights=True,
                         verbose=True):
    """
    Description:
    ------------
        ECA-CD DSI / clutter cancellation for a full capture, processed CPI by CPI so it fits in RAM.
        Drop-in comparable with clutterCancellation.Wiener_SMI_Batch_Process: same 1D complex inputs, returns
        (clean_surv, weights) with clean_surv the same length as the input.

    Parameters:
    -----------
        :param ref_ch: Full 1D complex reference array (a np.load mmap_mode='r' array is fine)
        :param surv_ch: Full 1D complex surveillance array
        :param fs: Sampling frequency in Hz (e.g. d_Fs)
        :param seg_s: Batches (segments) per second. Use the SAME value as passive_processing so the batch grid
                      lines up. Batch length L = int(fs / seg_s); seg_s sets the unambiguous Doppler (+/- seg_s/2).
        :param cpi_time_sec: Time over which one set of ECA-CD weights is estimated (Doppler resolution
                             1/cpi_time_sec). Number of batches per CPI M = round(cpi_time_sec * seg_s).
        :param doppler_bins: P, cancel Doppler shifts -P..P (0 = zero-Doppler only / ECA-C).
                             See velocity_to_doppler_bins().
        :param doppler_spacing: Spacing between Doppler shifts in bins (1.0 = 1/cpi_time_sec Hz). Clutter that sits
                                between the shifts is only partly cancelled, so 0.5 with twice the P is more robust.
        :param max_delay_samples: Overlap-save length K: clutter delayed by up to K samples is subtracted without
                                  batch-edge wrap-around. Must be <= L. 0 = plain circular batches as in the paper.
        :param iterations: Estimate-and-subtract passes per CPI. 1 = ECA-CD as in the paper. 2-3 = re-estimate on
                           the residual, which takes out what the batch edges leave behind on delayed clutter
                           (about 10 dB deeper in testing). Each pass costs about the same as the first.
        :param diagonal_loading: Relative diagonal loading for each carrier's (2P+1)x(2P+1) matrix.
        :param store_weights: Keep the per-CPI pass 1 weights ((num_cpi, 2P+1, Nfft) array). Set False to save RAM.
        :param verbose: Print progress and per-CPI suppression.

    Return values:
    --------------
        :return clean_surv: Filtered 1D complex64 surveillance array (same length as input). Trailing samples
                            that do not fill a whole batch are passed through unfiltered.
        :return weights: np.array of per-CPI pass 1 weights (num_cpi x (2P+1) x Nfft), or empty array. weights[i, P] is
                         the zero-Doppler channel frequency response for CPI i; its IFFT is the clutter impulse
                         response (range profile of what was cancelled).
    """
    if ref_ch.shape[0] != surv_ch.shape[0]:
        print("ERROR: Input arrays must be the same size.")
        return None

    N_total = ref_ch.shape[0]
    L = int(fs / seg_s)
    if max_delay_samples > L:
        print(f"ERROR: max_delay_samples ({max_delay_samples}) must be <= batch length L ({L}).")
        return None
    M_cpi = int(round(cpi_time_sec * seg_s))
    Nfft = sfft.next_fast_len(L + max_delay_samples) if max_delay_samples > 0 else L
    if Nfft - L > L:
        Nfft = L + max_delay_samples
    K = Nfft - L
    n_batches_total = N_total // L
    num_cpis = int(np.ceil(n_batches_total / M_cpi))
    n_dop = 2 * doppler_bins + 1
    min_batches = 4 * n_dop  # below this the per-carrier LS starts removing targets as well

    if verbose:
        print("--- Starting ECA-CD Processing ---")
        print(f"Total Samples: {N_total} | Batch length L: {L} samples ({1e3 * L / fs:.3f} ms)")
        print(f"FFT size: {Nfft} (max delay {K} samples) | Batches per CPI: {M_cpi} ({cpi_time_sec} s)")
        print(f"Doppler shifts: {n_dop} (+/- {doppler_bins} x {doppler_spacing / cpi_time_sec:.3f} Hz)")
        print(f"Total CPIs: {num_cpis}")
        if M_cpi < 10 * n_dop:
            print(f"WARNING: only {M_cpi} batches per CPI for {n_dop} Doppler columns, expect target loss.")

    clean_surv = np.array(surv_ch, dtype=np.complex64, copy=True)  # unfiltered tail is passed through
    all_weights = []
    start_time = time.time()

    for i in range(num_cpis):
        first_batch = i * M_cpi
        M = min(M_cpi, n_batches_total - first_batch)
        cpi_start = first_batch * L

        if M < min_batches:
            if verbose:
                print(f"CPI {i + 1}/{num_cpis}: only {M} batches, passed through unfiltered.")
            continue

        # Overlap-save batches: each one also carries the K samples before it. For the surveillance, the K samples
        # before the CPI come from the already cleaned output (zeros at the very start of the capture).
        ref_segs = _get_cpi_segments(ref_ch, cpi_start - K, M, L, Nfft)
        surv_series = np.empty(K + M * L, dtype=np.complex64)
        surv_series[:K] = _get_cpi_segments(clean_surv, cpi_start - K, 1, 0, K)[0] if K else 0
        surv_series[K:] = surv_ch[cpi_start:cpi_start + M * L]

        clean_cpi, W = ECA_CD_CPI(ref_segs, surv_series, doppler_bins, doppler_spacing, iterations, diagonal_loading)
        clean_surv[cpi_start:cpi_start + M * L] = clean_cpi

        if store_weights:
            all_weights.append(W[0])  # pass 1 weights = per-carrier channel estimate

        if verbose:
            p_in = np.mean(np.abs(surv_series[K:]) ** 2)
            p_out = np.mean(np.abs(clean_cpi) ** 2)
            print(f"Processed CPI {i + 1}/{num_cpis} | Power reduction: {10 * np.log10(p_in / p_out):.1f} dB | "
                  f"Elapsed Time: {time.time() - start_time:.2f}s")

        del ref_segs, surv_series, clean_cpi

    if verbose:
        print(f"--- Processing Complete in {time.time() - start_time:.2f} seconds ---")

    return clean_surv, np.array(all_weights)


# =============================================================================
# SYNTHETIC SELF-TEST (no data needed)
# =============================================================================

def _rd_map(ref, surv, fs, seg_s, max_lag):
    """Small batches range-Doppler map for the self-test (matches passive_processing with seg_percent=100)."""
    L = int(fs / seg_s)
    M = len(ref) // L
    r = ref[:M * L].reshape(M, L)
    s = surv[:M * L].reshape(M, L)
    n = sfft.next_fast_len(2 * L)
    cc = sfft.ifft(sfft.fft(s, n, axis=1) * np.conj(sfft.fft(r, n, axis=1)), axis=1)[:, :max_lag + 1]
    rd = sfft.fftshift(sfft.fft(cc * np.hanning(M)[:, None], axis=0), axes=0)
    dop = sfft.fftshift(sfft.fftfreq(M, 1 / seg_s))
    return np.abs(rd) ** 2, dop


def run_synthetic_test(seed=0, iterations=3, max_delay_samples=2048):
    """
    Synthetic check at the real sample rate: 60 dB DSI + static and slow-moving multipath + a weak moving target.
    Prints the clutter suppression, and confirms the target survives in the range-Doppler map.
    """
    rng = np.random.default_rng(seed)
    fs = 15e6
    seg_s = 1000
    N = int(fs * 1.0)
    t = np.arange(N) / fs

    # Band-limited (DVB-T-like, 7.6 MHz occupied) reference with empty guard bands
    spec = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
    spec[np.abs(sfft.fftfreq(N, 1 / fs)) > 3.8e6] = 0
    ref = sfft.ifft(spec, workers=-1).astype(np.complex64)
    ref /= np.std(ref)
    del spec

    def echo(k, fd=0.0):
        y = np.zeros_like(ref)
        y[k:] = ref[:N - k]
        return (y * np.exp(1j * 2 * np.pi * fd * t)).astype(np.complex64)

    surv = ((rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)).astype(np.complex64)  # 0 dB noise
    for amp, lag, fd in [(1e3, 0, 0.0), (3e2, 5, 0.0), (1e2, 40, 0.4), (30, 300, -0.8), (10, 1500, 0.0)]:
        surv += amp * echo(lag, fd)
    tgt_lag, tgt_fd = 200, 37.0
    surv += 0.3 * echo(tgt_lag, tgt_fd)

    clean, W = ECA_CD_Batch_Process(ref, surv, fs, seg_s, cpi_time_sec=0.5, doppler_bins=2, doppler_spacing=0.5,
                                    max_delay_samples=max_delay_samples, iterations=iterations, verbose=True)

    rd_out, dop = _rd_map(ref, clean, fs, seg_s, 400)
    rd_ref, _ = _rd_map(ref, 0.3 * echo(tgt_lag, tgt_fd), fs, seg_s, 400)  # target on its own, for target loss
    t_d = np.argmin(np.abs(dop - tgt_fd))
    print("\n=== Synthetic test results ===")
    print(f"Surveillance power before/after: {10 * np.log10(np.mean(np.abs(surv) ** 2)):.1f} / "
          f"{10 * np.log10(np.mean(np.abs(clean) ** 2)):.1f} dB (noise floor is 0 dB)")
    print(f"Target loss: {10 * np.log10(rd_ref[t_d, tgt_lag] / rd_out[t_d, tgt_lag]):.2f} dB")
    print(f"Target SNR in RD map after ECA-CD: {10 * np.log10(rd_out[t_d, tgt_lag] / np.median(rd_out)):.1f} dB")
    pk = np.unravel_index(np.argmax(rd_out), rd_out.shape)
    print(f"Strongest cell after ECA-CD: lag {pk[1]}, Doppler {dop[pk[0]]:.1f} Hz "
          f"(target is lag {tgt_lag}, {tgt_fd} Hz)")
    return clean, W


# =============================================================================
# PROCESSING SCRIPT (mirrors weiner_filtering_batch_dictionaries_test.py)
# =============================================================================

if __name__ == "__main__":

    run_synthetic = False  # True = run the synthetic self-test only, no data needed

    if run_synthetic:
        run_synthetic_test()
        raise SystemExit

    c = 3e8
    decimation_factor = 32  # Decimation factor used during capture
    Fs = 3.84e9  # Sample rate
    d_Fs = Fs / (8 * decimation_factor)  # Decimated sample rate

    # ECA-CD parameters
    seg_s = 1000  # Batches per second: use the same seg_s you use in DSP.passive_processing
    cpi_time_sec = 0.5  # Weight estimation interval (Doppler resolution = 1 / cpi_time_sec)
    filter_max_velocity = 1  # Clutter velocity (m/s) to cancel either side of zero Doppler
    doppler_spacing = 1.0  # Doppler shift spacing in bins
    max_delay_samples = 2048  # Overlap at batch edges, matches the Wiener filter tap length
    iterations = 3  # 1 = ECA-CD as in the paper, 3 = deeper null for about 3x the run time

    batch_process_flag = True  # Set to True to process all captures in the directories
    iq_matrices_exist_flag = True

    experiments_config = [
        {
            "directory": r'G:\ARESTOR_DATA\SIREN\east_fort_day_3',
            "centre_freq": 518e6
        },
        {
            "directory": r'G:\ARESTOR_DATA\SIREN\east_fort_day_1',
            "centre_freq": 494e6
        },
    ]

    for exp in experiments_config:
        experiment_directory = exp["directory"]
        centre_freq = exp["centre_freq"]
        wavelength = c / centre_freq
        doppler_bins = velocity_to_doppler_bins(filter_max_velocity, wavelength, cpi_time_sec, doppler_spacing)

        print(f"\n{'=' * 50}")
        print(f"Starting ECA-CD batch process for directory: {experiment_directory}")
        print(f"Centre Frequency: {centre_freq} Hz (Wavelength: {wavelength:.3f} m) | P = {doppler_bins}")
        print(f"{'=' * 50}")

        if not os.path.exists(experiment_directory):
            print(f"Warning: Directory {experiment_directory} does not exist. Skipping...")
            continue

        if batch_process_flag:
            captures = os.listdir(experiment_directory)
        else:
            captures = ['ulm_day_3_pc12_5']

        for capture in captures:
            directory = fr'{experiment_directory}\{capture}\passive_data_test_1'

            if not os.path.exists(directory):
                print(f"Skipping {capture}: 'passive_data_test_1' subfolder not found.")
                continue

            print(f"\nProcessing capture: {capture}")

            if iq_matrices_exist_flag and os.path.exists(directory + r'\ref_iq.npy'):
                ref_iq = np.load(directory + r'\ref_iq.npy', mmap_mode='r')
                surv_iq = np.load(directory + r'\surv_iq.npy', mmap_mode='r')
                print(f"IQ Matrices loaded. ref shape: {ref_iq.shape}, surv shape: {surv_iq.shape}")
            else:
                try:
                    ref_data = np.load(directory + r'\passive_ref_output.npy')
                    surv_data = np.load(directory + r'\passive_surv_output.npy')
                except FileNotFoundError:
                    print(f"Data files not found in {directory}. Skipping...")
                    continue

                ref_iq = (ref_data[0::2] + 1j * ref_data[1::2]).astype(np.complex64)
                surv_iq = (surv_data[0::2] + 1j * surv_data[1::2]).astype(np.complex64)
                del surv_data, ref_data
                gc.collect()

                np.save(directory + r'\ref_iq.npy', ref_iq)
                np.save(directory + r'\surv_iq.npy', surv_iq)
                print(f"Completed loading and iq conversion. ref shape: {ref_iq.shape}, surv shape: {surv_iq.shape}")

            filtered_surv_channel, weights = ECA_CD_Batch_Process(
                ref_iq, surv_iq, d_Fs, seg_s, cpi_time_sec, doppler_bins, doppler_spacing, max_delay_samples,
                iterations
            )
            print("Finished filtering data. Saving...")

            tag = f'{seg_s}_{cpi_time_sec}_{doppler_bins}_{iterations}'
            np.save(directory + fr'\{tag}_eca_cd_filtered_surv_channel.npy', filtered_surv_channel)
            np.save(directory + fr'\{tag}_eca_cd_filter_weights.npy', weights)
            print(fr"Successfully saved filtered data to: {directory}")

            del filtered_surv_channel, weights, ref_iq, surv_iq
            gc.collect()
            print("Memory cleared, moving onto next capture.")

    print("\nFinished ECA-CD filtering of all IQ data across all directories.")
