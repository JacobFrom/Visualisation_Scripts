"""
Passive radar processing, wrapped up as a single easy-to-use class.

The :class:`PassiveProcessor` bundles the whole passive (e.g. DVB-T illuminator)
processing chain into one object. You create it once with the parameters of your
setup, then call the methods in order:

    from Passive_processing.passive_processor import PassiveProcessor

    radar = PassiveProcessor(centre_freq=570e6, decimation_factor=32)

    radar.set_range_window(min_range=0, max_range=4000)
    radar.load_iq(directory)                 # loads IQ AND Wiener-filters surv (default)
    radar.cross_correlate()                  # -> radar.cc_matrix, radar.range_axis_m
    radar.range_doppler()                    # -> radar.rd_slices, radar.velocity_axis
    radar.slices_to_db()                     # -> radar.rd_plots
    radar.make_rd_video(directory, 'rd_video')

Detection and analysis helpers are also available once you have range-Doppler
slices: :meth:`cfar_detect`, :meth:`extract_plots`, :meth:`filter_plots` and
:meth:`micro_doppler`.

Each method stores its result on the object (``self.<something>``) so the next
step finds it automatically. The numeric processing inside every method is
identical to the original standalone code in ``DSP_functions.py`` and the
``Testing`` modules (``CaCFAR.py``, ``clutterCancellation.py``,
``DetProcessing.py``); this class only reorganises it into a friendlier
interface.

The clutter-cancellation and detection algorithms are based on the pyAPRiL /
APRiL project by Tamás Pető (https://github.com/petotamas/APRiL), GPLv3.
"""

import os
import time

import numpy as np
from numpy import linalg as LA
import scipy as sp
import scipy.signal as signal
import scipy.fft as fft
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Plot:
    """
    Stores the parameters of a plot extracted from the range-Doppler map after
    detection, including range, Doppler, azimuth and elevation information.

    (Verbatim from ``Testing/DetProcessing.py`` - pyAPRiL hit processor.)
    """

    def __init__(self, plot_ID, range_cell, Doppler_cell):
        self.plot_ID = plot_ID  # Numerical identifier
        self.range_cells = []  # Filled up from hit matrix
        self.Doppler_cells = []  # Filled up from hit matrix
        self.sinr_inf = []  # Filled up from SINR matrix
        self.azimuth = 0  # Filled up from DoA estimation
        self.elevation = 0  # Filled up from DoA estimation

        self.range_cells.append(range_cell)
        self.Doppler_cells.append(Doppler_cell)

        self.weighted_rd = np.array([0, 0], dtype=float)  # [Doppler, range] cell indexes
        self.max_rd = np.array([0, 0])  # Doppler, range cell index that has maximum SINR

    def add_hit(self, range_cell, Doppler_cell):
        self.range_cells.append(range_cell)
        self.Doppler_cells.append(Doppler_cell)

    def max_snr_rd_calc(self):
        self.max_rd = [self.Doppler_cells[np.argmax(self.sinr_inf)], self.range_cells[np.argmax(self.sinr_inf)]]

    def central_mass_calc(self):
        # Convert sinr values from dB
        sinrs = 10 ** (np.array(self.sinr_inf) / 20)
        for hit_index in range(len(self.range_cells)):
            self.weighted_rd[1] += sinrs[hit_index] * self.range_cells[hit_index]
            self.weighted_rd[0] += sinrs[hit_index] * self.Doppler_cells[hit_index]

        self.weighted_rd /= sum(sinrs)


def hit_search(plot, hit_matrix, range_cell, Doppler_cell, d_max):
    """
    Recursively search for additional hits in the immediate vicinity of the given
    range and Doppler cell. (Verbatim from ``Testing/DetProcessing.py``.)
    """
    d_size = np.size(hit_matrix, 0)
    r_size = np.size(hit_matrix, 1)
    for d in np.arange(-d_max, d_max + 1, 1):
        for r in np.arange(-d_max, d_max + 1, 1):
            if 0 < Doppler_cell + d < d_size and 0 < range_cell + r < r_size:  # Out of hit matrix check
                if (hit_matrix[Doppler_cell + d, range_cell + r]) == 1:
                    plot.add_hit(range_cell + r, Doppler_cell + d)
                    hit_matrix[Doppler_cell + d, range_cell + r] = 0
                    hit_search(plot, hit_matrix, range_cell + r, Doppler_cell + d, d_max)


class PassiveProcessor:
    """End-to-end processing chain for a passive (illuminator-of-opportunity) radar capture."""

    def __init__(self, centre_freq=570e6, decimation_factor=32, Fs=3.84e9,
                 seg_s=2048, seg_percent=100, c=3e8, cmap='viridis'):
        """
        Parameters
        ----------
        centre_freq : float
            Illuminator centre frequency in Hz (e.g. a DVB-T channel).
        decimation_factor : int
            Decimation factor used during capture (capture dependent - check this!).
        Fs : float
            Sample rate before decimation in Hz.
        seg_s : int
            Number of segments per second (the passive equivalent of the PRF).
        seg_percent : float
            Percentage of each segment used in the cross-correlation.
        c : float
            Speed of light in m/s.
        cmap : str
            Matplotlib colormap used for the plots and videos.
        """
        # --- Radar / processing parameters ---
        self.centre_freq = centre_freq
        self.decimation_factor = decimation_factor
        self.Fs = Fs
        self.seg_s = seg_s
        self.seg_percent = seg_percent
        self.c = c
        self.cmap = cmap

        # Derived quantities
        self.d_Fs = Fs / (8 * decimation_factor)  # Decimated sample rate
        self.wavelength = c / centre_freq
        self.prf = seg_s  # segments per second is the passive PRF equivalent

        # Range window (see set_range_window)
        self.min_range = None
        self.max_range = None
        self.min_lag_samples = None
        self.max_lag_samples = None
        self.set_range_window()  # apply defaults

        # --- Results (filled in as you call the methods) ---
        self.ref_iq = None
        self.surv_iq = None
        self.capture_period = None
        self.filter_weights = None

        self.cc_matrix = None
        self.range_axis_m = None

        self.rd_slices = None
        self.rd_plots = None
        self.velocity_axis = None
        self.number_cpi = None
        self.cpi_stride = None

    # ------------------------------------------------------------------ #
    #  Configuration and loading
    # ------------------------------------------------------------------ #
    def set_range_window(self, min_range=0, max_range=4000):
        """
        Set the range window (in metres) to process and convert it to the
        corresponding minimum/maximum correlation lag in samples.
        """
        self.min_range = min_range
        self.max_range = max_range
        self.min_lag_samples = np.ceil(min_range * self.d_Fs / self.c).astype(int)  # Minimum lag in samples
        self.max_lag_samples = np.ceil(max_range * self.d_Fs / self.c).astype(int)  # Maximum lag in samples

    def load_iq(self, directory, ref_file='passive_ref_output.npy',
                surv_file='passive_surv_output.npy', is_raw=True, save_converted=False,
                filter_surv=True, filter_taps=2048, block_duration=0.1, smoothing_weight=1,
                save_filtered=False):
        """
        Load the reference and surveillance channels and (by default) clean the
        surveillance channel with Wiener clutter cancellation, ready for
        cross-correlation.

        Parameters
        ----------
        directory : str
            Folder containing the data files.
        ref_file, surv_file : str
            File names for the reference / surveillance data.
        is_raw : bool
            If True, the files are interleaved I/Q real arrays that get converted
            to complex here. If False, they are already-complex ``.npy`` arrays.
        save_converted : bool
            If True (and ``is_raw``), save the converted complex arrays as
            ``ref_iq.npy`` / ``surv_iq.npy`` next to the source data. The saved
            copies are the *unfiltered* IQ (filtering, if any, happens after).
        filter_surv : bool
            If True (the default), the surveillance channel is automatically
            passed through Wiener clutter cancellation
            (:meth:`wiener_clutter_cancel`) using the reference channel before it
            is used, so a non-expert gets a clutter-suppressed channel into
            :meth:`cross_correlate` without any extra steps. Set to False if your
            surveillance data is already filtered, or if you want to run a
            different cancellation step yourself.
        filter_taps, block_duration, smoothing_weight :
            Passed straight through to :meth:`wiener_clutter_cancel` when
            ``filter_surv`` is True.
        save_filtered : bool
            If True (and ``filter_surv`` is True), the filtered surveillance
            channel and its weights are saved into ``directory`` using the
            standard ``<block>_<taps>_<smoothing>_weiner_filtered_surv_channel.npy``
            naming, so you can reload them later instead of re-filtering.

        Sets ``self.ref_iq``, ``self.surv_iq`` and ``self.capture_period``. When
        ``filter_surv`` is True, ``self.surv_iq`` holds the filtered channel and
        the weights are stored in ``self.filter_weights``.
        """
        if is_raw:
            ref_data = np.load(os.path.join(directory, ref_file))
            surv_data = np.load(os.path.join(directory, surv_file))
            print("Loaded data")
            ref_iq = (ref_data[0::2] + 1j * ref_data[1::2]).astype(np.complex64)  # Combine into complex numbers
            surv_iq = (surv_data[0::2] + 1j * surv_data[1::2]).astype(np.complex64)  # Combine into complex numbers
            del surv_data, ref_data  # Release memory if I/Q parts are large
            if save_converted:
                np.save(os.path.join(directory, 'ref_iq.npy'), ref_iq)
                np.save(os.path.join(directory, 'surv_iq.npy'), surv_iq)
        else:
            ref_iq = np.load(os.path.join(directory, ref_file))
            surv_iq = np.load(os.path.join(directory, surv_file))

        capture_period = len(ref_iq) / self.d_Fs  # Duration of the capture in seconds
        print(f"IQ Matrices ready. reference data shape: {ref_iq.shape}, surveillance data shape: {surv_iq.shape}")

        self.ref_iq = ref_iq
        self.surv_iq = surv_iq
        self.capture_period = capture_period

        if filter_surv:
            print("Auto-filtering surveillance channel (Wiener clutter cancellation)...")
            self.wiener_clutter_cancel(filter_taps=filter_taps,
                                       block_duration=block_duration,
                                       smoothing_weight=smoothing_weight,
                                       save_directory=directory if save_filtered else None)

        return self.ref_iq, self.surv_iq

    # ------------------------------------------------------------------ #
    #  Clutter cancellation (pyAPRiL channel preparation)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _shift(x, i):
        """Similar to np.roll, but shifts in zeros rather than circularly wrapping."""
        N = x.shape[0]
        if np.abs(i) >= N:
            return np.zeros(N)
        if i == 0:
            return x
        shifted = np.roll(x, i)
        if i < 0:
            shifted[np.mod(N + i, N):] = np.zeros(np.abs(i))
        if i > 0:
            shifted[0:i] = np.zeros(np.abs(i))
        return shifted

    @staticmethod
    def _pruned_correlation(y, x, clen):
        """
        Calculate the values of the cross-correlation between [N-1 : N+clen-1].
        (Verbatim from ``Testing/clutterCancellation.py``.)
        """
        # --input check--
        N = x.shape[0]
        if N != y.shape[0] or clen > (N + 1) / 2:
            print('ERROR:not shapeable')
            return None

        # --calculation--
        # set up input matrices pad zeros if not multiply of the correlation length
        cols = clen - 1
        rows = np.int32(N / (cols)) + 1
        zeropads = cols * rows - N
        x = np.concatenate((x, np.zeros(zeropads, dtype=np.complex64)))
        y = np.concatenate((y, np.zeros(zeropads, dtype=np.complex64)))

        # shaping inputs into matrices
        xp = np.reshape(x, (rows, cols))
        yp = np.reshape(y, (rows, cols))

        # padding matrices for FFT
        ypp = np.vstack([yp[1:, :], np.zeros(cols, dtype=np.complex64)])
        yp = np.concatenate([yp, ypp], axis=1, dtype=np.complex64)
        xp = np.concatenate([xp, np.zeros((rows, cols), dtype=np.complex64)], axis=1)

        # execute FFT on the matrices
        xpw = sp.fft.fft(xp, axis=1)
        bpw = sp.fft.fft(yp, axis=1)

        corr_batches = np.fliplr(np.fft.fftshift(sp.fft.ifft(np.multiply(xpw, bpw.conj()).astype(np.complex64), axis=1)).conj()[:, 0:clen]).astype(np.complex64)

        # sum each value in a column of the batched correlation matrix
        return np.sum(corr_batches, axis=0)

    @staticmethod
    def _fast_pruned_correlation(x, y, K):
        """
        Calculate the first K lags of the cross-correlation using fast 1D FFTs.
        Zero-padding prevents circular aliasing (wrap-around errors).
        (Verbatim from ``Testing/clutterCancellation.py``.)
        """
        N = len(x)
        # Find the optimal size for the FFT to maximize speed
        fast_len = fft.next_fast_len(N + K - 1)

        # Transform to frequency domain
        X = fft.fft(x, n=fast_len)
        Y = fft.fft(y, n=fast_len)

        # Multiply and transform back to time domain
        R = fft.ifft(X * np.conj(Y))

        # Return only the first K lags
        return R[:K].astype(np.complex64)

    def wiener_clutter_cancel(self, filter_taps=2048, block_duration=0.5, smoothing_weight=None,
                              save_directory=None, save_name=None, save_weights=True):
        """
        Batched Wiener (SMI) clutter cancellation - the primary clutter-removal
        step. Breaks the capture into time blocks so the filter weights adapt to
        the changing environment, carving a deep zero-Doppler notch without
        overflowing RAM.

        Filters ``self.surv_iq`` using ``self.ref_iq`` and replaces
        ``self.surv_iq`` with the cleaned channel. Weights are stored in
        ``self.filter_weights``.

        Parameters
        ----------
        filter_taps, block_duration, smoothing_weight :
            Filter configuration (see the pyAPRiL batched Wiener filter).
        save_directory : str or None
            If given, the filtered surveillance channel is saved to this folder
            as an ``.npy`` file. The folder is created if it does not exist.
        save_name : str or None
            Base file name (without extension) for the saved channel. If None, a
            name is built from the filter parameters following the original
            convention, e.g. ``0.1_2048_1_weiner_filtered_surv_channel``.
        save_weights : bool
            If True (and ``save_directory`` is given), also save the filter
            weights alongside as ``<base>_weiner_filter_weights.npy``.

        (Verbatim ``Wiener_SMI_Batch_Process`` from ``Testing/clutterCancellation.py``.)
        """
        ref_ch = self.ref_iq
        surv_ch = self.surv_iq
        K = filter_taps
        fs = self.d_Fs
        block_time_sec = block_duration

        if ref_ch.shape[0] != surv_ch.shape[0]:
            print("ERROR: Input arrays must be the same size.")
            return None

        N_total = ref_ch.shape[0]
        block_size = int(fs * block_time_sec)
        num_blocks = int(np.ceil(N_total / block_size))

        print(f"--- Starting Batch Processing ---")
        print(f"Total Samples: {N_total}")
        print(f"Block Size: {block_size} samples ({block_time_sec} sec)")
        print(f"Total Blocks: {num_blocks}")

        # Pre-allocate the final output array to save memory fragmentation
        clean_surv = np.zeros(N_total, dtype=np.complex64)
        all_weights = []  # Optional: Store weights to plot how the environment changes

        start_time = time.time()

        for i in range(num_blocks):
            # Calculate start and end indices for the current chunk
            start_idx = i * block_size
            end_idx = min(start_idx + block_size, N_total)

            # Extract the chunk
            ref_block = ref_ch[start_idx:end_idx]
            surv_block = surv_ch[start_idx:end_idx]
            N_block = ref_block.shape[0]

            # Safety catch for the very last block if it's too tiny to filter
            if N_block < K:
                clean_surv[start_idx:end_idx] = surv_block
                continue

            # 1. Calculate correlations for just this specific block of time
            r_auto = self._fast_pruned_correlation(ref_block, ref_block, K)
            r_cross = self._fast_pruned_correlation(surv_block, ref_block, K)

            # 2. Fast Toeplitz Solver to get optimal weights for this exact moment
            if smoothing_weight is not None:
                # Code to test smoothing:
                w_current = la.solve_toeplitz(r_auto, r_cross)
                # Apply Exponential Moving Average (EMA) to smooth the transitions
                # alpha controls the memory: 1.0 = no smoothing, 0.1 = heavy smoothing
                alpha = smoothing_weight

                if i == 0:
                    w = w_current  # Initialize for the first block
                else:
                    w = (alpha * w_current) + ((1 - alpha) * w)
            else:
                w = la.solve_toeplitz(r_auto, r_cross)
                all_weights.append(w)
            # 3. Apply the filter weights using fast FFT convolution
            synth_clutter = signal.fftconvolve(ref_block, w, mode='full')[:N_block]

            # 4. Subtract the dynamically synthesized clutter
            clean_surv[start_idx:end_idx] = surv_block - synth_clutter

            # Progress tracking
            if (i + 1) % 10 == 0 or (i + 1) == num_blocks:
                elapsed = time.time() - start_time
                print(f"Processed block {i + 1}/{num_blocks} | Elapsed Time: {elapsed:.2f}s")

        print(f"--- Processing Complete in {time.time() - start_time:.2f} seconds ---")

        self.surv_iq = clean_surv
        self.filter_weights = np.array(all_weights)

        # Optionally save the filtered surveillance channel (and weights) to disk
        if save_directory is not None:
            os.makedirs(save_directory, exist_ok=True)
            if save_name is None:
                # Match the original file-naming convention
                save_name = f'{block_duration}_{filter_taps}_{smoothing_weight}_weiner_filtered_surv_channel'
            surv_path = os.path.join(save_directory, save_name + '.npy')
            np.save(surv_path, clean_surv)
            print(f"Saved filtered surveillance channel to {surv_path}")
            if save_weights:
                if 'filtered_surv_channel' in save_name:
                    weights_name = save_name.replace('filtered_surv_channel', 'filter_weights')
                else:
                    weights_name = save_name + '_weights'
                weights_path = os.path.join(save_directory, weights_name + '.npy')
                np.save(weights_path, self.filter_weights)
                print(f"Saved filter weights to {weights_path}")

        return clean_surv, np.array(all_weights)

    def wiener_smi(self, K, imp="direct_for"):
        """
        Time-domain Wiener filtering of the surveillance channel using Sample
        Matrix Inversion (SMI). ``imp="direct_for"`` is memory efficient;
        ``imp="direct_matrix"`` is faster but uses more RAM.

        (Verbatim ``Wiener_SMI`` from ``Testing/clutterCancellation.py``.)

        Returns ``(filt_ch, w)`` and stores the filtered channel in
        ``self.surv_iq``.
        """
        ref_ch = self.ref_iq
        surv_ch = self.surv_iq

        # -- input check --
        if imp != "direct_matrix" and imp != "direct_for":
            print("WARNING: Unidentified implementation type")
            print("WARNING: direct_matrix method is used")
            imp = "direct_matrix"

        # -- calculation --
        N = ref_ch.size  # Get the number of samples of the coherent processing interval
        R = np.zeros((K, K), dtype=complex)  # Autocorrelation matrix allocation
        r = np.zeros((1, K), dtype=complex)  # cross correlation vector allocation
        filt_ch = np.zeros(N, dtype=complex)  # Filtered surveillance channel allocation

        if imp == "direct_matrix":
            # Prepare signal delay matrix for fast TD filtering
            X = np.zeros((N, K), dtype=complex)  # K dimensional signal subspace matrix
            X[:, 0] = ref_ch
            for i in np.arange(1, K):
                X[:, i] = np.pad(ref_ch, (i, 0), mode='constant')[:-i]
            X = X.T

            # Estimate the autocorrelation matrix and the cross-correlation vector
            R = np.dot(X, X.conj().T)
            r = np.dot(X, surv_ch.conj())

            # Calculate coefficient vector
            w = np.dot(np.linalg.inv(R), np.transpose(r))

            # Perform filtering
            filt_ch = surv_ch - np.dot(X.T, w.conj().T)

        elif imp == "direct_for":
            # Estimate the autocorrelation matrix and the cross-correlation vector
            for i in np.arange(K, N):
                R += np.outer(ref_ch[i - K:i][::-1], np.conjugate(ref_ch[i - K:i][::-1]))
                r += (ref_ch[i - K:i][::-1]) * np.conjugate(surv_ch[i - 1])

            # Normalization with the sample size
            R = np.divide(R, N - K)
            r = np.divide(r, N - K)

            # Calculate coefficient vector
            w = np.dot(np.linalg.inv(R), np.transpose(r))

            # Prepare signal vectors for filtering
            filt_ch = np.zeros((N + 1), dtype=complex)  # Filtered surveillance channel
            ext_ref = np.concatenate((np.zeros(K, dtype=complex), ref_ch), axis=0)  # extend signal with zeros
            x_n = np.zeros(K, dtype=complex)  # The last K samples of the reference signal

            # Filtering
            for n in np.arange(0, N + 1, 1):
                x_n[:] = ext_ref[n:n + K]  # Select last "l" element in the reference channel
                x_n = x_n[::-1]  # To preserve FIR filter format
                filt_ch[n] = np.dot(np.transpose(w.conj()), x_n)[0]  # Calculate filtered output
            filt_ch = filt_ch[1:N + 1]
            filt_ch = surv_ch - filt_ch

        self.surv_iq = filt_ch
        self.filter_weights = w
        return filt_ch, w

    def wiener_smi_mre(self, K):
        """
        Wiener filtering with the Minimum Redundance Estimation (MRE) technique -
        only the first column of the autocorrelation matrix is estimated,
        reducing the work from KxK to K.

        (Verbatim ``Wiener_SMI_MRE`` from ``Testing/clutterCancellation.py``.)
        """
        ref_ch = self.ref_iq
        surv_ch = self.surv_iq

        # --- Input check ---
        if ref_ch.shape[0] != surv_ch.shape[0]:
            print("ERROR: No output is generated")
            return None

        N = ref_ch.shape[0]  # Number of time samples
        R = np.zeros((K, K), dtype=np.complex64)  # Autocorrelation mtx.
        print("Arrays initialised, beginning pruned correlation...")
        R[:, 0] = self._pruned_correlation(ref_ch, ref_ch, K)  # Calc. first column of the autocorr. matrix
        r = self._pruned_correlation(surv_ch, ref_ch, K)  # Cross-correlation vector

        print("pruned correlation Complete, moving onto filtering")
        # Complete the R matrix based on its Hermitian and Toeplitz property
        for k in range(1, K):
            R[:, k] = self._shift(R[:, 0], k)

        R += np.transpose(np.conjugate(R))
        R *= (np.ones(K) - np.eye(K) * 0.5)

        w = np.dot(LA.inv(R), r)  # weight vector

        filt_ch = surv_ch - np.convolve(ref_ch, w, mode='full')[0:N]  # subtract the zero doppler clutter
        self.surv_iq = filt_ch
        self.filter_weights = w
        return filt_ch, w

    @staticmethod
    def gen_subspace_indexes(K, D):
        """
        Generate a signal subspace parameter list for the ECA algorithm variants.
        (Verbatim ``gen_subspace_indexes`` from ``Testing/clutterCancellation.py``.)
        """
        rd_index_list = []  # Used to store range-Doppler indexes on which the filtering have to performed
        # Create initial filter list
        for i in np.arange(-D, D + 1, 1):  # Doppler shifts
            for j in np.arange(0, K):  # Runs on signal delays
                rd_index_list.append([i, j])

        return rd_index_list

    def ecas(self, subspace_list, T, Na):
        """
        Extensive Cancellation Algorithm - Sliding window (ECA-S). Filters the
        surveillance channel in the time domain; coefficient estimation uses a
        symmetric window extension controlled by ``Na``.

        (Verbatim ``ECAS`` from ``Testing/clutterCancellation.py``.)
        """
        ref_ch = self.ref_iq
        surv_ch = self.surv_iq

        # TODO: If the batch size is not multiple of the CPI the signals should be extend with zeros
        N = ref_ch.size
        batch_size = N // T

        # -- input check --
        if N % T:
            print("ERROR: Improper batch size, filtering is bypassed")
            print("ERROR: No output is generated")
            return None, None

        for k in subspace_list:
            if k[1] > batch_size:
                print("ERROR: Time delay is greater than the size of the batch, "
                      "this is not handled properly. Time delay is bypassed")
                print("ERROR: No output is generated")
                return None, None
            if k[1] > (batch_size + Na):
                print("ERROR: Time delay is greater than the size of the batch plus the sliding window, "
                      "this is not handled properly. Time delay is bypassed")
                print("ERROR: No output is generated")
                return None, None
        # -- calculation --
        KD = len(subspace_list)  # Signal subspace dimension

        extended_ref_ch = np.zeros(N + 2 * Na, dtype=complex)
        extended_ref_ch[Na:Na + N] = ref_ch[:]
        extended_surv_ch = np.zeros(N + 2 * Na, dtype=complex)
        extended_surv_ch[Na:Na + N] = surv_ch[:]

        filtered_surv_ch = np.zeros(N, dtype=complex)

        for l in np.arange(0, N, batch_size):  # Runs on batches
            surv_signal_batch = surv_ch[l: l + batch_size]  # Filtering will be performed on this block
            surv_signal_batch_ta = extended_surv_ch[l:l + batch_size + 2 * Na]  # "w" will be estimated on this block
            x = np.zeros((batch_size, KD), dtype=complex)  # Signal subspace for filtering
            x_ta = np.zeros((batch_size + 2 * Na, KD), dtype=complex)  # Signal subspace for the estimation of the "w"

            # Generate matrices for filtering and estimation from the current batch
            subspace_vector_index = -1  # Used to index the subspace matrix (X) inside the for loops
            for k in subspace_list:  # Select current subspace vector parameter
                subspace_vector_index += 1

                doppler_shift_array_ta = np.exp(np.arange(l - Na, l + batch_size + Na, 1) * 1j * 2 * np.pi * k[0] / (2 * N))
                doppler_shift_array = np.exp(np.arange(l, l + batch_size, 1) * 1j * 2 * np.pi * k[0] / (2 * N))
                if not l:  # For the first batch zero values must be inserted (l=0)
                    x[:, subspace_vector_index] = self._shift(ref_ch[l:l + batch_size], k[1])
                    x_ta[:, subspace_vector_index] = self._shift(extended_ref_ch[l:l + batch_size + 2 * Na], k[1])
                else:
                    x[:, subspace_vector_index] = ref_ch[l - k[1]:l + batch_size - k[1]]
                    x_ta[:, subspace_vector_index] = extended_ref_ch[l - k[1]:l + 2 * Na + batch_size - k[1]]

                x[:, subspace_vector_index] *= doppler_shift_array  # Doppler shift
                x_ta[:, subspace_vector_index] *= doppler_shift_array_ta  # Doppler shift

            # Orthogonal subspace projection
            x = np.matrix(x)  # N x KD matrix
            x_ta = np.matrix(x_ta)  # N x KD matrix
            surv_signal = np.matrix(surv_signal_batch).getT()  # column vector
            surv_signal_ta = np.matrix(surv_signal_batch_ta).getT()  # column vector
            corr_matrix = x_ta.getH() * x_ta  # KD x KD
            corr_vector = x_ta.getH() * surv_signal_ta  # KD x 1
            weight_vector = corr_matrix.getI() * corr_vector  # coefficient vector
            filtered_surv_ch[l:l + batch_size] = np.array(surv_signal - x * weight_vector)[:, 0]

        self.surv_iq = filtered_surv_ch
        return filtered_surv_ch

    @staticmethod
    def change_to_log_scale(input_array, dynamic_range, data_type="power", normalize=False):
        """Convert an array to a log (dB) scale, clipped to ``dynamic_range``."""
        if data_type == "power":
            multiplier = 10
        elif data_type == "voltage":
            multiplier = 20
        else:
            print("Unidentified dataType, handled as power")
            multiplier = 10
        array_abs = np.abs(input_array)
        if normalize:
            array_abs = np.divide(array_abs, np.max(array_abs))
        array_log = multiplier * np.log10(array_abs)

        maximum = np.max(array_log)
        for i in np.nditer(array_log, op_flags=['readwrite']):
            if i < maximum - dynamic_range:
                i[...] = maximum - dynamic_range

        return array_log

    # ------------------------------------------------------------------ #
    #  Cross-correlation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _py_xcorr(surv_iq_matrix, ref_iq_matrix, max_lag, min_lag=0):
        """
        Cross-correlation between surveillance and reference IQ matrices.
        (Verbatim ``py_xcorr`` from ``DSP_functions.py``.)
        """
        full_corr = signal.correlate(surv_iq_matrix, ref_iq_matrix, mode='full', method='auto')
        full_lags = signal.correlation_lags(len(surv_iq_matrix), len(ref_iq_matrix), mode='full')

        # Extract the relevant lags and cross-correlation values
        zero_lag_idx = len(full_corr) // 2
        # With center point known for 0th lag, we can calculate the indices for the desired lags
        min_lag_idx = zero_lag_idx + min_lag
        max_lag_idx = zero_lag_idx + max_lag + 1  # +1 to include the max lag in the output

        # Now to extract the relevant part of the cross-correlation matrix
        cc_matrix = full_corr[min_lag_idx:max_lag_idx]
        lags = full_lags[min_lag_idx:max_lag_idx]

        return cc_matrix, lags

    def cross_correlate(self):
        """
        Segment the reference and surveillance channels and cross-correlate them
        segment by segment to build the range/slow-time beat-frequency (cc)
        matrix.

        Sets ``self.cc_matrix`` and ``self.range_axis_m``.
        (Verbatim ``passive_processing`` from ``DSP_functions.py``.)
        """
        surv_iq = self.surv_iq
        ref_iq = self.ref_iq
        seg_s = self.seg_s
        seg_percent = self.seg_percent
        Fs = self.d_Fs
        max_lag_samples = self.max_lag_samples
        min_lag_samples = self.min_lag_samples

        # Set up the segmented matrixes for processing
        seg_size = int(Fs / seg_s)  # Segment size in samples
        num_segments = int(np.floor(len(ref_iq) / seg_size))  # Number of segments in the reference data

        ref_iq = ref_iq[0:int((num_segments * seg_size))]
        surv_iq = surv_iq[0:int((num_segments * seg_size))]  # Ensure both signals have the same length

        seg_ref_channel = np.reshape(ref_iq, (num_segments, seg_size))
        seg_surv_channel = np.reshape(surv_iq, (num_segments, seg_size))

        # Calculate the decimated reference channel
        cc_size = np.round(seg_size * (seg_percent / 100))
        dec_ref_channel = seg_ref_channel[:, 0:int(cc_size)]  # Decimate the reference channel
        dec_surv_channel = seg_surv_channel[:, 0:int(cc_size)]  # Decimate the surveillance channel
        print(f"Decimated reference channel shape: {dec_ref_channel.shape}, "
              f"Decimated surveillance channel shape: {dec_surv_channel.shape}")

        # Here we begin the cross-correlation processing
        # Start with specifying the CC matrix idx of interest for extraction later
        num_range_bins_out = max_lag_samples - min_lag_samples + 1  # Number of range bins in the output

        cc_matrix = np.zeros((num_segments, num_range_bins_out), dtype=np.complex64)  # Initialize the cc matrix

        start_time = time.perf_counter()
        for i in range(num_segments):
            temp_cc_matrix, temp_lags = self._py_xcorr(dec_surv_channel[i, :], dec_ref_channel[i, :],
                                                       max_lag=max_lag_samples,
                                                       min_lag=min_lag_samples)

            cc_matrix[i, :] = temp_cc_matrix
            print(f"Finished processing {i} of {num_segments} segments of data.")
        end_time = time.perf_counter()
        processing_duration = end_time - start_time
        print(f"Finished processing {num_segments} segments of data in {processing_duration:.2f} seconds.")

        no_range_bins = int(cc_matrix.shape[1])
        print("Number of range bins:", no_range_bins)
        range_axis_m = np.linspace(self.min_range, self.max_range, no_range_bins)

        self.cc_matrix = cc_matrix
        self.range_axis_m = range_axis_m
        return cc_matrix

    # ------------------------------------------------------------------ #
    #  Range-Doppler processing
    # ------------------------------------------------------------------ #
    def range_doppler(self, cpi_length=1, cpi_overlap=0.9, rd_window='blackmanharris'):
        """
        Perform range-Doppler slicing on the cross-correlation matrix.

        Unlike the active version, the slices are kept **complex** (no dB
        conversion) so they can be re-used for detection and micro-Doppler.

        Sets ``self.rd_slices``, ``self.velocity_axis``, ``self.number_cpi`` and
        ``self.cpi_stride``.
        (Verbatim ``passive_range_Doppler_slicing`` from ``DSP_functions.py``.)
        """
        beat_frequencies = self.cc_matrix
        prf = self.prf
        wavelength = self.wavelength

        # Configure the parameters for range-Doppler processing
        pulse_interval = 1 / prf
        num_pulses = beat_frequencies.shape[0]
        pulses_per_cpi = int(cpi_length * prf)
        cpi_stride = int(np.round(pulses_per_cpi * (1 - cpi_overlap)))

        window = signal.get_window(rd_window, pulses_per_cpi)

        number_cpi = int(np.ceil((num_pulses - pulses_per_cpi) / cpi_stride))

        print("Pulses per CPI:", pulses_per_cpi)
        print("CPI stride:", cpi_stride)

        # configure the axis for the range-Doppler slices
        slice_doppler_axis_unshifted = fft.fftfreq(pulses_per_cpi, d=pulse_interval)  # doppler axis in Hz
        slice_doppler_axis = fft.fftshift(slice_doppler_axis_unshifted)  # Shift to center zero frequency
        velocity_axis = (wavelength * slice_doppler_axis) / 2  # Convert to velocity in m/s

        # Now onto the slicing
        rd_slices_list = []  # Pre allocate the list for range-Doppler slices
        for i in range(number_cpi):
            start_index = i * cpi_stride
            end_index = start_index + pulses_per_cpi

            windowed_cpi = beat_frequencies[start_index:end_index] * window[:, np.newaxis]  # Apply window to CPI
            rd_slice = fft.fft(windowed_cpi, axis=0)
            rd_slice = fft.fftshift(rd_slice, axes=0)

            rd_slices_list.append(rd_slice)  # Append the slice to the list
            print(f'Finished slice: {len(rd_slices_list)} of {number_cpi}. RD Slice Shape: {rd_slice.shape}')
        print(f"Finished processing. Number of slices processed: {len(rd_slices_list)}")

        self.rd_slices = rd_slices_list
        self.velocity_axis = velocity_axis
        self.number_cpi = number_cpi
        self.cpi_stride = cpi_stride
        return rd_slices_list, velocity_axis, number_cpi, cpi_stride

    def slices_to_db(self):
        """
        Convert the complex range-Doppler slices to normalised dB plots ready for
        video creation. Sets ``self.rd_plots``.
        """
        rd_plots = []
        for slice in self.rd_slices:
            slice_plot = (20 * np.log10(np.abs(slice) / np.max(np.abs(slice))))
            rd_plots.append(slice_plot)
            print(f"Appended slice {len(rd_plots)}")
        print(f"Finished creating RD plots. RD plots shape: {len(rd_plots)}")
        self.rd_plots = rd_plots
        return rd_plots

    def make_rd_video(self, directory, save_as, v_min=-35, v_max=0, x_lim=None,
                      max_doppler=100, fps=None, sharpening=False):
        """
        Create an ``.mp4`` video from the range-Doppler slices (uses
        ``self.rd_plots`` if available, otherwise ``self.rd_slices``).

        ``x_lim`` defaults to ``[0, max_range]`` and the velocity (y) axis is
        limited to ``[-max_doppler, max_doppler]``. Requires ``ffmpeg`` on PATH.
        (Verbatim ``create_rd_video`` from ``DSP_functions.py``.)
        """
        rd_slices_list = self.rd_plots if self.rd_plots is not None else self.rd_slices
        prf = self.prf
        cpi_stride = self.cpi_stride
        range_axis = self.range_axis_m
        velocity_axis = self.velocity_axis

        if x_lim is None:
            x_lim = [0, self.max_range]
        y_lim = [-max_doppler, max_doppler]

        if fps is None:
            fps = 1 / (self.capture_period / self.number_cpi)  # Frames per second for the output video
        FPS = fps

        os.makedirs(directory, exist_ok=True)
        VIDEO_FILENAME = save_as + '.mp4'  # Output video filename
        VIDEO_SAVE = os.path.join(directory, VIDEO_FILENAME)
        COLOR_VMIN = v_min  # Min value for the color scale (dB) - adjust as needed
        COLOR_VMAX = v_max  # Max value for the color scale (dB)

        # --- Setup the Figure ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # Initial plot setup using the first frame
        initial_data = rd_slices_list[0]

        im = ax.imshow(initial_data,
                       aspect='auto',
                       origin='lower',
                       extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
                       vmin=COLOR_VMIN,
                       vmax=COLOR_VMAX,
                       cmap=self.cmap)

        # Add labels and title (base title)
        ax.set_xlabel('Range [m]')
        ax.set_ylabel('Velocity [m/s]')
        ax.set_xlim(x_lim)
        # Set the y-axis (Velocity) limits
        ax.set_ylim(y_lim)
        base_title = 'Range-Doppler Plot Evolution'
        title_text = ax.set_title(base_title)  # Store title object for updates

        # Add colorbar
        cbar = fig.colorbar(im, label='Norm Power [dB]')
        plt.tight_layout()  # Adjust layout to fit colorbar

        def update(frame_num):
            if sharpening:
                current_rd_slice = rd_slices_list[frame_num].copy()

                noise_threshold = v_min + 2  # Set a noise threshold for the video
                current_rd_slice[current_rd_slice < noise_threshold] = COLOR_VMIN

                im.set_data(current_rd_slice)  # Update image data
            else:
                current_rd_slice = rd_slices_list[frame_num]
                im.set_data(current_rd_slice)
            # Update title
            start_time_sec = (frame_num * cpi_stride) / prf

            current_title = f'{base_title}\nFrame: {frame_num + 1}/{len(rd_slices_list)}, Time ≈ {start_time_sec:.2f} s'

            title_text.set_text(current_title)

            return [im, title_text]  # Return updated artists

        # --- Create and Save Animation ---
        print(f"Creating animation with {len(rd_slices_list)} frames...")
        animation_start_time = time.perf_counter()
        ani = animation.FuncAnimation(fig,
                                      update,
                                      frames=len(rd_slices_list),
                                      interval=1000 / FPS,  # ms delay between frames
                                      blit=True,
                                      repeat=False)
        animation_end_time = time.perf_counter()
        animation_duration = animation_end_time - animation_start_time
        print(f'Animation creation took {animation_duration:.2f} seconds.')
        print(f"Finished creating animation with {len(rd_slices_list)} frames. Now saving to {VIDEO_SAVE}...")
        saving_start_time = time.perf_counter()
        # IMPORTANT: Ensure ffmpeg is installed and available in user path or this won't work
        ani.save(str(VIDEO_SAVE), writer='ffmpeg', fps=FPS, dpi=150)
        saving_end_time = time.perf_counter()
        saving_duration = saving_end_time - saving_start_time
        print(f"Animation saved to {VIDEO_SAVE} and saving took {saving_duration} seconds.")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Detection (CA-CFAR) - shared with ActiveProcessor
    # ------------------------------------------------------------------ #
    def cfar_detect(self, rd_matrix, window_params=(8, 8, 2, 2), threshold_db=23):
        """
        Cell-Averaging Constant False Alarm Rate (CA-CFAR) detection on a single
        range-Doppler matrix.

        Parameters
        ----------
        rd_matrix : complex ndarray
            Range-Doppler map (Doppler x Range) to run detection on.
        window_params : sequence of 4 ints
            [Est. window width, Est. window height, Guard width, Guard height].
        threshold_db : float
            Threshold above the estimated average noise power, in dB.

        Returns
        -------
        hit_matrix : bool ndarray
            True where the SNR exceeded the threshold.
        rd_snr : ndarray
            The estimated signal-to-noise ratio map.

        (Verbatim ``CA_CFAR`` from ``Testing/CaCFAR.py``, based on
        https://github.com/petotamas/APRiL.)
        """
        rd_size = rd_matrix.shape

        # Unpack the list of window parameters into individual variables for readability
        win_width = window_params[0]
        win_height = window_params[1]
        guard_width = window_params[2]
        guard_height = window_params[3]

        # Create the sliding window "kernel" (mask) initially filled with 1s (Reference Cells)
        mask = np.ones((2 * win_height + 1, 2 * win_width + 1), dtype=bool)

        # Set the center region (Guard Cells and CUT) to 0 so they are excluded from the noise calculation
        mask[win_height - guard_height:win_height + 1 + guard_height,
             win_width - guard_width:win_width + 1 + guard_width] = 0

        # Convert the threshold from decibels (dB) to a linear power ratio
        threshold = 10 ** (threshold_db / 10)

        # Pre-calculate how many valid reference cells exist for every pixel position
        num_valid_cells_in_window = signal.convolve2d(np.ones(rd_size, dtype=float), mask, mode='same')

        # Convert the complex values (voltage) of the input matrix to Power (magnitude squared)
        rd_matrix = np.abs(rd_matrix) ** 2

        # 1. Sum the noise power around every cell
        rd_windowed_sum = signal.convolve2d(rd_matrix, mask, mode='same')

        # 2. Calculate the Average Noise Power
        rd_avg_noise_power = rd_windowed_sum / num_valid_cells_in_window

        # 3. Calculate SNR (Signal-to-Noise Ratio)
        rd_snr = rd_matrix / rd_avg_noise_power

        # 4. Determine Hits
        hit_matrix = rd_snr > threshold

        return hit_matrix, rd_snr

    # ------------------------------------------------------------------ #
    #  Plot (target) extraction and filtering
    # ------------------------------------------------------------------ #
    def extract_plots(self, hit_matrix_orig, sinr_matrix=None, d_max=1):
        """
        Combine groups of adjacent hits into target plots (clustering). If
        ``sinr_matrix`` is given, SINR values weight the plot's centre of mass.

        Returns a list of :class:`Plot` objects.
        (Verbatim ``plot_extractor`` from ``Testing/DetProcessing.py``.)
        """
        Doppler_cells = np.size(hit_matrix_orig, 0)
        range_cells = np.size(hit_matrix_orig, 1)
        hit_matrix = np.zeros((Doppler_cells, range_cells), dtype=int)
        hit_matrix[:] = hit_matrix_orig[:]

        if sinr_matrix is None:
            sinr_matrix = np.ones((Doppler_cells, range_cells), dtype=int)

        plot_list = []
        plot_cntr = 0

        for d in range(Doppler_cells):
            for r in range(range_cells):
                if hit_matrix[d, r] == 1:
                    # Register hit
                    plot_inst0 = Plot(plot_cntr, r, d)
                    plot_list.append(plot_inst0)
                    plot_cntr += 1

                    # Delete hit from hit matrix
                    hit_matrix[d, r] = 0

                    # Search for additional hits in the vicinity of the founded plot
                    hit_search(plot_inst0, hit_matrix, r, d, d_max)

        # Fill up plots with SINR information
        for plot in plot_list:
            for hit_index in range(len(plot.range_cells)):
                r = plot.range_cells[hit_index]
                D = plot.Doppler_cells[hit_index]
                plot.sinr_inf.append(sinr_matrix[D, r])  # Add SINR information

        # Calculate weighted and maxium range-Doppler cell indexes
        for plot in plot_list:
            plot.central_mass_calc()
            plot.max_snr_rd_calc()
        return plot_list

    def filter_plots(self, plot_list, range_gate, Doppler_gate, rd_size):
        """
        Apply range-gate and Doppler-gate filtering to a plot list, removing
        close-range clutter and near-zero-Doppler (static) plots.
        (Verbatim ``plot_filter`` from ``Testing/DetProcessing.py``.)
        """
        filter_list = []
        # Filter close range plots
        for plot in plot_list:
            if plot.weighted_rd[1] <= range_gate:
                filter_list.append(plot)
                print("Range gate - plot filtered: {0}".format(plot.weighted_rd))
        for plot_rm in filter_list:
            plot_list.remove(plot_rm)

        filter_list = []
        # Filter small Doppler plots
        for plot in plot_list:
            if abs(plot.weighted_rd[0] - (rd_size[0] - 1) // 2) <= Doppler_gate:
                filter_list.append(plot)
                print("Doppler gate: -plot filtered: {0}".format(plot.weighted_rd))
        for plot_rm in filter_list:
            plot_list.remove(plot_rm)

        return plot_list

    # ------------------------------------------------------------------ #
    #  Micro-Doppler - shared with ActiveProcessor
    # ------------------------------------------------------------------ #
    def micro_doppler(self, rd_slice_list, range_bin, slice_under_test, target_velocity,
                      notch_velocity, prf=None, NFFT=1024, Noverlap=None,
                      directory=None, save_as=None, target='Target', show=True):
        """
        Build a micro-Doppler spectrogram for a single range bin, reconstructed
        from complex range-Doppler slices.

        ``rd_slice_list`` must be a 3D complex array indexed
        ``[slice, doppler, range]`` (the complex slices from :meth:`range_doppler`,
        e.g. as saved to ``rd_slice_matrix.npy``). ``target_velocity`` recentres
        the bulk motion to 0 Hz; ``notch_velocity`` sets the main-body notch width.
        (Verbatim from ``Testing/micro_doppler_testing_RD_complete.py``.)
        """
        if prf is None:
            prf = self.prf
        if Noverlap is None:
            Noverlap = NFFT - 1

        wavelength = self.wavelength
        target_range = 20 * range_bin

        f_shift_hz = (2 * target_velocity / wavelength)

        start_frame = slice_under_test
        end_frame = slice_under_test + 1

        # --- Time-Domain Reconstruction (Stitching) ---
        slow_time_list = []
        for i in range(start_frame, end_frame):
            doppler_slice = rd_slice_list[i, :, range_bin]

            # Undo the zero-centering to prepare for IFFT
            doppler_unshifted = fft.ifftshift(doppler_slice)

            # Revert to slow-time
            slow_time_slice = fft.ifft(doppler_unshifted)

            slow_time_list.append(slow_time_slice)

        # Stitch them all together into one continuous 1D array
        stitched_slow_time = np.concatenate(slow_time_list)
        print(f"Stitched slow time matrix shape: {stitched_slow_time.shape}")

        # --- Bulk Motion Shifting ---
        N_samples = len(stitched_slow_time)
        t = np.arange(N_samples) / prf

        shifted_slow_time = stitched_slow_time * np.exp(-1j * 2 * np.pi * f_shift_hz * t)

        # --- Short-Time Fourier Transform (STFT) ---
        frequencies, times, Zxx = signal.stft(
            shifted_slow_time,
            fs=prf,
            window='hann',
            nperseg=NFFT,
            noverlap=Noverlap,
            nfft=NFFT,
            return_onesided=False  # Mandatory for complex radar data
        )

        # Move the new 0 Hz (the bulk motion) to the center of the arrays
        frequencies = fft.fftshift(frequencies)
        Zxx = fft.fftshift(Zxx, axes=0)

        # --- Main Body Cancellation (Notching) ---
        notch_width_hz = ((2 * notch_velocity) / wavelength)  # Total width of the suppression band
        notch_indices = np.abs(frequencies) < (notch_width_hz / 2)

        # Suppress the main body signal (tiny number instead of 0 to avoid log(0) errors)
        Zxx[notch_indices, :] = 1e-12

        # --- Dynamic Range & Decibel Conversion ---
        Zxx_mag = np.abs(Zxx)
        Zxx_mag = np.maximum(Zxx_mag, 1e-12)  # Floor to prevent log(0)
        Zxx_db = 25 * np.log10(Zxx_mag)

        # 99.5th percentile prevents single massive clutter spikes from washing out the plot
        vmax = np.percentile(Zxx_db, 99.5)
        vmin = vmax - 30  # dynamic range

        # --- Plotting ---
        plt.figure(figsize=(14, 6))
        pc = plt.pcolormesh(times, frequencies, Zxx_db, cmap=self.cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(pc, label='Power (dB)')
        plt.ylabel('Doppler Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title(f'Micro-Doppler Spectrogram of {target} at {target_range:.0f}m FFT Length: {NFFT}')
        plt.tight_layout()

        if directory is not None and save_as is not None:
            os.makedirs(directory, exist_ok=True)
            save_path = os.path.join(directory, save_as + '.png')
            plt.savefig(save_path)
            print(f"Micro-Doppler spectrogram saved to {save_path}")
        if show:
            plt.show()
        plt.close()

        return frequencies, times, Zxx_db

    # ------------------------------------------------------------------ #
    #  Power Spectral Density diagnostic
    # ------------------------------------------------------------------ #
    def plot_psd(self, directory, filename, start=0, block_pulses=100000, prf=1024,
                 normalize=False, show=True):
        """
        Quick Power Spectral Density plot of a raw reference capture, useful as a
        sanity check on the illuminator spectrum.
        (Verbatim from ``Passive_processing/PSD_passive.py``.)
        """
        data = np.load(os.path.join(directory, filename))

        total_decimation = 8 * self.decimation_factor
        Fs = self.Fs

        data = data[0::2] + 1j * data[1::2]
        N = start  # Number of samples for fft
        end = N + block_pulses * prf

        fft_data = data[N:end]  # Take the block of samples for the fft

        fft_data = fft_data * np.hamming(len(fft_data))  # Apply hamming window function to the data

        if normalize:
            PSD = np.abs(np.fft.fft(fft_data)) ** 2 / (len(fft_data) * (Fs / total_decimation))
        else:
            PSD = np.abs(np.fft.fft(fft_data)) ** 2  # Non-normalised Power Spectral Density

        PSD_dB = 10.0 * np.log10(PSD)  # Convert to dB
        PSD_shifted = np.fft.fftshift(PSD_dB)  # Shift the PSD to center around 0

        f = np.arange(-((Fs / total_decimation) / 2), ((Fs / total_decimation) / 2),
                      (Fs / total_decimation) / len(fft_data))  # Centered around 0 Hz
        f += self.centre_freq  # Shift the frequency axis to the centre frequency

        plt.plot(f, PSD_shifted)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        plt.grid(True)
        if show:
            plt.show()
        plt.close()
