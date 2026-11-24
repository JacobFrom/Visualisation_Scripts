import os
import numpy as np
import scipy as sp
import scipy.fft as fft
import scipy.signal as signal
import matplotlib.pyplot as plt

# ==========================================================================
# Micro-Doppler spectrogram + rotor parameter analysis
#
# Adapted from Testing/micro_doppler_testing_RD_complete.py. Same idea:
#   RD slice -> IFFT back to slow time -> shift body to 0 Hz -> STFT -> plot
# Changes from the original script:
#   * Several consecutive RD slices are stitched into one continuous slow-time
#     record (the Blackman-Harris window applied in the RD slicing is undone,
#     and only the middle 0.1 s of each slice is kept, where the window is ~1).
#   * The main body notch is applied BEFORE the STFT (on the slow-time FFT),
#     not before the IFFT of the RD slice.
#   * Two spectrograms are made: a long window (spectral lines) and a short
#     window (blade flashes), and both are saved at paper size.
#   * Rotor measurements are printed and saved to a text file.
# ==========================================================================

# ==========================================
# 1. Configuration
# ==========================================
directory = r'G:\ARESTOR_DATA\SIREN\east_fort_day_3\east_fort_capture_10\passive_data_test_1'
rd_slice_matrix_path = os.path.join(directory, 'rd_slices', '4000_rd_slice_matrix.npy')
save_directory = os.path.join(directory, 'claude_md_figures')

target = "Airbus H125"
target_tag = "h125"          # short name used in the saved file names

# Radar parameters (same as the original script)
c = sp.constants.c
centre_freq = 518e6          # CHECK: carrier of the DVB-T channel used for this capture
wavelength = c / centre_freq
prf = 2048                   # segments per second from the cross-correlation
range_bin = 19
slice_under_test = 184       # first RD slice to use
cpi_length = 1.0             # s, CPI used in the RD slicing
cpi_overlap = 0.9            # overlap used in the RD slicing
rd_window = 'blackmanharris' # window used in the RD slicing
num_slices = 10              # slices to stitch. Each adds 0.1 s. Match it to the time the target stays in the range bin

# Target body motion. Set target_velocity = None to find the body line automatically
target_velocity = 8.7        # m/s, bistatic velocity of the body (f = 2v/lambda, as before)
notch_velocity = 1           # m/s, total width of the body notch (as before)
edge_trim_s = 0.05           # s cut from each end of the record after the notch filter

# Geometry, used only for the tip speed / blade length estimates
beta_deg = 19.0              # bistatic angle at the target
theta_deg = 90.0             # angle between rotor axis and bisector. ~90 deg for a helicopter
                             # rotor. For a nose propeller this is roughly delta (heading angle)

# Spectrogram settings
NFFT_long = 256              # 0.125 s window -> 8 Hz resolution, shows the spectral lines
NFFT_short = 64              # ~31 ms window -> blade flash figure (same as your original plots)
NFFT_flash = 16              # ~7.8 ms window, used only to MEASURE the flash interval. Use 8 for propellers
nfft_short_padded = 256      # zero padding for the flash MEASUREMENT window only. The figures are
                             # not padded: padding interpolates the noise into blobs and worms
dynamic_range_db = 25        # colour scale range below the 99.5th percentile (as in your original script)
colour_floor_above_noise_db = 6  # anything less than this above the median (noise) level is drawn dark
doppler_plot_limit_hz = 700  # Doppler axis shown is +/- this. Set to None for the full +/- prf/2

# Search limits for the measurements
rotor_rate_search_hz = (10, 250)       # where to look for the spectral line spacing F_r
flash_interval_search_ms = (2, 60)     # where to look for the time between blade flashes
extent_threshold_db = 10               # Doppler extent = bins this far above the noise floor
acf_peak_threshold = 0.2               # an autocorrelation peak must stand this far above its
                                       # neighbouring dips (its 'prominence') to count
blade_counts_to_test = [2, 3, 4, 5, 6]

# Figure size for the paper. 1.72 in wide fits four panels across a two-column
# IEEE page (figure*). Use 3.45 in wide for a single-column figure.
fig_width_in = 1.72
fig_height_in = 1.45
font_size = 7
show_colorbar = False        # the 0 to -40 dB range can be stated in the caption instead
short_plot_seconds = 0.5     # the short-window figure only shows this much time, so the flashes are visible

os.makedirs(save_directory, exist_ok=True)

# Load the data
rd_slice_list = np.load(rd_slice_matrix_path)
print(f"RD Slice List Shape: {rd_slice_list.shape}")
if not np.iscomplexobj(rd_slice_list):
    print("WARNING: the RD matrix is not complex, so the IFFT back to slow time will not be valid")

# ==========================================
# 2. Time-Domain Reconstruction (Stitching)
# ==========================================
samples_per_cpi = int(cpi_length * prf)                              # 2048
cpi_stride = int(np.round(samples_per_cpi * (1 - cpi_overlap)))      # 205, same as DSP_functions
rd_window_values = signal.get_window(rd_window, samples_per_cpi)

# The middle cpi_stride samples of each slice line up end to end with the next slice
keep_start = (samples_per_cpi - cpi_stride) // 2
keep_end = keep_start + cpi_stride

last_slice = min(slice_under_test + num_slices, rd_slice_list.shape[0])
slow_time_list = []

for i in range(slice_under_test, last_slice):
    doppler_slice = rd_slice_list[i, :, range_bin]

    # Undo the zero-centering and go back to slow time (as in the original script)
    slow_time_slice = fft.ifft(fft.ifftshift(doppler_slice))

    # Undo the RD window, then keep only the middle part of this slice
    slow_time_slice = slow_time_slice / rd_window_values
    slow_time_list.append(slow_time_slice[keep_start:keep_end])

stitched_slow_time = np.concatenate(slow_time_list)
N_samples = len(stitched_slow_time)
t = np.arange(N_samples) / prf
start_time_s = (slice_under_test * cpi_stride + keep_start) / prf    # time from the start of the capture
print(f"Stitched slow time: {N_samples} samples = {N_samples / prf:.2f} s, "
      f"starting {start_time_s:.2f} s into the capture")

# ==========================================
# 3. Bulk Motion Shifting
# ==========================================
# Doppler spectrum of the whole stitched record (before any shift)
raw_spectrum = fft.fftshift(fft.fft(stitched_slow_time * signal.get_window('hann', N_samples)))
raw_freqs = fft.fftshift(fft.fftfreq(N_samples, d=1 / prf))

if target_velocity is None:
    # Take the strongest Doppler line as the body (ignore anything within 1 Hz of zero)
    search = np.abs(raw_freqs) > 1
    body_doppler_hz = raw_freqs[search][np.argmax(np.abs(raw_spectrum[search]))]
else:
    body_doppler_hz = 2 * target_velocity / wavelength

shifted_slow_time = stitched_slow_time * np.exp(-1j * 2 * np.pi * body_doppler_hz * t)

# ==========================================
# 4. Main Body Cancellation (Notching) - now BEFORE the STFT
# ==========================================
# The body now sits at 0 Hz, so a high-pass filter removes it. filtfilt runs the
# filter forwards and backwards, so there is no phase shift or time delay.
notch_width_hz = (2 * notch_velocity) / wavelength
b, a = signal.butter(4, (notch_width_hz / 2) / (prf / 2), btype='highpass')
notched_slow_time = signal.filtfilt(b, a, shifted_slow_time, padlen=N_samples - 1)

# The filter leaves a short transient at each end of the record, so cut off edge_trim_s
# from both ends (this shortens the record by 2 * edge_trim_s)
trim = int(edge_trim_s * prf)
shifted_slow_time = shifted_slow_time[trim:N_samples - trim]
notched_slow_time = notched_slow_time[trim:N_samples - trim]
N_samples = len(notched_slow_time)
start_time_s = start_time_s + trim / prf

# Doppler spectra before and after the notch (Hann windowed), used for the measurements
hann = signal.get_window('hann', N_samples)
freqs = fft.fftshift(fft.fftfreq(N_samples, d=1 / prf))
df = freqs[1] - freqs[0]
shifted_spectrum = np.abs(fft.fftshift(fft.fft(shifted_slow_time * hann)))
notched_spectrum = np.abs(fft.fftshift(fft.fft(notched_slow_time * hann)))
in_notch = np.abs(freqs) < (notch_width_hz / 2)
body_peak_db = 20 * np.log10(np.max(shifted_spectrum[in_notch]))
md_peak_db = 20 * np.log10(np.max(notched_spectrum[~in_notch]))

# ==========================================
# 5. Short-Time Fourier Transforms (long and short window)
# ==========================================
def make_spectrogram(x, nperseg, nfft):
    f, times, Z = signal.stft(x, fs=prf, window='hann', nperseg=nperseg, noverlap=nperseg - 1,
                              nfft=nfft, return_onesided=False, boundary=None, padded=False)
    f = fft.fftshift(f)
    Z = fft.fftshift(Z, axes=0)
    power_db = 20 * np.log10(np.maximum(np.abs(Z), 1e-12))
    # 0 dB = 99.5th percentile, like your original script, so the noise floor sits dark
    power_db = power_db - np.percentile(power_db, 99.5)
    return f, times, power_db

f_long, t_long, S_long_db = make_spectrogram(notched_slow_time, NFFT_long, NFFT_long)
f_short, t_short, S_short_db = make_spectrogram(notched_slow_time, NFFT_short, NFFT_short)   # no padding, like your original
f_flash, t_flash, S_flash_db = make_spectrogram(notched_slow_time, NFFT_flash, nfft_short_padded)

# ==========================================
# 6. Measurements
# ==========================================
# 6a. Doppler extent B: average the long spectrogram over time and find the
#     outermost bins that sit extent_threshold_db above the noise floor
mean_long_db = 10 * np.log10(np.mean(10 ** (S_long_db / 10), axis=1))
noise_floor_db = np.percentile(mean_long_db, 10)   # 10th percentile, so a wide rotor does not raise it
occupied = np.where(mean_long_db > noise_floor_db + extent_threshold_db)[0]
if len(occupied) > 5:
    f_upper = f_long[occupied[-1]]
    f_lower = f_long[occupied[0]]
else:
    # Nothing above the threshold: fall back to the whole Doppler axis
    print("WARNING: no micro-Doppler found above the threshold, so B is not valid")
    f_upper, f_lower = f_long[-1], f_long[0]
doppler_extent_hz = f_upper - f_lower

# 6b. Spectral line spacing F_r: autocorrelation of the notched Doppler spectrum
#     (only the part inside the extent B). Equally spaced lines give
#     autocorrelation peaks at multiples of the spacing.
in_extent = (freqs >= f_lower) & (freqs <= f_upper) & ~in_notch
mag = notched_spectrum[in_extent]
mag = mag - np.mean(mag)
spec_acf = np.correlate(mag, mag, mode='full')[len(mag) - 1:]
spec_acf = spec_acf / spec_acf[0]
spec_lags_hz = np.arange(len(spec_acf)) * df
in_range = (spec_lags_hz >= rotor_rate_search_hz[0]) & (spec_lags_hz <= rotor_rate_search_hz[1])
# As for the flashes below, take the FIRST strong peak (the 2x, 3x ... peaks can be bigger)
spec_peaks, _ = signal.find_peaks(spec_acf, prominence=acf_peak_threshold)
spec_peaks = spec_peaks[in_range[spec_peaks]]
if len(spec_peaks) > 0:
    rotor_rate_hz = spec_lags_hz[spec_peaks[0]]
else:   # no clear peak: fall back to the biggest value in the search range
    rotor_rate_hz = spec_lags_hz[in_range][np.argmax(spec_acf[in_range])]

# 6c. Flash interval: sum the short spectrogram power over Doppler to get a
#     "flash envelope" in time, then autocorrelate it.
envelope = np.sum(10 ** (S_flash_db / 10), axis=0)
env = envelope - np.mean(envelope)
env_acf = np.correlate(env, env, mode='full')[len(env) - 1:]
env_acf = env_acf / env_acf[0]
env_lags_ms = np.arange(len(env_acf)) / prf * 1e3
in_range = (env_lags_ms >= flash_interval_search_ms[0]) & (env_lags_ms <= flash_interval_search_ms[1])
# Take the FIRST strong peak, not the biggest one. With an odd number of blades the
# approaching and receding flashes differ in strength, so the peak at two flash
# intervals can be bigger than the peak at one.
acf_peaks, _ = signal.find_peaks(env_acf, prominence=acf_peak_threshold)
acf_peaks = acf_peaks[in_range[acf_peaks]]
if len(acf_peaks) > 0:
    flash_interval_ms = env_lags_ms[acf_peaks[0]]
else:   # no clear peak: fall back to the biggest value in the search range
    flash_interval_ms = env_lags_ms[in_range][np.argmax(env_acf[in_range])]

# 6d. Individual flashes: peaks in the envelope, and their width (duration)
noise_level = np.median(envelope)
min_spacing = max(1, int(0.6 * flash_interval_ms * 1e-3 * prf))
flash_peaks, _ = signal.find_peaks(envelope, height=4 * noise_level, distance=min_spacing)
if len(flash_peaks) > 0:
    widths = signal.peak_widths(envelope, flash_peaks, rel_height=0.5)[0]
    flash_duration_ms = np.median(widths) / prf * 1e3
else:
    flash_duration_ms = np.nan
short_window_ms = NFFT_flash / prf * 1e3

# 6e. Parity. Even blades: the approaching and receding blades flash at the SAME time,
#     so the positive and negative Doppler flashes line up. Odd blades: they alternate,
#     so the negative flashes sit halfway between the positive ones.
#     Cross-correlate the positive-Doppler and negative-Doppler flash envelopes and see
#     where the peak is: lag ~0 -> even, lag ~half a period -> odd.
flash_rate_hz = 1e3 / flash_interval_ms
ratio = flash_rate_hz / rotor_rate_hz          # ~1 (even, or odd with one side weak), ~2 (odd)
side_band = np.abs(f_flash) > notch_width_hz / 2
env_pos = np.sum(10 ** (S_flash_db[(f_flash > 0) & side_band] / 10), axis=0)
env_neg = np.sum(10 ** (S_flash_db[(f_flash < 0) & side_band] / 10), axis=0)
env_pos = env_pos - np.mean(env_pos)
env_neg = env_neg - np.mean(env_neg)
period = int(round(prf / rotor_rate_hz))       # samples in one F_r period
xcorr = np.correlate(env_pos, env_neg, mode='full')
centre = len(env_neg) - 1
lags = np.arange(-(period // 2), period // 2 + 1)
best_lag = lags[np.argmax(xcorr[centre + lags])]
pos_neg_offset = abs(best_lag) / period        # 0 -> even, 0.5 -> odd
parity = "even" if pos_neg_offset < 0.25 else "odd"
# F_r from the flashes: the flash rate is 1x or 2x F_r, whichever is closer
rotor_rate_from_flashes_hz = flash_rate_hz / max(1, round(ratio))

# 6f. Tip speed (from B), then rpm and blade length for each blade count
beta = np.deg2rad(beta_deg)
theta = np.deg2rad(theta_deg)
tip_speed = wavelength * doppler_extent_hz / (4 * np.sin(theta) * np.cos(beta / 2))
predicted_flash_ms = 2 * np.sin(theta) / doppler_extent_hz * 1e3 if doppler_extent_hz > 0 else np.nan

# ==========================================
# 7. Print and save the results
# ==========================================
lines = [
    f"Target: {target}",
    f"Capture: {directory}",
    f"Range bin {range_bin} ({range_bin * 20} m bistatic range), slices {slice_under_test}-{last_slice - 1}",
    f"Record: {N_samples / prf:.2f} s starting {start_time_s:.2f} s into the capture",
    f"Carrier {centre_freq / 1e6:.1f} MHz, wavelength {wavelength:.3f} m, PRF {prf} Hz",
    "",
    "--- Body ---",
    f"Body Doppler: {body_doppler_hz:.1f} Hz  (bistatic velocity {body_doppler_hz * wavelength / 2:.2f} m/s)",
    f"Body velocity / cos(beta/2): {body_doppler_hz * wavelength / 2 / np.cos(beta / 2):.2f} m/s  (= v cos(delta))",
    f"Strongest micro-Doppler bin is {body_peak_db - md_peak_db:.1f} dB below the body peak",
    "",
    "--- Micro-Doppler ---",
    f"Doppler extent: {f_lower:.0f} Hz to {f_upper:.0f} Hz, B = {doppler_extent_hz:.0f} Hz "
    f"(>{extent_threshold_db} dB above the noise floor)",
    f"Spectral line spacing F_r: {rotor_rate_hz:.2f} Hz  (+/- {df:.2f} Hz frequency bin)",
    f"Flash interval: {flash_interval_ms:.2f} ms  (+/- {1e3 / prf:.2f} ms sample)  -> flash rate {flash_rate_hz:.2f} Hz",
    f"Flash rate / F_r = {ratio:.2f}  (2 means odd blades; 1 means even, or odd with one side weak)",
    f"Positive vs negative flash offset = {pos_neg_offset:.2f} of a period ({abs(best_lag) / prf * 1e3:.1f} ms)"
    f"  ->  {parity} number of blades  (0 = even, 0.5 = odd)",
    f"Flashes detected: {len(flash_peaks)}, median duration {flash_duration_ms:.2f} ms "
    f"(short STFT window is {short_window_ms:.2f} ms, so durations below this are not resolved)",
    f"Predicted flash duration 2 sin(theta) / B: {predicted_flash_ms:.2f} ms",
    "",
    f"--- Rotor estimates (beta = {beta_deg} deg, theta = {theta_deg} deg) ---",
    f"Tip speed L*Omega = lambda B / (4 sin(theta) cos(beta/2)) = {tip_speed:.1f} m/s",
    "rpm (lines) uses F_r from the line spacing. rpm (flashes) uses F_r from the flash",
    "interval, which is usually more precise for short records.",
    "Blades   rpm (lines)   rpm (flashes)   blade length (m)",
]
# F_r from the flash interval: flash rate = F_r for even blades, 2 F_r for odd blades
for n_blades in blade_counts_to_test:
    omega = 2 * np.pi * rotor_rate_from_flashes_hz / n_blades
    rpm_lines = 60 * rotor_rate_hz / n_blades
    rpm_flashes = 60 * rotor_rate_from_flashes_hz / n_blades
    blade_length = tip_speed / omega
    match = "" if (n_blades % 2 == 0) == (parity == "even") else "  (parity does not match)"
    lines.append(f"{n_blades:>4}   {rpm_lines:11.1f}   {rpm_flashes:13.1f}   {blade_length:10.2f}{match}")

if abs(f_upper) > 0.9 * prf / 2 or abs(f_lower) > 0.9 * prf / 2:
    lines.append("WARNING: the micro-Doppler reaches the edge of the Doppler axis, so B may be aliased")

results_text = "\n".join(lines)
print(results_text)
results_name = f"{target_tag}_slice_{slice_under_test}_bin_{range_bin}_results.txt"
with open(os.path.join(save_directory, results_name), 'w') as results_file:
    results_file.write(results_text)

# ==========================================
# 8. Paper-sized spectrogram figures
# ==========================================
plt.rcParams.update({'font.size': font_size, 'font.family': 'serif'})

def save_paper_spectrogram(times, f, power_db, name, plot_seconds=None):
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    # Bottom of the colour scale: just above the noise floor, but never more than dynamic_range_db down
    colour_min = max(-dynamic_range_db, np.median(power_db) + colour_floor_above_noise_db)
    pc = ax.pcolormesh(times, f, power_db, cmap='viridis', shading='auto',
                       vmin=colour_min, vmax=0, rasterized=True)
    ax.set_xlabel('Time (s)', labelpad=1)
    ax.set_ylabel('Doppler (Hz)', labelpad=1)
    ax.tick_params(length=2, pad=1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_xlim(times[0], times[-1])
    if plot_seconds is not None:
        ax.set_xlim(times[0], times[0] + plot_seconds)
    if doppler_plot_limit_hz is not None:
        ax.set_ylim(-doppler_plot_limit_hz, doppler_plot_limit_hz)
    if show_colorbar:
        fig.colorbar(pc, ax=ax, label='Power (dB)')
    # bbox_inches='tight' trims the white border around the axes
    fig.savefig(os.path.join(save_directory, name + '.pdf'), bbox_inches='tight', pad_inches=0.01)
    fig.savefig(os.path.join(save_directory, name + '.png'), dpi=600, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved {name}.pdf and .png in {save_directory}")

base_name = f"{target_tag}_slice_{slice_under_test}_bin_{range_bin}"
save_paper_spectrogram(t_long, f_long, S_long_db, f"{base_name}_md_long_{NFFT_long}")
save_paper_spectrogram(t_short, f_short, S_short_db, f"{base_name}_md_short_{NFFT_short}", short_plot_seconds)

# ==========================================
# 9. Diagnostic figure (to check the measurements, not for the paper)
# ==========================================
plt.rcParams.update({'font.size': 9})
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(freqs + body_doppler_hz, 20 * np.log10(shifted_spectrum + 1e-12), lw=0.6, label='before notch')
axs[0, 0].plot(freqs + body_doppler_hz, 20 * np.log10(notched_spectrum + 1e-12), lw=0.6, label='after notch')
axs[0, 0].axvline(body_doppler_hz, color='r', ls='--', lw=0.8, label='body')
axs[0, 0].axvspan(f_lower + body_doppler_hz, f_upper + body_doppler_hz, color='g', alpha=0.15, label='extent B')
axs[0, 0].set_xlabel('Doppler (Hz)')
axs[0, 0].set_ylabel('Power (dB)')
axs[0, 0].set_title('Doppler spectrum of the stitched record (unshifted axis)')
axs[0, 0].legend()

axs[0, 1].plot(t_flash, 10 * np.log10(envelope), lw=0.6)
axs[0, 1].plot(t_flash[flash_peaks], 10 * np.log10(envelope[flash_peaks]), 'rx')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Power (dB)')
axs[0, 1].set_title(f'Flash envelope (short window), {len(flash_peaks)} flashes')

axs[1, 0].plot(spec_lags_hz, spec_acf, lw=0.6)
axs[1, 0].axvline(rotor_rate_hz, color='r', ls='--', lw=0.8)
axs[1, 0].set_xlim(0, rotor_rate_search_hz[1] * 2)
axs[1, 0].set_xlabel('Frequency lag (Hz)')
axs[1, 0].set_title(f'Spectrum autocorrelation, F_r = {rotor_rate_hz:.2f} Hz')

axs[1, 1].plot(env_lags_ms, env_acf, lw=0.6)
axs[1, 1].axvline(flash_interval_ms, color='r', ls='--', lw=0.8)
axs[1, 1].set_xlim(0, flash_interval_search_ms[1] * 2)
axs[1, 1].set_xlabel('Time lag (ms)')
axs[1, 1].set_title(f'Envelope autocorrelation, flash interval = {flash_interval_ms:.2f} ms')

fig.suptitle(f'{target}: measurement checks')
fig.tight_layout()
fig.savefig(os.path.join(save_directory, f"{base_name}_diagnostics.png"), dpi=150)
plt.show()
