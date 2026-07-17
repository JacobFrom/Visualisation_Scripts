import numpy as np
import scipy as sp
import scipy.fft as fft
import scipy.signal as signal
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Loading
# ==========================================
directory = r'G:\ARESTOR_DATA\Ulm_trial\tests\day_2\ulm_day_2_3020\passive_data_test_1'
# Fixed the string formatting for the path
rd_slice_matrix_path = fr'{directory}\rd_slices\4000_rd_slice_matrix.npy'

# Load the data (Make sure this file exists at this path!)
rd_slice_list = np.load(rd_slice_matrix_path)
print(f"RD Slice List Shape: {rd_slice_list.shape}")

target = "Inspire 2"

# Radar Parameters
decimation_factor = 32
c = sp.constants.c
Fs = 3.84e9
d_Fs = Fs / (8 * decimation_factor)
centre_freq = 570e6
wavelength = c / centre_freq
target_velocity = -8
notch_velocity = 2


f_shift_hz = (2 * (target_velocity) / wavelength)
# We use your hardcoded PRF as requested
prf = 2048
range_bin = 100
slice_under_test = 525
NFFT = 1024
Noverlap = NFFT - 1
target_range = 20*range_bin
micro_doppler_save_name = f"slice_{slice_under_test}_micro_doppler_spectrum_target_at_{range_bin}_FFT_Length_{NFFT}_olap_{Noverlap}.png"


# NOTE: Expanded the end_frame so you actually have enough slow-time data to see micro-Doppler!
# Adjust this based on how many slices make up your 0.6 seconds of total_time_s
start_frame = slice_under_test
end_frame = slice_under_test + 1

# ==========================================
# 2. Time-Domain Reconstruction (Stitching)
# ==========================================
slow_time_list = []

for i in range(start_frame, end_frame):
    doppler_slice = rd_slice_list[i, :, range_bin]

    # Undo the zero-centering to prepare for IFFT
    doppler_unshifted = fft.ifftshift(doppler_slice)

    # Revert to slow-time
    slow_time_slice = fft.ifft(doppler_unshifted)

    # Append to our list (faster than concatenating in the loop)
    slow_time_list.append(slow_time_slice)

# Stitch them all together into one continuous 1D array
stitched_slow_time = np.concatenate(slow_time_list)
print(f"Stitched slow time matrix shape: {stitched_slow_time.shape}")

# ==========================================
# 3. Bulk Motion Shifting
# ==========================================
N_samples = len(stitched_slow_time)
t = np.arange(N_samples) / prf

# Define the shift needed to bring your target's main body to 0 Hz.
# (e.g., If the 40 m/s bulk motion creates a 150 Hz Doppler shift, set this to 150)

shifted_slow_time = stitched_slow_time * np.exp(-1j * 2 * np.pi * f_shift_hz * t)

# ==========================================
# 4. Short-Time Fourier Transform (STFT)
# ==========================================

frequencies, times, Zxx = signal.stft(
    shifted_slow_time,
    fs=prf,
    window='hann',
    nperseg=NFFT,
    noverlap=Noverlap,
    nfft=NFFT,
    return_onesided=False # Mandatory for complex radar data
)

# Move the new 0 Hz (the bulk motion) to the center of the arrays
frequencies = fft.fftshift(frequencies)
Zxx = fft.fftshift(Zxx, axes=0)

# ==========================================
# 5. Main Body Cancellation (Notching)
# ==========================================
notch_width_hz = ((2*notch_velocity) /wavelength) # Total width of the suppression band (e.g., -10 Hz to +10 Hz)
notch_indices = np.abs(frequencies) < (notch_width_hz / 2)

# Suppress the main body signal (using a tiny number instead of 0 to avoid log(0) errors)
Zxx[notch_indices, :] = 1e-12

# ==========================================
# 6. Dynamic Range & Decibel Conversion
# ==========================================
Zxx_mag = np.abs(Zxx)
Zxx_mag = np.maximum(Zxx_mag, 1e-12) # Floor to prevent log(0)
Zxx_db = 25 * np.log10(Zxx_mag)

# Set dynamic range based on data distribution
# 99.5th percentile prevents single massive clutter spikes from washing out the plot
vmax = np.percentile(Zxx_db, 99.5)
vmin = vmax - 30 # 45 dB dynamic range (adjust this if you want more/less contrast)

# ==========================================
# 7. Plotting
# ==========================================
plt.figure(figsize=(14, 6))

# pcolormesh is the standard for rendering these types of matrices
pc = plt.pcolormesh(
    times,
    frequencies,
    Zxx_db,
    # Smooths the bins. Change to 'flat' if you want to see exact FFT bins.
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)

plt.colorbar(pc, label='Power (dB)')
plt.ylabel('Doppler Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title(f'Micro-Doppler Spectrogram of {target} at {target_range:.0f}m FFT Length: {NFFT}')
plt.tight_layout()
plt.show()
plt.savefig(fr"{directory}\micro_dopp_figures\{micro_doppler_save_name}")
print(f"Micro-Doppler spectrogram saved as {micro_doppler_save_name} in {directory}\micro_dopp_figures")
