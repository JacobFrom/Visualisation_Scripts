import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib.animation as animation
import time
from scipy.signal import get_window, chirp
import scipy.fft as fft


directory = r'G:\ARESTOR_DATA\Ulm_trial\tests\day_3\ulm_day_3_pc12_5\passive_data_test_1'
microdoppler_directory = directory + r'\micro_doppler_plots'
rd_slice_list = np.load(fr'{directory} + \rd_slices\12000_rd_slice_matrix.npy')
print(f"RD Slice List Shape: {rd_slice_list.shape}")

decimation_factor = 32  # Decimation factor used during capture
Fs = 3.84e9  # Sample rate after decimation
d_Fs = Fs / (8 * decimation_factor)  # Decimated sample rate

prf = d_Fs / rd_slice_list.shape[1]  # Pulse Repetition Frequency in Hz derived from number of Doppler bins

range_bin = 331
slice_under_test = 612
total_time_s = 0.6
prf = 2048

NFFT = 64

num_cpi = 10000


start_frame = slice_under_test
end_frame = slice_under_test +1
stitched_slow_time = np.array([])
for i in range(start_frame, end_frame):
    doppler_slice = rd_slice_list[i, :, range_bin]

    # Undo the zero-centering (crucial before IFFT)
    doppler_shifted = fft.ifftshift(doppler_slice)

    # Revert to slow-time using scipy.fft
    slow_time_slice = fft.ifft(doppler_shifted)

    # Stitch them together
    stitched_slow_time = np.concatenate((stitched_slow_time, slow_time_slice))
print(f"Stiched slow time matrix shape: {stitched_slow_time.shape}")

plt.figure(figsize=(14, 6))

# NFFT=64, noverlap=63 for maximum temporal continuity
# sides='twosided' is mandatory for complex radar data
Pxx, freqs, bins, im = plt.specgram(
    stitched_slow_time,
    NFFT=NFFT,
    Fs=prf,
    mode='magnitude',
    noverlap=(NFFT -1),
    sides='twosided',
    scale='dB',
    cmap='viridis',
    #vmin=-50,
    #vmax= 0,
)

