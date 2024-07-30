import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from classes import waveform

# Define waveform parameters
signal = waveform(2.44e9, 80e6, 500e-6, 1024)
# Define sample rate, capture period and number of pulses
Fs = 3.84e9
capture_period = 5

# Load in deramped data from ARESTOR
deramped_data = np.load(r'E:\Trial_2_Day_2_tests\Run4\trial_2_test_2\active_data\active_deramp_output_0_2.npy')
# Apply windowing function

window = np.hamming(deramped_data.shape[0])
windowed_data = deramped_data * window[:, np.newaxis]
# Perform FFT on windowed data
fft_data = np.fft.fft(windowed_data, axis=0)
# Keep only positive beat frequencies
num_rows = fft_data.shape[0]
positive_beat_freqs = fft_data[:num_rows // 2, :]

#MTI Filtering
num_pulses = positive_beat_freqs.shape[1]
MTI_data = np.zeros_like(positive_beat_freqs)
signal.range_bins = len(MTI_data)
for i in range(1, num_pulses):
    MTI_data[:, i] = positive_beat_freqs[:, i] - positive_beat_freqs[:, i-1]


