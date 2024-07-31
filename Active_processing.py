import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from classes import waveform

# Define waveform parameters
signal = waveform(2.44e9, 80e6, 500e-6, 1024)
# Define sample rate, capture period and number of pulses
Fs = 3.84e9
capture_period = 15

# Load in deramped data from ARESTOR
data = np.load(r'E:\Trial_2_Day_2_tests\Run8\trial_2_test_8\active_data\active_deramp_output_0_8.npy')

iq_data = data[0::2] + 1j*data[1::2] #Convert the data to complex numbers to make a baseband array.
deramped_data = iq_data.reshape((int(signal.prf*capture_period)), 940).T #Reshape the baseband array to the correct dimensions.
#This operation is typically used to organize the samples of a signal into a 2D array where each row corresponds to
# one pulse of the signal. This can be useful for further processing of the signal.
# Apply windowing function
window = np.hamming(deramped_data.shape[0])
#print("Window Shape:", np.shape(window))
windowed_data = deramped_data * window[:, np.newaxis]
# Perform FFT on windowed data
fft_data = np.fft.fft(windowed_data, axis=0)
# Keep only positive beat frequencies
num_rows = fft_data.shape[0]
positive_beat_freqs = fft_data[:num_rows // 2, :]
print("Positive BF shape:", np.shape(positive_beat_freqs))
print("Maximum Value:", np.max(positive_beat_freqs))
#MTI Filtering
num_pulses = positive_beat_freqs.shape[1]
MTI_data = np.zeros_like(positive_beat_freqs)
print("MTI Data Shape:", np.shape(MTI_data))
signal.range_bins = np.shape(MTI_data[0])
for i in range(1, num_pulses):
    MTI_data[:, i] = positive_beat_freqs[:, i] - positive_beat_freqs[:, i - 1]

#Derive appropriate range and time axis

freq = np.fft.fftfreq(fft_data.shape[0], 1. / (Fs / (8 * 256)))
slope = signal.chirp_bandwidth / signal.chirp_period
ranges = (freq * 3e8) / (2 * slope)
print("Shape of range bins:", ranges.shape)

range_axis = ranges[:ranges.shape[0] // 2]
range_bin_size = ranges[1]
time_axis = np.linspace(0, num_pulses * signal.chirp_period, num_pulses)




#RTI Plot
RTI_plot = np.transpose(20 * np.log10(np.abs(positive_beat_freqs) / np.max(np.abs(positive_beat_freqs))))
print("RTI Plot Shape:", RTI_plot.shape)
plt.figure()
plt.pcolormesh(range_axis, time_axis, RTI_plot)
plt.xlim([0, 200])
plt.xlabel('Range [m]')
plt.ylabel('Time [s]')
plt.colorbar()
plt.show()



