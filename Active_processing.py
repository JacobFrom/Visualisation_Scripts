import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from classes import waveform

# Define waveform parameters
signal = waveform(2.44e9, 80e6, 500e-6, 1024)
# Define sample rate, capture period and number of pulses
Fs = 3.84e9
capture_period = 5

# Load in deramped data from ARESTOR
data = np.load(r'E:\Trial_2_Day_2_tests\Run4\trial_2_test_2\active_data\active_deramp_output_0_2.npy')

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
num_rows = fft_data.shape[0] #Number of frequency bins
positive_beat_freqs = fft_data[:num_rows // 2, :]
#Print statements below act as sanity checks

print("Positive BF shape:", np.shape(positive_beat_freqs))
PBF_max_value = np.max(np.abs(positive_beat_freqs))
PBF_min_value = np.min(np.abs(positive_beat_freqs))
print("Positive BF Max Value:", PBF_max_value)
print("Positive BF Min Value:", PBF_min_value)

#MTI Filtering
num_pulses = fft_data.shape[1] #Number of pulses
MTI_data = np.zeros_like(positive_beat_freqs) #Create empty array for MTI
signal.range_bins = np.shape(MTI_data[0]) #Assign number of range bins to the signal object
for i in range(1, num_pulses): #Iterate through pulses missing first pulse as nothing to subtract from
    MTI_data[:, i] = positive_beat_freqs[:, i] - positive_beat_freqs[:, i - 1]
print("MTI Data Shape:", np.shape(MTI_data)) # Verify shape of MTI data
#Three Pulse MTI
h = np.array([1, -2, 1]) #Define filter coefficients

"""
#Attempt at IIR filter:
filter_order = 10
nyquist_rate = Fs / 2
normalised_cutoff = 0.021
cutoff_frequency = normalised_cutoff * nyquist_rate
b, a = sp.butter(filter_order, cutoff_frequency / nyquist_rate, btype = 'high')
n_range_bins = len(MTI_data)
for i in range(n_range_bins):
    MTI_data[i, :] = sp.filtfilt(b, a, MTI_data[i, :])
"""



#Derive appropriate range and time axis
freq = np.fft.fftfreq(fft_data.shape[0], 1. / (Fs / (8 * 256)))
slope = signal.chirp_bandwidth / signal.chirp_period
ranges = (freq * 3e8) / (2 * slope)
print("Shape of range bins:", ranges.shape)

range_axis = ranges[:ranges.shape[0] // 2]
range_bin_size = ranges[1]
time_axis = np.linspace(0, capture_period, num_pulses)
MTI_time_axis = np.linspace(0, capture_period, num_pulses)
MTI_plot_flag = False
RTI_plot_flag = True

if RTI_plot_flag:
    #RTI Plot
    RTI_plot = np.transpose(20 * np.log10(np.abs(positive_beat_freqs) / np.max(np.abs(positive_beat_freqs))))
    print("RTI Plot Shape:", RTI_plot.shape)
    plt.figure()
    plt.pcolormesh(range_axis, time_axis, RTI_plot, vmin=-60, vmax=0)
    plt.xlim([0, 80])
    plt.xlabel('Range [m]')
    plt.ylabel('Time [s]')
    plt.colorbar()
    plt.show()
MTI_max_value = np.max(np.abs(MTI_data))
MTI_min_value = np.min(np.abs(MTI_data))
print("MTI Max Value:", MTI_max_value)
print("MTI Min Value:", MTI_min_value)

#MTI RTI Plot
if MTI_plot_flag:
    MTI_plot = np.transpose(20 * np.log10(np.abs(MTI_data) / np.max(np.abs(MTI_data))))
    print("MTI Plot Shape:", MTI_plot.shape)
    plt.figure()
    plt.pcolormesh(range_axis, MTI_time_axis, MTI_plot, vmin=-30, vmax=0)
    plt.xlim([0, 120])
    plt.xlabel('Range [m]')
    plt.ylabel('Time [s]')
    plt.colorbar()
    plt.show()




