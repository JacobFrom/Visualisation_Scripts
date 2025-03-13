import os
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from classes import waveform
from scipy.signal import get_window

# Define waveform parameters
signal = waveform(5.8e9, 80e6, 500e-6, 1024)
# Define sample rate, capture period and number of pulses
Fs = 3.84e9
capture_period = 5
prf = 1024
centre_freq = 2.44e9
# Load in deramped data from ARESTOR
directory = r'E:\loki_testing_day_1'

#data = np.load(r'E:\Trial_2_Day_2_tests\Run8\trial_2_test_8\active_data\active_deramp_output_0_8.npy')

data = np.load(directory + r'\s_band_test_1\active_deramp_output.npy')
iq_data = data[:, 0::2] + 1j*data[:, 1::2] #Convert the data to complex numbers to make a baseband array.
#deramped_data = iq_data.reshape((int(signal.prf*capture_period)), -1).T

deramped_data = iq_data.T #Reshape the baseband array to the correct dimensions.
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

print("Shape of positive_beat_freqs:", positive_beat_freqs.shape)
#First stage of pre processing is complete. Now we perform some sanity checks and organise the data for further processing.

num_pulses = fft_data.shape[1] #Number of pulses
print("Number of Pulses:", num_pulses)
range_bins = np.shape(positive_beat_freqs[0]) #Assign number of range bins to the signal object
#Derive appropriate range and time axis
freq = np.fft.fftfreq(fft_data.shape[0], 1. / (Fs / (8 * 256)))
slope = signal.chirp_bandwidth / signal.chirp_period #Derive the slope of the chirp signal, This is used to calculate the range of each range bin in metres.
ranges = (freq * 3e8) / (2 * slope) #Calculate the range of each range bin in metres.

print("Shape of range bins:", ranges.shape)

#Pre define the range and time axis for the Range-Time plot

range_axis = ranges[:ranges.shape[0] // 2]

range_bin_size = ranges[1]
print(range_bin_size)
time_axis = np.linspace(0, capture_period, num_pulses)

#To generate the RTI plot, change flag value to True

RTI_plot_flag = True

if RTI_plot_flag:
    #RTI Plot
    RTI_plot = np.transpose(20 * np.log10(np.abs(positive_beat_freqs) / np.max(np.abs(positive_beat_freqs))))
    print("RTI Plot Shape:", RTI_plot.shape)
    plt.figure()
    plt.pcolormesh(range_axis, time_axis, RTI_plot, vmin=-50, vmax=0)
    plt.xlim([0, 50])
    plt.xlabel('Range [m]')
    plt.ylabel('Time [s]')
    plt.colorbar()
    plt.show()



micro_doppler_flag = False

if micro_doppler_flag:
    #First stage is to define duration and range bin/bins of interest
    selected_range_bin = 4 #Select the range bin that corresponds to where the movement under analysis occured
    stft_data = positive_beat_freqs[selected_range_bin, :] #This assigns the stft_data variable to the range bin of choice and contains all pulses
    print("Shape of STFT Data:", stft_data.shape)

    #Next we define the parameters for -the coherent processing intervals:
    cpi = 0.5 #Define the coherent processing interval in seconds
    cpi_overlap = 0.9  #Define the overlap between each coherent processing intervals
    pulses_per_cpi = int(cpi * prf) #How many pulses to include in each CPI, aka fft length/NFFT points
    number_of_overlap = int(cpi_overlap * pulses_per_cpi)
    window_func = get_window('hann', pulses_per_cpi)  # Define the windowing function to be applied to the data

    plt.specgram(stft_data, NFFT=pulses_per_cpi, Fs=Fs, noverlap=number_of_overlap, window=window_func, mode='magnitude', Fc=centre_freq, scale='dB')
    plt.show()
