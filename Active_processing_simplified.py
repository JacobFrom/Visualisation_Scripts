import os
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from classes import waveform
from dsp_functions import rangeDopplerslice, create_video

# Define waveform parameters
signal = waveform(2.44e9, 80e6, 500e-6, 1024)
# Define sample rate, capture period and number of pulses
Fs = 3.84e9
capture_period = 15

# Load in deramped data from ARESTOR
directory = r'G:\Trial_2_Day_2_tests\Run8\trial_2_test_8\active_data'

#data = np.load(r'E:\Trial_2_Day_2_tests\Run8\trial_2_test_8\active_data\active_deramp_output_0_8.npy')

data = np.load(directory + r'\active_deramp_output_0_8.npy')
iq_data = data[0::2] + 1j*data[1::2] #Convert the data to complex numbers to make a baseband array.
deramped_data = iq_data.reshape((int(signal.prf*capture_period)), -1).T #Reshape the baseband array to the correct dimensions.
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

#First stage of pre processing is complete. Now we perform some sanity checks and organise the data for further processing.

num_pulses = fft_data.shape[1] #Number of pulses
signal.range_bins = np.shape(positive_beat_freqs[0]) #Assign number of range bins to the signal object
#Derive appropriate range and time axis
freq = np.fft.fftfreq(fft_data.shape[0], 1. / (Fs / (8 * 256)))
slope = signal.chirp_bandwidth / signal.chirp_period #Derive the slope of the chirp signal, This is used to calculate the range of each range bin in metres.
ranges = (freq * 3e8) / (2 * slope) #Calculate the range of each range bin in metres.

print("Shape of range bins:", ranges.shape)

#Pre define the range and time axis for the Range-Time plot

range_axis = ranges[:ranges.shape[0] // 2]
range_bin_size = ranges[1]
time_axis = np.linspace(0, capture_period, num_pulses)

#To generate the RTI plot, change flag value to True

RTI_plot_flag = False

if RTI_plot_flag:
    #RTI Plot
    RTI_plot = np.transpose(20 * np.log10(np.abs(masked_positive_beat_freqs) / np.max(np.abs(masked_positive_beat_freqs))))
    print("RTI Plot Shape:", RTI_plot.shape)
    plt.figure()
    plt.pcolormesh(range_axis_masked, time_axis_masked, RTI_plot, vmin=-30, vmax=-10)
    plt.xlim([20, 100])
    plt.xlabel('Range [m]')
    plt.ylabel('Time [s]')
    plt.colorbar()
    plt.show()



micro_doppler_flag = True

if micro_doppler_flag:
    #First stage is to define range bins of interest
    start_range_bin = 0
    end_range_bin = 10
