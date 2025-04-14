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
capture_period = 20
prf = 1024
# Load in deramped data from ARESTOR
directory = r'/home/jacob/Documents/C_Band_Data/C_band_testing_ota'

#data = np.load(r'E:\Trial_2_Day_2_tests\Run8\trial_2_test_8\active_data\active_deramp_output_0_8.npy')

data = np.load(directory + r'/jacob_waving_moving.npy')
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

RTI_plot_flag = False

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



micro_doppler_flag = True

if micro_doppler_flag:
    #First stage is to define duration and range bin/bins of interest
    selected_range_bin = 5 #Select the range bin that corresponds to where the movement under analysis occured
    stft_data = positive_beat_freqs[selected_range_bin, :] #This assigns the stft_data variable to the range bin of choice and contains all pulses
    print("Shape of STFT Data:", stft_data.shape)

    #Next we define the parameters for -the coherent processing intervals:
    cpi = 0.5 #Define the coherent processing interval in seconds
    cpi_overlap = cpi / 2  #Define the overlap between each coherent processing interval, standard is 50% cpi
    pulses_per_cpi = int(cpi * prf)  # Define the number of pulses per cpi, this will be used for the windowing function
    cpi_slide = int(np.round(pulses_per_cpi * (1-cpi_overlap))) #This will define the number of pulses to slide the window by
    window_func = get_window('hann', pulses_per_cpi) #Define the windowing function to be applied to the data
    zero_padding = 1 #Define the factor for zero padding the Doppler FFT
    num_cpis = int(np.ceil((positive_beat_freqs.shape[1] - pulses_per_cpi) / cpi_slide)) #Calculate the number of cpis to be processed
    print("Number of CPIs:", num_cpis)
    num_doppler_bins = pulses_per_cpi * zero_padding #Define the number of Doppler bins
    doppler_bin_size = prf / pulses_per_cpi #Define the size of each Doppler bin
    #if doppler_bins % 2 == 0:
        #doppler_bins += 1 #Ensure the number of Doppler bins is odd
    print("Number of Doppler Bins:", num_doppler_bins)
    #Finally, define the order of operations for the STFT and the Axis
    micro_doppler_slices = np.zeros((int(num_cpis), num_doppler_bins), dtype=complex)
    print("Shape of Micro-Doppler Slices:", micro_doppler_slices.shape)

    doppler_axis = np.arange(-prf/2, prf/2, doppler_bin_size)

    doppler_time_axis = np.linspace(0, capture_period, num_cpis)

    for i in range(num_cpis):
        start = i * cpi_slide
        end = start + pulses_per_cpi
        cpi_data = stft_data[start:end]
        windowed_data = cpi_data * window_func
        stfted_data = np.fft.fft(windowed_data, axis=0)
        micro_doppler_slices[i] = stfted_data.T
    print("Shape of Micro-Doppler Slices:", micro_doppler_slices.shape)
#Now we can plot the micro-doppler slices
    plt.figure()
    plt.pcolormesh(doppler_time_axis, doppler_axis, 20 * np.log10(np.abs(micro_doppler_slices)), shading='Auto')
    plt.xlabel('CPI Number')
    plt.ylabel('Doppler Frequency')
    plt.colorbar()
    plt.show()

print('Hello')




