import os
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal


directory = r'E:\trial_3_data\run_5\active_test_2'

data = np.load(directory + r'\2.npy')

print(data.shape)

iq_data = data[:, 0::2] + 1j*data[:, 1::2] #Convert the data to complex numbers to make a baseband array.

# Below are the parameters of the capture
c = 3e8
centre_freq = 5.8e9 #active centre frequency
bandwidth = 80e6 #chirp bandwidth
chirp_period = 500e-6 #chirp length
prf = 1024  # Pulse Repitition Frequency

Fs = 3.84e9
decimated_sample_rate = Fs / (8*256) #Decimated sample rate

capture_period = 20

#Begin the processing by windowing and then FFTing the data

window = np.hamming(iq_data.shape[1]) #Create a hamming window function
print("Window Shape:", np.shape(window))
windowed_data = iq_data * window #Apply hamming window function to the data, np.newaxis adds a new axis to the window array, allowing for element-wise multiplication

print ("Windowed data shape", windowed_data.shape)

#perform fft on the newly windowed data
fft_data = np.fft.fft(windowed_data, axis=1) #Perform FFT on windowed data along the samples

num_pulses = fft_data.shape[0] #Number of pulses
# Keep only positive beat frequencies
num_samples = fft_data.shape[1]
num_range_bins = num_samples // 2 #Number of range bins
positive_beat_frequencies = fft_data[:, :num_range_bins]


print("Shape of positive_beat_freqs:", positive_beat_frequencies.shape)

slope = bandwidth / chirp_period #Derive the slope of the chirp signal, This is used to calculate the range of each range bin in metres.
range_bin_size = c * decimated_sample_rate / (2 * slope * num_samples) #Calculate the range of each range bin in metres.

#print("Ranges:", range_bin_size)

#Derive appropriate range and time axis
range_axis = np.linspace(0, (num_range_bins-1)*range_bin_size, num_range_bins)#Create a range axis from 0 to the maximum range in metres

#print("Shape of range axis:", range_axis)

time_axis = np.linspace(0, capture_period, num_pulses)

#print("Shape of time axis:", time_axis)
#If you wish to crop some data out use following code:

#first_N_bins_to_crop = 10 #Number of range bins to crop from the start of the data
#cropped_data = positive_beat_frequencies[:, first_N_bins_to_crop:num_range_bins]
#print("Shape of cropped_data:", cropped_data.shape)

#To generate the RTI plot, change flag value to True
RTI_plot_flag = True

if RTI_plot_flag:
    RTI_plot = (20 * np.log10(np.abs(positive_beat_frequencies) / np.max(np.abs(positive_beat_frequencies))))
    #or to plot cropped data to remove close in clutter
    #RTI_plot = (20 * np.log10(np.abs(cropped_data) / np.max(np.abs(cropped_data))))
    plt.figure()
    plt.pcolormesh(range_axis, time_axis, RTI_plot, vmin=-50, vmax=0)
    #To plot cropped bins:
    #plt.pcolormesh(range_axis[first_N_bins_to_crop:], time_axis, RTI_plot, vmin=-50, vmax=0)
    plt.xlim([0, 120])
    plt.xlabel('Range [m]')
    plt.ylabel('Time [s]')
    plt.colorbar(label = 'Relative Magnitude [dB]')
    plt.show()

#can use below if you want imshow:
# if RTI_plot_flag:
#
#     RTI_plot = (20 * np.log10(np.abs(positive_beat_frequencies) / np.max(np.abs(positive_beat_frequencies))))
#     plt.imshow(RTI_plot, aspect='auto', extent=[range_axis[0], range_axis[-1], time_axis[0], time_axis[-1]], vmin=-50, vmax=0)
#     plt.xlabel('Range [m]')
#     plt.ylabel('Time [s]')
#     plt.colorbar(label = 'Relative Magnitude [dB]')
#     plt.title('Range-Time Intensity Plot')
#     plt.xlim([0, 120])
#     plt.show()



# #Now lets make apply and MTI and produce another plot:

