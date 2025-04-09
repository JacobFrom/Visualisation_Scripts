import os
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal
import imageio


directory = r'G:\trial_3_data\run_7\active_test_1'

data = np.load(directory + r'\1.npy')

print(data.shape)

iq_data = data[:, 0::2] + 1j*data[:, 1::2] #Convert the data to complex numbers to make a baseband array.

# Below are the parameters of the capture
s_band = 2.44e9 #active centre frequency
c_band = 5.8e9

c = 3e8
centre_freq = s_band #active centre frequency
bandwidth = 80e6 #chirp bandwidth
chirp_period = 500e-6 #chirp length
prf = 1024  # Pulse Repitition Frequency

Fs = 3.84e9
decimated_sample_rate = Fs / (8*256) #Decimated sample rate

capture_period = 30

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

print("Shape of time axis:", time_axis)

#Code for removing close in clutter is in elipses
#If you wish to crop some data out use following code:

#first_N_bins_to_crop = 10 #Number of range bins to crop from the start of the data
#cropped_data = positive_beat_frequencies[:, first_N_bins_to_crop:num_range_bins]
#print("Shape of cropped_data:", cropped_data.shape)

#To generate the RTI plot, change flag value to True

RTI_plot_flag = False

#code for pcolormesh code is inside the elipses
# if RTI_plot_flag:
#     RTI_plot = (20 * np.log10(np.abs(positive_beat_frequencies) / np.max(np.abs(positive_beat_frequencies))))
#     #or to plot cropped data to remove close in clutter
#     #RTI_plot = (20 * np.log10(np.abs(cropped_data) / np.max(np.abs(cropped_data))))
#     plt.figure()
#     plt.pcolormesh(range_axis, time_axis, RTI_plot, vmin=-50, vmax=0)
#     #To plot cropped bins:
#     #plt.pcolormesh(range_axis[first_N_bins_to_crop:], time_axis, RTI_plot, vmin=-50, vmax=0)
#     plt.xlim([0, 120])
#     plt.xlabel('Range [m]')
#     plt.ylabel('Time [s]')
#     plt.colorbar(label = 'Relative Magnitude [dB]')
#     plt.show()

# #can use below if you want imshow:
if RTI_plot_flag:

    RTI_plot = (20 * np.log10(np.abs(positive_beat_frequencies) / np.max(np.abs(positive_beat_frequencies))))
    plt.imshow(RTI_plot, aspect='auto', origin="lower", extent=[range_axis[0], range_axis[-1], time_axis[0], time_axis[-1]], vmin=-50, vmax=0)
    plt.xlabel('Range [m]')
    plt.ylabel('Time [s]')
    plt.colorbar(label = 'Magnitude [dB]')
    plt.title('Range-Time Intensity Plot')
    plt.xlim([0, 100])
    #Now save the plot to the data directory
    rti_plot_filename = 'RTI_plot.png'
    full_rti_save_path = os.path.join(directory, rti_plot_filename)
    print(f"Saving RTI plot to: {full_rti_save_path}")
    try:
        plt.savefig(full_rti_save_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving RTI plot: {e}")
    #plt.show()
    plt.close()

# #Now lets make and apply an MTI filter and produce another plot:

#Creating 2 pulse canceller

MIT_plot_flag = False

if MIT_plot_flag:
    mti_filtered_data = positive_beat_frequencies[1:, :] - positive_beat_frequencies[:-1, :] #Subtract the previous pulse from the current pulse
    print("Shape of mti_filtered_data:", mti_filtered_data.shape)
    #need to re-establish the time axis as now we have 1 less pulse, we have lost the very first pulse of the capture
    mti_time_axis = np.linspace(0, capture_period, num_pulses-1) #Create a time axis from 0 to the maximum time in seconds

    print("Shape of mti_time_axis:", mti_time_axis)

    MTI_plot = (20 * np.log10(np.abs(mti_filtered_data) / np.max(np.abs(mti_filtered_data))))
#pcolormesh code is inside the elipses
    # plt.figure()
    # plt.pcolormesh(range_axis, mti_time_axis, MTI_plot, vmin=-50, vmax=0)
    # plt.xlim([0, 120])
    # plt.xlabel('Range [m]')
    # plt.ylabel('Time [s]')
    # plt.colorbar(label = 'Relative Magnitude [dB]')
    # plt.show()

    #or for imshow instead:

    plt.imshow(MTI_plot, aspect='auto', origin="lower", extent=[range_axis[0], range_axis[-1], mti_time_axis[0], mti_time_axis[-1]], vmin=-50, vmax=0)
    plt.xlabel('Range [m]')
    plt.ylabel('Time [s]')
    plt.colorbar(label = 'Relative Magnitude [dB]')
    plt.title('MTI-Filtered Range-Time Intensity Plot')
    plt.xlim([0, 100])
    #Now save the plot to the data directory
    mti_plot_filename = 'MTI_plot.png'
    full_mti_save_path = os.path.join(directory, mti_plot_filename)
    print(f"Saving MTI plot to: {full_mti_save_path}")
    try:
        plt.savefig(full_mti_save_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving MTI plot: {e}")
    #plt.show()
    plt.close()


#Now we perform some range doppler processing and make a video

rd_processing_flag = True

if rd_processing_flag:
    #step 1, fft along pulse dimension to get range doppler information
    rd_matrix = np.fft.fft(positive_beat_frequencies, axis=0) #Perform FFT along the pulse dimension
    print("Shape of rd_matrix:", rd_matrix.shape)
    #step 2, create a range doppler matrix, axes 0 now represents doppler information across whole capture for each range bin
    rd_matrix = np.fft.fftshift(rd_matrix, axes=0) #Shift the zero frequency component to the center of the spectrum
    print("Shape of rd_matrix after shift:", rd_matrix.shape)
    #Step 3, convert to db scale and normalise
    rd_plot = (20 * np.log10(np.abs(rd_matrix) / np.max(np.abs(rd_matrix))))
    #Step 4, create doppler axis, first in frequency, then convert to velocity
    pulse_interval = 1 / prf #Calculate the pulse interval in seconds
    #print("Pulse Interval:", pulse_interval)
    doppler_axis_unshifted = np.fft.fftfreq(num_pulses, d=pulse_interval) #Calculate the doppler axis in Hz
    doppler_axis = np.fft.fftshift(doppler_axis_unshifted) #Shift the doppler axis to match the shifted rd_matrix
    #print("Shape of doppler axis:", doppler_axis.shape)
    #print(f"Doppler frequency range from {doppler_axis[0]} to {doppler_axis[-1]} Hz")

    #Step 5, convert to velocity
    wavelength = c / centre_freq #Calculate the wavelength in metres
    velocity_axis = (doppler_axis * wavelength) / 2 #Calculate the velocity axis in m/s
    print(f"Velocity range (m/s): {velocity_axis[0]:.2f} m/s to {velocity_axis[-1]:.2f} m/s") #Here the :.2f formats the float to 2 decimal places
    #Step 6, create the plot
    plt.figure()
    plt.imshow(rd_plot, aspect='auto', origin="lower", extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]], vmin=-100, vmax=0)
    plt.xlabel('Range [m]')
    plt.ylabel('Velocity [m/s]')
    plt.colorbar(label = 'Magnitude [dB]')
    plt.title('Range-Doppler Spectogram Plot')
    plt.xlim([0, 100])
    plt.ylim([-25, 25])
    #plt.show()
    plt.close()




