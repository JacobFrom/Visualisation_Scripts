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
#Print statements below act as sanity checks

print("Positive BF shape:", np.shape(positive_beat_freqs))
PBF_max_value = np.max(np.abs(positive_beat_freqs))
PBF_min_value = np.min(np.abs(positive_beat_freqs))
print("Positive BF Max Value:", PBF_max_value)
print("Positive BF Min Value:", PBF_min_value)

#MTI Filtering
num_pulses = fft_data.shape[1] #Number of pulses
#MTI_data = np.zeros_like(positive_beat_freqs) #Create empty array for MTI
signal.range_bins = np.shape(positive_beat_freqs[0]) #Assign number of range bins to the signal object
#for i in range(num_pulses - 1): #Iterate through pulses missing first pulse as nothing to subtract from
#   MTI_data[:, i] = positive_beat_freqs[:, i + 1] - positive_beat_freqs[:, i]
#print("MTI Data Shape:", np.shape(MTI_data)) # Verify shape of MTI data


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
bin_plot_flag = False
mask_flag = False
range_doppler_flag = False
micro_doppler_flag = False

if mask_flag:
    #Masking of interfering pulses

    # Calculate the power of each pulse
    pulse_power = np.sum(np.abs(positive_beat_freqs) ** 2, axis=0)

    #Identify pulses with power above a certain threshold
    threshold = np.mean(pulse_power) + 10 * np.std(pulse_power)  #Threshold based on plots
    interfering_pulses = pulse_power > threshold

    #Create a mask to exclude the identified pulses
    mask = ~interfering_pulses

    # Apply the mask to create a new array without interfering pulses
    cleaned_positive_beat_freqs = positive_beat_freqs[:, mask]

    #Masking of close in range bins
    num_excluded_bins = 15

    mask_bins = np.arange(cleaned_positive_beat_freqs.shape[0]) >= num_excluded_bins

    masked_positive_beat_freqs = cleaned_positive_beat_freqs[mask_bins, :]
    print("Shape of masked positive beat freqs:", masked_positive_beat_freqs.shape)
    range_axis_masked = ranges[:(ranges.shape[0] // 2) - num_excluded_bins]
    time_axis_masked = np.linspace(0, capture_period, masked_positive_beat_freqs.shape[1])

    #Masked data MTI:
    MTI_freqs = masked_positive_beat_freqs
    MTI_data = np.zeros_like(masked_positive_beat_freqs)
    for i in range(masked_positive_beat_freqs.shape[1] - 4):
        MTI_data[:, i] = (
                MTI_freqs[:, i + 4]    #x(n+4)
                - 4 * MTI_freqs[:, i + 3] #-4x(n+3)
                + 6 * MTI_freqs[:, i + 2] #6x(n+2)
                - 4 * MTI_freqs[:, i + 1] #-4x(n+1)
                + MTI_freqs[:, i])      #x(n)


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
MTI_max_value = np.max(np.abs(MTI_data))
MTI_min_value = np.min(np.abs(MTI_data))
print("MTI Max Value:", MTI_max_value)
print("MTI Min Value:", MTI_min_value)

#MTI RTI Plot
if MTI_plot_flag:
    MTI_plot = np.transpose(20 * np.log10(np.abs(MTI_data) / np.max(np.abs(MTI_data))))
    print("MTI Plot Shape:", MTI_plot.shape)
    plt.figure()
    plt.pcolormesh(range_axis_masked, time_axis_masked, MTI_plot, vmin=-40, vmax=0)
    plt.xlim([0, 120])
    plt.xlabel('Range [m]')
    plt.ylabel('Time [s]')
    plt.colorbar()
    plt.show()
range_bin_idx = 100
range_bin_data = positive_beat_freqs[range_bin_idx, :]
power_range_bin = 20 * np.log10(np.abs(range_bin_data) / np.max(np.abs(range_bin_data)))
if bin_plot_flag:
    plt.figure()
    plt.plot(power_range_bin)
    plt.xlabel('Pulse Number')
    plt.ylabel('Power [dB]')
    plt.title(f'Power Plot for Range bin {range_bin_idx}')
    plt.grid(True)
    plt.show()


if range_doppler_flag:
    #Range Doppler Processing:
    C = sp.constants.speed_of_light #Speed of light in m/s
    cpi = 0.5 #Coherent Processing Interval in seconds
    cpi_overlap = 0.8 #Overlap between CPIs
    doppler_window = 'hann' #Window type for Doppler FFT
    lambda_c = C / signal.centre_freq #Wavelength in m
    zero_padding = 1    #Zero padding factor for Doppler FFT 1 = none, 2 = 100% zero padding
    max_range = 100 #Maximum range for plotting

    number_cpi, pulses_per_cpi, range_doppler_slices = rangeDopplerslice(positive_beat_freqs, cpi, signal.prf, cpi_overlap,
    zero_padding, doppler_window)

range_doppler_slice_flag = False

if range_doppler_slice_flag:
    slice_index = 120
    slice_to_plot = range_doppler_slices[slice_index]
    doppler_axis = np.linspace(-signal.prf / 2, signal.prf / 2, slice_to_plot.shape[1])
    doppler_velocity_axis = doppler_axis * lambda_c / 2
    doppler_range_axis = np.linspace(0, max_range, positive_beat_freqs.shape[0])
    plt.figure(figsize=(12, 8))
    plt.title("Range-Doppler Slice")
    plt.xlabel("Range bin number")
    plt.ylabel("Doppler Velocity [m/s]")
    plt.xlim([0, 30])
    plt.ylim([-20, 20])
    plt.imshow(20 * np.log10(np.abs(slice_to_plot)), aspect='auto', cmap='jet', extent=[doppler_range_axis[0], doppler_range_axis[-1], doppler_velocity_axis[0], doppler_velocity_axis[-1]])
    plt.colorbar(label='Power (dB)')
    plt.show()

if micro_doppler_flag:
    
"""
os.makedirs(directory, exist_ok=True)

video_filename = "active_range-Doppler_video1.avi"
video_path = os.path.join(directory, video_filename)

#Create the video
video_title ="Active Range-Doppler Video"
dynamic_range = 60
max_range_video = 100
max_doppler = 20
frame_rate = 1/(capture_period / number_cpi)

create_video(range_doppler_slices, frame_rate, doppler_range_axis, max_range_video, -doppler_velocity_axis, max_doppler,
             dynamic_range, video_path, video_title)
print("Video created successfully")

"""