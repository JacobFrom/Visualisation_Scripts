import os
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from classes import waveform
import cv2
from dsp_functions import rangeDopplerslice, create_video

raw_ref_data = np.load(r'G:\passive_data\old_3_sec_wall_cap\ref_stream\passive_ref_output_0_3.npy') #Load in reference channel data
raw_surv_data = np.load(r'G:\passive_data\old_3_sec_wall_cap\ref_stream\passive_ref_altered.npy') # Load in surveillance channel data


Fs = 3.84e9 #Sample Rate of the ADC
capture_period = 3 #Capture length of the data
decimation_block = 4 #Decimation block size in ARESTOR build
decimation_c_c = 8 #Decimation specified in command and control
max_range = 100 #Maximum range of interest in metres
centre_frequency = 530e6  #Central frequency of the capture (usually the same as central f of IoO)
segments_per_sec = 1000  #How many segments per second for batch processing, analagous to PRF for active
seg_percent = 100 #What percentage of the segments to batch process, 100% is all segments, lower is faster but loses some coherent integration gains

ref_iq = raw_ref_data[0::2] + 1j*raw_ref_data[1::2] #Convert the data to complex numbers to make a baseband array.

surv_iq = raw_surv_data[0::2] + 1j*raw_surv_data[1::2] #Convert the data to complex numbers to make a baseband array.

#Create a false target at a given range in the surveillance channel:

shift_factor = 30 #How many samples to shift the target by
passive_sample_rate = Fs / (decimation_block * decimation_c_c) #Calculate the true sample rate of passive data after decimation
range_bin_size = (1/passive_sample_rate) * (sp.constants.c / 2) #Calculate the size of a single range bin, in metres
print("Expected target range = ", shift_factor * range_bin_size, " m")

surv_shifted = np.roll(surv_iq, shift_factor) #Shift the surveillance channel data by the shift factor

#Batch processing for passive data set
segment_size = int(passive_sample_rate/segments_per_sec) #Calculate the size of each segment

seg_ref_channel = np.reshape(ref_iq, (segment_size, int(ref_iq.shape[0] / segment_size))) #Reshape the reference channel data into segments

seg_surv_channel = np.reshape(surv_shifted, (segment_size, int(surv_shifted.shape[0] / segment_size))) #Reshape the surveillance channel data into segments

cross_correlation_size = int(segment_size * (seg_percent / 100)) #Calculate the size of the cross correlation

decimated_ref_channel = seg_ref_channel[:cross_correlation_size, :] #Decimate the reference channel data
decimated_sur_channel = seg_surv_channel[:cross_correlation_size, :] #Decimate the surveillance channel data

#Time domain based cross correlation:
zero_padding = 1 #Zero padding factor
num_rows = zero_padding * (2 * max_range + 1) #Number of rows in the cross correlation matrix
num_cols = int(ref_iq.shape[0] / segment_size) #Number of columns in the cross correlation matrix

cc_matrix = np.zeros((num_rows, num_cols), dtype=complex) #Create an empty cross correlation matrix
ref_self_ambg = np.zeros((num_rows, num_cols), dtype=complex) #Create an empty reference self ambiguity matrix

max_lag = int(np.ceil(max_range / range_bin_size)) #Calculate the maximum lag in samples

for i in range(seg_ref_channel.shape[1]): #Iterate through the segments
    segment_sur = decimated_sur_channel[:, i] #Select the ith segment of the surveillance channel
    segment_ref = decimated_ref_channel[:, i] #Select the ith segment of the reference channel
    result = np.zeros(2 * max_lag + 1, dtype=complex) #Create an empty array to store the cross correlation result
    result_ambg = np.zeros(2 * max_lag + 1, dtype=complex) #Create an empty array to store the reference self ambiguity result
    for lag in range(-max_lag, max_lag + 1): #Iterate through the lags
        if lag <0: #Compute cross correlation and self ambiguity for negative lags by calculating the dot product of the overlapping segments
            result[lag + max_lag] = np.dot(segment_sur[-lag:], segment_ref[:len(segment_ref) + lag])
            result_ambg[lag + max_lag] = np.dot(segment_ref[-lag:], segment_ref[:len(segment_ref) + lag])
        else: #Compute cross correlation and self ambiguity for positive lags by calculating the dot product of the overlapping segments
            result[lag + max_lag] = np.dot(segment_sur[:len(segment_sur) - lag], segment_ref[lag:])
            result_ambg[lag + max_lag] = np.dot(segment_ref[:len(segment_ref) - lag], segment_ref[lag:])
    cc_matrix[:, i] = result #Store the result in the cross correlation matrix
    ref_self_ambg[:, i] = result_ambg #Store the result in the reference self ambiguity matrix

bin_zero = np.argmax(cc_matrix[:, 0]) #Find the zero bin of the cross correlation matrix
cc_matrix = cc_matrix[bin_zero:bin_zero + max_range + 1, :] #Select only the positive range bins
ref_self_ambg = ref_self_ambg[bin_zero:bin_zero + max_range + 1, :] #Select only the positive range bins

#Range doppler slices:
C = sp.constants.c #Speed of light in m/s
cpi = 0.5 #Coherent Processing Interval in seconds
cpi_overlap = 0.8 #Overlap between CPIs
doppler_window = 'blackmanharris' #Window type for Doppler FFT
lambda_c = C / centre_frequency #Wavelength in m
zero_padding = 1    #Zero padding factor for Doppler FFT 1 = none, 2 = 100% zero padding
max_range = 100 #Maximum range for plotting
prf = segments_per_sec #Pulse Repetition Frequency in Hz

number_cpi, pulses_per_cpi, range_doppler_slices = rangeDopplerslice(cc_matrix, cpi, prf, cpi_overlap,
zero_padding, doppler_window)

range_doppler_slice_flag = True
slice_index = 20
slice_to_plot = range_doppler_slices[slice_index]
doppler_axis = np.linspace(-prf/2, prf/2, slice_to_plot.shape[1])
doppler_velocity_axis = doppler_axis * lambda_c / 2
doppler_range_axis = np.linspace(0, max_range, cc_matrix.shape[0])

if range_doppler_slice_flag:
    plt.figure(figsize=(12, 8))
    plt.title("Range-Doppler Slice")
    plt.xlabel("Range bin number")
    plt.ylabel("Doppler Velocity [m/s]")
    plt.xlim([0, 30])
    plt.ylim([-20, 20])
    plt.imshow(20 * np.log10(np.abs(slice_to_plot)), aspect='auto', cmap='jet', extent=[doppler_range_axis[0], doppler_range_axis[-1], doppler_velocity_axis[0], doppler_velocity_axis[-1]])
    plt.colorbar(label='Power (dB)')
    plt.show()

