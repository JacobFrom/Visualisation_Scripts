import os
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal
import matplotlib.animation as animation
import time
import processing_functions as pf

# --- Configuration ---
experiment_codename = 'run_7'

capture_duration = 30
centre_freq = 498e6  #Centre freq of DVB-T capture DVB-T

decimator_ip = 32 #Choose between 4x or 32x
decimator_tile = 8


base_data_directory = r'G:\trial_3_data'
data_subdirectory = 'passive_data_test_1'
output_video_directory = fr'D:\ARESTOR_DATA\trial_3_rd_videos\{experiment_codename}\passive_test_1'

input_data_dir = os.path.join(base_data_directory, experiment_codename, data_subdirectory)


ref_filename = 'passive_ref_output.npy'
surv_filename = 'passive_surv_output.npy'

ref_path = os.path.join(input_data_dir, ref_filename)
surv_path = os.path.join(input_data_dir, surv_filename)

os.makedirs(output_video_directory, exist_ok=True) # Ensure output dir exists

ref_data_raw = np.load(ref_path)
surv_data_raw = np.load(surv_path)


#De interleave and convert to complex

ref_i = ref_data_raw[0::2] # Select elements 0, 2, 4,... (Real/I)
ref_q = ref_data_raw[1::2] # Select elements 1, 3, 5,... (Imaginary/Q)
ref_iq = (ref_i + 1j * ref_q).astype(np.complex64)   # Combine into complex numbers
del ref_i, ref_q # Optional: Release memory if I/Q parts are large

# Surveillance Channel
# Check if data length is even
if surv_data_raw.shape[0] % 2 != 0:
    print("Warning: Surveillance data has odd number of samples. Trimming last sample.")
    surv_data_raw = surv_data_raw[:-1]

surv_i = surv_data_raw[0::2] # Select elements 0, 2, 4,... (Real/I)
surv_q = surv_data_raw[1::2] # Select elements 1, 3, 5,... (Imaginary/Q)
surv_iq = (surv_i + 1j * surv_q).astype(np.complex64)   # Combine into complex numbers
del surv_i, surv_q # Optional: Release memory

# Optional: Delete raw data if memory is tight, but only AFTER successful conversion
del ref_data_raw, surv_data_raw

print(f"Shape of complex reference data (ref_iq): {ref_iq.shape}") # Should be (450000000,)
print(f"Shape of complex surveillance data (surv_iq): {surv_iq.shape}") # Should be (450000000,)

#Calculate Effective Sample Rate

Fs_original = 3.84e9 # Original sample rate
decimation_factor = decimator_ip * decimator_tile #Decimation factor from decimator ip block multiplied by tile decimation

# Calculate Effective Sample Rate
Fs_eff = Fs_original / decimation_factor
print(f"Effective Sample Rate (Fs_eff): {Fs_eff / 1e6:.2f} MHz ({Fs_eff} Hz)")

# --- Processing Parameters ---
C = 3e8  # Speed of light in m/s
wavelength = C / centre_freq  # Wavelength in meters

# CPI settings
cpi_duration_sec = 0.5 # Coherent Processing Interval in seconds
cpi_overlap = 0.8 # Overlap between consecutive CPIs (0 to 1)

# Range/Delay settings
max_range_interest_meters = 280 # Max bistatic range of interest (20 km example)
# Convert max range to max delay. NOTE: For passive bistatic radar, Range = c * delay.
# This 'Range' is the path length difference: (Tx -> Target -> Rx) - (Tx -> Rx)
max_delay_sec = max_range_interest_meters / C

print(f"Maximum delay calculated based on range of interest: {max_delay_sec * 1e6:.2f} us")

#Doppler settings
doppler_resolution_hz = 1 / cpi_duration_sec # Doppler resolution in Hz

samples_per_cpi = int(np.ceil(cpi_duration_sec * Fs_eff))
print(f"Samples per CPI: {samples_per_cpi}")
cpi_stride = int(samples_per_cpi * (1 - cpi_overlap))

max_delay_samples = int(np.ceil(max_delay_sec * Fs_eff))

#calculate total number of CPIs that can be processed
total_samples = ref_iq.shape[0]

num_cpis = int(np.floor((total_samples - samples_per_cpi) / cpi_stride)) + 1
print(f"Total number of CPIs that can be processed: {num_cpis}")

slow_time_step = cpi_stride / Fs_eff
###################################################################################
#Here is where I would like to add some DSI in the future but for now we skip
#
#
#
#
###################################################################################
#Now to start frequency based cross correlation
start_time = time.time()
# Initialize the output matrix to store results
# Rows = CPIs, Columns = Delay Bins
correlation_matrix = np.zeros((num_cpis, max_delay_samples), dtype=np.complex64)

n_fft = int(2**np.ceil(np.log2(samples_per_cpi))) #Pad to next power of 2

print(f"Using FFT length: {n_fft}")

for i in range(num_cpis):
    start_index = i * cpi_stride
    end_index = start_index + samples_per_cpi

    ref_cpi = ref_iq[start_index:end_index].astype(np.complex64)
    surv_cpi = surv_iq[start_index:end_index].astype(np.complex64)
    #Now begin fft based cross correlation
    fft_surv = np.fft.fft(surv_cpi, n=n_fft)
    #Need the complex conjugate fft of ref signal
    fft_ref_conj = np.conj(np.fft.fft(ref_cpi, n=n_fft))
    #Perform the cross correlation
    correlation_freq = fft_surv * fft_ref_conj
    #Now perform the inverse fft
    correlation_time = np.fft.ifft(correlation_freq)
    # Store the result in the correlation matrix
    correlation_matrix[i, :] = correlation_time[0:max_delay_samples]

elapsed_time = time.time() - start_time
print(f"--- Cross-correlation loop finished in {elapsed_time:.2f} seconds ---")
print(f"Output correlation matrix shape: {correlation_matrix.shape}") # Should be (num_cpis, max_delay_samples)

# --- Range-Doppler Processing ---
N_doppler_choice = 32
rd_overlap = 0.95
rd_cpi = N_doppler_choice * slow_time_step
# Generate range doppler slices


rd_slices_list, velocity_axis = pf.get_passive_rd_slices(correlation_matrix, slow_time_step, wavelength, rd_cpi, rd_overlap)

print(f"Number of RD slices: {len(rd_slices_list)}")

# --- Plotting ---
delay_axis_sec = np.arange(max_delay_samples) / Fs_eff
range_axis_m = delay_axis_sec * C #No division by 2 for bistatic since it is time delay of arrival calculated with respect to Tx and Rx

print(f"Range Axis (m) = {range_axis_m}")
print(f"Velocity Axis (m/s) = {velocity_axis[0]:.2f} m/s to {velocity_axis[-1]:.2f} m/s")




