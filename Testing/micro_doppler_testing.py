import time
import numpy as np
from CaCFAR import CA_CFAR as CFAR
import DetProcessing as dp
import matplotlib.pyplot as plt
import DSP_functions as DSP

# Define the folder path where the data is stored
directory = fr'G:\ARESTOR_DATA\SIREN\east_fort_day_1\east_fort_capture_10\passive_data_test_1'

#First, set all the parameters for the CFAR processing

# Load the 3D array containing all the Range-Doppler matrices
rd_matrices = np.load(directory + r'\rd_slice_matrix_2.npy')
cc_matrix  = np.load(directory + r'\3km_cc_matrix.npy')
#Set the CA window parameters, format: [window width, window height, guard cell width, guard cell height]
window_params = [8, 8, 2, 2]
# Set the detection threshold in dB
threshold_dB = 12
# Get the shape of a single matrix (rows=Doppler, cols=Range) for configuration
data_shape = rd_matrices[0].shape
# Calculate max range in meters, this assumes 20m per range cell (True for single channel DVB-T passive radar)
max_range_m = data_shape[1] * 20

# Create a linear array for the Range axis (0 to max range)
range_axis_m = np.linspace(0, max_range_m, int(data_shape[1]))
# Set the floor for the dB plot (minimum intensity)
dynamic_range = -50
# Calculate time between pulses based on number of Doppler cells
pulse_interval = 1 / int(data_shape[0])
# Speed of light constant
c = 3e8
# Carrier frequency of the radar
centre_freq = 494e6
# Calculate signal wavelength
wavelength = c / centre_freq
# Generate the frequency axis for Doppler (unshifted)
slice_doppler_axis_unshifted = np.fft.fftfreq(data_shape[0], d=pulse_interval)
# Shift the zero frequency to the center of the array
slice_doppler_axis = np.fft.fftshift(slice_doppler_axis_unshifted)
# Convert the frequency axis to Velocity (m/s)
velocity_axis = (wavelength * slice_doppler_axis) / 2

# Toggle this flag: False = calculate fresh CFAR data; True = load previous results
CFAR_hit_matrix_exists_flag = False

#Next, set all the parameters for the micro-Doppler processing
cpi = 0.5 # Coherent Processing Interval in seconds
cpi_overlap = 0.8 # Overlap of the CPI
prf = int(data_shape[0]) # Pulse Repetition Frequency in Hz derived from number of Doppler bins
decimation_factor = 32 # Decimation factor used during capture
Fs =  (3.84e9) / (8 * decimation_factor) # Sample rate after decimation

micro_doppler_flag: True

# ==========================================
# CFAR PROCESSING SECTION
# ==========================================
if not CFAR_hit_matrix_exists_flag:
    # Initialize empty lists to store the detection results and SNR maps
    hit_results = []
    snr_results = []

    print(f"Processing {len(rd_matrices)} slices...")
    # Start the timer to track performance
    start = time.time()

    # Create the CFAR processor object once with our settings
    ca_cfar = CFAR(window_params, threshold_dB, data_shape)

    # Loop through every single matrix in the stack
    for i, rd_matrix in enumerate(rd_matrices):
        # Run the CFAR algorithm on the current slice
        hit_matrix, rd_snr = ca_cfar(rd_matrix)
        #Set last 3 range bins to false to avoid false detections at max range from CFAR/Cross correlation artefacts
        hit_matrix[:, -3:] = False

        # Save the boolean hits (detections) to our list
        hit_results.append(hit_matrix)
        # Save the SNR values to our list
        snr_results.append(rd_snr)

    # Stop the timer
    end = time.time()

    print(f"CFAR processing time Complete. Time taken: {end - start} seconds. Saving results...")
    # Define file paths for saving the output
    hit_save = directory + r'\cfar_hit_matrix.npy'
    snr_save = directory + r'\cfar_sinr_matrix.npy'

    # Save the list of hit matrices to disk
    np.save(hit_save, hit_results)
    # Save the list of SNR matrices to disk (Corrected from hit_results to snr_results)
    np.save(snr_save, snr_results)
    print(f"CFAR processing complete. Hit matrix and SNR matrix saved in {directory}.")

else:
    print("Loading pre-computed CFAR results from disk...")
    # Load the previously saved boolean hit matrices
    hit_results = np.load(directory + r'\cfar_hit_matrix.npy')
    # Load the previously saved SNR matrices
    snr_results = np.load(directory + r'\cfar_sinr_matrix.npy')
    print("CFAR results loaded.")

# ==========================================
#MICRO-DOPPLER PROCESSING SECTION
# ==========================================
DSP.plot_micro_doppler(100, cc_matrix, 'cc_matrix', cpi, cpi_overlap, prf, Fs, centre_freq, slice_number=31)
