import clutterCancellation as CC
import numpy as np
from numpy import linalg as LA

import os
import time
import numpy as np
import scipy as sp
import DSP_functions as DSP
import ECA_test_functions as TF
import gc


# Below are the parameters of the radar capture and fixed processing parameters. Generally don't change these apart from
# the CF/Decimation factor which will be capture dependent.
c = 3e8
centre_freq = 498e6  # DVB-T centre frequency
decimation_factor = 32  # Decimation factor used during capture
Fs = 3.84e9  # Sample rate
d_Fs = Fs / (8 * decimation_factor)  # Decimated sample rate
block_duration = 0.1

filter_taps = 2048
smoothing_weight = 1

wavelength = c / centre_freq  # Wavelength of the radar signal


batch_process_flag = False  # Set to True to process all datasets in the directory, False to process a single dataset
iq_matrices_exist_flag = False

experiment_directory = fr'G:\ARESTOR_DATA\trial_3_captures\C_band\32x' #Directory containing the experiment data

if batch_process_flag:
    captures = os.listdir(experiment_directory)
else:
    captures = ['run_7']  # Set to a specific capture directory if not batch processing

for capture in captures:

    experiment_codename = capture  # Set the experiment codename to the current capture directory

    directory = fr'{experiment_directory}\{experiment_codename}\passive_data_test_1'

    if iq_matrices_exist_flag:
        # ref_iq = np.load(directory + r'\ref_iq.npy')
        # surv_iq = np.load(directory + r'\surv_iq.npy')
        #mmap mode:
        ref_iq = np.load(directory + r'\ref_iq.npy')
        surv_iq = np.load(directory + r'\surv_iq.npy')
        capture_period = len(ref_iq) / d_Fs  # Calculate the duration of the capture in seconds
        print(f"IQ Matrices successfully loaded from memory. reference data shape: {ref_iq.shape}, surveillance data shape:"
              f" {surv_iq.shape}")
    else:
        ref_data = np.load(directory + r'\passive_ref_output.npy')
        surv_data = np.load(directory + r'\passive_surv_output.npy')
        print("Loaded data")
        ref_iq = (ref_data[0::2] + 1j * ref_data[1::2]).astype(np.complex64)  # Combine into complex numbers
        surv_iq = (surv_data[0::2] + 1j * surv_data[1::2]).astype(np.complex64)  # Combine into complex numbers
        del surv_data, ref_data  #Release memory if I/Q parts are large
        gc.collect()

        np.save(directory + r'\ref_iq.npy', ref_iq)
        np.save(directory + r'\surv_iq.npy', surv_iq)
        capture_period = len(ref_iq) / d_Fs  # Calculate the duration of the capture in seconds

        print(f"Completed loading and iq conversion of data. reference data shape: {ref_iq.shape}, surveillance data shape:"
              f" {surv_iq.shape}")


    filtered_surv_channel, weights = CC.Wiener_SMI_Batch_Process(ref_iq, surv_iq, filter_taps, d_Fs, block_duration, smoothing_weight)
    print("Finished filtering data. Saving...")
    np.save(directory + fr'\{block_duration}_{filter_taps}_{smoothing_weight}_weiner_filtered_surv_channel.npy', filtered_surv_channel)
    np.save(directory + fr'\{block_duration}_{filter_taps}_{smoothing_weight}_weiner_filter_weights.npy', weights)
    print(fr"Successfully saved filtered data to: {directory}\{block_duration}_{filter_taps}_{smoothing_weight}_weiner_filtered_surv_channel.npy")
    del filtered_surv_channel, weights, ref_iq, surv_iq
    gc.collect()
    print("Memory cleared, moving onto next capture (if applicable).")
print("Finished Filtering of all IQ data")
