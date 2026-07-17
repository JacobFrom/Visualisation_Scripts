import clutterCancellation as CC
import numpy as np

import os
import gc

# =============================================================================
# FIXED PARAMETERS
# =============================================================================
c = 3e8
decimation_factor = 32  # Decimation factor used during capture
Fs = 3.84e9  # Sample rate
d_Fs = Fs / (8 * decimation_factor)  # Decimated sample rate
block_duration = 0.1

filter_taps = 2048
smoothing_weight = 1

batch_process_flag = True  # Set to True to process all datasets in the directories
iq_matrices_exist_flag = False

# =============================================================================
# BATCH CONFIGURATION
# Define your trial directories and their associated centre frequencies here
# =============================================================================
experiments_config = [
    {
        "directory": r'G:\ARESTOR_DATA\SIREN\east_fort_day_3',
        "centre_freq": 518e6
    },
    {
        "directory": r'G:\ARESTOR_DATA\SIREN\east_fort_day_1',
        "centre_freq": 494e6  # Replace with actual frequency
    },
    # Add as many dictionaries as you need...
]

# =============================================================================
# PROCESSING LOOP
# =============================================================================

# 1. Outer loop: Iterate through the different main experiment directories
for exp in experiments_config:
    experiment_directory = exp["directory"]
    centre_freq = exp["centre_freq"]
    wavelength = c / centre_freq  # Recalculate wavelength for the new frequency

    print(f"\n{'=' * 50}")
    print(f"Starting batch process for directory: {experiment_directory}")
    print(f"Centre Frequency: {centre_freq} Hz (Wavelength: {wavelength:.3f} m)")
    print(f"{'=' * 50}")

    if not os.path.exists(experiment_directory):
        print(f"Warning: Directory {experiment_directory} does not exist. Skipping...")
        continue

    # Determine which captures to process in the current directory
    if batch_process_flag:
        captures = os.listdir(experiment_directory)
    else:
        # Note: If batch_process_flag is False, it will look for this exact folder
        # inside EVERY experiment_directory listed above.
        captures = ['ulm_day_3_pc12_5']

        # 2. Inner loop: Iterate through the captures within the current directory
    for capture in captures:
        experiment_codename = capture

        # Construct the full path using os.path.join for safety, or keep your f-string format
        directory = fr'{experiment_directory}\{experiment_codename}\passive_data_test_1'

        if not os.path.exists(directory):
            print(f"Skipping {capture}: 'passive_data_test_1' subfolder not found.")
            continue

        print(f"\nProcessing capture: {capture}")

        if iq_matrices_exist_flag:
            ref_iq = np.load(directory + r'\ref_iq.npy')
            surv_iq = np.load(directory + r'\surv_iq.npy')
            capture_period = len(ref_iq) / d_Fs
            print(
                f"IQ Matrices successfully loaded from memory. ref shape: {ref_iq.shape}, surv shape: {surv_iq.shape}")
        else:
            try:
                ref_data = np.load(directory + r'\passive_ref_output.npy')
                surv_data = np.load(directory + r'\passive_surv_output.npy')
            except FileNotFoundError:
                print(f"Data files not found in {directory}. Skipping...")
                continue

            print("Loaded data")
            ref_iq = (ref_data[0::2] + 1j * ref_data[1::2]).astype(np.complex64)
            surv_iq = (surv_data[0::2] + 1j * surv_data[1::2]).astype(np.complex64)
            del surv_data, ref_data
            gc.collect()

            np.save(directory + r'\ref_iq.npy', ref_iq)
            np.save(directory + r'\surv_iq.npy', surv_iq)
            capture_period = len(ref_iq) / d_Fs

            print(f"Completed loading and iq conversion. ref shape: {ref_iq.shape}, surv shape: {surv_iq.shape}")

        # Run the filter
        filtered_surv_channel, weights = CC.Wiener_SMI_Batch_Process(
            ref_iq, surv_iq, filter_taps, d_Fs, block_duration, smoothing_weight
        )
        print("Finished filtering data. Saving...")

        np.save(directory + fr'\{block_duration}_{filter_taps}_{smoothing_weight}_weiner_filtered_surv_channel.npy',
                filtered_surv_channel)
        np.save(directory + fr'\{block_duration}_{filter_taps}_{smoothing_weight}_weiner_filter_weights.npy', weights)
        print(fr"Successfully saved filtered data to: {directory}")

        # Cleanup memory for the next capture
        del filtered_surv_channel, weights, ref_iq, surv_iq
        gc.collect()
        print("Memory cleared, moving onto next capture.")

print("\nFinished Filtering of all IQ data across all directories.")