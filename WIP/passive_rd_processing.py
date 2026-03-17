import os
import numpy as np
import DSP_functions as DSP
from DSP_functions import create_rd_video
from scipy import signal

# Below are the parameters of the radar capture and fixed processing parameters. Generally don't change these apart from
# the CF/Decimation factor which will be capture dependent.
c = 3e8
centre_freq = 570e6  # active centre frequency
decimation_factor = 32  # Decimation factor used during capture
Fs = 3.84e9  # Sample rate after decimation
d_Fs = Fs / (8 * decimation_factor)  # Decimated sample rate
seg_s = 2048 # Number of segments per second, similar to PRF
seg_percent = 100 #Number of segments used in xcorr

#These are the changable parameters of the processing, they will affect the correlations.
min_range = 0  # Minimum range in meters
min_lag_samples = np.ceil(min_range * d_Fs / c).astype(int)  # Minimum lag in samples
max_range = 10000  # Maximum range in meters
max_lag_samples = np.ceil(max_range * d_Fs / c).astype(int)  # Maximum lag in samples

#These are the parameters of the range-Doppler processing, they will affect the range-Doppler map and subsequent video.
PRF = seg_s # Pulse Repetition Frequency in Hz used in RD slicing function, segments per second is passive equivalent in this case
cpi_length = 1 # Coherent Processing Interval in seconds, how long to average the RD map for
cpi_overlap = 0.9 # Overlap of the CPI, 0.8 means 80% overlap
rd_window = 'blackmanharris' # Window function for Doppler processing, can be 'hamming', 'blackman', 'blackmanharris', etc.
wavelength = c / centre_freq  # Wavelength of the radar signal

#These are the parameters of the range-Doppler video creation, they will affect the video output.
CMAP = 'viridis'  # Colormap for the range-Doppler video
unfiltered_VIDEO_FILENAME = f'{max_range}_0.1_2048_1_wiener_filtered_30db_cropped'  # Output video filename
unfiltered_VMIN = -30  # Min value for the color scale (dB) - adjust as needed
unfiltered_VMAX = 0  # Max value for the color scale (dB)
vid_x_lim = [0, max_range]  # X-axis limits for the video

#Decide here if you want to batch process or process individual datasets
batch_process_flag = False # Set to True to process all datasets in the directory, False to process a single dataset


correlation_exists_flag = False
filtered_iq_exists_flag = True

experiment_directory = fr'G:\ARESTOR_DATA\Ulm_trial\tests\day_3' # Directory containing the experiment data

if batch_process_flag:
    captures = os.listdir(experiment_directory)
else:
    captures = ['ulm_day_3_pc12_3']  # Set to a specific capture directory if not batch processing

for capture in captures:

    experiment_codename = capture  # Set the experiment codename to the current capture directory

    directory = fr'{experiment_directory}\{experiment_codename}\passive_data_test_1'
    if correlation_exists_flag:
        if not filtered_iq_exists_flag:
            cc_matrix = np.load(directory + r'\cc_matrix.npy')

            capture_period = 70  # Calculate the duration of the capture in seconds
            print("Unfiltered Cross-Correlation Matrix and Self-Ambiguity Matrix successfully loaded from memory. No Need to load IQ")
        else:
            cc_matrix = np.load(directory + r'\fixed_filtered_cc_matrix.npy')
            capture_period = 70  # Calculate the duration of the capture in seconds
            print("Filtered Cross-Correlation Matrix successfully loaded from memory. No Need to load IQ")
    elif filtered_iq_exists_flag:
        ref_iq = np.load(directory + r'\ref_iq.npy')
        surv_iq = np.load(directory + r'\0.1_2048_1_weiner_filtered_surv_channel.npy')
        capture_period = len(ref_iq) / d_Fs  # Calculate the duration of the capture in seconds
        print(f"IQ Matrices successfully loaded from memory. reference data shape: {ref_iq.shape}, surveillance data shape:"
              f" {surv_iq.shape}")
        cc_matrix = DSP.passive_processing(surv_iq, ref_iq, seg_s, seg_percent,
                                                                      d_Fs, max_lag_samples, min_lag_samples)
        print(f"Completed cross-correlation of data. cc_matrix shape: "
              f"{cc_matrix.shape}, cc_lags shape: . Deleting raw data to save memory.")
        print(f"Saving filtered CC_matrix to {directory}")
        np.save(directory + fr'\{max_range}_filtered_cc_matrix.npy', cc_matrix)

    else:
        ref_data = np.load(directory + r'\passive_ref_output.npy')
        surv_data = np.load(directory + r'\passive_surv_output.npy')
        print("Loaded data")
        ref_iq = (ref_data[0::2] + 1j * ref_data[1::2]).astype(np.complex64)  # Combine into complex numbers
        surv_iq = (surv_data[0::2] + 1j * surv_data[1::2]).astype(np.complex64)  # Combine into complex numbers
        del surv_data, ref_data  # Release memory if I/Q parts are large
        np.save(directory + r'\ref_iq', ref_iq)
        np.save(directory + r'\surv_iq', surv_iq)
        capture_period = len(ref_iq) / d_Fs  # Calculate the duration of the capture in seconds
        print(f"Completed loading and iq conversion of data. reference data shape: {ref_iq.shape}, surveillance data shape:"
              f" {surv_iq.shape}")
        #Now we will segment the data and cross-correlate.
        #self_ambg_matrix, cc_matrix, cc_lags = DSP.passive_processing(surv_iq, ref_iq, seg_s, seg_percent,
        #                                                              d_Fs, max_lag_samples, min_lag_samples)

        cc_matrix = DSP.passive_processing(surv_iq, ref_iq, seg_s, seg_percent,
                                           d_Fs, max_lag_samples, min_lag_samples)
        print(f"Completed cross-correlation of data. cc_matrix shape: "
              f"{cc_matrix.shape}, cc_lags shape: . Deleting raw data to save memory.")

        print("Saving CC_matrix to harddrive")
        np.save(directory + fr'\{max_range}_cc_matrix.npy', cc_matrix)


        del ref_iq, surv_iq  # Release memory if I/Q parts are large
        print("Deleted raw data, moving on to range-Doppler processing.")
        #We also need to make a range axis using our lag samples and the decimated sample rate.
        #range_axis_m = (cc_lags * c) / d_Fs  # Range axis in meters, using the formula range = c * lag / Fs. Due to TDOA, no need for factor of 2
    no_range_bins = int(cc_matrix.shape[1])
    print("Number of range bins:", no_range_bins)
    range_axis_m = np.linspace(min_range, max_range, no_range_bins)

    #Now we will perform range-Doppler processing on the cross-correlation matrix. First we will get the range-Doppler
    #slices for the cross-correlation matrix, then we will get the self_ambg slices for the self-ambiguity matrix and
    #subsequent DSI processing.

    rd_slices, velocity_axis, number_cpi, cpi_stride = DSP.passive_range_Doppler_slicing(cc_matrix, cpi_length,
                                                                                      cpi_overlap, PRF, rd_window,
                                                                                      wavelength)
    rd_slice_directory = directory + r'\rd_slices'
    os.makedirs(rd_slice_directory, exist_ok=True)
    np.save(rd_slice_directory + fr'\{max_range}_rd_slice_matrix.npy', rd_slices)
    print(f"Completed range-Doppler slicing of cross-correlation matrix. rd_slices shape: {len(rd_slices)}. RD Slices saved to {rd_slice_directory}.")

    rd_plots = []
    for slice in rd_slices:
        slice_plot = (20 * np.log10(np.abs(slice) / np.max(np.abs(slice))))
        rd_plots.append(slice_plot)
        print(f"Appended slice {len(rd_plots)}")
    print(f"Finished creating RD plots. RD plots shape: {len(rd_plots)}")
    #Now we will create the range-Doppler video from the range-Doppler slices.
    unfiltered_rd_video_flag = True  # Set to True to create unfiltered RD video.
    if unfiltered_rd_video_flag:

        FPS = 1 / ((capture_period / number_cpi))  # Frames per second for the output video
        video_directory = directory + r'\rd_videos'  # Directory to save the output videos
        os.makedirs(video_directory, exist_ok=True)
        #VIDEO_SAVE = os.path.join(video_directory, unfiltered_VIDEO_FILENAME)
        #doppler_limit = [velocity_axis[0], velocity_axis[-1]]
        doppler_limit = [-150, 150]
        DSP.create_rd_video(FPS, rd_plots, PRF, cpi_stride, range_axis_m, velocity_axis,
                            unfiltered_VMIN, unfiltered_VMAX, CMAP, vid_x_lim, video_directory,
                            unfiltered_VIDEO_FILENAME, doppler_limit, sharpening=False)


print("Have you checked the centre frequency and decimation factor are correct for this capture?")


