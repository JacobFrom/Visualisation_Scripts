import os
import numpy as np
import DSP_functions as DSP
from DSP_functions import create_rd_video
from scipy import signal

# Below are the parameters of the radar capture and fixed processing parameters. Generally don't change these apart from
# the CF/Decimation factor which will be capture dependent.
c = 3e8
centre_freq = 570e6  # active centre frequency
decimation_factor = 32
Fs = 3.84e9  # Sample rate after decimation
d_Fs = Fs / (8 * decimation_factor)  # Decimated sample rate
seg_s = 1000 # Number of segments per second, similar to PRF
seg_percent = 100 #Number of segments used in xcorr

#These are the changable parameters of the processing, they will affect the correlations.
min_range = 0  # Minimum range in meters
min_lag_samples = np.ceil(min_range * d_Fs / c).astype(int)  # Minimum lag in samples
max_range = 4000  # Maximum range in meters
max_lag_samples = np.ceil(max_range * d_Fs / c).astype(int)  # Maximum lag in samples

#These are the parameters of the range-Doppler processing, they will affect the range-Doppler map and subsequent video.
PRF = seg_s # Pulse Repetition Frequency in Hz used in RD slicing function, segments per second is passive equivalent in this case
cpi_length = 0.5 # Coherent Processing Interval in seconds, how long to average the RD map for
cpi_overlap = 0.8 # Overlap of the CPI, 0.8 means 80% overlap
rd_window = 'blackmanharris' # Window function for Doppler processing, can be 'hamming', 'blackman', 'blackmanharris', etc.
wavelength = c / centre_freq  # Wavelength of the radar signal

#These are the parameters of the range-Doppler video creation, they will affect the video output.
CMAP = 'inferno'  # Colormap for the range-Doppler video

unfiltered_VIDEO_FILENAME = 'no_sharpening_test_rd_video'  # Output video filename
unfiltered_VMIN = -50  # Min value for the color scale (dB) - adjust as needed
unfiltered_VMAX = 0  # Max value for the color scale (dB)
vid_x_lim = [0, max_range]  # X-axis limits for the video

#Decide here if you want to batch process or process individual datasets
batch_process_flag = False  # Set to True to process all datasets in the directory, False to process a single dataset

correlation_exists_flag = True

experiment_directory = fr'D:\ARESTOR\Data\Ulm_trial\tests\day_3' # Directory containing the experiment data

if batch_process_flag:
    captures = os.listdir(experiment_directory)
else:
    captures = ['ulm_day_3_pc12_5']  # Set to a specific capture directory if not batch processing

for capture in captures:

    experiment_codename = capture  # Set the experiment codename to the current capture directory

    directory = fr'{experiment_directory}\{experiment_codename}\passive_data_test_1'
    if correlation_exists_flag:
        cc_matrix = np.load(directory + r'\cc_matrix.npy')
        self_ambg_matrix = np.load(directory + r'\self_ambg_matrix.npy')
        capture_period = 70
        print("Cross-Correlation Matrix and Self-Ambiguity Matrix successfully loaded from memory. No Need to load IQ")
    else:
        ref_data = np.load(directory + r'\passive_ref_output.npy')
        surv_data = np.load(directory + r'\passive_surv_output.npy')
        print("Loaded data")
        ref_i = ref_data[0::2]  # Select elements 0, 2, 4,... (Real/I)
        ref_q = ref_data[1::2]  # Select elements 1, 3, 5,... (Imaginary/Q)
        ref_iq = (ref_i + 1j * ref_q)  # Combine into complex numbers
        del ref_i, ref_q  #Release memory if I/Q parts are large
        surv_i = surv_data[0::2]  # Select elements 0, 2, 4,... (Real/I)
        surv_q = surv_data[1::2]  # Select elements 1, 3, 5,... (Imaginary/Q)
        surv_iq = (surv_i + 1j * surv_q)  # Combine into complex numbers
        del surv_i, surv_q  #Release memory if I/Q parts are large
        del ref_data, surv_data #Release memory if I/Q parts are large
        capture_period = len(ref_iq) / d_Fs  # Calculate the duration of the capture in seconds
        print(f"Completed loading and iq conversion of data. reference data shape: {ref_iq.shape}, surveillance data shape:"
              f" {surv_iq.shape}")
        #Now we will segment the data and cross-correlate.
        #self_ambg_matrix, cc_matrix, cc_lags = DSP.passive_processing(surv_iq, ref_iq, seg_s, seg_percent,
        #                                                              d_Fs, max_lag_samples, min_lag_samples)

        self_ambg_matrix, cc_matrix = DSP.passive_processing(surv_iq, ref_iq, seg_s, seg_percent,
                                                                      d_Fs, max_lag_samples, min_lag_samples)
        print(f"Completed cross-correlation of data. self_ambg_matrix shape: {self_ambg_matrix.shape}, cc_matrix shape: "
              f"{cc_matrix.shape}, cc_lags shape: . Deleting raw data to save memory.")

        print("Saving CC_matrix and self_ambg_matrix to harddrive")
        np.save(directory + r'\cc_matrix.npy', cc_matrix)
        np.save(directory + r'\self_ambg_matrix.npy', self_ambg_matrix)

        del ref_iq, surv_iq  # Release memory if I/Q parts are large
        print("Deleted raw data, moving on to range-Doppler processing.")
        #We also need to make a range axis using our lag samples and the decimated sample rate.
        #range_axis_m = (cc_lags * c) / d_Fs  # Range axis in meters, using the formula range = c * lag / Fs. Due to TDOA, no need for factor of 2
    no_range_bins = int(len(cc_matrix))
    print("Number of range bins:", no_range_bins)
    range_axis_m = np.linspace(min_range, max_range, no_range_bins)

    #Now we will perform range-Doppler processing on the cross-correlation matrix. First we will get the range-Doppler
    #slices for the cross-correlation matrix, then we will get the self_ambg slices for the self-ambiguity matrix and
    #subsequent DSI processing.

    rd_slices, velocity_axis, number_cpi, cpi_stride = DSP.range_Doppler_slicing(cc_matrix, cpi_length,
                                                                                      cpi_overlap, PRF, rd_window,
                                                                                      wavelength, radar_mode='passive')
    print(f"Completed range-Doppler slicing of cross-correlation matrix. rd_slices shape: {len(rd_slices)}, moving"
          f" on to self-ambiguity matrix slicing.")
    self_ambg_slices, _, _, _ = DSP.range_Doppler_slicing(self_ambg_matrix, cpi_length, cpi_overlap,
                                                  PRF, rd_window, wavelength, radar_mode='passive')
    print(f"Completed self-ambiguity matrix slicing. self_ambg_slices shape: {len(self_ambg_slices)}, moving on to "
          f" range-Doppler video creation.")

    #Now we will create the range-Doppler video from the range-Doppler slices.
    unfiltered_rd_video_flag = True  # Set to True to create unfiltered RD video.
    if unfiltered_rd_video_flag:
        FPS = 1 / ((capture_period / number_cpi))  # Frames per second for the output video
        VIDEO_SAVE = os.path.join(directory, unfiltered_VIDEO_FILENAME)
        doppler_limit = [velocity_axis[0], velocity_axis[-1]]
        DSP.create_rd_video(FPS, rd_slices, PRF, cpi_stride, range_axis_m, velocity_axis,
                            unfiltered_VMIN, unfiltered_VMAX, CMAP, vid_x_lim, directory,
                            unfiltered_VIDEO_FILENAME, doppler_limit, sharpening=False)


    #DSI CLEANING CODE
    p = 0.1 #Loop Gain
    threshold = 1e-5 #Absolute magnitue threshold
    max_iterations = 100 #Maximum number of CLEANing iterations per slice
    num_slices = int(len(rd_slices))
    middle_cpi_idx = np.floor(num_slices / 2).astype(int) #Because of pythons 0 based indexing this may or may not be wrong. If wrong add +1
    doppler_bins_self = self_ambg_slices[middle_cpi_idx].shape[0]
    zero_doppler_idx_self = (doppler_bins_self + 1) / 2
    self_ambg_zero_doppler_cut = self_ambg_slices[middle_cpi_idx][zero_doppler_idx_self,:]
    #Now send to DSI function

    cleaned_rd_slices = np.zeros(1, num_slices)

    for i in range(num_slices):
        cleaned_rd_slices[i] = simple_zero_doppler_cleaning[]




