"""
Passive radar processing driver / example.

This is a thin, readable example of how to use :class:`PassiveProcessor`. All of
the actual signal processing lives inside the class (see
``Passive_processing/passive_processor.py``); here we set the parameters of the
capture, choose how the data is loaded, and call the methods in order.

Three loading routes are supported via the flags below:
  * a pre-computed cross-correlation matrix already exists on disk, or
  * pre-filtered surveillance IQ (+ reference IQ) exists, so we only need to
    cross-correlate, or
  * only the raw capture exists, so we load, IQ-convert and cross-correlate.
"""

import os

import numpy as np

from passive_processor import PassiveProcessor

# ---------------------------------------------------------------------------- #
#  Radar / processing parameters. Generally only change the centre frequency and
#  decimation factor - those are capture dependent.
# ---------------------------------------------------------------------------- #
centre_freq = 570e6      # illuminator centre frequency
decimation_factor = 32   # Decimation factor used during capture
Fs = 3.84e9              # Sample rate before decimation
seg_s = 2048             # Number of segments per second (passive PRF equivalent)
seg_percent = 100        # Percentage of each segment used in xcorr

# Changeable range window (affects the correlations)
min_range = 0            # Minimum range in metres
max_range = 7980         # Maximum range in metres
max_doppler = 100

# Range-Doppler parameters
cpi_length = 1           # Coherent Processing Interval in seconds
cpi_overlap = 0.9        # Overlap of the CPI (0.9 = 90% overlap)
rd_window = 'blackmanharris'

# Range-Doppler video parameters
CMAP = 'viridis'
VIDEO_FILENAME = f'agentic_{max_range}_35db_cropped'
VMIN = -35    # Min value for the color scale (dB)
VMAX = 0      # Max value for the color scale (dB)
vid_x_lim = [0, max_range]

# Loading routes
batch_process_flag = False       # True: process all datasets in the directory
correlation_exists_flag = False  # True: load a pre-computed cc_matrix
filtered_iq_exists_flag = False   # True: load ref_iq + pre-filtered surv channel

experiment_directory = fr'G:\ARESTOR_DATA\Ulm_trial\tests\day_2'  # Directory containing the experiment data

if batch_process_flag:
    captures = os.listdir(experiment_directory)
else:
    captures = ['ulm_day_2_heli']  # Specific capture directory if not batch processing

# ---------------------------------------------------------------------------- #
#  Processing loop
# ---------------------------------------------------------------------------- #
for capture in captures:

    experiment_codename = capture  # Set the experiment codename to the current capture directory

    directory = fr'{experiment_directory}\{experiment_codename}\passive_data_test_1'

    # Build the processor once with our radar parameters
    radar = PassiveProcessor(centre_freq=centre_freq, decimation_factor=decimation_factor,
                             Fs=Fs, seg_s=seg_s, seg_percent=seg_percent, cmap=CMAP)
    radar.set_range_window(min_range=min_range, max_range=max_range)

    # --- Obtain the cross-correlation matrix by one of three routes ---
    if correlation_exists_flag:
        if not filtered_iq_exists_flag:
            radar.cc_matrix = np.load(directory + r'\cc_matrix.npy')
            radar.capture_period = 70  # Duration of the capture in seconds
            print("Unfiltered Cross-Correlation Matrix loaded from memory. No need to load IQ.")
        else:
            radar.cc_matrix = np.load(directory + r'\fixed_filtered_cc_matrix.npy')
            radar.capture_period = 70  # Duration of the capture in seconds
            print("Filtered Cross-Correlation Matrix loaded from memory. No need to load IQ.")
        # Rebuild the range axis for the loaded matrix
        no_range_bins = int(radar.cc_matrix.shape[1])
        radar.range_axis_m = np.linspace(min_range, max_range, no_range_bins)

    elif filtered_iq_exists_flag:
        # Reference IQ + a pre-filtered (Wiener) surveillance channel already exist
        radar.ref_iq = np.load(directory + r'\ref_iq.npy')
        radar.surv_iq = np.load(directory + r'\0.1_2048_1_weiner_filtered_surv_channel.npy')
        radar.capture_period = len(radar.ref_iq) / radar.d_Fs
        print(f"IQ Matrices loaded from memory. reference shape: {radar.ref_iq.shape}, "
              f"surveillance shape: {radar.surv_iq.shape}")

        radar.cross_correlate()
        print(f"Completed cross-correlation. cc_matrix shape: {radar.cc_matrix.shape}. Saving...")
        np.save(directory + fr'\{max_range}_filtered_cc_matrix.npy', radar.cc_matrix)

    else:
        # Only the raw capture exists: load, IQ-convert, save, Wiener-filter the
        # surveillance channel (automatic - see load_iq's filter_surv), then
        # cross-correlate. Pass filter_surv=False to load_iq to skip the filtering.
        radar.load_iq(directory, ref_file='passive_ref_output.npy',
                      surv_file='passive_surv_output.npy', is_raw=True, save_converted=True)
        print(f"Loaded and IQ-converted data. reference shape: {radar.ref_iq.shape}, "
              f"surveillance shape: {radar.surv_iq.shape}")

        radar.cross_correlate()
        print(f"Completed cross-correlation. cc_matrix shape: {radar.cc_matrix.shape}. Saving...")
        np.save(directory + fr'\{max_range}_cc_matrix.npy', radar.cc_matrix)
        # Free memory
        radar.ref_iq = None
        radar.surv_iq = None
        print("Deleted raw data, moving on to range-Doppler processing.")

    # --- Range-Doppler slicing ---
    radar.range_doppler(cpi_length=cpi_length, cpi_overlap=cpi_overlap, rd_window=rd_window)

    rd_slice_directory = directory + r'\rd_slices'
    os.makedirs(rd_slice_directory, exist_ok=True)
    np.save(rd_slice_directory + fr'\{max_range}_rd_slice_matrix.npy', radar.rd_slices)
    print(f"Completed range-Doppler slicing. {len(radar.rd_slices)} slices saved to {rd_slice_directory}.")

    # Convert the complex slices to normalised dB for plotting
    radar.slices_to_db()

    # --- Range-Doppler video ---
    RD_video_flag = True
    if RD_video_flag:
        video_directory = directory + r'\rd_videos'
        radar.make_rd_video(video_directory, VIDEO_FILENAME,
                            v_min=VMIN, v_max=VMAX,
                            x_lim=vid_x_lim, max_doppler=max_doppler, sharpening=False)

print("Have you checked the centre frequency and decimation factor are correct for this capture?")
