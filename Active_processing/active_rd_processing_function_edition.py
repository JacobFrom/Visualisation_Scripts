"""
Active radar processing driver / example.

This is a thin, readable example of how to use :class:`ActiveProcessor`. All of
the actual signal processing lives inside the class (see
``Active_processing/active_processor.py``); here we just set the parameters of
the capture and call the methods in order.

To process your own data, edit the parameters and ``experiment_directory`` below.
"""

import os

from active_processor import ActiveProcessor

# ---------------------------------------------------------------------------- #
#  Radar capture parameters (edit these for your capture)
# ---------------------------------------------------------------------------- #
s_band = 2.44e9  # active centre frequency
c_band = 5.8e9

centre_freq = c_band   # active centre frequency
bandwidth = 80e6       # chirp bandwidth
chirp_period = 500e-6  # chirp length
prf = 1024             # Pulse Repetition Frequency

Fs = 3.84e9
decimation = 8 * 256   # Total decimation factor -> decimated_sample_rate = Fs / decimation

CMAP = 'viridis'       # Colormap for the plots

# Which processing steps to run
RTI_plot_flag = True
MTI_plot_flag = True
rd_processing_flag = True

# Decide here if you want to batch process or process individual datasets
batch_process_flag = False  # True: process all datasets in the directory, False: a single dataset

experiment_directory = fr'G:\ARESTOR_DATA\trial_3_captures\C_band\32x'  # Directory containing the experiment data

if batch_process_flag:
    captures = os.listdir(experiment_directory)
else:
    captures = [fr'\run_7']  # Specific capture directory if not batch processing

# ---------------------------------------------------------------------------- #
#  Processing loop
# ---------------------------------------------------------------------------- #
for capture in captures:

    experiment_codename = capture  # Set the experiment codename to the current capture directory

    directory = fr'{experiment_directory}\{experiment_codename}\active_test_1'

    # Build the processor once with our radar parameters
    radar = ActiveProcessor(centre_freq=centre_freq, bandwidth=bandwidth,
                            chirp_period=chirp_period, prf=prf, Fs=Fs,
                            decimation=decimation, cmap=CMAP)

    # Load the capture and compute the (positive) beat frequencies + range/time axes
    radar.load_capture(directory, filename='1.npy')
    radar.compute_beat_frequencies()

    # Range-Time Intensity plot
    if RTI_plot_flag:
        radar.plot_rti(directory, 'agentic_test_rti_plot', v_min=-50, v_max=0, x_lim=[0, 200])
        print("RTI plot created and saved successfully.")

    # Two-pulse MTI canceller plot
    if MTI_plot_flag:
        radar.plot_mti(directory, 'agentic_test_mti_plot', v_min=-50, v_max=0, x_lim=[0, 200])
        print("MTI plot created and saved successfully.")

    # Range-Doppler processing and video
    if rd_processing_flag:
        print("Starting Range-Doppler processing...")

        radar.range_doppler(cpi_length=0.5, cpi_overlap=0.9, rd_window='blackmanharris')

        # RD video. Video creation has some built-in filtering (sharpening) you may wish to adjust.
        # IMPORTANT: Ensure ffmpeg is installed and available on PATH or this won't work.
        # Install instructions: https://ffmpeg.org/download.html
        radar.make_rd_video(directory, 'agentic_range_Doppler_video_200m',
                            v_min=-50, v_max=0, x_lim=[0, 200], y_lim=None, sharpening=False)

        print(f"Finished processing dataset: {experiment_codename}")
