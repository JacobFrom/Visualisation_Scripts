import numpy as np
import DSP_functions as dsp
import scipy
import matplotlib.pyplot as plt

Fs = 3.84e9  # Base Sample Rate
total_decimation = 8 * 8  # tile_decimation * block_decimation

eff_bandwidth = Fs / total_decimation
fc = 800e6  # 800 MHz Center frequency

# Using a power of 2 makes the FFT compute much faster.
num_samples = 2**19
limit = num_samples * 2  # *2 because I and Q are interleaved in the raw array

plot_PSD_flag = True

plot_raw_adc_flag = True

# --- Load and Pre-process your real data here ---
test_directory = r'/home/uceejpw/x_band_testing_september_2026_data/ADC_2345_cable_GDB_gain_18_25MHz'
ch0_data = np.load(test_directory + r'/channel_2_output.npy')
ch1_data = np.load(test_directory + r'/channel_3_output.npy')
ch2_data = np.load(test_directory + r'/channel_4_output.npy')
ch3_data = np.load(test_directory + r'/channel_5_output.npy')


ch0_data = ch0_data[0:limit:2] + 1j * ch0_data[1:limit:2]
ch1_data = ch1_data[0:limit:2] + 1j * ch1_data[1:limit:2]
ch2_data = ch2_data[0:limit:2] + 1j * ch2_data[1:limit:2]
ch3_data = ch3_data[0:limit:2] + 1j * ch3_data[1:limit:2]


if plot_PSD_flag:

    dsp.plot_PSD_channels(ch0_data, ch1_data, ch2_data, ch3_data, fc, eff_bandwidth, test_directory, normalize=False)

if plot_raw_adc_flag:
    dsp.plot_raw_adc(ch0_data, ch1_data, ch2_data, ch3_data, test_directory, samples_to_plot=2**19, y_limit=100)