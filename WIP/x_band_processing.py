import numpy as np
import DSP_functions as dsp
import scipy
import matplotlib.pyplot as plt

ch0_data = np.load(r'D:\ARESTOR\x_band_july_testing\21_07_tests\x_band_test_6\channel_2_output.npy')
ch1_data = np.load(r'D:\ARESTOR\x_band_july_testing\21_07_tests\x_band_test_6\channel_3_output.npy')
ch2_data = np.load(r'D:\ARESTOR\x_band_july_testing\21_07_tests\x_band_test_6\channel_4_output.npy')
ch3_data = np.load(r'D:\ARESTOR\x_band_july_testing\21_07_tests\x_band_test_6\channel_5_output.npy')

#Below are the parameters of the waveform

prf = 1024 #Pulse Repitition Frequency

tile_decimation = 8
block_decimation = 4
total_decimation = tile_decimation * block_decimation

capture_period = 3

Fs = 3.84e9 #Sample Rate
total_samples = (Fs / total_decimation) * capture_period

centre_freq = 800e6

ch0_data = ch0_data[0::2] + 1j*ch0_data[1::2]
ch1_data = ch1_data[0::2] + 1j*ch1_data[1::2]
ch2_data = ch2_data[0::2] + 1j*ch2_data[1::2]
ch3_data = ch3_data[0::2] + 1j*ch3_data[1::2]

#N = 0 #Number of samples for fft
#end = N + 100000*1024

phase_data = dsp.analyze_rf_phases(ch0_data, ch1_data, ch2_data, ch3_data, fast_compare=True, plot=False)

plot = dsp.plot_phase_differences(ch0_data, ch1_data, ch2_data, ch3_data)
