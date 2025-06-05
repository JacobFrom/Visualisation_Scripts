import numpy as np
import matplotlib.pyplot as plt
from legacy_working_scripts.classes import waveform
from old_scripts.dsp_functions import get_RT

#The data is the output of the active mode capture

data = np.load(r'E:\Trial_2_Day_2_tests\Run8\trial_2_test_8\active_data\active_deramp_output_0_8.npy')


sig = waveform(2.440e9, 80e6, 500e-6, 1024)

capture_period = 5
Fs = 3.84e9 #Sample Rate

#In order to pass the parameters to the function, we need to define the baseband signal.

print(np.max(data), np.min(data))
print(data.shape)

#if already reshaped in script use the following 2 lines:
iq_data = data[:, 0::2] + 1j*data[:, 1::2] #Convert the data to complex numbers to make a baseband array.
#deramped_data = iq_data.reshape((int(signal.prf*capture_period)), -1).T
deramped_data = iq_data.T #Reshape the baseband array to the correct dimensions.
##############################################################
#else use the next 2 lines:

#baseband = data[0::2] + 1j*data[1::2] #Convert the data to complex numbers to make a baseband array.
#baseband = baseband.reshape((int(sig.prf*capture_period), -1)) #Reshape the baseband array to the correct dimensions.
#This operation is typically used to organize the samples of a signal into a 2D array where each row corresponds to
# one pulse of the signal. This can be useful for further processing of the signal.

plt.figure()

ranges, times, fft_in_dB = get_RT(deramped_data, Fs, sig.chirp_bandwidth, sig.chirp_period, sig.prf, 50)
print(fft_in_dB.shape)
plt.pcolormesh(times, ranges, fft_in_dB.T, cmap='jet')
plt.xlabel("Time [s]")
plt.ylabel("Range [m]")
plt.colorbar()

plt.show()

