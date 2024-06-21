import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from classes import waveform
from dsp_functions import get_RT

#The data is the output of the active mode capture

data = np.load(r'/home/uceejpw/Trial_2_lab_tests/Active_tests/2x10m_cable_160bw/active_deramp_output_0_7.npy')


sig = waveform(2.44e9, 160e6, 500e-6, 1024)

capture_period = 8
Fs = 3.84e9 #Sample Rate

#In order to pass the parameters to the function, we need to define the baseband signal.

print(np.max(data), np.min(data))
print(data.shape)
baseband = data[0::2] + 1j*data[1::2] #Convert the data to complex numbers to make a baseband array.
baseband = baseband.reshape((int(sig.prf*capture_period), -1)) #Reshape the baseband array to the correct dimensions.
#This operation is typically used to organize the samples of a signal into a 2D array where each row corresponds to
# one pulse of the signal. This can be useful for further processing of the signal.

plt.figure()

ranges, times, fft_in_dB = get_RT(baseband, Fs, sig.chirp_bandwidth, sig.chirp_period, sig.prf, 50)
print(fft_in_dB.shape)
plt.pcolormesh(times, ranges, fft_in_dB.T, cmap='jet')
plt.xlabel("Time [s]")
plt.ylabel("Range [m]")
plt.colorbar()

plt.show()

