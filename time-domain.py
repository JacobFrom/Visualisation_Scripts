import numpy as np
from classes import waveform
import matplotlib as mp
import matplotlib.pyplot as plt


sig = waveform(5800e6, 80e6, 500e-6, 1024)

capture_period = 10
Fs = 3.84e9 #Sample Rate

#data = np.load(r'D:\Data\C_band_testing\test_1.npy')
data = np.load(r'C:\Users\jster\Downloads\large_cable_loop_back.npy')
reshaped_flag = False
if reshaped_flag:
    baseband = data[:, 0::2] + 1j * data[:, 1::2]
else:
    baseband = data[0::2] + 1j * data[1::2]
    baseband = baseband.reshape((int(sig.prf * capture_period), -1))

print(baseband.shape)
#samples = np.arange(baseband.shape[1]) / (Fs / (8 * 256))
samples = np.arange(baseband.shape[1])
print(samples.shape)
amplitude = np.abs(baseband)
plt.figure()
#plt.plot(times, amplitude[8000, :])
plt.plot(samples, amplitude.T)

plt.show()