import numpy as np
from classes import waveform
import matplotlib as mp
import matplotlib.pyplot as plt


sig = waveform(5800e6, 80e6, 500e-6, 1024)

capture_period = 20
Fs = 3.84e9 #Sample Rate

data = np.load(r'/home/uceejpw/C_band_testing/Data/C_band_testing_ota/jacob_waving_moving.npy')

reshaped_flag = True
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
print(amplitude.shape)

plt.figure()

#plt.plot(samples, amplitude[5000])
#plt.plot(samples, amplitude.T)
plt.plot(samples, baseband.T)
#plt.plot(samples, baseband[5000])
#plt.plot(amplitude.T)

plt.show()