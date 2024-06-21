import numpy as np
from classes import waveform
import matplotlib as mp
import matplotlib.pyplot as plt


sig = waveform(530e6, 80e6, 1, 1024)

capture_period = 3
Fs = 3.84e9 #Sample Rate

data = np.load(r'/home/uceejpw/Trial_2_lab_tests/3s_blank_cap/passive_ref_output_0_1.npy')

baseband = data[0::2] + 1j*data[1::2]
baseband = baseband.reshape((int(sig.prf*capture_period), -1))

print(baseband.shape)
times = np.arange(baseband.shape[1]) / (Fs / (8 * 256))
print(times.shape)
amplitude = np.abs(baseband)
plt.figure()
plt.plot(times, amplitude[1000])

plt.show()