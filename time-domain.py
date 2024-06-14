import numpy as np
from classes import waveform
import matplotlib as mp
import matplotlib.pyplot as plt


sig = waveform(5.8e9, 80e6, 500e-6, 1024)

capture_period = 5
Fs = 3.84e9 #Sample Rate

data = np.load(r'/home/uceejpw/May_lab_tests/Test 4 - Active with half length of test 3/test_2/active_deramp_output_0_2.npy')

baseband = data[0::2] + 1j*data[1::2]
baseband = baseband.reshape((int(sig.prf*capture_period), -1))

print(baseband.shape)
times = np.arange(baseband.shape[1]) / (Fs / (8 * 256))
print(times.shape)
amplitude = np.abs(baseband)
plt.figure()
plt.plot(times, amplitude[5002])

plt.show()