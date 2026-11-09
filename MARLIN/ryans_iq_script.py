import numpy as np
import matplotlib.pyplot as plt
import os

int_filt = np.ones(10)

fs = 3.84e9 / 32  # Sampling frequency in Hz

plt.figure()
test_number = 1
channel_of_interest = 5

iq_directory = fr"/home/uceejpw//MARLIN_OCT_2026/day_3/IQ_SS_AX_INTL_T1_1_GDB_0dB/Raw_IQ_Capture_1"

raw_data = np.load(os.path.join(iq_directory, f'channel_{channel_of_interest}_iq.npy'))

# raw_data = raw_data[100000:200000]
cmplx_data = raw_data[0::2] + 1.0j * raw_data[1::2]
time = np.arange(len(cmplx_data)) / fs
mag = np.abs(cmplx_data)

integrated_mag = np.convolve(int_filt, mag, mode="same")
power = mag ** 2
power_db = 10 * np.log10(power)
# plt.plot(time, cmplx_data.real, label="I")
# plt.plot(time, cmplx_data.imag, label="Q")
plt.plot(time, integrated_mag, label="Power")
plt.legend()
plt.title(f'Channel {channel_of_interest} IQ')
plt.xlabel("Time (s)")
plt.ylabel("Magnitude")
# plt.ylabel("Power (dB)")
plt.show()
# plt.figure(figsize=(8.5, 8.5))
# plt.specgram(cmplx_data, Fs=fs)
# plt.xlabel("Time (s)")
# plt.ylabel("Freq (Hz)")
# plt.show()