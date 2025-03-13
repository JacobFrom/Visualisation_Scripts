import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Qt5Agg')

#The data is the output of the active mode capture

data = np.load(r'E:\loki_testing_day_1\s_band_test_1\active_deramp_output.npy')

#Below are the parameters of the waveform

prf = 1024 #Pulse Repitition Frequency
capture_period = 5
max_range = 50
Fs = 3.84e9 #Sample Rate
chirp_bandwidth = 80e6
chirp_period = 500e-6
centre_freq = 2440e6
reshaped_flag = True
if reshaped_flag:
    data = data[:, 0::2] + 1j*data[:, 1::2]
else:
    data = data[0::2] + 1j*data[1::2]

seconds_of_interest = 5
N = prf * seconds_of_interest#Number of samples for fft

fft_data = data[0:N, :] #Take the first N pulses for the fft
window = np.hamming(fft_data.shape[0]) #Create a hamming window function
windowed_data = fft_data * window[:, np.newaxis] #Apply hamming window function to the data

PSD = np.abs(np.fft.fft(windowed_data))**2 / (N * (Fs)) #Calculate the Power Spectral Density

PSD_dB = 10.0*np.log10(PSD) #Convert to dB

PSD_shifted = np.fft.fftshift(PSD_dB) #Shift the PSD to center around 0

f = np.arange(0, (Fs / (8*256)), (Fs / (8*256))/N) #Start, Stop, Step. Centred around 0 Hz
f += centre_freq #Shift the frequency axis to the centre frequency

plt.plot(f, PSD_shifted)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)

plt.show()
