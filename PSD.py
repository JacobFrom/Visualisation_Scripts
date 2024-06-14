import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Qt5Agg')

#The data is the output of the active mode capture

data = np.load(r'/home/uceejpw/May_lab_tests/Test 5 - Active with 3 connected long cables/test_2/active_deramp_output_0_2.npy')

#Below are the parameters of the waveform

prf = 1024 #Pulse Repitition Frequency
capture_period = 5
max_range = 100
Fs = 3.84e9 #Sample Rate
chirp_bandwidth = 80e6
chirp_period = 500e-6
centre_freq = 5.8e9

N = 1024 #Number of samples for fft

fft_data = data[0:N] #Take the first N samples for the fft

fft_data = fft_data * np.hamming(len(fft_data)) #Apply hamming window function to the data

PSD = np.abs(np.fft.fft(fft_data))**2 / (N * (Fs)) #Calculate the Power Spectral Density

PSD_dB = 10.0*np.log10(PSD) #Convert to dB

PSD_shifted = np.fft.fftshift(PSD_dB) #Shift the PSD to center around 0

f = np.arange(0, (Fs / (8*256))/2.0, (Fs / (8*256))/N) #Start, Stop, Step. Centred around 0 Hz
#f += centre_freq #Shift the frequency axis to the centre frequency

plt.plot(f, PSD_shifted)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)

plt.show()
