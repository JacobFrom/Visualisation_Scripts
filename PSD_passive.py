import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Qt5Agg')

#The data is the output of the active mode capture

data = np.load(r'/media/uceejpw/External_SSD_JF/Loki_v2_testing_day_2/passive_data_test_3/passive_ref_output.npy')

#Below are the parameters of the waveform

prf = 1024 #Pulse Repitition Frequency

tile_decimation = 8
block_decimation = 4
total_decimation = tile_decimation * block_decimation

capture_period = 5

Fs = 3.84e9 #Sample Rate
total_samples = (Fs / total_decimation) * capture_period

centre_freq = 530e6

data = data[0::2] + 1j*data[1::2]

N = 0 #Number of samples for fft
end = N + 50000*1024

fft_data = data[N:end] #Take the first N samples for the fft

fft_data = fft_data * np.hamming(len(fft_data)) #Apply hamming window function to the data

PSD = np.abs(np.fft.fft(fft_data))**2 / (len(fft_data) * (Fs/total_decimation)) #Calculate the Power Spectral Density


PSD_dB = 10.0*np.log10(PSD) #Convert to dB

PSD_shifted = np.fft.fftshift(PSD_dB) #Shift the PSD to center around 0


f = np.arange(-((Fs/ total_decimation)/2), ((Fs/ total_decimation)/2), (Fs / (total_decimation))/len(fft_data)) #Start, Stop, Step. Centred around 0 Hz
f += centre_freq #Shift the frequency axis to the centre frequency

plt.plot(f, PSD_shifted)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)

plt.show()
