import numpy as np
import matplotlib.pyplot as plt
from original_messy_code.legacy_working_scripts.classes import waveform
# Define waveform parameters
signal = waveform(5.8e9, 80e6, 500e-6, 1024)
# Define sample rate, capture period and number of pulses
Fs = 3.84e9
capture_period = 10

# Load in deramped data from ARESTOR
data = np.load(r'D:\Data\C_band_testing\test_1.npy')

iq_data = data[0::2] + 1j*data[1::2] #Convert the data to complex numbers to make a baseband array.

#deramped_data = iq_data.reshape((int(signal.prf*capture_period)), -1).T #Reshape the baseband array to the correct dimensions.

deramped_data = iq_data.T
#This operation is typically used to organize the samples of a signal into a 2D array where each row corresponds to
# one pulse of the signal. This can be useful for further processing of the signal.
# Apply windowing function
window = np.hamming(deramped_data.shape[0])
#print("Window Shape:", np.shape(window))
windowed_data = deramped_data * window[:, np.newaxis]
# Perform FFT on windowed data
fft_data = np.fft.fft(windowed_data, axis=0)
# Keep only positive beat frequencies
num_rows = fft_data.shape[0] #Number of frequency bins
positive_beat_freqs = fft_data[:num_rows // 2, :]
#Print statements below act as sanity checks



# Assuming positive_beat_freqs is already defined and populated

# Calculate the power of each pulse
pulse_power = np.sum(np.abs(positive_beat_freqs) ** 2, axis=0)

# Identify pulses with power above a certain threshold
threshold = np.mean(pulse_power) + 5 * np.std(pulse_power)  #Threshold based on plots
interfering_pulses = pulse_power > threshold

# Step 3: Create a mask to exclude the identified pulses
mask = ~interfering_pulses

# Step 4: Apply the mask to create a new array without interfering pulses
cleaned_positive_beat_freqs = positive_beat_freqs[:, mask]

# Print the shape of the new array to verify
print("Shape of cleaned_positive_beat_freqs:", cleaned_positive_beat_freqs.shape)

# Optional: Plot the power of pulses to visualize the threshold and interfering pulses
plt.figure()
plt.plot(pulse_power, label='Pulse Power')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel('Pulse Index')
plt.ylabel('Power')
plt.title('Pulse Power with Interfering Pulses Highlighted')
plt.legend()
plt.grid(True)
plt.show()
