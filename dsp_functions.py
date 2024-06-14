import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')
def get_psd(iq_data, centre_frequency, sample_rate, number_of_samples):
    """
    This function calculates the Power Spectral Density of a given signal
    :param iq_data: The signal data
    :param centre_frequency: The centre frequency of the signal
    :param sample_rate: The sample rate of the signal
    :param number_of_samples: The number of samples that the signal was captured for
    :return: The Power Spectral Density plot of the signal
    """
    data = iq_data # The data is the output of the active mode capture

    N = number_of_samples  # Number of samples for fft

    fft_data = data[0:N]  # Take the first N samples for the fft

    fft_data = fft_data * np.hamming(len(fft_data))  # Apply hamming window function to the data

    PSD = np.abs(np.fft.fft(fft_data)) ** 2 / (N * sample_rate)  # Calculate the Power Spectral Density

    PSD_dB = 10.0 * np.log10(PSD)  # Convert to dB

    PSD_shifted = np.fft.fftshift(PSD_dB)  # Shift the PSD to center around 0

    f = np.arange(sample_rate / -2.0, sample_rate / 2.0, sample_rate / N)  # Start, Stop, Step. Centred around 0 Hz
    f += centre_frequency  # Shift the frequency axis to the centre frequency
    plt.plot(f, PSD_shifted)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)

    plt.show()
    return PSD_shifted


def get_RT(baseband, sample_rate, chirp_bandwidth, chirp_period, prf, max_range):
    """
    This function calculates the Range-Time plot of a given signal.
    :param baseband: This is the baseband signal data, usually the output of the active mode capture reshaped appropriately.
    :param sample_rate: The ADC sample rate for the experiment.
    :param chirp_bandwidth: The bandwidth of the chirp signal.
    :param chirp_period: The period of the chirp signal.
    :param prf: The pulse repetition frequency of the signal.
    :param max_range: The maximum range of interest for the plot.
    :return: The ranges vs time, and fft_in_dB of the signal.
    """
    baseband -= np.mean(baseband, axis=-1, keepdims=True)
    baseband *= np.hanning(baseband.shape[1])
    print(baseband.shape)
    fft = np.fft.fft(baseband, axis=-1)
    freq = np.fft.fftfreq(fft.shape[1], 1. / (sample_rate / (8 * 256))) #Factoring in the decimation

    ranges = freq * (3e8 * chirp_period) / (2.0 * chirp_bandwidth)

    max_idx = np.argmin(np.fabs(ranges - max_range))

    ranges = ranges[0:max_idx]

    fft_in_dB = 20 * np.log10(np.abs(fft[:, 0:max_idx]))

    times = np.arange(baseband.shape[0]) / prf
    print(times.shape)
    return ranges, times, fft_in_dB