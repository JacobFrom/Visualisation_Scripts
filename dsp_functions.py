import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import get_window
import cv2
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

def rangeDopplerslice(radar_matrix, cpi, prf, cpi_overlap, zero_padding, window_type):
    """
    Processes radar data to create range-Doppler slices.

    Parameters:
    radar_matrix (numpy.ndarray): Matrix containing raw radar data (range bins x pulses).
    cpi (float): Coherent Processing Interval in seconds.
    PRF (float): Pulse Repetition Frequency in Hz.
    cpi_overlap (float): Overlap between consecutive CPIs (0 to 1).
    zero_padding (int): Factor for zero-padding the Doppler FFT.
    window_type (str): Type of window to apply (e.g., 'hann', 'hamming').

    Returns:
    tuple: (number_cpi, pulses_per_cpi, range_doppler_slices)
    - number_cpi (int): Number of CPIs processed.
    - pulses_per_cpi (int): Number of pulses per CPI.
    - range_doppler_slices (list): List of range-Doppler maps for each CPI.
    """
    pulses_per_cpi = int(np.ceil(cpi * prf))
    cpi_stride = int(np.round(pulses_per_cpi * (1-cpi_overlap)))
    number_cpi = int(np.ceil((radar_matrix.shape[1] - pulses_per_cpi) / cpi_stride))
    doppler_bins = (pulses_per_cpi * zero_padding)
    if doppler_bins % 2 == 0:
        doppler_bins += 1 #Ensure that the number of doppler bins is odd

    range_doppler_slices = [np.zeros((radar_matrix.shape[0], doppler_bins), dtype=complex) for _ in range(number_cpi)]

    # Loop through each CPI and process the data
    for i in range(number_cpi):
        # Extract the current window of pulses from the radar data
        start_index = i * cpi_stride
        end_index = start_index + pulses_per_cpi
        cpi_window = radar_matrix[:, start_index:end_index]

        # Apply the windowing function to reduce sidelobes
        window = get_window(window_type, pulses_per_cpi)
        cpi_window *= window

        # Compute the FFT along the pulse dimension (Doppler processing)
        caf = np.fft.fftshift(np.fft.fft(cpi_window, n=doppler_bins, axis=1), axes=1)

        # Store the result in the range-Doppler slices array
        range_doppler_slices[i] = caf.T

    return number_cpi, pulses_per_cpi, range_doppler_slices

def create_video(range_doppler_slices, frame_rate, doppler_range_axis, max_range, doppler_axis, max_doppler, dynamic_range,
                 video_name, video_title):
    fig, ax = plt.subplots(figsize=(10, 6))
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (640, 480))
    try:

        for idx, slice_ in enumerate(range_doppler_slices): # Each range doppler slice is processed in a loop, idx is the index, slice is the slice data
            ax.clear() #Clear axis ready for next slice
            ax.set_title(f'{video_title} - Frame {idx + 1}') #Title indicates frame number
            ax.set_xlabel('Range (m)') #Set x axis label
            ax.set_ylabel('Velocity (m/s)') #set y axis label

            power_db = 20 * np.log10(np.abs(slice_)) #Calculate power in dB
            im = ax.imshow(power_db, extent=[doppler_range_axis[0], doppler_range_axis[-1], doppler_axis[0], doppler_axis[-1]],
                           aspect='auto', vmin=power_db.max() - dynamic_range, vmax=power_db.max(), cmap='jet')
            ax.set_xlim([0, max_range]) #Set x axis limits
            ax.set_ylim([-max_doppler, max_doppler]) #Set y axis limits
            plt.colorbar(im, ax=ax, label='Power (dB)')
            plt.draw()

            #Convert the plot to an image and write to the video
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img)

    finally:
        out.release()
        plt.close(fig)

