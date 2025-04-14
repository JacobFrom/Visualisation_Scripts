import numpy as np
import scipy.signal as signal
import time

from active_rd_processing_final import velocity_axis


def get_passive_rd_slices(correlation_matrix, slow_time_step, wavelength, cpi_sec, cpi_overlap):

    #Calculate the number of samples in the slow time domain
    slow_time_samples = correlation_matrix.shape[0]
    #Calculate the number of samples per CPI
    samples_per_cpi = int(np.ceil(cpi_sec / slow_time_step))
    #Calculate the CPI stride
    cpi_stride = int(np.round(samples_per_cpi * (1 - cpi_overlap)))
    #Calculate the number of CPIs that can be processed
    num_rd_slices = int(np.floor((slow_time_samples - samples_per_cpi) / cpi_stride)) + 1

    #Generate axis for plots that will be made using these slices

    doppler_axis_hz = np.fft.fftshift(np.fft.fftfreq(samples_per_cpi, d=slow_time_step))
    velocity_axis = doppler_axis_hz * wavelength / 2

    #Create window for the slow time domain
    window = signal.get_window('blackmanharris', samples_per_cpi)

    rd_slices_list = []

    for i in range(num_rd_slices):
        start_idx = i * cpi_stride
        end_idx = start_idx + samples_per_cpi
        windowed_cpi = correlation_matrix[start_idx:end_idx, :] * window[:, np.newaxis]
        rd_slice = np.fft.fft(windowed_cpi, axis=0)
        rd_slice = np.fft.fftshift(rd_slice, axes=0)
        slice_plot = (20 * np.log10(np.abs(rd_slice) / np.max(np.abs(rd_slice))))

        rd_slices_list.append(slice_plot)

    return rd_slices_list, velocity_axis

# def get_active_rd_slices:
