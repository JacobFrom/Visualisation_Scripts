import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib.animation as animation
import time

def plot_RTI(beat_frequencies_array, range_axis, time_axis, v_min, v_max, CMAP, x_lim, directory):

    """
    Plots a Range-Time Intensity (RTI) plot of the beat frequencies array.

    Parameters:
    - beat_frequencies_array: 2D numpy array of beat frequencies.
    - range_axis: 1D numpy array representing the range axis.
    - time_axis: 1D numpy array representing the time axis.
    - vmin: Minimum value for color scaling.
    - vmax: Maximum value for color scaling.
    - cmap: Colormap to use for the plot.
    - xlim: Tuple specifying the x-axis limits (start, end).
    - directory: Directory to save the plot.
    """

    RTI_plot = (20 * np.log10(np.abs(beat_frequencies_array) / np.max(np.abs(beat_frequencies_array))))
    plt.imshow(RTI_plot, aspect='auto', origin='lower', extent=[range_axis[0], range_axis[-1], time_axis[0], time_axis[-1]], vmin=v_min, vmax=v_max, cmap=CMAP)
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Range (m)')
    plt.ylabel('Time (s)')
    plt.title('Range-Time Intensity (RTI) Plot')
    plt.xlim(x_lim)

    rti_plot_filename = 'RTI_plot.png'
    full_rti_save_path = os.path.join(directory, rti_plot_filename)

    print('Saving RTI plot...')
    try:
        plt.savefig(full_rti_save_path, dpi=300, bbox_inches='tight')
        print(f"RTI plot saved successfully to {full_rti_save_path}")
    except Exception as e:
        print(f"ERROR: Could not save RTI plot: {e}")

    plt.close()  # Close the plot to free up memory

def plot_MTI(MTI_filtered_data, range_axis, mti_time_axis, mti_v_min, mti_v_max, CMAP, x_lim, directory):
    """
    Plots a Moving Target Indicator (MTI) plot of the filtered data.

    Parameters:
    - MTI_filtered_data: 2D numpy array of MTI filtered data.
    - range_axis: 1D numpy array representing the range axis.
    - mti_time_axis: 1D numpy array representing the time axis.
    - mti_v_min: Minimum value for color scaling.
    - mti_v_max: Maximum value for color scaling.
    - cmap: Colormap to use for the plot.
    - xlim: Tuple specifying the x-axis limits (start, end).
    - directory: Directory to save the plot.
    """

    MTI_plot = (20 * np.log10(np.abs(MTI_filtered_data) / np.max(np.abs(MTI_filtered_data))))
    plt.imshow(MTI_plot, aspect='auto', origin='lower', extent=[range_axis[0], range_axis[-1], mti_time_axis[0], mti_time_axis[-1]], vmin=mti_v_min, vmax=mti_v_max, cmap=CMAP)
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Range (m)')
    plt.ylabel('Time (s)')
    plt.title('Moving Target Indicator (MTI) Plot')
    plt.xlim(x_lim)

    mti_plot_filename = 'MTI_plot.png'
    full_mti_save_path = os.path.join(directory, mti_plot_filename)

    print('Saving MTI plot...')
    try:
        plt.savefig(full_mti_save_path, dpi=300, bbox_inches='tight')
        print(f"MTI plot saved successfully to {full_mti_save_path}")
    except Exception as e:
        print(f"ERROR: Could not save MTI plot: {e}")

    plt.close()  # Close the plot to free up memory

def range_Doppler_slicing(beat_frequencies, cpi_length, cpi_overlap, prf, rd_window, wavelength):
    """
    Performs range-Doppler slicing on the beat frequencies data.

    Parameters:
    - beat_frequencies: 2D numpy array of beat frequencies.
    - cpi_length: Length of the coherent processing interval (CPI).
    - cpi_overlap: Overlap percentage for the CPI.
    - prf: Pulse repetition frequency.
    - rd_window: Window function for range-Doppler processing.
    - wavelength: Wavelength of the radar signal.

    Returns:
    - rd_slices_list: List of range-Doppler slices.
    - velocity_axis: Velocity axis corresponding to the Doppler frequency.
    """
    #Configure the parameters for range-Doppler processing

    pulse_interval = 1 / prf
    num_pulses = beat_frequencies.shape[0]
    pulses_per_cpi = int(cpi_length * prf)
    cpi_stride = int(np.round(pulses_per_cpi * (1-cpi_overlap)))

    window = signal.get_window(rd_window, pulses_per_cpi)

    number_cpi = int(np.ceil((num_pulses - pulses_per_cpi) / cpi_stride))

    #configure the axis for the range-Doppler slices
    slice_doppler_axis_unshifted = np.fft.fftfreq(pulses_per_cpi, d=pulse_interval) #This calculates doppler axis in Hz
    slice_doppler_axis = np.fft.fftshift(slice_doppler_axis_unshifted) #Shift to center zero frequency
    velocity_axis = (wavelength*slice_doppler_axis) / 2 # Convert to velocity in m/s

    #Now onto the slicing
    rd_slices_list = [] #Pre allocate the list for range-Doppler slices
    for i in range(number_cpi):
        start_index = i * cpi_stride
        end_index = start_index + pulses_per_cpi

        windowed_cpi = beat_frequencies[start_index:end_index] * window[:, np.newaxis]  # Apply the window to the current CPI
        rd_slice = np.fft.fft(windowed_cpi, axis=0)
        rd_slice = np.fft.fftshift(rd_slice, axes=0)
        slice_plot = (20 * np.log10(np.abs(rd_slice) / np.max(np.abs(rd_slice))))

        rd_slices_list.append(slice_plot)  # Append the slice to the list
        print(f'Finished slice: {len(rd_slices_list)} of {number_cpi}')
    print(f"Finished processing. Number of slices processed: {len(rd_slices_list)}")
    return rd_slices_list, velocity_axis