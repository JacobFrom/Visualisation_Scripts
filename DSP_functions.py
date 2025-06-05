import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib.animation as animation
import time

def plot_RTI(beat_frequencies_array, range_axis, time_axis, v_min, v_max, CMAP, x_lim, directory, save_as):

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

    rti_plot_filename = save_as + '.png'
    full_rti_save_path = os.path.join(directory, rti_plot_filename)

    print('Saving RTI plot...')
    try:
        plt.savefig(full_rti_save_path, dpi=300, bbox_inches='tight')
        print(f"RTI plot saved successfully to {full_rti_save_path}")
    except Exception as e:
        print(f"ERROR: Could not save RTI plot: {e}")

    plt.close()  # Close the plot to free up memory

def plot_MTI(MTI_filtered_data, range_axis, mti_time_axis, mti_v_min, mti_v_max, CMAP, x_lim, directory, save_as):
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

    mti_plot_filename = save_as + '.png'
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

    print("Pulses per CPI:", pulses_per_cpi)
    print("CPI stride:", cpi_stride)

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
    return rd_slices_list, velocity_axis, number_cpi, cpi_stride


def create_rd_video(FPS, rd_slices_list, prf, cpi_stride, range_axis, velocity_axis, vid_v_min, vid_v_max, CMAP, x_lim, directory, save_as):

    """
    Creates a video from the range-Doppler slices.

    Parameters:
    - FPS: Frames per second for the output video.
    - rd_slices_list: List of range-Doppler slices.
    - range_axis: 1D numpy array representing the range axis.
    - velocity_axis: 1D numpy array representing the velocity axis.
    - vid_v_min: Minimum value for color scaling in the video.
    - vid_v_max: Maximum value for color scaling in the video.
    - cmap: Colormap to use for the video.
    - xlim: Tuple specifying the x-axis limits (start, end).
    - directory: Directory to save the video.
    """
    VIDEO_FILENAME = save_as + '.mp4'  # Output video filename
    VIDEO_SAVE = os.path.join(directory, VIDEO_FILENAME)
    COLOR_VMIN = vid_v_min  # Min value for the color scale (dB) - adjust as needed
    COLOR_VMAX = vid_v_max    # Max value for the color scale (dB)


    # --- Setup the Figure ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initial plot setup using the first frame

    initial_data = rd_slices_list[0]
        #initial_data = processed_slices[0]

    im = ax.imshow(initial_data,
                    aspect='auto',
                    origin='lower',
                    extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
                    vmin=COLOR_VMIN,
                    vmax=COLOR_VMAX,
                    cmap=CMAP)

    #Add labels and title (base title)
    ax.set_xlabel('Range [m]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_xlim(x_lim)
    # Set the y-axis (Velocity) limits
    ax.set_ylim(velocity_axis[0], velocity_axis[-1])
    base_title = 'Range-Doppler Plot Evolution'
    title_text = ax.set_title(base_title) # Store title object for updates

    # Add colorbar
    cbar = fig.colorbar(im, label='Norm Power [dB]')
    plt.tight_layout() # Adjust layout to fit colorbar

    #Now need to creat the update function for the animation, will nest it inside the create_video function to avoid confusion
    #Will also include some filtering inside the update function to clean up the video
    def update(frame_num):

        #current_rd_slice = rd_slices_list[frame_num]
        #current_rd_slice = processed_slices[frame_num]
        #current_rd_slice = processed_slices[frame_num].copy()
        current_rd_slice = rd_slices_list[frame_num].copy()

        noise_threshold = vid_v_min + 4 # Set a noise threshold for the video
        # Apply noise thresholding
        current_rd_slice[current_rd_slice < noise_threshold] = COLOR_VMIN

        im.set_data(current_rd_slice)  # Update image data

        # Update title
        start_time_sec = (frame_num * cpi_stride) / prf

        current_title = f'{base_title}\nFrame: {frame_num + 1}/{len(rd_slices_list)}, Time ≈ {start_time_sec:.2f} s'
        #current_title = f'{base_title}\nFrame: {frame_num + 1}/{len(processed_slices)}, Time ≈ {start_time_sec:.2f} s'

        title_text.set_text(current_title)

        return [im, title_text]  # Return updated artists

    # --- Create and Save Animation ---
    print(f"Creating animation with {len(rd_slices_list)} frames...")
    animation_start_time = time.perf_counter()
    ani = animation.FuncAnimation(fig,
                                  update,
                                  frames=len(rd_slices_list),
                                  interval=1000 / FPS,  #ms delay between frames
                                  blit=True,
                                  repeat=False)
    animation_end_time = time.perf_counter()
    animation_duration = animation_end_time - animation_start_time
    print(f'Animation creation took {animation_duration:.2f} seconds.')
    print(f"Finished creating animation with {len(rd_slices_list)} frames. Now saving to {VIDEO_SAVE}...")
    # Save the animation
    saving_start_time = time.perf_counter()
    ani.save(str(VIDEO_SAVE), writer='ffmpeg', fps=FPS, dpi=150)
    saving_end_time = time.perf_counter()
    saving_duration = saving_end_time - saving_start_time
    print(f"Animation saved to {VIDEO_SAVE} and saving took {saving_duration} seconds.")
    plt.close(fig)