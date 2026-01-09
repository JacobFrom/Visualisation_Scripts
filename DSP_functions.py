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
    - number_cpi: Number of CPIs processed.
    - cpi_stride: Stride between CPIs.
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


def create_rd_video(FPS, rd_slices_list, prf, cpi_stride, range_axis, velocity_axis, vid_v_min, vid_v_max, CMAP, x_lim, directory, save_as, y_lim=None, sharpening=True):

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
    - save_as: Base name for the output video file.
    - y_lim: Tuple specifying the y-axis limits (start, end). If None, defaults to the range of velocity_axis.
    - sharpening: Boolean flag to apply sharpening to the video frames. In this case sharpening is my own invention where it will suppress noise below a certain threshold (currently v_min + 2 dB) in the video frames. Change to false to remove this effect.
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
    ax.set_ylim(y_lim)
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
        if sharpening:
            current_rd_slice = rd_slices_list[frame_num].copy()

            noise_threshold = vid_v_min + 2 # Set a noise threshold for the video
        # Apply noise thresholding
            current_rd_slice[current_rd_slice < noise_threshold] = COLOR_VMIN

            im.set_data(current_rd_slice)  # Update image data
        else:
            current_rd_slice = rd_slices_list[frame_num]
            im.set_data(current_rd_slice)
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
    ani.save(str(VIDEO_SAVE), writer='ffmpeg', fps=FPS, dpi=150) #IMPORTANT: Ensure ffmpeg is installed and available in user path or this won't work
    saving_end_time = time.perf_counter()
    saving_duration = saving_end_time - saving_start_time
    print(f"Animation saved to {VIDEO_SAVE} and saving took {saving_duration} seconds.")
    plt.close(fig)

def py_xcorr(surv_iq_matrix, ref_iq_matrix, max_lag, min_lag=0):
    """
    Performs cross-correlation between reference and surveillance IQ matrices.

    Parameters:
    - ref_iq_matrix: 2D numpy array of reference IQ data.
    - surv_iq_matrix: 2D numpy array of surveillance IQ data.
    - max_lag: Maximum lag in samples.
    - min_lag: Minimum lag in samples.

    Returns:
    - cc_matrix: Cross-correlation matrix.
    - lags: 1D numpy array of lag idx corresponding to the cross-correlation values.
    """
    full_corr = signal.correlate(surv_iq_matrix, ref_iq_matrix, mode='full', method='auto')
    full_lags = signal.correlation_lags(len(surv_iq_matrix), len(ref_iq_matrix), mode='full')

    # Extract the relevant lags and cross-correlation values
    zero_lag_idx = len(full_corr) // 2
    #With center point known for 0th lag, we can calculate the indices for the desired lags
    min_lag_idx = zero_lag_idx + min_lag
    max_lag_idx = zero_lag_idx + max_lag + 1  # +1 to include the max lag in the output

    #Now to extract the relevant part of the cross-correlation matrix
    cc_matrix = full_corr[min_lag_idx:max_lag_idx]
    lags = full_lags[min_lag_idx:max_lag_idx]

    return cc_matrix, lags




def passive_processing(surv_iq, ref_iq, seg_s, seg_percent, Fs, max_lag_samples, min_lag_samples=0):
    """
    Processes the reference and surveillance IQ data to compute the beat frequencies.

    Parameters:
    - ref_iq: 1D numpy array of reference IQ data.
    - surv_iq: 1D numpy array of surveillance IQ data.
    - seg_s: Number of segments per second.
    - seg_percent: Percentage of segments used in cross-correlation.
    - Fs: Sample rate after decimation.
    - max_lag_samples: Maximum lag in samples.

    Returns:
    - dec_ref_channel: Decimated reference channel data.
    - self_ambg_matrix: Self-ambiguity matrix of the reference data.
    - cc_matrix: Cross-correlation matrix of the reference and surveillance data.
    """
    # Set up the segmented matrixes for processing
    seg_size = int(Fs / seg_s) # Segment size in samples
    num_segments = int(np.floor(len(ref_iq)/seg_size))  # Number of segments in the reference data

    ref_iq = ref_iq[0:int((num_segments*seg_size))]
    surv_iq = surv_iq[0:int((num_segments*seg_size))]  # Ensure both signals have the same length

    seg_ref_channel = np.reshape(ref_iq, (num_segments, seg_size))
    seg_surv_channel = np.reshape(surv_iq, (num_segments, seg_size))

    # Calculate the decimated reference channel
    cc_size = np.round(seg_size * (seg_percent/100))
    dec_ref_channel = seg_ref_channel[:, 0:int(cc_size)]  # Decimate the reference channel
    dec_surv_channel = seg_surv_channel[:, 0:int(cc_size)]  # Decimate the surveillance channel
    print(f"Decimated reference channel shape: {dec_ref_channel.shape}, Decimated surveillance channel shape: {dec_surv_channel.shape}")

    #Here we begin the cross-correlation processing
    #Start with specifying the CC matrix idx of interest for extraction later
    num_range_bins_out = max_lag_samples - min_lag_samples + 1  # Number of range bins in the output

    cc_matrix = np.zeros((num_segments, num_range_bins_out), dtype=complex) # Initialize thecross-correlation matrix
    #self_ambg_matrix = np.zeros((num_segments, num_range_bins_out),dtype=complex)

    start_time = time.perf_counter()
    # Calculate the self-ambiguity matrix for the reference data
    for i in range(num_segments):

        temp_cc_matrix, temp_lags = py_xcorr(dec_surv_channel[i, :], dec_ref_channel[i, :],
                                             max_lag=max_lag_samples,
                                             min_lag=min_lag_samples)
        #temp_sa_matrix, _ = py_xcorr(dec_ref_channel[i, :], dec_ref_channel[i, :],
                                                 #max_lag=max_lag_samples,
                                                #min_lag=min_lag_samples)


        cc_matrix[i, :] = temp_cc_matrix
        #self_ambg_matrix[i, :] = temp_sa_matrix
        print(f"Finished processing {i} of {num_segments} segments of data.")
    #cc_lags = temp_lags # Store the lags from the last segment processed
    end_time = time.perf_counter()
    processing_duration = end_time - start_time
    print(f"Finished processing {num_segments} segments of data in {processing_duration:.2f} seconds.")
    #return self_ambg_matrix, cc_matrix #cc_lags
    return cc_matrix

def simple_zero_doppler_cleaning(caf_slice, self_ambg_zero_doppler, max_iterations, loop_gain_p, threshold_abs):



    doppler_bins, range_bins = np.shape(caf_slice)
    zero_doppler_idx = doppler_bins // 2
    print(f"Number of doppler bins = {doppler_bins}, zero doppler idx = {zero_doppler_idx}")
    self_ambg_zero_doppler = self_ambg_zero_doppler.flatten()
    print("Shape of self_ambg_zero_doppler = ", self_ambg_zero_doppler.shape)
    abs_self_ambg_zero_doppler = np.abs(self_ambg_zero_doppler)
    peak_idx_self = np.argmax(abs_self_ambg_zero_doppler)
    normalised_self_ambg_zero_doppler = self_ambg_zero_doppler / self_ambg_zero_doppler[peak_idx_self]
    center_idx_self_ambg = peak_idx_self

    residual_caf = caf_slice



    for iter in range(max_iterations):
        zero_doppler_cut_residual = residual_caf[zero_doppler_idx, :]
        #peak_abs_val, delay_idx = np.max(np.abs(zero_doppler_cut_residual))
        peak_abs_val = np.max(np.abs(zero_doppler_cut_residual))
        delay_idx = np.argmax(np.abs(zero_doppler_cut_residual))
        if peak_abs_val < threshold_abs:
            break
        scale_factor = zero_doppler_cut_residual[delay_idx]

        shift_amount = delay_idx - center_idx_self_ambg
        shifted_component_zd = np.zeros(range_bins, dtype=normalised_self_ambg_zero_doppler.dtype)
        #print(f"Dtype of caf_slice: {caf_slice.dtype}")
        #print(f"Dtype of self_ambg_zero_doppler_input: {self_ambg_zero_doppler.dtype}")
        if shift_amount == 0:
            shifted_component_zd = normalised_self_ambg_zero_doppler
        elif shift_amount > 0:
            len_to_copy = min(len(normalised_self_ambg_zero_doppler), range_bins - shift_amount)
            if len_to_copy > 0:
                shifted_component_zd[shift_amount : shift_amount + len_to_copy] = normalised_self_ambg_zero_doppler[:len_to_copy]
        else:
            len_to_copy = min(len(normalised_self_ambg_zero_doppler), range_bins + shift_amount)
            if len_to_copy > 0:
                shifted_component_zd[: len_to_copy] = normalised_self_ambg_zero_doppler[-shift_amount : -shift_amount + len_to_copy]
        component_to_subtract = loop_gain_p * scale_factor * shifted_component_zd
        residual_caf[zero_doppler_idx, :] = residual_caf[zero_doppler_idx, :] - component_to_subtract

    cleaned_caf_slice = residual_caf

    return cleaned_caf_slice

def passive_range_Doppler_slicing(beat_frequencies, cpi_length, cpi_overlap, prf, rd_window, wavelength):
    """
    Performs range-Doppler slicing on the beat frequencies data.

    Parameters:
    - beat_frequencies: 2D numpy array of beat frequencies.
    - cpi_length: Length of the coherent processing interval (CPI).
    - cpi_overlap: Overlap percentage for the CPI.
    - prf: Pulse repetition frequency.
    - rd_window: Window function for range-Doppler processing.
    - wavelength: Wavelength of the radar signal.
    - radar_mode: Radar mode, accepts 'passive', 'active').

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


        rd_slices_list.append(rd_slice)  # Append the slice to the list
        print(f'Finished slice: {len(rd_slices_list)} of {number_cpi}')
    print(f"Finished processing. Number of slices processed: {len(rd_slices_list)}")
    return rd_slices_list, velocity_axis, number_cpi, cpi_stride

