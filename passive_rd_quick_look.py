import numpy as np
import scipy.signal as signal
import time
import os # Only needed if adding progress prints with PID, otherwise optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define complex type consistently
complex_dtype = np.complex64

def generate_quick_look_rds(ref_iq, surv_iq, Fs_eff, centre_freq,
                            seg_s, # How many correlation segments per second (e.g., 1000)
                            max_range_meters, # Max range (excess path) in meters
                            doppler_duration_sec, # How much time each RD slice represents
                            doppler_overlap, # Overlap for RD slices
                            doppler_window='blackmanharris'): # Window for Doppler FFT
    """
    Simplified 'Quick Look' Passive RD Slice Generator (Reshape-First Method).

    Args:
        ref_iq (np.ndarray): Full 1D complex reference signal (complex64 recommended).
        surv_iq (np.ndarray): Full 1D complex surveillance signal (complex64 recommended).
        Fs_eff (float): Effective sample rate (Hz).
        centre_freq (float): Centre frequency (Hz) for wavelength calculation.
        seg_s (int): Segments per second for correlation stage (e.g., 1000 Hz).
                     Determines slow_time_step and max unambiguous velocity.
        max_range_meters (float): Max excess path length (m) to calculate correlation for.
        doppler_duration_sec (float): Duration (s) represented by each Doppler FFT / RD slice.
        doppler_overlap (float): Fractional overlap (0-1) between consecutive RD slices.
        doppler_window (str, optional): Window type for Doppler FFT. Defaults to 'hann'.

    Returns:
        tuple: (rd_slices_list, range_axis_m, velocity_axis), or ([], [], []) on error.
               rd_slices_list contains 2D arrays (Doppler Bin x Range Bin) in norm. dB.
    """
    print(f"\n--- Starting Quick Look RD Generation (seg_s={seg_s}) ---")
    start_time = time.perf_counter()

    C = 3e8
    wavelength = C / centre_freq

    # --- 1. Calculate Correlation Stage Parameters & Reshape ---
    max_delay_sec = max_range_meters / C
    max_delay_samples = int(np.ceil(max_delay_sec * Fs_eff))
    seg_duration = 1.0 / seg_s
    seg_size = int(np.round(seg_duration * Fs_eff)) # Samples per short correlation segment

    if seg_size < 2:
        print("ERROR: seg_s too high for Fs_eff (segment size < 2).")
        return [], np.array([]), np.array([])

    total_samples = min(ref_iq.shape[0], surv_iq.shape[0])
    num_segments = total_samples // seg_size # Total correlation segments

    if num_segments < 2:
        print("ERROR: Not enough data for multiple segments.")
        return [], np.array([]), np.array([])

    print(f"Using {num_segments} segments of {seg_size} samples ({seg_duration*1000:.1f} ms).")
    print(f"Max correlation delay: {max_delay_samples} samples ({max_range_meters} m)")

    # Trim and reshape input data (Using Fortran order for column-major access)
    trimmed_len = num_segments * seg_size
    try:
        reshaped_ref = ref_iq[:trimmed_len].reshape((seg_size, num_segments), order='F').astype(complex_dtype)
        reshaped_surv = surv_iq[:trimmed_len].reshape((seg_size, num_segments), order='F').astype(complex_dtype)
    except MemoryError:
         print("ERROR: Not enough memory to reshape arrays. Try reducing data length or parameters.")
         return [], np.array([]), np.array([])
    except Exception as e:
         print(f"ERROR during reshape: {e}")
         return [], np.array([]), np.array([])

    # --- 2. Fast Cross-Correlation (Segment by Segment) ---
    # Output matrix: rows = segments (slow time), cols = lags
    correlation_matrix = np.zeros((num_segments, max_delay_samples), dtype=complex_dtype)
    n_fft_corr = seg_size # Use exact length for correlation FFT (faster?)

    print("Performing quick cross-correlation...")
    corr_start_time = time.perf_counter()
    for i in range(num_segments): # Loop columns (segments)
        ref_seg = reshaped_ref[:, i]
        surv_seg = reshaped_surv[:, i]

        # Use SciPy correlate (method='fft' default)
        full_corr = signal.correlate(surv_seg, ref_seg, mode='full')
        zero_lag_idx = seg_size - 1
        lags_needed = full_corr[zero_lag_idx : zero_lag_idx + max_delay_samples]
        correlation_matrix[i, :len(lags_needed)] = lags_needed

    del reshaped_ref, reshaped_surv # Free memory
    print(f"Correlation finished in {time.perf_counter() - corr_start_time:.2f} s.")

    # --- 3. Doppler Processing Parameters & Axis ---
    slow_time_step_ql = 1.0 / seg_s # Time between rows of correlation_matrix
    N_doppler = int(np.round(doppler_duration_sec / slow_time_step_ql))
    if N_doppler < 2: N_doppler = 2
    if N_doppler > num_segments: N_doppler = num_segments

    doppler_stride = int(np.round(N_doppler * (1.0 - doppler_overlap)))
    if doppler_stride < 1: doppler_stride = 1
    num_rd_slices = int(np.floor((num_segments - N_doppler) / doppler_stride)) + 1

    print(f"Doppler processing: N_doppler={N_doppler}, Stride={doppler_stride}, Num Slices={num_rd_slices}")

    velocity_axis = np.fft.fftshift(np.fft.fftfreq(N_doppler, d=slow_time_step_ql)) * wavelength / 2

    # --- 4. Doppler FFT Loop ---
    print("Performing Doppler processing...")
    rd_slices_list = []
    window = signal.get_window(doppler_window, N_doppler)[:, np.newaxis]

    for i in range(num_rd_slices):
        start_row = i * doppler_stride
        end_row = start_row + N_doppler
        if end_row > num_segments: end_row = num_segments

        block = correlation_matrix[start_row:end_row, :]
        current_N_doppler = block.shape[0]
        if current_N_doppler < 2: continue

        current_window = window if current_N_doppler == N_doppler else signal.get_window(doppler_window, current_N_doppler)[:, np.newaxis]

        rd_slice_shifted = np.fft.fftshift(np.fft.fft(block * current_window, n=current_N_doppler, axis=0), axes=0)

        with np.errstate(divide='ignore'):
            slice_plot = 10 * np.log10(np.abs(rd_slice_shifted)**2)
        max_val = np.nanmax(slice_plot)
        if np.isfinite(max_val): slice_plot -= max_val
        slice_plot[np.isneginf(slice_plot)] = -120.0 # Basic floor

        rd_slices_list.append(slice_plot)

    # --- 5. Calculate Range Axis ---
    delay_axis_sec = np.arange(max_delay_samples) / Fs_eff
    range_axis_m = delay_axis_sec * C

    total_time = time.perf_counter() - start_time
    print(f"--- Quick Look Processing Finished in {total_time:.2f} seconds ---")

    return rd_slices_list, range_axis_m, velocity_axis, doppler_stride

# ============================================
# --- Example Usage ---
# ============================================
if __name__ == "__main__":
    # --- Configuration ---
    experiment_codename = 'run_7'

    capture_duration = 30
    centre_freq = 498e6  # Centre freq of DVB-T capture DVB-T

    decimator_ip = 32  # Choose between 4x or 32x
    decimator_tile = 8

    base_data_directory = r'G:\trial_3_data'
    data_subdirectory = 'passive_data_test_1'
    output_video_directory = fr'D:\ARESTOR_DATA\trial_3_rd_videos\{experiment_codename}\passive_test_1'

    input_data_dir = os.path.join(base_data_directory, experiment_codename, data_subdirectory)

    ref_filename = 'passive_ref_output.npy'
    surv_filename = 'passive_surv_output.npy'

    ref_path = os.path.join(input_data_dir, ref_filename)
    surv_path = os.path.join(input_data_dir, surv_filename)

    os.makedirs(output_video_directory, exist_ok=True)  # Ensure output dir exists

    ref_data_raw = np.load(ref_path)
    surv_data_raw = np.load(surv_path)

    complex_dtype = np.dtype('complex64')
    # De interleave and convert to complex

    ref_i = ref_data_raw[0::2]  # Select elements 0, 2, 4,... (Real/I)
    ref_q = ref_data_raw[1::2]  # Select elements 1, 3, 5,... (Imaginary/Q)
    ref_iq = (ref_i + 1j * ref_q).astype(np.complex64)  # Combine into complex numbers
    del ref_i, ref_q  # Optional: Release memory if I/Q parts are large

    # Surveillance Channel
    # Check if data length is even
    if surv_data_raw.shape[0] % 2 != 0:
        print("Warning: Surveillance data has odd number of samples. Trimming last sample.")
        surv_data_raw = surv_data_raw[:-1]

    surv_i = surv_data_raw[0::2]  # Select elements 0, 2, 4,... (Real/I)
    surv_q = surv_data_raw[1::2]  # Select elements 1, 3, 5,... (Imaginary/Q)
    surv_iq = (surv_i + 1j * surv_q).astype(np.complex64)  # Combine into complex numbers
    del surv_i, surv_q  # Optional: Release memory

    # Optional: Delete raw data if memory is tight, but only AFTER successful conversion
    del ref_data_raw, surv_data_raw

    print(f"Shape of complex reference data (ref_iq): {ref_iq.shape}")  # Should be (450000000,)
    print(f"Shape of complex surveillance data (surv_iq): {surv_iq.shape}")  # Should be (450000000,)

    # Calculate Effective Sample Rate

    Fs_original = 3.84e9  # Original sample rate
    decimation_factor = decimator_ip * decimator_tile  # Decimation factor from decimator ip block multiplied by tile decimation

    # Calculate Effective Sample Rate
    Fs_eff = Fs_original / decimation_factor
    print(f"Effective Sample Rate (Fs_eff): {Fs_eff / 1e6:.2f} MHz ({Fs_eff} Hz)")

    # --- Processing Parameters ---
    C = 3e8  # Speed of light in m/s
    wavelength = C / centre_freq  # Wavelength in meters
    # --- Quick Look Parameters ---
    quick_seg_s = 1000       # Hz -> sets Vmax to +/- 150 m/s, uses 1ms correlation segments
    quick_max_range = 280    # Meters (excess path) -> sets max_delay_samples=14
    quick_doppler_dur = 1.0  # Seconds -> Sets duration over which Doppler FFT is done
    quick_doppler_olap = 0.75# Overlap for Doppler FFT window
    quick_doppler_win = 'blackmanharris' # Window for Doppler FFT

    # --- Run Quick Look ---
    # Use try/except here if you want robustness for the call
    q_slices, q_range, q_velocity, doppler_stride = generate_quick_look_rds(
        ref_iq, surv_iq, Fs_eff, centre_freq,
        seg_s=quick_seg_s,
        max_range_meters=quick_max_range,
        doppler_duration_sec=quick_doppler_dur,
        doppler_overlap=quick_doppler_olap,

    )

    # --- Use the results ---
    if q_slices:
        print(f"\nQuick Look Results:")
        print(f"  Number of slices: {len(q_slices)}")
        print(f"  Slice shape: {q_slices[0].shape}")
        print(f"  Range axis points: {q_range.shape[0]}")
        print(f"  Velocity axis points: {q_velocity.shape[0]}")
        print(f"  Velocity range: {q_velocity[0]:.2f} m/s to {q_velocity[-1]:.2f} m/s")

        # Now you can pass q_slices, q_range, q_velocity to an animation function
        # Example: create_rd_animation(q_slices, q_range, q_velocity, ...)
    else:
        print("Quick Look processing failed or produced no slices.")

    FPS = 1 / ((capture_duration / len(q_slices)))  # Frames per second for the output video
    VIDEO_FILENAME = os.path.join(output_video_directory, 'quick_rd.mp4') # Output video filename
    COLOR_VMIN = -63  # Min value for the color scale (dB) - adjust as needed
    COLOR_VMAX = -10  # Max value for the color scale (dB)
    CMAP = 'inferno'
    # --- Setup the Figure ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initial plot setup using the first frame

    initial_data = q_slices[0]
    # initial_data = processed_slices[0]

    im = ax.imshow(initial_data,
                   aspect='auto',
                   origin='lower',
                   extent=[q_range[0], q_range[-1], q_velocity[0], q_velocity[-1]],
                   vmin=COLOR_VMIN,
                   vmax=COLOR_VMAX,
                   cmap=CMAP)

    # Add labels and title (base title)
    ax.set_xlabel('Range [m]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_xlim(q_range[0], 90)
    # Set the y-axis (Velocity) limits
    ax.set_ylim(-13.4, 13.4)
    base_title = 'Range-Doppler Map Evolution'
    title_text = ax.set_title(base_title)  # Store title object for updates

    # Add colorbar
    cbar = fig.colorbar(im, label='Norm Power [dB]')
    plt.tight_layout()  # Adjust layout to fit colorbar


# --- Update Function for Animation ---
def update(frame_num):
    # current_rd_slice = rd_slices_list[frame_num]
    # current_rd_slice = processed_slices[frame_num]
    # current_rd_slice = processed_slices[frame_num].copy()
    current_rd_slice = q_slices[frame_num].copy()

    noise_threshold = -58
    # Apply noise thresholding
    current_rd_slice[current_rd_slice < noise_threshold] = COLOR_VMIN

    im.set_data(current_rd_slice)  # Update image data

    # Update title
    start_time_sec = (frame_num * doppler_stride) / quick_seg_s

    current_title = f'{base_title}\nFrame: {frame_num + 1}/{len(q_slices)}, Time ≈ {start_time_sec:.2f} s'
    # current_title = f'{base_title}\nFrame: {frame_num + 1}/{len(processed_slices)}, Time ≈ {start_time_sec:.2f} s'

    title_text.set_text(current_title)

    return [im, title_text]  # Return updated artists


# --- Create and Save Animation ---

print(f"Creating animation with {len(q_slices)} frames...")
# print(f"Creating animation with {len(processed_slices)} frames...")
start_time = time.perf_counter()

ani = animation.FuncAnimation(fig,
                              update,

                              frames=len(q_slices),
                              # frames=len(processed_slices),

                              interval=1000 / FPS,  # ms delay
                              blit=True,  # Use blitting for efficiency
                              repeat=False)

ani.save(VIDEO_FILENAME, writer='ffmpeg', fps=FPS, dpi=150)

end_time = time.perf_counter()
duration = end_time - start_time
print(f"Animation saved to {VIDEO_FILENAME}")
print(f"Animation creation and saving took {duration:.2f} seconds.")

plt.close(fig)