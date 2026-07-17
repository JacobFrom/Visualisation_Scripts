"""
Active radar processing, wrapped up as a single easy-to-use class.

The :class:`ActiveProcessor` bundles every step of the active (FMCW) processing
chain into one object. You create it once with the parameters of your radar, then
call the methods in order:

    from Active_processing.active_processor import ActiveProcessor

    radar = ActiveProcessor(centre_freq=5.8e9, bandwidth=80e6,
                            chirp_period=500e-6, prf=1024)

    radar.load_capture(directory)          # -> radar.iq_data
    radar.compute_beat_frequencies()       # -> radar.beat_frequencies, range/time axes
    radar.plot_rti(directory, 'rti_plot')
    radar.plot_mti(directory, 'mti_plot')
    radar.range_doppler()                  # -> radar.rd_slices, radar.velocity_axis
    radar.make_rd_video(directory, 'rd_video')

Each method stores its result on the object (``self.<something>``) so the next
step can find it automatically - you do not have to pass arrays around by hand.

The numeric processing inside every method is identical to the original
standalone functions in ``DSP_functions.py`` / the ``Testing`` scripts; this
class only reorganises that code into a friendlier interface.
"""

import os
import time

import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class ActiveProcessor:
    """End-to-end processing chain for an active (FMCW) radar capture."""

    def __init__(self, centre_freq=5.8e9, bandwidth=80e6, chirp_period=500e-6,
                 prf=1024, Fs=3.84e9, decimation=8 * 256, c=3e8, cmap='viridis'):
        """
        Parameters
        ----------
        centre_freq : float
            Radar centre frequency in Hz (e.g. ``2.44e9`` for S-band,
            ``5.8e9`` for C-band).
        bandwidth : float
            Chirp bandwidth in Hz.
        chirp_period : float
            Chirp length in seconds.
        prf : float
            Pulse Repetition Frequency in Hz.
        Fs : float
            Sample rate before decimation in Hz.
        decimation : int
            Total decimation factor applied during capture (default ``8 * 256``).
        c : float
            Speed of light in m/s.
        cmap : str
            Matplotlib colormap used for the plots and videos.
        """
        # --- Radar / waveform parameters ---
        self.centre_freq = centre_freq
        self.bandwidth = bandwidth
        self.chirp_period = chirp_period
        self.prf = prf
        self.Fs = Fs
        self.decimation = decimation
        self.c = c
        self.cmap = cmap

        # Derived quantities
        self.decimated_sample_rate = Fs / decimation
        self.wavelength = c / centre_freq

        # --- Results (filled in as you call the methods) ---
        self.iq_data = None
        self.capture_period = None
        self.beat_frequencies = None
        self.range_axis = None
        self.time_axis = None
        self.num_pulses = None
        self.num_samples = None
        self.num_range_bins = None

        self.rd_slices = None
        self.velocity_axis = None
        self.number_cpi = None
        self.cpi_stride = None

    # ------------------------------------------------------------------ #
    #  Loading and pre-processing
    # ------------------------------------------------------------------ #
    def load_capture(self, directory, filename='1.npy'):
        """
        Load a raw active capture from ``directory/filename`` and convert it to a
        complex baseband array.

        Sets ``self.iq_data`` and ``self.capture_period``.
        """
        data = np.load(os.path.join(directory, filename))

        print(data.shape)

        # Convert the data to complex numbers to make a baseband array.
        iq_data = data[:, 0::2] + 1j * data[:, 1::2]

        capture_period = data.shape[0] / self.prf  # Capture period in seconds

        self.iq_data = iq_data
        self.capture_period = capture_period
        return iq_data

    def compute_beat_frequencies(self):
        """
        Window and FFT the raw pulses to obtain the positive beat frequencies,
        and build the matching range and time axes.

        Sets ``self.beat_frequencies``, ``self.range_axis`` and
        ``self.time_axis``.
        """
        iq_data = self.iq_data

        # Begin the processing by windowing and then FFTing the data
        window = np.hamming(iq_data.shape[1])  # Create a hamming window function
        print("Window Shape:", np.shape(window))
        windowed_data = iq_data * window  # Apply hamming window function to the data

        print("Windowed data shape", windowed_data.shape)

        # perform fft on the newly windowed data
        fft_data = np.fft.fft(windowed_data, axis=1)  # FFT on windowed data along the samples

        num_pulses = fft_data.shape[0]  # Number of pulses
        # Keep only positive beat frequencies
        num_samples = fft_data.shape[1]
        num_range_bins = num_samples // 2  # Number of range bins
        positive_beat_frequencies = fft_data[:, :num_range_bins]

        print("Shape of positive_beat_freqs:", positive_beat_frequencies.shape)

        # Derive the slope of the chirp signal, used to calculate the range of each range bin in metres.
        slope = self.bandwidth / self.chirp_period
        range_bin_size = self.c * self.decimated_sample_rate / (2 * slope * num_samples)

        print("Ranges:", range_bin_size)

        # Derive appropriate range and time axis
        range_axis = np.linspace(0, (num_range_bins - 1) * range_bin_size, num_range_bins)

        print("Max range:", range_axis[-1])

        time_axis = np.linspace(0, self.capture_period, num_pulses)

        print("Shape of time axis:", time_axis)

        self.beat_frequencies = positive_beat_frequencies
        self.range_axis = range_axis
        self.time_axis = time_axis
        self.num_pulses = num_pulses
        self.num_samples = num_samples
        self.num_range_bins = num_range_bins
        return positive_beat_frequencies

    # ------------------------------------------------------------------ #
    #  Range-Time / MTI plots
    # ------------------------------------------------------------------ #
    def plot_rti(self, directory, save_as, v_min=-50, v_max=0, x_lim=(0, 200)):
        """Plot and save a Range-Time Intensity (RTI) plot of the beat frequencies."""
        beat_frequencies_array = self.beat_frequencies
        range_axis = self.range_axis
        time_axis = self.time_axis

        RTI_plot = (20 * np.log10(np.abs(beat_frequencies_array) / np.max(np.abs(beat_frequencies_array))))
        plt.imshow(RTI_plot, aspect='auto', origin='lower',
                   extent=[range_axis[0], range_axis[-1], time_axis[0], time_axis[-1]],
                   vmin=v_min, vmax=v_max, cmap=self.cmap)
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

    def plot_mti(self, directory, save_as, v_min=-50, v_max=0, x_lim=(0, 200)):
        """
        Apply a simple two-pulse MTI canceller to the beat frequencies and plot
        the result. The first pulse of the capture is lost by the differencing.
        """
        positive_beat_frequencies = self.beat_frequencies
        range_axis = self.range_axis

        # Subtract the previous pulse from the current pulse
        mti_filtered_data = positive_beat_frequencies[1:, :] - positive_beat_frequencies[:-1, :]
        print("Shape of mti_filtered_data:", mti_filtered_data.shape)
        # Re-establish the time axis; we have 1 less pulse (lost the very first pulse of the capture)
        mti_time_axis = np.linspace(0, self.capture_period, self.num_pulses - 1)

        print("Shape of mti_time_axis:", mti_time_axis)

        MTI_plot = (20 * np.log10(np.abs(mti_filtered_data) / np.max(np.abs(mti_filtered_data))))
        plt.imshow(MTI_plot, aspect='auto', origin='lower',
                   extent=[range_axis[0], range_axis[-1], mti_time_axis[0], mti_time_axis[-1]],
                   vmin=v_min, vmax=v_max, cmap=self.cmap)
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

    # ------------------------------------------------------------------ #
    #  Range-Doppler processing
    # ------------------------------------------------------------------ #
    def range_doppler(self, cpi_length=0.5, cpi_overlap=0.9, rd_window='blackmanharris'):
        """
        Perform range-Doppler slicing on the beat frequencies.

        Sets ``self.rd_slices`` (list of dB-scaled slices), ``self.velocity_axis``,
        ``self.number_cpi`` and ``self.cpi_stride``.
        """
        beat_frequencies = self.beat_frequencies
        prf = self.prf
        wavelength = self.wavelength

        # Configure the parameters for range-Doppler processing
        pulse_interval = 1 / prf
        num_pulses = beat_frequencies.shape[0]
        pulses_per_cpi = int(cpi_length * prf)
        cpi_stride = int(np.round(pulses_per_cpi * (1 - cpi_overlap)))

        window = signal.get_window(rd_window, pulses_per_cpi)

        number_cpi = int(np.ceil((num_pulses - pulses_per_cpi) / cpi_stride))

        print("Pulses per CPI:", pulses_per_cpi)
        print("CPI stride:", cpi_stride)

        # configure the axis for the range-Doppler slices
        slice_doppler_axis_unshifted = fft.fftfreq(pulses_per_cpi, d=pulse_interval)  # doppler axis in Hz
        slice_doppler_axis = fft.fftshift(slice_doppler_axis_unshifted)  # Shift to center zero frequency
        velocity_axis = (wavelength * slice_doppler_axis) / 2  # Convert to velocity in m/s

        # Now onto the slicing
        rd_slices_list = []  # Pre allocate the list for range-Doppler slices
        for i in range(number_cpi):
            start_index = i * cpi_stride
            end_index = start_index + pulses_per_cpi

            windowed_cpi = beat_frequencies[start_index:end_index] * window[:, np.newaxis]  # Apply window to CPI
            rd_slice = fft.fft(windowed_cpi, axis=0)
            rd_slice = fft.fftshift(rd_slice, axes=0)
            slice_plot = (20 * np.log10(np.abs(rd_slice) / np.max(np.abs(rd_slice))))

            rd_slices_list.append(slice_plot)  # Append the slice to the list
            print(f'Finished slice: {len(rd_slices_list)} of {number_cpi}')
        print(f"Finished processing. Number of slices processed: {len(rd_slices_list)}")

        self.rd_slices = rd_slices_list
        self.velocity_axis = velocity_axis
        self.number_cpi = number_cpi
        self.cpi_stride = cpi_stride
        return rd_slices_list, velocity_axis, number_cpi, cpi_stride

    def make_rd_video(self, directory, save_as, v_min=-50, v_max=0, x_lim=(0, 200),
                      y_lim=None, fps=None, sharpening=False):
        """
        Create an ``.mp4`` video from the range-Doppler slices.

        ``sharpening`` suppresses noise below ``v_min + 2`` dB in each frame.
        If ``fps`` is ``None`` it is derived from the capture period so the video
        plays back in real time. Requires ``ffmpeg`` on the system PATH.
        """
        rd_slices_list = self.rd_slices
        prf = self.prf
        cpi_stride = self.cpi_stride
        range_axis = self.range_axis
        velocity_axis = self.velocity_axis

        if fps is None:
            fps = 1 / (self.capture_period / self.number_cpi)  # Frames per second for the output video
        FPS = fps

        VIDEO_FILENAME = save_as + '.mp4'  # Output video filename
        VIDEO_SAVE = os.path.join(directory, VIDEO_FILENAME)
        COLOR_VMIN = v_min  # Min value for the color scale (dB) - adjust as needed
        COLOR_VMAX = v_max  # Max value for the color scale (dB)

        # --- Setup the Figure ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # Initial plot setup using the first frame
        initial_data = rd_slices_list[0]

        im = ax.imshow(initial_data,
                       aspect='auto',
                       origin='lower',
                       extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
                       vmin=COLOR_VMIN,
                       vmax=COLOR_VMAX,
                       cmap=self.cmap)

        # Add labels and title (base title)
        ax.set_xlabel('Range [m]')
        ax.set_ylabel('Velocity [m/s]')
        ax.set_xlim(x_lim)
        # Set the y-axis (Velocity) limits
        ax.set_ylim(y_lim)
        base_title = 'Range-Doppler Plot Evolution'
        title_text = ax.set_title(base_title)  # Store title object for updates

        # Add colorbar
        cbar = fig.colorbar(im, label='Norm Power [dB]')
        plt.tight_layout()  # Adjust layout to fit colorbar

        # Update function for the animation, nested to avoid confusion.
        # Includes some filtering to clean up the video.
        def update(frame_num):
            if sharpening:
                current_rd_slice = rd_slices_list[frame_num].copy()

                noise_threshold = v_min + 2  # Set a noise threshold for the video
                # Apply noise thresholding
                current_rd_slice[current_rd_slice < noise_threshold] = COLOR_VMIN

                im.set_data(current_rd_slice)  # Update image data
            else:
                current_rd_slice = rd_slices_list[frame_num]
                im.set_data(current_rd_slice)
            # Update title
            start_time_sec = (frame_num * cpi_stride) / prf

            current_title = f'{base_title}\nFrame: {frame_num + 1}/{len(rd_slices_list)}, Time ≈ {start_time_sec:.2f} s'

            title_text.set_text(current_title)

            return [im, title_text]  # Return updated artists

        # --- Create and Save Animation ---
        print(f"Creating animation with {len(rd_slices_list)} frames...")
        animation_start_time = time.perf_counter()
        ani = animation.FuncAnimation(fig,
                                      update,
                                      frames=len(rd_slices_list),
                                      interval=1000 / FPS,  # ms delay between frames
                                      blit=True,
                                      repeat=False)
        animation_end_time = time.perf_counter()
        animation_duration = animation_end_time - animation_start_time
        print(f'Animation creation took {animation_duration:.2f} seconds.')
        print(f"Finished creating animation with {len(rd_slices_list)} frames. Now saving to {VIDEO_SAVE}...")
        # Save the animation
        saving_start_time = time.perf_counter()
        # IMPORTANT: Ensure ffmpeg is installed and available in user path or this won't work
        ani.save(str(VIDEO_SAVE), writer='ffmpeg', fps=FPS, dpi=150)
        saving_end_time = time.perf_counter()
        saving_duration = saving_end_time - saving_start_time
        print(f"Animation saved to {VIDEO_SAVE} and saving took {saving_duration} seconds.")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Detection (CA-CFAR) - shared with PassiveProcessor
    # ------------------------------------------------------------------ #
    def cfar_detect(self, rd_matrix, window_params=(8, 8, 2, 2), threshold_db=23):
        """
        Cell-Averaging Constant False Alarm Rate (CA-CFAR) detection on a single
        range-Doppler matrix.

        Parameters
        ----------
        rd_matrix : complex ndarray
            Range-Doppler map (R x D) to run detection on.
        window_params : sequence of 4 ints
            [Est. window width, Est. window height, Guard width, Guard height].
        threshold_db : float
            Threshold above the estimated average noise power, in dB.

        Returns
        -------
        hit_matrix : bool ndarray
            True where the SNR exceeded the threshold.
        rd_snr : ndarray
            The estimated signal-to-noise ratio map.

        Implementation based on https://github.com/petotamas/APRiL
        """
        rd_size = rd_matrix.shape

        # Unpack the list of window parameters into individual variables for readability
        win_width = window_params[0]
        win_height = window_params[1]
        guard_width = window_params[2]
        guard_height = window_params[3]

        # Create the sliding window "kernel" (mask) initially filled with 1s (Reference Cells)
        # The size covers the full span: Reference + Guard + CUT (Cell Under Test)
        mask = np.ones((2 * win_height + 1, 2 * win_width + 1), dtype=bool)

        # Set the center region (Guard Cells and CUT) to 0 so they are excluded from the noise calculation
        mask[win_height - guard_height:win_height + 1 + guard_height,
             win_width - guard_width:win_width + 1 + guard_width] = 0

        # Convert the threshold from decibels (dB) to a linear power ratio
        threshold = 10 ** (threshold_db / 10)

        # Pre-calculate how many valid reference cells exist for every pixel position
        num_valid_cells_in_window = signal.convolve2d(np.ones(rd_size, dtype=float), mask, mode='same')

        # Convert the complex values (voltage) of the input matrix to Power (magnitude squared)
        rd_matrix = np.abs(rd_matrix) ** 2

        # 1. Sum the noise power around every cell
        rd_windowed_sum = signal.convolve2d(rd_matrix, mask, mode='same')

        # 2. Calculate the Average Noise Power
        rd_avg_noise_power = rd_windowed_sum / num_valid_cells_in_window

        # 3. Calculate SNR (Signal-to-Noise Ratio)
        rd_snr = rd_matrix / rd_avg_noise_power

        # 4. Determine Hits
        hit_matrix = rd_snr > threshold

        # Return the boolean detection map and the calculated SNR map
        return hit_matrix, rd_snr

    # ------------------------------------------------------------------ #
    #  Micro-Doppler - shared with PassiveProcessor
    # ------------------------------------------------------------------ #
    def micro_doppler(self, rd_slice_list, range_bin, slice_under_test, target_velocity,
                      notch_velocity, prf=None, NFFT=1024, Noverlap=None,
                      directory=None, save_as=None, target='Target', show=True):
        """
        Build a micro-Doppler spectrogram for a single range bin, reconstructed
        from complex range-Doppler slices.

        ``rd_slice_list`` must be a 3D complex array indexed
        ``[slice, doppler, range]`` (i.e. the complex slices, not the dB slices
        produced by :meth:`range_doppler`). ``target_velocity`` recentres the
        bulk motion to 0 Hz; ``notch_velocity`` sets the width of the main-body
        notch.
        """
        if prf is None:
            prf = self.prf
        if Noverlap is None:
            Noverlap = NFFT - 1

        wavelength = self.wavelength
        target_range = 20 * range_bin

        f_shift_hz = (2 * target_velocity / wavelength)

        start_frame = slice_under_test
        end_frame = slice_under_test + 1

        # --- Time-Domain Reconstruction (Stitching) ---
        slow_time_list = []
        for i in range(start_frame, end_frame):
            doppler_slice = rd_slice_list[i, :, range_bin]

            # Undo the zero-centering to prepare for IFFT
            doppler_unshifted = fft.ifftshift(doppler_slice)

            # Revert to slow-time
            slow_time_slice = fft.ifft(doppler_unshifted)

            slow_time_list.append(slow_time_slice)

        # Stitch them all together into one continuous 1D array
        stitched_slow_time = np.concatenate(slow_time_list)
        print(f"Stitched slow time matrix shape: {stitched_slow_time.shape}")

        # --- Bulk Motion Shifting ---
        N_samples = len(stitched_slow_time)
        t = np.arange(N_samples) / prf

        shifted_slow_time = stitched_slow_time * np.exp(-1j * 2 * np.pi * f_shift_hz * t)

        # --- Short-Time Fourier Transform (STFT) ---
        frequencies, times, Zxx = signal.stft(
            shifted_slow_time,
            fs=prf,
            window='hann',
            nperseg=NFFT,
            noverlap=Noverlap,
            nfft=NFFT,
            return_onesided=False  # Mandatory for complex radar data
        )

        # Move the new 0 Hz (the bulk motion) to the center of the arrays
        frequencies = fft.fftshift(frequencies)
        Zxx = fft.fftshift(Zxx, axes=0)

        # --- Main Body Cancellation (Notching) ---
        notch_width_hz = ((2 * notch_velocity) / wavelength)  # Total width of the suppression band
        notch_indices = np.abs(frequencies) < (notch_width_hz / 2)

        # Suppress the main body signal (tiny number instead of 0 to avoid log(0) errors)
        Zxx[notch_indices, :] = 1e-12

        # --- Dynamic Range & Decibel Conversion ---
        Zxx_mag = np.abs(Zxx)
        Zxx_mag = np.maximum(Zxx_mag, 1e-12)  # Floor to prevent log(0)
        Zxx_db = 25 * np.log10(Zxx_mag)

        # 99.5th percentile prevents single massive clutter spikes from washing out the plot
        vmax = np.percentile(Zxx_db, 99.5)
        vmin = vmax - 30  # dynamic range

        # --- Plotting ---
        plt.figure(figsize=(14, 6))
        pc = plt.pcolormesh(times, frequencies, Zxx_db, cmap=self.cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(pc, label='Power (dB)')
        plt.ylabel('Doppler Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title(f'Micro-Doppler Spectrogram of {target} at {target_range:.0f}m FFT Length: {NFFT}')
        plt.tight_layout()

        if directory is not None and save_as is not None:
            os.makedirs(directory, exist_ok=True)
            save_path = os.path.join(directory, save_as + '.png')
            plt.savefig(save_path)
            print(f"Micro-Doppler spectrogram saved to {save_path}")
        if show:
            plt.show()
        plt.close()

        return frequencies, times, Zxx_db

    # ------------------------------------------------------------------ #
    #  Power Spectral Density diagnostic
    # ------------------------------------------------------------------ #
    def plot_psd(self, directory, filename, seconds_of_interest=5, reshaped=True, show=True):
        """
        Quick Power Spectral Density plot of a raw active capture, useful as a
        sanity check on the spectrum before running the full chain.
        """
        data = np.load(os.path.join(directory, filename))

        if reshaped:
            data = data[:, 0::2] + 1j * data[:, 1::2]
        else:
            data = data[0::2] + 1j * data[1::2]

        prf = self.prf
        Fs = self.Fs

        N = prf * seconds_of_interest  # Number of samples for fft

        fft_data = data[0:N, :]  # Take the first N pulses for the fft
        window = np.hamming(fft_data.shape[0])  # Create a hamming window function
        windowed_data = fft_data * window[:, np.newaxis]  # Apply hamming window function to the data

        PSD = np.abs(np.fft.fft(windowed_data)) ** 2 / (N * Fs)  # Power Spectral Density

        PSD_dB = 10.0 * np.log10(PSD)  # Convert to dB

        PSD_shifted = np.fft.fftshift(PSD_dB)  # Shift the PSD to center around 0

        f = np.arange(0, (Fs / self.decimation), (Fs / self.decimation) / N)  # Start, Stop, Step
        f += self.centre_freq  # Shift the frequency axis to the centre frequency

        plt.plot(f, PSD_shifted)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        plt.grid(True)
        if show:
            plt.show()
        plt.close()
