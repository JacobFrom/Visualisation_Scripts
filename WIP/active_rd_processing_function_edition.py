import os
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal
import matplotlib.animation as animation
from scipy.ndimage import median_filter
import time
import DSP_functions as DSP

# Below are the parameters of the radar capture
s_band = 2.44e9  # active centre frequency
c_band = 5.8e9

c = 3e8
centre_freq = c_band  # active centre frequency
bandwidth = 80e6  # chirp bandwidth
chirp_period = 500e-6  # chirp length
prf = 1024  # Pulse Repitition Frequency

Fs = 3.84e9
decimated_sample_rate = Fs / (8 * 256)  # Decimated sample rate

#Decide here if you want to batch process or process individual datasets
batch_process_flag = False  # Set to True to process all datasets in the directory, False to process a single dataset

experiment_directory = fr'G:\ARESTOR_DATA\dorchester_active_jf' # Directory containing the experiment data

if batch_process_flag:
    captures = os.listdir(experiment_directory)
else:
    captures = 'dorch_drone1_run4'  # Set to a specific capture directory if not batch processing

for capture in captures:

    experiment_codename = capture  # Set the experiment codename to the current capture directory

    directory = fr'{experiment_directory}\{experiment_codename}\active_test_1'


    data = np.load(directory + r'\1.npy')

    print(data.shape)

    iq_data = data[:, 0::2] + 1j*data[:, 1::2] #Convert the data to complex numbers to make a baseband array.

    capture_period = data.shape[0] / prf #Calculate the capture period in seconds

    #Begin the processing by windowing and then FFTing the data

    window = np.hamming(iq_data.shape[1]) #Create a hamming window function
    print("Window Shape:", np.shape(window))
    windowed_data = iq_data * window #Apply hamming window function to the data, np.newaxis adds a new axis to the window array, allowing for element-wise multiplication

    print ("Windowed data shape", windowed_data.shape)

    #perform fft on the newly windowed data
    fft_data = np.fft.fft(windowed_data, axis=1) #Perform FFT on windowed data along the samples

    num_pulses = fft_data.shape[0] #Number of pulses
    # Keep only positive beat frequencies
    num_samples = fft_data.shape[1]
    num_range_bins = num_samples // 2 #Number of range bins
    positive_beat_frequencies = fft_data[:, :num_range_bins]


    print("Shape of positive_beat_freqs:", positive_beat_frequencies.shape)

    slope = bandwidth / chirp_period #Derive the slope of the chirp signal, This is used to calculate the range of each range bin in metres.
    range_bin_size = c * decimated_sample_rate / (2 * slope * num_samples) #Calculate the range of each range bin in metres.

    print("Ranges:", range_bin_size)

    #Derive appropriate range and time axis
    range_axis = np.linspace(0, (num_range_bins-1)*range_bin_size, num_range_bins)#Create a range axis from 0 to the maximum range in metres

    #print("Shape of range axis:", range_axis)

    time_axis = np.linspace(0, capture_period, num_pulses)

    print("Shape of time axis:", time_axis)

    CMAP = 'inferno' #Colormap for the plots

    #Code for removing close in clutter is in elipses
    #If you wish to crop some data out use following code:

    #first_N_bins_to_crop = 10 #Number of range bins to crop from the start of the data
    #cropped_data = positive_beat_frequencies[:, first_N_bins_to_crop:num_range_bins]
    #print("Shape of cropped_data:", cropped_data.shape)

    #To generate the RTI plot, change flag value to True

    RTI_plot_flag = False

    if RTI_plot_flag:

        v_min = -50  # Minimum value for RTI plot
        v_max = 0    # Maximum value for RTI plot
        x_lim = [0, 120]  # X-axis limits for the RTI plot

        DSP.plot_RTI(positive_beat_frequencies, range_axis, time_axis, v_min, v_max,CMAP, x_lim, directory)
        print("RTI plot created and saved successfully.")

    # #Now lets make and apply an MTI filter and produce another plot:

    MTI_plot_flag = True

    if MTI_plot_flag:

        mti_filtered_data = positive_beat_frequencies[1:, :] - positive_beat_frequencies[:-1, :] #Subtract the previous pulse from the current pulse
        print("Shape of mti_filtered_data:", mti_filtered_data.shape)
        #need to re-establish the time axis as now we have 1 less pulse, we have lost the very first pulse of the capture
        mti_time_axis = np.linspace(0, capture_period, num_pulses-1) #Create a time axis from 0 to the maximum time in seconds

        #Here we can also set MTI specific V-min and V-max values if desired, or can leave them as above
        mti_v_min = -50  # Minimum value for MTI plot
        mti_v_max = 0   # Maximum value for MTI plot
        mti_x_lim = [0, 120] # Range limit for the MTI plot

        print("Shape of mti_time_axis:", mti_time_axis)

        DSP.plot_MTI(mti_filtered_data,range_axis, mti_time_axis, mti_v_min, mti_v_max, CMAP, mti_x_lim, directory)  #Now we can plot the MTI filtered data

        print("MTI plot created and saved successfully.")
        del mti_filtered_data, mti_time_axis  # Free up memory by deleting the MTI filtered data and time axis
        print("MTI filtered data and time axis deleted to free up memory.")
    #Now we perform some range doppler processing and make a video

    rd_processing_flag = True

    if rd_processing_flag:
        print("Starting Range-Doppler processing...")

        pulse_interval = 1 / prf #Calculate the pulse interval in seconds

        wavelength = c / centre_freq #Calculate the wavelength in metres

        cpi_length = 0.5 #Define the coherent processing interval in seconds
        cpi_overlap = 0.9  #Define the overlap between each coherent processing intervals

        rd_window = 'blackmanharris'

        pulses_per_cpi = int(cpi_length * prf) #How many pulses to include in each CPI, aka fft length/NFFT points
        cpi_stride = int(np.round(pulses_per_cpi * (1-cpi_overlap))) #Stride length for the next CPI
        print("Pulses per CPI:", pulses_per_cpi)
        print("CPI stride:", cpi_stride)
        window = signal.get_window('blackmanharris', pulses_per_cpi)  # Define the windowing function to be applied to the data
        number_cpi = int(np.ceil((num_pulses - pulses_per_cpi) / cpi_stride)) #Calculate the number of CPIs
        print("Number of CPIs:", number_cpi)

        slice_doppler_axis_unshifted = np.fft.fftfreq(pulses_per_cpi, d=pulse_interval) #Calculate the doppler axis in Hz
        slice_doppler_axis = np.fft.fftshift(slice_doppler_axis_unshifted) #Shift the doppler axis to match the shifted rd_matrix
        velocity_axis = (wavelength * slice_doppler_axis) / 2 #Calculate the velocity axis in m/s
        print(f"Velocity range (m/s): {velocity_axis[0]:.2f} m/s to {velocity_axis[-1]:.2f} m/s") #Here the :.2f formats the float to 2 decimal places

        rd_slices_list = [] #List to hold the range doppler slices
        for i in range(number_cpi):
            start_index = i * cpi_stride
            end_index = start_index + pulses_per_cpi
            print("Start index:", start_index)
            print("End index:", end_index)
            windowed_cpi = positive_beat_frequencies[start_index:end_index] * window[:, np.newaxis] #Apply the window to the current CPI
            rd_slice = np.fft.fft(windowed_cpi, axis=0)
            rd_slice = np.fft.fftshift(rd_slice, axes=0)
            slice_plot = (20 * np.log10(np.abs(rd_slice) / np.max(np.abs(rd_slice))))

            rd_slices_list.append(slice_plot) #Append the slice to the list
            print(f'Finished slice: {len(rd_slices_list)} of {number_cpi}')


        print(f"Finished processing. Number of slices processed: {len(rd_slices_list)}")
        #Next step is to create an animation
        #now lets use matplotlib animation

        # #here we attempt some filtering on the list
        #
        # processed_slices = []
        # filter_size = 3
        #
        # for i, rd_slice in enumerate(rd_slices_list):
        #     # Apply median filter to the slice
        #     filtered_slice = median_filter(rd_slice, size=(filter_size))
        #     processed_slices.append(filtered_slice)
        #
        # print(f"Finished filtering. Number of slices processed: {len(processed_slices)}")


        FPS = 1/((capture_period/number_cpi))  # Frames per second for the output video
        VIDEO_FILENAME = f'{CMAP}_bmanh_rd_video.mp4'  # Output video filename
        VIDEO_SAVE = os.path.join(directory, VIDEO_FILENAME)
        COLOR_VMIN = -63  # Min value for the color scale (dB) - adjust as needed
        COLOR_VMAX = -10    # Max value for the color scale (dB)


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
        ax.set_xlim(range_axis[0], 150)
        # Set the y-axis (Velocity) limits
        ax.set_ylim(-13.4, 13.4)
        base_title = 'Range-Doppler Map Evolution'
        title_text = ax.set_title(base_title) # Store title object for updates

    # Add colorbar
        cbar = fig.colorbar(im, label='Norm Power [dB]')
        plt.tight_layout() # Adjust layout to fit colorbar

    # --- Update Function for Animation ---
    def update(frame_num):

        #current_rd_slice = rd_slices_list[frame_num]
        #current_rd_slice = processed_slices[frame_num]
        #current_rd_slice = processed_slices[frame_num].copy()
        current_rd_slice = rd_slices_list[frame_num].copy()

        noise_threshold = -58
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
    #print(f"Creating animation with {len(processed_slices)} frames...")
    start_time = time.perf_counter()

    ani = animation.FuncAnimation(fig,
                                  update,

                                  frames=len(rd_slices_list),
                                  #frames=len(processed_slices),

                                  interval=1000 / FPS,  # ms delay
                                  blit=True,  # Use blitting for efficiency
                                  repeat=False)

    ani.save(VIDEO_SAVE, writer='ffmpeg', fps=FPS, dpi=150)

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Animation saved to {VIDEO_SAVE}")
    print(f"Animation creation and saving took {duration:.2f} seconds.")

    plt.close(fig)
    print(f"Finished processing dataset: {experiment_codename}")