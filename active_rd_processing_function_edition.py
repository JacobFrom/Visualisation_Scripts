import os
import numpy as np
import DSP_functions as DSP
from DSP_functions import create_rd_video

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
    captures = ['dorch_drone7_run2']  # Set to a specific capture directory if not batch processing

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

    RTI_plot_flag = True

    if RTI_plot_flag:

        v_min = -50  # Minimum value for RTI plot
        v_max = 0    # Maximum value for RTI plot
        x_lim = [0, 120]  # X-axis limits for the RTI plot
        RTI_file_name = f'function_test_rti_plot'  # Output RTI plot filename

        DSP.plot_RTI(positive_beat_frequencies, range_axis, time_axis, v_min, v_max,CMAP, x_lim, directory, RTI_file_name)
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
        mti_file_name = f'function_test_mti_plot'  # Output MTI plot filename

        print("Shape of mti_time_axis:", mti_time_axis)

        DSP.plot_MTI(mti_filtered_data,range_axis, mti_time_axis, mti_v_min, mti_v_max, CMAP, mti_x_lim, directory, mti_file_name)  #Now we can plot the MTI filtered data

        print("MTI plot created and saved successfully.")
        del mti_filtered_data, mti_time_axis  # Free up memory by deleting the MTI filtered data and time axis
        print("MTI filtered data and time axis deleted to free up memory.")
    #Now we perform some range doppler processing and make a video

    rd_processing_flag = True

    if rd_processing_flag:
        print("Starting Range-Doppler processing...")

        wavelength = c / centre_freq #Calculate the wavelength in metres

        cpi_length = 0.5 #Define the coherent processing interval in seconds
        cpi_overlap = 0.9  #Define the overlap between each coherent processing intervals

        rd_window = 'blackmanharris'

        rd_slices, velocity_axis, number_cpi, cpi_stride = DSP.range_Doppler_slicing(positive_beat_frequencies, cpi_length, cpi_overlap, prf, rd_window, wavelength) #Perform range-Doppler slicing

        print(f"Finished processing. Number of slices processed: {len(rd_slices)}")

        """ RD slicing is finished, below code is for creating range-Doppler Videos. Note: Video creation function has some built-in filtering
        That you may wish to adjust based on how your videos look."""


        FPS = 1/((capture_period/number_cpi))  # Frames per second for the output video


        VIDEO_FILENAME = f'function_test_rd_video_3'  # Output video filename
        VIDEO_SAVE = os.path.join(directory, VIDEO_FILENAME)
        COLOR_VMIN = -60  # Min value for the color scale (dB) - adjust as needed
        COLOR_VMAX = 0    # Max value for the color scale (dB)
        vid_x_lim = [0, 120]  # X-axis limits for the video

        create_rd_video(FPS, rd_slices, prf, cpi_stride, range_axis, velocity_axis, COLOR_VMIN, COLOR_VMAX, CMAP, vid_x_lim, directory, VIDEO_FILENAME)

        print(f"Finished processing dataset: {experiment_codename}")