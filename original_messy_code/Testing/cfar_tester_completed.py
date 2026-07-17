import time
import numpy as np
from original_messy_code.old_scripts.CaCFAR_orig import CA_CFAR as CFAR
import DetProcessing as dp
import matplotlib.pyplot as plt
import os

# Define the folder path where the data is stored
directory = fr'G:\ARESTOR_DATA\Ulm_trial\tests\day_2\ulm_day_2_3020\passive_data_test_1'

# Load the large 3D array containing all the Range-Doppler matrices
rd_matrices = np.load(directory + r'\rd_slices\4000_rd_slice_matrix.npy')

#Set the CA window parameters, format: [window width, window height, guard cell width, guard cell height]
window_params = [8, 8, 2, 2]
# Set the detection threshold in dB
threshold_dB = 23
# Get the shape of a single matrix (rows=Doppler, cols=Range) for configuration
data_shape = rd_matrices[0].shape
# Calculate max range in meters, this assumes 20m per range cell (True for single channel DVB-T passive radar)
max_range_m = data_shape[1] * 20

# Toggle this flag: False = calculate fresh CFAR data; True = load previous results
CFAR_hit_matrix_exists_flag = True

# ==========================================
# CFAR PROCESSING SECTION
# ==========================================
if not CFAR_hit_matrix_exists_flag:
    # Initialize empty lists to store the detection results and SNR maps
    hit_results = []
    snr_results = []

    print(f"Processing {len(rd_matrices)} slices...")
    # Start the timer to track performance
    start = time.time()

    # Create the CFAR processor object once with our settings
    ca_cfar = CFAR(window_params, threshold_dB, data_shape)

    # Loop through every single matrix in the stack
    for i, rd_matrix in enumerate(rd_matrices):
        # Run the CFAR algorithm on the current slice
        hit_matrix, rd_snr = ca_cfar(rd_matrix)
        #Set last 3 range bins to false to avoid false detections at max range from CFAR/Cross correlation artefacts
        hit_matrix[:, -3:] = False

        # Save the boolean hits (detections) to our list
        hit_results.append(hit_matrix)
        # Save the SNR values to our list
        snr_results.append(rd_snr)

    # Stop the timer
    end = time.time()

    print(f"CFAR processing time Complete. Time taken: {end - start} seconds. Saving results...")
    # Define file paths for saving the output
    cfar_directory = directory + r'\cfar_results'
    os.makedirs(cfar_directory, exist_ok=True)
    hit_save = cfar_directory + fr'\{max_range_m}_cfar_hit_matrix.npy'
    snr_save = cfar_directory + fr'\{max_range_m}_cfar_sinr_matrix.npy'

    # Save the list of hit matrices to disk
    np.save(hit_save, hit_results)
    # Save the list of SNR matrices to disk (Corrected from hit_results to snr_results)
    np.save(snr_save, snr_results)
    print(f"CFAR processing complete. Hit matrix and SNR matrix saved in {cfar_directory}.")

else:
    print("Loading pre-computed CFAR results from disk...")
    # Load the previously saved boolean hit matrices
    hit_results = np.load(directory + fr'\cfar_results\{max_range_m}_cfar_hit_matrix.npy')
    # Load the previously saved SNR matrices
    snr_results = np.load(directory + fr'\cfar_results\{max_range_m}_cfar_sinr_matrix.npy')
    print("CFAR results loaded.")

# ==========================================
# VISUALIZATION SECTION
# ==========================================

# Choose which slice (frame) to look at
slice_number = 525

# Extract the specific Hit matrix for this frame
hit_matrix = hit_results[slice_number]
# Extract the specific SNR matrix for this frame
rd_snr = snr_results[slice_number]
# Extract the original raw data matrix for the background image
rd_matrix = rd_matrices[slice_number]

print("Generating plots...")
# Create a linear array for the Range axis (0 to max range)
range_axis_m = np.linspace(0, max_range_m, int(data_shape[1]))
# Set the floor for the dB plot (minimum intensity)
dynamic_range = -35
# Calculate time between pulses based on number of Doppler cells
pulse_interval = 1 / int(data_shape[0])
# Speed of light constant
c = 3e8
# Carrier frequency of the radar
centre_freq = 570e6
# Calculate signal wavelength
wavelength = c / centre_freq

# Generate the frequency axis for Doppler (unshifted)
slice_doppler_axis_unshifted = np.fft.fftfreq(data_shape[0], d=pulse_interval)
# Shift the zero frequency to the center of the array
slice_doppler_axis = np.fft.fftshift(slice_doppler_axis_unshifted)
# Convert the frequency axis to Velocity (m/s)
velocity_axis = (wavelength * slice_doppler_axis) / 2

# Group adjacent detection pixels into single targets (Clustering) - See DetProcessing.py-> plot_extractor function
plots = dp.plot_extractor(hit_matrix, rd_snr, d_max=2)

# Filter out bad targets (static clutter or close-range noise) - See DetProcessing.py-> plot_filter function
plots = dp.plot_filter(plots, range_gate=5, Doppler_gate=2, rd_size=data_shape)

print(f"Extracted {len(plots)} targets from the raw hits.")

# Create a figure with two side-by-side subplots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# --- Left Plot: Raw CFAR Hit Matrix ---
# Display the boolean hit matrix (Black/White)
ax[0].imshow(hit_matrix, aspect='auto', interpolation='nearest', origin='lower',
             extent=[range_axis_m[0], range_axis_m[-1], velocity_axis[0], velocity_axis[-1]],
             cmap='gray')
ax[0].set_title("Raw CFAR Hit Matrix (Boolean)")
ax[0].set_xlabel("Range (m)")
ax[0].set_ylabel("Velocity (m/s)")

# --- Right Plot: Range-Doppler Map with Target Overlays ---

# Convert the raw complex data to decibels (dB), normalized to the max value
rd_db = (20 * np.log10(np.abs(rd_matrix) / np.max(np.abs(rd_matrix))))

# Display the background heatmap (Power in dB)
im = ax[1].imshow(rd_db, aspect='auto', interpolation='nearest', origin='lower',
                  extent=[range_axis_m[0], range_axis_m[-1], velocity_axis[0], velocity_axis[-1]],
                  vmin=dynamic_range, vmax=0,
                  cmap='viridis')
# Add a colorbar to show the scale
plt.colorbar(im, ax=ax[1], label='Normalised Power (dB)')

# Loop through every detected target to plot an 'X' on it
for i, p in enumerate(plots):
    # Get the row (Doppler) and column (Range) indices from the target
    rng_coord = p.weighted_rd[1]
    dop_coord = p.weighted_rd[0]

    target_snr_linear = rd_snr[int(round(dop_coord)), int(round(rng_coord))]

    # Convert linear SNR to dB for easier reading
    target_snr_db = 10 * np.log10(target_snr_linear)

    print(f"Target {i + 1}: Range (m)={range_axis_m[int(round(rng_coord))]}, (Velocity m/s)={velocity_axis[int(round(dop_coord))]}, SNR={target_snr_db:.2f} dB")


    # Plot a green 'X' at these coordinates
    ax[1].scatter(range_axis_m[int(round(rng_coord))], velocity_axis[int(round(dop_coord))], #Since axes are in meters and m/s, rather than converting coords, just index the bins.
                  color='green', s=25,
                  marker='x', linewidth=1, label='Target')

# Check if any targets exist to avoid errors when making the legend
handles, labels = ax[1].get_legend_handles_labels()
if handles:
    # Add a single legend entry (prevents duplicate labels)
    ax[1].legend([handles[0]], [labels[0]], loc='upper right')

# Set titles and labels for the second plot
ax[1].set_title(f"Range-Doppler Slice with {len(plots)} Detections")
ax[1].set_xlabel("Range (m)")
ax[1].set_ylabel("Velocity (m/s)")

# Adjust layout to prevent overlapping text
plt.tight_layout()
# Render the final window
plt.show()
plt.savefig(directory + fr'\cfar_results\cfar_detection_slice_{slice_number}.png')
print("Plots generated and saved.")
