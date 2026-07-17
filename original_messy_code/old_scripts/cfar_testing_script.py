import os
import time
import numpy as np
from CaCFAR import CA_CFAR as CFAR
import DetProcessing as dp
import DSP_functions as DSP
import matplotlib.pyplot as plt

directory = fr'G:\ARESTOR_DATA\Ulm_trial\tests\day_2\ulm_day_2_heli\passive_data_test_1\CFAR_testing'
rd_matrices = np.load(directory + r'\rd_slice_matrix_2.npy')

window_params = [8, 8, 2, 2]  # [window width, window height, guard cell width, guard cell height]
threshold_dB = 15  # Threshold level in dB
data_shape = rd_matrices[0].shape
max_range_m = data_shape[1] * 20  # Assuming 1 range cell = 20 meters (True for single channel DVB-T passive radar)

CFAR_hit_matrix_exists_flag = True

# ==========================================
# CFAR PROCESSING SECTION
# ==========================================
if not CFAR_hit_matrix_exists_flag:
    hit_results = []
    snr_results = []
    print(f"Processing {len(rd_matrices)} slices...")
    start = time.time()

    ca_cfar = CFAR(window_params, threshold_dB, data_shape)

    for i, rd_matrix in enumerate(rd_matrices):
        # Perform CFAR on the current matrix
        hit_matrix, rd_snr = ca_cfar(rd_matrix)

        # Populate the results lists
        hit_results.append(hit_matrix)
        snr_results.append(rd_snr)

    end = time.time()

    print(f"CFAR processing time Complete. Time taken: {end - start} seconds. Saving results...")
    hit_save = directory + r'\cfar_hit_matrix.npy'
    snr_save = directory + r'\cfar_sinr_matrix.npy'
    np.save(hit_save, hit_results)
    np.save(snr_save, snr_results)
    print(f"CFAR processing complete. Hit matrix and SNR matrix saved in {directory}.")
else:
    print("Loading pre-computed CFAR results from disk...")
    hit_results = np.load(directory + r'\cfar_hit_matrix.npy')
    snr_results = np.load(directory + r'\cfar_sinr_matrix.npy')
    print("CFAR results loaded.")

# ==========================================
# VISUALIZATION SECTION
# ==========================================
#Choosing a single slice for visualization

i = 120 # Index of the slice to visualize

hit_matrix = hit_results[i]
rd_snr = snr_results[i]
rd_matrix = rd_matrices[i]

print("Generating plots...")
range_axis_m = np.linspace(0, max_range_m, int(data_shape[1]))
dynamic_range = -60  # dB
pulse_interval = 1/int(data_shape[0]) # Assuming PRF = number of Doppler cells
c = 3e8  # Speed of light in m/s
centre_freq = 570e6  # Centre frequency in Hz
wavelength = c / centre_freq  # Wavelength in meters
# configure the axis for the range-Doppler slices
slice_doppler_axis_unshifted = np.fft.fftfreq(data_shape[0], d=pulse_interval)  # This calculates doppler axis in Hz
slice_doppler_axis = np.fft.fftshift(slice_doppler_axis_unshifted)  # Shift to center zero frequency
velocity_axis = (wavelength * slice_doppler_axis) / 2  # Convert to velocity in m/s
# 1. Extract "Plots" (Clusters) from the raw hits
# This groups adjacent hit pixels into single targets and calculates their centers
# d_max=2 means hits within 2 cells of each other are grouped together
plots = dp.plot_extractor(hit_matrix, rd_snr, d_max=2)

# Optional: Filter out clutter (e.g., static objects at 0 Doppler or very close range)
plots = dp.plot_filter(plots, range_gate=5, Doppler_gate=2, rd_size=data_shape)

print(f"Extracted {len(plots)} targets from the raw hits.")

# 2. Setup the Figure
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# --- Left Plot: Raw CFAR Hit Matrix ---
# This shows exactly which cells passed the threshold
ax[0].imshow(hit_matrix, aspect='auto', interpolation='nearest', origin='lower',
             extent=[range_axis_m[0], range_axis_m[-1], velocity_axis[0], velocity_axis[-1]],
             cmap='gray')
ax[0].set_title("Raw CFAR Hit Matrix (Boolean)")
ax[0].set_xlabel("Range Cell (Index)")
ax[0].set_ylabel("Doppler Cell (Index)")

# --- Right Plot: Range-Doppler Map with Target Overlays ---
# Convert complex RD matrix to dB for better visualization

rd_db = (20 * np.log10(np.abs(rd_matrix) / np.max(np.abs(rd_matrix))))
#rd_db = 10 * np.log10(np.abs(rd_matrix) ** 2 + 1e-10)

# Plot the background intensity
im = ax[1].imshow(rd_db, aspect='auto', interpolation='nearest', origin='lower',
                  vmin = dynamic_range, vmax = 0,
                  cmap='viridis')
plt.colorbar(im, ax=ax[1], label='Power (dB)')

# Overlay the Extracted Targets (Centroids)
# The Plot class stores position in .weighted_rd as [Doppler, Range]
for p in plots:
    # Extract coordinates (Note: weighted_rd is [Row/Doppler, Col/Range])
    rng_coord = p.weighted_rd[1]
    dop_coord = p.weighted_rd[0]

    # Scatter plot: x=Range, y=Doppler
    ax[1].scatter(rng_coord, dop_coord, color='red', s=100, marker='x', linewidth=1, label='Target')

# Handle legend (only add one label to avoid clutter)
handles, labels = ax[1].get_legend_handles_labels()
if handles:
    ax[1].legend([handles[0]], [labels[0]], loc='upper right')

ax[1].set_title(f"Range-Doppler Map with {len(plots)} Detections")
ax[1].set_xlabel("Range Cell (Index)")
ax[1].set_ylabel("Doppler Cell (Index)")

plt.tight_layout()
plt.show()