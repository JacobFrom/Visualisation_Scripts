import os
import numpy as np
import scipy as sp
import scipy.fft as fft
import matplotlib.pyplot as plt

# ==========================================================================
# Paper-sized range-Doppler slice plot
#
# Plots ONE slice from the saved RD slice matrix (rows = Doppler, columns =
# range, Doppler already fftshifted, 20 m per range bin), at the same size and
# font as the micro-Doppler figures from md_spectrogram_analysis.py.
#
# Use the SAME slice_under_test / num_slices / range_bin as the micro-Doppler
# run and the script picks the RD slice whose CPI is centred on the middle of
# the micro-Doppler record, so the two figures show the same moment.
# ==========================================================================

# ==========================================
# 1. Configuration
# ==========================================
directory = r'G:\ARESTOR_DATA\SIREN\east_fort_day_3\east_fort_capture_10\passive_data_test_1'
rd_slice_matrix_path = os.path.join(directory, 'rd_slices', '4000_rd_slice_matrix.npy')
save_directory = os.path.join(directory, 'claude_md_figures')

target_tag = "h125"          # short name used in the saved file name

# Same values as the micro-Doppler run for this target
centre_freq = 518e6          # CHECK: carrier of the DVB-T channel used for this capture
prf = 2048
range_bin = 19
slice_under_test = 179       # first slice of the micro-Doppler record
num_slices = 10              # slices in the micro-Doppler record
rd_slice_to_plot = None      # None = the slice centred on the micro-Doppler record. Or give a slice number
cpi_length = 1.0             # s
cpi_overlap = 0.9
range_bin_m = 20             # bistatic range per bin

# Plot settings
y_axis = 'velocity'          # 'velocity' (m/s, = lambda f / 2 as in DSP_functions) or 'doppler' (Hz)
range_plot_limits_m = (0, 2000)   # None for the full range axis
y_plot_limit = 60            # +/- this many m/s (or Hz if y_axis = 'doppler'). None for the full axis
dynamic_range_db = 40        # colour scale: 0 dB = strongest cell, bottom = -dynamic_range_db
mark_target = False          # True draws a small circle around the target cell (range_bin, body Doppler)
target_velocity = None       # m/s, only used for the marker. None = strongest cell in range_bin away from 0 Hz

# Same paper size and font as the micro-Doppler figures
fig_width_in = 1.72
fig_height_in = 1.45
font_size = 7
show_colorbar = False

os.makedirs(save_directory, exist_ok=True)

# ==========================================
# 2. Load the slice and build the axes
# ==========================================
rd_slice_list = np.load(rd_slice_matrix_path, mmap_mode='r')
print(f"RD Slice List Shape: {rd_slice_list.shape}")

cpi_stride = int(np.round(cpi_length * prf * (1 - cpi_overlap)))      # 205
if rd_slice_to_plot is None:
    # The micro-Doppler record runs from the middle of slice_under_test to the middle of the
    # last slice, so the slice centred on the record is half way between them
    rd_slice_to_plot = slice_under_test + (num_slices - 1) // 2
rd_slice = np.asarray(rd_slice_list[rd_slice_to_plot])
n_doppler, n_range = rd_slice.shape

wavelength = sp.constants.c / centre_freq
doppler_axis = fft.fftshift(fft.fftfreq(n_doppler, d=1 / prf))      # Hz, same as DSP_functions
velocity_axis = wavelength * doppler_axis / 2                         # m/s, same as DSP_functions
range_axis = np.arange(n_range) * range_bin_m                         # m
y_values = velocity_axis if y_axis == 'velocity' else doppler_axis
y_label = 'Velocity (m/s)' if y_axis == 'velocity' else 'Doppler (Hz)'

centre_time_s = (rd_slice_to_plot * cpi_stride) / prf + cpi_length / 2
print(f"Plotting RD slice {rd_slice_to_plot}, CPI centred {centre_time_s:.2f} s into the capture")

# 0 dB = strongest cell, as in your other RD plots
rd_db = 20 * np.log10(np.abs(rd_slice) / np.max(np.abs(rd_slice)) + 1e-12)

# ==========================================
# 3. Paper-sized figure
# ==========================================
plt.rcParams.update({'font.size': font_size, 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
half_bin_y = (y_values[1] - y_values[0]) / 2
im = ax.imshow(rd_db, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis',
               extent=[range_axis[0] - range_bin_m / 2, range_axis[-1] + range_bin_m / 2,
                       y_values[0] - half_bin_y, y_values[-1] + half_bin_y],
               vmin=-dynamic_range_db, vmax=0, rasterized=True)

if mark_target:
    if target_velocity is None:
        column = np.abs(rd_slice[:, range_bin])
        column[np.abs(doppler_axis) < 2] = 0        # ignore the zero-Doppler clutter
        target_doppler = doppler_axis[np.argmax(column)]
    else:
        target_doppler = 2 * target_velocity / wavelength
    target_y = wavelength * target_doppler / 2 if y_axis == 'velocity' else target_doppler
    ax.plot(range_bin * range_bin_m, target_y, 'o', ms=6, mfc='none', mec='r', mew=0.6)

ax.set_xlabel('Bistatic range (m)', labelpad=1)
ax.set_ylabel(y_label, labelpad=1)
ax.tick_params(length=2, pad=1)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
if range_plot_limits_m is not None:
    ax.set_xlim(*range_plot_limits_m)
if y_plot_limit is not None:
    ax.set_ylim(-y_plot_limit, y_plot_limit)
if show_colorbar:
    fig.colorbar(im, ax=ax, label='Power (dB)')

name = f"{target_tag}_slice_{rd_slice_to_plot}_rd"
fig.savefig(os.path.join(save_directory, name + '.pdf'), bbox_inches='tight', pad_inches=0.01)
fig.savefig(os.path.join(save_directory, name + '.png'), dpi=600, bbox_inches='tight', pad_inches=0.01)
print(f"Saved {name}.pdf and .png in {save_directory}")
plt.show()
