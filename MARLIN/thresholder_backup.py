import numpy as np
import os

iq_directory = r"/home/uceejpw//MARLIN_OCT_2026/day_1/IQ_T9_1/Raw_IQ_Capture_1"
channel_of_interest = 3

WINDOW = 5                   # moving-sum length (your 5-sample integration)
FRACTION = 0.55              # start threshold: this far from the noise level towards the peak (as before)
K_SIGMA = 7                  # ...but never closer than K noise standard deviations above the noise level.
                             # 7 gives ~0 noise-only triggers; 6 gives ~30/s.
END_SIGMA = 2                # end threshold this many noise sigmas BELOW the noise level: stops weak
                             # pulses splitting into several PDWs (3 starts merging separate pulses)
FS = 3.84e9 / 32             # IQ sample rate
EXPECTED_PULSE_RATE = 500  # e.g. 1024 for your chirp: prints how many pulses to expect

iq_raw_data = np.load(os.path.join(iq_directory, f'channel_{channel_of_interest}_iq.npy'))
i_data = iq_raw_data[0::2].astype(np.float32, copy=False)
q_data = iq_raw_data[1::2].astype(np.float32, copy=False)
mag = np.hypot(i_data, q_data)
integrated_mag = np.convolve(mag, np.ones(WINDOW, np.float32), mode='valid')

# Noise level and spread from the median and median absolute deviation: unlike the mean and
# standard deviation, pulses in the capture barely move them.
noise_level = float(np.median(integrated_mag))
noise_sigma = 1.4826 * float(np.median(np.abs(integrated_mag - noise_level)))
integrated_max = float(integrated_mag.max())

fraction_threshold = noise_level + FRACTION * (integrated_max - noise_level)
noise_floor_threshold = noise_level + K_SIGMA * noise_sigma

current_start_capture_threshold = int(round(max(fraction_threshold, noise_floor_threshold)))
current_end_capture_threshold = int(round(noise_level - END_SIGMA * noise_sigma))

print(f"noise level (median) = {noise_level:.1f}, noise sigma = {noise_sigma:.1f}")
print(f"integrated_max = {integrated_max:.1f}  ({(integrated_max - noise_level) / noise_sigma:.1f} sigma above noise)")
print(f"fraction rule = {fraction_threshold:.0f}, noise floor ({K_SIGMA} sigma) = {noise_floor_threshold:.0f}")
if noise_floor_threshold > fraction_threshold:
    print("  -> weak signal: the noise floor sets the start threshold (the fraction rule would sit in the noise)")
if integrated_max < noise_floor_threshold:
    print("  -> WARNING: nothing in this capture rises above the noise floor; no pulses will be detected")

# Check: how many triggers these thresholds give on this capture (start above `start`,
# end when the integrated magnitude falls below `end`, as the IFM does)
above = (integrated_mag >= current_end_capture_threshold).view(np.int8)
starts = np.flatnonzero(np.diff(above, prepend=np.int8(0)) == 1)
n_trig = int((np.maximum.reduceat(integrated_mag, starts) > current_start_capture_threshold).sum()) if starts.size else 0
duration = integrated_mag.size / FS
print(f"triggers in this capture: {n_trig} in {duration:.2f} s ({n_trig / duration:.0f}/s)"
      + (f", expected ~{EXPECTED_PULSE_RATE * duration:.0f} pulses" if EXPECTED_PULSE_RATE else ""))

print(f"current_start_capture_threshold = {current_start_capture_threshold}")
print(f"current_end_capture_threshold = {current_end_capture_threshold}")