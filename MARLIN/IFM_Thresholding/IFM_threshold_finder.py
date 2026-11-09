import numpy as np
import os


test_number = 1
iq_directory = r"/home/uceejpw//MARLIN_OCT_2026/day_3/IQ_OS_SS_T1_2_GDB_0dB/Raw_IQ_Capture_1"

channel_of_interest = 5

iq_raw_data = np.load(os.path.join(iq_directory, f'channel_{channel_of_interest}_iq.npy'))

i_data = iq_raw_data[0::2].astype(np.float32, copy=False)
q_data = iq_raw_data[1::2].astype(np.float32, copy=False)
mag = np.hypot(i_data, q_data)
integrated_mag = (mag[:-4] +
                  mag[1:-3] +
                  mag[2:-2] +
                  mag[3:-1] +
                  mag[4:]
                  )
integrated_mean = integrated_mag.mean()
integrated_max = integrated_mag.max()

print(f"integrated_mean = {integrated_mean}")
print(f"integrated_max = {integrated_max}")


current_start_capture_threshold = int(round(
                      integrated_mean + 0.55 * (integrated_max - integrated_mean)
                    ))

current_end_capture_threshold = int(round(integrated_mean))

print(f"current_start_capture_threshold = {current_start_capture_threshold}")
print(f"current_end_capture_threshold = {current_end_capture_threshold}")