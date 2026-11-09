import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------
# Config: these must match the values passed to the callback in extra_args
# ---------------------------------------------------------------------

main_directory = r"/home/uceejpw/x_band_testing_september_2026_data"
final_dir = r"/IFM_final_day"
last_dir = r"/IFM_testing_front_end"
data = fr"{main_directory}{final_dir}/pdw_only_IFM_test_11_signal_-10dB_GDB_18dB/"
file = "PDW_unformatted_channel_5.npy"

filename = os.path.join(data, file)
out_csv = fr"{data}/PDW_formatted_offline.csv"

band = 2  # 1, 2, 5 or 9
adc_sample_rate = 3.84e9
adc_decimation = 8
other_decimation = 4
# IMPORTANT: use the same value the callback received as extra_args[2].
# If your pulse widths are wrong, check this first.
effective_sample_rate = adc_sample_rate / (adc_decimation * other_decimation)

sband_cmplx_mixer_freq = 2.438e9
cband_cmplx_mixer_freq = None  # fill in if band == 5
cband_mixer_freq = None  # fill in if band == 5
testing_mixer_freq = 798e6  # fill in if band == 1
x_band_cf = 9.75e9  # fill in if band == 9

drop_zero_power = False  # the callback keeps these (power = -inf); set True to drop them


# ---------------------------------------------------------------------


def decode_signed_phase_diff_q4_12(word):
    """Q4.12 signed 16-bit value -> radians/sample.
    (Assumed to match your decode_signed_phase_diff_q4_12; swap in the real one if it differs.)"""
    word = int(word) & 0xFFFF
    if word & 0x8000:
        word -= 0x10000
    return word / (2 ** 12)


np_arr = np.load(filename)
print(f"dtype={np_arr.dtype}, length={len(np_arr)}")

# Guard reshape into 4-word PDW rows
if len(np_arr) % 4 != 0:
    remainder = len(np_arr) % 4
    print(f"Warning: length {len(np_arr)} not divisible by 4. Truncating last {remainder} value(s).")
    np_arr = np_arr[:len(np_arr) - remainder]

pdw_words = np.reshape(np_arr, (-1, 4)).astype(np.int64)
print(f"PDW rows after reshape = {len(pdw_words)}")

rows = []
k = effective_sample_rate / (2 * np.pi)

for row in pdw_words:
    pulseID = int(row[0] & 0xFFFF)
    pulseWidth_raw = int((row[0] >> 16) & 0xFFFF)
    restTime_raw = int(row[1] & 0xFFFF)
    pulseWidth = pulseWidth_raw / effective_sample_rate
    restTime = restTime_raw / effective_sample_rate

    centre_word = (row[1] >> 16) & 0xFFFF
    start_word = row[3] & 0xFFFF
    stop_word = (row[3] >> 16) & 0xFFFF

    # Reject incomplete / invalid IFM descriptors
    if pulseWidth == 0 or start_word == 0x0000:
        continue

    sampled_centre_frequency = k * decode_signed_phase_diff_q4_12(centre_word)
    startFrequency = k * decode_signed_phase_diff_q4_12(start_word)
    stopFrequency = k * decode_signed_phase_diff_q4_12(stop_word)

    if band == 2:
        unwrappedCentreFrequency = sband_cmplx_mixer_freq - sampled_centre_frequency
        startFrequency = sband_cmplx_mixer_freq - startFrequency
        stopFrequency = sband_cmplx_mixer_freq - stopFrequency
    elif band == 5:
        unwrappedCentreFrequency = cband_mixer_freq + (sampled_centre_frequency - cband_cmplx_mixer_freq)
        startFrequency = cband_mixer_freq + (startFrequency - cband_cmplx_mixer_freq)
        stopFrequency = cband_mixer_freq + (stopFrequency - cband_cmplx_mixer_freq)
    elif band == 1:
        unwrappedCentreFrequency = adc_sample_rate - (sampled_centre_frequency + testing_mixer_freq)
        startFrequency = testing_mixer_freq + startFrequency
        stopFrequency = testing_mixer_freq + stopFrequency
    elif band == 9:
        unwrappedCentreFrequency = x_band_cf + sampled_centre_frequency
        startFrequency = x_band_cf + startFrequency
        stopFrequency = x_band_cf + stopFrequency
    else:
        raise ValueError(f"Unsupported band value: {band}")

    if row[2] > 0:
        averagePower = 10 * np.log10(row[2])
    else:
        averagePower = -np.inf
        if drop_zero_power:
            continue

    if np.isfinite(startFrequency) and np.isfinite(stopFrequency):
        calculatedCentreFrequency = (startFrequency + stopFrequency) / 2
    else:
        calculatedCentreFrequency = np.nan

    rows.append({
        'PulseID': pulseID,
        'PulseWidth': pulseWidth,
        'RestTime': restTime,
        'CentreFrequency': calculatedCentreFrequency,
        'AveragePower': averagePower,
        'StartFreq': startFrequency,
        'StopFreq': stopFrequency,
        'Bandwidth': abs(stopFrequency - startFrequency),
    })

PDW = pd.DataFrame(rows)
PDW.to_csv(out_csv)

print(f"\n{len(PDW)} valid pulses decoded -> {out_csv}")
print(PDW.head(10).to_string())

