import numpy as np

# ---- Config: fill these in ----
filename = r'/home/uceejpw/x_band_testing_september_2026_data/IFM_testing_day_1/test_3/PDW_unformatted_2.npy'

band = 2
sband_cmplx_mixer_freq = 798e6

adc_sample_rate = 3.84e9
adc_decimation = 8
other_decimation = 4
effective_sample_rate = adc_sample_rate / (adc_decimation * other_decimation)
# --------------------------------

if band == 2 and sband_cmplx_mixer_freq is None:
    raise ValueError("sband_cmplx_mixer_freq must be set for band == 2")

np_arr = np.load(filename)

# Pack 4 Int32's together for each PDW
new_array = np.reshape(np_arr, (-1, 4))

# Loop through the packed PDW array and extract parameters
for x in range(len(new_array)):

    pulseID = np.int16(new_array[x][0] & 0xFFFF)
    pulseWidth = np.int16((new_array[x][0] >> 16) & 0xFFFF) * (1 / effective_sample_rate)

    if pulseWidth != 0:

        restTime = np.int16(new_array[x][1] & 0xFFFF) * (1 / effective_sample_rate)

        average_phase = np.int16((new_array[x][1] >> 16) & 0xFFFF)
        if average_phase & 0x8000:
            average_phase -= 0x10000  # Sign-extend to negative number
        average_phase = average_phase / (2 ** 12)
        sampledCentreFrequency = (1 / (2 * np.pi)) * effective_sample_rate * average_phase

        start_phase = (new_array[x][3] & 0xFFFF)
        if start_phase & 0x8000:
            start_phase -= 0x10000  # Sign-extend to negative number
        start_phase = start_phase / (2 ** 12)
        startFrequency = (1 / (2 * np.pi)) * effective_sample_rate * start_phase

        stop_phase = ((new_array[x][3] >> 16) & 0xFFFF)
        if stop_phase & 0x8000:
            stop_phase -= 0x10000  # Sign-extend to negative number
        stop_phase = stop_phase / (2 ** 12)
        stopFrequency = (1 / (2 * np.pi)) * effective_sample_rate * stop_phase

        # Band 2, 1st Nyquist zone: just add the mixer frequency, no fold-over
        unwrappedCentreFrequency = sampledCentreFrequency + sband_cmplx_mixer_freq
        startFrequency = startFrequency + sband_cmplx_mixer_freq
        stopFrequency = stopFrequency + sband_cmplx_mixer_freq

        averagePower = 10 * np.log10(new_array[x][2])

        print("\n")
        print("PulseID: " + str(pulseID))
        print("PulseWidth: " + str(pulseWidth))
        print("RestTime: " + str(restTime))
        print("CentreFrequency: " + str(unwrappedCentreFrequency))
        print("AveragePower: " + str(averagePower))
        print("StartFreq: " + str(startFrequency))
        print("StopFreq: " + str(stopFrequency))