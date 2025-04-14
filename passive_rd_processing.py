import os
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal
import matplotlib.animation as animation
import time

experiment_codename = 'run_7'
directory = fr'G:\trial_3_data\{experiment_codename}\passive_data_test_1'


os.makedirs(fr'D:\ARESTOR_DATA\trial_3_rd_videos\{experiment_codename}', exist_ok=True)

ref_data = np.load(directory + r'\passive_ref_output.npy')

surv_data = np.load(directory + r'\passive_surv_output.npy')

print(ref_data.shape)
