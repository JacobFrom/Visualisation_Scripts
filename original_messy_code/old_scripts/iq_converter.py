import os
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

save_directory = r'G:\passive_data\old_3_sec_wall_cap\ref_stream\converted_to_iq'
os.makedirs(save_directory, exist_ok=True)
ref_file_path = os.path.join(save_directory, 'ref_iq.mat')
surv_file_path = os.path.join(save_directory, 'surv_iq.npy')
raw_arestor_ref_data = np.load(r'G:\passive_data\old_3_sec_wall_cap\ref_stream\passive_ref_output_0_3.npy') #Load in reference channel data
raw_arestor_surv_data = np.load(r'G:\passive_data\old_3_sec_wall_cap\ref_stream\passive_ref_altered.npy') # Load in surveillance channel data

ref_iq = raw_arestor_ref_data[0::2] + 1j * raw_arestor_ref_data[1::2]
surv_iq = raw_arestor_surv_data[0::2] + 1j * raw_arestor_surv_data[1::2]

mdic_ref_iq = {"ref_iq": ref_iq, "label": "Reference Channel IQ Data"}

savemat(ref_file_path, mdic_ref_iq)

print("Files saved to:", save_directory)
