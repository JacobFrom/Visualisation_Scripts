import numpy as np
import matplotlib
import matplotlib.pyplot as plt

arr = np.load((r'/home/uceejpw/Trial_2_lab_tests/DVB-T_tests/test2_3s_ref_capture/passive_ref_output_0_3.npy'), mmap_mode='r')

I = arr[0:480000:2]
plt.plot(I)
#Q = arr[1:480000:2]
#plt.plot(Q)
plt.show()