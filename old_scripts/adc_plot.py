import numpy as np
import matplotlib
import matplotlib.pyplot as plt

arr = np.load((r'/home/uceejpw/Trial_2_Field_tests/passive_surv_output_0_7.npy'), mmap_mode='r')

I = arr[0:480000:2]
plt.plot(I)
#Q = arr[1:480000:2]
#plt.plot(Q)
plt.show()