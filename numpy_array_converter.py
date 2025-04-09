import numpy as np
import h5py
import os  # Import the 'os' module

npy_file = r'D:\ARESTOR_DATA\Trial_2_Day_2_tests\Run8\trial_2_test_8\passive_data\passive_surv_output_0_8.npy'
mat_file = r'D:\ARESTOR_DATA\Trial_2_Day_2_tests\Run8\trial_2_test_8\passive_data\passive_surv_output_0_8.mat'

# --- Diagnostic Step 1: Check if the .npy file exists and is readable ---
if not os.path.exists(npy_file):
    print(f"Error: .npy file not found: {npy_file}")
    exit()

try:
    data = np.load(npy_file, mmap_mode='r')
except Exception as e:
    print(f"Error loading .npy file: {e}")
    exit()

print(f"Successfully loaded .npy file. Data type: {data.dtype}, Shape: {data.shape}")

# --- Diagnostic Step 2:  Create a very small .mat file first ---
try:
    with h5py.File(mat_file, 'w') as f:
        test_dataset = f.create_dataset('test_data', data=np.array([1, 2, 3]))  # Tiny dataset
    print("Successfully created a small test .mat file.")
except Exception as e:
    print(f"Error creating test .mat file: {e}")
    exit()

# --- Now try to create the full .mat file (with chunking) ---
chunk_size = 1000000
try:
    with h5py.File(mat_file, 'w') as f: # Overwrite the test file.
        dataset = f.create_dataset('data',
                                   shape=data.shape,
                                   dtype=data.dtype,
                                   chunks=(chunk_size,),
                                   compression="gzip")
        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i:i + chunk_size]
            dataset[i:i + chunk_size] = chunk
        print(f"Successfully wrote data to {mat_file}")
        print(f"Final file size: {os.path.getsize(mat_file)} bytes")

except Exception as e:
    print(f"Error writing to .mat file: {e}")
    exit()