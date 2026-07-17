import os
import numpy as np
import time
from original_messy_code.broken_scripts import passive_processing_functions as pf
import multiprocessing
from multiprocessing import shared_memory

# --- Processing Parameters ---
C = 3e8  # Speed of light in m/s
float_dtype = np.float32
complex_dtype = np.complex64
#multiprocessing attempt
#-worker initializer function-
def worker_init(shm_ref_name_g, shm_surv_name_g, ref_shape_g, surv_shape_g, dtype_g):

    global worker_ref_iq, worker_surv_iq # Make arrays global *within this worker*

    # Access the existing shared memory
    existing_shm_ref = shared_memory.SharedMemory(name=shm_ref_name_g)
    existing_shm_surv = shared_memory.SharedMemory(name=shm_surv_name_g)

    # Create NumPy array views pointing to the shared memory buffers
    worker_ref_iq = np.ndarray(ref_shape_g, dtype=dtype_g, buffer=existing_shm_ref.buf)
    worker_surv_iq = np.ndarray(surv_shape_g, dtype=dtype_g, buffer=existing_shm_surv.buf)
    # Note: Keep existing_shm_ref/surv open while worker uses the arrays

# --- Worker Task Function (Simplified - No Filtering) ---
# Processes ONE CPI, using shared memory arrays directly
# --- Worker Task Function (Modified with Prints) ---
def process_one_cpi_shared_no_filter(i, params):
    # --- ADD PRINT ---
    print(f"[Worker {os.getpid()}] START processing CPI {i}", flush=True)
    try: # Add try block for worker errors
        # Unpack parameters
        cpi_stride = params['cpi_stride']
        samples_per_cpi = params['samples_per_cpi']
        n_fft = params['n_fft']
        max_delay_samples = params['max_delay_samples']

        global worker_ref_iq, worker_surv_iq

        start_index = i * cpi_stride
        end_index = start_index + samples_per_cpi
        if end_index > worker_ref_iq.shape[0]: end_index = worker_ref_iq.shape[0]

        ref_cpi = worker_ref_iq[start_index:end_index]
        surv_cpi = worker_surv_iq[start_index:end_index]

        current_segment_len = ref_cpi.shape[0]
        if current_segment_len < 2:
            print(f"[Worker {os.getpid()}] FINISH processing CPI {i} (segment too short)", flush=True)
            return np.zeros(max_delay_samples, dtype=complex_dtype)

        current_n_fft = n_fft if current_segment_len == samples_per_cpi else current_segment_len

        # --- Perform cross-correlation ---
        # print(f"[Worker {os.getpid()}] CPI {i} - Starting FFTs...", flush=True) # Optional finer print
        fft_surv = np.fft.fft(surv_cpi, n=current_n_fft)
        fft_ref_conj = np.conj(np.fft.fft(ref_cpi, n=current_n_fft))
        # print(f"[Worker {os.getpid()}] CPI {i} - Starting Multiply...", flush=True) # Optional finer print
        correlation_freq = fft_surv * fft_ref_conj
        # print(f"[Worker {os.getpid()}] CPI {i} - Starting IFFT...", flush=True) # Optional finer print
        correlation_time = np.fft.ifft(correlation_freq)
        # print(f"[Worker {os.getpid()}] CPI {i} - FFTs/IFFT Done.", flush=True) # Optional finer print


        result_row = np.zeros(max_delay_samples, dtype=complex_dtype)
        num_lags_to_extract = min(max_delay_samples, len(correlation_time))
        result_row[:num_lags_to_extract] = correlation_time[0 : num_lags_to_extract]

        # --- ADD PRINT ---
        print(f"[Worker {os.getpid()}] FINISH processing CPI {i}", flush=True)
        return result_row
    except Exception as worker_e:
         print(f"[Worker {os.getpid()}] !!!!! EXCEPTION in CPI {i}: {type(worker_e).__name__} - {worker_e}", flush=True)
         # Return zeros or raise error depending on how you want pool to handle it
         return np.zeros(max_delay_samples, dtype=complex_dtype)



# --- Main Execution Context ---
if __name__ == "__main__":
    # ================================================================
    # --- Ensure all setup code is ABOVE this point ---
    # Including loading data into initial ref_iq, surv_iq
    # AND ensuring they are dtype=complex_dtype before sharing:
    # ref_iq = ref_iq.astype(complex_dtype)
    # surv_iq = surv_iq.astype(complex_dtype)
    # Also calculate parameters like:
    # num_cpis, max_delay_samples, samples_per_cpi, cpi_stride
    # Also, choose n_fft (e.g., samples_per_cpi or padded power of 2)
    # ================================================================
    # --- Configuration ---
    experiment_codename = 'run_7'

    capture_duration = 30
    centre_freq = 498e6  # Centre freq of DVB-T capture DVB-T

    decimator_ip = 32  # Choose between 4x or 32x
    decimator_tile = 8
    wavelength = C / centre_freq  # Wavelength in meters

    base_data_directory = r'G:\trial_3_data'
    data_subdirectory = 'passive_data_test_1'
    output_video_directory = fr'D:\ARESTOR_DATA\trial_3_rd_videos\{experiment_codename}\passive_test_1'

    input_data_dir = os.path.join(base_data_directory, experiment_codename, data_subdirectory)

    ref_filename = 'passive_ref_output.npy'
    surv_filename = 'passive_surv_output.npy'

    ref_path = os.path.join(input_data_dir, ref_filename)
    surv_path = os.path.join(input_data_dir, surv_filename)

    os.makedirs(output_video_directory, exist_ok=True)  # Ensure output dir exists

    ref_data_raw = np.load(ref_path)
    surv_data_raw = np.load(surv_path)

    # --- De-interleave and Convert to Complex (Minimal Pre-allocation) ---
    print("De-interleaving and creating complex signals (using complex64 efficiently)...")


    # Reference Channel
    if ref_data_raw.shape[0] % 2 != 0:  # Ensure even length for pairing
        ref_data_raw = ref_data_raw[:-1]  # Trim last sample if odd
    num_complex_samples_ref = ref_data_raw.shape[0] // 2  # Calculate number of complex samples
    ref_iq = np.empty(num_complex_samples_ref, dtype=complex_dtype)  # 1. Pre-allocate final array
    ref_iq.real = ref_data_raw[0::2].astype(float_dtype)  # 2. Fill real part (I)
    ref_iq.imag = ref_data_raw[1::2].astype(float_dtype)  # 3. Fill imaginary part (Q)
    del ref_data_raw  # Delete raw int16 data to free memory

    # Surveillance Channel
    if surv_data_raw.shape[0] % 2 != 0:  # Ensure even length for pairing
        surv_data_raw = surv_data_raw[:-1]  # Trim last sample if odd
    num_complex_samples_surv = surv_data_raw.shape[0] // 2  # Calculate number of complex samples
    surv_iq = np.empty(num_complex_samples_surv, dtype=complex_dtype)  # 1. Pre-allocate final array
    surv_iq.real = surv_data_raw[0::2].astype(float_dtype)  # 2. Fill real part (I)
    surv_iq.imag = surv_data_raw[1::2].astype(float_dtype)  # 3. Fill imaginary part (Q)
    del surv_data_raw  # Delete raw int16 data to free memory
    # --- End De-interleave Section ---

    # --- Continue with the rest of your script ---
    print(f"Shape of complex reference data (ref_iq): {ref_iq.shape}")
    print(f"Shape of complex surveillance data (surv_iq): {surv_iq.shape}")

    # Calculate Effective Sample Rate

    Fs_original = 3.84e9  # Original sample rate
    decimation_factor = decimator_ip * decimator_tile  # Decimation factor from decimator ip block multiplied by tile decimation

    # Calculate Effective Sample Rate
    Fs_eff = Fs_original / decimation_factor
    print(f"Effective Sample Rate (Fs_eff): {Fs_eff / 1e6:.2f} MHz ({Fs_eff} Hz)")

    # CPI settings
    cpi_duration_sec = 0.5  # Coherent Processing Interval in seconds
    cpi_overlap = 0.98  # Overlap between consecutive CPIs (0 to 1)

    # Range/Delay settings
    max_range_interest_meters = 280  # Max bistatic range of interest (20 km example)
    # Convert max range to max delay. NOTE: For passive bistatic radar, Range = c * delay.
    # This 'Range' is the path length difference: (Tx -> Target -> Rx) - (Tx -> Rx)
    max_delay_sec = max_range_interest_meters / C

    print(f"Maximum delay calculated based on range of interest: {max_delay_sec * 1e6:.2f} us")

    # Doppler settings

    samples_per_cpi = int(np.ceil(cpi_duration_sec * Fs_eff))
    print(f"Samples per CPI: {samples_per_cpi}")
    cpi_stride = int(samples_per_cpi * (1 - cpi_overlap))

    max_delay_samples = int(np.ceil(max_delay_sec * Fs_eff))

    # calculate total number of CPIs that can be processed
    total_samples = ref_iq.shape[0]

    num_cpis = int(np.floor((total_samples - samples_per_cpi) / cpi_stride)) + 1
    print(f"Total number of CPIs that can be processed: {num_cpis}")

    slow_time_step = cpi_stride / Fs_eff

    # Rows = CPIs, Columns = Delay Bins
    correlation_matrix = np.zeros((num_cpis, max_delay_samples), dtype=np.complex64)

    n_fft = int(2 ** np.ceil(np.log2(samples_per_cpi)))  # Pad to next power of 2
    # --- Shared Memory Setup ---
    shm_ref = None  # Initialize to None
    shm_surv = None
    try:
        print("Setting up shared memory...")
        # --- Try creating shm_ref ---
        try:
            shm_ref = shared_memory.SharedMemory(create=True, size=ref_iq.nbytes)
            print(f"Allocated shared memory: ref='{shm_ref.name}'")
        except FileExistsError:
            print(
                "ERROR: Shared memory block for ref ALREADY EXISTS. Clean up needed (try restarting IDE/PC again or manual cleanup).")
            exit()  # Stop execution
        except Exception as e_ref_create:
            print(f"ERROR creating ref shared memory: {type(e_ref_create).__name__} - {e_ref_create}")
            exit()  # Stop execution

        # --- Try creating shm_surv (only if shm_ref succeeded) ---
        try:
            shm_surv = shared_memory.SharedMemory(create=True, size=surv_iq.nbytes)
            print(f"Allocated shared memory: surv='{shm_surv.name}'")
        except FileExistsError:
            print("ERROR: Shared memory block for surv ALREADY EXISTS. Clean up needed.")
            exit()  # Stop execution
        except Exception as e_surv_create:
            print(f"ERROR creating surv shared memory: {type(e_surv_create).__name__} - {e_surv_create}")
            exit()  # Stop execution

        # --- If we got here, BOTH shm_ref and shm_surv should be valid objects ---
        print("Shared memory objects created successfully.")

        # Create NumPy array views mapped to these shared blocks
        ref_iq_shm = np.ndarray(ref_iq.shape, dtype=ref_iq.dtype, buffer=shm_ref.buf)
        surv_iq_shm = np.ndarray(surv_iq.shape, dtype=surv_iq.dtype, buffer=shm_surv.buf)

        print("Copying data to shared memory...")
        ref_iq_shm[:] = ref_iq[:]
        surv_iq_shm[:] = surv_iq[:]
        print("Data copied.")

        # --- Multiprocessing Setup ---
        print("\nStarting cross-correlation using multiprocessing (NO filtering)...")
        start_loop_time = time.perf_counter()

        num_processes = 4  # Keep low for testing
        print(f"Using {num_processes} worker processes.")

        n_fft = samples_per_cpi

        worker_params = {  # Ensure all needed params are here
            'cpi_stride': cpi_stride,
            'samples_per_cpi': samples_per_cpi,
            'n_fft': n_fft,
            'max_delay_samples': max_delay_samples
        }

        # Create init_args ONLY if shm objects are valid
        init_args = (shm_ref.name, shm_surv.name, ref_iq.shape, surv_iq.shape, complex_dtype)
        print(f"Initializer Args prepared: {init_args[0]}, {init_args[1]}, ...")  # Verify names look okay

        map_args = [(i, worker_params) for i in range(num_cpis)]

        # --- Create and run the processing pool ---
        print("Creating Pool...")
        with multiprocessing.Pool(processes=num_processes, initializer=worker_init, initargs=init_args) as pool:
            print(f"\n>>> Calling pool.starmap for {num_cpis} tasks...", flush=True)
            list_of_results = pool.starmap(process_one_cpi_shared_no_filter, map_args)
            print(f">>> pool.starmap finished. Got {len(list_of_results)} results.", flush=True) # Check if it gets here
        # ... assemble matrix ...
        # --- Assemble the final matrix ---
        # ... (stack results) ...
        correlation_matrix = np.stack(list_of_results, axis=0)

        end_loop_time = time.perf_counter()
        # ... (print time, shape) ...

    except Exception as e:
        # This catches errors AFTER shared memory creation but before pool finishes maybe?
        print(f"An error occurred during Pool execution or result assembly: {type(e).__name__} - {e}")
        import traceback

        traceback.print_exc()  # Print detailed traceback

    finally:
        # --- Crucial Cleanup ---
        print("Cleaning up shared memory...")
        if shm_ref is not None:
            try:
                shm_ref.close()
                shm_ref.unlink()
            except Exception as e_clean_ref:
                print(f"Error cleaning up ref shared memory: {e_clean_ref}")
        if shm_surv is not None:
            try:
                shm_surv.close()
                shm_surv.unlink()
            except Exception as e_clean_surv:
                print(f"Error cleaning up surv shared memory: {e_clean_surv}")
        print("Shared memory cleaned up.")
    # --- Range-Doppler Processing ---
    N_doppler_choice = 64
    rd_overlap = 0.95
    rd_cpi = N_doppler_choice * slow_time_step
    # Generate range doppler slices


    rd_slices_list, velocity_axis = pf.get_passive_rd_slices(correlation_matrix, slow_time_step, wavelength, rd_cpi, rd_overlap)

    print(f"Number of RD slices: {len(rd_slices_list)}")

    # --- Plotting ---
    delay_axis_sec = np.arange(max_delay_samples) / Fs_eff
    range_axis_m = delay_axis_sec * C #No division by 2 for bistatic since it is time delay of arrival calculated with respect to Tx and Rx

    print(f"Range Axis (m) = {range_axis_m}")
    print(f"Velocity Axis (m/s) = {velocity_axis[0]:.2f} m/s to {velocity_axis[-1]:.2f} m/s")
