import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
import os

def shift(x, i):
    """
        Description:
        -----------
        Similar to np.roll function, but not circularly shift values
        Example:
        x = |x0|x1|...|xN-1|
        y = shift(x,2)
        x --> y: |0|0|x0|x1|...|xN-3|

        Parameters:
        -----------
        :param:x : input array on which the roll will be performed
        :param i : delay value [sample]

        :type i :int
        :type x: N x 1 complex numpy array
        Return values:
        --------------
        :return shifted : shifted version of x
        :rtype shifted: N x 1 complex numpy array

    """
    N = x.shape[0]
    if np.abs(i) >= N:
        return np.zeros(N)
    if i == 0:
        return x
    shifted = np.roll(x, i)
    if i < 0:
        shifted[np.mod(N + i, N):] = np.zeros(np.abs(i))
    if i > 0:
        shifted[0:i] = np.zeros(np.abs(i))
    return shifted

def ECAS(ref_ch, surv_ch, subspace_list, T, Na):
    """

    Description:
    ------------
        Performs filtering on the surveillance channel in the time domain. The ECA-S algorithm is a modification
        of the ECA-B alg. where the filtering and the coefficient estimation is made on different block sizes.
        The window used for the coefficient estimation is the symmetrical extension of the window that is used for the
        filtering. This extension can be controlled via the Na parameter. Na is the one sided window extension
        measured in signal samples.
    Parameters:
    -----------
        :param ref_ch        : Signal component to suppress (reference channel)
        :param surv_ch       : Channel to filter (surveillance channel)
        :param subspace_list : Contains the time delay and Doppler shift which
                               is used to span the signal subspace
        :param T             : Number of batches
        :param Na            : One sided window extension measured in samples

        :type ref_ch        : N by 1 complex numpy array
        :type surv_ch       : N by 1 complex numpy array
        :type subspace_list : 2D python list
        :type:T             : int
        :type Na            : int

    Return values:
    --------------
        :return filt_ch: Filtered surveillance channel
        :return w: Time domain filter coefficient vector

        :rtype N by 1 complex numpy array
        :rtype K by 1 complex numpy array

        :return None: General failure
    """
    # TODO: If the batch size is not multiple of the CPI the signals should be extend with zeros
    N = ref_ch.size
    batch_size = N // T

    # -- input check --
    if N % T:
        print("ERROR: Improper batch size, filtering is bypassed")
        print("ERROR: No output is generated")
        return None, None

    for k in subspace_list:
        if k[1] > batch_size:
            print("ERROR: Time delay is greater than the size of the batch, "
                  "this is not handled properly. Time delay is bypassed")
            print("ERROR: No output is generated")
            return None, None
        if k[1] > (batch_size + Na):
            print("ERROR: Time delay is greater than the size of the batch plus the sliding window, "
                  "this is not handled properly. Time delay is bypassed")
            print("ERROR: No output is generated")
            return None, None
    # -- calculation --
    KD = len(subspace_list)  # Signal subspace dimension

    extended_ref_ch = np.zeros(N + 2 * Na).astype(np.complex64)
    extended_ref_ch[Na:Na + N] = ref_ch[:]
    extended_surv_ch = np.zeros(N + 2 * Na).astype(np.complex64)
    extended_surv_ch[Na:Na + N] = surv_ch[:]

    filtered_surv_ch = np.zeros(N).astype(np.complex64)

    for l in np.arange(0, N, batch_size):  # Runs on batches
        surv_signal_batch = surv_ch[l: l + batch_size]  # Filtering will be performed on this block
        surv_signal_batch_ta = extended_surv_ch[l:l + batch_size + 2 * Na]  # "w" will be estimated on this block
        x = np.zeros((batch_size, KD)).astype(np.complex64) # Signal subspace for filtering
        x_ta = np.zeros((batch_size + 2 * Na, KD)).astype(np.complex64)  # Signal subspace for the estimation of the "w"

        # Generate matrices for filtering and estimation from the current batch
        subspace_vector_index = -1  # Used to index the subspace matrix (X) inside the for loops
        for k in subspace_list:  # Select current subspace vector parameter
            # print("Progress: %d %%" %(((subspace_vector_index + 2) / KD) * ((l + batch_size) / N) * 100))
            subspace_vector_index += 1

            doppler_shift_array_ta = np.exp(np.arange(l - Na, l + batch_size + Na, 1) * 1j * 2 * np.pi * k[0] / (2 * N))
            doppler_shift_array = np.exp(np.arange(l, l + batch_size, 1) * 1j * 2 * np.pi * k[0] / (2 * N))
            if not l:  # For the first batch zero values must be inserted (l=0)
                x[:, subspace_vector_index] = shift(ref_ch[l:l + batch_size], k[1])
                x_ta[:, subspace_vector_index] = shift(extended_ref_ch[l:l + batch_size + 2 * Na], k[1])
            else:
                x[:, subspace_vector_index] = ref_ch[l - k[1]:l + batch_size - k[1]]
                x_ta[:, subspace_vector_index] = extended_ref_ch[l - k[1]:l + 2 * Na + batch_size - k[1]]

            x[:, subspace_vector_index] *= doppler_shift_array  # Doppler shift
            x_ta[:, subspace_vector_index] *= doppler_shift_array_ta  # Doppler shift

        #Orthogonal subspace projection
        #Old and heavy version:

        x = np.matrix(x)  # N x KD matrix
        x_ta = np.matrix(x_ta)  # N x KD matrix
        surv_signal = np.matrix(surv_signal_batch).getT()  # column vector
        surv_signal_ta = np.matrix(surv_signal_batch_ta).getT()  # column vector
        corr_matrix = x_ta.getH() * x_ta  # KD x KD
        corr_vector = x_ta.getH() * surv_signal_ta  # KD x 1
        weight_vector = corr_matrix.getI() * corr_vector  # coefficient vector
        filtered_surv_ch[l:l + batch_size] = np.array(surv_signal - x * weight_vector)[:, 0]

        # #Attempt at Modernising and speeding up:
        #
        # # No need to convert to np.matrix anymore
        # surv_signal_ta = surv_signal_batch_ta[:, None]
        # corr_matrix = x_ta.conj().T @ x_ta  # Use @ for matrix multiplication
        # corr_vector = x_ta.conj().T @ surv_signal_ta
        # weight_vector = np.linalg.solve(corr_matrix, corr_vector)  # MUCH faster than .getI()
        # filtered_surv_ch[l:l + batch_size] = (surv_signal_batch - (x @ weight_vector)).flatten()

    return filtered_surv_ch, weight_vector


def gen_subspace_indexes(K, D):
    """
    Description:
    -----------
    Generate a signal subspace parameter list for the ECA algorithm variants.

    Parameters:
    -----------
        :param K: Number of considered time delay
        :param D: Maximum Doppler frequency value

        :type: K: int
        :type: D: int
    Return values:
    --------------
        :return: Generated subspace list
        :rtype: 2D python list
    """
    rd_index_list = []  # Used to store range-Doppler indexes on which the filtering have to performed
    # Create initial filter list
    for i in np.arange(-D, D + 1, 1):  # Doppler shifts
        for j in np.arange(0, K):  # Runs on signal delays
            rd_index_list.append([i , j])

    return rd_index_list


def mmap_ECAS(ref_ch, surv_ch, subspace_list, T, Na):
    N = ref_ch.size
    batch_size = N // T
    KD = len(subspace_list)

    # Path for M.2 Drive Offloading
    mmap_dir = r'G:\mmap'
    if not os.path.exists(mmap_dir):
        os.makedirs(mmap_dir)

    # 1. Setup Memory Mapped Files to save RAM
    ext_ref_path = os.path.join(mmap_dir, 'ext_ref.dat')
    ext_sur_path = os.path.join(mmap_dir, 'ext_sur.dat')
    filt_res_path = os.path.join(mmap_dir, 'filt_res.dat')

    extended_ref_ch = np.memmap(ext_ref_path, dtype='complex64', mode='w+', shape=(N + 2 * Na,))
    extended_surv_ch = np.memmap(ext_sur_path, dtype='complex64', mode='w+', shape=(N + 2 * Na,))

    # Copy data to M.2
    extended_ref_ch[Na:Na + N] = ref_ch[:]
    extended_surv_ch[Na:Na + N] = surv_ch[:]

    # Pre-allocate output on disk
    filtered_surv_ch = np.memmap(filt_res_path, dtype='complex64', mode='w+', shape=(N,))

    # 2. RADICAL SPEED FIX: Pre-calculate Doppler Lookup Tables
    # We calculate the "shape" of the doppler once so we don't call np.exp in the loop
    unique_dopplers = np.unique([k[0] for k in subspace_list])
    doppler_lookup = {}
    doppler_lookup_ta = {}

    t_filt_base = np.arange(batch_size)
    t_ta_base = np.arange(batch_size + 2 * Na) - Na

    for d_val in unique_dopplers:
        # Core rotation stems
        doppler_lookup[d_val] = np.exp(1j * 2 * np.pi * d_val * t_filt_base / (2 * N)).astype(np.complex64)
        doppler_lookup_ta[d_val] = np.exp(1j * 2 * np.pi * d_val * t_ta_base / (2 * N)).astype(np.complex64)

    # 3. Main Processing Loop
    for l in tqdm(np.arange(0, N, batch_size), desc="ECA-S Processing"):
        surv_signal_batch = surv_ch[l: l + batch_size]
        surv_signal_batch_ta = extended_surv_ch[l:l + batch_size + 2 * Na]

        x = np.zeros((batch_size, KD), dtype=np.complex64)
        x_ta = np.zeros((batch_size + 2 * Na, KD), dtype=np.complex64)

        for idx, k in enumerate(subspace_list):
            d_val = k[0]
            delay = k[1]

            # Slicing logic (Matches your original shift-based logic)
            if not l:
                x[:, idx] = shift(ref_ch[l:l + batch_size], delay)
                x_ta[:, idx] = shift(extended_ref_ch[l:l + batch_size + 2 * Na], delay)
            else:
                x[:, idx] = ref_ch[l - delay: l + batch_size - delay]
                x_ta[:, idx] = extended_ref_ch[l - delay: l + 2 * Na + batch_size - delay]

            # Applying Doppler via Phase Rotation (Much faster than np.exp)
            batch_phase = np.exp(1j * 2 * np.pi * d_val * l / (2 * N)).astype(np.complex64)
            x[:, idx] *= (doppler_lookup[d_val] * batch_phase)
            x_ta[:, idx] *= (doppler_lookup_ta[d_val] * batch_phase)

        # 4. Optimized Matrix Math (Mathematically identical to np.matrix)
        # Using @ and linalg.solve avoids heavy np.matrix overhead
        corr_matrix = x_ta.conj().T @ x_ta
        corr_vector = x_ta.conj().T @ surv_signal_batch_ta[:, None]

        try:
            w = np.linalg.solve(corr_matrix, corr_vector)
            # Subtract reconstructed interference
            filtered_surv_ch[l:l + batch_size] = surv_signal_batch - (x @ w).ravel()
        except np.linalg.LinAlgError:
            filtered_surv_ch[l:l + batch_size] = surv_signal_batch

    # 5. Flush and Return
    filtered_surv_ch.flush()
    result = np.array(filtered_surv_ch)

    # Cleanup file handles and temp files
    del extended_ref_ch, extended_surv_ch, filtered_surv_ch
    for f in [ext_ref_path, ext_sur_path, filt_res_path]:
        try:
            os.remove(f)
        except:
            pass

    return result, w
