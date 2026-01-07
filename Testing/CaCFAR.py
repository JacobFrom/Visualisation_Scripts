import numpy as np
from scipy import signal


class CA_CFAR():
    """
    Description:
    ------------
        Cell Averaging - Constant False Alarm Rate algorithm

        Performs an automatic detection on the input range-Doppler matrix with an adaptive thresholding.
        The threshold level is determined for each cell in the range-Doppler map with the estimation
        of the power level of its surrounding noise. The average power of the noise is estimated on a
        rectangular window, that is defined around the CUT (Cell Under Test). In order the mitigate the effect
        of the target reflection energy spreading some cells are left out from the calculation in the immediate
        vicinity of the CUT. These cells are the guard cells.
        The size of the estimation window and guard window can be set with the win_param parameter.

    Implementation notes:
    ---------------------
        Implementation based on https://github.com/petotamas/APRiL
    Parameters:
    -----------
    :param win_param: Parameters of the noise power estimation window
                      [Est. window length, Est. window width, Guard window length, Guard window width]
    :param threshold: Threshold level above the estimated average noise power
    :type win_param: python list with 4 elements
    :type threshold: float
    Return values:
    --------------
    """

def __init__(self, win_param, threshold, rd_size):
    # Unpack the list of window parameters into individual variables for readability
    win_width = win_param[0]
    win_height = win_param[1]
    guard_width = win_param[2]
    guard_height = win_param[3]

    # Create the sliding window "kernel" (mask) initially filled with 1s (Reference Cells)
    # The size covers the full span: Reference + Guard + CUT (Cell Under Test)
    self.mask = np.ones((2 * win_height + 1, 2 * win_width + 1), dtype=bool)

    #

    # Set the center region (Guard Cells and CUT) to 0 so they are excluded from the noise calculation
    self.mask[
        win_height - guard_height:win_height + 1 + guard_height, win_width - guard_width:win_width + 1 + guard_width] = 0

    # Convert the threshold from decibels (dB) to a linear power ratio
    self.threshold = 10 ** (threshold / 10)

    # Pre-calculate how many valid reference cells exist for every pixel position
    # We create a dummy matrix of 1s (same size as our data) and convolve it with our mask.
    # This creates a map that tells us exactly how many cells we are averaging over at any point,
    # which automatically handles the "edge cases" where the window hangs off the side of the map.
    self.num_valid_cells_in_window = signal.convolve2d(np.ones(rd_size, dtype=float), self.mask, mode='same')


def __call__(self, rd_matrix):
    """
    Description:
    ------------
        Performs the automatic detection on the input range-Doppler matrix.

    Implementation notes:
    ---------------------
    Parameters:
    -----------
    :param rd_matrix: Range-Doppler map on which the automatic detection should be performed
    :type rd_matrix: R x D complex numpy array
    Return values:
    --------------
    :return hit_matrix: Calculated hit matrix
    """
    # Convert the complex values (voltage) of the input matrix to Power (magnitude squared)
    rd_matrix = np.abs(rd_matrix) ** 2

    # 1. Sum the noise power around every cell
    # Scipy's convolve2d efficiently "slides" our mask over the entire matrix at once.
    # For every pixel, it multiplies the neighbors by the mask (1 for ref cells, 0 for guard) and sums them.
    rd_windowed_sum = signal.convolve2d(rd_matrix, self.mask, mode='same')

    # 2. Calculate the Average Noise Power
    # Divide the summed noise power by the count of valid cells (handling edges correctly)
    rd_avg_noise_power = rd_windowed_sum / self.num_valid_cells_in_window

    # 3. Calculate SNR (Signal-to-Noise Ratio)
    # Divide the power of the Cell Under Test (the pixel itself) by the estimated noise floor
    rd_snr = rd_matrix / rd_avg_noise_power

    # 4. Determine Hits
    # Create a boolean matrix where True means the SNR exceeded the threshold
    hit_matrix = rd_snr > self.threshold

    # Return the boolean detection map and the calculated SNR map
    return hit_matrix, rd_snr