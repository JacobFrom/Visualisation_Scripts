import numpy as np
import matplotlib.pyplot as plt
import os


def plot_gain_steps_psd(capture_dirs, gain_labels, fs, limit, save_dir=None):
    """
    Plots overlaid PSDs for 4 different gain captures across 4 channels.
    Creates a 2x2 grid (1 subplot per channel).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Gain Step Analysis: 6 dB Increments (PSD)", fontsize=16)

    # Loop over the 4 physical channels (2, 3, 4, 5)
    for ch_idx in range(4):
        physical_ch = ch_idx + 2
        ax = axes[ch_idx // 2, ch_idx % 2]

        # Loop over the 4 different gain captures
        for cap_idx, capture_dir in enumerate(capture_dirs):
            file_path = os.path.join(capture_dir, f'channel_{physical_ch}_output.npy')

            try:
                # Load just enough data to reach our limit
                raw_data = np.load(file_path)
                complex_data = raw_data[0:limit:2] + 1j * raw_data[1:limit:2]

                # Plot the PSD. Using a slightly lower alpha to see overlapping lines
                ax.psd(complex_data, NFFT=2048, Fs=fs,
                       label=f"{gain_labels[cap_idx]}", alpha=0.8)
            except FileNotFoundError:
                print(f"Warning: Could not find {file_path}")

        ax.set_title(f"Channel {physical_ch} Signal Power")
        # Zoom in to see the 0 Hz DC and the 2 MHz signal peak
        ax.set_xlim(-5e6, 5e6)

        # Adding a grid and legend to easily measure the 6 dB jumps
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')

    plt.tight_layout()

    if save_dir:
        # Force Python to create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, 'gain_step_comparison.png')
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved successfully to: {save_path}")
        except Exception as e:
            # Fallback: Save to the current working directory if permissions fail
            fallback_path = 'gain_step_comparison_fallback.png'
            plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
            print(f"Path error. Saved fallback plot to current folder: {fallback_path}")
            print(f"Error details: {e}")

    plt.show()


if __name__ == "__main__":
    # --- System Parameters ---
    Fs = 3.84e9
    total_decimation = 64
    eff_bandwidth = Fs / total_decimation

    num_samples = 2 ** 19
    limit = num_samples * 2

    # --- Define your 4 capture directories here ---
    # Update these paths to point to your actual 4 folders
    base_dir = r'/home/uceejpw/x_band_testing_september_2026_data'

    capture_dirs = [
        os.path.join(base_dir, 'ADC_2345_cable_GDB_gain_0_2MHz'),
        os.path.join(base_dir, 'ADC_2345_cable_GDB_gain_6_2MHz'),
        os.path.join(base_dir, 'ADC_2345_cable_GDB_gain_12_2MHz'),
        os.path.join(base_dir, 'ADC_2345_cable_GDB_gain_18_2MHz')
    ]

    gain_labels = ["Gain 0 dB", "Gain 6 dB", "Gain 12 dB", "Gain 18 dB"]

    print("Loading data and generating overlaid plots...")
    # Saving to the highest gain folder for convenience
    plot_gain_steps_psd(capture_dirs, gain_labels, eff_bandwidth, limit, save_dir=capture_dirs[-1])