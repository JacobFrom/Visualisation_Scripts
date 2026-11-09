import numpy as np
import matplotlib.pyplot as plt
import os


def plot_beginner_rf_basics(data_list, fs, save_dir=None):
    """
    Generates the PSD (Frequency Domain) and Time-Domain Magnitude for 4 channels.
    Uses a 4x2 grid: Rows = Channels, Columns = (PSD, Time-Domain)
    """
    # 4 rows (channels), 2 columns (plot types)
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle("RF Fundamentals: PSD vs Time-Domain Magnitude", fontsize=16)

    for ch in range(4):
        ch_data = data_list[ch]
        physical_ch = ch + 2

        # --- Plot 1: Power Spectral Density (Left Column) ---
        ax_psd = axes[ch, 0]
        ax_psd.psd(ch_data, NFFT=2048, Fs=fs, color='blue', alpha=0.8)
        ax_psd.set_title(f"Channel {physical_ch} - Frequency Domain (PSD)")
        # Zoom in slightly to clearly see the 0 Hz DC offset and the 2 MHz Signal
        ax_psd.set_xlim(-10e6, 10e6)

        # --- Plot 2: Time-Domain Magnitude (Right Column) ---
        magnitude = np.abs(ch_data)

        ax_time = axes[ch, 1]
        num_samples_to_plot = 2 ** 16
        t = np.arange(num_samples_to_plot) * (1e6 / fs)  # Time in microseconds

        ax_time.plot(t, magnitude[:num_samples_to_plot], color='orange')
        ax_time.set_title(f"Channel {physical_ch} - Time Domain (Magnitude)")
        ax_time.set_xlabel("Time (microseconds)")
        ax_time.set_ylabel("Amplitude")
        ax_time.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # --- Save the plot before showing ---
    if save_dir:
        save_path = os.path.join(save_dir, 'rf_fundamentals_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {save_path}")

    plt.show()


if __name__ == "__main__":
    # --- System Parameters & Data Loading ---
    Fs = 3.84e9  # Base Sample Rate
    total_decimation = 8 * 8  # tile_decimation * block_decimation
    eff_bandwidth = Fs / total_decimation

    # Define the limit to prevent massive memory usage
    num_samples = 2 ** 21
    limit = num_samples * 2  # *2 because I and Q are interleaved

    test_directory = r'/home/uceejpw/x_band_testing_september_2026_data/ADC_2367_cable_GDB_gain_0_2MHz'

    print(f"Loading arrays from {test_directory}...")
    ch0_raw = np.load(os.path.join(test_directory, 'channel_2_output.npy'))
    ch1_raw = np.load(os.path.join(test_directory, 'channel_3_output.npy'))
    ch2_raw = np.load(os.path.join(test_directory, 'channel_4_output.npy'))
    ch3_raw = np.load(os.path.join(test_directory, 'channel_5_output.npy'))

    # Construct Complex Arrays
    ch0_data = ch0_raw[0:limit:2] + 1j * ch0_raw[1:limit:2]
    ch1_data = ch1_raw[0:limit:2] + 1j * ch1_raw[1:limit:2]
    ch2_data = ch2_raw[0:limit:2] + 1j * ch2_raw[1:limit:2]
    ch3_data = ch3_raw[0:limit:2] + 1j * ch3_raw[1:limit:2]

    iq_data = [ch0_data, ch1_data, ch2_data, ch3_data]

    # --- Execute Plots ---
    print("Generating plots...")
    # Pass the test_directory to the save_dir argument
    plot_beginner_rf_basics(iq_data, fs=eff_bandwidth, save_dir=test_directory)