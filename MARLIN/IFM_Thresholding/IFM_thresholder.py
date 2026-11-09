"""
Estimate IFM noise / trigger thresholds from a raw IQ capture.

Method
------
1. Form the same detection statistic you currently plot (|IQ| integrated with a
   boxcar of int_len samples).
2. Robustly estimate the noise floor (iterative sigma clipping), then find the
   pulse plateau level with an Otsu split between "noise" and "signal".
3. Mask out the pulses (plus a guard band) and re-measure the noise from the
   pulse-free samples only: mean, sigma, peak.
4. Thresholds:
     noise threshold   = noise_mean + noise_sigma_low    * noise_sigma
     trigger threshold = noise_mean + trigger_sigma_high * noise_sigma
                         (raised to just above the largest noise-only sample)
5. Cross-check with a threshold sweep: the number of pulses detected should be
   constant over a wide range of thresholds. That "plateau" is where it is safe
   to set the trigger.

Usage:  python ifm_threshold_analysis.py [path/to/channel_X_iq.npy]
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
iq_directory = r"/home/uceejpw/x_band_testing_september_2026_data/IFM_final_week_testing/S_band_IFM_test_1_signal_0dB_GDB_18dB"
channel = 4
iq_file = os.path.join(iq_directory, f"channel_{channel}_iq.npy")
if len(sys.argv) > 1:
    iq_file = sys.argv[1]

fs = 3.84e9 / 32  # effective sample rate (Hz)
int_len = 10  # boxcar length; should match your IFM's integration
segment = None  # e.g. (100000, 200000) to analyse a slice, or None

min_pulse_us = 0.1  # ignore detections shorter than this
noise_sigma_low = 3.0  # noise threshold  = mean + this * sigma
trigger_sigma_high = 6.0  # trigger threshold = mean + this * sigma
guard_samples = 2 * int_len  # guard band around pulses when measuring noise


# ---------------------------------------------------------------------


def detect_pulses(x, high, low, min_len):
    """Hysteresis detector. A pulse is a run of x > low that contains at least
    one sample > high and lasts >= min_len samples. Returns (starts, ends)
    with ends exclusive."""
    above = (x > low).astype(np.int8)
    d = np.diff(above, prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    hi_cum = np.concatenate(([0], np.cumsum(x > high)))
    has_hi = (hi_cum[ends] - hi_cum[starts]) > 0
    keep = has_hi & ((ends - starts) >= min_len)
    return starts[keep], ends[keep]


def otsu_split(values, bins=256):
    """Threshold that best separates a bimodal distribution."""
    hist, edges = np.histogram(values, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(float)
    w0 = np.cumsum(w)
    w1 = w0[-1] - w0
    s = np.cumsum(w * centres)
    m0 = s / np.maximum(w0, 1)
    m1 = (s[-1] - s) / np.maximum(w1, 1)
    return centres[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def decimate_max(t, x, max_pts=200_000):
    """Max-pool for plotting so narrow pulses are not lost."""
    step = max(1, len(x) // max_pts)
    n = len(x) // step * step
    return t[:n:step], x[:n].reshape(-1, step).max(axis=1)


# ---------------------------------------------------------------------
# Load and form the detection statistic
# ---------------------------------------------------------------------
raw = np.load(iq_file).astype(np.float64)
iq = raw[0::2] + 1j * raw[1::2]
if segment is not None:
    iq = iq[segment[0]:segment[1]]

mag = np.abs(iq)
x = np.convolve(mag, np.ones(int_len), mode="same")
x = x[int_len:-int_len]  # drop convolution edge effects
t = (np.arange(len(x)) + int_len) / fs
min_len = max(1, int(np.ceil(min_pulse_us * 1e-6 * fs)))

print(f"File: {iq_file}")
print(f"{len(iq)} IQ samples ({len(iq) / fs * 1e3:.3f} ms), int_len={int_len}")

# ---------------------------------------------------------------------
# Pass 1: robust noise estimate and signal level
# ---------------------------------------------------------------------
mu = np.median(x)
sd = 1.4826 * np.median(np.abs(x - mu))
for _ in range(20):
    keep = x < mu + 4 * sd
    mu_new, sd_new = x[keep].mean(), x[keep].std()
    converged = abs(mu_new - mu) < 1e-6 * mu and abs(sd_new - sd) < 1e-6 * sd
    mu, sd = mu_new, sd_new
    if converged:
        break

split = 10 ** otsu_split(np.log10(np.maximum(x, 1e-12)))
p_start, p_end = detect_pulses(x, split, split, min_len)
if len(p_start) == 0:
    print("\nNo pulses found by the initial split. Either there is no signal in "
          "this capture or it is too close to the noise to separate.")
    sys.exit(1)

signal_level = np.median(x[x > split])

# ---------------------------------------------------------------------
# Pass 2: noise statistics from pulse-free samples only
# ---------------------------------------------------------------------
mask = np.zeros(len(x), dtype=bool)
for s, e in zip(p_start, p_end):
    mask[max(0, s - guard_samples):e + guard_samples] = True
noise = x[~mask]
if len(noise) < 1000:
    print("Warning: very few pulse-free samples; noise estimate is unreliable.")

noise_mean = noise.mean()
noise_std = noise.std()
noise_peak = noise.max()

noise_thr = noise_mean + noise_sigma_low * noise_std
trigger_thr = max(noise_mean + trigger_sigma_high * noise_std, 1.05 * noise_peak)

# ---------------------------------------------------------------------
# Validate: detect with these thresholds
# ---------------------------------------------------------------------
d_start, d_end = detect_pulses(x, trigger_thr, noise_thr, min_len)
widths = (d_end - d_start) / fs
pri = np.diff(d_start) / fs

# ---------------------------------------------------------------------
# Threshold sweep: pulse count vs threshold
# ---------------------------------------------------------------------
sweep_lo = noise_mean + 2 * noise_std
sweep_hi = 1.1 * signal_level
thr_grid = np.geomspace(sweep_lo, sweep_hi, 150)
counts = np.array([len(detect_pulses(x, th, th, min_len)[0]) for th in thr_grid])

plateau = None
nz = counts[counts > 0]
if len(nz):
    vals, c = np.unique(nz, return_counts=True)
    mode = vals[np.argmax(c)]
    eq = (counts == mode).astype(int)
    d = np.diff(np.concatenate(([0], eq, [0])))
    rs, re = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    i = np.argmax(re - rs)
    plateau = (thr_grid[rs[i]], thr_grid[re[i] - 1], mode)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
def table(label, v):
    pm = v / int_len
    return (f"  {label:<22} {v:>14.2f} | {pm:>12.2f} | {pm ** 2:>14.1f} | "
            f"{20 * np.log10(max(pm, 1e-12)):>8.2f}")


print("\n=== Levels ===")
print(f"  {'':<22} {'integrated mag':>14} | {'per-sample mag':>12} | "
      f"{'per-sample pwr':>14} | {'dB (mag)':>8}")
print(table("noise mean", noise_mean))
print(table("noise peak (observed)", noise_peak))
print(table("NOISE THRESHOLD", noise_thr))
print(table("TRIGGER THRESHOLD", trigger_thr))
print(table("signal plateau", signal_level))
print(f"\n  noise sigma = {noise_std:.2f} (integrated mag units)")
print(f"  SNR (plateau vs noise mean) = "
      f"{20 * np.log10(signal_level / noise_mean):.1f} dB")

print("\n=== Detection with these thresholds ===")
print(f"  pulses detected : {len(d_start)}")
if len(d_start):
    print(f"  median width    : {np.median(widths) * 1e6:.3f} us "
          f"(measured at the noise threshold, so slightly wider than true)")
if len(pri):
    print(f"  median PRI      : {np.median(pri) * 1e6:.3f} us")

print("\n=== Checks ===")
ok = True
if trigger_thr > noise_mean + 0.5 * (signal_level - noise_mean):
    ok = False
    print("  ! Trigger threshold is above the midpoint between noise and signal: "
          "weak pulses may be missed. SNR is marginal.")
if signal_level < noise_mean + 8 * noise_std:
    ok = False
    print("  ! Signal plateau is less than 8 sigma above the noise: expect "
          "unreliable detection whatever thresholds you pick.")
if plateau is not None:
    p_lo, p_hi, p_n = plateau
    print(f"  Sweep plateau: {p_n} pulses over thresholds "
          f"{p_lo:.1f} to {p_hi:.1f} (integrated mag)")
    if trigger_thr < p_lo:
        ok = False
        print("  ! Trigger is below the stable plateau: noise is probably "
              "creating extra detections.")
    elif trigger_thr > p_hi:
        ok = False
        print("  ! Trigger is above the stable plateau: pulses are being lost.")
    if len(d_start) != p_n:
        ok = False
        print(f"  ! Hysteresis detector found {len(d_start)} pulses but the "
              f"plateau count is {p_n}.")
else:
    print("  No stable plateau found in the threshold sweep.")
if ok:
    print("  All checks passed.")

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(11, 11))

tp, xp = decimate_max(t, x)
ax[0].plot(tp * 1e6, xp, lw=0.7, label="integrated |IQ|")
ax[0].axhline(noise_mean, color="grey", ls=":", label="noise mean")
ax[0].axhline(noise_thr, color="orange", ls="--", label="noise threshold")
ax[0].axhline(trigger_thr, color="red", ls="--", label="trigger threshold")
ax[0].axhline(signal_level, color="green", ls=":", label="signal plateau")
for s, e in zip(d_start, d_end):
    ax[0].axvspan(t[s] * 1e6, t[e - 1] * 1e6, color="green", alpha=0.15)
ax[0].set_xlabel("Time (us)")
ax[0].set_ylabel("Integrated magnitude")
ax[0].set_title(f"Channel {channel}: detection statistic and thresholds")
ax[0].legend(loc="upper right", fontsize=8)

ax[1].hist(x, bins=300, color="steelblue")
ax[1].set_yscale("log")
for v, c, lab in [(noise_thr, "orange", "noise thr"), (trigger_thr, "red", "trigger thr"),
                  (signal_level, "green", "signal")]:
    ax[1].axvline(v, color=c, ls="--", label=lab)
ax[1].set_xlabel("Integrated magnitude")
ax[1].set_ylabel("Samples (log)")
ax[1].set_title("Histogram: noise peak on the left, pulse plateau on the right")
ax[1].legend(fontsize=8)

ax[2].semilogx(thr_grid, counts, ".-")
if plateau is not None:
    ax[2].axvspan(plateau[0], plateau[1], color="green", alpha=0.2, label="stable plateau")
ax[2].axvline(noise_thr, color="orange", ls="--", label="noise thr")
ax[2].axvline(trigger_thr, color="red", ls="--", label="trigger thr")
ax[2].set_xlabel("Threshold (integrated magnitude)")
ax[2].set_ylabel("Pulses detected")
ax[2].set_title("Pulse count vs threshold (pick the trigger inside the flat region)")
ax[2].legend(fontsize=8)

plt.tight_layout()
plt.show()
