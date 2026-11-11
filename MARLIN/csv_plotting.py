"""
plot_pdws.py - plot streamed PDWs as they are: no grouping, no colour coding.
Everything is drawn in one colour with transparency, so dense clusters show up darker.

PRF is not a field in the PDWs; it is derived from arrival times (TOA = TimeInSegment_s - PulseWidth),
using only pairs of PDWs from the same acquisition (times are exact only within one acquisition).
PRF_METHOD chooses which earlier PDW the interval is measured back to:

  "previous"  the most recent earlier PDW at least 1/PRF_MAX_HZ back, whatever it was. When transmitters
              interleave, many intervals span two different transmitters (e.g. 11 us -> 90 kHz).
  "histogram" the most recent earlier PDW whose interval matches a peak in the arrival-time
              difference histogram (figure 4). The histogram uses the differences between ALL pairs of
              PDWs (not just neighbours): each transmitter's PRI builds a sharp peak, intervals between
              unrelated transmitters spread into a flat background. Assumes only PRF_MAX_HZ.
  "similar"   the most recent earlier PDW whose bandwidth is within SIMILAR_BW_FRACTION and power
              within SIMILAR_POWER_DB (searching up to SIMILAR_LOOKBACK PDWs back). Interleaved
              transmitters with different parameters then get their own PRI, but this assumes a
              transmitter's pulses look alike.

Figures:
  1. 3D scatter: PRF x power x bandwidth
  2. Power vs time over the whole capture
  3. Pulse timeline: a few milliseconds of PDWs (line height = power, bar = pulse width)
  4. Arrival-time difference histogram with the detected PRI peaks
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===========================================================================
#  SETTINGS
# ===========================================================================
CSV_PATH = r"E:\MARLIN_OCTOBER_2026_DAY_2\IFM_T29_1\stream_IFM_test_1\PDW_formatted_online_3_stream.csv"

PRF_METHOD = "histogram"    # "histogram", "previous" or "similar" (see above)
PRF_MAX_HZ = 5000           # known maximum PRF: earlier PDWs closer than 1/PRF_MAX_HZ (200 us) are skipped
                            # as references, since no real PRI can be that short. None = no limit
SIMILAR_BW_FRACTION = 0.30
SIMILAR_POWER_DB = 2.5
SIMILAR_LOOKBACK = 30

# Arrival-time difference histogram (figure 4, and PRF_METHOD = "histogram")
HIST_MAX_PRI_US = 3000      # longest PRI searched for
HIST_BIN_US = 0.5           # histogram bin width
HIST_LOOKBACK = 60          # PDWs searched back per PDW (must reach HIST_MAX_PRI_US at your pulse density)
PEAK_FACTOR = 8             # a peak must be this many times the histogram's typical (median) level
PEAK_TOL_US = 1.0           # "histogram" method: an interval matches a peak if within this
HARMONIC_TOL = 0.003        # peaks within 0.3% of a multiple (or a sum of two) of smaller peaks are marked derived
PRF_AXIS_MAX_HZ = 5000      # 3D plot PRF axis runs 0..this (PDWs with a higher PRF are left out)
PRF_LOG_AXIS = False        # True = logarithmic PRF axis instead (shows every PRF, but stretched)

MAX_SCATTER_POINTS = 30000  # subsample the scatter plots for speed
POINT_COLOR = "tab:blue"
POINT_ALPHA = 0.15
TIMELINE_MS = 3.0           # length of the pulse-timeline window
TIMELINE_START_S = None     # start of the timeline (TimeInSegment_s); None = start of the first acquisition
MAX_PW_US = 200             # PDWs longer than this are left out of the timeline (partial pulses)
SAVE_PLOTS = True           # save PNGs next to the CSV, named <csv name>_<figure>.png
SHOW = True
# ===========================================================================

d = pd.read_csv(CSV_PATH)
d["TOA_s"] = d["TimeInSegment_s"] - d["PulseWidth"]
d = d.sort_values(["Segment", "Pass", "AcqIndex", "TOA_s"]).reset_index(drop=True)
t = d["TOA_s"].values
acq = ((d["Segment"] * 100000 + d["Pass"]) * 1000 + d["AcqIndex"]).values
bw = d["Bandwidth"].values / 1e6                                   # MHz
pwr = d["AveragePower"].values                                     # dB
pw = d["PulseWidth"].values * 1e6                                  # us
time_s = d["UnixTime_approx"].values - d["UnixTime_approx"].min()

# ---- PRI / PRF per PDW
pri = np.full(len(d), np.nan)
min_pri = 1.0 / PRF_MAX_HZ if PRF_MAX_HZ else 0.0
if PRF_METHOD not in ("previous", "similar", "histogram"):
    raise ValueError('PRF_METHOD must be "histogram", "previous" or "similar"')

# ---- arrival-time difference histogram (all pairs within an acquisition, min_pri..HIST_MAX_PRI_US)
diffs = []
for k in range(1, HIST_LOOKBACK + 1):
    cur = np.arange(k, len(d))
    gap = t[cur] - t[cur - k]
    ok = (acq[cur] == acq[cur - k]) & (gap >= min_pri) & (gap <= HIST_MAX_PRI_US * 1e-6)
    diffs.append(gap[ok])
diffs_us = np.concatenate(diffs) * 1e6
bins = np.arange(min_pri * 1e6, HIST_MAX_PRI_US + HIST_BIN_US, HIST_BIN_US)
hist, _ = np.histogram(diffs_us, bins)
centres = (bins[:-1] + bins[1:]) / 2
w = max(1, int(round(2.0 / HIST_BIN_US)))                         # peaks at least ~2 us apart
level = np.median(hist) + 1
peaks = []
for i in np.flatnonzero(hist > PEAK_FACTOR * level):
    lo, hi = max(0, i - w), min(len(hist), i + w + 1)
    if hist[i] == hist[lo:hi].max() and (not peaks or i - peaks[-1][0] > w):
        c = np.sum(centres[lo:hi] * hist[lo:hi]) / np.sum(hist[lo:hi])  # centroid
        peaks.append((i, c, int(hist[lo:hi].sum())))
fund, derived = [], []
for i, c, n in sorted(peaks, key=lambda x: x[1]):
    mult = [f for f in fund if round(c / f) >= 2 and abs(c / f - round(c / f)) < HARMONIC_TOL * round(c / f)]
    sums = [(a, b) for a in fund for b in fund if a <= b and abs(c - (a + b)) < HARMONIC_TOL * c]
    (derived if (mult or sums) else fund).append(c)
print(f"Difference histogram: {len(peaks)} peaks between {min_pri * 1e6:.0f} and {HIST_MAX_PRI_US} us")
for c in sorted(fund):
    print(f"   PRI {c:9.2f} us = PRF {1e6 / c:8.1f} Hz")
if derived:
    print(f"   (+ {len(derived)} peaks at multiples or sums of those: missed pulses / stagger cycles)")

# ---- PRI per PDW
todo = np.ones(len(d), bool)
for k in range(1, max(SIMILAR_LOOKBACK, HIST_LOOKBACK) + 1):
    cur = np.arange(k, len(d))
    gap = t[cur] - t[cur - k]
    ok = todo[cur] & (acq[cur] == acq[cur - k]) & (gap >= min_pri)
    if PRF_METHOD == "similar":
        ok &= ((np.abs(bw[cur] - bw[cur - k]) < SIMILAR_BW_FRACTION * np.maximum(bw[cur], 1e-3)) &
               (np.abs(pwr[cur] - pwr[cur - k]) < SIMILAR_POWER_DB))
    elif PRF_METHOD == "histogram":
        if not fund:
            break
        g_us = gap * 1e6
        ok &= np.min(np.abs(g_us[:, None] - np.array(fund)[None, :]), axis=1) < PEAK_TOL_US
    i = cur[ok]
    pri[i] = gap[ok]
    todo[i] = False
with np.errstate(divide="ignore"):
    prf = 1.0 / pri                                                # Hz (inf for zero intervals)

print(f"{len(d):,} PDWs over {time_s.max():.1f} s | PRF method: {PRF_METHOD} | "
      f"PRF available for {np.isfinite(prf).sum():,} PDWs")


def hz_label(f):
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"), (1, "Hz")):
        if f >= scale:
            return f"{f / scale:g} {unit}"
    return f"{f:g} Hz"


SAVE_DIR = os.path.dirname(os.path.abspath(CSV_PATH))
CSV_STEM = os.path.splitext(os.path.basename(CSV_PATH))[0]


def show(fig, filename):
    fig.tight_layout()
    if SAVE_PLOTS:
        path = os.path.join(SAVE_DIR, f"{CSV_STEM}_{filename}")
        fig.savefig(path, dpi=130)
        print(f"saved {path}")


rng = np.random.default_rng(0)
sub = rng.choice(len(d), min(MAX_SCATTER_POINTS, len(d)), replace=False)

# ---- Figure 1: 3D scatter (PRF, power, bandwidth)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(projection="3d")
if PRF_LOG_AXIS:
    s = sub[np.isfinite(prf[sub]) & (prf[sub] > 0)]
    x = np.log10(prf[s])
    ax.scatter(x, pwr[s], bw[s], s=2, alpha=POINT_ALPHA, color=POINT_COLOR)
    decades = np.arange(np.floor(x.min()), np.ceil(x.max()) + 1)
    ax.set_xticks(decades)
    ax.set_xticklabels([hz_label(10 ** e) for e in decades])
    ax.set_xlabel("PRF (log scale)")
else:
    s = sub[np.isfinite(prf[sub]) & (prf[sub] <= PRF_AXIS_MAX_HZ)]
    ax.scatter(prf[s], pwr[s], bw[s], s=2, alpha=POINT_ALPHA, color=POINT_COLOR)
    ax.set_xlim(0, PRF_AXIS_MAX_HZ)
    ax.set_xlabel("PRF (Hz)")
ax.set_ylabel("Power (dB)")
ax.set_zlabel("Bandwidth (MHz)")
ax.set_title(f"3D Scatterplot of PDWs from realtime csv: PRF / Power / Bandwidth")
show(fig, "1_scatter3d.png")

# ---- Figure 2: power vs time
fig, ax = plt.subplots(figsize=(11, 4))
ax.scatter(time_s[sub], pwr[sub], s=1, alpha=POINT_ALPHA, color=POINT_COLOR)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Power (dB)")
ax.set_title("Received Power vs Time")
show(fig, "2_power_vs_time.png")

# ---- Figure 3: pulse timeline
if TIMELINE_START_S is None:
    k0 = 0
    t0 = t[k0] + 0.2e-3                                            # skip the acquisition's first edge
else:
    k0 = int(np.searchsorted(t, TIMELINE_START_S))
    t0 = TIMELINE_START_S
w = (acq == acq[k0]) & (t >= t0) & (t < t0 + TIMELINE_MS * 1e-3) & (pw < MAX_PW_US)
fig, ax = plt.subplots(figsize=(12, 4))
ax.vlines((t[w] - t0) * 1e3, 0, pwr[w], color=POINT_COLOR, lw=1.5)
ax.hlines(pwr[w], (t[w] - t0) * 1e3, (t[w] - t0 + pw[w] * 1e-6) * 1e3, color=POINT_COLOR, lw=3)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Power (dB)")
ax.set_title(f"Pulse timeline: {TIMELINE_MS:g} ms (Line = Arrival time, Bar Length = Pulse Width)")
show(fig, "3_pulse_timeline.png")

# ---- Figure 4: arrival-time difference histogram
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(centres, hist, color=POINT_COLOR, lw=0.8)
for c in fund:
    ax.axvline(c, color="red", lw=0.8, alpha=0.7)
    ax.text(c, ax.get_ylim()[1] * 0.95, f" {c:.1f} us\n {1e6 / c:.0f} Hz", color="red", fontsize=7, va="top")
for c in derived:
    ax.axvline(c, color="grey", lw=0.6, ls=":")
ax.set_yscale("log")
ax.set_xlabel("Time difference between PDWs (us)")
ax.set_ylabel("count")
ax.set_title("Arrival-time differences, all PDW pairs in each acquisition "
             "(red = PRI peaks, grey dotted = multiples/sums)")
show(fig, "4_difference_histogram.png")

if SHOW:
    plt.show()