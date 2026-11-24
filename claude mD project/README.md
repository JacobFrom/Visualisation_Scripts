# claude mD project

`md_spectrogram_analysis.py` is an adapted version of
`Testing/micro_doppler_testing_RD_complete.py`. It makes paper-sized micro-Doppler
spectrograms and measures the rotor parameters needed for the paper.

To use it, edit the configuration block at the top (capture directory, target name,
carrier frequency, range bin, first slice, body velocity, bistatic angle) and run it.
It writes to `<capture directory>/claude_md_figures/`:

- `*_md_long_256.pdf/.png`: 0.125 s window spectrogram (spectral lines)
- `*_md_short_64.pdf/.png`: ~31 ms window spectrogram (blade flashes, as in the original script)
- `*_results.txt`: body Doppler, Doppler extent B, line spacing F_r, flash interval,
  blade parity, tip speed, and rpm / blade length for 2 to 6 blades
- `*_diagnostics.png`: plots to check each measurement was picked correctly

The figures are 1.72 in x 1.45 in with 7 pt text, so four fit across a two-column
IEEE page in a `figure*`. Include them with `\includegraphics[width=\linewidth]`.

## Propeller version (fixed wing)

`md_spectrogram_analysis_propeller.py` is the same processing with settings for nose propellers
(PC-12, Delphin). The helicopter script is unchanged. Set the lines marked `CHECK` (capture folder,
target, carrier, range bin, first slice, body velocity) before running. Differences:

1. Line spacing search up to 400 Hz. rpm from the line spacing is the value to trust.
2. `NFFT_short = 16` (figure) and `NFFT_flash = 8` (measurement) to try to resolve flashes ~4-8 ms
   apart. If they do not resolve, ignore the flash interval and parity.
3. `num_slices = 3` and `edge_trim_s = 0.01`: an aircraft crosses a 20 m bin in 20 / (2 v) s.
   The script prints this crossing time.
4. `theta_deg = 90`, so tip speed and blade length are lower bounds (true theta ~ heading angle).
5. Wider body notch (2 m/s) and the full +/- PRF/2 Doppler axis, so aliased lines can be seen.
