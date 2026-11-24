# claude mD project

`md_spectrogram_analysis.py` is an adapted version of
`Testing/micro_doppler_testing_RD_complete.py`. It makes paper-sized micro-Doppler
spectrograms and measures the rotor parameters needed for the paper.

To use it, edit the configuration block at the top (capture directory, target name,
carrier frequency, range bin, first slice, body velocity, bistatic angle) and run it.
It writes to `<capture directory>/claude_md_figures/`:

- `*_md_long_1024.pdf/.png`: 0.5 s window spectrogram (spectral lines)
- `*_md_short_8.pdf/.png`: ~4 ms window spectrogram (blade flashes)
- `*_results.txt`: body Doppler, Doppler extent B, line spacing F_r, flash interval,
  blade parity, tip speed, and rpm / blade length for 2 to 6 blades
- `*_diagnostics.png`: plots to check each measurement was picked correctly

The figures are 1.72 in x 1.45 in with 7 pt text, so four fit across a two-column
IEEE page in a `figure*`. Include them with `\includegraphics[width=\linewidth]`.
