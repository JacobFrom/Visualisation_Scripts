This repository represents over a year of work in signal processing. The master branch has all the original code in its messy glory, but this branch serves as a starting point for members of the group that just need some signal processing done on data they have collected but do not want to write their own code base from scratch.
Essentially, I took the main branch of this repository and had a Claude agent (opus 4.5 I believe) refactor the entire codebase into this class based approach. 

The signal processing scripts are found in their relevant active or passive directories.

Active_processing contains the following:
- A basic PSD plotting script
- active_processor.py: This script contains the active processing class and all of the relevent methods (class functions) for a multitude of signal processing functions such as RTI plotting, MTI plotting, generation of RD videos and cell averaged CFAR options.
- active_rd_processing_function_edition: A poorly named script that serves an important purpose. It is the script you should use for your signal processing, calling any relevant methods you wish and filling out the relevant paramaters of your waveforms

Passive_processing contains the following:
-A basic PSD plotting script for your channels
-passive_processor.py: This script is similar to its active counterpart, however there are more functions within this class. The functions include surveillance channel weiner filtering, a bespoke cross correlation function and some other handy functions. Much of the processing is done by default within the class so you do not need to be an expert to use it.
-passive_rd_processing.py: This script is the one you should use for your passive processing, much of it is already set up to process your passive captures into nicely filtered RD videos. The user is free to set their waveform parameters, data directories and save filename preferences, as well as any other additional methods they wish to call or properties of those methods they wish to change. The processing algorithms are computationally efficient, a 70 second ~10 MHz bandwidth passive capture takes roughly 2 minutes to process from start to finish.

Future work: 
-Add a micro-Doppler method to both active and passive processing that uses user input + CFAR outputs to automatically plot relevant range bins
-Increase computational efficiency
-Add array based processing methods such as DOA and general channel comparisons of important things such as amplitude and phase differences between channels
-ES processing: A third directory containing ES processing algorithms
