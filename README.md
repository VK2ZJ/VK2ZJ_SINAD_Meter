# VK2ZJ SINAD Meter

This is a little python app to measure SINAD (Signal-to-Noise and Distortion ratio) with a PC audio input, such as a USB audio interface.

The primary use case is evaluating radios, such as when measuring FM receiver sensitivity. It's time to let those old SINADDERs have some deserved rest on their museum shelves.

VK2ZJ_SINAD_Meter.exe is the main app. No installation required - just download and run it. Note: You may get a warning from Windows or your antivirus the first time you run the app. I haven't learned how to sign executables yet.

Broadly, there are three methods for SINAD measurements, which differ by the type of filter applied before the main measurement itself:
1) Most hams use no well defined band pass filter, except the equipment under test itself. This is typically the case when using old test gear such as the SINADDER.
2) ARRL Test Procedures Manual specifies the use of the HP 339A Distortion Measurement Set with its 30kHz low-pass filter enabled.
3) Professional analogue radio standards, such as the ETSI EN 300 086, use the Psophometric filter as per ITU-T O.41 - see https://en.wikipedia.org/wiki/Psophometric_weighting.

Method 1) and 2) will produce the same SINAD in most cases.. unless your radio somehow produces ultrasonic noise above 30kHz. Method 3) will produce higher SINAD values with most radios, because the Psophometric filter removes some of the noise.

The first number on the top of the window is the SINAD without any filter, except the sound card's anti-aliasing filter. The second number is the SINAD with the Psophometric filter. Use the non-filtered (first) value when comparing results to ARRL lab's and most other ham radio specs. Use the 2nd value (Psopho-filter) for evaluating professional LMR / marine / etc radios against their specs.

Limitations:
* The app expects a 1kHz tone, which is the standard for testing radio sensitivity.
* The notch filter in the SINAD algorithm implementation is only 50Hz wide, so you must use a good signal generator with the modulation frequency accuracy or ±25Hz.
* The result will only be as good as the audio input of your PC. Turn off all input processing effects, like Dolby, AGC, etc.
* The app won't measure very high SINAD such as what might be needed for high-end audio equiment. Most audio input devices / chipsets don't have the required performance.
* The input level must not cause clipping in the audio input.. adjust your radio volume while monitoring the waveform plot the app gives you, to spot-check for clipping and other issues.

Tested on Windows only so far, but the underlying python code should work on most OSes.

Verification:
Thanks G0HZU for providing some test vectors. The underlying DSP library matches the test vector within a fraction of a dB. I'm keen for others to provide other test vectors. You can process any wav files recorded at 44.1kHz or higer sample rate, using the sinad_wav.exe command line program. Just open the Windows command line, type sinad_wav.exe <wave_filename>, and you'll get a plot of SINAD over time. The program sinad_wav uses the same library as the main app. The library itself is dsp.py.

Additional info on each file in this repo:
* LICENSE.. as usual
* build_sinad_wav.bat builds the sinad_wav.exe executable.
* build_VK2ZJ_SINAD_Meter.bat builds the maain app executable.
* install_deps.bat installs all required libraries for running the Python code directly from command line, and for building the two executables.
* sinad_wav.exe is the command line SINAD plotter executable.
* VK2ZJ_SINAD_Meter.exe is the main executable.
* VK2ZJ_SINAD_Meter.ico is just the icon.
* README.md is this file.
* G0HZU_12dB_SINAD_OOK2.png is the SINAD plot obtained by running "sinad_wav.exe G0HZU_12dB_SINAD_OOK2.wav".
* VK2ZJ_SINAD_Meter.png is the main app screenshot.
* dsp.py is the library with the SINAD algorithm implementation.
* sinad_wav.py is the main source code for sinad_wav.exe.
* test_plot_filters.py plots the response of the two filters from dsp.py. Basically, a unit test.
* VK2ZJ_SINAD_Meter.pyw is the top level source for the main app.
* requirements.txt contains the list of dependencies.
* G0HZU_**.wav are the test vectors.


![alt text](https://github.com/VK2ZJ/VK2ZJ_SINAD_Meter/blob/main/VK2ZJ_SINAD_Meter.png?raw=true)
