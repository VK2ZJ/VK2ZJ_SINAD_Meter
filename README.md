# VK2ZJ_SINAD_Meter
Little python app to measure SINAD with an audio input.

SINAD stands for Signal-to-Noise and Distortion ratio.

The specific method implemented here is what has been in use in the radio comms (and analog telephony) industry. 

The signal is passed through the Psophometric filter as per ITU-T O.41 (see https://en.wikipedia.org/wiki/Psophometric_weighting).

The app expexts a 1kHz tone.

Tested on Windows only so far, but should work on other systems.

![alt text](https://github.com/VK2ZJ/VK2ZJ_SINAD_Meter/blob/main/VK2ZJ_SINAD_Meter.png?raw=true)
