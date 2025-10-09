import dsp
import numpy as np
import scipy.signal
import scipy.io
import matplotlib.pyplot as plt

# Plot the psophometric filter response
plt.figure(1)
sample_rate = 44e3
coeffs, state = dsp.psophometric_filter(sample_rate)
freq = (
    0.1, 16.66, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 
    1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 8000, sample_rate/2)
gain = (
    -85.0, -85.0, -63.0, -41.0, -21.0, -10.6, -6.3, -3.6, -2.0, -0.9, -0.0,
     0.6, 1.0, -0.0, -0.9, -1.7, -2.4, -3.0, -4.2, -5.6, -8.5, -15.0, -25.0,
     -36.0, -43.0, -80, -80)
w, h = scipy.signal.freqz(coeffs, fs = sample_rate, worN = 8192)
plt.semilogx(freq, gain, '+', label = 'Required')
plt.semilogx(w, 20 * np.log10(np.abs(h)), label='Actual')
plt.title('Psophometric Filter Response')
plt.xlim(100, 10000)
plt.ylim(-50, 10)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()


# Plot the notch filter response
plt.figure(2)
coeffs, state = dsp.notch_filter(sample_rate, 1e3)
w, h = scipy.signal.sosfreqz(coeffs, fs = sample_rate, worN = 8192)
plt.plot(w, 20 * np.log10(np.abs(h)))
plt.title('Notch Filter Response')
plt.xlim(100, 10000)
plt.ylim(-80, 10)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()

plt.show()
