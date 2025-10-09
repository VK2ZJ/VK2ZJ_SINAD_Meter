"""VK2ZJ sinad_wav

Plots SINAD from a pre-recorded wav file. 

Hard coded for 1kHz sine wave.

Use case: VHF/UHF radio measurements.

Usage:
  sinad_wav.py <filename>
  sinad_wav.py (-h | --help)

Options:
  -h --help     Show this screen.
"""

import dsp
import numpy as np
import scipy.signal
import scipy.io
import matplotlib.pyplot as plt
import docopt


def plot_sinad(filename):
    # Plot the SINAD for test vectors
    plt.figure()
    plt.title('SINAD test (%s)' % filename)

    sample_rate, samples_int = scipy.io.wavfile.read(filename)
    samples = samples_int.astype(np.float64)

    # Calculate the filter coefficients for the sample rate used in the recording
    bpf_coeffs, bpf_state = dsp.psophometric_filter(sample_rate)
    notch_filter_coeffs, notch_filter_state = dsp.notch_filter(sample_rate, 1e3)

    # Plot the SINAD for each small chunk of samples (No band pass filter)
    n_samples_per_chunk = sample_rate
    n_chunks = len(samples) // n_samples_per_chunk
    sinad_plot_no_filter = []
    for i in range(n_chunks):
        chunk = samples[n_samples_per_chunk * i : n_samples_per_chunk * (i + 1)]
        sinad, bpf_state, notch_filter_state = dsp.calculate_SINAD(
            sample_rate,
            chunk,
            None,
            bpf_state,
            notch_filter_coeffs,
            notch_filter_state)
        sinad_plot_no_filter.append(sinad)

    plt.plot(sinad_plot_no_filter) 

    # Plot the SINAD for each small chunk of samples (Psophometric filter)
    sinad_plot = []
    for i in range(n_chunks):
        chunk = samples[n_samples_per_chunk * i : n_samples_per_chunk * (i + 1)]
        sinad, bpf_state, notch_filter_state = dsp.calculate_SINAD(
            sample_rate,
            chunk,
            bpf_coeffs,
            bpf_state,
            notch_filter_coeffs,
            notch_filter_state)
        sinad_plot.append(sinad)

    plt.plot(sinad_plot)    

    plt.legend(('No Bandpass Filter', 'Psophometric Filter'))
    plt.grid()
    plt.ylim(-10, 60)
    plt.xlabel('Time [s]')
    plt.ylabel('SINAD (dB)')
    plt.show()


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    filename = arguments['<filename>']
    plot_sinad(filename)

