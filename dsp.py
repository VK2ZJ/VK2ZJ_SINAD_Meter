import numpy as np
import scipy.signal


def psophometric_filter(sample_rate):
    # Psophometric filter design as per ITU-T O.41 (see https://en.wikipedia.org/wiki/Psophometric_weighting)
    freq = (
        0, 16.66, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 
        1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 8000, 
        sample_rate/2)
    gain = (
        -85.0, -85.0, -63.0, -41.0, -21.0, -10.6, -6.3, -3.6, -2.0, -0.9, -0.0,
         0.6, 1.0, -0.0, -0.9, -1.7, -2.4, -3.0, -4.2, -5.6, -8.5, -15.0, -25.0,
         -36.0, -43.0, -80, -80)
    gain_lin = 10**(np.array(gain)/20.0)
    gain_lin[0] = 0
    gain_lin[-1] = 0
    coeffs = scipy.signal.firwin2(1001, freq, gain_lin, fs = sample_rate, antisymmetric = True)
    state = scipy.signal.lfilter_zi(coeffs, [1])
    return coeffs, state


def notch_filter(sample_rate, notch_frequency, bandwidth = 50, n_coeffs = 16):
    # Notch filter coefficients for removing the test tone
    coeffs = scipy.signal.iirfilter(
        n_coeffs,
        [notch_frequency - bandwidth/2, notch_frequency + bandwidth/2],
        fs = sample_rate,
        rp = 0.1,
        rs = 150,
        btype = 'bandstop',
        ftype = 'ellip',
        output = 'sos')
    state = scipy.signal.sosfilt_zi(coeffs)
    return coeffs, state    


def calculate_SINAD(sample_rate, samples, bpf_coeffs, bpf_state, notch_filter_coeffs, notch_filter_state):
    # Calculate SINAD for a block of samples

    # Band-pass filter the audio, if a filter was provided
    if bpf_coeffs is None:
        bpf_samples = samples
    else:
        bpf_samples, bpf_state = scipy.signal.lfilter(bpf_coeffs, [1], samples, zi = bpf_state)

    # Calculate the signal + noise + distortion power, in dB
    sind_power_dB = 20 * np.log10(np.sqrt(np.mean(bpf_samples**2)))

    # Notch out the tone
    notched_samples, notch_filter_state = scipy.signal.sosfilt(notch_filter_coeffs, bpf_samples, zi = notch_filter_state)

    # Calculate the noise + distortion power, in dB
    nd_power_dB = 20 * np.log10(np.sqrt(np.mean(notched_samples**2)))

    # SINAD definition:
    SINAD = sind_power_dB - nd_power_dB

    # Return SINAD and filter state to facilitate continuity of processing
    return SINAD, bpf_state, notch_filter_state
