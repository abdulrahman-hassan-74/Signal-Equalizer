import numpy as np
import pywt
from signal_processor import compute_ifft


def _apply_gain_fourier(signal, sample_rate, freq_ranges, gain):
    """Apply gain to specific frequency ranges using Fourier transform."""
    n           = len(signal)
    fft_result  = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    for i, freq in enumerate(frequencies):
        for (min_f, max_f) in freq_ranges:
            if min_f <= freq <= max_f:
                fft_result[i] *= gain
                break

    return compute_ifft(fft_result, n)


def _apply_gain_wavelet(signal, sample_rate, freq_ranges, gain, wavelet="db4"):
    """Apply gain to specific frequency ranges using wavelet decomposition."""
    coeffs   = pywt.wavedec(signal, wavelet)
    n_levels = len(coeffs)

    for level_idx in range(1, n_levels):
        # Estimate the frequency band this wavelet level covers
        level_max_freq = sample_rate / (2 ** level_idx)
        level_min_freq = sample_rate / (2 ** (level_idx + 1))

        for (min_f, max_f) in freq_ranges:
            if level_min_freq < max_f and level_max_freq > min_f:
                coeffs[level_idx] = coeffs[level_idx] * gain
                break

    reconstructed = pywt.waverec(coeffs, wavelet)
    return reconstructed[:len(signal)]   # ensure same length


def apply_gain(signal, sample_rate, freq_ranges, gain,
               method="fourier", wavelet="db4"):
    """
    Apply a gain multiplier to specific frequency ranges of a signal.

    Parameters:
        signal      : numpy array of audio samples
        sample_rate : integer (e.g. 44100)
        freq_ranges : list of [min_hz, max_hz] pairs
                      e.g. [[200, 800], [2000, 4000]]
        gain        : float from 0.0 (silence) to 2.0 (double volume)
        method      : 'fourier' or 'wavelet'
        wavelet     : wavelet name e.g. 'db4', 'sym5', 'morlet'

    Returns:
        Modified signal as numpy array
    """
    if method == "wavelet":
        return _apply_gain_wavelet(signal, sample_rate, freq_ranges,
                                   gain, wavelet)
    return _apply_gain_fourier(signal, sample_rate, freq_ranges, gain)