import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram
import pywt


def compute_fft(signal, sample_rate):
    n           = len(signal)
    fft_result  = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    magnitudes  = np.abs(fft_result) / n

    # Limit to audible range 0–20kHz
    max_freq = min(20000, sample_rate // 2)
    mask     = frequencies <= max_freq
    freqs    = frequencies[mask]
    mags     = magnitudes[mask]

    # Smart downsample: keep max 2000 points but preserve peaks
    total = len(freqs)
    if total <= 2000:
        step = 1
    else:
        step = total // 2000

    result = []
    for i in range(0, total, step):
        # For each step-window, keep the bin with the highest magnitude
        # This ensures spikes are never skipped
        end   = min(i + step, total)
        peak  = int(np.argmax(mags[i:end])) + i
        result.append({
            "frequency": float(freqs[peak]),
            "magnitude": float(mags[peak])
        })

    return result


def compute_ifft(fft_complex, n_samples):
    """
    Convert FFT back to time-domain signal.
    fft_complex: complex numpy array (output of np.fft.rfft)
    n_samples:   original signal length
    """
    return np.fft.irfft(fft_complex, n=n_samples)


def compute_spectrogram(signal, sample_rate):
    """
    Compute a 2D spectrogram of the signal.
    Returns a 2D list [frequency_bins][time_bins] with values 0.0 to 1.0.
    """
    _, _, Sxx = scipy_spectrogram(signal, fs=sample_rate, nperseg=256)

    Sxx = np.log10(Sxx + 1e-10)  # log10 instead of log1p
    Sxx = Sxx - Sxx.min()  # shift so min = 0
    Sxx = Sxx / (Sxx.max() + 1e-9)  # normalize to 0-1

    return Sxx.tolist()



def apply_wavelet(signal, wavelet_name):
    """
    Decomposes a signal into wavelet coefficients.
    signal:       numpy array of audio samples
    wavelet_name: string like 'db4', 'sym5', 'morlet'
    Returns:      list of coefficient arrays (one per frequency level)
    """
    coefficients = pywt.wavedec(signal, wavelet_name)
    return coefficients


def inverse_wavelet(coefficients, wavelet_name):
    """
    Reconstructs a signal from wavelet coefficients.
    coefficients: list of arrays (output of apply_wavelet)
    wavelet_name: must be the same wavelet used in apply_wavelet
    Returns:      numpy array of audio samples
    """
    reconstructed = pywt.waverec(coefficients, wavelet_name)
    return reconstructed