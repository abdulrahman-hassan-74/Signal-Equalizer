import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram
import pywt


def compute_fft(signal, sample_rate):
    """
    Compute high-resolution FFT.
    Returns list of {frequency, magnitude} dicts — up to 4000 points
    using peak-preserving downsampling so spikes are never missed.
    """
    n           = len(signal)
    fft_result  = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    magnitudes  = np.abs(fft_result) / np.sqrt(n)   # sqrt(n) keeps magnitudes visible

    # Limit to audible range 0–20 kHz
    max_freq = min(20000, sample_rate // 2)
    mask     = frequencies <= max_freq
    freqs    = frequencies[mask]
    mags     = magnitudes[mask]

    total = len(freqs)
    MAX_POINTS = 4000   # enough for a sharp chart without slowing browser

    if total <= MAX_POINTS:
        return [{"frequency": float(f), "magnitude": float(m)}
                for f, m in zip(freqs, mags)]

    # Peak-preserving downsample: within each window keep the bin with max magnitude
    # This ensures narrow spikes (single tones) are never skipped
    step   = total // MAX_POINTS
    result = []
    for i in range(0, total, step):
        end  = min(i + step, total)
        peak = int(np.argmax(mags[i:end])) + i
        result.append({"frequency": float(freqs[peak]),
                        "magnitude": float(mags[peak])})
    return result


def compute_ifft(fft_complex, n_samples):
    """Convert FFT back to time-domain signal."""
    return np.fft.irfft(fft_complex, n=n_samples)


def compute_spectrogram(signal, sample_rate):
    """
    Compute 2D spectrogram normalized to 0–1.
    Returns 2D list [freq_bins][time_bins].
    """
    _, _, Sxx = scipy_spectrogram(signal, fs=sample_rate, nperseg=256)
    Sxx = np.log10(Sxx + 1e-10)
    Sxx = Sxx - Sxx.min()
    Sxx = Sxx / (Sxx.max() + 1e-9)
    return Sxx.tolist()


def apply_wavelet(signal, wavelet_name):
     """Decompose signal into wavelet coefficients."""
     return pywt.wavedec(signal, wavelet_name)


def inverse_wavelet(coefficients, wavelet_name):
    """Reconstruct signal from wavelet coefficients."""
    return pywt.waverec(coefficients, wavelet_name)