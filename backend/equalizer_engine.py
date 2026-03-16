import numpy as np
import pywt
from signal_processor import compute_ifft

# Pre-defined optimal wavelet per custom mode — NOT user-selectable
OPTIMAL_WAVELETS = {
    "instruments": "db6",   # Daubechies 6 — tonal sustained audio
    "animals":     "db4",   # Daubechies 4 — short transient sounds
    "voices":      "haar",  # Haar — assigned by instructor
    "ecg":         "db4",   # Daubechies 4 — biomedical standard
}


def _apply_gain_fourier(signal, sample_rate, freq_ranges, gain):
    """System A: apply gain to frequency ranges using FFT."""
    n           = len(signal)
    fft_result  = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    for i, freq in enumerate(frequencies):
        for (min_f, max_f) in freq_ranges:
            if min_f <= freq <= max_f:
                fft_result[i] *= gain
                break
    return compute_ifft(fft_result, n)


def apply_wavelet_gains(signal, sample_rate, bands_with_gains, wavelet):
    """
    System B: apply per-band gains using wavelet decomposition.
    bands_with_gains: [{'id':1, 'freq_ranges':[[f1,f2]], 'gain':1.5}, ...]
    """
    coeffs   = pywt.wavedec(signal, wavelet)
    n_levels = len(coeffs)
    for band in bands_with_gains:
        gain        = float(band["gain"])
        freq_ranges = band["freq_ranges"]
        for level_idx in range(1, n_levels):
            level_max = sample_rate / (2 ** level_idx)
            level_min = sample_rate / (2 ** (level_idx + 1))
            for (min_f, max_f) in freq_ranges:
                if level_min < max_f and level_max > min_f:
                    coeffs[level_idx] = coeffs[level_idx] * gain
                    break
    return pywt.waverec(coeffs, wavelet)[:len(signal)]


def get_wavelet_band_energies(signal, sample_rate, bands, wavelet):
    """
    Compute RMS energy per band in wavelet domain.
    Returns: [{'band_id':1, 'name':'Violin', 'energy':0.043}, ...]
    """
    coeffs   = pywt.wavedec(signal, wavelet)
    n_levels = len(coeffs)
    result   = []
    for band in bands:
        freq_ranges  = band.get("freq_ranges", [])
        name         = band.get("name", band.get("label", f'Band {band["id"]}'))
        total_energy = 0.0
        for level_idx in range(1, n_levels):
            level_max = sample_rate / (2 ** level_idx)
            level_min = sample_rate / (2 ** (level_idx + 1))
            for (min_f, max_f) in freq_ranges:
                if level_min < max_f and level_max > min_f:
                    total_energy += float(np.sqrt(np.mean(coeffs[level_idx] ** 2)))
                    break
        result.append({"band_id": band["id"], "name": name,
                        "energy": round(total_energy, 6)})
    return result


def apply_gain(signal, sample_rate, freq_ranges, gain,
               method="fourier", wavelet="db4"):
    """Backward-compatible single-band interface."""
    if method == "wavelet":
        return apply_wavelet_gains(
            signal, sample_rate,
            [{"id": 0, "freq_ranges": freq_ranges, "gain": gain}], wavelet)
    return _apply_gain_fourier(signal, sample_rate, freq_ranges, gain)