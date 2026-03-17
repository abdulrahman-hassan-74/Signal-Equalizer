import numpy as np
import pywt
from signal_processor import compute_ifft

# Pre-defined optimal wavelet per custom mode
OPTIMAL_WAVELETS = {
    "instruments": "db6",   # Daubechies 6 — tonal sustained audio
    "animals":     "db4",   # Daubechies 4 — short transient sounds
    "voices":      "haar",  # Haar — assigned by instructor
    "ecg":         "db4",   # Daubechies 4 — biomedical standard
}


def _dwt_level_freq_map(sample_rate, n_levels):
    """
    Return (f_lo, f_hi) for every coeffs index produced by pywt.wavedec.

    pywt.wavedec returns:
      coeffs[0]      = approximation cA_L  →  [0,           sr/2^L     ]
      coeffs[1]      = detail cD_L         →  [sr/2^(L+1),  sr/2^L     ]  coarsest
      coeffs[2]      = detail cD_(L-1)     →  [sr/2^L,      sr/2^(L-1) ]
      ...
      coeffs[L]      = detail cD_1         →  [sr/4,        sr/2        ]  finest
    """
    freq_map = []
    # Index 0 — approximation
    freq_map.append((0.0, sample_rate / (2 ** n_levels)))
    # Indices 1..n_levels — detail bands (coarse → fine)
    for i in range(1, n_levels + 1):
        p    = n_levels - i + 1              # pywt level: 1=finest, L=coarsest
        f_lo = sample_rate / (2 ** (p + 1))
        f_hi = sample_rate / (2 ** p)
        freq_map.append((f_lo, f_hi))
    return freq_map   # len == n_levels + 1 == len(coeffs)


def _overlap_hz(band_ranges, lev_lo, lev_hi):
    """Total Hz overlap between a band's freq_ranges and a DWT level [lev_lo, lev_hi]."""
    total = 0.0
    for (min_f, max_f) in band_ranges:
        lo = max(min_f, lev_lo)
        hi = min(max_f, lev_hi)
        if hi > lo:
            total += hi - lo
    return total


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
    System B — correct DWT equalization with exclusive level assignment.

    Algorithm
    ---------
    1.  Decompose:  coeffs = pywt.wavedec(signal, wavelet)
    2.  Map each coeffs index to its (f_lo, f_hi) via _dwt_level_freq_map.
    3.  For each DWT level, find the band with the GREATEST Hz overlap.
        That band exclusively owns the level → no overwriting between bands.
    4.  Scale all coeffs in one pass (including approximation index 0).
    5.  Reconstruct: pywt.waverec(coeffs, wavelet).

    Exclusive assignment prevents e.g. Bass=0 from silencing a DWT level
    that belongs primarily to Kick=1.0.
    """
    coeffs   = pywt.wavedec(signal, wavelet)
    n_levels = len(coeffs) - 1
    freq_map = _dwt_level_freq_map(sample_rate, n_levels)

    # Default: identity gain on every level
    level_gains = [1.0] * len(coeffs)

    for coeff_idx, (lev_lo, lev_hi) in enumerate(freq_map):
        best_overlap = 0.0
        best_gain    = 1.0           # no band owns it → keep as-is
        for band in bands_with_gains:
            ov = _overlap_hz(band["freq_ranges"], lev_lo, lev_hi)
            if ov > best_overlap:
                best_overlap = ov
                best_gain    = float(band["gain"])
        level_gains[coeff_idx] = best_gain

    # Apply in a single pass — ALL levels including approximation (idx 0)
    for idx in range(len(coeffs)):
        g = level_gains[idx]
        if abs(g - 1.0) > 1e-9:
            coeffs[idx] = coeffs[idx] * g

    return pywt.waverec(coeffs, wavelet)[:len(signal)]


def get_wavelet_band_energies(signal, sample_rate, bands, wavelet):
    """
    Compute RMS energy per user-band in the wavelet domain using
    the same exclusive-assignment rule as apply_wavelet_gains.

    Returns list of dicts: band_id, name, energy, dwt_levels.
    dwt_levels = [(f_lo, f_hi)] of DWT levels exclusively owned by this band.
    """
    coeffs   = pywt.wavedec(signal, wavelet)
    n_levels = len(coeffs) - 1
    freq_map = _dwt_level_freq_map(sample_rate, n_levels)

    # Build exclusive ownership map: coeff_idx → band index
    owner = {}   # coeff_idx → index into bands list
    for coeff_idx, (lev_lo, lev_hi) in enumerate(freq_map):
        best_overlap = 0.0
        best_band    = None
        for band_idx, band in enumerate(bands):
            ov = _overlap_hz(band.get("freq_ranges", []), lev_lo, lev_hi)
            if ov > best_overlap:
                best_overlap = ov
                best_band    = band_idx
        if best_band is not None:
            owner[coeff_idx] = best_band

    # Accumulate energy per band
    energy_map  = {i: 0.0  for i in range(len(bands))}
    levels_map  = {i: []   for i in range(len(bands))}
    for coeff_idx, band_idx in owner.items():
        lev_lo, lev_hi = freq_map[coeff_idx]
        rms = float(np.sqrt(np.mean(coeffs[coeff_idx] ** 2)))
        energy_map[band_idx] += rms
        levels_map[band_idx].append((round(lev_lo, 1), round(lev_hi, 1)))

    result = []
    for i, band in enumerate(bands):
        name = band.get("name") or band.get("label") or f'Band {band["id"]}'
        result.append({
            "band_id":    band["id"],
            "name":       name,
            "energy":     round(energy_map[i], 6),
            "dwt_levels": levels_map[i],
        })
    return result


def apply_gain(signal, sample_rate, freq_ranges, gain,
               method="fourier", wavelet="db4"):
    """Backward-compatible single-band interface."""
    if method == "wavelet":
        return apply_wavelet_gains(
            signal, sample_rate,
            [{"id": 0, "freq_ranges": freq_ranges, "gain": gain}], wavelet)
    return _apply_gain_fourier(signal, sample_rate, freq_ranges, gain)