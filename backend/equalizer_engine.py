"""
equalizer_engine.py — System B: Wavelet Packet Decomposition Equalizer
=======================================================================
Replaces the broken standard-DWT approach with Wavelet Packet Decomposition
(WPD), which produces a BALANCED binary tree so energy is distributed
uniformly across all user-defined bands.

Root cause of the original bug
-------------------------------
Standard DWT (pywt.wavedec) is an octave-band filter:
  - The approximation coefficients at the deepest level accumulate ALL
    low-frequency energy through repeated low-pass iterations.
  - Every band's freq_ranges overlapped the approximation more than the
    detail levels, so band 0 always "won" and the others received gain 0
    → silence / no equalization.

WPD fix
-------
WPD decomposes BOTH approximation AND detail sub-bands recursively, producing
2^L leaf nodes at level L.  Each node covers exactly sr / 2^(L+1) Hz, giving
uniform frequency resolution.  Energy is spread correctly across all bands.

Optimal wavelets (research-backed)
------------------------------------
  instruments : db6   — smooth, 12-sample support, good for sustained tonal
  animals     : db4   — 8-sample support, captures short transient bio calls
  voices      : db4   — speech has both transient & quasi-periodic components
  ecg         : db4   — Daubechies 4 is the de-facto ECG wavelet (literature)
"""

import numpy as np
import pywt
from signal_processor import compute_ifft

# ── Optimal wavelet per mode ─────────────────────────────────────────────────
OPTIMAL_WAVELETS = {
    "instruments": "db6",
    "animals":     "db4",
    "voices":      "db4",
    "ecg":         "db4",
}

# Default: aim for 2^4 = 16 leaf nodes → 4 nodes per slider (4-slider setup)
_DEFAULT_WPD_LEVEL = 4


# ══════════════════════════════════════════════════════════════════════════════
#  CORE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _wpd_node_freq_range(path: str, sample_rate: int) -> tuple:
    """
    Return (f_lo, f_hi) for a WPD node identified by its path string.

    Algorithm — start with the full [0, Nyquist] band, then for each
    character in the path:
        'a' (approximation / low-pass)  → hi = mid   (lower half)
        'd' (detail       / high-pass)  → lo = mid   (upper half)

    This is the idealised brick-wall model; it holds to first order for
    all compactly-supported orthogonal wavelets and is the standard mapping
    used in WPD-based filterbank analysis.
    """
    lo, hi = 0.0, float(sample_rate) / 2.0
    for ch in path:
        mid = (lo + hi) * 0.5
        if ch == 'a':
            hi = mid
        else:          # 'd'
            lo = mid
    return lo, hi


def _overlap_hz(band_ranges: list, f_lo: float, f_hi: float) -> float:
    """Total Hz overlap between a user band's freq_ranges and a node [f_lo, f_hi]."""
    total = 0.0
    for (min_f, max_f) in band_ranges:
        lo = max(min_f, f_lo)
        hi = min(max_f, f_hi)
        if hi > lo:
            total += hi - lo
    return total


def _choose_wpd_level(signal_len: int, wavelet: str,
                      sample_rate: int = None, bands: list = None,
                      target_subbands: int = 16) -> int:
    """
    Select WPD decomposition level L, capped by signal length / wavelet filter.

    Default behaviour
    -----------------
    Aim for 2^L ≥ target_subbands (default 16) leaf nodes.

    Adaptive mode (when sample_rate + bands are supplied)
    ------------------------------------------------------
    Narrow frequency bands need finer WPD resolution.  The necessary
    condition is:

        node_bandwidth  ≤  narrowest_band_span
        (nyq / 2^L)     ≤  min_span
        L               ≥  log2(nyq / min_span)

    This ensures every band can exclusively own at least one leaf node
    (the node whose centre overlaps it more than any adjacent band).

    Example: Bass [20–300 Hz], SR=44 100 Hz
        min_span = 280 Hz, nyq = 22 050 Hz
        L_needed = ceil(log2(22 050 / 280)) = ceil(6.3) = 7  → 128 nodes @ 172 Hz each
    """
    w     = pywt.Wavelet(wavelet)
    max_l = min(pywt.dwt_max_level(signal_len, w.dec_len), 10)  # hard cap at 10

    desired = int(np.ceil(np.log2(max(target_subbands, 2))))

    if sample_rate is not None and bands is not None:
        nyq      = float(sample_rate) / 2.0
        min_span = float('inf')
        for band in bands:
            for (lo, hi) in band.get("freq_ranges", []):
                if hi > lo:
                    min_span = min(min_span, hi - lo)
        if min_span < float('inf') and min_span > 0 and nyq > 0:
            # Enough resolution so narrowest band dominates its node(s)
            adaptive = int(np.ceil(np.log2(nyq / min_span)))
            desired  = max(desired, adaptive)

    return int(np.clip(desired, 1, max(1, max_l)))


# ══════════════════════════════════════════════════════════════════════════════
#  EXCLUSIVE BAND ASSIGNMENT  (shared by apply + energy functions)
# ══════════════════════════════════════════════════════════════════════════════

def _build_node_ownership(leaf_nodes: list, bands: list,
                          sample_rate: int) -> dict:
    """
    For every WPD leaf node, find the band (from `bands`) with the greatest
    Hz overlap.  Returns {node.path: band_index}.

    Exclusive assignment ensures no band can zero-out a node that belongs
    primarily to another band.
    """
    ownership = {}
    for node in leaf_nodes:
        f_lo, f_hi = _wpd_node_freq_range(node.path, sample_rate)
        best_ov, best_idx = 0.0, None
        for band_idx, band in enumerate(bands):
            ov = _overlap_hz(band.get("freq_ranges", []), f_lo, f_hi)
            if ov > best_ov:
                best_ov  = ov
                best_idx = band_idx
        if best_idx is not None:
            ownership[node.path] = best_idx
    return ownership


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM B — WPD EQUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def apply_wpd_gains(signal: np.ndarray, sample_rate: int,
                    bands_with_gains: list, wavelet: str,
                    level: int = None) -> np.ndarray:
    """
    Apply per-band gains using Wavelet Packet Decomposition.

    Algorithm
    ---------
    1. Choose decomposition level L (default: target 16 sub-bands).
    2. Decompose signal into 2^L leaf nodes via pywt.WaveletPacket.
    3. Map each node to its exact [f_lo, f_hi] with _wpd_node_freq_range.
    4. Exclusive assignment: each node owned by the band with greatest overlap.
    5. Scale node.data by the owning band's gain (skip if gain ≈ 1.0).
    6. Reconstruct with inverse WPD from modified leaf nodes.
    7. Trim / zero-pad to match original signal length.

    Parameters
    ----------
    signal          : 1-D float32/float64 array
    sample_rate     : Hz
    bands_with_gains: list of dicts with keys 'freq_ranges' and 'gain'
    wavelet         : PyWavelets wavelet name (e.g. 'db4', 'haar')
    level           : WPD decomposition level; auto-selected if None
    """
    n      = len(signal)
    sig64  = signal.astype(np.float64)

    if level is None:
        level = _choose_wpd_level(n, wavelet, sample_rate, bands_with_gains)

    # ── Step 1: Decompose ────────────────────────────────────────────────────
    wp         = pywt.WaveletPacket(data=sig64, wavelet=wavelet, mode='per')
    leaf_nodes = wp.get_level(level, 'natural')

    # ── Step 2: Exclusive band ownership ────────────────────────────────────
    ownership  = _build_node_ownership(leaf_nodes, bands_with_gains, sample_rate)

    # ── Step 3: Apply gains to node coefficients ─────────────────────────────
    for node in leaf_nodes:
        band_idx = ownership.get(node.path)
        if band_idx is None:
            continue
        gain = float(bands_with_gains[band_idx].get("gain", 1.0))
        if abs(gain - 1.0) > 1e-9:
            node.data = node.data * gain

    # ── Step 4: Reconstruct from modified leaf nodes ─────────────────────────
    # Populate a fresh WP tree from only the leaf nodes so PyWavelets
    # reconstructs bottom-up without touching stale ancestor cache.
    new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='per')
    for node in leaf_nodes:
        new_wp[node.path] = node.data

    result = new_wp.reconstruct(update=False)

    # ── Step 5: Match original length (periodic padding may extend slightly) ─
    if len(result) >= n:
        result = result[:n]
    else:
        result = np.concatenate([result, np.zeros(n - len(result))])

    return result.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  WPD ENERGY ANALYSIS  (for visualisation)
# ══════════════════════════════════════════════════════════════════════════════

def get_wpd_band_energies(signal: np.ndarray, sample_rate: int,
                          bands: list, wavelet: str,
                          level: int = None) -> list:
    """
    Compute RMS energy per user-defined band using WPD with exclusive
    node assignment.

    Returns a list of dicts, one per band:

        {
          "band_id"      : int,
          "name"         : str,
          "energy"       : float,          # total RMS sum over owned nodes
          "wpd_nodes"    : [(f_lo, f_hi)], # frequency intervals of owned nodes
          "node_energies": [float, ...],   # per-node RMS (parallel to wpd_nodes)
          "wpd_level"    : int,            # decomposition level used
        }

    The per-node data enables the frontend to draw a frequency-ordered
    energy spectrum rather than a simple bar chart.
    """
    n      = len(signal)
    sig64  = signal.astype(np.float64)

    if level is None:
        level = _choose_wpd_level(n, wavelet, sample_rate, bands)

    # Decompose
    wp         = pywt.WaveletPacket(data=sig64, wavelet=wavelet, mode='per')
    leaf_nodes = wp.get_level(level, 'natural')

    # Exclusive ownership
    ownership  = _build_node_ownership(leaf_nodes, bands, sample_rate)

    # Accumulate per band
    energy_map      = {i: 0.0 for i in range(len(bands))}
    node_ranges_map = {i: []  for i in range(len(bands))}
    node_rms_map    = {i: []  for i in range(len(bands))}

    for node in leaf_nodes:
        band_idx = ownership.get(node.path)
        if band_idx is None:
            continue
        f_lo, f_hi = _wpd_node_freq_range(node.path, sample_rate)
        rms = float(np.sqrt(np.mean(node.data ** 2)))
        energy_map[band_idx]      += rms
        node_ranges_map[band_idx].append((round(f_lo, 1), round(f_hi, 1)))
        node_rms_map[band_idx].append(round(rms, 7))

    # Build result list
    result = []
    for i, band in enumerate(bands):
        name = (band.get("name") or band.get("label") or f'Band {band["id"]}')
        result.append({
            "band_id":       band["id"],
            "name":          name,
            "energy":        round(energy_map[i], 6),
            "wpd_nodes":     node_ranges_map[i],   # replaces old dwt_levels
            "node_energies": node_rms_map[i],       # NEW: per-node RMS for spectrum plot
            "wpd_level":     level,
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM A — FOURIER (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_gain_fourier(signal: np.ndarray, sample_rate: int,
                        freq_ranges: list, gain: float) -> np.ndarray:
    """System A: apply a scalar gain to a set of frequency ranges via FFT."""
    n           = len(signal)
    fft_result  = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    for i, freq in enumerate(frequencies):
        for (min_f, max_f) in freq_ranges:
            if min_f <= freq <= max_f:
                fft_result[i] *= gain
                break
    return compute_ifft(fft_result, n)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKWARD-COMPATIBLE WRAPPERS  (main.py imports these names unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def apply_wavelet_gains(signal, sample_rate, bands_with_gains, wavelet):
    """Delegates to the new WPD-based apply_wpd_gains."""
    return apply_wpd_gains(signal, sample_rate, bands_with_gains, wavelet)


def get_wavelet_band_energies(signal, sample_rate, bands, wavelet):
    """
    Delegates to get_wpd_band_energies.
    Returns the same outer shape as before but with 'wpd_nodes' /
    'node_energies' / 'wpd_level' instead of the old 'dwt_levels'.
    """
    return get_wpd_band_energies(signal, sample_rate, bands, wavelet)


def apply_gain(signal, sample_rate, freq_ranges, gain,
               method="fourier", wavelet="db4", mode="generic"):
    """Backward-compatible single-band interface."""
    if method == "wavelet":
        return apply_wavelet_gains(
            signal, sample_rate,
            [{"id": 0, "freq_ranges": freq_ranges, "gain": gain}], wavelet)
    if mode in ("voices", "instruments"):
        return _apply_gain_windowed(
            signal, sample_rate,
            [{"freq_ranges": freq_ranges, "gain": gain}],
            window_type="gaussian")
    return _apply_gain_fourier(signal, sample_rate, freq_ranges, gain)


def _apply_gain_windowed(signal, sample_rate, bands_with_gains, window_type="gaussian"):
    """
    System A enhanced: smooth Gaussian windowed gains.
    Used for voices and instruments modes to avoid sharp cutoff artifacts.
    """
    n             = len(signal)
    fft_result    = np.fft.rfft(signal)
    frequencies   = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    gain_envelope = np.ones(len(frequencies))

    for band in bands_with_gains:
        gain        = float(band["gain"])
        sigma_ratio = band.get("sigma_ratio", 0.5)  # tight = more separation

        for (min_f, max_f) in band["freq_ranges"]:
            f_c    = (min_f + max_f) / 2.0
            sigma  = (max_f - min_f) * sigma_ratio
            window = np.exp(-((frequencies - f_c) ** 2) / (2 * sigma ** 2))
            gain_envelope += window * (gain - 1.0)

    fft_result *= gain_envelope
    return compute_ifft(fft_result, n)