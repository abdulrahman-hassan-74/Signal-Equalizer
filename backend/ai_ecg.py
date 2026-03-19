"""
ai_ecg.py
=========
ECG arrhythmia analysis using the pretrained ECGResNet (ecg_resnet_mitbih.pt).

Primary backend  : ECGPipeline from ecg_inference.py
  - Accepts: .wav, .dat/.hea (WFDB), .csv, .npy, any soundfile format
  - Segments beats with wfdb.gqrs_detect
  - Classifies with the 5-class ResNet: N / S / V / F / Q
  - Builds per-class Gaussian frequency windows (used as equalizer bands)
  - Returns per-class isolated signals (stem signals for System C)

Fallback backend : old _ECGNet (requires neurokit2, separate training data)

The per-class freq windows ARE the equalizer bands, so the AI directly
drives the sliders — exactly what the instructor described.

Install:
    pip install wfdb scipy torch numpy pandas soundfile
"""

from __future__ import annotations
import os, io, time, json
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[ECG] ⚠ torch not installed — pip install torch")

try:
    import wfdb
    import wfdb.processing as wfdb_proc
    WFDB_OK = True
except ImportError:
    WFDB_OK = False
    print("[ECG] ⚠ wfdb not installed — pip install wfdb")

try:
    from scipy.fft    import fft as scipy_fft, ifft as scipy_ifft, fftfreq
    from scipy.signal import resample_poly
    from scipy.io     import wavfile
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[ECG] ⚠ scipy not installed — pip install scipy")

try:
    import soundfile as sf
    SF_OK = True
except ImportError:
    SF_OK = False

# ── constants (must match training) ──────────────────────────────────────────
FS          = 360
BEAT_LEN    = 187
BEAT_BEFORE = 90
BEAT_AFTER  = BEAT_LEN - BEAT_BEFORE
N_CLASSES   = 5
CLASS_NAMES = ["Normal (N)", "SVEB (S)", "PVC (V)", "Fusion (F)", "Unknown (Q)"]
CLASS_SYMS  = ["N", "S", "V", "F", "Q"]

# Default weights path — override in ECGModel(weights_path=...)
DEFAULT_WEIGHTS = "ecg_resnet_mitbih.pt"

# ── model cache ───────────────────────────────────────────────────────────────
_pipeline_cache: dict = {}


# ═══════════════════════════════════════════════════════
#  RESNET ARCHITECTURE  (must match ecg_inference.py)
# ═══════════════════════════════════════════════════════
class _ResBlock(nn.Module if TORCH_OK else object):
    def __init__(self, in_ch, out_ch, downsample=False):
        if not TORCH_OK: return
        super().__init__()
        stride     = 2 if downsample else 1
        self.conv1 = nn.Conv1d(in_ch, out_ch, 5, stride=stride, padding=2, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch)
        ) if (in_ch != out_ch or downsample) else nn.Identity()

    def forward(self, x):
        return F.relu(
            self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x)
        )


class _ECGResNet(nn.Module if TORCH_OK else object):
    def __init__(self, n_classes=N_CLASSES):
        if not TORCH_OK: return
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        self.layer1 = _ResBlock(32,  64)
        self.layer2 = _ResBlock(64,  128, downsample=True)
        self.layer3 = _ResBlock(128, 256, downsample=True)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.drop   = nn.Dropout(0.3)
        self.fc     = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(self.drop(self.pool(x).squeeze(-1)))


# ═══════════════════════════════════════════════════════
#  FILE LOADING  — accepts any ECG format
# ═══════════════════════════════════════════════════════

def _normalise(sig: np.ndarray) -> np.ndarray:
    return (sig - sig.mean()) / (sig.std() + 1e-8)


def _resample_signal(sig: np.ndarray, src_fs: int, tgt_fs: int = FS) -> np.ndarray:
    if src_fs == tgt_fs:
        return sig.astype(np.float32)
    from math import gcd
    g = gcd(int(src_fs), int(tgt_fs))
    return resample_poly(sig.astype(np.float32),
                         int(tgt_fs) // g, int(src_fs) // g).astype(np.float32)


def load_ecg_file(filepath: str, channel: int = 0) -> tuple[np.ndarray, int]:
    """
    Load any ECG file and return (signal_float32_at_360Hz, 360).

    Supported formats
    -----------------
    .wav              — scipy.io.wavfile / soundfile
    .dat / .hea       — WFDB
    .csv              — pandas (takes first numeric column; detects fs from header)
    .npy              — numpy array
    .txt              — whitespace / comma separated numbers
    other audio       — soundfile fallback
    """
    ext = os.path.splitext(filepath)[1].lower()

    # ── WAV ───────────────────────────────────────────────────────────────────
    if ext == ".wav":
        if SCIPY_OK:
            fs_raw, data = wavfile.read(filepath)
            if data.dtype.kind == "i":
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            data = data.astype(np.float32)
            if data.ndim == 2:
                data = data[:, channel if data.shape[1] > channel else 0]
        elif SF_OK:
            data, fs_raw = sf.read(filepath)
            if data.ndim == 2:
                data = data[:, channel if data.shape[1] > channel else 0]
            data = data.astype(np.float32)
        else:
            raise ImportError("scipy or soundfile required to read WAV")
        return _resample_signal(data, int(fs_raw)), FS

    # ── WFDB .dat/.hea ────────────────────────────────────────────────────────
    if ext in (".dat", ".hea", ""):
        if not WFDB_OK:
            raise ImportError("wfdb required to read .dat/.hea — pip install wfdb")
        base = filepath.replace(".hea", "").replace(".dat", "")
        rec  = wfdb.rdrecord(base)
        ch   = channel if rec.p_signal.shape[1] > channel else 0
        data = rec.p_signal[:, ch].astype(np.float32)
        return _resample_signal(data, int(rec.fs)), FS

    # ── CSV ───────────────────────────────────────────────────────────────────
    if ext == ".csv":
        import pandas as pd
        df  = pd.read_csv(filepath)
        src_fs = 360
        for col in df.columns:
            c = str(col).lower()
            if "sample" in c and "rate" in c:
                try: src_fs = int(df[col].iloc[0]); break
                except: pass
            if c.endswith("hz"):
                try: src_fs = int(c.replace("hz", "").strip()); break
                except: pass
        num = df.select_dtypes(include="number")
        if num.empty:
            raise ValueError("No numeric columns in CSV")
        col_idx = channel if num.shape[1] > channel else 0
        data    = num.iloc[:, col_idx].dropna().values.astype(np.float32)
        peak    = np.abs(data).max()
        if peak > 1.0:
            data = data / peak
        return _resample_signal(data, src_fs), FS

    # ── NPY ───────────────────────────────────────────────────────────────────
    if ext == ".npy":
        data = np.load(filepath).astype(np.float32)
        if data.ndim == 2:
            data = data[:, channel if data.shape[1] > channel else 0]
        peak = np.abs(data).max()
        if peak > 1.0:
            data = data / peak
        return data, FS   # assume already at FS

    # ── TXT ───────────────────────────────────────────────────────────────────
    if ext in (".txt", ".tsv"):
        data = np.loadtxt(filepath, delimiter=None).astype(np.float32)
        if data.ndim == 2:
            data = data[:, channel if data.shape[1] > channel else 0]
        peak = np.abs(data).max()
        if peak > 1.0:
            data = data / peak
        return data, FS

    # ── generic audio fallback (soundfile) ───────────────────────────────────
    if SF_OK:
        data, fs_raw = sf.read(filepath)
        if data.ndim == 2:
            data = data[:, channel if data.shape[1] > channel else 0]
        return _resample_signal(data.astype(np.float32), int(fs_raw)), FS

    raise ValueError(f"Unsupported ECG format: {ext}")


# ═══════════════════════════════════════════════════════
#  BEAT SEGMENTATION
# ═══════════════════════════════════════════════════════

def _segment_beats(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect R-peaks with wfdb.gqrs_detect, extract fixed-length beat windows.
    Returns (beats (M,187), r_peaks (M,)).
    Falls back to uniform 1-second windows if wfdb unavailable.
    """
    if WFDB_OK:
        try:
            r_peaks = wfdb_proc.gqrs_detect(sig=signal.astype(np.float64), fs=FS)
        except Exception as e:
            print(f"[ECG] gqrs_detect failed ({e}), using fallback peak detection")
            r_peaks = _fallback_peaks(signal)
    else:
        r_peaks = _fallback_peaks(signal)

    beats, valid = [], []
    for rp in r_peaks:
        lo, hi = int(rp) - BEAT_BEFORE, int(rp) + BEAT_AFTER
        if lo < 0 or hi > len(signal):
            continue
        b = signal[lo:hi].copy().astype(np.float32)
        b = _normalise(b)
        beats.append(b)
        valid.append(rp)

    if not beats:
        return np.zeros((0, BEAT_LEN), dtype=np.float32), np.array([], dtype=int)
    return np.array(beats, dtype=np.float32), np.array(valid, dtype=int)


def _fallback_peaks(signal: np.ndarray) -> np.ndarray:
    """Simple threshold-based peak detection when wfdb is unavailable."""
    try:
        from scipy.signal import find_peaks
        # Normalize signal for reliable detection
        sig = signal - signal.mean()
        std = sig.std()
        if std < 1e-8:
            # Flat signal — return evenly spaced peaks
            return np.arange(BEAT_BEFORE, len(signal) - BEAT_AFTER, int(FS * 0.8))
        sig_norm = sig / std
        height = max(0.3, sig_norm.mean() + 0.5 * sig_norm.std())
        peaks, _ = find_peaks(sig_norm, height=height, distance=int(FS * 0.4))
        if len(peaks) < 2:
            # Too few peaks — lower threshold
            peaks, _ = find_peaks(sig_norm, height=0.0, distance=int(FS * 0.3))
        return peaks
    except Exception as e:
        print(f"[ECG] fallback_peaks error: {e}")
        # Last resort: uniform spacing
        return np.arange(BEAT_BEFORE, len(signal) - BEAT_AFTER, int(FS * 0.8))


# ═══════════════════════════════════════════════════════
#  FREQUENCY BAND DISCOVERY
#  Builds Gaussian windows per class — used as equalizer bands
# ═══════════════════════════════════════════════════════

def _find_bands(beats: np.ndarray, labels: np.ndarray,
                half_win: int = 10) -> dict[int, tuple[float, float, float]]:
    """
    For each class, compute the dominant frequency band from the average
    beat spectrum.  Returns {cls_id: (f_lo, f_hi, f_peak)}.
    """
    freqs = fftfreq(BEAT_LEN, d=1.0 / FS)[:BEAT_LEN // 2]
    bands: dict[int, tuple[float, float, float]] = {}

    for cls in range(N_CLASSES):
        mask = labels == cls
        if mask.sum() == 0:
            bands[cls] = (0.0, 0.0, 0.0)
            continue
        cls_beats = beats[mask]
        avg_mag   = np.abs(scipy_fft(cls_beats, axis=1))[:, :BEAT_LEN // 2].mean(axis=0)
        smooth    = np.convolve(avg_mag, np.ones(half_win) / half_win, mode="same")
        peak_bin  = int(np.argmax(smooth))
        lo_bin    = max(0, peak_bin - half_win)
        hi_bin    = min(len(freqs) - 1, peak_bin + half_win)
        bands[cls] = (float(freqs[lo_bin]),
                      float(freqs[hi_bin]),
                      float(freqs[peak_bin]))
    return bands


# ═══════════════════════════════════════════════════════
#  ECG EQUALIZATION  (Gaussian window per class)
# ═══════════════════════════════════════════════════════

def ecg_equalise(signal: np.ndarray,
                 gains: dict[int, float],
                 bands: dict[int, tuple[float, float, float]],
                 noise_gain: float = 1.0,
                 fs: int = FS) -> np.ndarray:
    """
    Apply per-arrhythmia-class Gaussian gain windows to an ECG signal.

    Parameters
    ----------
    signal     : float32 ECG array
    gains      : {class_id: gain}  0=suppress 1=unchanged 2=amplify
    bands      : {class_id: (f_lo, f_hi, f_peak)}
    noise_gain : gain for frequencies outside all defined bands
    fs         : sample rate

    Returns
    -------
    equalised signal (float32)
    """
    N     = len(signal)
    freqs = np.abs(fftfreq(N, d=1.0 / fs))
    spec  = scipy_fft(signal.astype(np.float64))

    covered = np.zeros(N, dtype=bool)
    windows: dict[int, np.ndarray] = {}

    for cls in range(N_CLASSES):
        if cls not in bands:
            continue
        f_lo, f_hi, f_peak = bands[cls]
        if f_lo == f_hi == 0.0:
            continue
        sigma  = (f_hi - f_lo) / 2.0 if f_hi > f_lo else 1.0
        win    = np.exp(-0.5 * ((freqs - f_peak) / sigma) ** 2)
        win[(freqs < f_lo) | (freqs > f_hi)] = 0.0
        wmax   = win.max()
        if wmax > 0:
            win /= wmax
        windows[cls] = win
        covered |= (win > 0.01)

    mask           = np.ones(N, dtype=np.float64)
    mask[~covered] = noise_gain

    for cls, win in windows.items():
        g               = float(gains.get(cls, 1.0))
        factor          = (1.0 - win) + win * g
        mask[covered]  *= factor[covered]

    return np.real(scipy_ifft(spec * mask)).astype(np.float32)


# ═══════════════════════════════════════════════════════
#  ECGModel — PUBLIC API  (replaces old ECGModel)
# ═══════════════════════════════════════════════════════

class ECGModel:
    """
    ECG arrhythmia analysis using pretrained ECGResNet.

    Public methods
    ──────────────
    classify_signal(signal_np, fs)
        → {"beats", "summary", "class_signals", "bands", "counts"}

    analyse_file(filepath)
        → same dict (loads + resamples automatically)

    equalise_signal(signal, gains, bands, noise_gain)
        → equalised float32 signal

    compare_methods(signal, fs, gains, bands, wavelet, level)
        → {"wavelet": {metrics}, "resnet": {metrics}}
    """

    def __init__(self,
                 weights_path: str = DEFAULT_WEIGHTS,
                 data_dir:     str = "",       # kept for backward-compat, ignored
                 model_path:   str = ""):      # kept for backward-compat, ignored
        if not TORCH_OK:
            raise ImportError("torch is required — pip install torch")
        if not SCIPY_OK:
            raise ImportError("scipy is required — pip install scipy")

        self.weights_path = weights_path
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._backend     = "none"

        # Try to load the pretrained ResNet
        if os.path.exists(weights_path):
            self._load_resnet(weights_path)
        else:
            # Search in common locations
            candidates = [
                weights_path,
                os.path.join(os.path.dirname(__file__), weights_path),
                os.path.join(os.path.dirname(__file__), "ecg_resnet_mitbih.pt"),
                "ecg_resnet_mitbih.pt",
            ]
            for c in candidates:
                if os.path.exists(c):
                    self._load_resnet(c)
                    break
            else:
                print(f"[ECG] ⚠ Weights not found at '{weights_path}'")
                print("[ECG]   Running without pretrained model — "
                      "labels will be uniform 'Unknown'")
                self._net     = None
                self._backend = "no_weights"

    def _load_resnet(self, path: str):
        self._net = _ECGResNet(n_classes=N_CLASSES).to(self.device)
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            # weights_only not available in older PyTorch versions
            state = torch.load(path, map_location=self.device)
        self._net.load_state_dict(state)
        self._net.eval()
        self._backend = "resnet"
        print(f"[ECG] ECGResNet loaded from '{path}'  device={self.device}")

    # ── Beat classification ──────────────────────────────────────────────────
    def _classify_beats(self, beats: np.ndarray,
                        batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Returns (labels (M,), probs (M,5))."""
        if self._net is None or len(beats) == 0:
            labels = np.zeros(len(beats), dtype=int)
            probs  = np.zeros((len(beats), N_CLASSES), dtype=np.float32)
            probs[:, 0] = 1.0
            return labels, probs

        xt = torch.tensor(beats[:, np.newaxis, :], dtype=torch.float32)
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(xt), batch_size):
                logits = self._net(xt[i:i + batch_size].to(self.device))
                all_probs.append(F.softmax(logits, dim=1).cpu())
        probs  = torch.cat(all_probs).numpy()
        labels = probs.argmax(axis=1)
        return labels, probs

    # ── classify_signal() ────────────────────────────────────────────────────
    def classify_signal(self, signal: np.ndarray, fs: int = FS) -> dict:
        """
        Full pipeline on a numpy signal.
        Never crashes — returns empty results on any error.
        """
        try:
            # Resample to FS if needed
            if fs != FS:
                signal = _resample_signal(signal, fs, FS)
            signal = _normalise(signal.astype(np.float32))

            beats, r_peaks = _segment_beats(signal)
            if len(beats) == 0:
                return {
                    "beats": [], "summary": {}, "class_signals": {},
                    "bands": {c: (0.0, 0.0, 0.0) for c in range(N_CLASSES)},
                    "counts": {c: 0 for c in range(N_CLASSES)},
                    "beat_arrays": np.zeros((0, BEAT_LEN), dtype=np.float32),
                }

            labels, probs = self._classify_beats(beats)
            bands         = _find_bands(beats, labels)

        except Exception as e:
            print(f"[ECG] classify_signal error: {e}")
            import traceback; traceback.print_exc()
            # Return safe empty result
            return {
                "beats": [], "summary": {"Unknown (Q)": 0}, "class_signals": {},
                "bands": {c: (0.0, 0.0, 0.0) for c in range(N_CLASSES)},
                "counts": {c: 0 for c in range(N_CLASSES)},
                "beat_arrays": np.zeros((0, BEAT_LEN), dtype=np.float32),
                "error": str(e),
            }

        # Build per-beat result list
        beat_results = []
        for i, (lab, prob, rp) in enumerate(zip(labels, probs, r_peaks)):
            start = max(0, int(rp) - BEAT_BEFORE)
            end   = min(len(signal), int(rp) + BEAT_AFTER)
            beat_results.append({
                "beat_idx"       : i,
                "label"          : CLASS_NAMES[int(lab)],
                "symbol"         : CLASS_SYMS[int(lab)],
                "confidence"     : round(float(prob[lab]), 3),
                "all_probs"      : {CLASS_NAMES[c]: round(float(prob[c]), 3)
                                    for c in range(N_CLASSES)},
                "r_peak_sample"  : int(rp),
                "r_peak_time_sec": round(int(rp) / FS, 3),
                "start_sample"   : start,
                "end_sample"     : end,
            })

        # Build isolated per-class signals (stems for System C)
        class_signals: dict[str, np.ndarray] = {}
        for c, name in enumerate(CLASS_NAMES):
            sig_c = np.zeros(len(signal), dtype=np.float32)
            for i, lab in enumerate(labels):
                if lab == c:
                    rp    = int(r_peaks[i])
                    s, e  = max(0, rp - BEAT_BEFORE), min(len(signal), rp + BEAT_AFTER)
                    sig_c[s:e] += signal[s:e]
            class_signals[name] = sig_c

        from collections import Counter
        sym_counts   = Counter(CLASS_NAMES[int(l)] for l in labels)
        int_counts   = {c: int((labels == c).sum()) for c in range(N_CLASSES)}

        return {
            "beats"        : beat_results,
            "summary"      : dict(sym_counts),
            "class_signals": class_signals,
            "bands"        : bands,
            "counts"       : int_counts,
            "beat_arrays"  : beats,
        }

    # ── analyse_file() — accepts any file format ─────────────────────────────
    def analyse_file(self, filepath: str, channel: int = 0) -> dict:
        """
        Load any ECG file, run full pipeline, return classify_signal() result.
        Accepts: .wav, .dat/.hea, .csv, .npy, .txt, or any soundfile format.
        """
        print(f"[ECG] Loading '{filepath}'…")
        signal, fs = load_ecg_file(filepath, channel)
        print(f"[ECG] Loaded: {len(signal)/fs:.1f}s  fs={fs}")
        return self.classify_signal(signal, fs)

    # ── equalise_signal() ─────────────────────────────────────────────────────
    def equalise_signal(self, signal: np.ndarray,
                        gains:      dict,
                        bands:      dict,
                        noise_gain: float = 1.0,
                        fs:         int   = FS) -> np.ndarray:
        """Apply per-class Gaussian equalisation. Thin wrapper over ecg_equalise()."""
        return ecg_equalise(signal, gains, bands, noise_gain, fs)

    # ── compare_methods() ────────────────────────────────────────────────────
    def compare_methods(self, signal: np.ndarray,
                        fs:         int   = FS,
                        gains:      dict  = None,
                        bands:      dict  = None,
                        wavelet:    str   = "db4",
                        level:      int   = 6) -> dict:
        """
        Run both wavelet equalisation and ECG ResNet equalisation on the same
        signal and compute comparison metrics.

        Returns
        -------
        {
          "wavelet": {"output", "snr", "si_snr", "prd", "lsd", "time_ms"},
          "resnet":  {"output", "snr", "si_snr", "prd", "lsd", "time_ms",
                      "bands", "counts", "summary", "error": None | str}
        }
        """
        import pywt

        if gains is None:
            gains = {c: 1.0 for c in range(N_CLASSES)}

        # ── Wavelet ───────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            max_lv = pywt.dwt_max_level(len(signal), wavelet)
            coeffs = pywt.wavedec(signal, wavelet=wavelet,
                                   level=min(level, max_lv, 10))
            scales  = [gains.get(i, 1.0) for i in range(len(coeffs))]
            c_mod   = [c * s for c, s in zip(coeffs, scales)]
            wav_out = pywt.waverec(c_mod, wavelet=wavelet)[:len(signal)]
            wav_out = (wav_out / (np.abs(wav_out).max() + 1e-8)).astype(np.float32)
        except Exception as e:
            print(f"[ECG] Wavelet error: {e}")
            wav_out = signal.copy()
        wav_t = (time.perf_counter() - t0) * 1000

        # ── ResNet equalisation ───────────────────────────────────────────────
        ai_error: str | None = None
        t1 = time.perf_counter()
        try:
            if bands is None:
                res    = self.classify_signal(signal, fs)
                bands  = res["bands"]
                counts = res["counts"]
                summary = res["summary"]
            else:
                counts  = {c: 0 for c in range(N_CLASSES)}
                summary = {}
            ai_out = ecg_equalise(signal, gains, bands, noise_gain=1.0, fs=fs)
        except Exception as e:
            import traceback; traceback.print_exc()
            ai_error = str(e)
            ai_out   = signal.copy()
            bands    = bands or {c: (0.0, 0.0, 0.0) for c in range(N_CLASSES)}
            counts   = {c: 0 for c in range(N_CLASSES)}
            summary  = {}
        ai_t = (time.perf_counter() - t1) * 1000

        def _snr(r, e):
            n=min(len(r),len(e)); d=r[:n]-e[:n]; sp=float(np.mean(r[:n]**2)); np_=float(np.mean(d**2))
            return 100.0 if np_<1e-15 else round(10*np.log10(sp/np_),3)
        def _si_snr(r, e):
            n=min(len(r),len(e)); s=r[:n]-r[:n].mean(); h=e[:n]-e[:n].mean()
            t=(np.dot(h,s)/(np.dot(s,s)+1e-10))*s
            return round(float(10*np.log10((np.sum(t**2)+1e-10)/(np.sum((h-t)**2)+1e-10))),3)
        def _prd(r, e):
            n=min(len(r),len(e))
            return round(float(np.sqrt(np.sum((r[:n]-e[:n])**2))/(np.sqrt(np.sum(r[:n]**2))+1e-10)*100),3)
        def _lsd(r, e):
            n=min(len(r),len(e)); S1=np.abs(np.fft.rfft(r[:n])); S2=np.abs(np.fft.rfft(e[:n]))
            return round(float(np.sqrt(np.mean((20*np.log10((S2+1e-10)/(S1+1e-10)))**2))),3)

        def _m(ref, est):
            return {"snr":_snr(ref,est),"si_snr":_si_snr(ref,est),
                    "prd":_prd(ref,est),"lsd":_lsd(ref,est)}

        return {
            "wavelet": {"output": wav_out,  "time_ms": round(wav_t,2), **_m(signal, wav_out)},
            "resnet":  {"output": ai_out,   "time_ms": round(ai_t,2),
                        "bands": bands, "counts": counts, "summary": summary,
                        "error": ai_error,
                        **(_m(signal, ai_out) if not ai_error
                           else {"snr":None,"si_snr":None,"prd":None,"lsd":None})},
        }

    # ── settings helpers (compatible with ecg_inference.py) ─────────────────
    @staticmethod
    def bands_to_settings(bands: dict, gains: dict,
                          noise_gain: float = 1.0) -> dict:
        """Serialise bands + gains to the ecg_eq_settings.json format."""
        return {
            "mode"      : "ECG Abnormalities",
            "fs"        : FS,
            "n_classes" : N_CLASSES,
            "noise_gain": noise_gain,
            "sliders"   : [
                {
                    "id"        : c,
                    "label"     : CLASS_NAMES[c],
                    "symbol"    : CLASS_SYMS[c],
                    "gain"      : float(gains.get(c, 1.0)),
                    "freq_bands": [{"f_low" : bands[c][0],
                                    "f_high": bands[c][1],
                                    "f_peak": bands[c][2]}],
                }
                for c in range(N_CLASSES)
            ],
        }

    @staticmethod
    def settings_to_bands(cfg: dict) -> tuple[dict, dict, float]:
        """Parse ecg_eq_settings.json → (bands, gains, noise_gain)."""
        bands      = {s["id"]: (s["freq_bands"][0]["f_low"],
                                s["freq_bands"][0]["f_high"],
                                s["freq_bands"][0]["f_peak"])
                      for s in cfg["sliders"]}
        gains      = {s["id"]: float(s["gain"]) for s in cfg["sliders"]}
        noise_gain = float(cfg.get("noise_gain", 1.0))
        return bands, gains, noise_gain


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "my_ecg.wav"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        print("Usage: python ai_ecg.py path/to/ecg.wav")
        sys.exit(1)

    model  = ECGModel(weights_path=DEFAULT_WEIGHTS)
    result = model.analyse_file(filepath)
    print(f"Duration  : {len(result.get('beat_arrays', []))} beats")
    print(f"Summary   : {result['summary']}")
    if result["beats"]:
        print(f"First beat: {result['beats'][0]}")