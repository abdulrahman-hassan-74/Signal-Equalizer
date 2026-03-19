"""
ai_human.py
===========
Multi-speaker separation using SepFormer (primary) with ConvTasNet fallback.

SepFormer gives better separation quality, especially on mixed speech.
Falls back to ConvTasNet (asteroid) if SepFormer weights are unavailable.

Usage:
    from ai_human import HumanModel
    model = HumanModel()
    result = model.separate("mixed_voices.wav")
    # result["sources"] → [{"speaker_id": 1, "label": "Male (120 Hz)",
    #                        "waveform": np.array, "sr": int,
    #                        "duration_sec": 12.3}, ...]
    #
    # Also compare wavelet vs AI:
    result2 = model.compare_methods(signal_np, sample_rate, scales=[1.0, 1.0])
    # result2 → {"wavelet": {...metrics}, "ai": {...metrics}}

Install for SepFormer:
    pip install speechbrain==0.5.16 huggingface_hub

Install for ConvTasNet fallback:
    pip install asteroid-filterbanks
"""

import os
import io
import time
import numpy as np
import soundfile as sf
from math import gcd

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[Human] ⚠ torch not installed — run: pip install torch")

try:
    import pywt
    PYWT_OK = True
except ImportError:
    PYWT_OK = False

try:
    from scipy.signal import resample_poly
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

from ai_shared import peak_norm, load_audio

# ── SepFormer constants ────────────────────────────────────────────────────────
SEPFORMER_SR  = 8000
MODEL_DIR     = os.path.abspath("pretrained_models/sepformer-libri2mix")
_model_cache  = {}


# ═══════════════════════════════════════════════════════
#  AUDIO HELPERS
# ═══════════════════════════════════════════════════════

def _resample(signal: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample signal from from_sr to to_sr using polyphase filter."""
    if from_sr == to_sr:
        return signal.astype(np.float32)
    if SCIPY_OK:
        g = gcd(int(from_sr), int(to_sr))
        return resample_poly(signal.astype(np.float32),
                             int(to_sr) // g, int(from_sr) // g)
    # fallback: simple linear interpolation
    n_out = int(len(signal) * to_sr / from_sr)
    return np.interp(np.linspace(0, len(signal), n_out),
                     np.arange(len(signal)), signal).astype(np.float32)


def _match_length(s: np.ndarray, n: int) -> np.ndarray:
    return s[:n] if len(s) >= n else np.pad(s, (0, n - len(s)))


# ═══════════════════════════════════════════════════════
#  SEPFORMER LOADER
# ═══════════════════════════════════════════════════════

def _download_sepformer_files():
    """Download only the 4 needed checkpoint files via huggingface_hub."""
    from huggingface_hub import hf_hub_download
    import shutil

    os.makedirs(MODEL_DIR, exist_ok=True)
    needed = ["hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt"]

    for fname in needed:
        dst = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(dst):
            print(f"  [Human] Downloading {fname}…")
            try:
                tmp = hf_hub_download(
                    repo_id  = "speechbrain/sepformer-libri2mix",
                    filename = fname,
                )
                shutil.copy2(tmp, dst)
                print(f"  [Human] ✓ {fname}")
            except Exception as e:
                print(f"  [Human] ✗ {fname}: {e}")
                raise


def _load_sepformer():
    """
    Load SepFormer.
    Strategy:
      1. Try speechbrain from_hparams on local MODEL_DIR (no internet needed).
      2. If that fails, try downloading files first, then load.
      3. If speechbrain is unavailable entirely, raise ImportError so caller
         can fall back to ConvTasNet.
    """
    if "sepformer" in _model_cache:
        return _model_cache["sepformer"]

    try:
        from speechbrain.inference.separation import SepformerSeparation
    except ImportError:
        raise ImportError(
            "speechbrain not installed — "
            "run: pip install speechbrain==0.5.16"
        )

    # Patch torchaudio if needed
    try:
        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: []
    except Exception:
        pass

    # Patch huggingface_hub to remove deprecated kwarg
    try:
        import huggingface_hub.file_download as hfd
        _orig = hfd.hf_hub_download
        def _patched(*a, **kw):
            kw.pop("use_auth_token", None)
            return _orig(*a, **kw)
        hfd.hf_hub_download = _patched
    except Exception:
        pass

    # Download files if missing
    required = ["hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt"]
    if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required):
        print("[Human] Model files missing — downloading…")
        _download_sepformer_files()

    print(f"[Human] Loading SepFormer from {MODEL_DIR}…")
    model = SepformerSeparation.from_hparams(
        source   = MODEL_DIR,
        savedir  = MODEL_DIR,
        run_opts = {"device": "cpu"},
    )
    model.hparams.encoder.eval()
    model.hparams.masknet.eval()
    model.hparams.decoder.eval()

    _model_cache["sepformer"] = model
    print("[Human] SepFormer ready ✓")
    return model


def _separate_with_sepformer(audio: np.ndarray, orig_sr: int) -> tuple:
    """Run SepFormer; returns (list_of_numpy_arrays, elapsed_ms)."""
    model  = _load_sepformer()
    sig_8k = _resample(audio, orig_sr, SEPFORMER_SR)
    tensor = torch.tensor(sig_8k, dtype=torch.float32).unsqueeze(0)  # (1, T)

    t0 = time.perf_counter()
    with torch.no_grad():
        est = model.separate_batch(tensor)   # (1, T, n_spk)
    elapsed = (time.perf_counter() - t0) * 1000

    sources = []
    n_spk = est.shape[-1]
    for i in range(n_spk):
        src = est[0, :, i].cpu().numpy().astype(np.float32)
        src = _resample(src, SEPFORMER_SR, orig_sr)
        src = _match_length(src, len(audio))
        src = peak_norm(src)
        sources.append(src)

    return sources, elapsed


# ═══════════════════════════════════════════════════════
#  CONVTASNET FALLBACK (original code)
# ═══════════════════════════════════════════════════════

def _load_convtasnet():
    if "convtasnet" in _model_cache:
        return _model_cache["convtasnet"]
    from asteroid.models import ConvTasNet
    model = ConvTasNet.from_pretrained(
        "JorisCos/ConvTasNet_Libri2Mix_sepclean_8k"
    )
    model.eval()
    _model_cache["convtasnet"] = model
    print("[Human] ConvTasNet ready ✓")
    return model


def _split_convtasnet(wav, depth, max_depth, sources, threshold=0.05):
    """Recursive 2-way split using ConvTasNet."""
    if depth >= max_depth or len(wav) < SEPFORMER_SR:
        sources.append(wav); return
    model = _load_convtasnet()
    t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(t)
    s1 = out[0, 0].cpu().numpy()
    s2 = out[0, 1].cpu().numpy()
    total = np.mean(wav ** 2) + 1e-8
    for s in (s1, s2):
        if np.mean(s ** 2) / total >= threshold:
            _split_convtasnet(s, depth + 1, max_depth, sources, threshold)


def _separate_with_convtasnet(audio: np.ndarray, orig_sr: int,
                               max_depth: int = 2) -> tuple:
    audio_8k = _resample(audio, orig_sr, SEPFORMER_SR)
    t0 = time.perf_counter()
    srcs: list = []
    _split_convtasnet(peak_norm(audio_8k), 0, max_depth, srcs)
    elapsed = (time.perf_counter() - t0) * 1000
    out = []
    for s in srcs:
        s_r = _resample(s, SEPFORMER_SR, orig_sr)
        s_r = _match_length(peak_norm(s_r), len(audio))
        out.append(s_r)
    return out, elapsed


# ═══════════════════════════════════════════════════════
#  PITCH-BASED LABEL ESTIMATION
# ═══════════════════════════════════════════════════════

def _estimate_speaker_label(audio: np.ndarray, sr: int) -> str:
    """Classify speaker type from median F0."""
    try:
        import librosa
        f0, voiced, _ = librosa.pyin(
            audio, fmin=50, fmax=500, sr=sr, frame_length=2048
        )
        f0_v = f0[voiced] if voiced is not None else np.array([])
        if len(f0_v) == 0:
            return "Speaker"
        mf0 = float(np.median(f0_v))
        if mf0 < 120:   return f"Male adult ({mf0:.0f} Hz)"
        if mf0 < 165:   return f"Male ({mf0:.0f} Hz)"
        if mf0 < 255:   return f"Female ({mf0:.0f} Hz)"
        if mf0 < 350:   return f"Female (high) ({mf0:.0f} Hz)"
        return f"Child ({mf0:.0f} Hz)"
    except Exception:
        return "Speaker"


# ═══════════════════════════════════════════════════════
#  WAVELET EQUALIZATION  (for compare_methods)
# ═══════════════════════════════════════════════════════

def _wavelet_equalize(signal: np.ndarray, scales: list,
                      wavelet: str = "coif3", level: int = 6) -> tuple:
    """Apply per-level wavelet gains. Returns (output, elapsed_ms)."""
    if not PYWT_OK:
        return signal.copy(), 0.0

    t0     = time.perf_counter()
    max_lv = pywt.dwt_max_level(len(signal), wavelet)
    coeffs = pywt.wavedec(signal, wavelet=wavelet,
                           level=min(level, max_lv, 10))
    padded = list(scales) + [1.0] * (len(coeffs) - len(scales))
    c_mod  = [c * s for c, s in zip(coeffs, padded[: len(coeffs)])]
    out    = pywt.waverec(c_mod, wavelet=wavelet)[: len(signal)]
    out    = peak_norm(out.astype(np.float32))
    elapsed = (time.perf_counter() - t0) * 1000
    return out, elapsed


# ═══════════════════════════════════════════════════════
#  METRICS HELPERS
# ═══════════════════════════════════════════════════════

def _snr(ref, est):
    n = min(len(ref), len(est))
    d = ref[:n] - est[:n]
    sp = float(np.mean(ref[:n] ** 2))
    np_ = float(np.mean(d ** 2))
    return 100.0 if np_ < 1e-15 else round(10 * np.log10(sp / np_), 3)


def _si_snr(ref, est):
    n = min(len(ref), len(est))
    s = ref[:n] - ref[:n].mean()
    h = est[:n] - est[:n].mean()
    t = (np.dot(h, s) / (np.dot(s, s) + 1e-10)) * s
    return round(float(10 * np.log10(
        (np.sum(t ** 2) + 1e-10) / (np.sum((h - t) ** 2) + 1e-10)
    )), 3)


def _prd(ref, est):
    n = min(len(ref), len(est))
    return round(float(np.sqrt(np.sum((ref[:n] - est[:n]) ** 2)) /
                        (np.sqrt(np.sum(ref[:n] ** 2)) + 1e-10) * 100), 3)


def _lsd(ref, est):
    n = min(len(ref), len(est))
    S1 = np.abs(np.fft.rfft(ref[:n]))
    S2 = np.abs(np.fft.rfft(est[:n]))
    return round(float(np.sqrt(np.mean(
        (20 * np.log10((S2 + 1e-10) / (S1 + 1e-10))) ** 2
    ))), 3)


# ═══════════════════════════════════════════════════════
#  HumanModel — PUBLIC API
# ═══════════════════════════════════════════════════════

class HumanModel:
    """
    Speaker separation using SepFormer (primary) or ConvTasNet (fallback).

    Public methods
    ──────────────
    separate(audio_path)                      → dict with "sources" list
    separate_signal(signal_np, sr)            → same format
    compare_methods(signal_np, sr, scales)    → dict with wavelet + ai metrics
    """

    def __init__(self,
                 use_sepformer: bool = True,
                 target_sr:     int  = SEPFORMER_SR):
        self.use_sepformer = use_sepformer
        self.target_sr     = target_sr

        if not TORCH_OK:
            raise ImportError("torch is required. Run: pip install torch")

        # Eagerly probe which backend is available
        self._backend = "none"
        if use_sepformer:
            try:
                _load_sepformer()
                self._backend = "sepformer"
            except Exception as e:
                print(f"[Human] SepFormer unavailable ({e}), trying ConvTasNet…")
        if self._backend == "none":
            try:
                _load_convtasnet()
                self._backend = "convtasnet"
            except Exception as e:
                print(f"[Human] ConvTasNet also unavailable ({e})")

        print(f"[Human] Backend: {self._backend}")

    # ── Internal separation dispatcher ──────────────────────
    def _do_separate(self, audio: np.ndarray, sr: int) -> tuple:
        """
        Returns (sources: list[np.ndarray], elapsed_ms: float).
        Each source is float32, same length as audio, peak-normalised.
        """
        audio = peak_norm(audio.astype(np.float32))

        if self._backend == "sepformer":
            return _separate_with_sepformer(audio, sr)

        if self._backend == "convtasnet":
            return _separate_with_convtasnet(audio, sr)

        # Last resort: frequency-band decomposition (no ML)
        print("[Human] ⚠ No ML model available — using frequency split")
        fft  = np.fft.rfft(audio)
        half = len(fft) // 2
        lo   = np.fft.irfft(np.concatenate([fft[:half], np.zeros(len(fft) - half)]),
                             n=len(audio)).astype(np.float32)
        hi   = np.fft.irfft(np.concatenate([np.zeros(half), fft[half:]]),
                             n=len(audio)).astype(np.float32)
        return [peak_norm(lo), peak_norm(hi)], 0.0

    # ── separate() — accepts file path ──────────────────────
    def separate(self, audio_path: str,
                 output_dir: str = "outputs/human") -> dict:
        """
        Separate speakers from an audio file.

        Returns
        -------
        {
          "sources": [
            {"speaker_id": 1, "label": "Male (120 Hz)",
             "waveform": np.array(float32), "sr": int,
             "duration_sec": float, "path": str},
            …
          ],
          "backend": "sepformer" | "convtasnet" | "fallback",
          "elapsed_ms": float
        }
        """
        import librosa
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return self.separate_signal(audio, sr, output_dir)

    # ── separate_signal() — accepts numpy array ──────────────
    def separate_signal(self, audio: np.ndarray, sr: int,
                        output_dir: str = "outputs/human") -> dict:
        """Same as separate() but takes numpy array directly."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"[Human] Separating  {len(audio)/sr:.1f}s  sr={sr}  "
              f"backend={self._backend}")

        sources, elapsed = self._do_separate(audio, sr)
        print(f"[Human] Got {len(sources)} source(s)  ({elapsed:.0f} ms)")

        results = []
        for i, src in enumerate(sources):
            label   = _estimate_speaker_label(src, sr)
            out_pth = os.path.join(output_dir, f"speaker_{i+1}.wav")
            sf.write(out_pth, src * 0.9, sr)
            results.append({
                "speaker_id":   i + 1,
                "label":        label,
                "waveform":     src,
                "sr":           sr,
                "duration_sec": round(len(src) / sr, 2),
                "path":         out_pth,
            })
            print(f"  [Human] Speaker {i+1}: {label}")

        return {
            "sources":    results,
            "backend":    self._backend,
            "elapsed_ms": round(elapsed, 2),
        }

    # ── compare_methods() — wavelet vs AI ────────────────────
    def compare_methods(self, signal: np.ndarray, sr: int,
                        scales:  list  = None,
                        wavelet: str   = "coif3",
                        level:   int   = 6) -> dict:
        """
        Run both wavelet equalisation and AI separation with the same
        per-speaker gains, then compute comparison metrics.

        Parameters
        ----------
        signal  : mono float32 array
        sr      : sample rate
        scales  : list of per-speaker gain values (default all 1.0)
        wavelet : wavelet name for System B (default: coif3)
        level   : decomposition levels

        Returns
        -------
        {
          "wavelet": {
            "output":    np.array,
            "snr":       float,    "si_snr": float,
            "prd":       float,    "lsd":    float,
            "time_ms":   float
          },
          "ai": {
            "output":   np.array,
            "sources":  [np.array, …],
            "snr":      float,  "si_snr": float,
            "prd":      float,  "lsd":    float,
            "time_ms":  float,
            "error":    str | None
          }
        }
        """
        if scales is None:
            scales = [1.0, 1.0]

        # ── Wavelet ───────────────────────────────────────────
        wav_out, wav_t = _wavelet_equalize(signal, scales, wavelet, level)

        # ── AI ────────────────────────────────────────────────
        ai_error: str | None = None
        try:
            raw_srcs, ai_t = self._do_separate(signal, sr)
            n   = len(signal)
            out = np.zeros(n, dtype=np.float32)
            for j, src in enumerate(raw_srcs):
                g = scales[j] if j < len(scales) else 1.0
                out += _match_length(src, n) * g
            ai_out = peak_norm(out)
        except Exception as e:
            ai_error = str(e)
            ai_out   = signal.copy()
            raw_srcs = []
            ai_t     = 0.0

        def _metrics(ref, est):
            return {
                "snr":    _snr(ref, est),
                "si_snr": _si_snr(ref, est),
                "prd":    _prd(ref, est),
                "lsd":    _lsd(ref, est),
            }

        return {
            "wavelet": {
                "output":   wav_out,
                "time_ms":  round(wav_t, 2),
                **_metrics(signal, wav_out),
            },
            "ai": {
                "output":   ai_out,
                "sources":  raw_srcs,
                "time_ms":  round(ai_t, 2),
                "error":    ai_error,
                **(_metrics(signal, ai_out) if not ai_error else
                   {"snr": None, "si_snr": None, "prd": None, "lsd": None}),
            },
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "mixed_voices.wav"
    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        print("Usage: python ai_human.py path/to/mixed_voices.wav")
        sys.exit(1)
    model  = HumanModel()
    result = model.separate(audio_file)
    print(f"\nBackend  : {result['backend']}")
    print(f"Elapsed  : {result['elapsed_ms']:.0f} ms")
    for s in result["sources"]:
        print(f"  Speaker {s['speaker_id']}: {s['label']}  "
              f"{s['duration_sec']:.1f}s  → {s['path']}")