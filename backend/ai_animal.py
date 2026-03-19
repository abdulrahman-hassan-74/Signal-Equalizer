"""
ai_animal.py
============
Animal sound isolation using YAMNet (primary) + Wiener soft masking,
with the original 1D-CNN separator as an automatic fallback.

YAMNet strategy (no training required):
  1. Load audio at 16 kHz
  2. Run YAMNet to get per-frame class scores (521 classes)
  3. Map YAMNet classes → animal groups (Dog, Cat, Bird, …)
  4. Build per-group Wiener soft masks from score curves
  5. Apply masks to STFT → iSTFT → one isolated WAV per animal group

Fallback (if tensorflow / tensorflow_hub unavailable):
  Uses the original 1D-CNN separator + sklearn GBM classifier.

Install for YAMNet backend:
    pip install tensorflow tensorflow-hub librosa soundfile resampy scipy

Install for CNN fallback:
    pip install torch joblib librosa
"""

from __future__ import annotations
import os, time, io
import numpy as np
import soundfile as sf

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    print("[Animal] ⚠ librosa not installed — pip install librosa")

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_OK = True
except ImportError:
    TF_OK = False
    print("[Animal] ⚠ tensorflow/tensorflow_hub not installed — YAMNet unavailable")

try:
    import torch, torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

try:
    from scipy.signal import stft as scipy_stft, istft as scipy_istft
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[Animal] ⚠ scipy not installed — pip install scipy")

from ai_shared import peak_norm

# ── constants ─────────────────────────────────────────────────────────────────
YAMNET_SR   = 16_000          # YAMNet requires 16 kHz
YAMNET_URL  = "https://tfhub.dev/google/yamnet/1"
YAMNET_CSV  = ("https://raw.githubusercontent.com/tensorflow/models/master/"
               "research/audioset/yamnet/yamnet_class_map.csv")

CNN_SR      = 22_050          # original CNN separator sample rate

# Mask sharpness exponent — higher = cleaner isolation but may clip quiet sounds
MASK_POWER  = 2.5

# Minimum peak YAMNet score for a group to be included in output
DETECT_THRESH = 0.05

# STFT parameters (for Wiener masking)
N_FFT   = 1024
HOP_LEN = 256
WIN_LEN = 1024

# ── YAMNet animal group definitions ───────────────────────────────────────────
# Each entry maps a friendly label → list of YAMNet keyword fragments.
# The keywords are matched against YAMNet's 521 class names (case-insensitive).
ANIMAL_GROUPS: dict[str, list[str]] = {
    "Dog"      : ["dog", "bark", "howl", "whimper", "growl"],
    "Cat"      : ["cat", "meow", "purr", "hiss"],
    "Bird"     : ["bird", "chirp", "tweet", "crow", "rooster", "owl",
                  "squawk", "parrot", "duck", "goose", "turkey"],
    "Frog"     : ["frog", "croak", "toad"],
    "Horse"    : ["horse", "neigh", "whinny"],
    "Insect"   : ["insect", "cricket", "bee", "fly", "buzz", "cicada"],
    "Primate"  : ["monkey", "primate", "gibbon", "chimpanzee", "ape"],
    "Livestock": ["cow", "moo", "sheep", "goat", "pig", "oink", "hen",
                  "chicken", "cluck"],
    "Wild"     : ["lion", "tiger", "elephant", "wolf", "bear", "deer",
                  "fox", "snake", "whale", "dolphin"],
}

# ── module-level caches ───────────────────────────────────────────────────────
_yamnet_cache:     dict = {}   # {"model": ..., "class_names": [...], "group_indices": {...}}
_cnn_model_cache:  dict = {}


# ═══════════════════════════════════════════════════════════════════
#  YAMNET BACKEND
# ═══════════════════════════════════════════════════════════════════

def _load_yamnet() -> dict:
    """Load YAMNet model + class names + group→index mapping (cached)."""
    if _yamnet_cache:
        return _yamnet_cache

    import csv, urllib.request

    print("[Animal] Loading YAMNet from TF Hub…")
    model = hub.load(YAMNET_URL)

    # Download class map
    with urllib.request.urlopen(YAMNET_CSV) as resp:
        reader = csv.DictReader(resp.read().decode().splitlines())
        class_names = [row["display_name"] for row in reader]

    # Map each group → list of YAMNet class indices
    group_indices: dict[str, list[int]] = {}
    for group, keywords in ANIMAL_GROUPS.items():
        idxs = [i for i, name in enumerate(class_names)
                if any(kw in name.lower() for kw in keywords)]
        if idxs:
            group_indices[group] = idxs
            matched = [class_names[i] for i in idxs[:5]]
            print(f"  [Animal] {group:<12}: {matched}{'…' if len(idxs)>5 else ''}")

    _yamnet_cache["model"]         = model
    _yamnet_cache["class_names"]   = class_names
    _yamnet_cache["group_indices"] = group_indices
    print(f"[Animal] YAMNet ready — {len(group_indices)} animal groups mapped.")
    return _yamnet_cache


def _yamnet_separate(audio_16k: np.ndarray, duration: float) -> dict[str, np.ndarray]:
    """
    Run YAMNet inference + Wiener masking.
    Returns {group_label: isolated_signal_float32} for detected groups only.
    """
    cache  = _load_yamnet()
    model  = cache["model"]
    g_idx  = cache["group_indices"]

    # ── YAMNet inference ─────────────────────────────────────────────────────
    waveform = tf.cast(audio_16k, tf.float32)
    scores_np, _, _ = model(waveform)          # (n_frames, 521)
    scores_np = scores_np.numpy()
    n_frames  = scores_np.shape[0]

    # Per-group score curve (max over member classes per frame)
    t_yamnet = np.linspace(0, duration, n_frames)
    group_scores: dict[str, np.ndarray] = {}
    for group, idxs in g_idx.items():
        curve = scores_np[:, idxs].max(axis=1)
        if curve.max() >= DETECT_THRESH:
            group_scores[group] = curve

    if not group_scores:
        print("[Animal] YAMNet: no animals detected above threshold.")
        return {}

    detected = ", ".join(
        f"{g}(peak={v.max():.2f})" for g, v in group_scores.items()
    )
    print(f"[Animal] YAMNet detected: {detected}")

    # ── STFT of the original signal ──────────────────────────────────────────
    f_bins, t_stft_arr, S = scipy_stft(
        audio_16k, fs=YAMNET_SR,
        nperseg=WIN_LEN, noverlap=WIN_LEN - HOP_LEN, nfft=N_FFT
    )
    S_mag   = np.abs(S)
    S_phase = np.angle(S)
    n_time  = S_mag.shape[1]

    t_stft = np.linspace(0, duration, n_time)

    # ── Interpolate score curves onto STFT time grid ─────────────────────────
    groups_ordered = list(group_scores.keys())
    score_matrix   = np.zeros((len(groups_ordered), n_time))
    for i, g in enumerate(groups_ordered):
        score_matrix[i] = np.interp(t_stft, t_yamnet, group_scores[g])

    # Background row so masks sum to 1
    background  = np.clip(1.0 - score_matrix.sum(axis=0), 0, None)
    all_scores  = np.vstack([score_matrix, background])
    total       = all_scores.sum(axis=0, keepdims=True) + 1e-8
    masks       = all_scores / total          # Wiener masks (n_groups+1, n_time)

    # ── Apply masks and reconstruct ──────────────────────────────────────────
    isolated: dict[str, np.ndarray] = {}
    for i, group in enumerate(groups_ordered):
        soft_mask = (masks[i][np.newaxis, :] ** MASK_POWER)   # (1, n_time)
        S_animal  = S_mag * soft_mask * np.exp(1j * S_phase)
        _, track  = scipy_istft(
            S_animal, fs=YAMNET_SR,
            nperseg=WIN_LEN, noverlap=WIN_LEN - HOP_LEN, nfft=N_FFT
        )
        track = peak_norm(track.astype(np.float32)) * 0.9
        isolated[group] = track
        print(f"  [Animal] Isolated: {group}")

    return isolated


# ═══════════════════════════════════════════════════════════════════
#  CNN FALLBACK BACKEND  (original code, unchanged logic)
# ═══════════════════════════════════════════════════════════════════

class _AnimalSeparatorCNN(nn.Module if TORCH_OK else object):
    def __init__(self, n_sources: int = 4):
        if not TORCH_OK:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4), nn.ReLU(),
            nn.Conv1d(32, 64, 9, padding=4), nn.ReLU(),
            nn.Conv1d(64, 64, 9, padding=4), nn.ReLU(),
            nn.Conv1d(64, n_sources, 9, padding=4),
        )
    def forward(self, x):
        return self.net(x)


def _load_cnn(separator_path: str = "animal_separator.pth",
              classifier_path: str = "animal_sound_model.pkl",
              encoder_path:    str = "animal_label_encoder.pkl",
              n_sources:       int = 4) -> dict:
    """Load (or retrieve from cache) the CNN separator + GBM classifier."""
    key = (separator_path, classifier_path, encoder_path, n_sources)
    if key in _cnn_model_cache:
        return _cnn_model_cache[key]

    sep = _AnimalSeparatorCNN(n_sources)
    if os.path.exists(separator_path):
        sep.load_state_dict(torch.load(separator_path, map_location="cpu"))
        print(f"[Animal] CNN separator loaded from {separator_path}")
    else:
        print(f"[Animal] ⚠ CNN weights not found at '{separator_path}' — untrained")
    sep.eval()

    clf = enc = None
    if JOBLIB_OK and os.path.exists(classifier_path) and os.path.exists(encoder_path):
        clf = joblib.load(classifier_path)
        enc = joblib.load(encoder_path)
        print(f"[Animal] GBM classifier loaded  classes={list(enc.classes_)}")
    else:
        print("[Animal] ⚠ GBM classifier not found — sources will be labelled source_N")

    result = {"sep": sep, "clf": clf, "enc": enc, "n_sources": n_sources}
    _cnn_model_cache[key] = result
    return result


def _mel_feature(audio: np.ndarray, sr: int = CNN_SR) -> np.ndarray:
    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel)
    return np.mean(mel_db.T, axis=0)


def _classify_cnn(audio: np.ndarray, idx: int, clf, enc) -> tuple[str, float]:
    if clf is None or enc is None:
        return f"source_{idx + 1}", 0.0
    try:
        feat  = _mel_feature(audio).reshape(1, -1)
        probs = clf.predict_proba(feat)[0]
        top   = probs.argmax()
        return enc.classes_[top], float(probs[top])
    except Exception as e:
        print(f"  [Animal] GBM classify error: {e}")
        return f"source_{idx + 1}", 0.0


def _cnn_separate(audio: np.ndarray, sr: int,
                  separator_path:  str = "animal_separator.pth",
                  classifier_path: str = "animal_sound_model.pkl",
                  encoder_path:    str = "animal_label_encoder.pkl",
                  n_sources:       int = 4) -> dict[str, np.ndarray]:
    """Run CNN separator. Returns {label: waveform_float32}."""
    m   = _load_cnn(separator_path, classifier_path, encoder_path, n_sources)
    sep = m["sep"]; clf = m["clf"]; enc = m["enc"]

    # Resample to CNN_SR if needed
    if sr != CNN_SR and LIBROSA_OK:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=CNN_SR)

    audio = peak_norm(audio.astype(np.float32))
    x     = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)   # (1, 1, T)
    with torch.no_grad():
        separated = sep(x)   # (1, n_sources, T)

    isolated: dict[str, np.ndarray] = {}
    for i in range(n_sources):
        src           = peak_norm(separated[0, i].numpy())
        label, conf   = _classify_cnn(src, i, clf, enc)
        # De-duplicate labels
        base  = label
        count = 1
        while label in isolated:
            count += 1; label = f"{base}_{count}"
        isolated[label] = src
        print(f"  [Animal] Source {i+1}: {label} ({conf*100:.1f}%)")

    return isolated


# ═══════════════════════════════════════════════════════════════════
#  METRICS HELPERS  (same as ai_human.py for consistency)
# ═══════════════════════════════════════════════════════════════════

def _snr(ref, est):
    n = min(len(ref), len(est)); d = ref[:n] - est[:n]
    sp = float(np.mean(ref[:n]**2)); np_ = float(np.mean(d**2))
    return 100.0 if np_ < 1e-15 else round(10*np.log10(sp/np_), 3)

def _si_snr(ref, est):
    n=min(len(ref),len(est)); s=ref[:n]-ref[:n].mean(); h=est[:n]-est[:n].mean()
    t=(np.dot(h,s)/(np.dot(s,s)+1e-10))*s
    return round(float(10*np.log10((np.sum(t**2)+1e-10)/(np.sum((h-t)**2)+1e-10))),3)

def _prd(ref, est):
    n=min(len(ref),len(est))
    return round(float(np.sqrt(np.sum((ref[:n]-est[:n])**2))/(np.sqrt(np.sum(ref[:n]**2))+1e-10)*100),3)

def _lsd(ref, est):
    n=min(len(ref),len(est))
    S1=np.abs(np.fft.rfft(ref[:n])); S2=np.abs(np.fft.rfft(est[:n]))
    return round(float(np.sqrt(np.mean((20*np.log10((S2+1e-10)/(S1+1e-10)))**2))),3)


# ═══════════════════════════════════════════════════════════════════
#  AnimalModel — PUBLIC API
# ═══════════════════════════════════════════════════════════════════

class AnimalModel:
    """
    Animal sound isolation.

    Primary backend  : YAMNet + Wiener soft masking (no training needed).
    Fallback backend : 1D-CNN separator + GBM classifier.

    Public methods
    ──────────────
    separate_and_classify(audio_path)
        → {"sources": [{label, confidence, waveform, sr}, ...]}

    compare_methods(signal_np, sr, scales)
        → {"wavelet": {metrics…}, "yamnet": {metrics…}}
    """

    def __init__(self,
                 use_yamnet:       bool = True,
                 separator_path:   str  = "animal_separator.pth",
                 classifier_path:  str  = "animal_sound_model.pkl",
                 encoder_path:     str  = "animal_label_encoder.pkl",
                 n_sources:        int  = 4):

        self.use_yamnet      = use_yamnet
        self.separator_path  = separator_path
        self.classifier_path = classifier_path
        self.encoder_path    = encoder_path
        self.n_sources       = n_sources

        # Probe which backend is available
        self._backend = "none"
        if use_yamnet and TF_OK and SCIPY_OK and LIBROSA_OK:
            try:
                _load_yamnet()
                self._backend = "yamnet"
            except Exception as e:
                print(f"[Animal] YAMNet load failed ({e}), trying CNN fallback…")

        if self._backend == "none" and TORCH_OK and LIBROSA_OK:
            try:
                _load_cnn(separator_path, classifier_path, encoder_path, n_sources)
                self._backend = "cnn"
            except Exception as e:
                print(f"[Animal] CNN load also failed: {e}")

        print(f"[Animal] Backend: {self._backend}")

    # ── Internal dispatcher ──────────────────────────────────────────────────
    def _do_separate(self, audio: np.ndarray,
                     sr: int) -> dict[str, np.ndarray]:
        """
        Returns {label: waveform_float32} for all detected animal groups.
        Works regardless of backend.
        """
        audio = peak_norm(audio.astype(np.float32))

        if self._backend == "yamnet":
            # YAMNet needs 16 kHz
            audio_16k = (librosa.resample(audio, orig_sr=sr, target_sr=YAMNET_SR)
                         if sr != YAMNET_SR else audio)
            duration  = len(audio_16k) / YAMNET_SR
            return _yamnet_separate(audio_16k, duration)

        if self._backend == "cnn":
            return _cnn_separate(audio, sr,
                                 self.separator_path,
                                 self.classifier_path,
                                 self.encoder_path,
                                 self.n_sources)

        # Last resort: frequency-band split labelled by band
        print("[Animal] ⚠ No ML backend available — using frequency split")
        fft   = np.fft.rfft(audio)
        bands = np.array_split(np.arange(len(fft)), 4)
        result: dict[str, np.ndarray] = {}
        band_names = ["Sub-bass", "Bass", "Mid", "High"]
        for name, idx in zip(band_names, bands):
            m    = np.zeros(len(fft), dtype=complex)
            m[idx] = fft[idx]
            sig  = np.fft.irfft(m, n=len(audio)).astype(np.float32)
            result[name] = peak_norm(sig)
        return result

    # ── separate_and_classify() — accepts file path ──────────────────────────
    def separate_and_classify(self, audio_path: str) -> dict:
        """
        Separate and classify animal sounds in a mixed audio file.

        Parameters
        ----------
        audio_path : path to any WAV/MP3/OGG/FLAC file

        Returns
        -------
        {
          "sources": [
            {"label": "Dog", "confidence": 1.0,
             "waveform": np.array(float32), "sr": int}, …
          ],
          "backend": "yamnet" | "cnn" | "fallback",
          "elapsed_ms": float
        }
        """
        if not LIBROSA_OK:
            raise ImportError("librosa is required")

        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return self.separate_signal(audio, int(sr))

    # ── separate_signal() — accepts numpy array ──────────────────────────────
    def separate_signal(self, audio: np.ndarray, sr: int) -> dict:
        """Same as separate_and_classify() but takes a numpy array."""
        print(f"[Animal] Separating {len(audio)/sr:.1f}s  sr={sr}  backend={self._backend}")
        t0 = time.perf_counter()
        isolated = self._do_separate(audio, sr)
        elapsed  = (time.perf_counter() - t0) * 1000

        # YAMNet outputs are at YAMNET_SR; resample back to original sr if needed
        out_sr = YAMNET_SR if self._backend == "yamnet" else CNN_SR if self._backend == "cnn" else sr

        sources = []
        for label, waveform in isolated.items():
            sources.append({
                "label"      : label,
                "confidence" : 1.0 if self._backend == "yamnet" else 0.0,
                "waveform"   : waveform.astype(np.float32),
                "sr"         : out_sr,
            })

        return {
            "sources"    : sources,
            "backend"    : self._backend,
            "elapsed_ms" : round(elapsed, 2),
        }

    # ── compare_methods() — YAMNet/CNN vs Wavelet ────────────────────────────
    def compare_methods(self, signal: np.ndarray, sr: int,
                        scales:  list  = None,
                        wavelet: str   = "db4",
                        level:   int   = 6) -> dict:
        """
        Compare YAMNet isolation vs wavelet equalisation on the same signal.

        Parameters
        ----------
        signal  : mono float32 array (original sample rate)
        sr      : sample rate
        scales  : per-source gain list (default all 1.0)
        wavelet : wavelet for System B (default: db4 — optimal for animals)
        level   : decomposition levels

        Returns
        -------
        {
          "wavelet": {"output": np.array, "snr", "si_snr", "prd", "lsd", "time_ms"},
          "yamnet":  {"output": np.array, "sources": [np.array,…],
                      "snr", "si_snr", "prd", "lsd", "time_ms",
                      "backend": str, "error": str|None}
        }
        """
        import pywt
        if scales is None:
            scales = [1.0] * self.n_sources

        # ── Wavelet equalisation ──────────────────────────────────────────────
        t0     = time.perf_counter()
        try:
            max_lv = pywt.dwt_max_level(len(signal), wavelet)
            coeffs = pywt.wavedec(signal, wavelet=wavelet,
                                   level=min(level, max_lv, 10))
            padded = list(scales) + [1.0] * (len(coeffs) - len(scales))
            c_mod  = [c * s for c, s in zip(coeffs, padded[:len(coeffs)])]
            wav_out = pywt.waverec(c_mod, wavelet=wavelet)[:len(signal)]
            wav_out = peak_norm(wav_out.astype(np.float32))
        except Exception as e:
            print(f"[Animal] Wavelet error: {e}")
            wav_out = signal.copy()
        wav_t = (time.perf_counter() - t0) * 1000

        # ── YAMNet / CNN isolation ────────────────────────────────────────────
        ai_error: str | None = None
        try:
            isolated    = self._do_separate(signal, sr)
            sources_arr = list(isolated.values())
            n           = len(signal)
            out_sr      = YAMNET_SR if self._backend == "yamnet" else sr

            # Mix isolated sources with their scale gains
            mixed = np.zeros(n, dtype=np.float32)
            for j, src in enumerate(sources_arr):
                # Resample back to original sr if YAMNet changed it
                if out_sr != sr and LIBROSA_OK:
                    src = librosa.resample(src, orig_sr=out_sr, target_sr=sr)
                # Match length
                if len(src) >= n:
                    src = src[:n]
                else:
                    src = np.pad(src, (0, n - len(src)))
                g = scales[j] if j < len(scales) else 1.0
                mixed += src * g

            ai_out = peak_norm(mixed)
        except Exception as e:
            import traceback; traceback.print_exc()
            ai_error = str(e)
            ai_out   = signal.copy()
            sources_arr = []
        ai_t = (time.perf_counter() - t0) * 1000 - wav_t

        def _metrics(ref, est):
            return {"snr":    _snr(ref, est),
                    "si_snr": _si_snr(ref, est),
                    "prd":    _prd(ref, est),
                    "lsd":    _lsd(ref, est)}

        return {
            "wavelet": {
                "output":   wav_out,
                "time_ms":  round(wav_t, 2),
                **_metrics(signal, wav_out),
            },
            "yamnet": {
                "output":   ai_out,
                "sources":  sources_arr,
                "time_ms":  round(ai_t, 2),
                "backend":  self._backend,
                "error":    ai_error,
                **(_metrics(signal, ai_out) if not ai_error
                   else {"snr": None, "si_snr": None, "prd": None, "lsd": None}),
            },
        }


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "mixed_animals.wav"
    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        print("Usage: python ai_animal.py path/to/mixed_animals.wav")
        sys.exit(1)

    model  = AnimalModel()
    result = model.separate_and_classify(audio_file)
    print(f"\nBackend  : {result['backend']}")
    print(f"Elapsed  : {result['elapsed_ms']:.0f} ms")
    for s in result["sources"]:
        print(f"  {s['label']:<14}  confidence={s['confidence']*100:.0f}%"
              f"  sr={s['sr']}")