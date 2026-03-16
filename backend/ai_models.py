"""
ai_models.py
============
Unified AI backend for all 4 modes:
  - ECGModel       : arrhythmia detection + component isolation
  - AnimalModel    : animal sound separation + classification
  - HumanModel     : multi-speaker separation
  - MusicModel     : music source separation (demucs)

Usage example (from main.py / FastAPI):
    from ai_models import ECGModel, AnimalModel, HumanModel, MusicModel
"""

import os, glob, warnings
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from pydub import AudioSegment
import librosa
import joblib
import neurokit2 as nk
import wfdb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _peak_norm(x: np.ndarray) -> np.ndarray:
    peak = np.abs(x).max()
    return x / (peak + 1e-8) if peak > 1e-8 else x


def _load_audio_sf(path: str, target_sr: int = 8000) -> tuple[np.ndarray, int]:
    """Load any audio file → mono float32 numpy array at target_sr."""
    tmp = "_tmp_convert.wav"
    seg = AudioSegment.from_file(path)
    seg = seg.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
    seg.export(tmp, format="wav")
    data, sr = sf.read(tmp)
    os.remove(tmp)
    return data.astype(np.float32), sr


# Patch torch.load for old checkpoints (asteroid / sklearn)
_orig_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _safe_load


# ─────────────────────────────────────────────────────────────────────────────
#  1. ECG Model
# ─────────────────────────────────────────────────────────────────────────────

class _ECGNet(nn.Module):
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32,  7, padding=3), nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128,5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128,256, 3, padding=1),nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))


class ECGModel:
    """
    Auto-trains on any folder of MIT-BIH records (.hea/.dat/.atr).
    Detects arrhythmia type per beat AND isolates per-class signals.

    Parameters
    ----------
    data_dir   : folder containing .hea files
    model_path : where to save/load the trained weights
    window     : samples per beat (default 360 = 1 s @ 360 Hz)
    """

    ARRHYTHMIA_MAP = {
        # annotation symbol → class name
        "N": "Normal", "L": "LBBB", "R": "RBBB",
        "A": "PAC",    "V": "PVC",  "f": "AFib",
        "/": "Paced",  "F": "Fusion",
    }

    def __init__(self, data_dir: str, model_path: str = "ecg_model.pt", window: int = 360):
        self.data_dir   = data_dir
        self.model_path = model_path
        self.window     = window
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"

        # Discover records automatically
        hea_files = glob.glob(os.path.join(data_dir, "*.hea"))
        self.record_ids = [os.path.splitext(os.path.basename(f))[0] for f in hea_files]
        print(f"[ECG] Found {len(self.record_ids)} records: {self.record_ids}")

        # Discover class names from annotations
        self.class_names, self.record_labels = self._scan_classes()
        print(f"[ECG] Classes detected: {self.class_names}")

        # Load or train
        if os.path.exists(model_path):
            self._load(model_path)
        else:
            print("[ECG] No saved model found — training now...")
            self._train()

    # ── Internal: scan annotations to find dominant class per record ─────────
    def _scan_classes(self):
        from collections import Counter
        record_labels = {}
        all_classes   = set()

        for rec_id in self.record_ids:
            path = os.path.join(self.data_dir, rec_id)
            try:
                ann = wfdb.rdann(path, "atr")
                counts = Counter(ann.symbol)
                # dominant non-artifact symbol
                dominant = max(
                    ((s, c) for s, c in counts.items() if s in self.ARRHYTHMIA_MAP),
                    key=lambda x: x[1], default=(None, 0)
                )
                if dominant[0]:
                    cls = self.ARRHYTHMIA_MAP[dominant[0]]
                    record_labels[rec_id] = cls
                    all_classes.add(cls)
            except Exception as e:
                print(f"  [ECG] Could not read annotations for {rec_id}: {e}")

        class_names = sorted(all_classes)
        return class_names, record_labels

    # ── Internal: extract beats with R-peak detection ────────────────────────
    def _extract_beats(self, signal: np.ndarray, fs: int):
        try:
            _, info = nk.ecg_peaks(signal, sampling_rate=fs)
            r_peaks = info["ECG_R_Peaks"]
        except Exception:
            return np.array([]), np.array([])

        half   = self.window // 2
        beats, peaks = [], []
        for r in r_peaks:
            if r - half >= 0 and r + half <= len(signal):
                b = signal[r - half: r + half].astype(np.float32)
                b = (b - b.mean()) / (b.std() + 1e-8)
                beats.append(b)
                peaks.append(r)
        return np.array(beats, dtype=np.float32), np.array(peaks)

    # ── Internal: train ──────────────────────────────────────────────────────
    def _train(self):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        from torch.utils.data import TensorDataset, DataLoader

        all_beats, all_labels = [], []
        for rec_id, cls in self.record_labels.items():
            path   = os.path.join(self.data_dir, rec_id)
            record = wfdb.rdrecord(path)
            signal = record.p_signal[:, 0].astype(np.float32)
            fs     = record.fs
            beats, _ = self._extract_beats(signal, fs)
            if len(beats) == 0:
                continue
            idx = self.class_names.index(cls)
            all_beats.append(beats)
            all_labels.extend([idx] * len(beats))
            print(f"  [ECG] {rec_id} ({cls}): {len(beats)} beats")

        X = np.concatenate(all_beats)
        y = np.array(all_labels)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)
        tr_dl = DataLoader(
            TensorDataset(torch.from_numpy(X_tr).unsqueeze(1), torch.from_numpy(y_tr).long()),
            batch_size=64, shuffle=True)
        te_dl = DataLoader(
            TensorDataset(torch.from_numpy(X_te).unsqueeze(1), torch.from_numpy(y_te).long()),
            batch_size=64)

        self.net = _ECGNet(n_classes=len(self.class_names)).to(self.device)
        opt  = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        sch  = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

        for ep in range(25):
            self.net.train()
            total = correct = 0
            for xb, yb in tr_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.net(xb)
                loss = crit(out, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                correct += (out.argmax(1) == yb).sum().item()
                total   += len(yb)
            sch.step()
            if ep % 5 == 0 or ep == 24:
                print(f"  [ECG] Epoch {ep+1:02d}  acc={correct/total*100:.1f}%")

        # Evaluate
        self.net.eval()
        preds_all, true_all = [], []
        with torch.no_grad():
            for xb, yb in te_dl:
                preds_all.extend(self.net(xb.to(self.device)).argmax(1).cpu().numpy())
                true_all.extend(yb.numpy())
        print(classification_report(true_all, preds_all, target_names=self.class_names))

        torch.save({"state_dict": self.net.state_dict(),
                    "class_names": self.class_names,
                    "window": self.window}, self.model_path)
        print(f"[ECG] Model saved → {self.model_path}")

    # ── Internal: load ───────────────────────────────────────────────────────
    def _load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.class_names = ckpt.get("class_names", self.class_names)
        self.window      = ckpt.get("window", self.window)
        self.net = _ECGNet(n_classes=len(self.class_names))
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        print(f"[ECG] Loaded model from {path}  classes={self.class_names}")

    # ── Public: classify_signal ──────────────────────────────────────────────
    def classify_signal(self, signal: np.ndarray, fs: int = 360) -> dict:
        """
        Classify every beat in a signal.

        Returns
        -------
        {
          "beats": [{"beat_idx", "label", "confidence", "all_probs",
                     "r_peak_sample", "r_peak_time_sec"}, ...],
          "summary": {"Normal": 120, "PVC": 30, ...},
          "class_signals": {"Normal": np.array, "PVC": np.array, ...}
        }
        The class_signals are full-length arrays (same length as signal)
        containing ONLY the samples belonging to that arrhythmia type.
        """
        beats, r_peaks = self._extract_beats(signal, fs)
        if len(beats) == 0:
            return {"beats": [], "summary": {}, "class_signals": {}}

        x = torch.from_numpy(beats).unsqueeze(1)
        with torch.no_grad():
            probs = torch.softmax(self.net(x), dim=1).numpy()

        half    = self.window // 2
        results = []
        # Accumulators for per-class signals
        class_signals = {c: np.zeros_like(signal) for c in self.class_names}

        for i, (p, r) in enumerate(zip(probs, r_peaks)):
            top   = p.argmax()
            label = self.class_names[top]
            results.append({
                "beat_idx"       : i,
                "label"          : label,
                "confidence"     : round(float(p[top]), 3),
                "all_probs"      : {c: round(float(p[j]), 3) for j, c in enumerate(self.class_names)},
                "r_peak_sample"  : int(r),
                "r_peak_time_sec": round(r / fs, 3),
            })
            # Copy the raw beat into the per-class accumulator
            s = max(0, int(r) - half)
            e = min(len(signal), int(r) + half)
            class_signals[label][s:e] += signal[s:e]

        from collections import Counter
        summary = dict(Counter(r["label"] for r in results))

        return {"beats": results, "summary": summary, "class_signals": class_signals}


# ─────────────────────────────────────────────────────────────────────────────
#  2. Animal Model
# ─────────────────────────────────────────────────────────────────────────────

class _AnimalSeparator(nn.Module):
    def __init__(self, n_sources: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4), nn.ReLU(),
            nn.Conv1d(32, 64, 9, padding=4), nn.ReLU(),
            nn.Conv1d(64, 64, 9, padding=4), nn.ReLU(),
            nn.Conv1d(64, n_sources, 9, padding=4),
        )
    def forward(self, x): return self.net(x)


class AnimalModel:
    """
    Separate animal sounds + classify each source using PANNs.

    Parameters
    ----------
    separator_path : path to animal_separator.pth
    classifier_path: path to animal_sound_model.pkl
    encoder_path   : path to animal_label_encoder.pkl
    n_sources      : number of animals in the mix (default 4)
    """

    SR = 22050

    def __init__(self,
                 separator_path : str = "animal_separator.pth",
                 classifier_path: str = "animal_sound_model.pkl",
                 encoder_path   : str = "animal_label_encoder.pkl",
                 n_sources      : int = 4):
        self.n_sources = n_sources

        # ── Separator ────────────────────────────────────────────────────────
        self.separator = _AnimalSeparator(n_sources)
        if os.path.exists(separator_path):
            self.separator.load_state_dict(torch.load(separator_path, map_location="cpu"))
            print(f"[Animal] Separator loaded from {separator_path}")
        else:
            print(f"[Animal] ⚠ Separator weights not found at {separator_path}")
        self.separator.eval()

        # ── Classifier (sklearn GBM) ──────────────────────────────────────
        self.clf     = joblib.load(classifier_path) if os.path.exists(classifier_path) else None
        self.encoder = joblib.load(encoder_path)    if os.path.exists(encoder_path)    else None
        if self.clf:
            print(f"[Animal] Classifier loaded  classes={list(self.encoder.classes_)}")

    # ── Internal: mel feature ────────────────────────────────────────────────
    def _mel_feature(self, audio: np.ndarray) -> np.ndarray:
        mel    = librosa.feature.melspectrogram(y=audio, sr=self.SR, n_mels=128)
        mel_db = librosa.power_to_db(mel)
        return np.mean(mel_db.T, axis=0)

    # ── Internal: classify a single source waveform ──────────────────────────
    def _classify_source(self, audio: np.ndarray) -> tuple[str, float]:
        if self.clf is None or self.encoder is None:
            return "unknown", 0.0
        feat  = self._mel_feature(audio).reshape(1, -1)
        probs = self.clf.predict_proba(feat)[0]
        idx   = probs.argmax()
        return self.encoder.classes_[idx], float(probs[idx])

    # ── Public: separate_and_classify ────────────────────────────────────────
    def separate_and_classify(self, audio_path: str) -> dict:
        """
        Load any audio file, separate sources, classify each.

        Returns
        -------
        {
          "sources": [
            {"label": "dog", "confidence": 0.93,
             "waveform": np.array, "sr": 22050}, ...
          ]
        }
        """
        audio, _ = librosa.load(audio_path, sr=self.SR, mono=True)
        audio     = _peak_norm(audio).astype(np.float32)

        x = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1,1,T)
        with torch.no_grad():
            separated = self.separator(x)   # (1, n_sources, T)

        results = []
        for i in range(self.n_sources):
            src   = separated[0, i].numpy()
            src   = _peak_norm(src)
            label, conf = self._classify_source(src)
            results.append({"label": label, "confidence": conf,
                            "waveform": src, "sr": self.SR})
            print(f"  [Animal] Source {i+1}: {label} ({conf*100:.1f}%)")

        return {"sources": results}


# ─────────────────────────────────────────────────────────────────────────────
#  3. Human Voice Model
# ─────────────────────────────────────────────────────────────────────────────

class HumanModel:
    """
    Separate multiple speakers from a mixed audio file.
    Uses ConvTasNet (asteroid) with cascade splitting for >2 speakers.

    Parameters
    ----------
    pretrained : asteroid HuggingFace model id
    max_depth  : cascade depth  (depth=1 → 2 speakers, depth=2 → up to 4)
    target_sr  : sample rate expected by the model (8000 for ConvTasNet)
    """

    def __init__(self,
                 pretrained: str = "JorisCos/ConvTasNet_Libri2Mix_sepclean_8k",
                 max_depth : int = 2,
                 target_sr : int = 8000):
        from asteroid.models import ConvTasNet
        self.sr        = target_sr
        self.max_depth = max_depth
        print("[Human] Loading ConvTasNet model...")
        self.model = ConvTasNet.from_pretrained(pretrained)
        self.model.eval()
        print("[Human] Model ready.")

    # ── Internal: recursive cascade split ────────────────────────────────────
    def _split(self, wav: np.ndarray, depth: int, sources: list, threshold: float = 0.05):
        if depth >= self.max_depth or len(wav) < self.sr:
            sources.append(wav)
            return

        t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = self.model(t)   # (1, 2, T)

        s1 = out[0, 0].cpu().numpy()
        s2 = out[0, 1].cpu().numpy()
        total = np.mean(wav ** 2) + 1e-8

        for s in (s1, s2):
            if np.mean(s ** 2) / total >= threshold:
                self._split(s, depth + 1, sources, threshold)
            else:
                print(f"  [Human] depth={depth} — silent source skipped")

    # ── Public: separate ──────────────────────────────────────────────────────
    def separate(self, audio_path: str, output_dir: str = "outputs/human") -> dict:
        """
        Separate speakers from audio_path.

        Returns
        -------
        {
          "sources": [
            {"speaker_id": 1, "waveform": np.array, "sr": 8000,
             "duration_sec": 12.3}, ...
          ]
        }
        """
        os.makedirs(output_dir, exist_ok=True)
        audio, _ = _load_audio_sf(audio_path, self.sr)
        audio     = _peak_norm(audio)
        print(f"[Human] duration={len(audio)/self.sr:.1f}s")

        all_sources: list[np.ndarray] = []
        self._split(audio, depth=0, sources=all_sources)
        print(f"[Human] Separated into {len(all_sources)} speakers")

        results = []
        for i, src in enumerate(all_sources):
            src = _peak_norm(src) * 0.9
            out_path = os.path.join(output_dir, f"speaker_{i+1}.wav")
            sf.write(out_path, src, self.sr)
            results.append({"speaker_id": i + 1,
                            "waveform"  : src,
                            "sr"        : self.sr,
                            "duration_sec": round(len(src) / self.sr, 2),
                            "path"      : out_path})
            print(f"  [Human] Speaker {i+1} saved → {out_path}")

        return {"sources": results}


# ─────────────────────────────────────────────────────────────────────────────
#  4. Music Model (Demucs)
# ─────────────────────────────────────────────────────────────────────────────

class MusicModel:
    """
    Separate music into stems using Demucs (drums/bass/vocals/other).
    Runs demucs CLI — make sure it is installed:  pip install demucs

    Parameters
    ----------
    output_root : where demucs writes separated folders
    model_name  : demucs model (default 'htdemucs')
    """

    STEMS = ["drums", "bass", "vocals", "other"]

    def __init__(self, output_root: str = "separated", model_name: str = "htdemucs"):
        self.output_root = output_root
        self.model_name  = model_name

    # ── Public: separate ─────────────────────────────────────────────────────
    def separate(self, audio_path: str) -> dict:
        """
        Run demucs on audio_path.

        Returns
        -------
        {
          "stems": {
            "drums"  : {"waveform": np.array, "sr": int, "path": str},
            "bass"   : {...},
            "vocals" : {...},
            "other"  : {...},
          }
        }
        """
        import subprocess
        audio_path = os.path.abspath(audio_path)
        print(f"[Music] Running demucs on {os.path.basename(audio_path)} ...")
        result = subprocess.run(
            ["demucs", "-n", self.model_name, "-o", self.output_root, "--mp3", audio_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Demucs failed:\n{result.stderr}")

        # Locate output folder
        base    = os.path.splitext(os.path.basename(audio_path))[0]
        stem_dir = os.path.join(self.output_root, self.model_name, base)
        if not os.path.isdir(stem_dir):
            raise FileNotFoundError(f"Demucs output not found at {stem_dir}")

        stems = {}
        for stem in self.STEMS:
            # demucs may output .wav or .mp3
            for ext in ("wav", "mp3"):
                p = os.path.join(stem_dir, f"{stem}.{ext}")
                if os.path.exists(p):
                    wav, sr = librosa.load(p, sr=None, mono=True)
                    stems[stem] = {"waveform": wav.astype(np.float32),
                                   "sr": sr, "path": p}
                    print(f"  [Music] {stem:8s} → {p}")
                    break
            else:
                print(f"  [Music] ⚠ {stem} not found in {stem_dir}")

        return {"stems": stems}


# ─────────────────────────────────────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ai_models.py — self test")
    print("=" * 60)

    # ── ECG ──────────────────────────────────────────────────────────────────
    ECG_DIR = r"D:\2nd year SBME\Second semester\DSP\project\task 2\data\ecg 02\ecg 02"
    if os.path.isdir(ECG_DIR):
        ecg = ECGModel(data_dir=ECG_DIR)
        rec = wfdb.rdrecord(os.path.join(ECG_DIR, os.listdir(ECG_DIR)[0].replace(".hea", "")))
        sig = rec.p_signal[:360 * 30, 0]
        out = ecg.classify_signal(sig, rec.fs)
        print(f"\n[ECG] Summary: {out['summary']}")
        print(f"[ECG] First beat: {out['beats'][0] if out['beats'] else 'none'}")
    else:
        print("[ECG] Skipped — data dir not found")

    # ── Animal ───────────────────────────────────────────────────────────────
    ANIMAL_AUDIO = r"D:\2nd year SBME\Second semester\DSP\project\task 2\data\studio-mixer.wav"
    if os.path.exists(ANIMAL_AUDIO):
        animal = AnimalModel()
        out    = animal.separate_and_classify(ANIMAL_AUDIO)
        print(f"\n[Animal] Sources: {[s['label'] for s in out['sources']]}")
    else:
        print("[Animal] Skipped — audio file not found")

    # ── Human ────────────────────────────────────────────────────────────────
    HUMAN_AUDIO = r"D:\2nd year SBME\Second semester\DSP\project\task 2\data\humans.wav"
    if os.path.exists(HUMAN_AUDIO):
        human = HumanModel()
        out   = human.separate(HUMAN_AUDIO)
        print(f"\n[Human] Speakers found: {len(out['sources'])}")
    else:
        print("[Human] Skipped — audio file not found")

    print("\n✅ ai_models.py loaded successfully")
