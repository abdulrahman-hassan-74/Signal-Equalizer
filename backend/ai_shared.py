"""
ai_shared.py
============
Shared utilities used by all AI model files.
Import this in each ai_*.py file.
"""

import os
import warnings
import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment

warnings.filterwarnings("ignore")

# ── Patch torch.load for old checkpoints ─────────────────────────────────────
_orig_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _safe_load


# ── Normalize signal to [-1, 1] ──────────────────────────────────────────────
def peak_norm(x: np.ndarray) -> np.ndarray:
    peak = np.abs(x).max()
    return x / (peak + 1e-8) if peak > 1e-8 else x


# ── Load any audio file → mono float32 at target_sr ─────────────────────────
def load_audio(path: str, target_sr: int = 8000) -> tuple:
    """
    Load any audio file (wav/mp3/m4a/ogg) → (numpy float32 array, sample_rate).
    Converts to mono and resamples to target_sr.
    """
    tmp = f"_tmp_load_{os.getpid()}.wav"
    try:
        seg = AudioSegment.from_file(path)
        seg = seg.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
        seg.export(tmp, format="wav")
        data, sr = sf.read(tmp)
        return data.astype(np.float32), sr
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)