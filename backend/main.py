"""
main.py — FastAPI Backend
===========================
Three-system equalizer: Frequency (FFT) | Wavelet | AI Separation

New routes for System C (AI):
  POST /ai/separate   — run AI separation once, cache components server-side
  POST /ai/mix        — weighted sum of cached components (instant)
"""

import numpy as np
import soundfile as sf
import base64, io, uuid, os, tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict

from signal_processor  import compute_fft, compute_spectrogram
from equalizer_engine  import (_apply_gain_fourier, apply_wavelet_gains,
                                get_wavelet_band_energies, OPTIMAL_WAVELETS)
from settings_manager  import load_settings, save_settings

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── In-memory stores ──────────────────────────────────────────────────────────
signals:       dict = {}   # sid → (np.float32, sample_rate)
ai_components: dict = {}   # sid → {"mode", "components": [{name, waveform, sr}]}

CUSTOM_MODES = {"instruments", "animals", "voices", "ecg"}

# ── Lazy AI model singletons ──────────────────────────────────────────────────
_ecg_model    = None
_animal_model = None
_human_model  = None
_music_model  = None

ECG_WEIGHTS = os.path.join(os.path.dirname(__file__),
              "ecg_pretrainedmodel", "ECG_standalone_version", "ecg_resnet_mitbih.pt")


def _get_ecg_model():
    global _ecg_model
    if _ecg_model is None:
        from ai_ecg import ECGModel
        _ecg_model = ECGModel(weights_path=ECG_WEIGHTS)
    return _ecg_model

def _get_animal_model():
    global _animal_model
    if _animal_model is None:
        from ai_animal import AnimalModel
        _animal_model = AnimalModel()
    return _animal_model

def _get_human_model():
    global _human_model
    if _human_model is None:
        from ai_human import HumanModel
        _human_model = HumanModel()
    return _human_model

def _get_music_model():
    global _music_model
    if _music_model is None:
        from ai_music import MusicModel
        _music_model = MusicModel()
    return _music_model


# ── Shared helpers ────────────────────────────────────────────────────────────

def _signal_to_tempfile(signal: np.ndarray, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, signal.astype(np.float32), sr)
    return tmp.name

def _snr_db(original: np.ndarray, modified: np.ndarray) -> float:
    n = min(len(original), len(modified))
    sig, mod = original[:n], modified[:n]
    noise = sig - mod
    sp = float(np.mean(sig ** 2))
    np_ = float(np.mean(noise ** 2)) + 1e-12
    return round(10 * np.log10(sp / np_), 2)

def _encode_signal(signal: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, signal.astype(np.float32), sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()

def _resample_to(signal: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """Resample signal to target sample rate."""
    if src_sr == tgt_sr:
        return signal
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(tgt_sr, src_sr)
    return resample_poly(signal, tgt_sr // g, src_sr // g).astype(np.float32)

def _pad_or_trim(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or trim to target_len."""
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.concatenate([arr, np.zeros(target_len - len(arr), dtype=np.float32)])


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/ping")
@app.get("/health")
def ping():
    return {"message": "hello", "signals_cached": len(signals)}


@app.post("/upload")
async def upload_signal(file: UploadFile = File(...)):
    data, sr = sf.read(io.BytesIO(await file.read()))
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    sid  = str(uuid.uuid4())[:8]
    signals[sid] = (data, sr)
    # Clear any cached AI components for this slot
    ai_components.pop(sid, None)
    return {"signal_id": sid, "sample_rate": int(sr),
            "duration": float(len(data) / sr)}


@app.get("/signal/{sid}")
def get_signal(sid: str):
    if sid not in signals:
        return {"error": "Signal not found"}
    data, sr = signals[sid]
    step = max(1, len(data) // 5000)
    return {"samples": data[::step].tolist(),
            "sample_rate": int(sr), "full_length": len(data)}


# ── System A + B: Equalize (FFT + Wavelet) ───────────────────────────────────

class GainBand(BaseModel):
    band_id:     int
    freq_ranges: List[List[float]]
    gain:        float

class EqualizeRequest(BaseModel):
    signal_id:     str
    freq_gains:    List[GainBand]
    wavelet_gains: Optional[List[GainBand]] = []
    mode:          str = "generic"

@app.post("/equalize")
def equalize(req: EqualizeRequest):
    if req.signal_id not in signals:
        return {"error": "Signal not found"}
    signal, sr = signals[req.signal_id]
    result = signal.copy()

    for band in req.freq_gains:
        result = _apply_gain_fourier(result, sr, band.freq_ranges, band.gain)

    wavelet_used = None
    if req.mode in CUSTOM_MODES and req.wavelet_gains:
        wavelet_used = OPTIMAL_WAVELETS.get(req.mode, "db4")
        bands = [{"id": b.band_id, "freq_ranges": b.freq_ranges, "gain": b.gain}
                 for b in req.wavelet_gains]
        result = apply_wavelet_gains(result, sr, bands, wavelet_used)

    b64 = _encode_signal(result, sr)
    wav_in, wav_out = [], []
    if req.mode in CUSTOM_MODES and req.wavelet_gains and wavelet_used:
        try:
            from settings_manager import load_settings
            raw_cfg = load_settings(req.mode)
            sliders = raw_cfg.get("sliders", [])
            name_by_id  = {int(s["id"]): (s.get("name") or s.get("label") or f"Band {i+1}")
                           for i, s in enumerate(sliders) if "id" in s}
            name_by_idx = {i: (s.get("name") or s.get("label") or f"Band {i+1}")
                           for i, s in enumerate(sliders)}
        except Exception:
            name_by_id = {}; name_by_idx = {}
        bi = [{"id": b.band_id, "freq_ranges": b.freq_ranges,
               "name": name_by_id.get(b.band_id) or name_by_idx.get(i, f"Band {b.band_id}")}
              for i, b in enumerate(req.wavelet_gains)]
        wav_in  = get_wavelet_band_energies(signal, sr, bi, wavelet_used)
        wav_out = get_wavelet_band_energies(result, sr, bi, wavelet_used)

    return {
        "output_signal_b64":       b64,
        "fft_input":               compute_fft(signal, sr),
        "fft_output":              compute_fft(result, sr),
        "spectrogram_output":      compute_spectrogram(result, sr),
        "wavelet_energies_input":  wav_in,
        "wavelet_energies_output": wav_out,
        "wavelet_used":            wavelet_used,
    }


@app.get("/spectrogram/{sid}")
def get_spectrogram(sid: str):
    if sid not in signals:
        return {"error": "Signal not found"}
    data, sr = signals[sid]
    return {"spectrogram": compute_spectrogram(data, sr)}


@app.get("/wavelet-compare/{sid}/{mode}")
def wavelet_compare(sid: str, mode: str):
    if sid not in signals:
        return {"error": "Signal not found"}
    if mode not in CUSTOM_MODES:
        return {"error": "Wavelet comparison only for custom modes"}
    signal, sr = signals[sid]
    wavelet = OPTIMAL_WAVELETS.get(mode, "db4")
    tr, tg  = [[500, 2000]], 0.5
    rf = _apply_gain_fourier(signal, sr, tr, tg)
    rw = apply_wavelet_gains(signal, sr, [{"id":1,"freq_ranges":tr,"gain":tg}], wavelet)
    sf_ = _snr_db(signal, rf); sw = _snr_db(signal, rw)
    return {"mode": mode, "wavelet_used": wavelet,
            "snr_fourier_db": sf_, "snr_wavelet_db": sw,
            "better": "wavelet" if sw > sf_ else "fourier"}


@app.get("/settings/{mode_name}")
def get_settings(mode_name: str):
    try:    return load_settings(mode_name)
    except: return {"error": f"Mode '{mode_name}' not found"}

class SaveSettingsRequest(BaseModel):
    mode_name: str
    config: dict

@app.post("/settings/save")
def save_settings_route(body: SaveSettingsRequest):
    return {"success": save_settings(body.mode_name, body.config)}

@app.get("/modes/list")
def list_modes():
    return ["instruments", "animals", "voices", "ecg", "generic"]

@app.get("/mode-info/{mode}")
def mode_info(mode: str):
    if mode in CUSTOM_MODES:
        return {"mode": mode, "type": "custom",
                "optimal_wavelet": OPTIMAL_WAVELETS.get(mode, "db4")}
    return {"mode": mode, "type": "generic", "optimal_wavelet": None}


# ── System C: AI Separation ───────────────────────────────────────────────────

class AISeparateRequest(BaseModel):
    signal_id: str
    mode:      str

@app.post("/ai/separate")
def ai_separate(req: AISeparateRequest):
    """
    Run AI source separation ONCE. Stores components server-side.
    Returns component names + base64 waveforms so frontend can do weighted sum.
    This is the slow step — only called when user activates AI system.
    """
    if req.signal_id not in signals:
        return {"error": "Signal not found"}

    signal, sr = signals[req.signal_id]
    mode = req.mode
    components = []   # [{id, name, waveform(float32), sr}]

    try:
        # ── ECG ──────────────────────────────────────────────────────────────
        if mode == "ecg":
            model  = _get_ecg_model()
            result = model.classify_signal(signal, fs=sr)
            class_signals = result.get("class_signals", {})
            counts        = result.get("counts", {})
            summary       = result.get("summary", {})
            for i, (name, wav) in enumerate(class_signals.items()):
                wav = _pad_or_trim(wav.astype(np.float32), len(signal))
                components.append({"id": i, "name": name,
                                   "waveform": wav, "sr": sr})
            extra = {"model_used": "ECGResNet (MIT-BIH)",
                     "beat_summary": summary, "counts": counts,
                     "bands": {k: list(v) for k, v in
                               result.get("bands", {}).items()}}

        # ── Instruments ───────────────────────────────────────────────────────
        elif mode == "instruments":
            model    = _get_music_model()
            tmp_path = _signal_to_tempfile(signal, sr)
            try:
                result = model.separate(tmp_path)
            finally:
                os.unlink(tmp_path)
            stems = result.get("stems", {})
            for i, (name, data) in enumerate(stems.items()):
                wav = _resample_to(data["waveform"], data["sr"], sr)
                wav = _pad_or_trim(wav, len(signal))
                components.append({"id": i, "name": name.capitalize(),
                                   "waveform": wav, "sr": sr})
            extra = {"model_used": "Demucs htdemucs",
                     "stems_found": list(stems.keys())}

        # ── Animals ───────────────────────────────────────────────────────────
        elif mode == "animals":
            model    = _get_animal_model()
            tmp_path = _signal_to_tempfile(signal, sr)
            try:
                result = model.separate_and_classify(tmp_path)
            finally:
                os.unlink(tmp_path)
            sources = result.get("sources", [])
            for i, s in enumerate(sources):
                wav = _resample_to(s["waveform"], s["sr"], sr)
                wav = _pad_or_trim(wav, len(signal))
                name = s.get("label") or f"Source {i+1}"
                components.append({"id": i, "name": name,
                                   "waveform": wav, "sr": sr})
            extra = {"model_used": "YAMNet + Wiener masking",
                     "total_sources": len(sources)}

        # ── Voices ────────────────────────────────────────────────────────────
        elif mode == "voices":
            model    = _get_human_model()
            tmp_path = _signal_to_tempfile(signal, sr)
            try:
                result = model.separate(tmp_path)
            finally:
                os.unlink(tmp_path)
            sources = result.get("sources", [])
            for s in sources:
                wav = _resample_to(s["waveform"], s["sr"], sr)
                wav = _pad_or_trim(wav, len(signal))
                name = s.get("label") or f"Speaker {s.get('speaker_id', s['id'] if 'id' in s else len(components)+1)}"
                components.append({"id": s.get("speaker_id", len(components)),
                                   "name": name, "waveform": wav, "sr": sr})
            extra = {"model_used": "ConvTasNet (asteroid)",
                     "speakers_found": len(sources)}

        else:
            return {"error": f"AI separation not supported for mode: {mode}"}

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e), "mode": mode}

    # Cache components (waveforms stay as numpy arrays server-side)
    ai_components[req.signal_id] = {
        "mode":       mode,
        "sr":         sr,
        "sig_len":    len(signal),
        "components": components,
    }

    # Return component metadata + base64 waveforms for browser-side mixing
    out_components = []
    for c in components:
        rms = float(np.sqrt(np.mean(c["waveform"] ** 2)))
        out_components.append({
            "id":           c["id"],
            "name":         c["name"],
            "rms_energy":   round(rms, 5),
            "duration_sec": round(len(c["waveform"]) / c["sr"], 2),
            # Downsample for waveform display (5000 pts max)
            "samples_b64":  _encode_signal(c["waveform"], c["sr"]),
        })

    return {
        "signal_id":  req.signal_id,
        "mode":       mode,
        "components": out_components,
        **extra,
    }


# ── System C: AI Mix (weighted sum, instant) ─────────────────────────────────

class ComponentGain(BaseModel):
    component_id: int
    gain:         float

class AIMixRequest(BaseModel):
    signal_id: str
    gains:     List[ComponentGain]

@app.post("/ai/mix")
def ai_mix(req: AIMixRequest):
    """
    Weighted sum of AI-separated components.
    output = Σ (component_i * gain_i)
    Instant — separation already done by /ai/separate.
    Returns output_signal_b64 + FFT + spectrogram (like /equalize).
    """
    if req.signal_id not in signals:
        return {"error": "Signal not found"}
    if req.signal_id not in ai_components:
        return {"error": "AI separation not yet run. Call /ai/separate first."}

    signal, sr = signals[req.signal_id]
    cached     = ai_components[req.signal_id]
    comps      = cached["components"]
    sig_len    = cached["sig_len"]

    # Build gain lookup
    gain_map = {g.component_id: g.gain for g in req.gains}

    # Weighted sum
    result = np.zeros(sig_len, dtype=np.float32)
    for c in comps:
        g = gain_map.get(c["id"], 1.0)
        result += c["waveform"] * g

    # Prevent clipping
    peak = np.abs(result).max()
    if peak > 1.0:
        result = result / peak

    return {
        "output_signal_b64":  _encode_signal(result, sr),
        "fft_input":          compute_fft(signal, sr),
        "fft_output":         compute_fft(result, sr),
        "spectrogram_output": compute_spectrogram(result, sr),
    }


# ── AI Analysis (comparison panel — unchanged) ───────────────────────────────

class AIRequest(BaseModel):
    signal_id: str
    mode:      str

@app.post("/ai/run")
def run_ai(req: AIRequest):
    """Quick analysis for the AI Compare panel (not the equalizer)."""
    if req.signal_id not in signals:
        return {"error": "Signal not found"}
    signal, sr = signals[req.signal_id]

    if req.mode == "ecg":
        try:
            model  = _get_ecg_model()
            result = model.classify_signal(signal, fs=sr)
            beats  = result.get("beats", [])
            summary = result.get("summary", {})
            r_peaks = [b["r_peak_sample"] for b in beats]
            if len(r_peaks) > 1:
                rr = np.diff(r_peaks) / sr
                hr, rr_std = round(float(60.0/rr.mean()),1), round(float(rr.std()*1000),1)
            else:
                hr, rr_std = 0.0, 0.0
            dominant = max(summary, key=summary.get) if summary else "—"
            return {"mode":"ecg","heart_rate":hr,"condition":dominant,
                    "peak_count":len(beats),"rr_std_ms":rr_std,
                    "beat_summary":summary,"model_used":"ECGResNet (MIT-BIH)"}
        except Exception as e:
            return {"error": str(e), "mode": "ecg"}

    else:
        from scipy.fft import rfft, rfftfreq
        n   = len(signal)
        fft = np.abs(rfft(signal)) / n
        frq = rfftfreq(n, d=1.0 / sr)
        peak_idx = int(np.argmax(fft))
        return {"mode": req.mode, "model_used": "Spectral analysis",
                "peak_frequency": round(float(frq[peak_idx]),1),
                "rms_energy":    round(float(np.sqrt(np.mean(signal**2))),5),
                "peak_amplitude": round(float(np.max(np.abs(signal))),5)}