import numpy as np
import soundfile as sf
import base64
import io
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from signal_processor import compute_fft, compute_spectrogram
from equalizer_engine import apply_gain
from settings_manager import load_settings, save_settings
#from ai_models import run_ecg_classifier, analyze_audio_signal, compare_results

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage: signal_id -> (numpy_array, sample_rate)
signals = {}


# ═══════════════════════════════════════════════════════
#  ROUTE 1 — Health check  (also /ping for compatibility)
# ═══════════════════════════════════════════════════════
@app.get("/ping")
@app.get("/health")
def ping():
    return {"message": "hello", "signals_cached": len(signals)}


# ═══════════════════════════════════════════════════════
#  ROUTE 2 — Upload a signal file
# ═══════════════════════════════════════════════════════
@app.post("/upload")
async def upload_signal(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    data, sr  = sf.read(io.BytesIO(raw_bytes))

    if len(data.shape) > 1:
        data = data[:, 0]                    # stereo → mono

    data = data.astype(np.float32)
    sid  = str(uuid.uuid4())[:8]
    signals[sid] = (data, sr)

    return {
        "signal_id":   sid,
        "sample_rate": int(sr),
        "duration":    float(len(data) / sr),
    }


# ═══════════════════════════════════════════════════════
#  ROUTE 3 — Get signal samples for waveform drawing
# ═══════════════════════════════════════════════════════
@app.get("/signal/{sid}")
def get_signal(sid: str):
    if sid not in signals:
        return {"error": "Signal not found"}

    data, sr = signals[sid]
    step     = max(1, len(data) // 5000)     # max 5000 points

    return {
        "samples":     data[::step].tolist(),
        "sample_rate": int(sr),
        "full_length": len(data),
    }


# ═══════════════════════════════════════════════════════
#  ROUTE 4 — Equalize a signal
# ═══════════════════════════════════════════════════════
class GainBand(BaseModel):
    band_id:     int
    freq_ranges: List[List[float]]
    gain:        float

class EqualizeRequest(BaseModel):
    signal_id: str
    gains:     List[GainBand]
    method:    str = "fourier"
    wavelet:   str = "db4"

@app.post("/equalize")
def equalize(req: EqualizeRequest):
    if req.signal_id not in signals:
        return {"error": "Signal not found"}

    signal, sr = signals[req.signal_id]
    result     = signal.copy()

    for band in req.gains:
        result = apply_gain(
            result, sr,
            band.freq_ranges,
            band.gain,
            method  = req.method,
            wavelet = req.wavelet
        )

    # Encode output as base64 WAV
    buf = io.BytesIO()
    sf.write(buf, result.astype(np.float32), sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "output_signal_b64":  b64,
        "fft_input":          compute_fft(signal, sr),
        "fft_output":         compute_fft(result, sr),
        "spectrogram_output": compute_spectrogram(result, sr),
    }


# ═══════════════════════════════════════════════════════
#  ROUTE 5 — Get spectrogram of original signal
# ═══════════════════════════════════════════════════════
@app.get("/spectrogram/{sid}")
def get_spectrogram(sid: str):
    if sid not in signals:
        return {"error": "Signal not found"}
    data, sr = signals[sid]
    return {"spectrogram": compute_spectrogram(data, sr)}


# ═══════════════════════════════════════════════════════
#  ROUTE 6 — Compare Fourier vs Wavelet quality
# ═══════════════════════════════════════════════════════
@app.get("/wavelet-compare/{sid}/{mode}")
def wavelet_compare(sid: str, mode: str):
    if sid not in signals:
        return {"error": "Signal not found"}

    signal, sr = signals[sid]

    wavelet_map = {
        "instruments": "db6",
        "animals":     "db6",
        "voices":      "sym5",
        "ecg":         "db4",
        "generic":     "db4",
    }
    wavelet     = wavelet_map.get(mode, "db4")
    test_ranges = [[500, 2000]]
    test_gain   = 0.5

    result_f = apply_gain(signal, sr, test_ranges, test_gain, method="fourier")
    result_w = apply_gain(signal, sr, test_ranges, test_gain,
                          method="wavelet", wavelet=wavelet)

    def snr(orig, mod):
        noise = orig - mod[:len(orig)]
        sp    = float(np.mean(orig ** 2))
        np_   = float(np.mean(noise ** 2)) + 1e-10
        return round(10 * np.log10(sp / np_), 2)

    return {
        "mode":           mode,
        "wavelet_used":   wavelet,
        "snr_fourier_db": snr(signal, result_f),
        "snr_wavelet_db": snr(signal, result_w),
        "better": "wavelet" if snr(signal, result_w) > snr(signal, result_f) else "fourier",
    }


# ═══════════════════════════════════════════════════════
#  ROUTE 7 — Get settings for a mode
# ═══════════════════════════════════════════════════════
@app.get("/settings/{mode_name}")
def get_settings(mode_name: str):
    try:
        return load_settings(mode_name)
    except FileNotFoundError:
        return {"error": f"Mode '{mode_name}' not found"}


# ═══════════════════════════════════════════════════════
#  ROUTE 8 — Save settings for a mode
# ═══════════════════════════════════════════════════════
class SaveSettingsRequest(BaseModel):
    mode_name: str
    config:    dict

@app.post("/settings/save")
def save_settings_route(body: SaveSettingsRequest):
    success = save_settings(body.mode_name, body.config)
    return {"success": success}


# ═══════════════════════════════════════════════════════
#  ROUTE 9 — List all available modes
# ═══════════════════════════════════════════════════════
@app.get("/modes/list")
def list_modes():
    return ["instruments", "animals", "voices", "ecg", "generic"]


# ═══════════════════════════════════════════════════════
#  ROUTE 10 — Run AI model for current mode
# ═══════════════════════════════════════════════════════
"""

class AIRequest(BaseModel):
    signal_id: str
    mode:      str

@app.post("/ai/run")
def run_ai(req: AIRequest):
    if req.signal_id not in signals:
        return {"error": "Signal not found"}

    signal, sr = signals[req.signal_id]

    if req.mode == "ecg":
        result = run_ecg_classifier(signal, sr)
    else:
        result = analyze_audio_signal(signal, sr, req.mode)

    # Also compare AI result signal energy vs equalized if we had output
    # (basic comparison using the original signal as reference)
    result["mode"] = req.mode
    return result


"""