import numpy as np
import soundfile as sf
import base64, io, uuid, os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from signal_processor import compute_fft, compute_spectrogram
from equalizer_engine import (_apply_gain_fourier, apply_wavelet_gains,
                               get_wavelet_band_energies, OPTIMAL_WAVELETS, apply_gain)
from settings_manager import load_settings, save_settings

# AI models are imported lazily inside the loader functions below
# so startup doesn't fail if a package is missing for one mode

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

signals     = {}
ai_stems    = {}   # cache: signal_id -> {mode -> {stem_name: np.array}}
_ecg_analysis_cache = {}  # cache: signal_id -> {result, bands}
CUSTOM_MODES = {"instruments", "animals", "voices", "ecg"}

ECG_DATA_DIR = r"D:\2nd year SBME\Second semester\DSP\project\task 2\data\ecg\ecg"

# ── Lazy AI model loaders ────────────────────────────────────────────────────
_ecg_model = _animal_model = _human_model = _music_model = None

def _get_ecg():
    global _ecg_model
    if _ecg_model is None:
        from ai_ecg import ECGModel
        # ECGResNet (ecg_resnet_mitbih.pt) is the primary backend.
        # Falls back gracefully if the weights file is missing.
        _ecg_model = ECGModel(weights_path="ecg_resnet_mitbih.pt")
    return _ecg_model

def _get_animal():
    global _animal_model
    if _animal_model is None:
        from ai_animal import AnimalModel
        # YAMNet is the primary backend; 1D-CNN is the automatic fallback
        _animal_model = AnimalModel(use_yamnet=True)
    return _animal_model

def _get_human():
    global _human_model
    if _human_model is None:
        from ai_human import HumanModel
        # SepFormer is the primary backend; ConvTasNet is the automatic fallback
        _human_model = HumanModel(use_sepformer=True)
    return _human_model

def _get_music():
    global _music_model
    if _music_model is None:
        from ai_music import MusicModel
        _music_model = MusicModel()
    return _music_model

def _tmp(name): return f"_tmp_{name}_{uuid.uuid4().hex[:6]}.wav"
def _rm(p):
    try: os.remove(p)
    except: pass


# ═══════════════════════════════════════════════════════
#  ROUTE 1 — Health check
# ═══════════════════════════════════════════════════════
@app.get("/api/ping")
@app.get("/api/health")
def ping():
    return {"message": "hello", "signals_cached": len(signals)}


# ═══════════════════════════════════════════════════════
#  ROUTE 2 — Upload
# ═══════════════════════════════════════════════════════
@app.post("/api/upload")
async def upload_signal(file: UploadFile = File(...)):
    raw   = await file.read()
    fname = (file.filename or "").lower()
    try:
        if fname.endswith(".mp3") or fname.endswith(".m4a"):
            from pydub import AudioSegment
            fmt   = "mp3" if fname.endswith(".mp3") else "m4a"
            seg   = AudioSegment.from_file(io.BytesIO(raw), format=fmt)
            buf   = io.BytesIO(); seg.export(buf, format="wav"); buf.seek(0)
            data, sr = sf.read(buf)

        elif fname.endswith(".csv"):
            # ECG / generic CSV support
            # Accepts: single column of numbers, or multi-column (takes first numeric col)
            # Auto-detects sample rate from header if present, else defaults to 360 Hz (MIT-BIH)
            import pandas as pd
            df = pd.read_csv(io.BytesIO(raw))

            # Try to detect sample rate from column names like "360Hz" or header comment
            sr = 360  # default MIT-BIH ECG sample rate
            for col in df.columns:
                c = str(col).lower()
                if 'sample' in c and 'rate' in c:
                    try: sr = int(df[col].iloc[0]); break
                    except: pass
                if c.endswith('hz'):
                    try: sr = int(c.replace('hz','').strip()); break
                    except: pass

            # Take first numeric column
            num_cols = df.select_dtypes(include='number')
            if num_cols.empty:
                raise ValueError("No numeric columns found in CSV")
            data = num_cols.iloc[:, 0].dropna().values.astype(np.float32)

            # Normalize to [-1, 1] if values are outside audio range
            peak = np.abs(data).max()
            if peak > 1.0:
                data = data / peak

        elif fname.endswith(".dat") or fname.endswith(".hea"):
            # WFDB format — save to temp files so wfdb can read them
            import tempfile, shutil
            with tempfile.TemporaryDirectory() as tmpdir:
                base = os.path.splitext(file.filename)[0]
                # Write the uploaded file
                with open(os.path.join(tmpdir, file.filename), "wb") as fh:
                    fh.write(raw)
                from ai_ecg import load_ecg_file
                data, sr = load_ecg_file(os.path.join(tmpdir, file.filename))
                data = data.astype(np.float32)

        elif fname.endswith(".npy"):
            data = np.load(io.BytesIO(raw)).astype(np.float32)
            if data.ndim == 2: data = data[:, 0]
            sr   = 360

        elif fname.endswith(".txt") or fname.endswith(".tsv"):
            data = np.loadtxt(io.StringIO(raw.decode("utf-8", errors="ignore"))).astype(np.float32)
            if data.ndim == 2: data = data[:, 0]
            peak = np.abs(data).max()
            if peak > 1.0: data = data / peak
            sr   = 360

        else:
            data, sr = sf.read(io.BytesIO(raw))

    except Exception as e:
        raise HTTPException(400, detail=f"Cannot read '{file.filename}': {e}")
    if len(data.shape) > 1: data = data[:, 0]
    data = data.astype(np.float32)
    sid  = str(uuid.uuid4())[:8]
    signals[sid] = (data, sr)
    return {"signal_id": sid, "sample_rate": int(sr), "duration": float(len(data)/sr)}


# ═══════════════════════════════════════════════════════
#  ROUTE 3 — Get signal samples
# ═══════════════════════════════════════════════════════
@app.get("/api/signal/{sid}")
def get_signal(sid: str):
    if sid not in signals: return {"error": "Signal not found"}
    data, sr = signals[sid]
    step = max(1, len(data)//5000)
    return {"samples": data[::step].tolist(), "sample_rate": int(sr), "full_length": len(data)}


# ═══════════════════════════════════════════════════════
#  ROUTE 4 — Equalize (Fourier + Wavelet)
# ═══════════════════════════════════════════════════════
class GainBand(BaseModel):
    band_id:     int
    freq_ranges: List[List[float]]
    gain:        float

class EqualizeRequest(BaseModel):
    signal_id:     str
    freq_gains:    List[GainBand]
    wavelet_gains: Optional[List[GainBand]] = []
    mode:          str = "generic"

@app.post("/api/equalize")
def equalize(req: EqualizeRequest):
    if req.signal_id not in signals: return {"error": "Signal not found"}
    signal, sr = signals[req.signal_id]
    result = signal.copy()

    for b in req.freq_gains:
        result = _apply_gain_fourier(result, sr, b.freq_ranges, b.gain)

    wavelet_used = None
    if req.mode in CUSTOM_MODES and req.wavelet_gains:
        wavelet_used = OPTIMAL_WAVELETS.get(req.mode, "db4")
        bands = [{"id": b.band_id, "freq_ranges": b.freq_ranges, "gain": b.gain}
                 for b in req.wavelet_gains]
        result = apply_wavelet_gains(result, sr, bands, wavelet_used)

    buf = io.BytesIO()
    sf.write(buf, result.astype(np.float32), sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode()

    wav_in, wav_out = [], []
    if req.mode in CUSTOM_MODES and req.wavelet_gains and wavelet_used:
        bi = [{"id": b.band_id, "freq_ranges": b.freq_ranges, "name": ""}
              for b in req.wavelet_gains]
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


# ═══════════════════════════════════════════════════════
#  ROUTE 5 — Spectrogram
# ═══════════════════════════════════════════════════════
@app.get("/api/spectrogram/{sid}")
def get_spectrogram(sid: str):
    if sid not in signals: return {"error": "Signal not found"}
    data, sr = signals[sid]
    return {"spectrogram": compute_spectrogram(data, sr)}


# ═══════════════════════════════════════════════════════
#  ROUTE 6 — Wavelet compare
# ═══════════════════════════════════════════════════════
@app.get("/api/wavelet-compare/{sid}/{mode}")
def wavelet_compare(sid: str, mode: str):
    if sid not in signals: return {"error": "Signal not found"}
    if mode not in CUSTOM_MODES: return {"error": "Custom modes only"}
    signal, sr = signals[sid]
    wavelet = OPTIMAL_WAVELETS.get(mode, "db4")
    tr, tg  = [[500, 2000]], 0.5
    rf = _apply_gain_fourier(signal, sr, tr, tg)
    rw = apply_wavelet_gains(signal, sr, [{"id":1,"freq_ranges":tr,"gain":tg}], wavelet)
    def snr(o, m):
        n = o - m[:len(o)]; sp = float(np.mean(o**2))
        return round(10*np.log10(sp/(float(np.mean(n**2))+1e-10)), 2)
    sf_ = snr(signal, rf); sw = snr(signal, rw)
    return {"mode": mode, "wavelet_used": wavelet,
            "snr_fourier_db": sf_, "snr_wavelet_db": sw,
            "better": "wavelet" if sw > sf_ else "fourier"}


# ═══════════════════════════════════════════════════════
#  ROUTE 7 — Get settings
# ═══════════════════════════════════════════════════════
@app.get("/api/settings/{mode_name}")
def get_settings(mode_name: str):
    try:    return load_settings(mode_name)
    except: return {"error": f"Mode '{mode_name}' not found"}


# ═══════════════════════════════════════════════════════
#  ROUTE 8 — Save settings
# ═══════════════════════════════════════════════════════
class SaveSettingsRequest(BaseModel):
    mode_name: str
    config:    dict

@app.post("/api/settings/save")
def save_settings_route(body: SaveSettingsRequest):
    return {"success": save_settings(body.mode_name, body.config)}


# ═══════════════════════════════════════════════════════
#  ROUTE 9 — List modes
# ═══════════════════════════════════════════════════════
@app.get("/api/modes/list")
def list_modes():
    return ["instruments", "animals", "voices", "ecg", "generic"]


# ═══════════════════════════════════════════════════════
#  ROUTE 10 — Mode info
# ═══════════════════════════════════════════════════════
@app.get("/api/mode-info/{mode}")
def mode_info(mode: str):
    if mode in CUSTOM_MODES:
        return {"mode": mode, "type": "custom",
                "optimal_wavelet": OPTIMAL_WAVELETS.get(mode, "db4")}
    return {"mode": mode, "type": "generic", "optimal_wavelet": None}


# ═══════════════════════════════════════════════════════
#  ROUTE 11 — AI stems discovery
#  Called once per signal+mode to get component names.
#  Frontend uses names to build AI equalizer sliders.
# ═══════════════════════════════════════════════════════
class AIStemRequest(BaseModel):
    signal_id: str
    mode:      str

@app.post("/api/ai-stems")
def get_ai_stems(req: AIStemRequest):
    if req.signal_id not in signals: return {"error": "Signal not found"}
    if req.mode not in CUSTOM_MODES:  return {"error": "Custom modes only"}

    signal, sr = signals[req.signal_id]
    cache_key  = f"{req.signal_id}_{req.mode}"

    # Return cached stems if available
    if cache_key in ai_stems:
        names = list(ai_stems[cache_key].keys())
        return {"mode": req.mode, "stem_names": names, "count": len(names), "cached": True}

    tmp = _tmp("stems")
    sf.write(tmp, signal, sr)
    try:
        stems: dict[str, np.ndarray] = {}

        if req.mode == "instruments":
            result = _get_music().separate(tmp)
            for name, data in result.get("stems", {}).items():
                raw = data["waveform"]
                if data["sr"] != sr:
                    import librosa
                    raw = librosa.resample(raw, orig_sr=data["sr"], target_sr=sr)
                stems[name] = raw.astype(np.float32)

        elif req.mode == "animals":
            result = _get_animal().separate_signal(signal, sr)
            for i, src in enumerate(result.get("sources", [])):
                label = src.get("label", f"source_{i+1}")
                wav   = np.array(src["waveform"], dtype=np.float32)
                # Resample to signal sr if backend used a different rate
                src_sr = src.get("sr", sr)
                if src_sr != sr:
                    import librosa as _lb
                    wav = _lb.resample(wav, orig_sr=src_sr, target_sr=sr)
                stems[label] = wav

        elif req.mode == "voices":
            result = _get_human().separate(tmp)
            for src in result.get("sources", []):
                label = src.get("label") or f"speaker_{src['speaker_id']}"
                stems[label] = np.array(src["waveform"], dtype=np.float32)

        elif req.mode == "ecg":
            result = _get_ecg().classify_signal(signal, sr)
            # Use the pre-built class_signals from the pipeline
            class_sigs = result.get("class_signals", {})
            # Only include classes that actually have beats
            counts = result.get("counts", {})
            from ai_ecg import CLASS_NAMES as ECG_CLASS_NAMES
            for i, name in enumerate(ECG_CLASS_NAMES):
                if counts.get(i, 0) > 0 and name in class_sigs:
                    stems[name] = class_sigs[name].astype(np.float32)
            # Store bands for /api/ecg/equalize
            _ecg_analysis_cache[req.signal_id] = {
                "result": result,
                "bands":  result.get("bands", {}),
            }
            if not stems:
                stems = {"ECG Signal": signal.copy()}

        # Cache for use in /api/ai-equalize
        ai_stems[cache_key] = stems
        return {"mode": req.mode, "stem_names": list(stems.keys()),
                "count": len(stems), "cached": False}

    except Exception as e:
        import traceback
        return {"error": str(e), "detail": traceback.format_exc()}
    finally:
        _rm(tmp)



# ═══════════════════════════════════════════════════════
#  ROUTE 11b — Per-stem data (waveform + FFT + audio)
# ═══════════════════════════════════════════════════════
class StemDataRequest(BaseModel):
    signal_id: str
    mode:      str
    stem_name: str

@app.post("/api/ai-stem-data")
def get_stem_data(req: StemDataRequest):
    cache_key = f"{req.signal_id}_{req.mode}"
    if cache_key not in ai_stems:
        return {"error": "Stems not found — run /api/ai-stems first"}
    if req.stem_name not in ai_stems[cache_key]:
        return {"error": f"Stem not found"}

    stem_sig = ai_stems[cache_key][req.stem_name]
    _, sr    = signals[req.signal_id]

    step    = max(1, len(stem_sig) // 5000)
    samples = stem_sig[::step].tolist()
    fft_data = compute_fft(stem_sig, sr)

    buf = io.BytesIO()
    sf.write(buf, stem_sig.astype(np.float32), sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "stem_name":   req.stem_name,
        "samples":     samples,
        "sample_rate": int(sr),
        "duration":    round(len(stem_sig) / sr, 2),
        "fft":         fft_data,
        "audio_b64":   b64,
        "rms_energy":  round(float(np.sqrt(np.mean(stem_sig**2))), 6),
    }


# ═══════════════════════════════════════════════════════
#  ROUTE 12 — AI equalize (System C)
#  Mixes AI-separated stems weighted by slider gains.
#  gain dict: {"drums": 1.5, "bass": 0.3, ...}
# ═══════════════════════════════════════════════════════
class AIEqualizeRequest(BaseModel):
    signal_id: str
    mode:      str
    gains:     dict   # {stem_name: float}

@app.post("/api/ai-equalize")
def ai_equalize(req: AIEqualizeRequest):
    if req.signal_id not in signals: return {"error": "Signal not found"}
    if req.mode not in CUSTOM_MODES:  return {"error": "Custom modes only"}

    signal, sr    = signals[req.signal_id]
    cache_key     = f"{req.signal_id}_{req.mode}"

    # Auto-run stem separation if not cached yet
    if cache_key not in ai_stems:
        stem_req = AIStemRequest(signal_id=req.signal_id, mode=req.mode)
        res = get_ai_stems(stem_req)
        if "error" in res: return res

    stems = ai_stems[cache_key]
    if not stems: return {"error": "No stems available"}

    # Mix: output = sum(stem * gain)
    min_len = min(len(s) for s in stems.values())
    output  = np.zeros(min_len, dtype=np.float32)
    stem_info = []

    for name, stem_sig in stems.items():
        gain = float(req.gains.get(name, 1.0))
        s    = stem_sig[:min_len] if len(stem_sig) >= min_len \
               else np.pad(stem_sig, (0, min_len - len(stem_sig)))
        output += s * gain
        stem_info.append({
            "name":       name,
            "gain_used":  gain,
            "rms_energy": float(np.sqrt(np.mean(s**2))),
        })

    # Normalize to prevent clipping
    peak = np.abs(output).max()
    if peak > 1.0: output /= peak

    buf = io.BytesIO()
    sf.write(buf, output.astype(np.float32), sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "output_signal_b64":  b64,
        "fft_input":          compute_fft(signal, sr),
        "fft_output":         compute_fft(output, sr),
        "spectrogram_output": compute_spectrogram(output, sr),
        "stems":              stem_info,
        "stem_names":         list(stems.keys()),
    }


# ═══════════════════════════════════════════════════════
#  ROUTE 13 — AI run (analysis / comparison)
# ═══════════════════════════════════════════════════════
class AIRequest(BaseModel):
    signal_id: str
    mode:      str

def _signal_stats(signal: np.ndarray, sr: int) -> dict:
    """Compute analysis stats that the frontend expects for all modes."""
    fft_result  = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), d=1.0/sr)
    magnitudes  = np.abs(fft_result) / len(signal)
    peak_idx    = int(np.argmax(magnitudes))
    rms         = float(np.sqrt(np.mean(signal**2)))
    peak_amp    = float(np.max(np.abs(signal)))
    # SNR vs silence baseline
    noise_floor = float(np.mean(magnitudes)) + 1e-10
    snr_db      = round(10 * np.log10((rms**2) / (noise_floor**2 + 1e-10)), 2)
    return {
        "peak_frequency": float(frequencies[peak_idx]),
        "rms_energy":     round(rms, 6),
        "peak_amplitude": round(peak_amp, 6),
        "snr_db":         snr_db,
    }

@app.post("/api/ai/run")
def run_ai(req: AIRequest):
    if req.signal_id not in signals: return {"error": "Signal not found"}
    signal, sr = signals[req.signal_id]
    tmp = _tmp("ai")
    sf.write(tmp, signal, sr)
    try:
        stats = _signal_stats(signal, sr)

        if req.mode == "ecg":
            try:
                result  = _get_ecg().classify_signal(signal, sr)
                beats   = result.get("beats", [])
                summary = result.get("summary", {})
                counts  = result.get("counts", {})
                bands   = result.get("bands", {})
                r_times = [b["r_peak_time_sec"] for b in beats if "r_peak_time_sec" in b]
                if len(r_times) > 1:
                    rr_int    = np.diff(r_times)
                    heart_rate  = round(60.0 / float(np.mean(rr_int)), 1)
                    rr_std_ms   = round(float(np.std(rr_int)) * 1000, 1)
                else:
                    heart_rate, rr_std_ms = 0.0, 0.0
                dominant = max(summary, key=summary.get) if summary else "Unknown"
                # Build slider config from discovered bands
                from ai_ecg import ECGModel as _ECGModel, CLASS_NAMES as ECG_CLASS_NAMES
                settings = _ECGModel.bands_to_settings(
                    bands, {c: 1.0 for c in range(5)}, noise_gain=1.0)
                # Cache analysis result
                _ecg_analysis_cache[req.signal_id] = {"result": result, "bands": bands}
                return {"mode":        "ecg",
                        "heart_rate":  heart_rate,
                        "condition":   dominant,
                        "peak_count":  len(beats),
                        "rr_std_ms":   rr_std_ms,
                        "summary":     summary,
                        "counts":      counts,
                        "bands":       {str(k): list(v) for k, v in bands.items()},
                        "settings":    settings,
                        **stats}
            except Exception as e:
                return {"mode": "ecg", "error": str(e), **stats}

        elif req.mode == "animals":
            try:
                result  = _get_animal().separate_signal(signal, sr)
                sources = result.get("sources", [])
                top     = max(sources, key=lambda s: s.get("confidence", 0)) if sources else {}
                return {"mode":       "animals",
                        "backend":    result.get("backend", "unknown"),
                        "elapsed_ms": result.get("elapsed_ms", 0),
                        "sources":    [{"label":      s["label"],
                                        "confidence": round(float(s["confidence"]), 4)}
                                       for s in sources],
                        "top_label":  top.get("label", "—"),
                        **stats}
            except Exception as e:
                return {"mode": "animals", "error": str(e), **stats}

        elif req.mode == "voices":
            try:
                result  = _get_human().separate(tmp)
                sources = result.get("sources", [])
                return {"mode":          "voices",
                        "speaker_count": len(sources),
                        "backend":       result.get("backend", "unknown"),
                        "elapsed_ms":    result.get("elapsed_ms", 0),
                        "speakers":      [{"speaker_id":   s["speaker_id"],
                                           "label":        s.get("label", ""),
                                           "duration_sec": s["duration_sec"]}
                                          for s in sources],
                        **stats}
            except Exception as e:
                return {"mode": "voices", "error": str(e), **stats}

        elif req.mode == "instruments":
            try:
                result = _get_music().separate(tmp)
                stems  = result.get("stems", {})
                return {"mode": "instruments", "stems": list(stems.keys()), **stats}
            except Exception as e:
                return {"mode": "instruments", "error": str(e), **stats}

        else:
            return {"mode": req.mode, **stats}

    except Exception as e:
        return {"error": str(e)}
    finally:
        _rm(tmp)

# ═══════════════════════════════════════════════════════
#  ROUTE 14 — Voices: Wavelet vs AI comparison
#  POST /api/voices/compare
# ═══════════════════════════════════════════════════════
class VoicesCompareRequest(BaseModel):
    signal_id: str
    scales:    List[float] = [1.0, 1.0]
    wavelet:   str         = "coif3"
    level:     int         = 6

@app.post("/api/voices/compare")
def voices_compare(req: VoicesCompareRequest):
    if req.signal_id not in signals:
        return {"error": "Signal not found"}

    signal, sr = signals[req.signal_id]
    try:
        results = _get_human().compare_methods(
            signal  = signal,
            sr      = sr,
            scales  = req.scales,
            wavelet = req.wavelet,
            level   = req.level,
        )

        def _enrich(r):
            out = r.get("output")
            if out is None:
                return r
            out_np = np.array(out, dtype=np.float32)
            buf = io.BytesIO()
            sf.write(buf, out_np, sr, format="WAV")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return {
                **{k: v for k, v in r.items() if k != "output"},
                "fft_output":         compute_fft(out_np, sr),
                "spectrogram_output": compute_spectrogram(out_np, sr),
                "output_signal_b64":  b64,
            }

        def _strip(r):
            return {k: v for k, v in r.items()
                    if not isinstance(v, np.ndarray)
                    and not (isinstance(v, list) and v
                             and isinstance(v[0], np.ndarray))}

        wav_r = _strip(_enrich(results["wavelet"]))
        ai_r  = _strip(_enrich(results["ai"]))
        ai_r["backend"] = _get_human()._backend

        return {"wavelet": wav_r, "ai": ai_r}

    except Exception as e:
        import traceback
        return {"error": str(e), "detail": traceback.format_exc()}


# ═══════════════════════════════════════════════════════
#  ROUTE 15 — Animals: YAMNet vs Wavelet comparison
#  POST /api/animals/compare
# ═══════════════════════════════════════════════════════
class AnimalsCompareRequest(BaseModel):
    signal_id: str
    scales:    List[float] = [1.0, 1.0, 1.0, 1.0]
    wavelet:   str         = "db4"
    level:     int         = 6

@app.post("/api/animals/compare")
def animals_compare(req: AnimalsCompareRequest):
    """
    Run both wavelet equalisation and YAMNet isolation on the same signal,
    apply per-animal scales, then return comparison metrics + audio.

    Response identical structure to /api/voices/compare:
    {
      "wavelet": { snr, si_snr, prd, lsd, time_ms, fft_output, spectrogram_output, output_signal_b64 },
      "yamnet":  { same fields + backend + error }
    }
    """
    if req.signal_id not in signals:
        return {"error": "Signal not found"}

    signal, sr = signals[req.signal_id]
    try:
        results = _get_animal().compare_methods(
            signal  = signal,
            sr      = sr,
            scales  = req.scales,
            wavelet = req.wavelet,
            level   = req.level,
        )

        def _enrich(r):
            out = r.get("output")
            if out is None:
                return r
            out_np = np.array(out, dtype=np.float32)
            buf = io.BytesIO()
            sf.write(buf, out_np, sr, format="WAV")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return {
                **{k: v for k, v in r.items() if k != "output"},
                "fft_output":         compute_fft(out_np, sr),
                "spectrogram_output": compute_spectrogram(out_np, sr),
                "output_signal_b64":  b64,
            }

        def _strip(r):
            return {k: v for k, v in r.items()
                    if not isinstance(v, np.ndarray)
                    and not (isinstance(v, list) and v
                             and isinstance(v[0], np.ndarray))}

        wav_r = _strip(_enrich(results["wavelet"]))
        ai_r  = _strip(_enrich(results["yamnet"]))
        ai_r["backend"] = _get_animal()._backend

        return {"wavelet": wav_r, "yamnet": ai_r}

    except Exception as e:
        import traceback
        return {"error": str(e), "detail": traceback.format_exc()}


# ═══════════════════════════════════════════════════════
#  ROUTE 16 — ECG: AI-driven per-class equalisation
#  POST /api/ecg/equalize
#  Uses the ResNet-discovered frequency bands as equalizer windows.
#  This IS System C for ECG mode.
# ═══════════════════════════════════════════════════════
class ECGEqualizeRequest(BaseModel):
    signal_id:  str
    gains:      dict        # {class_id_str: float}  e.g. {"0":1.0,"2":0.0}
    noise_gain: float = 1.0

@app.post("/api/ecg/equalize")
def ecg_equalize_route(req: ECGEqualizeRequest):
    """
    Apply per-arrhythmia-class Gaussian frequency gains to the ECG signal.

    Runs classify_signal() first (cached) to discover frequency bands,
    then applies the per-class Gaussian windows with the given gains.

    Response:  same structure as /api/equalize
    """
    if req.signal_id not in signals:
        return {"error": "Signal not found"}

    signal, sr = signals[req.signal_id]

    # Convert str keys → int
    gains = {int(k): float(v) for k, v in req.gains.items()}

    # Get cached analysis or run fresh
    if req.signal_id in _ecg_analysis_cache:
        cached = _ecg_analysis_cache[req.signal_id]
        bands  = cached["bands"]
        result = cached["result"]
    else:
        result = _get_ecg().classify_signal(signal, sr)
        bands  = result.get("bands", {})
        _ecg_analysis_cache[req.signal_id] = {"result": result, "bands": bands}

    from ai_ecg import ecg_equalise, CLASS_NAMES as ECG_CLASS_NAMES, ECGModel as _ECGModel
    output = ecg_equalise(signal, gains, bands,
                          noise_gain=req.noise_gain, fs=sr)

    buf = io.BytesIO()
    sf.write(buf, output.astype(np.float32), sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode()

    # Build settings JSON so frontend can display the band info
    settings = _ECGModel.bands_to_settings(bands, gains, req.noise_gain)

    return {
        "output_signal_b64":  b64,
        "fft_input":          compute_fft(signal, sr),
        "fft_output":         compute_fft(output, sr),
        "spectrogram_output": compute_spectrogram(output, sr),
        "bands":              {str(k): list(v) for k, v in bands.items()},
        "settings":           settings,
        "summary":            result.get("summary", {}),
        "counts":             result.get("counts", {}),
    }


# ═══════════════════════════════════════════════════════
#  ROUTE 17 — ECG: Analyse (classify beats, find bands)
#  POST /api/ecg/analyse
#  Called once per uploaded ECG to get beat classification + bands.
# ═══════════════════════════════════════════════════════
class ECGAnalyseRequest(BaseModel):
    signal_id: str

@app.post("/api/ecg/analyse")
def ecg_analyse_route(req: ECGAnalyseRequest):
    """
    Run full ECG pipeline: segment → classify → find freq bands.

    Response
    --------
    {
      "beats":     [{beat_idx, label, symbol, confidence, all_probs,
                     r_peak_sample, r_peak_time_sec}, ...],
      "summary":   {"Normal (N)": 120, "PVC (V)": 30, ...},
      "counts":    {0: 120, 2: 30, ...},
      "bands":     {"0": [f_lo, f_hi, f_peak], ...},
      "settings":  { full ecg_eq_settings.json structure },
      "heart_rate": float,
      "rr_std_ms":  float,
    }
    """
    if req.signal_id not in signals:
        return {"error": "Signal not found"}

    signal, sr = signals[req.signal_id]

    try:
        result  = _get_ecg().classify_signal(signal, sr)
        # Check if classify_signal itself returned an error
        if result.get("error"):
            return {"error": result["error"],
                    "beats": [], "summary": {}, "counts": {},
                    "bands": {}, "settings": {}, "heart_rate": 0.0, "rr_std_ms": 0.0}

        beats   = result.get("beats", [])
        summary = result.get("summary", {})
        counts  = result.get("counts", {})
        bands   = result.get("bands", {})

        _ecg_analysis_cache[req.signal_id] = {"result": result, "bands": bands}

        r_times = [b["r_peak_time_sec"] for b in beats]
        if len(r_times) > 1:
            rr_int     = np.diff(r_times)
            heart_rate = round(60.0 / float(np.mean(rr_int)), 1)
            rr_std_ms  = round(float(np.std(rr_int)) * 1000, 1)
        else:
            heart_rate, rr_std_ms = 0.0, 0.0

        from ai_ecg import ECGModel as _ECGModel
        settings = _ECGModel.bands_to_settings(
            bands, {c: 1.0 for c in range(5)}, noise_gain=1.0)

        # Trim beat list to first 500 to keep response size reasonable
        return {
            "beats"      : beats[:500],
            "total_beats": len(beats),
            "summary"    : summary,
            "counts"     : counts,
            "bands"      : {str(k): list(v) for k, v in bands.items()},
            "settings"   : settings,
            "heart_rate" : heart_rate,
            "rr_std_ms"  : rr_std_ms,
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "detail": traceback.format_exc()}


# ═══════════════════════════════════════════════════════
#  ROUTE 18 — ECG: Wavelet vs ResNet comparison
#  POST /api/ecg/compare
# ═══════════════════════════════════════════════════════
class ECGCompareRequest(BaseModel):
    signal_id:  str
    gains:      dict  = {}     # {class_id_str: float}
    wavelet:    str   = "db4"
    level:      int   = 6
    noise_gain: float = 1.0

@app.post("/api/ecg/compare")
def ecg_compare_route(req: ECGCompareRequest):
    """
    Run both wavelet equalisation and ResNet-guided equalisation and
    return comparison metrics.  Structure mirrors /api/voices/compare.
    """
    if req.signal_id not in signals:
        return {"error": "Signal not found"}

    signal, sr = signals[req.signal_id]
    gains = {int(k): float(v) for k, v in req.gains.items()}

    # Fetch cached bands or run fresh
    if req.signal_id in _ecg_analysis_cache:
        bands = _ecg_analysis_cache[req.signal_id]["bands"]
    else:
        bands = None   # compare_methods will compute them

    try:
        results = _get_ecg().compare_methods(
            signal  = signal,
            fs      = sr,
            gains   = gains,
            bands   = bands,
            wavelet = req.wavelet,
            level   = req.level,
        )

        def _enrich(r):
            out = r.get("output")
            if out is None: return r
            out_np = np.array(out, dtype=np.float32)
            buf = io.BytesIO()
            sf.write(buf, out_np, sr, format="WAV")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return {
                **{k: v for k, v in r.items() if k != "output"},
                "fft_output":         compute_fft(out_np, sr),
                "spectrogram_output": compute_spectrogram(out_np, sr),
                "output_signal_b64":  b64,
            }

        def _strip(r):
            return {k: v for k, v in r.items()
                    if not isinstance(v, np.ndarray)
                    and not (isinstance(v, list) and v
                             and isinstance(v[0], np.ndarray))}

        wav_r    = _strip(_enrich(results["wavelet"]))
        resnet_r = _strip(_enrich(results["resnet"]))
        # Serialise bands
        if "bands" in resnet_r and isinstance(resnet_r["bands"], dict):
            resnet_r["bands"] = {str(k): list(v)
                                 for k, v in resnet_r["bands"].items()}
        resnet_r["backend"] = _get_ecg()._backend

        return {"wavelet": wav_r, "resnet": resnet_r}

    except Exception as e:
        import traceback
        return {"error": str(e), "detail": traceback.format_exc()}