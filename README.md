# ⚡ Signal Equalizer — DSP Task 2

A full-stack web application for interactive signal equalization using Fourier transforms, Wavelet decomposition, and AI-based source separation. Built with a Python FastAPI backend and a pure HTML/CSS/JavaScript frontend.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [How to Use](#how-to-use)
- [Mode System](#mode-system)
- [Three Equalization Systems](#three-equalization-systems)
- [Settings File Format](#settings-file-format)
- [API Reference](#api-reference)
- [Technical Details](#technical-details)
- [Team](#team)

---

## Overview

The Signal Equalizer allows users to upload an audio or ECG signal, then interactively adjust the magnitude of specific frequency components using sliders. Changes are reflected immediately across the waveform viewers, frequency spectrum chart, wavelet energy chart, and spectrograms.

The app supports three architecturally distinct equalization systems and two mode types:

| Mode Type | Modes | Available Systems |
|---|---|---|
| **Generic** | Generic | System A — Frequency (FFT) only |
| **Custom** | Instruments, Animals, Voices, ECG | System A (FFT) + System B (Wavelet) + System C (AI) |

---

## Features

### Signal Display
- **Two linked cine viewers** — input (blue) and output (red) waveforms displayed side by side
- **Synchronized scrolling** — zoom and pan on either viewer and both update identically
- **Playback controls** — Play, Pause, Stop, Speed control (0.25× to 4×)
- **Zoom & Pan** — buttons and mouse scroll wheel, with drag-to-pan
- **Input audio playback** — the global **🔊 Input** button in the playback bar always plays the raw uploaded signal
- **Per-system output audio** — each system row has its own **🔊 Play** button that plays only that system's processed output

### Frequency Analysis
- **FFT frequency spectrum** — input and output plotted together with toggle-able visibility
- **Linear scale** — full frequency range 0 Hz to Nyquist
- **Audiogram scale** — logarithmic X axis at 125, 250, 500, 1k, 2k, 4k, 8k Hz (ISO 8253-1 hearing test format)
- Scale toggle does not interrupt or reset any functionality
- Interactive: scroll-to-zoom, drag-to-pan, double-click to reset, hover tooltip with exact values

### Spectrograms
- **Two spectrograms** — input and output in a narrow dedicated column (190 px)
- **High-resolution axes** — both frequency (Hz/kHz) and time (s/ms) axes with major + minor ticks, device-pixel-ratio aware rendering
- **Axes adapt** to the signal's actual sample rate so ECG (360 Hz) shows the correct 0–180 Hz range
- **Live update** — output spectrogram updates within ~55 ms of any slider change
- **Toggle show/hide** — single button hides both spectrograms

### Equalizer — Generic Mode
- **Schema-driven** — load a JSON schema file to auto-generate sliders for arbitrary frequency bands
- **Manual band adding** — click **＋ Add Band Manually** to define bands one by one
- **Save & Load** — export your current band configuration as JSON and reload it later
- **Purely FFT-based** — no wavelet or AI controls appear in this mode

### Equalizer — Custom Modes (Three Systems)

Custom modes display three independent equalization systems with a clear visual hierarchy:

- **System A — Frequency** occupies a full-width top row (largest chart, most detail)
- **System B — Wavelet** and **System C — AI** share a side-by-side bottom row

Only one system is active at a time. The active system drives the output viewers and spectrograms. Inactive systems are visually dimmed and disabled.

---

## Project Structure

```
Task2_Signal_Equalizer/
│
├── backend/
│   ├── main.py                        ← FastAPI server — all API routes
│   ├── signal_processor.py            ← FFT, IFFT, Spectrogram, Wavelet functions
│   ├── equalizer_engine.py            ← Core equalization: FFT + Wavelet (exclusive DWT)
│   ├── settings_manager.py            ← load/save/validate JSON settings files
│   ├── ai_models.py                   ← AI signal analysis functions
│   ├── ai_music.py                    ← Music stem separation via Demucs
│   ├── ai_animal.py                   ← Animal sound classification + separation
│   ├── ai_human.py                    ← Speaker separation via ConvTasNet (asteroid)
│   ├── ai_ecg.py                      ← ECG arrhythmia classification (ResNet)
│   ├── ai_shared.py                   ← Shared audio loading utilities
│   ├── ecg_inference.py               ← Standalone ECG pipeline (ECGPipeline class)
│   └── generate_synthetic.py          ← Run once to create the test WAV file
│
├── frontend/
│   └── index.html                     ← Complete frontend (HTML + CSS + JS)
│
├── ecg_pretrainedmodel/
│   ├── ECG_standalone_version/
│   │   ├── ecg_resnet_mitbih.pt       ← Pretrained ECG ResNet weights
│   │   ├── ecg_inference.py           ← Standalone inference module
│   │   └── ecg_eq_settings.json       ← ECG equalizer band configuration
│   ├── ECG_Inference_Equalizer.ipynb  ← Training + evaluation notebook
│   └── ECG_Setup_PhysioNet.ipynb      ← Dataset setup notebook
│
└── data/
    ├── settings/
    │   ├── instruments.json           ← Musical Instruments mode config
    │   ├── animals.json               ← Animal Sounds mode config
    │   ├── voices.json                ← Human Voices mode config
    │   ├── ecg.json                   ← ECG Abnormalities mode config
    │   └── generic.json               ← Empty template for Generic mode
    └── samples/
        ├── synthetic_signal.wav       ← Generated test signal (10 pure sine waves)
        └── ...                        ← Your own audio/ECG files
```

---

## Installation

### Prerequisites
- Python 3.10 or newer
- A modern browser (Chrome recommended)
- ffmpeg installed and on PATH (required by Demucs for music separation)

### Step 1 — Clone or download the project

```bash
git clone <repository-url>
cd Task2_Signal_Equalizer
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate
```

### Step 3 — Install core dependencies

```bash
pip install fastapi uvicorn numpy scipy pywavelets soundfile python-multipart
```

### Step 4 — Install AI dependencies (optional but recommended)

Each AI system has its own dependencies:

```bash
# System C — Music (Instruments mode)
pip install demucs
pip install soundfile==0.11.0      # fixes SoundFileRuntimeError with Demucs

# System C — Animals
pip install torch torchaudio

# System C — Human Voices
pip install asteroid

# System C — ECG
pip install torch wfdb             # wfdb for WFDB file format support

# ECG AI analysis panel
pip install neurokit2
```

> **Windows note:** If Demucs fails with `[WinError 2]`, the app automatically falls back to `python -m demucs`. If you see `No module named demucs`, run `pip install demucs` inside your active `.venv`. If you see a `TorchCodec` error, run `pip install torchcodec` or downgrade torchaudio: `pip install torchaudio==2.1.2`.

### Step 5 — Install ffmpeg (required for Demucs)

**Windows:**
```bash
winget install ffmpeg
```
Or download from https://ffmpeg.org/download.html and add to PATH.

**Verify:**
```bash
ffmpeg -version
```

### Step 6 — Generate the synthetic test signal

```bash
cd backend
python generate_synthetic.py
```

This creates `data/samples/synthetic_signal.wav` — a 5-second signal containing 10 pure sine waves at 100, 300, 500, 1000, 2000, 4000, 6000, 8000, 10000, and 12000 Hz, ideal for verifying equalization behaviour.

---

## Running the App

### Step 1 — Start the backend server

```bash
cd backend
uvicorn main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Started reloader process
```

### Step 2 — Open the frontend

Open `frontend/index.html` directly in Chrome. The toolbar should show **● Backend OK** in green. If it shows **Backend Offline**, make sure Step 1 is running.

---

## How to Use

### Uploading a Signal

1. Click **📂 Load Signal** in the toolbar
2. Select any `.wav`, `.mp3`, `.ogg`, or `.flac` file
3. The input waveform, spectrogram, and FFT chart appear automatically
4. All sliders start at 1.0× (no change)

### Moving Sliders

- Drag any slider up or down — all charts update immediately with a ~55 ms debounce
- The value label above each slider shows the current gain (e.g. `0.0×` = silence, `2.0×` = double amplitude)
- Each system has its own **↺ Reset** button to return all its sliders to 1.0×

### Switching the Active System

Three methods — all equivalent:
1. Click the **tab bar** at the top of the custom layout (System A / System B / System C)
2. Click anywhere on a **dimmed (inactive) system panel** — it activates immediately
3. Click the **panel header** of any system

When you switch, the newly activated system's sliders reset to 1.0× so you start from a clean state.

### Audio Playback

| Control | What it plays |
|---|---|
| **🔊 Input** (playback bar) | The raw uploaded signal — always, regardless of active system |
| **🔊 Play** (System A row) | The FFT-equalized output |
| **🔊 Play** (System B row) | The Wavelet-equalized output |
| **🔊 Play** (System C row) | The AI-mixed output |
| **▶ Play** (playback bar) | Scrolls both cine viewers forward (visual only) |

### Frequency Scale

- **Linear** — 0 Hz to Nyquist, evenly spaced
- **Audiogram** — 125 Hz to 8 kHz, logarithmic (ISO 8253-1 standard)

---

## Mode System

### Generic Mode

For arbitrary frequency-based equalization. Purely FFT-based — no Wavelet or AI systems appear.

Load a JSON schema file to auto-generate sliders, or add bands manually. Save and reload configurations as JSON.

### Custom Modes

Available modes: **Instruments**, **Animals**, **Voices**, **ECG**

Each shows all three equalization systems. Use the system tab bar or click any panel to switch between them.

---

## Three Equalization Systems

### System A — Frequency Domain (Blue)

Uses the Fast Fourier Transform (FFT). Each slider scales the FFT magnitude bins that fall within the slider's configured frequency range.

- Best for **precise frequency targeting** (e.g. exactly 200–400 Hz)
- Chart: continuous input/output frequency spectrum
- Works in all modes including Generic

### System B — Wavelet Domain (Purple)

Uses the Discrete Wavelet Transform (DWT). Each slider controls the RMS energy of the wavelet coefficients that correspond to its frequency range.

**Algorithm — Exclusive DWT Level Assignment:**

The DWT decomposes a signal into dyadic frequency octaves. Each `coeffs` index maps to a specific frequency range:

```
coeffs[0]   = approximation  →  [0,       sr/2^L    ]  (DC + lowest)
coeffs[1]   = detail cD_L    →  [sr/2^(L+1), sr/2^L ]  (coarsest detail)
coeffs[2]   = detail cD_(L-1)→  [sr/2^L,  sr/2^(L-1)]
...
coeffs[L]   = detail cD_1    →  [sr/4,    sr/2       ]  (finest detail)
```

For each DWT level, the band with the **greatest Hz overlap** gets exclusive ownership. This prevents bands from compounding gains on shared levels. Setting a slider to 0 truly silences those frequencies — including the approximation coefficients.

**Optimal wavelets per mode:**

| Mode | Wavelet | Reason |
|---|---|---|
| Musical Instruments | Daubechies db6 | High vanishing moments — captures smooth tonal signals |
| Animal Sounds | Daubechies db4 | Compact support — handles short transient bursts |
| Human Voices | Haar | Fast, simple, assigned by course requirements |
| ECG Abnormalities | Daubechies db4 | Established biomedical standard for QRS analysis |

The chart shows per-band relative energy (%) for input and output, with DWT frequency ranges shown as sub-labels below each bar. Hover over a bar for an exact tooltip.

> **Note:** Wavelet equalization is less frequency-precise than FFT because DWT bands are dyadic octaves. Its advantage is better time-frequency resolution — it captures transients and signal attacks more faithfully.

### System C — AI Source Separation (Teal)

Uses pretrained deep learning models to separate the signal into its constituent sources, then reconstructs the output as a **weighted sum**:

```
output = Σ (component_i × gain_i)
```

**Workflow:**
1. Activate System C by clicking its tab or panel
2. Click **🔬 Run Separation** — this runs the AI model (slow, ~5–30 seconds)
3. Sliders appear — one per separated component with its RMS energy shown
4. Adjust slider gains; the weighted sum updates instantly (no re-separation needed)
5. At 1.0× all gains, the output equals the original signal (no lossy artefact)

**AI models per mode:**

| Mode | Model | Components |
|---|---|---|
| Instruments | Demucs (htdemucs) | Drums, Bass, Vocals, Other |
| Animals | YAMNet + Wiener masking | Per detected animal sound class |
| Voices | ConvTasNet (asteroid) | Per speaker |
| ECG | ECGResNet (MIT-BIH trained) | Normal, SVEB, PVC, Fusion, Unknown |

---

## Settings File Format

All mode configurations are stored as JSON in `data/settings/`.

```json
{
  "mode": "Musical Instruments",
  "sliders": [
    {
      "id": 1,
      "name": "Kick Drum",
      "ranges": [
        { "min": 40, "max": 100 }
      ]
    },
    {
      "id": 2,
      "name": "Bass Guitar",
      "ranges": [
        { "min": 100, "max": 400 }
      ]
    },
    {
      "id": 3,
      "name": "Flute",
      "ranges": [
        { "min": 1500, "max": 4000 }
      ]
    },
    {
      "id": 4,
      "name": "Cymbals",
      "ranges": [
        { "min": 8000, "max": 16000 }
      ]
    }
  ]
}
```

A single slider can cover multiple non-contiguous frequency ranges:

```json
{
  "id": 3,
  "name": "Drums",
  "ranges": [
    { "min": 50,   "max": 200  },
    { "min": 4000, "max": 8000 }
  ]
}
```

> The `name` field in the settings file is used by System A (FFT) and System B (Wavelet) sliders. System C (AI) uses names returned by the AI model itself (e.g. `"Drums"`, `"Bass"`, `"Normal (N)"`).

---

## API Reference

Backend runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

| Method | Route | Description |
|---|---|---|
| GET | `/ping` | Health check |
| POST | `/upload` | Upload audio file. Returns `signal_id`, `sample_rate`, `duration` |
| GET | `/signal/{sid}` | Get downsampled waveform for drawing |
| POST | `/equalize` | Apply FFT + Wavelet gains. Returns output + FFT + spectrogram + wavelet energies |
| GET | `/spectrogram/{sid}` | Get 2D spectrogram of original signal |
| GET | `/wavelet-compare/{sid}/{mode}` | Compare Fourier vs Wavelet SNR |
| GET | `/settings/{mode_name}` | Load mode configuration JSON |
| POST | `/settings/save` | Save mode configuration JSON |
| GET | `/mode-info/{mode}` | Returns mode type and optimal wavelet |
| POST | `/ai/separate` | Run AI source separation (slow). Caches components server-side |
| POST | `/ai/mix` | Weighted sum of cached AI components (instant). Returns output + FFT + spectrogram |
| POST | `/ai/run` | Quick AI analysis for the AI Compare panel |

### POST /ai/separate

Runs the AI model for the given mode and signal. Call this once per signal per mode.

```json
{ "signal_id": "ab9b6acb", "mode": "instruments" }
```

Response includes `components` array with `id`, `name`, `rms_energy`, and `samples_b64` per component.

### POST /ai/mix

Instant weighted sum using cached components. Call this on every slider change.

```json
{
  "signal_id": "ab9b6acb",
  "gains": [
    { "component_id": 0, "gain": 1.0 },
    { "component_id": 1, "gain": 0.0 },
    { "component_id": 2, "gain": 1.5 },
    { "component_id": 3, "gain": 1.0 }
  ]
}
```

When all gains equal 1.0, the original signal is returned directly to avoid lossy reconstruction artefacts.

---

## Technical Details

### Signal Processing Pipeline

```
Upload WAV/MP3/OGG/FLAC
      ↓
Backend: soundfile → mono Float32 numpy array → stored in memory dict
      ↓
User moves a slider (55 ms debounce)
      ↓
POST /equalize  (System A or B)
  ├─ System A: rfft → multiply bins in freq_ranges → irfft
  └─ System B: wavedec → exclusive DWT level assignment → scale coeffs → waverec

POST /ai/mix  (System C, after /ai/separate)
  └─ output = Σ (component_waveform × gain)
             returns original signal if all gains == 1.0
      ↓
Result encoded as base64 WAV → browser
      ↓
Browser: OfflineAudioContext.decodeAudioData → Float32Array → CineViewer
FFT chart, wavelet energy chart, spectrogram all updated
```

### Key Design Decisions

**Exclusive DWT level ownership**
In the wavelet system, each DWT coefficient array is assigned to exactly one user band — the one with the greatest Hz overlap. This prevents multiple bands from compounding gains on shared levels (the previous bug where Bass=0 would silence Kick=1.0 levels they shared).

**AI separation caching**
The slow AI model run (`/ai/separate`) happens once per signal and caches the component waveforms on the server. Subsequent slider moves call `/ai/mix` which is a pure numpy weighted sum — instant regardless of signal length.

**Schema-driven, zero hardcoding**
The frontend has no hardcoded mode knowledge. It reads a JSON settings file and generates sliders dynamically. Adding a new mode requires only a new JSON file.

**Per-system audio**
Three independent `AudioBufferSourceNode` instances are managed — one per system. Playing one stops the others. The global `🔊 Input` button encodes the raw `Float32Array` to WAV in the browser (no server round-trip) and plays the original signal independently.

**Spectrogram axis resolution**
Both axes use a two-tier tick system (major + minor) with step sizes computed via a `_niceF()` algorithm that adapts to panel size and signal duration. All rendering uses `devicePixelRatio` scaling for sharp display on HiDPI screens.

### Supported File Formats

| Format | Notes |
|---|---|
| WAV | Recommended. Lossless, fastest |
| MP3 | Supported via Web Audio API |
| OGG | Supported |
| FLAC | Supported |
| CSV | ECG signals — single column of amplitude values |

---

## Team

| Member | Role | Files |
|---|---|---|
| Abdullah Gamil | Backend Core + Signal Engine + Wavelet Algorithm | `signal_processor.py`, `equalizer_engine.py`, `main.py` |
| Abdulrahman Hassan | Modes + Settings + Data + AI Integration | `settings_manager.py`, `ai_music.py`, `ai_animal.py`, `ai_human.py`, all `data/settings/*.json` |
| Saga Sadek | Frontend UI + Visualization + 3-System Layout | `index.html` — FFT chart, spectrograms with axes, 3-way toggle, system panels, CSS design system |
| Alaa Essam | Cine Viewers + Audio + ECG AI + AI Equalizer | `index.html` — CineViewer, per-system AudioPlayer, AI separation UI, `ai_ecg.py`, `ecg_inference.py` |

---

*Signal Equalizer — Task 2, Digital Signal Processing Course*
