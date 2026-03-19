<div align="center">

<!-- LOGO -->
<img width="823" height="235" alt="Signal Equalizer Logo" src="https://github.com/user-attachments/assets/86aaab8c-b0b9-4653-965b-1a8de87714db" />
<br/>

# ⚡ Signal Equalizer Studio
### DSP Task 2 — Digital Signal Processing Course

**An interactive full-stack web app for signal equalization**
using Fourier transforms, Wavelet decomposition, and AI-based source separation.

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?style=flat-square)
![JavaScript](https://img.shields.io/badge/JavaScript-frontend-f7df1e?style=flat-square&logo=javascript&logoColor=black)

</div>

---

## 📋 Table of Contents

- [Application Overview](#-application-overview)
- [Modes](#-modes)
  - [🎸 Instruments](#-instruments-mode)
  - [🐾 Animals](#-animals-mode)
  - [🗣️ Voices](#️-voices-mode)
  - [❤️ ECG](#️-ecg-mode)
  - [⚙️ Generic](#️-generic-mode)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [API Reference](#-api-reference)
- [Technical Details](#-technical-details)
- [Team](#-team)

---

## 🖥️ Application Overview

The Signal Equalizer is a browser-based app that lets you upload any audio or ECG signal and interactively adjust its frequency components in real time. Every slider move instantly updates the waveform viewers, frequency spectrum, wavelet energy chart, and spectrograms — all synchronized.

---

### Main Interface

The toolbar at the top gives you instant access to every control — mode selector, file upload, config save/load, and AI compare.
<img width="1907" height="422" alt="Image" src="https://github.com/user-attachments/assets/b5080eeb-b474-429b-bad6-3744793b3efd" />
![Full Application View](docs/screenshots/full-app.png)

---

### Waveform Viewers

Two linked cine viewers — **blue** for input, **pink** for output. Zoom with the scroll wheel, pan by dragging, and both viewers stay in perfect sync.

---

### Playback & Controls

Play, Pause, Stop, and adjustable speed (0.25× to 4×). Zoom and pan buttons. Spectrogram toggle. Linear / Audiogram FFT scale switch.

---

### Frequency Spectrum Chart

Input and output plotted together in real time. **Linear** shows the full range from 0 Hz to Nyquist. **Audiogram** uses a logarithmic X axis at 125 Hz → 8 kHz (ISO 8253-1 hearing-test standard).
<img width="1918" height="805" alt="Image" src="https://github.com/user-attachments/assets/e5a85f8b-1ad2-402a-989f-696fc4e598e9" />

![FFT Chart](docs/screenshots/fft-chart.png)

---

### Spectrogram Viewers

Two high-resolution 2D time-frequency heatmaps — input and output. Axes auto-adapt to the signal's actual sample rate. The output spectrogram updates within ~55 ms of any slider change.


---

### Three Equalization Systems

Each custom mode exposes three independent systems. Only one is active at a time — the active system drives the output viewers and audio. Inactive systems are visually dimmed.

---

### System A — Frequency Domain

Sliders scale the FFT magnitude bins within each configured frequency range. Precise, instant, and works in every mode.

<img width="1495" height="720" alt="Image" src="https://github.com/user-attachments/assets/a95509dd-bfd7-402c-90e1-ae19fab3732d" />

---

### System B — Wavelet Domain

Sliders control DWT coefficient energy per band using an exclusive level assignment that prevents bands from compounding gains on shared DWT levels.

<img width="947" height="610" alt="Image" src="https://github.com/user-attachments/assets/8b652d3b-e8a8-463f-ac61-7eb0ccd33432" />

---

### System C — AI Source Separation

Click **🔬 Separate Signal** once to run the AI model. Gain sliders appear immediately — one per separated component. Every slider move is an instant weighted sum; no re-separation needed.

<img width="792" height="528" alt="Image" src="https://github.com/user-attachments/assets/2b2c84b6-4f6d-4a27-8972-d7094bb86668" />

---

## 🎛️ Modes

---

### 🎸 Instruments Mode

Separates a music file into its individual instrument stems using the **Demucs htdemucs** deep learning model.

**Separated components:**

| Component | Description |
|---|---|
| 🥁 Drums | Kick, snare, cymbals and all percussion |
| 🎸 Bass | Bass guitar and low-end bass lines |
| 🎤 Vocals | Lead and backing vocals |
| 🎹 Other | All remaining instruments — guitars, keys, synths |

**How to use:**
1. Select **🎸 Instruments** from the Mode dropdown
2. Upload a music file (`.wav` or `.mp3`)
3. Use **System A** sliders to boost or cut specific frequency bands across the full mix
4. Switch to **System B** for wavelet-based equalization
5. Switch to **System C** → click **🔬 Separate Signal** — Demucs runs (~10–30 s)
6. Adjust per-stem gain sliders to remix the song (e.g. silence drums, boost vocals)
7. Click ▶ Play on any component card to listen to that stem in isolation

> 🎬 **Demo**
>
<video src="https://github.com/user-attachments/assets/5492ba66-f09e-4a22-974b-f8dee605bf66" controls width="800" style="border-radius: 20px; border: 3px solid #00ffff; box-shadow: 0 0 30px #00ffff66;">
    Your browser does not support the video tag.
  </video>

<video src="https://github.com/user-attachments/assets/09c4cb5f-cd67-4ac7-ab47-1f397239b782" controls width="900" style="border-radius: 20px; border: 3px solid #00ffff; box-shadow: 0 0 30px #00ffff;"></video>


<video src="https://github.com/user-attachments/assets/1ddb7326-7c25-4c66-80f3-1af1f95d0c6c" controls width="900" style="border-radius: 20px; border: 3px solid #00ffff; box-shadow: 0 0 30px #00ffff;"></video>


  

---

### 🐾 Animals Mode

Classifies and isolates animal sound sources using **YAMNet** (Google's pretrained audio classifier) combined with Wiener filter masking per detected class.

**Separated components:**
One component per detected animal sound class (e.g. Dog bark, Bird call, Cat meow) — depends on what's in the recording.

**How to use:**
1. Select **🐾 Animals** from the Mode dropdown
2. Upload a recording containing animal sounds
3. Use **System A** / **System B** sliders to equalize the full mix
4. Activate **System C** → click **🔬 Separate Signal** → one card per detected animal class
5. Mute or boost individual animal sources with the gain sliders
6. Click ▶ Play on any card to hear that animal in isolation
7. Use the **Animals Compare Panel** to benchmark YAMNet vs Wavelet with SNR, SI-SNR, PRD, and LSD metrics

> 🎬 **Demo**
>
<video src="https://github.com/user-attachments/assets/d5c2bc4a-dc26-4b06-96cd-762ad836b9c2" controls width="800" style="border-radius: 20px; border: 3px solid #00ffff; box-shadow: 0 0 30px #00ffff66;">
    Your browser does not support the video tag.
  </video>

<video src="https://github.com/user-attachments/assets/bb373789-37ae-47b4-b4a1-e183e77fd32c" controls width="800" style="border-radius: 20px; border: 3px solid #00ffff; box-shadow: 0 0 30px #00ffff66;">
    Your browser does not support the video tag.
  </video>

<video src="https://github.com/user-attachments/assets/fd5e1b3a-58f7-4749-8efe-891d60015b82" controls width="800" style="border-radius: 20px; border: 3px solid #00ffff; box-shadow: 0 0 30px #00ffff66;">
    Your browser does not support the video tag.
  </video>



---

### 🗣️ Voices Mode

Separates overlapping speech into individual speaker tracks using **ConvTasNet** via the Asteroid library.

**Separated components:**
Speaker 1, Speaker 2, Speaker 3, ... — as many as detected in the recording.

**How to use:**
1. Select **🗣️ Voices** from the Mode dropdown
2. Upload a recording with multiple or overlapping speakers
3. Use **System A** / **System B** sliders to equalize the full mix
4. Activate **System C** → click **🔬 Separate Signal** → one card per detected speaker
5. Adjust individual speaker volumes with the gain sliders
6. Click ▶ Play on any card to hear that speaker alone
7. Use the **Voices Compare Panel** to benchmark ConvTasNet vs Wavelet using SNR, SI-SNR, PRD, and LSD metrics

> 💡 **Tip:** Human voices overlap heavily in frequency. System A and B equalize the mix as a whole — for per-speaker volume control, use **System C**.

> 🎬 **Demo**
>
<video src="https://github.com/user-attachments/assets/ed4e66c2-02d5-4631-a980-cbc65694ca97" controls width="800" style="border-radius: 20px; border: 3px solid #00ffff; box-shadow: 0 0 30px #00ffff66;">
    Your browser does not support the video tag.
  </video>


  <video src="https://github.com/user-attachments/assets/7367a5c6-e263-4465-b375-9a3e5423223f" controls width="800" style="border-radius: 20px; border: 3px solid #00ffff; box-shadow: 0 0 30px #00ffff66;">
    Your browser does not support the video tag.
  </video>
  

<video src="https://github.com/user-attachments/assets/5638b33c-fce6-44ac-bdbb-a703b6fe4415" controls width="800" style="border-radius: 20px; border: 3px solid #00ffff; box-shadow: 0 0 30px #00ffff66;">
    Your browser does not support the video tag.
  </video>
  

---

### ❤️ ECG Mode

Classifies heartbeat arrhythmia types using a **ResNet** pretrained on the MIT-BIH Arrhythmia Database. Isolates each beat class as a separate signal component.

**Separated components — 5 classes (MIT-BIH standard):**

| Symbol | Class | Description |
|---|---|---|
| N | Normal | Normal sinus beat |
| S | SVEB | Supraventricular ectopic beat |
| V | PVC | Premature ventricular contraction |
| F | Fusion | Fusion of normal and ventricular beat |
| Q | Unknown | Unclassifiable beat |

**Supported file formats:** `.wav` · `.csv` (single column) · `.dat` + `.hea` (WFDB) · `.npy` · `.txt`

**How to use:**
1. Select **❤️ ECG** from the Mode dropdown
2. Upload an ECG file in any supported format
3. The signal is automatically resampled to **360 Hz** (MIT-BIH standard) for analysis
4. Use **System A** / **System B** sliders to apply frequency-domain equalization
5. Open the **ECG Analysis Panel** → click **🔬 Analyse**
   - See heart rate, RR variability, total beat count, and beat distribution per class
6. Activate **System C** → click **🔬 Separate Signal** → one card per arrhythmia class
7. Adjust per-class gain sliders to amplify or suppress specific beat types
8. Click **📊 Compare Wavelet vs ResNet** to benchmark both methods with full metrics

> 🎬 **Demo**
>
<video src="https://github.com/user-attachments/assets/526ff9b5-cfa4-465b-99ec-f96856161bdc" controls width="800" style="border-radius: 20px; border: 3px solid #00ffff; box-shadow: 0 0 30px #00ffff66;">
    Your browser does not support the video tag.
  </video>

---

### ⚙️ Generic Mode

Fully customizable frequency-domain equalization using any bands you define. Pure FFT — no wavelet or AI systems.

**How to use:**
1. Select **⚙️ Generic** from the Mode dropdown
2. Upload any audio signal
3. Click **📋 Load Schema** to import a JSON schema defining your bands
   — or click **＋ Add Band** to define bands manually (name + min/max Hz)
4. Drag sliders to boost or cut each band
5. Export your setup with **📤 Save Config** and reload it anytime

> 🎬 **Demo**
>
<video src="https://github.com/user-attachments/assets/316d6090-626c-4534-9466-44aa1e2047c4" controls width="800" style="border-radius: 20px; border: 3px solid #00ffff; box-shadow: 0 0 30px #00ffff66;">
    Your browser does not support the video tag.
  </video>
  
---

## ✨ Features

### Signal Display
- Two linked cine viewers — zoom, pan, drag; both sync automatically
- Playback controls with adjustable speed (0.25× to 4×)
- **🔊 Input** button always plays the raw uploaded signal
- **🔊 Play** per system plays only that system's processed output
- Zoom in/out, pan left/right, reset view buttons

### Frequency Analysis
- FFT chart with input and output plotted together in real time
- **Linear scale** — 0 Hz to Nyquist, full range
- **Audiogram scale** — 125 Hz to 8 kHz, logarithmic (ISO 8253-1)
- High-resolution spectrograms with time and frequency axes that auto-adapt to signal sample rate
- Output spectrogram updates within ~55 ms of any slider change

### Equalization
- **System A (FFT)** — `rfft → scale bins in freq_ranges → irfft`
- **System B (Wavelet)** — exclusive DWT level assignment prevents band overlap compounding
- **System C (AI)** — AI separation with instant gain-weighted mixing post-separation
- One active system at a time; switching resets its sliders to 1.0×

### AI Component Viewer
- Per-component cards with name, duration, and RMS energy
- Waveform thumbnail drawn from raw PCM bytes
- FFT spectrum computed entirely in-browser (radix-2 FFT, no server round-trip)
- Individual ▶ Play / ⏹ Stop per component
- Gain slider (0× to 2×) synced bidirectionally with System C sliders
- ECG audio fallback — 360 Hz WAVs automatically resampled in-browser for playback

### Generic Mode Extras
- Schema-driven sliders from any JSON band file
- Manual band editor — add and delete bands on the fly
- Save and reload configurations as JSON

### Supported File Formats

| Format | Notes |
|---|---|
| `.wav` | Recommended — lossless, fastest |
| `.mp3` | Supported via Web Audio API |
| `.ogg` | Supported |
| `.flac` | Supported |
| `.csv` | ECG — single column of amplitude values |
| `.dat` / `.hea` | WFDB format for MIT-BIH ECG records |
| `.npy` | NumPy array |
| `.txt` / `.tsv` | Whitespace or comma-separated numbers |

---

## 📁 Project Structure

```
Task2_Signal_Equalizer/
│
├── backend/
│   ├── main.py                       ← FastAPI server — all 22 API routes
│   ├── signal_processor.py           ← FFT, IFFT, Spectrogram, Wavelet functions
│   ├── equalizer_engine.py           ← Core equalization: FFT + exclusive DWT
│   ├── settings_manager.py           ← Load/save/validate JSON settings files
│   ├── ai_shared.py                  ← Shared audio loading utilities
│   ├── ai_music.py                   ← Music stem separation via Demucs
│   ├── ai_animal.py                  ← Animal classification + Wiener masking (YAMNet)
│   ├── ai_human.py                   ← Speaker separation via ConvTasNet (Asteroid)
│   ├── ai_ecg.py                     ← ECG arrhythmia classification (ResNet MIT-BIH)
│   └── generate_synthetic.py         ← Creates the 10-tone test WAV file
│
├── frontend/
│   └── index.html                    ← Complete frontend — HTML + CSS + JS (single file)
│
├── ecg_pretrainedmodel/
│   ├── ECG_standalone_version/
│   │   ├── ecg_resnet_mitbih.pt      ← Pretrained ECG ResNet weights
│   │   └── ecg_eq_settings.json      ← ECG equalizer band configuration
│   └── ECG_Inference_Equalizer.ipynb ← Training + evaluation notebook
│
└── data/
    ├── settings/
    │   ├── instruments.json          ← Musical Instruments mode config
    │   ├── animals.json              ← Animal Sounds mode config
    │   ├── voices.json               ← Human Voices mode config
    │   ├── ecg.json                  ← ECG Abnormalities mode config
    │   └── generic.json              ← Empty template for Generic mode
    └── samples/
        └── synthetic_signal.wav      ← 10-tone test signal
```

---

## ⚙️ Installation

### Prerequisites
- Python **3.10** or newer
- **Chrome** browser (recommended)
- **ffmpeg** on PATH — required by Demucs for music separation

### Step 1 — Clone the repository

```bash
git clone <repository-url>
cd Task2_Signal_Equalizer
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Mac / Linux:
source .venv/bin/activate
```

### Step 3 — Install core dependencies

```bash
pip install fastapi uvicorn numpy scipy pywavelets soundfile python-multipart pandas
```

### Step 4 — Install AI dependencies *(optional — only needed for System C)*

```bash
# Instruments mode — Demucs music separation
pip install demucs
pip install soundfile==0.11.0       # fixes SoundFileRuntimeError with Demucs

# Animals mode — YAMNet classifier
pip install torch torchaudio

# Voices mode — ConvTasNet speaker separation
pip install asteroid

# ECG mode — ResNet arrhythmia classification
pip install torch wfdb               # wfdb adds support for .dat/.hea WFDB files
```

> **Windows note:** If Demucs fails with `[WinError 2]`, the app automatically falls back to `python -m demucs`. If you see a `TorchCodec` error, run `pip install torchcodec` or downgrade with `pip install torchaudio==2.1.2`.

### Step 5 — Install ffmpeg

**Windows:**
```bash
winget install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

**Verify:**
```bash
ffmpeg -version
```

### Step 6 — Generate the test signal *(optional)*

```bash
cd backend
python generate_synthetic.py
```

Creates `data/samples/synthetic_signal.wav` — a 5-second signal with **10 pure sine waves** at 100, 300, 500, 1k, 2k, 4k, 6k, 8k, 10k, and 12k Hz. Load it into the app and each frequency appears as a sharp spike in the FFT chart — ideal for verifying equalization is working correctly.

---

## 🚀 Running the App

### Step 1 — Start the backend

```bash
cd backend
uvicorn main:app --reload
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Started reloader process
```

Interactive API docs: `http://localhost:8000/docs`

### Step 2 — Open the frontend

Open `frontend/index.html` directly in **Chrome**.

The toolbar should show **● Backend OK** in green.
If it shows **Backend Offline**, check that Step 1 is still running.

> No build step or proxy needed. CORS is fully enabled.

---

## 📡 API Reference

All routes are prefixed with `/api`. Interactive docs at `http://localhost:8000/docs`.

| Method | Route | Description |
|---|---|---|
| GET | `/api/ping` | Health check |
| POST | `/api/upload` | Upload file → `signal_id`, `sample_rate`, `duration` |
| GET | `/api/signal/{sid}` | Downsampled waveform for the cine viewer |
| POST | `/api/equalize` | Apply FFT + Wavelet gains → output WAV + FFT + spectrogram + wavelet energies |
| GET | `/api/spectrogram/{sid}` | 2D spectrogram of the original signal |
| GET | `/api/settings/{mode}` | Load mode configuration JSON |
| POST | `/api/settings/save` | Save mode configuration JSON |
| GET | `/api/mode-info/{mode}` | Mode type and optimal wavelet name |
| POST | `/api/ai/separate` | Run AI separation (slow, once per signal). Returns components with audio |
| POST | `/api/ai/mix` | Weighted mix of cached components (instant) |
| POST | `/api/ai/run` | Quick analysis for the AI Compare panel |
| POST | `/api/ecg/analyse` | Beat classification → heart rate, beat counts, frequency bands |
| POST | `/api/ecg/equalize` | Per-class Gaussian gain equalization |
| POST | `/api/ecg/compare` | Wavelet vs ResNet comparison — SNR, SI-SNR, PRD, LSD, Time |
| POST | `/api/voices/compare` | Wavelet vs ConvTasNet comparison with full metrics |
| POST | `/api/animals/compare` | Wavelet vs YAMNet comparison with full metrics |

---

## 🔧 Technical Details

### Signal Processing Pipeline

```
Upload WAV / MP3 / CSV / WFDB
           ↓
Backend: soundfile → mono Float32 numpy array → stored in memory
           ↓
User moves a slider  (55 ms debounce)
           ↓
POST /api/equalize                         POST /api/ai/mix
  ├─ System A (FFT):                         └─ output = Σ (component_i × gain_i)
  │    rfft → scale bins → irfft                    returns original if all gains = 1.0
  └─ System B (Wavelet):
       wavedec → exclusive DWT level assignment
       → scale coefficients → waverec
           ↓
Result encoded as base64 WAV → browser
           ↓
Browser: OfflineAudioContext.decodeAudioData → Float32Array
         → CineViewer + FFT chart + Wavelet chart + Spectrograms
```

### Key Design Decisions

**Exclusive DWT level ownership**
Each DWT coefficient array is assigned to exactly one user band — the one with the greatest Hz overlap. This prevents multiple sliders from compounding gains on shared levels. Setting a slider to 0× truly silences those frequencies.

**AI separation caching**
The slow AI model runs once per signal and caches the component waveforms server-side. Every subsequent slider move calls `/api/ai/mix` — a pure NumPy weighted sum — so it's instant regardless of signal length.

**ECG sample rate handling**
ECG files are resampled to 360 Hz before classification (MIT-BIH training standard), then each component is resampled back to the original file's sample rate for storage and playback. If no beats are detected, an FFT-band fallback always produces 5 components.

**Schema-driven frontend**
The frontend has zero hardcoded mode knowledge. All sliders are generated dynamically from a JSON settings file. Adding a new mode requires only a new JSON file in `data/settings/`.

**Per-system audio routing**
Three independent `AudioBufferSourceNode` instances — one per system. Playing one stops the others. The **🔊 Input** button encodes the raw `Float32Array` to WAV entirely in the browser with no server round-trip.

### Optimal Wavelet Per Mode

| Mode | Wavelet | Reason |
|---|---|---|
| Instruments | db6 | High vanishing moments — captures smooth tonal signals |
| Animals | db4 | Compact support — handles short transient bursts |
| Voices | Haar | Fast and simple |
| ECG | db4 | Established biomedical standard for QRS analysis |

### Settings File Format

```json
{
  "mode": "Musical Instruments",
  "sliders": [
    { "id": 1, "name": "Kick Drum",   "ranges": [{ "min": 40,   "max": 100   }] },
    { "id": 2, "name": "Bass Guitar", "ranges": [{ "min": 100,  "max": 400   }] },
    { "id": 3, "name": "Flute",       "ranges": [{ "min": 1500, "max": 4000  }] },
    { "id": 4, "name": "Cymbals",     "ranges": [{ "min": 8000, "max": 16000 }] }
  ]
}
```

A single slider can span multiple non-contiguous ranges:

```json
{ "id": 3, "name": "Drums", "ranges": [{ "min": 50, "max": 200 }, { "min": 4000, "max": 8000 }] }
```

---

## 👥 Team

| Name | GitHub |
|---|---|
| Abdullah Gamil | [@AbdullahGamil](https://github.com/AbdullahGamil) |
| Abdulrahman Hassan | [@AbdulrahmanHassan](https://github.com/AbdulrahmanHassan) |
| Saga Sadek | [@SagaSadek](https://github.com/SagaSadek) |
| Alaa Essam | [@AlaaEssam](https://github.com/Alaa-Essam5) |

---

<div align="center">

*Signal Equalizer — Task 2, Digital Signal Processing Course*

</div>
