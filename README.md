# ⚡ Signal Equalizer — DSP Task 2

A full-stack web application for interactive signal equalization using both Fourier and Wavelet transforms. Built with a Python FastAPI backend and a pure HTML/CSS/JavaScript frontend.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [How to Use](#how-to-use)
- [Mode System](#mode-system)
- [Settings File Format](#settings-file-format)
- [API Reference](#api-reference)
- [Technical Details](#technical-details)
- [Team](#team)

---

## Overview

The Signal Equalizer allows users to upload an audio or ECG signal, then interactively adjust the magnitude of specific frequency components using sliders. Changes are reflected immediately across the waveform viewers, frequency spectrum chart, and spectrograms.

The app supports two architecturally distinct modes:

| Mode Type | Modes | Equalization |
|---|---|---|
| **Generic** | Generic | Frequency domain (FFT) |
| **Custom** | Instruments, Animals, Voices, ECG | Dual system: FFT + Optimal Wavelet |

---

## Features

### Signal Display
- **Two linked cine viewers** — input (blue) and output (red) waveforms displayed side by side in large canvases
- **Synchronized scrolling** — zoom and pan on either viewer and both update identically
- **Playback controls** — Play, Pause, Stop, Speed control (0.25× to 4×)
- **Zoom & Pan** — buttons and mouse scroll wheel, with drag-to-pan
- **Audio playback** — plays the equalized output signal through the browser speakers

### Frequency Analysis
- **FFT frequency spectrum** — input and output plotted together on one chart
- **Linear scale** — full frequency range 0 Hz to Nyquist
- **Audiogram scale** — logarithmic X axis at 125, 250, 500, 1k, 2k, 4k, 8k Hz (hearing test format)
- Scale toggle does not interrupt or reset any functionality

### Spectrograms
- **Two spectrograms** — input and output displayed in parallel in large dedicated panels
- **Live update** — output spectrogram updates within 1 second of any slider change
- **Toggle show/hide** — single button hides both spectrograms and expands the frequency chart
- Color scale: dark blue = low energy, orange/yellow = high energy

### Equalizer — Generic Mode
- **Schema-driven** — load a JSON schema file to auto-generate sliders for arbitrary frequency bands
- **Manual band adding** — even without a schema, click **＋ Add Band Manually** to define bands one by one
- **Hybrid workflow** — load a schema to get a starting set of bands, then extend it by adding more manually
- **Save & Load** — export your current band configuration as a JSON file and reload it later
- **Purely frequency-based** — no wavelet controls appear in this mode

### Equalizer — Custom Modes (Dual System)
Each custom mode displays two independent control systems simultaneously:

**System A — Frequency Domain (blue)**
- FFT spectrum chart showing input vs output
- 4 sliders, each controlling the magnitude of one component in its frequency range

**System B — Wavelet Domain (purple)**
- Wavelet energy bar chart showing per-band RMS energy before and after
- 4 sliders for the same components, operating in the wavelet domain
- Uses a pre-assigned optimal wavelet per mode (not user-selectable)

| Mode | Optimal Wavelet | Reason |
|---|---|---|
| Musical Instruments | Daubechies db6 | Best for tonal sustained audio signals |
| Animal Sounds | Daubechies db4 | Good for short transient sounds |
| Human Voices | Haar | Assigned by course requirements |
| ECG Abnormalities | Daubechies db4 | Standard in biomedical signal processing |

### AI Analysis
- Runs signal analysis on the uploaded file
- **ECG mode** — peak detection, heart rate estimation, arrhythmia classification
- **Audio modes** — spectral features: peak frequency, RMS energy, spectral centroid, rolloff
- Results displayed in an expandable card panel

---

## Project Structure

```
Task2_Signal_Equalizer/
│
├── backend/
│   ├── main.py                  ← FastAPI server — all API routes
│   ├── signal_processor.py      ← FFT, IFFT, Spectrogram, Wavelet functions
│   ├── equalizer_engine.py      ← apply_gain() — core equalization logic
│   ├── settings_manager.py      ← load/save/validate JSON settings files
│   ├── ai_models.py             ← AI signal analysis functions
│   └── generate_synthetic.py    ← Run once to create the test WAV file
│
├── frontend/
│   └── index.html               ← Complete frontend (HTML + CSS + JS in one file)
│
└── data/
    ├── settings/
    │   ├── instruments.json     ← Musical Instruments mode config
    │   ├── animals.json         ← Animal Sounds mode config
    │   ├── voices.json          ← Human Voices mode config
    │   ├── ecg.json             ← ECG Abnormalities mode config
    │   └── generic.json         ← Empty template for Generic mode
    └── samples/
        ├── synthetic_signal.wav ← Generated test signal (10 pure sine waves)
        ├── instruments_mix.wav  ← Mixed instruments audio file
        ├── animals_mix.wav      ← Mixed animal sounds audio file
        ├── voices_mix.wav       ← Mixed voices audio file
        └── ecg_*.csv            ← ECG signal files from MIT-BIH database
```

---

## Installation

### Prerequisites
- Python 3.10 or newer
- A modern browser (Chrome recommended)
- Git (optional)

### Step 1 — Clone or download the project

```bash
git clone <repository-url>
cd Task2_Signal_Equalizer
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install fastapi
pip install uvicorn
pip install numpy
pip install scipy
pip install pywavelets
pip install soundfile
pip install python-multipart
pip install neurokit2          # optional — needed for ECG AI analysis
```

Or install everything at once:

```bash
pip install fastapi uvicorn numpy scipy pywavelets soundfile python-multipart neurokit2
```

### Step 4 — Verify installation

```bash
python -c "import fastapi, numpy, scipy, pywt, soundfile; print('All OK')"
```

### Step 5 — Generate the synthetic test signal

```bash
cd backend
python generate_synthetic.py
```

This creates `data/samples/synthetic_signal.wav` — a 5-second signal containing 10 pure sine waves at 100, 300, 500, 1000, 2000, 4000, 6000, 8000, 10000, and 12000 Hz.

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

The `--reload` flag means the server automatically restarts whenever you save a Python file.

### Step 2 — Open the frontend

Open `frontend/index.html` directly in Chrome:
- Double-click the file in your file explorer, or
- Drag it into a Chrome window

The top-right corner of the app should show **● Backend OK** in green. If it shows **Backend Offline**, make sure Step 1 is still running.

> **Important:** Keep the terminal with uvicorn running at all times while using the app.

---

## How to Use

### Uploading a Signal

1. Click **📂 Load Signal** in the toolbar
2. Select any `.wav`, `.mp3`, `.ogg`, or `.flac` file
3. The input waveform, input spectrogram, and FFT chart will all appear automatically
4. The output is initialized with all sliders at 1.0× (no change)

For testing, use the included `data/samples/synthetic_signal.wav` — it has 10 known frequency spikes that are easy to verify.

### Moving Sliders

- Drag any slider up or down
- All charts update **immediately** — no button press needed
- The value label above each slider shows the current gain (e.g. `0.0×` = silence, `2.0×` = double)

### Playback

| Button | Action |
|---|---|
| ▶ Play | Both viewers scroll forward simultaneously |
| ⏸ Pause | Both viewers freeze at the same position |
| ⏹ Stop | Both viewers return to the beginning |
| 🔊 Audio | Plays the equalized output through speakers |
| Speed slider | Changes playback speed from 0.25× to 4× |

### Zoom and Pan

| Control | Action |
|---|---|
| 🔍+ button | Zoom in — see less signal, more detail |
| 🔍− button | Zoom out — see more of the signal |
| Mouse scroll wheel | Zoom in/out on either canvas |
| Click and drag canvas | Pan left or right |
| ◀ ▶ buttons | Pan left or right by 20% of the window |
| ↺ Reset | Return to the default view (first 0.5 seconds) |

> Both viewers always stay synchronized. Zooming or panning one automatically updates the other.

### Frequency Scale

Use the **Scale** radio buttons in the playback bar:

- **Linear** — X axis from 0 Hz to the Nyquist frequency, evenly spaced
- **Audiogram** — X axis showing 125, 250, 500, 1000, 2000, 4000, 8000 Hz in logarithmic spacing, matching the standard hearing test format

Switching scales never resets any sliders or playback state.

### Spectrogram Toggle

Click **👁 Spectrograms** to hide both spectrogram panels. Click again to show them. The frequency chart expands to fill the space.

---

## Mode System

### Switching Modes

Use the **Mode** dropdown in the toolbar. The equalizer panel updates immediately with the correct sliders for that mode.

### Generic Mode

Generic mode is for **arbitrary frequency-based equalization**:

**Option A — Load a schema file (recommended):**
1. Switch to **⚙️ Generic** in the mode dropdown
2. Click **📋 Load Schema** in the toolbar and select a JSON file
3. Sliders are auto-generated — one per frequency band defined in the file
4. Each slider shows the frequency range as a tooltip (hover over it)
5. You can then add more bands manually on top of the loaded schema

**Option B — Build bands manually from scratch:**
1. Switch to **⚙️ Generic** — the equalizer panel starts empty
2. Click **＋ Add Band Manually** shown in the center of the empty panel
3. Enter a band name (e.g. `Bass`), a min frequency, and a max frequency
4. A new slider appears immediately
5. Repeat for as many bands as needed

**Saving your configuration:**
- Click **💾 Save Schema** to export all current bands as a JSON file
- Load that file later with **📋 Load Schema** to restore your exact setup
- The exported file is fully editable in any text editor

Generic mode only uses the **frequency domain (FFT)**. No wavelet controls appear.

### Custom Modes (Instruments / Animals / Voices / ECG)

Custom modes show the **Dual System layout**:

**Left panel — System A (Frequency Domain)**
- FFT chart showing input (blue) and output (red) frequency spectrum
- 4 frequency-domain sliders — one per component
- Moving these sliders applies gain directly to the FFT bins

**Right panel — System B (Wavelet Domain)**
- Wavelet energy bar chart — purple = input energy, orange = output energy per band
- 4 wavelet-domain sliders — same bands as System A but independent
- Moving these sliders applies gain to the wavelet coefficients
- The optimal wavelet is displayed as a badge (e.g. `haar`, `db4`, `db6`) — it is fixed and not user-selectable

Both systems stack: the final output signal is first processed by System A (Fourier), then by System B (Wavelet).

---

## Settings File Format

All mode configurations are stored as JSON files in `data/settings/`. These files can be edited with any text editor.

### Format

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

### Fields

| Field | Type | Description |
|---|---|---|
| `mode` | string | Display name of the mode |
| `sliders` | array | List of frequency band definitions |
| `sliders[].id` | integer | Unique identifier for this band |
| `sliders[].name` | string | Label shown under the slider |
| `sliders[].ranges` | array | One or more frequency ranges |
| `ranges[].min` | number | Minimum frequency in Hz |
| `ranges[].max` | number | Maximum frequency in Hz |

### Multiple Frequency Ranges Per Slider

A single slider can control multiple non-contiguous frequency ranges. For example, a Drums slider can cover both the kick (low) and cymbal (high) frequencies simultaneously:

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

---

## API Reference

The backend runs at `http://localhost:8000`. Interactive documentation is automatically available at `http://localhost:8000/docs` when the server is running.

| Method | Route | Description |
|---|---|---|
| GET | `/ping` or `/health` | Test connection. Returns `{"message": "hello"}` |
| POST | `/upload` | Upload audio file. Returns `signal_id`, `sample_rate`, `duration` |
| GET | `/signal/{sid}` | Get downsampled waveform samples for drawing |
| POST | `/equalize` | Apply frequency + wavelet gains. Returns output signal + FFT + spectrogram |
| GET | `/spectrogram/{sid}` | Get 2D spectrogram of the original signal |
| GET | `/wavelet-compare/{sid}/{mode}` | Compare Fourier vs Wavelet SNR for a mode |
| GET | `/settings/{mode_name}` | Load mode configuration from JSON file |
| POST | `/settings/save` | Save mode configuration to JSON file |
| GET | `/modes/list` | Returns list of available mode names |
| GET | `/mode-info/{mode}` | Returns mode type (`generic`/`custom`) and optimal wavelet |
| POST | `/ai/run` | Run AI signal analysis for a given mode |

### POST /equalize — Request Body

```json
{
  "signal_id": "ab9b6acb",
  "freq_gains": [
    {
      "band_id": 1,
      "freq_ranges": [[40, 100]],
      "gain": 0.0
    }
  ],
  "wavelet_gains": [
    {
      "band_id": 1,
      "freq_ranges": [[40, 100]],
      "gain": 1.5
    }
  ],
  "mode": "instruments"
}
```

### POST /equalize — Response

```json
{
  "output_signal_b64": "UklGRi...",
  "fft_input": [{"frequency": 100.0, "magnitude": 0.08}, ...],
  "fft_output": [{"frequency": 100.0, "magnitude": 0.0}, ...],
  "spectrogram_output": [[0.1, 0.3, ...], ...],
  "wavelet_energies_input":  [{"band_id": 1, "name": "Kick Drum", "energy": 0.043}, ...],
  "wavelet_energies_output": [{"band_id": 1, "name": "Kick Drum", "energy": 0.021}, ...],
  "wavelet_used": "db6"
}
```

---

## Technical Details

### Signal Processing Pipeline

```
Upload WAV file
      ↓
Backend reads with soundfile → converts stereo to mono → stores as Float32 numpy array
      ↓
User moves a slider
      ↓
POST /equalize called with freq_gains + wavelet_gains
      ↓
System A: np.fft.rfft → multiply bins → np.fft.irfft    (Fourier)
      ↓
System B: pywt.wavedec → multiply coeffs → pywt.waverec  (Wavelet)
      ↓
Result encoded as base64 WAV → sent to browser
      ↓
Browser decodes with OfflineAudioContext → draws output waveform
FFT chart updated, spectrogram updated, wavelet energy chart updated
```

### Key Design Decisions

**One `apply_gain()` for all modes**
The core equalization function accepts a list of `[min_hz, max_hz]` pairs. All modes — Instruments, Animals, Voices, ECG, Generic — call the same function with different frequency ranges. No mode-specific equalization code exists anywhere.

**Schema-driven sliders**
The equalizer panel has no hardcoded mode knowledge. It reads a JSON config and dynamically generates sliders. Switching modes means loading a different JSON — the UI builds itself automatically.

**Dual system architecture**
In custom modes, System A (Fourier) and System B (Wavelet) are independent slider sets that both apply to the same signal sequentially. This allows direct comparison: move a System A slider and see the FFT chart change; move a System B slider and see the wavelet energy chart change.

**Synchronized cine viewers**
Both viewers share `viewStart` and `viewEnd` values. Any zoom, pan, or playback event on one viewer immediately mirrors to the other via a `syncWith()` reference. No polling or events — direct synchronous call.

### Wavelet Research Summary

| Mode | Chosen Wavelet | Why This Wavelet |
|---|---|---|
| Musical Instruments | Daubechies db6 | High vanishing moments capture smooth tonal signals efficiently |
| Animal Sounds | Daubechies db4 | Compact support handles short transient bursts without edge artifacts |
| Human Voices | Haar | Simple, fast, separates speech energy at the instructor-specified level |
| ECG | Daubechies db4 | Established standard in biomedical literature for QRS complex analysis |
| Generic | N/A (Fourier only) | Generic mode is purely frequency-based by design |

### Supported File Formats

| Format | Notes |
|---|---|
| WAV | Recommended. Lossless, fastest to decode |
| MP3 | Supported via Web Audio API decoder in browser |
| OGG | Supported |
| FLAC | Supported |
| CSV | For ECG signals — single column of amplitude values |

### UI Layout Dimensions

The app is designed to use all available screen space. Key panel sizes:

| Panel | Height | Notes |
|---|---|---|
| Cine Viewers (×2) | 220 px | Wide side-by-side layout |
| Generic FFT chart | 340 px | Takes most of the analysis row |
| Generic Spectrograms (×2) | 340 px | 300 px wide each |
| Custom FFT chart (System A) | 220 px | Full-height chart |
| Custom Wavelet chart (System B) | 220 px | Full-height bar chart |
| Custom Spectrograms (×2) | 160 px min each | 300 px wide column |
| Equalizer sliders | 155 px | Scrollable horizontally |

All panels are responsive — the page scrolls naturally when the AI Analysis panel is open or when many sliders are added.

---

## Team

| Member | Role | Files |
|---|---|---|
| Abdullah Gamil | Backend Core + Signal Engine | `signal_processor.py`, `equalizer_engine.py`, `main.py` |
| Abdulrahman Hassan | Modes + Settings + Data | `settings_manager.py`, all `data/settings/*.json`, all audio sample files |
| Saga Sadek | Frontend UI + Visualization | `index.html` — FFT chart, spectrograms, equalizer panel, CSS |
| Alaa Essam | Cine Viewers + Audio + AI | `index.html` — CineViewer, AudioPlayer, api calls, state, AI panel |

---

*Signal Equalizer — Task 2, Digital Signal Processing Course*
