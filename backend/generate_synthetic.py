import numpy as np
import soundfile as sf
import os

# ── 1. Define the signal parameters ──────────────────────────────
sample_rate = 44100      # standard audio quality (44100 samples per second)
duration    = 5          # 5 seconds long
t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
# t is just a list of time values: [0, 0.000022, 0.000045, ..., 4.999977]
# it has exactly sample_rate * duration = 220500 values

# ── 2. Define the 10 frequencies we want in the signal ───────────
frequencies = [100, 300, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000]
# these are spread across the full hearing range
# each one will produce a visible spike in the FFT chart

# ── 3. Create the signal by adding pure sine waves ────────────────
signal = np.zeros(len(t))   # start with silence
for f in frequencies:
    signal += np.sin(2 * np.pi * f * t)
# np.sin(2 * pi * f * t) creates one pure tone at frequency f
# adding all of them together creates one complex signal

# ── 4. Normalize — prevent values going above 1.0 or below -1.0 ──
signal = signal / np.max(np.abs(signal))
# without this the signal would be too loud and distorted in audio players

# ── 5. Save as a WAV file ─────────────────────────────────────────
os.makedirs("../data/samples", exist_ok=True)
# exist_ok=True means: create the folder if it doesn't exist,
# but don't crash if it already exists

sf.write("../data/samples/synthetic_signal.wav", signal, sample_rate)
# sf = soundfile library
# writes the numpy array as a real WAV audio file

print("✅ Created: data/samples/synthetic_signal.wav")
print(f"   Duration:    {duration} seconds")
print(f"   Sample rate: {sample_rate} Hz")
print(f"   Frequencies: {frequencies}")
print("   Test: upload to the app → FFT chart must show exactly 10 spikes")