"""
ai_music.py
===========
Music source separation using Demucs.

What it does:
  Runs the Demucs CLI to separate a music file into 4 stems:
    drums / bass / vocals / other

Usage:
    from ai_music import MusicModel
    model  = MusicModel()
    result = model.separate("song.wav")
    # result["stems"] → {
    #   "drums":  {"waveform": np.array, "sr": 44100, "path": "..."},
    #   "bass":   {...},
    #   "vocals": {...},
    #   "other":  {...},
    # }

Requirements:
    pip install demucs
    pip install soundfile==0.11.0   ← fixes SoundFileRuntimeError
    (ffmpeg must be installed and on PATH for mp3 output)

Common errors:
    SoundFileRuntimeError  → downgrade soundfile: pip install soundfile==0.11.0
    Demucs failed          → check ffmpeg is installed: ffmpeg -version
    Output not found       → check output_root folder exists and is writable
"""

import os
import subprocess
import numpy as np

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    print("[Music] ⚠ librosa not installed — run: pip install librosa")


# ── MusicModel ────────────────────────────────────────────────────────────────
class MusicModel:

    STEMS = ["drums", "bass", "vocals", "other"]

    def __init__(self, output_root: str = "separated",
                 model_name: str = "htdemucs"):
        """
        Parameters
        ----------
        output_root : folder where demucs writes separated stems
        model_name  : demucs model to use (default: htdemucs)
                      other options: htdemucs_ft, mdx, mdx_extra
        """
        if not LIBROSA_OK:
            raise ImportError("librosa is required. Run: pip install librosa")

        self.output_root = output_root
        self.model_name  = model_name
        os.makedirs(output_root, exist_ok=True)
        print(f"[Music] MusicModel ready  model={model_name}")

    def separate(self, audio_path: str) -> dict:
        """
        Separate music into stems using Demucs.

        Parameters
        ----------
        audio_path : path to any audio file (wav/mp3/ogg/m4a)

        Returns
        -------
        {
          "stems": {
            "drums":  {"waveform": np.array, "sr": int, "path": str},
            "bass":   {...},
            "vocals": {...},
            "other":  {...},
          }
        }

        Raises
        ------
        RuntimeError  if demucs process fails
        FileNotFoundError  if output stems not found after demucs runs
        """
        if not LIBROSA_OK:
            raise ImportError("librosa is required. Run: pip install librosa")

        audio_path = os.path.abspath(audio_path)
        base       = os.path.splitext(os.path.basename(audio_path))[0]
        print(f"[Music] Running demucs on '{base}' ...")

        # Run demucs CLI — use --mp3 to avoid SoundFileRuntimeError
        cmd = ["demucs", "-n", self.model_name,
               "-o", self.output_root, "--mp3", audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # If mp3 fails, retry without --mp3 (wav output)
            print("[Music] mp3 output failed, retrying with wav...")
            cmd_wav = ["demucs", "-n", self.model_name,
                       "-o", self.output_root, audio_path]
            result = subprocess.run(cmd_wav, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Demucs failed (exit {result.returncode}):\n"
                f"STDOUT: {result.stdout[-500:]}\n"
                f"STDERR: {result.stderr[-500:]}"
            )

        # Find output folder — demucs creates: output_root/model_name/base/
        stem_dir = os.path.join(self.output_root, self.model_name, base)
        if not os.path.isdir(stem_dir):
            raise FileNotFoundError(
                f"Demucs output not found at '{stem_dir}'\n"
                f"Contents of output root: {os.listdir(self.output_root)}"
            )

        stems = {}
        for stem in self.STEMS:
            for ext in ("mp3", "wav"):
                p = os.path.join(stem_dir, f"{stem}.{ext}")
                if os.path.exists(p):
                    try:
                        if ext == "mp3":
                            # librosa can't always read mp3 — use pydub instead
                            from pydub import AudioSegment
                            import numpy as np
                            seg = AudioSegment.from_mp3(p)
                            seg = seg.set_channels(1)
                            sr  = seg.frame_rate
                            wav = np.array(seg.get_array_of_samples(), dtype=np.float32)
                            wav = wav / (2**15)   # normalize int16 → float32
                        else:
                            wav, sr = librosa.load(p, sr=None, mono=True)
                        stems[stem] = {
                            "waveform": wav.astype(np.float32),
                            "sr"      : sr,
                            "path"    : p,
                        }
                        duration = len(wav) / sr
                        print(f"  [Music] {stem:8s} ✓  {duration:.1f}s  {p}")
                    except Exception as e:
                        print(f"  [Music] {stem:8s} ✗  load error: {e}")
                    break
            else:
                print(f"  [Music] {stem:8s} ✗  not found in {stem_dir}")

        if not stems:
            raise FileNotFoundError(
                f"No stem files found in {stem_dir}. "
                f"Contents: {os.listdir(stem_dir)}"
            )

        return {"stems": stems}


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else \
        r"D:\2nd year SBME\Second semester\DSP\project\task 2\data\music.wav"

    if os.path.exists(audio_file):
        model  = MusicModel()
        result = model.separate(audio_file)
        print("\nStems separated:")
        for name, data in result["stems"].items():
            duration = len(data["waveform"]) / data["sr"]
            print(f"  {name:8s}  {duration:.1f}s  sr={data['sr']}  → {data['path']}")
    else:
        print(f"Audio file not found: {audio_file}")
        print("Usage: python ai_music.py path/to/music.wav")
        print("\nIf you get SoundFileRuntimeError:")
        print("  pip install soundfile==0.11.0")