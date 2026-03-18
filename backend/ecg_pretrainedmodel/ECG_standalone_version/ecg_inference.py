"""
ecg_inference.py
================
Standalone ECG arrhythmia inference + equalizer module.
Give this file + ecg_resnet_mitbih.pt to your teammate.

Usage
-----
from ecg_inference import ECGPipeline

pipe   = ECGPipeline(weights_path='ecg_resnet_mitbih.pt')
result = pipe.analyse('my_ecg.wav')        # or .dat/.hea

# result keys:
#   signal       (N,)     raw signal at 360 Hz
#   beat_labels  (M,)     predicted class per beat
#   beat_times   (M,2)    [[start_sec, end_sec], ...]
#   beat_probs   (M,5)    confidence scores
#   bands        dict     {cls: (f_lo, f_hi, f_peak)}
#   counts       dict     {cls: n_beats}

# Equalise
eq = pipe.equalise(result['signal'],
                   gains={0:1.0, 1:1.0, 2:0.0, 3:1.0, 4:1.0},
                   bands=result['bands'],
                   noise_gain=0.0)

# Save output
pipe.save_wav(eq, 'output.wav')
"""

import os
import json
import numpy as np
from scipy.fft    import fft, ifft, fftfreq
from scipy.io     import wavfile
from scipy.signal import resample_poly
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
import wfdb.processing as wfdb_proc


# ── Constants ──────────────────────────────────────────────────────────────
FS          = 360
BEAT_LEN    = 187
BEAT_BEFORE = 90
BEAT_AFTER  = BEAT_LEN - BEAT_BEFORE
N_CLASSES   = 5
CLASS_NAMES = ['Normal (N)', 'SVEB (S)', 'PVC (V)', 'Fusion (F)', 'Unknown (Q)']
CLASS_SYMS  = ['N', 'S', 'V', 'F', 'Q']


# ── Model architecture — must match training exactly ───────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        stride     = 2 if downsample else 1
        self.conv1 = nn.Conv1d(in_ch, out_ch, 5, stride=stride, padding=2, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch)
        ) if (in_ch != out_ch or downsample) else nn.Identity()

    def forward(self, x):
        return F.relu(
            self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x)
        )


class ECGResNet(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        self.layer1 = ResBlock(32,  64)
        self.layer2 = ResBlock(64,  128, downsample=True)
        self.layer3 = ResBlock(128, 256, downsample=True)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.drop   = nn.Dropout(0.3)
        self.fc     = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(self.drop(self.pool(x).squeeze(-1)))


# ── Main pipeline class ────────────────────────────────────────────────────
class ECGPipeline:
    """
    All-in-one ECG analysis pipeline.

    Parameters
    ----------
    weights_path : path to ecg_resnet_mitbih.pt
    device       : 'cuda' / 'cpu' / 'auto'
    """

    def __init__(self, weights_path: str, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = ECGResNet().to(self.device)
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        self.model.eval()
        print(f'Model loaded from {weights_path}  |  device={self.device}')

    # ── File loading ───────────────────────────────────────────────────────
    @staticmethod
    def _normalise(sig: np.ndarray) -> np.ndarray:
        return (sig - sig.mean()) / (sig.std() + 1e-8)

    @staticmethod
    def _resample(sig: np.ndarray, src_fs: int, tgt_fs: int = FS) -> np.ndarray:
        if src_fs == tgt_fs:
            return sig
        from math import gcd
        g = gcd(tgt_fs, src_fs)
        return resample_poly(sig, tgt_fs // g, src_fs // g).astype(np.float32)

    def load_file(self, filepath: str, channel: int = 0) -> tuple[np.ndarray, int]:
        """
        Load any ECG file (.wav or WFDB .dat/.hea).
        Returns (signal_360hz_normalised, 360).
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.wav':
            fs, data = wavfile.read(filepath)
            if data.dtype.kind == 'i':
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            data = data.astype(np.float32)
            if data.ndim == 2:
                data = data[:, 0]
            sig, src_fs = data, int(fs)
        else:
            base   = filepath.replace('.hea', '').replace('.dat', '')
            rec    = wfdb.rdrecord(base)
            sig    = rec.p_signal[:, channel].astype(np.float32)
            src_fs = int(rec.fs)

        sig = self._resample(sig, src_fs, FS)
        sig = self._normalise(sig)
        return sig, FS

    # ── Beat segmentation ──────────────────────────────────────────────────
    def _segment_beats(self, signal: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
        r_peaks        = wfdb_proc.gqrs_detect(sig=signal, fs=FS)
        beats, valid   = [], []
        for rp in r_peaks:
            lo, hi = rp - BEAT_BEFORE, rp + BEAT_AFTER
            if lo < 0 or hi > len(signal):
                continue
            beats.append(self._normalise(signal[lo:hi].copy()))
            valid.append(rp)
        return np.array(beats, dtype=np.float32), np.array(valid, dtype=int)

    # ── Classification ─────────────────────────────────────────────────────
    def _classify(self, beats: np.ndarray,
                  batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
        xt        = torch.tensor(beats[:, np.newaxis, :])
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(xt), batch_size):
                logits = self.model(xt[i:i+batch_size].to(self.device))
                all_probs.append(F.softmax(logits, dim=1).cpu())
        probs  = torch.cat(all_probs).numpy()
        return probs.argmax(axis=1), probs

    # ── Dominant frequency bands ───────────────────────────────────────────
    @staticmethod
    def _find_bands(beats: np.ndarray,
                    labels: np.ndarray,
                    half_win: int = 10) -> dict:
        freqs = fftfreq(BEAT_LEN, d=1/FS)[:BEAT_LEN // 2]
        bands = {}
        for cls in range(N_CLASSES):
            cls_beats = beats[labels == cls]
            if len(cls_beats) == 0:
                bands[cls] = (0.0, 0.0, 0.0)
                continue
            avg_mag  = np.abs(fft(cls_beats, axis=1))[:, :BEAT_LEN//2].mean(axis=0)
            smooth   = np.convolve(avg_mag, np.ones(half_win)/half_win, mode='same')
            peak_bin = int(np.argmax(smooth))
            lo_bin   = max(0, peak_bin - half_win)
            hi_bin   = min(len(freqs) - 1, peak_bin + half_win)
            bands[cls] = (float(freqs[lo_bin]),
                          float(freqs[hi_bin]),
                          float(freqs[peak_bin]))
        return bands

    # ── Main analysis entry point ──────────────────────────────────────────
    def analyse(self, filepath: str, channel: int = 0) -> dict:
        """
        Full pipeline: load → segment → classify → find frequencies.

        Returns
        -------
        dict with keys:
          signal       (N,)     normalised signal at 360 Hz
          fs           int      always 360
          r_peaks      (M,)     R-peak sample positions
          beats        (M,187)  beat windows
          beat_labels  (M,)     predicted class per beat
          beat_probs   (M,5)    softmax confidence scores
          beat_times   (M,2)    [[start_sec, end_sec], ...]
          bands        dict     {cls: (f_lo, f_hi, f_peak)}
          counts       dict     {cls: n_beats}
        """
        sig, fs        = self.load_file(filepath, channel)
        beats, valid_rp = self._segment_beats(sig)
        labels, probs  = self._classify(beats)
        bands          = self._find_bands(beats, labels)

        beat_times = np.column_stack([
            (valid_rp - BEAT_BEFORE) / fs,
            (valid_rp + BEAT_AFTER)  / fs
        ])
        counts = {c: int((labels == c).sum()) for c in range(N_CLASSES)}

        return dict(
            signal      = sig,
            fs          = fs,
            r_peaks     = valid_rp,
            beats       = beats,
            beat_labels = labels,
            beat_probs  = probs,
            beat_times  = beat_times,
            bands       = bands,
            counts      = counts
        )

    # ── Equalizer ──────────────────────────────────────────────────────────
    @staticmethod
    def equalise(signal: np.ndarray,
                 gains: dict,
                 bands: dict,
                 noise_gain: float = 1.0,
                 fs: int = FS) -> np.ndarray:
        """
        Apply per-class frequency gain to signal.

        Parameters
        ----------
        signal     : raw ECG (N,)
        gains      : {class_id: float}  0=suppress  1=unchanged  2=amplify
        bands      : from analyse()     {class_id: (f_lo, f_hi, f_peak)}
        noise_gain : gain for all frequencies outside every defined band
        fs         : sampling rate

        Returns
        -------
        equalised  : (N,) float32
        """
        N     = len(signal)
        freqs = np.abs(fftfreq(N, d=1/fs))
        spec  = fft(signal)

        covered = np.zeros(N, dtype=bool)
        windows = {}
        for cls in range(N_CLASSES):
            f_lo, f_hi, f_peak = bands[cls]
            if f_lo == f_hi == 0.0:
                continue
            sigma  = (f_hi - f_lo) / 2.0 if f_hi > f_lo else 1.0
            window = np.exp(-0.5 * ((freqs - f_peak) / sigma) ** 2)
            window[(freqs < f_lo) | (freqs > f_hi)] = 0.0
            w_max = window.max()
            if w_max > 0:
                window /= w_max
            windows[cls] = window
            covered |= window > 0.01

        mask           = np.ones(N, dtype=np.float64)
        mask[~covered] = noise_gain

        for cls, window in windows.items():
            gain          = gains.get(cls, 1.0)
            factor        = (1.0 - window) + window * gain
            mask[covered] *= factor[covered]

        return np.real(ifft(spec * mask)).astype(np.float32)

    # ── Save output ────────────────────────────────────────────────────────
    @staticmethod
    def save_wav(signal: np.ndarray, path: str, fs: int = FS) -> None:
        """Save float32 signal as 16-bit PCM WAV."""
        pcm = (signal / (np.abs(signal).max() + 1e-8) * 32767).astype(np.int16)
        wavfile.write(path, fs, pcm)
        print(f'Saved -> {path}')

    # ── Settings JSON ──────────────────────────────────────────────────────
    @staticmethod
    def save_settings(bands: dict, gains: dict,
                      noise_gain: float, path: str) -> None:
        """Save equalizer state to JSON (for frontend to load)."""
        cfg = {
            'mode'      : 'ECG Abnormalities',
            'fs'        : FS,
            'n_classes' : N_CLASSES,
            'noise_gain': noise_gain,
            'sliders'   : [
                {
                    'id'        : c,
                    'label'     : CLASS_NAMES[c],
                    'symbol'    : CLASS_SYMS[c],
                    'gain'      : gains.get(c, 1.0),
                    'freq_bands': [{'f_low' : bands[c][0],
                                    'f_high': bands[c][1],
                                    'f_peak': bands[c][2]}]
                }
                for c in range(N_CLASSES)
            ]
        }
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f'Settings saved -> {path}')

    @staticmethod
    def load_settings(path: str) -> tuple[dict, dict, float]:
        """Load equalizer state from JSON. Returns (bands, gains, noise_gain)."""
        with open(path) as f:
            cfg = json.load(f)
        bands = {
            s['id']: (s['freq_bands'][0]['f_low'],
                      s['freq_bands'][0]['f_high'],
                      s['freq_bands'][0]['f_peak'])
            for s in cfg['sliders']
        }
        gains      = {s['id']: s['gain'] for s in cfg['sliders']}
        noise_gain = cfg.get('noise_gain', 1.0)
        return bands, gains, noise_gain


# ── Quick usage example ────────────────────────────────────────────────────
if __name__ == '__main__':
    pipe   = ECGPipeline(weights_path='ecg_resnet_mitbih.pt')
    result = pipe.analyse('my_ecg.wav')

    print(f"Duration : {len(result['signal']) / result['fs']:.1f}s")
    print(f"Beats    : {len(result['beat_labels'])}")
    for c in range(N_CLASSES):
        print(f"  {CLASS_NAMES[c]}: {result['counts'][c]} beats")

    # Suppress PVC (class 2), keep everything else
    eq = pipe.equalise(
        signal     = result['signal'],
        gains      = {0:1.0, 1:1.0, 2:0.0, 3:1.0, 4:1.0},
        bands      = result['bands'],
        noise_gain = 0.0
    )
    pipe.save_wav(eq, 'output_no_pvc.wav')
