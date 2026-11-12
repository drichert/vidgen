"""Offline, frame-aligned audio analysis. Requires ffmpeg/ffprobe executables."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

SAMPLE_RATE = 22050


@dataclass(frozen=True)
class Analysis:
    duration: float
    fps: int
    features: dict[str, np.ndarray]

    def as_json(self) -> dict:
        return {
            "duration": self.duration,
            "fps": self.fps,
            "features": {key: np.round(value, 5).tolist() for key, value in self.features.items()},
        }


def load_mono(audio: Path, duration: float | None) -> np.ndarray:
    """Decode arbitrary ffmpeg-supported files to normalized float32 mono PCM."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(audio)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg could not decode {audio}: {proc.stderr.decode(errors='replace')}")
    samples = np.frombuffer(proc.stdout, dtype="<f4")
    if samples.size == 0:
        raise ValueError(f"No audio samples found in {audio}")
    return samples


def _normalize(data: np.ndarray) -> np.ndarray:
    """Scale robustly to 0..1; silence/flat signals remain zero."""
    lo, hi = np.percentile(data, (5, 95))
    if hi - lo < 1e-8:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0, 1).astype(np.float32)


def analyze(audio: Path, *, fps: int = 12, duration: float | None = None) -> Analysis:
    if fps < 1:
        raise ValueError("fps must be positive")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive")
    y = load_mono(audio, duration)
    actual_duration = len(y) / SAMPLE_RATE
    nframes = max(1, int(np.ceil(actual_duration * fps)))
    # STFT is time resolved; a nonbeat-based analysis works for sustained material too.
    frequencies, times, z = signal.stft(
        y, fs=SAMPLE_RATE, nperseg=2048, noverlap=1536, boundary="zeros"
    )
    mag = np.abs(z).astype(np.float32)
    power = mag * mag
    names = {"low": (30, 180), "mid": (180, 2000), "high": (2000, 9000)}
    raw: dict[str, np.ndarray] = {}
    for name, (lower, upper) in names.items():
        mask = (frequencies >= lower) & (frequencies < upper)
        raw[name] = np.log1p(1000 * np.sqrt(np.mean(power[mask], axis=0)))
    raw["rms"] = np.sqrt(np.mean(power, axis=0))
    # Positive spectral change measures articulations without imposing a beat grid.
    normalized_mag = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-8)
    delta = np.diff(normalized_mag, axis=1, prepend=normalized_mag[:, :1])
    raw["flux"] = np.sum(np.maximum(0, delta), axis=0)
    # slow RMS envelope drives long-form structure in drones and ambient tracks.
    raw["slow"] = gaussian_filter1d(raw["rms"], sigma=max(1, 2.0 / (times[1] - times[0])))
    frame_times = (np.arange(nframes) + 0.5) / fps
    frame_times = np.clip(frame_times, 0, actual_duration)
    features = {}
    for name, values in raw.items():
        smoothed = gaussian_filter1d(values, sigma=1.2 if name != "slow" else 3.0)
        features[name] = _normalize(np.interp(frame_times, times, smoothed))
    return Analysis(duration=actual_duration, fps=fps, features=features)
