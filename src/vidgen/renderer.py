"""Deterministic, deliberately lo-fi visual feedback. CPU-based and checkpoint-free."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .audio import Analysis

PRESETS = ("analog-feedback", "industrial-decay", "minimal-signal")


def _texture(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    texture = np.zeros((height, width), dtype=np.float32)
    for scale, gain in ((6, .43), (24, .33), (100, .18), (min(width, height), .06)):
        h = max(2, min(height, scale))
        w = max(2, min(width, int(scale * width / height)))
        field = rng.random((h, w), dtype=np.float32)
        texture += cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR) * gain
    texture = (np.clip(texture, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)


def load_source(source: Path | None, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    if source is None:
        return _texture(rng, width, height)
    img = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read source image: {source}")
    # fill the frame, cropping instead of distorting input aspect ratio
    h, w = img.shape[:2]
    scale = max(width / w, height / h)
    resized = cv2.resize(img, (int(math.ceil(w * scale)), int(math.ceil(h * scale))))
    x = (resized.shape[1] - width) // 2
    y = (resized.shape[0] - height) // 2
    return resized[y:y + height, x:x + width].copy()


class FeedbackRenderer:
    def __init__(self, source: np.ndarray, *, preset: str, seed: int, fps: int):
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset}. Choose from {PRESETS}")
        self.preset = preset
        self.source = source
        self.state = source.copy()
        self.height, self.width = source.shape[:2]
        self.rng = np.random.default_rng(seed)
        self.fps = fps
        self.x, self.y = np.meshgrid(
            np.arange(self.width, dtype=np.float32),
            np.arange(self.height, dtype=np.float32),
        )

    def frame(self, index: int, features: dict[str, float]) -> np.ndarray:
        t = index / self.fps
        low = float(features.get("low", 0))
        mid = float(features.get("mid", 0))
        high = float(features.get("high", 0))
        flux = float(features.get("flux", 0))
        slow = float(features.get("slow", 0))
        if self.preset == "minimal-signal":
            persistence, strength, tint = .974, 0.6, (0.90, 0.95, 1.00)
        elif self.preset == "industrial-decay":
            persistence, strength, tint = .985, 3.5, (0.70, 0.88, 1.04)
        else:
            persistence, strength, tint = .973, 2.0, (0.85, 0.95, 1.03)
        # Autonomous motion never freezes, even when input dynamics are constant.
        zoom = 1.001 + .011 * low + .002 * math.sin(t * .41)
        rotation = .085 * math.sin(t * .23) + .08 * mid
        matrix = cv2.getRotationMatrix2D((self.width / 2, self.height / 2), rotation, zoom)
        matrix[0, 2] += .35 * math.sin(t * .37)
        matrix[1, 2] += .35 * math.cos(t * .31)
        moved = cv2.warpAffine(self.state, matrix, (self.width, self.height), borderMode=cv2.BORDER_REFLECT101)
        warp = strength * (.3 + mid)
        map_x = self.x + warp * np.sin(self.y * .023 + t * .63 + low * 3.0)
        map_y = self.y + warp * np.cos(self.x * .017 + t * .43 + mid * 2.0)
        moved = cv2.remap(moved, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        if self.preset == "industrial-decay":
            edges = cv2.Canny(moved, 60, 120)
            moved = cv2.addWeighted(moved, .95, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), .15 + .2 * mid, 0)
        # Feed some of the original texture back in, preventing eventual flatness.
        frame = cv2.addWeighted(moved, persistence, self.source, 1 - persistence, 0)
        if self.preset == "minimal-signal":
            frame = cv2.GaussianBlur(frame, (3, 3), .3)
        else:
            if flux > .45:
                lines = int(1 + 5 * flux)
                for _ in range(lines):
                    y = int(self.rng.integers(0, self.height))
                    block_h = int(self.rng.integers(1, max(2, self.height // 35)))
                    shift = int(self.rng.integers(-18, 19) * flux)
                    frame[y:y + block_h] = np.roll(frame[y:y + block_h], shift, axis=1)
        # Quantization, slight electronic flicker, grit and palette tint.
        levels = 24 if self.preset == "minimal-signal" else 12
        frame = ((frame // (256 // levels)) * (256 // levels)).astype(np.float32)
        frame *= (.83 + .22 * slow) * np.asarray(tint, dtype=np.float32)
        if self.preset != "minimal-signal":
            noise = self.rng.normal(0, 2 + high * 17, (self.height, self.width, 1))
            frame += noise
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        self.state = frame
        return frame


def render(
    *, audio: Path, output: Path, analysis: Analysis, source: Path | None,
    preset: str, width: int, height: int, seed: int, crf: int = 20,
) -> None:
    if width < 32 or height < 32 or width % 2 or height % 2:
        raise ValueError("width and height must be even and at least 32")
    if not (0 <= crf <= 51):
        raise ValueError("crf must be 0..51")
    rng = np.random.default_rng(seed)
    visual = FeedbackRenderer(load_source(source, width, height, rng), preset=preset, seed=seed, fps=analysis.fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(analysis.fps), "-i", "pipe:0", "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-t", f"{analysis.duration:.6f}",
        "-movflags", "+faststart", str(output),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for i in range(len(analysis.features["rms"])):
            frame_features = {k: float(v[i]) for k, v in analysis.features.items()}
            proc.stdin.write(visual.frame(i, frame_features).tobytes())
        proc.stdin.close()
        proc.stdin = None
        _, stderr = proc.communicate()
        if proc.returncode:
            raise RuntimeError(f"ffmpeg encoding failed: {stderr.decode(errors='replace')}")
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        output.unlink(missing_ok=True)
        raise
