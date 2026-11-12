"""Deterministic, deliberately lo-fi visual feedback. CPU-based and checkpoint-free."""
from __future__ import annotations

import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .audio import Analysis
from .modulation import ModulationEngine, Preset, builtin_names, load_preset

PRESETS = builtin_names()


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
    """Feedback renderer; every time-varying quantity comes from the preset's modulation engine."""

    def __init__(self, source: np.ndarray, *, preset: Preset | str | Path, seed: int, fps: int):
        self.preset = preset if isinstance(preset, Preset) else load_preset(preset)
        self.engine = ModulationEngine(self.preset, fps)
        self.settings = self.engine.settings
        self.source = source
        self.state = source.copy()
        self.height, self.width = source.shape[:2]
        self.rng = np.random.default_rng(seed)
        self.fps = fps
        self._levels: np.ndarray | None = None  # smoothed (black, white) points for autolevel
        self.x, self.y = np.meshgrid(
            np.arange(self.width, dtype=np.float32),
            np.arange(self.height, dtype=np.float32),
        )

    def frame(self, index: int, features: dict[str, float]) -> np.ndarray:
        t = index / self.fps
        p = self.engine.params(index, features)
        cfg = self.settings
        matrix = cv2.getRotationMatrix2D((self.width / 2, self.height / 2), p["rotation"], p["zoom"])
        matrix[0, 2] += p["shift_x"]
        matrix[1, 2] += p["shift_y"]
        moved = cv2.warpAffine(self.state, matrix, (self.width, self.height), borderMode=cv2.BORDER_REFLECT101)
        if p["warp"] > 0:
            speed_x, speed_y = cfg["warp_speed"]
            map_x = self.x + p["warp"] * np.sin(self.y * .023 + t * speed_x + p["warp_phase_x"])
            map_y = self.y + p["warp"] * np.cos(self.x * .017 + t * speed_y + p["warp_phase_y"])
            moved = cv2.remap(moved, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        if p["edge_gain"] > 0:
            edges = cv2.Canny(cv2.GaussianBlur(moved, (0, 0), 2.0), 25, 70)
            # Zero-mean edge overlay: adds structure without shifting overall brightness.
            edge_layer = edges.astype(np.float32)
            edge_layer -= edge_layer.mean()
            moved = np.clip(moved + p["edge_gain"] * edge_layer[..., None], 0, 255).astype(np.uint8)
        # Feed some of the original texture back in, preventing eventual flatness.
        frame = cv2.addWeighted(moved, p["persistence"], self.source, 1 - p["persistence"], 0)
        if cfg["blur"]:
            frame = cv2.GaussianBlur(frame, (3, 3), .3)
        glitch = p["glitch"]
        if glitch > cfg["glitch_threshold"]:
            for _ in range(int(1 + 5 * glitch)):
                y = int(self.rng.integers(0, self.height))
                block_h = int(self.rng.integers(1, max(2, self.height // 35)))
                shift = int(self.rng.integers(-18, 19) * glitch)
                frame[y:y + block_h] = np.roll(frame[y:y + block_h], shift, axis=1)
        # The feedback state stays neutral; gain, tint, quantization and grit only touch the
        # displayed copy, otherwise they compound every frame (fade to black / saturation).
        self.state = frame
        step = max(1, 256 // int(cfg["levels"]))
        out = (np.round(frame.astype(np.float32) / step) * step)
        out *= p["brightness"] * np.asarray(cfg["tint"], dtype=np.float32)
        out = self._tone(out, cfg)
        if p["grain"] > 0:
            out += self.rng.normal(0, p["grain"], (self.height, self.width, 1))
        frame = np.clip(out, 0, 255).astype(np.uint8)
        # Display-only palette: hue shift and saturation lift.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(p["hue_shift"])) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * cfg["saturation"] + 25, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


    def _tone(self, out: np.ndarray, cfg: dict) -> np.ndarray:
        """Display-only tonal stretch and gamma. Black/white points are smoothed over time so
        quiet and loud passages don't pump the whole image."""
        if cfg["autolevel"] > 0:
            lo_pct, hi_pct = cfg["autolevel_pct"]
            small = out[::4, ::4]
            target = np.percentile(small, [lo_pct, hi_pct]).astype(np.float32)
            target[1] = max(target[1], target[0] + 32)  # never amplify a near-flat frame
            self._levels = target if self._levels is None else self._levels + .08 * (target - self._levels)
            black, white = self._levels
            stretched = (out - black) * (255.0 / (white - black))
            out = out + cfg["autolevel"] * (stretched - out)
        if cfg["gamma"] != 1.0:
            out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, cfg["gamma"])
        return out


class RenderTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderInfo:
    ffmpeg_command: list[str]  # as run, with the final output path substituted for the temp file
    frames: int
    width: int
    height: int
    crf: int


def render(
    *, audio: Path, output: Path, analysis: Analysis, source: Path | None,
    preset: Preset | str | Path, width: int, height: int, seed: int, crf: int = 20,
    max_seconds: float | None = None,
) -> RenderInfo:
    """Render to ``<output>.partial.mp4`` and rename on success, so a cancelled, failed or timed-out
    render never leaves a truncated file or clobbers an existing output."""
    if width < 32 or height < 32 or width % 2 or height % 2:
        raise ValueError("width and height must be even and at least 32")
    if not (0 <= crf <= 51):
        raise ValueError("crf must be 0..51")
    rng = np.random.default_rng(seed)
    visual = FeedbackRenderer(load_source(source, width, height, rng), preset=preset, seed=seed, fps=analysis.fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial" + output.suffix)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(analysis.fps), "-i", "pipe:0", "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-t", f"{analysis.duration:.6f}",
        "-movflags", "+faststart", str(partial),
    ]
    deadline = None if max_seconds is None else time.monotonic() + max_seconds
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        frames = len(analysis.features["rms"])
        for i in range(frames):
            if deadline is not None and time.monotonic() > deadline:
                raise RenderTimeout(f"render exceeded --max-seconds {max_seconds} at frame {i}/{frames}")
            frame_features = {k: float(v[i]) for k, v in analysis.features.items()}
            proc.stdin.write(visual.frame(i, frame_features).tobytes())
        proc.stdin.close()
        proc.stdin = None
        _, stderr = proc.communicate()
        if proc.returncode:
            raise RuntimeError(f"ffmpeg encoding failed: {stderr.decode(errors='replace')}")
        os.replace(partial, output)
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        partial.unlink(missing_ok=True)
        raise
    return RenderInfo([*cmd[:-1], str(output)], frames, width, height, crf)
