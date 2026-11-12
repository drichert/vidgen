"""Fast offline tests; no GPU, network or ComfyUI installation required."""
import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from vidgen.audio import analyze
from vidgen.comfy import load_workflow, override_workflow
from vidgen.renderer import PRESETS, FeedbackRenderer, load_source, render


def make_audio(path: Path, seconds: float = .8) -> None:
    sr = 22050
    x = np.arange(int(sr * seconds)) / sr
    signal = (.3 * np.sin(2 * np.pi * 80 * x) + .1 * np.sin(2 * np.pi * 700 * x))
    data = np.asarray(signal * 32767, dtype="<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sr)
        writer.writeframes(data.tobytes())


def test_audio_features(tmp_path):
    path = tmp_path / "sample.wav"
    make_audio(path)
    data = analyze(path, fps=10, duration=.5)
    assert len(data.features["low"]) == 5
    assert set(data.features) == {"rms", "low", "mid", "high", "flux", "slow"}
    assert all(np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all() for v in data.features.values())


@pytest.mark.parametrize("preset", PRESETS)
def test_deterministic_visuals(preset):
    def frames():
        src = load_source(None, 96, 64, np.random.default_rng(42))
        engine = FeedbackRenderer(src, preset=preset, seed=42, fps=12)
        return [engine.frame(i, {"low": .4, "mid": .6, "high": .3, "flux": .5, "slow": .5}) for i in range(3)]
    assert all(np.array_equal(a, b) for a, b in zip(frames(), frames()))


def test_comfy_json_validation_and_input_overrides(tmp_path):
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps({"3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 12}}}))
    original = load_workflow(path)
    modified = override_workflow(original, ["3.seed=42", "3.steps=6"])
    assert modified["3"]["inputs"]["seed"] == 42
    assert original["3"]["inputs"]["seed"] == 1
    with pytest.raises(ValueError):
        override_workflow(original, ["4.seed=1"])


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_end_to_end_mp4(tmp_path):
    audio = tmp_path / "sample.wav"
    output = tmp_path / "clip.mp4"
    make_audio(audio)
    features = analyze(audio, fps=8, duration=.5)
    render(audio=audio, output=output, analysis=features, source=None,
           preset="analog-feedback", width=96, height=64, seed=7)
    assert output.exists() and output.stat().st_size > 1000
    if shutil.which("ffprobe"):
        data = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(output)], text=True)
        assert "video" in data and "audio" in data
