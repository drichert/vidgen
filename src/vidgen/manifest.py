"""Reproducibility manifest: everything needed to re-run, audit or cite a render."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .modulation import Preset

SCHEMA_VERSION = 1
MODEL_INPUT_FIELDS = ("ckpt_name", "lora_name", "vae_name", "unet_name", "clip_name", "control_net_name")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def preset_record(preset: Preset) -> dict[str, Any]:
    content = dataclasses.asdict(preset)
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return {"sha256": hashlib.sha256(canonical.encode()).hexdigest(), **content}


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("vidgen", "numpy", "opencv-python-headless", "scipy", "requests"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=cwd).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_state() -> dict[str, Any] | None:
    here = Path(__file__).parent
    commit = _run(["git", "rev-parse", "HEAD"], here)
    if not commit:
        return None
    return {"commit": commit, "dirty": bool(_run(["git", "status", "--porcelain"], here))}


def comfy_models(workflow: dict, licenses: dict[str, str]) -> list[dict[str, str]]:
    """Model files referenced by the workflow. Licenses are only what the operator supplied: never inferred."""
    seen: dict[str, dict[str, str]] = {}
    for node_id, node in workflow.items():
        for field in MODEL_INPUT_FIELDS:
            value = node.get("inputs", {}).get(field)
            if isinstance(value, str):
                seen.setdefault(value, {
                    "file": value, "node": f"{node_id}:{node.get('class_type')}.{field}",
                    "license": licenses.get(value, "UNVERIFIED"),
                    "note": "weights not hashed (file lives on the ComfyUI server)",
                })
    return sorted(seen.values(), key=lambda m: m["file"])


def build_manifest(
    *, audio: Path, output: Path, source: Path | None, preset: Preset, seed: int, fps: int,
    width: int, height: int, duration: float, crf: int, ffmpeg_command: list[str], frames: int,
    cli_args: list[str], comfy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ffmpeg_version = _run(["ffmpeg", "-version"])
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": _versions(),
        "git": _git_state(),
        "ffmpeg_version": ffmpeg_version.splitlines()[0] if ffmpeg_version else None,
        "command_line": cli_args,
        "audio": _file_record(audio),
        "source": _file_record(source) if source else {"kind": "procedural", "note": "generated from seed"},
        "preset": preset_record(preset),
        "seed": seed,
        "video": {"fps": fps, "width": width, "height": height, "duration_s": round(duration, 6), "frames": frames, "crf": crf},
        "ffmpeg_command": ffmpeg_command,
        "output": _file_record(output),
        "comfy": comfy,
        "reproducibility_note": "Decoded frames are expected to match on the same environment; encoded bytes may differ across ffmpeg/OpenCV builds.",
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, path)
