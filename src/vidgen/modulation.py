"""Declarative modulation: connect any audio feature (or LFO) to any named visual parameter.

A preset file (TOML) holds base ``[params]``, static ``[settings]`` and ordered ``[[mappings]]``.
Each mapping evaluates as: source -> smooth -> value*scale+offset -> clip -> combine into param.
"""
from __future__ import annotations

import copy
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PRESET_DIR = Path(__file__).parent / "presets"
FEATURES = ("low", "mid", "high", "rms", "flux", "slow")
SOURCES = FEATURES + ("lfo",)
MODES = ("add", "multiply", "blend")

# name -> (default base value, (min, max)); final values are always clamped to the bounds.
PARAMS: dict[str, tuple[float, tuple[float, float]]] = {
    "zoom": (1.0, (.9, 1.1)),
    "rotation": (0.0, (-5, 5)),
    "shift_x": (0.0, (-8, 8)),
    "shift_y": (0.0, (-8, 8)),
    "warp": (0.0, (0, 20)),
    "warp_phase_x": (0.0, (-100, 100)),
    "warp_phase_y": (0.0, (-100, 100)),
    "persistence": (.97, (.8, .999)),
    "edge_gain": (0.0, (0, 1)),
    "glitch": (0.0, (0, 1)),
    "brightness": (1.0, (.4, 1.6)),
    "hue_shift": (0.0, (-180, 180)),
    "grain": (0.0, (0, 40)),
}
SETTINGS: dict[str, Any] = {
    "tint": [1.0, 1.0, 1.0],  # BGR gain on the displayed copy only
    "levels": 12,
    "saturation": 1.0,
    "blur": False,
    "warp_speed": [.63, .43],  # rad/s drift of the warp field
    "glitch_threshold": .45,
}


class PresetError(ValueError):
    pass


@dataclass(frozen=True)
class Mapping:
    source: str
    param: str
    scale: float = 1.0
    offset: float = 0.0
    smooth: float = 0.0
    clip: tuple[float, float] | None = None
    mode: str = "add"
    amount: float = 1.0
    rate_hz: float = 0.0
    phase_deg: float = 0.0


@dataclass(frozen=True)
class Preset:
    name: str
    description: str = ""
    params: dict[str, float] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    mappings: tuple[Mapping, ...] = ()


def _num(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PresetError(f"{what} must be a finite number, got {value!r}")
    return float(value)


def _mapping(raw: dict[str, Any], i: int) -> Mapping:
    where = f"mappings[{i}]"
    allowed = {"feature", "param", "scale", "offset", "smooth", "clip", "mode", "amount", "rate_hz", "phase_deg"}
    unknown = set(raw) - allowed
    if unknown:
        raise PresetError(f"{where}: unknown keys {sorted(unknown)}")
    source, param = raw.get("feature"), raw.get("param")
    if source not in SOURCES:
        raise PresetError(f"{where}: feature must be one of {SOURCES}, got {source!r}")
    if param not in PARAMS:
        raise PresetError(f"{where}: param must be one of {sorted(PARAMS)}, got {param!r}")
    mode = raw.get("mode", "add")
    if mode not in MODES:
        raise PresetError(f"{where}: mode must be one of {MODES}, got {mode!r}")
    smooth = _num(raw.get("smooth", 0.0), f"{where}.smooth")
    if not 0 <= smooth < 1:
        raise PresetError(f"{where}: smooth must be in [0, 1)")
    amount = _num(raw.get("amount", 1.0), f"{where}.amount")
    if not 0 <= amount <= 1:
        raise PresetError(f"{where}: amount must be in [0, 1]")
    clip = raw.get("clip")
    if clip is not None:
        if not isinstance(clip, (list, tuple)) or len(clip) != 2:
            raise PresetError(f"{where}: clip must be [min, max]")
        clip = (_num(clip[0], f"{where}.clip[0]"), _num(clip[1], f"{where}.clip[1]"))
        if clip[0] > clip[1]:
            raise PresetError(f"{where}: clip min exceeds max")
    return Mapping(
        source=source, param=param, scale=_num(raw.get("scale", 1.0), f"{where}.scale"),
        offset=_num(raw.get("offset", 0.0), f"{where}.offset"), smooth=smooth, clip=clip, mode=mode,
        amount=amount, rate_hz=_num(raw.get("rate_hz", 0.0), f"{where}.rate_hz"),
        phase_deg=_num(raw.get("phase_deg", 0.0), f"{where}.phase_deg"),
    )


def parse_preset(data: dict[str, Any], default_name: str = "custom") -> Preset:
    unknown = set(data) - {"name", "description", "params", "settings", "mappings"}
    if unknown:
        raise PresetError(f"unknown top-level keys {sorted(unknown)}")
    params = {k: _num(v, f"params.{k}") for k, v in data.get("params", {}).items()}
    bad = set(params) - set(PARAMS)
    if bad:
        raise PresetError(f"unknown params {sorted(bad)}; known: {sorted(PARAMS)}")
    settings = copy.deepcopy(data.get("settings", {}))
    bad = set(settings) - set(SETTINGS)
    if bad:
        raise PresetError(f"unknown settings {sorted(bad)}; known: {sorted(SETTINGS)}")
    for key in ("tint", "warp_speed"):
        if key in settings:
            n = 3 if key == "tint" else 2
            if not isinstance(settings[key], list) or len(settings[key]) != n:
                raise PresetError(f"settings.{key} must be a list of {n} numbers")
            settings[key] = [_num(v, f"settings.{key}") for v in settings[key]]
    mappings = tuple(_mapping(m, i) for i, m in enumerate(data.get("mappings", [])))
    return Preset(str(data.get("name", default_name)), str(data.get("description", "")), params, settings, mappings)


def builtin_names() -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in PRESET_DIR.glob("*.toml")))


def load_preset(name_or_path: str | Path) -> Preset:
    """Load a built-in preset by name, or any preset TOML by path. Every call parses afresh."""
    path = Path(name_or_path)
    if not path.suffix:
        path = PRESET_DIR / f"{name_or_path}.toml"
        if not path.is_file():
            raise PresetError(f"Unknown preset {name_or_path!r}. Built-ins: {builtin_names()}")
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PresetError(f"Cannot read preset {path}: {exc}") from exc
    return parse_preset(data, default_name=path.stem)


class ModulationEngine:
    """Evaluates a preset frame by frame. Owns all smoothing state, so engines never interact."""

    def __init__(self, preset: Preset, fps: float):
        self.preset = preset
        self.fps = fps
        self._smoothed: list[float | None] = [None] * len(preset.mappings)
        self.base = {k: v[0] for k, v in PARAMS.items()}
        self.base.update(preset.params)
        self.settings = {**copy.deepcopy(SETTINGS), **copy.deepcopy(preset.settings)}

    def params(self, index: int, features: dict[str, float]) -> dict[str, float]:
        t = index / self.fps
        out = dict(self.base)
        for i, m in enumerate(self.preset.mappings):
            if m.source == "lfo":
                raw = math.sin(2 * math.pi * m.rate_hz * t + math.radians(m.phase_deg))
            else:
                raw = float(features.get(m.source, 0.0))
                if not math.isfinite(raw):
                    raw = 0.0
            prev = self._smoothed[i]
            value = raw if prev is None or m.smooth == 0 else m.smooth * prev + (1 - m.smooth) * raw
            self._smoothed[i] = value
            value = value * m.scale + m.offset
            if m.clip is not None:
                value = min(max(value, m.clip[0]), m.clip[1])
            cur = out[m.param]
            if m.mode == "add":
                out[m.param] = cur + value
            elif m.mode == "multiply":
                out[m.param] = cur * value
            else:
                out[m.param] = cur * (1 - m.amount) + value * m.amount
        for k, v in out.items():
            lo, hi = PARAMS[k][1]
            out[k] = min(max(v, lo), hi)
        return out
