import math

import numpy as np
import pytest

from vidgen.modulation import ModulationEngine, PresetError, builtin_names, load_preset, parse_preset
from vidgen.renderer import FeedbackRenderer, load_source

FEATS = {"low": 0, "mid": 0, "high": 0, "rms": 0, "flux": 0, "slow": 0}


def engine(mappings, params=None, fps=10):
    return ModulationEngine(parse_preset({"params": params or {}, "mappings": mappings}), fps)


def test_builtin_presets_load_and_have_lfo_and_audio_mappings():
    assert set(builtin_names()) >= {"analog-feedback", "image-feedback", "industrial-decay", "minimal-signal"}
    for name in builtin_names():
        preset = load_preset(name)
        assert {m.source for m in preset.mappings} & {"low", "mid", "high", "flux", "slow"}


def test_scale_offset_and_any_feature_to_any_param():
    e = engine([{"feature": "high", "param": "warp", "scale": .5, "offset": .1}], {"warp": 1.0})
    assert e.params(0, {**FEATS, "high": .4})["warp"] == pytest.approx(1.3)


def test_modes_add_multiply_blend():
    f = {**FEATS, "low": .5}
    add = engine([{"feature": "low", "param": "warp", "mode": "add"}], {"warp": 1.0}).params(0, f)["warp"]
    mul = engine([{"feature": "low", "param": "warp", "mode": "multiply", "scale": 3}], {"warp": 2.0}).params(0, f)["warp"]
    blend = engine([{"feature": "low", "param": "warp", "mode": "blend", "amount": .5, "scale": 4}], {"warp": 1.0}).params(0, f)["warp"]
    assert (add, mul, blend) == pytest.approx((1.5, 3.0, 1.5))  # blend: 1*(1-.5) + 2*.5


def test_clip_applies_before_combining_and_param_bounds_always_hold():
    e = engine([{"feature": "low", "param": "warp", "scale": 100, "clip": [0, 3]}], {"warp": 1})
    assert e.params(0, {**FEATS, "low": 1})["warp"] == pytest.approx(4)
    e = engine([{"feature": "low", "param": "warp", "scale": 1000}])
    assert e.params(0, {**FEATS, "low": 1})["warp"] == 20  # bound


def test_smoothing_is_exponential_and_stateful():
    e = engine([{"feature": "low", "param": "warp", "smooth": .5}])
    seq = [e.params(i, {**FEATS, "low": 1.0})["warp"] for i in range(3)]
    assert seq == pytest.approx([1.0, 1.0, 1.0])  # first sample seeds the filter
    e = engine([{"feature": "low", "param": "warp", "smooth": .5}])
    e.params(0, FEATS)
    seq = [e.params(i, {**FEATS, "low": 1.0})["warp"] for i in range(1, 4)]
    assert seq == pytest.approx([.5, .75, .875])


def test_lfo_is_deterministic_and_moves_without_audio():
    e = engine([{"feature": "lfo", "param": "shift_x", "scale": 2, "rate_hz": 1.0}])
    v = [e.params(i, FEATS)["shift_x"] for i in range(10)]
    assert v[0] == pytest.approx(0) and v[2] == pytest.approx(2 * math.sin(2 * math.pi * .2))
    assert len(set(np.round(v, 6))) > 3


def test_nonfinite_feature_is_ignored():
    e = engine([{"feature": "low", "param": "warp"}], {"warp": 1})
    assert e.params(0, {**FEATS, "low": float("nan")})["warp"] == 1


@pytest.mark.parametrize("bad", [
    {"feature": "bass", "param": "zoom"},
    {"feature": "low", "param": "nope"},
    {"feature": "low", "param": "zoom", "mode": "xor"},
    {"feature": "low", "param": "zoom", "smooth": 1.0},
    {"feature": "low", "param": "zoom", "scale": float("inf")},
    {"feature": "low", "param": "zoom", "clip": [2, 1]},
    {"feature": "low", "param": "zoom", "amount": 2},
    {"feature": "low", "param": "zoom", "typo": 1},
])
def test_invalid_mappings_rejected(bad):
    with pytest.raises(PresetError):
        parse_preset({"mappings": [bad]})


def test_invalid_params_settings_and_names_rejected():
    for data in ({"params": {"bogus": 1}}, {"settings": {"bogus": 1}}, {"extra": 1}, {"settings": {"tint": [1, 2]}}):
        with pytest.raises(PresetError):
            parse_preset(data)
    with pytest.raises(PresetError):
        load_preset("does-not-exist")


def test_engines_and_loads_are_isolated():
    a, b = load_preset("analog-feedback"), load_preset("analog-feedback")
    assert a is not b
    e1, e2 = ModulationEngine(a, 12), ModulationEngine(a, 12)
    e1.settings["levels"] = 3
    e1.base["zoom"] = 5
    assert e2.settings["levels"] == a.settings["levels"] and e2.base["zoom"] != 5
    assert load_preset("analog-feedback").settings.get("levels") == 12
    # smoothing state is per engine
    s1 = ModulationEngine(parse_preset({"mappings": [{"feature": "low", "param": "warp", "smooth": .9}]}), 10)
    s2 = ModulationEngine(s1.preset, 10)
    s1.params(0, {**FEATS, "low": 1})
    assert s2.params(0, FEATS)["warp"] == 0


def test_user_config_file_changes_three_renderer_params(tmp_path):
    cfg = tmp_path / "mine.toml"
    cfg.write_text("""
[params]
persistence = 0.95
[[mappings]]
feature = "rms"
param = "zoom"
scale = 0.05
[[mappings]]
feature = "flux"
param = "hue_shift"
scale = 40
[[mappings]]
feature = "high"
param = "warp"
scale = 6
""")
    e = ModulationEngine(load_preset(cfg), 12)
    quiet, loud = e.params(0, FEATS), e.params(1, {**FEATS, "rms": 1, "flux": 1, "high": 1})
    assert [loud[k] > quiet[k] for k in ("zoom", "hue_shift", "warp")] == [True] * 3


def test_same_audio_different_preset_files_differ_and_same_preset_reproduces():
    src = load_source(None, 96, 64, np.random.default_rng(1))
    feats = [{**FEATS, "low": .5, "mid": .4, "flux": .6, "high": .3, "slow": .5}] * 20

    def run(name):
        r = FeedbackRenderer(src, preset=name, seed=9, fps=12)
        return [r.frame(i, f) for i, f in enumerate(feats)][-1]
    assert np.array_equal(run("analog-feedback"), run("analog-feedback"))
    assert not np.array_equal(run("analog-feedback"), run("industrial-decay"))
