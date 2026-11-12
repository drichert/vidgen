import json
import shutil

import numpy as np
import pytest

from vidgen.audio import analyze
from vidgen.comfy import ComfyClient, ComfyError, workflow_sha256
from vidgen.manifest import build_manifest, comfy_models, sha256_file
from vidgen.modulation import load_preset
from vidgen.renderer import RenderTimeout, render

from test_smoke import make_audio

WORKFLOW = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
    "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
}
PNG = b"\x89PNG fake"


class FakeResponse:
    def __init__(self, payload=None, content=b""):
        self.payload, self.content = payload, content

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeComfy:
    """In-process stand-in for the ComfyUI HTTP API. Offline test double only."""

    def __init__(self):
        self.submitted, self.history, self.queue = [], {}, []

    def get(self, url, params=None, timeout=None):
        route = url.split("8188", 1)[1]
        if route == "/object_info":
            return FakeResponse({"CheckpointLoaderSimple": {}, "SaveImage": {}})
        if route == "/queue":
            return FakeResponse({"queue_running": [[0, pid, {}] for pid in self.queue], "queue_pending": []})
        if route.startswith("/history/"):
            pid = route.rsplit("/", 1)[1]
            return FakeResponse({pid: self.history[pid]} if pid in self.history else {})
        if route.startswith("/view"):
            return FakeResponse(content=PNG)
        raise AssertionError(route)

    def post(self, url, json=None, timeout=None):
        pid = f"p{len(self.submitted) + 1}"
        self.submitted.append(pid)
        self.queue.append(pid)
        return FakeResponse({"prompt_id": pid})

    def finish(self, pid):
        self.queue.remove(pid)
        self.history[pid] = {"status": {"status_str": "success", "completed": True},
                             "outputs": {"2": {"images": [{"filename": "a.png"}]}}}


def client(fake, timeout=.05):
    c = ComfyClient(timeout=timeout)
    c.session = fake
    return c


def test_comfy_timeout_then_rerun_resumes_same_prompt_without_resubmitting(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake, target, job = FakeComfy(), tmp_path / "s.png", tmp_path / "s.png.job.json"
    with pytest.raises(ComfyError, match="prompt_id=p1"):
        client(fake).generate(WORKFLOW, target, job_path=job)
    assert json.loads(job.read_text())["prompt_id"] == "p1"
    fake.finish("p1")
    result = client(fake).generate(WORKFLOW, target, job_path=job)
    assert (result.prompt_id, result.resumed) == ("p1", True)
    assert fake.submitted == ["p1"] and target.read_bytes() == PNG


def test_comfy_finished_job_is_reused_without_server_work(tmp_path):
    fake, target, job = FakeComfy(), tmp_path / "s.png", tmp_path / "s.png.job.json"
    fake.post = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not submit"))
    job.write_text(json.dumps({"prompt_id": "p9", "workflow_sha256": workflow_sha256(WORKFLOW), "status": "done"}))
    target.write_bytes(PNG)
    result = client(fake).generate(WORKFLOW, target, job_path=job)
    assert result.reused_file and result.prompt_id == "p9"


def test_comfy_changed_workflow_or_resubmit_or_vanished_job_queues_new_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    target, job = tmp_path / "s.png", tmp_path / "s.png.job.json"
    for record_hash, kwargs in (("other", {}), (workflow_sha256(WORKFLOW), {"resubmit": True}), (workflow_sha256(WORKFLOW), {})):
        fake = FakeComfy()
        job.write_text(json.dumps({"prompt_id": "old", "workflow_sha256": record_hash, "status": "submitted"}))
        with pytest.raises(ComfyError):
            client(fake).generate(WORKFLOW, target, job_path=job, **kwargs)
        assert fake.submitted == ["p1"]  # "old" is unknown to the server, or deliberately ignored


def test_workflow_hash_is_stable_and_order_independent():
    assert workflow_sha256(WORKFLOW) == workflow_sha256(json.loads(json.dumps(WORKFLOW, sort_keys=True)))
    changed = json.loads(json.dumps(WORKFLOW))
    changed["1"]["inputs"]["ckpt_name"] = "x"
    assert workflow_sha256(changed) != workflow_sha256(WORKFLOW)


def test_model_licenses_are_never_inferred():
    models = comfy_models(WORKFLOW, {})
    assert models[0]["file"] == "model.safetensors" and models[0]["license"] == "UNVERIFIED"
    assert comfy_models(WORKFLOW, {"model.safetensors": "Apache-2.0"})[0]["license"] == "Apache-2.0"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_manifest_contents_and_partial_file_handling(tmp_path):
    audio, out = tmp_path / "a.wav", tmp_path / "clip.mp4"
    make_audio(audio)
    feats = analyze(audio, fps=8, duration=.5)
    preset = load_preset("minimal-signal")
    info = render(audio=audio, output=out, analysis=feats, source=None, preset=preset, width=96, height=64, seed=3)
    assert info.frames == 4 and info.ffmpeg_command[-1] == str(out)
    assert not list(tmp_path.glob("*.partial*"))
    m = build_manifest(audio=audio, output=out, source=None, preset=preset, seed=3, fps=8, width=96, height=64,
                       duration=feats.duration, crf=20, ffmpeg_command=info.ffmpeg_command, frames=info.frames, cli_args=["render"])
    json.dumps(m)
    assert m["audio"]["sha256"] == sha256_file(audio) and m["output"]["sha256"] == sha256_file(out)
    assert m["seed"] == 3 and m["preset"]["name"] == "minimal-signal" and len(m["preset"]["mappings"]) > 0
    assert m["source"]["kind"] == "procedural" and m["comfy"] is None
    # editing a preset changes its recorded hash
    other = load_preset("analog-feedback")
    assert build_manifest.__globals__["preset_record"](other)["sha256"] != m["preset"]["sha256"]

    before = out.read_bytes()
    with pytest.raises(RenderTimeout):
        render(audio=audio, output=out, analysis=feats, source=None, preset=preset, width=96, height=64, seed=3, max_seconds=-1)
    assert out.read_bytes() == before and not list(tmp_path.glob("*.partial*"))
