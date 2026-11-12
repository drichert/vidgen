"""Minimal local ComfyUI API client; expects exported API-format workflow JSON."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests


class ComfyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComfyResult:
    path: Path
    prompt_id: str
    resumed: bool  # an earlier submission was collected or resumed; nothing new was queued
    workflow_sha256: str
    reused_file: bool = False


def workflow_sha256(workflow: dict) -> str:
    return hashlib.sha256(json.dumps(workflow, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_job(path: Path | None) -> dict | None:
    try:
        data = json.loads(path.read_text()) if path else None
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and "prompt_id" in data else None


def _write_job(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, path)


def load_workflow(path: Path) -> dict:
    raw = json.loads(path.read_text())
    workflow = raw.get("prompt", raw) if isinstance(raw, dict) else raw
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("Expected a nonempty ComfyUI API-format workflow")
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or "class_type" not in node or "inputs" not in node:
            raise ValueError(f"Node {node_id} is not API-format; export using File > Export Workflow (API)")
    return workflow


def override_workflow(workflow: dict, assignments: list[str]) -> dict:
    result = copy.deepcopy(workflow)
    for expression in assignments:
        if "=" not in expression:
            raise ValueError(f"Invalid --comfy-set {expression!r}; use NODE.FIELD=JSON_VALUE")
        path, value_text = expression.split("=", 1)
        if "." not in path:
            raise ValueError(f"Invalid --comfy-set {expression!r}; use NODE.FIELD=JSON_VALUE")
        node_id, field = path.split(".", 1)
        if node_id not in result or field not in result[node_id]["inputs"]:
            raise ValueError(f"Unknown ComfyUI input {node_id}.{field}")
        try:
            value = json.loads(value_text)
        except json.JSONDecodeError:
            value = value_text
        result[node_id]["inputs"][field] = value
    return result


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, route: str, **kwargs) -> requests.Response:
        response = self.session.get(f"{self.base_url}{route}", timeout=20, **kwargs)
        response.raise_for_status()
        return response

    def check_nodes(self, workflow: dict) -> None:
        try:
            available = self._get("/object_info").json()
        except requests.RequestException as exc:
            raise ComfyError(f"ComfyUI is not reachable at {self.base_url}: {exc}") from exc
        missing = sorted({node["class_type"] for node in workflow.values()} - set(available))
        if missing:
            raise ComfyError(f"Missing ComfyUI node classes: {', '.join(missing)}")

    def _find_job(self, prompt_id: str) -> tuple[str, dict | None]:
        """Where does the server know this prompt from: ('history', entry), ('queue', None) or ('unknown', None)."""
        try:
            entry = self._get(f"/history/{prompt_id}").json().get(prompt_id)
            if entry is not None:
                return "history", entry
            queue = self._get("/queue").json()
        except requests.RequestException as exc:
            raise ComfyError(f"ComfyUI status check failed for {prompt_id}: {exc}") from exc
        for item in list(queue.get("queue_running", [])) + list(queue.get("queue_pending", [])):
            if isinstance(item, (list, tuple)) and prompt_id in item[:2]:
                return "queue", None
        return "unknown", None

    def _submit(self, workflow: dict) -> str:
        try:
            response = self.session.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow, "client_id": str(uuid.uuid4())}, timeout=30,
            )
            response.raise_for_status()
            submitted = response.json()
        except requests.RequestException as exc:
            raise ComfyError(f"Failed to submit ComfyUI workflow: {exc}") from exc
        if "prompt_id" not in submitted:
            raise ComfyError(f"ComfyUI rejected workflow: {submitted}")
        return submitted["prompt_id"]

    def _fetch_image(self, prompt_id: str, entry: dict, target: Path) -> Path:
        status = entry.get("status", {})
        if status.get("status_str") == "error" or status.get("completed") is False:
            raise ComfyError(f"ComfyUI execution failed for {prompt_id}: {status}")
        for output_node in entry.get("outputs", {}).values():
            for image in output_node.get("images", []):
                params = {"filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": image.get("type", "output")}
                result = self._get("/view", params=params)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(result.content)
                return target
        raise ComfyError(f"Workflow {prompt_id} completed without an image output. Add a SaveImage node.")

    def generate(self, workflow: dict, target: Path, job_path: Path | None = None, resubmit: bool = False) -> ComfyResult:
        """Generate a still. With ``job_path`` the prompt_id is persisted before polling, so a rerun
        of the same workflow resumes or collects the earlier job instead of queueing a duplicate."""
        self.check_nodes(workflow)
        digest = workflow_sha256(workflow)
        prompt_id, resumed = None, False
        record = _read_job(job_path) if job_path and not resubmit else None
        if record and record.get("workflow_sha256") == digest:
            if record.get("status") == "done" and target.is_file():
                return ComfyResult(target, record["prompt_id"], True, digest, reused_file=True)
            where, _ = self._find_job(record["prompt_id"])
            if where != "unknown":
                prompt_id, resumed = record["prompt_id"], True
        if prompt_id is None:
            prompt_id = self._submit(workflow)
            if job_path:
                _write_job(job_path, {"prompt_id": prompt_id, "workflow_sha256": digest, "url": self.base_url,
                                      "submitted_utc": datetime.now(timezone.utc).isoformat(), "status": "submitted"})
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            where, entry = self._find_job(prompt_id)
            if entry is not None:
                self._fetch_image(prompt_id, entry, target)
                if job_path:
                    _write_job(job_path, {**(_read_job(job_path) or {}), "status": "done", "image": str(target)})
                return ComfyResult(target, prompt_id, resumed, digest)
            time.sleep(1.0)
        hint = f"; rerun the same command to resume it ({job_path})" if job_path else ""
        raise ComfyError(f"ComfyUI timed out after {self.timeout}s; prompt_id={prompt_id} (check server history before retrying{hint})")
