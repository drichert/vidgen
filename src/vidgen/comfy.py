"""Minimal local ComfyUI API client; expects exported API-format workflow JSON."""
from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path

import requests


class ComfyError(RuntimeError):
    pass


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

    def generate(self, workflow: dict, target: Path) -> Path:
        self.check_nodes(workflow)
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
        prompt_id = submitted["prompt_id"]
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                entry = self._get(f"/history/{prompt_id}").json().get(prompt_id)
            except requests.RequestException as exc:
                raise ComfyError(f"ComfyUI history failed for {prompt_id}: {exc}") from exc
            if entry is None:
                time.sleep(1.0)
                continue
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
        raise ComfyError(f"ComfyUI timed out after {self.timeout}s; prompt_id={prompt_id} (check server history before retrying)")
