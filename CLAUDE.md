# vidgen: Claude Code project instructions

Read [docs/CLAUDE_HANDOFF.md](docs/CLAUDE_HANDOFF.md) before continuing implementation.

- Purpose: general-purpose, genre-agnostic audio-reactive generative video tool. Megadroan is an example, not a hard-coded target.
- Hard constraint: use **only free/open-source software** and review AI model *weight* licenses independently. No hosted AI APIs, proprietary visualizers, paid plugins, or assumed open-licensed checkpoints.
- Hardware: **NVIDIA RTX 3060, 12 GB VRAM**. Prefer algorithmic rendering and short previews over frame-by-frame diffusion.
- ComfyUI: optional still-image provider on the user's local server, default http://127.0.0.1:8188. Use the API; inspect installed nodes/models instead of assuming them. Never start a long render or download a model without establishing need and licensing.
- Preserve the CPU-only path without ComfyUI: procedural or local-image input → audio feature extraction → feedback renderer → FFmpeg synchronized output.
- Commands: `uv sync --extra dev`, `uv run --extra dev pytest -q`, `uv run vidgen doctor`, and `uv run vidgen render --audio INPUT --duration 30 --output outputs/preview.mp4`.
- Prefer short, reproducible comparisons on the same audio with alternate presets. Keep generated media in `outputs/` and out of git.
- A successful unit test does **not** mean the ComfyUI integration is verified. Test it against the user's running instance and report results accurately.
- Next priority: data-driven modulation mappings, visual-quality evaluation, and a reproducibility manifest. See handoff milestones and acceptance tests.
