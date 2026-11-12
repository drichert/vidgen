# Claude handoff: vidgen v0.1 → modular video instrument

## Context and non-negotiable constraints

This repository is a **general-purpose**, local, audio-reactive generative video engine. It must work on any audio material, not just a specific artist, genre or track. ComfyUI is an **optional visual-source provider**, controlled by its *local* HTTP API. The reference GPU is **NVIDIA RTX 3060 12 GB**. All software/tools used must be free and genuinely open source. No commercial API, freemium proprietary editor, or bundled restricted-weight checkpoint. Do not mistake “downloadable weights” for “open source weights.”

## Existing implementation (verify it before changing)

- `uv run vidgen analyze`: frame-aligned RMS, low/mid/high band energies, spectral flux, slow dynamics.
- `uv run vidgen render`: procedural or existing-image OpenCV recursive-feedback renderer and ffmpeg audio/video muxing; three aesthetic presets.
- `--comfy-workflow`: optional local ComfyUI API-format still-image generation; node existence checks via `/object_info`, `POST /prompt`, polling `GET /history/{prompt_id}`, `GET /view` to fetch image.
- Unit/integration smoke tests are offline; no running ComfyUI or GPU required.

## Your first pass

1. Run `uv sync --extra dev`, `uv run --extra dev pytest -q`, and `uv run vidgen doctor` on the local development machine. Check ffmpeg's H.264 encoder (`ffmpeg -encoders | grep libx264`). Report any environmental blockers accurately.
2. Run a 10–30 second preview from a user-supplied audio excerpt. Use 640×360 at 12 fps; do not assume a local audio path or existing checkpoint. Check whether image changes, temporal smoothness and audio sync are genuinely convincing. Save preview files in ignored `outputs/`.
3. If ComfyUI is available, query `GET /object_info` and `GET /models/checkpoints` (where supported), inventory installed compatible nodes/models, and request/choose a checkpoint *only after verifying its model weights license*. Never silently download/replace model files. Export or construct a **valid API-format still workflow**, then validate with the real API.
4. Run at least two presets on the same audio, examine the outputs and compare low-frequency response, sustained motion, transient treatment, and aesthetic distinctiveness.
5. Implement a **declarative modulation mapping** format that connects any audio feature to any named visual parameter with scale, offset, smoothing, clipping, and blend/add semantics. Keep presets as separate files and include tests. Avoid hard-coding ‘bass always means zoom’ as a permanent design.
6. Add manifest-based reproducibility: versions, FFmpeg command, source hashes, checkpoint provenance/license (if ComfyUI used), seeds, mappings, output path, duration. Add graceful cancellation/timeout and idempotent ComfyUI job recovery without duplicating expensive prompts blindly.
7. Iterate on visual quality (color palettes, masks, multiscale feedback, controlled compositing, transitions), without making the engine a generic waveform/spectrum visualizer. Consider GLSL/ModernGL only if justified by profiling; keep CPU fallback.
8. Run tests and include exact commands, actual preview artifacts where feasible, and clear PR checkboxes separating completed from untested work. No performance claims without measurements on the target machine.

## Design constraints

- **Any audio file** FFmpeg can decode; no required BPM or beat grid; slow features for drone and ambient, flux for articulated material.
- **Any source:** procedural/noise, local image, ComfyUI-generated still; later can add loops/clips and layered sources.
- **No mandatory ComfyUI or GPU for initial renders.** Avoid a fragile custom-node dependency stack.
- **Determinism:** explicit seeds and bounded effect behavior. Provide a fast preview mode before high-quality output.
- **API:** never assume exported node IDs are stable; `NODE.FIELD=VALUE` overrides only existing inputs. Validate node schemas. A saved UI workflow cannot be passed directly as an API workflow.
- **Error recovery:** ComfyUI queue timeouts are ambiguous outcomes; capture `prompt_id` so callers can retrieve status rather than resubmitting automatically.
- **Security:** default ComfyUI to `127.0.0.1:8188`; do not expose it publicly or send private audio/files to hosted APIs.

## First milestone acceptance criteria

- [ ] Offline smoke tests pass on target machine.
- [ ] Both generated-texture and supplied-image renders produce playable MP4 with audio at requested duration.
- [ ] Changing seed/preset actually changes visuals; the same seed/settings reproduce decoded frames on the same environment.
- [ ] Sustained tones and percussive material both yield sensible curves (no NaN/inf or flatlining on every feature).
- [ ] A supplied ComfyUI API workflow can generate a still and drive video, with verified installed node/model support.
- [ ] A user-editable modulation config can influence at least three renderer parameters; tests validate mappings and preset isolation.
- [ ] Clear provenance and license records for optional generative models; no pretrained checkpoint assumed open-source.

## Future extensions (do not scope-creep v0.1)

Shaders, video-source reactivity, patch graph, live MIDI/OSC, GPU feedback, multitrack audio, real-time previews, persistent state checkpoints, robust timeline editing, and benchmarked 1080p output.
