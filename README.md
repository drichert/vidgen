# vidgen

A general-purpose **audio-reactive, generative video instrument**. Any audio, any imagery, no genre assumptions. The initial engine combines offline audio analysis with deterministic CPU-based visual feedback. An optional local **ComfyUI API** can generate the source image.

**Free and open source only.** Python, NumPy, SciPy, OpenCV, FFmpeg, requests, uv, and local ComfyUI are free/open-source tools. AI checkpoints are **not bundled**; independently review model-weight licenses before use, as downloadable weights are not automatically open source.

## Quickstart

Requires Python 3.11+, `uv`, `ffmpeg`/`ffprobe` on `PATH`.

```bash
uv sync --extra dev
uv run vidgen doctor
uv run vidgen render --audio /path/to/track.wav --output outputs/test.mp4 --duration 30
```

Try alternate aesthetics without changing analysis code:

```bash
uv run vidgen render --audio track.mp3 --preset industrial-decay --duration 30 --seed 42 --output outputs/industrial.mp4
uv run vidgen render --audio track.mp3 --preset minimal-signal --duration 30 --seed 42 --output outputs/minimal.mp4
```

Or pass an existing image:

```bash
uv run vidgen render --audio track.flac --source still.png --preset analog-feedback --duration 30 --output outputs/still.mp4
```

Default: **640×360, 12 fps, 30 seconds**, H.264 MP4 with original audio muxed as AAC. `--duration 0` renders the complete track. Larger or higher-frame-rate output costs CPU time but does not require additional diffusion steps or GPU memory.

## Optional local ComfyUI generation

Create a still-image ComfyUI workflow using models/nodes you already have installed. Include `SaveImage`. Export it from ComfyUI using **File → Export Workflow (API)** (ordinary frontend workflow JSON is not equivalent). Reference that file:

```bash
uv run vidgen render --audio track.wav --comfy-workflow workflows/still-api.json \
  --comfy-set 3.seed=42 --comfy-url http://127.0.0.1:8188 \
  --keep-comfy-source outputs/seed.png --output outputs/ai-feedback.mp4
```

`--comfy-set NODE.FIELD=VALUE` modifies an existing input in the exported JSON. `NODE` is the workflow node ID, not a fixed or portable preset. Numbers/booleans/arrays are parsed as JSON; other strings are accepted. The client checks `/object_info`, submits `/prompt`, polls `/history/{prompt_id}`, and downloads the first image using `/view`. It does not install custom nodes, download model weights, start ComfyUI, or require GPU-based video generation.

## Audio features and modularity

```bash
uv run vidgen analyze --audio track.wav --duration 30 --fps 12 --output outputs/features.json
```

Feature curves are normalized per clip: `low` (30–180 Hz), `mid` (180–2000 Hz), `high` (2–9 kHz), RMS, positive spectral flux, and slow RMS. Persistent visual motion continues through quiet or steady passages. The starting renderer uses presets to map features to feedback zoom, deformation, tearing, grain, and decay. The next milestone is to make these mappings data-driven rather than hard-coded.

The CLI is deliberately independent of ComfyUI for procedural and existing-image renders. The initial code uses a single frame-feedback buffer; it is not diffusion-per-frame or a conventional equalizer visualizer.

## Quality, licensing, constraints

- **RTX 3060 12 GB:** ComfyUI is optional and only generates still images; avoid prescribing any checkpoint until its license and local VRAM behavior are confirmed. The renderer and audio analyzer run on CPU.
- **Reproducibility:** fixed RNG seed, explicit preset, FPS, image, audio, and CLI parameters. Different ffmpeg/OpenCV builds may differ slightly in encoded bits.
- **Legal:** the project itself is MIT-licensed. Transitive dependencies and ffmpeg codecs may have distinct applicable open-source licenses. No AI model weights are included.
- **Not yet included:** MIDI or real-time audio capture, visual patch graph, GPU shader acceleration, crossfades, temporal segmentation, and render resume. They are proposed follow-up milestones, not claimed features.

## Development

```bash
uv run --extra dev pytest -q
```

See [Claude implementation handoff](docs/CLAUDE_HANDOFF.md) for continuation scope and explicit review checks.
