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

**Default still workflow: MOSS-Image-Base.** [workflows/moss-image-base-still.api.json](workflows/moss-image-base-still.api.json) uses ComfyUI's native MOSS-Image support (a ComfyUI build that includes it; `moss_image` must appear in the `CLIPLoader` types). Place the [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) files `moss_image_base_int8_convrot` (diffusion_models), `qwen3.5_9b_int8_convrot` (text_encoders) and `flux2-vae_small` (vae) yourself; vidgen never downloads weights. Its model card lists Apache-2.0, but verify each file's license yourself and record it:

```bash
uv run vidgen render --audio track.wav --comfy-workflow workflows/moss-image-base-still.api.json \
  --model-license moss_image_base_int8_convrot.safetensors=Apache-2.0 \
  --model-license qwen3.5_9b_int8_convrot.safetensors=Apache-2.0 \
  --model-license flux2-vae_small.safetensors=Apache-2.0 --output outputs/moss.mp4
```

Verified once on an RTX 3060 12 GB (ComfyUI 0.42.0): a 1024×576 still plus a 10 s 640×360 render took about 60 s end to end (single run, not a benchmark).

`--comfy-set NODE.FIELD=VALUE` modifies an existing input in the exported JSON. `NODE` is the workflow node ID, not a fixed or portable preset. Numbers/booleans/arrays are parsed as JSON; other strings are accepted. The client checks `/object_info`, submits `/prompt`, polls `/history/{prompt_id}`, and downloads the first image using `/view`. It does not install custom nodes, download model weights, start ComfyUI, or require GPU-based video generation.

## Audio features and modularity

```bash
uv run vidgen analyze --audio track.wav --duration 30 --fps 12 --output outputs/features.json
```

Feature curves are normalized per clip: `low` (30–180 Hz), `mid` (180–2000 Hz), `high` (2–9 kHz), RMS, positive spectral flux, and slow RMS. Persistent visual motion continues through quiet or steady passages. Presets are TOML files ([src/vidgen/presets/](src/vidgen/presets/)); `--preset` takes a built-in name or a path to your own file.

```toml
[params]                 # base values; each is clamped to fixed bounds (see PARAMS in modulation.py)
persistence = 0.973
warp = 0.6

[settings]               # static: tint (BGR), levels, saturation, blur, warp_speed, glitch_threshold
levels = 12

[[mappings]]             # applied in file order
feature = "mid"          # low | mid | high | rms | flux | slow | lfo
param = "warp"           # zoom rotation shift_x shift_y warp warp_phase_x warp_phase_y persistence
scale = 2.0              #   edge_gain glitch brightness hue_shift grain
offset = 0.0
smooth = 0.5             # 0 <= smooth < 1, exponential smoothing of the source
clip = [0.0, 1.0]        # optional, applied to the scaled value
mode = "add"             # add | multiply | blend (blend uses `amount`, 0..1)
# lfo only: rate_hz = 0.05, phase_deg = 90
```

Order per mapping: source, smooth, `value*scale+offset`, clip, combine into the parameter.

The CLI is deliberately independent of ComfyUI for procedural and existing-image renders. The initial code uses a single frame-feedback buffer; it is not diffusion-per-frame or a conventional equalizer visualizer.

## Quality, licensing, constraints

- **RTX 3060 12 GB:** ComfyUI is optional and only generates still images; the default MOSS-Image-Base workflow was exercised once on this GPU; confirm licenses and VRAM behavior for any other model. The renderer and audio analyzer run on CPU.
- **Reproducibility:** fixed RNG seed, explicit preset, FPS, image, audio, and CLI parameters. Different ffmpeg/OpenCV builds may differ slightly in encoded bits.
- **Manifest:** every render writes `<output>.manifest.json` with package/ffmpeg versions, git commit and dirty flag, the command line, SHA-256 of audio and source, the full preset content and hash, seed, video settings, the ffmpeg command, and the output hash. For ComfyUI sources it also records the prompt_id, workflow hash and content, overrides, and referenced model files. Model licenses are never inferred: they read `UNVERIFIED` unless you pass `--model-license FILE=LICENSE`, and the weights themselves are not hashed because they live on the server.
- **Cancellation:** video is written to `<name>.partial.mp4` and renamed only on success, so Ctrl-C, SIGTERM, `--max-seconds` or an ffmpeg failure never leaves a truncated file or overwrites an existing one (exit 130 on cancel).
- **ComfyUI recovery:** the prompt_id is saved to `<source>.job.json` before polling. Rerunning the same command collects or resumes that job instead of queueing a duplicate; a changed workflow, an unknown job (e.g. after a server restart) or `--comfy-resubmit` queues a new one. A timeout is an ambiguous outcome, so the error prints the prompt_id.
- **Legal:** the project itself is MIT-licensed. Transitive dependencies and ffmpeg codecs may have distinct applicable open-source licenses. No AI model weights are included.
- **Not yet included:** MIDI or real-time audio capture, visual patch graph, GPU shader acceleration, crossfades, temporal segmentation, and render resume. They are proposed follow-up milestones, not claimed features.

## Development

```bash
uv run --extra dev pytest -q
```

See [Claude implementation handoff](docs/CLAUDE_HANDOFF.md) for continuation scope and explicit review checks.
