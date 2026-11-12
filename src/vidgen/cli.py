"""Command-line interface for reproducible audio-reactive rendering."""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
from pathlib import Path

from .audio import analyze
from .comfy import ComfyClient, ComfyError, load_workflow, override_workflow
from .modulation import PresetError, load_preset
from .manifest import build_manifest, comfy_models, write_manifest
from .renderer import PRESETS, RenderTimeout, render


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vidgen", description="Audio-reactive, lo-fi generative video")
    commands = p.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check local executable dependencies")
    doctor.add_argument("--comfy-url", default=None)
    analysis = commands.add_parser("analyze", help="Write frame-aligned audio features to JSON")
    analysis.add_argument("--audio", type=Path, required=True)
    analysis.add_argument("--duration", type=float)
    analysis.add_argument("--fps", type=int, default=12)
    analysis.add_argument("--output", type=Path, required=True)
    video = commands.add_parser("render", help="Render synchronized MP4 using procedural or supplied visuals")
    video.add_argument("--audio", type=Path, required=True)
    video.add_argument("--output", type=Path, required=True)
    video.add_argument("--source", type=Path, help="Optional still image for the visual feedback buffer")
    video.add_argument("--preset", default="analog-feedback", help=f"Built-in name {PRESETS} or path to a preset .toml")
    video.add_argument("--duration", type=float, default=30.0, help="Clip duration; omit via --duration 0 for full track")
    video.add_argument("--fps", type=int, default=12)
    video.add_argument("--width", type=int, default=640)
    video.add_argument("--height", type=int, default=360)
    video.add_argument("--seed", type=int, default=42)
    video.add_argument("--crf", type=int, default=20)
    video.add_argument("--features-output", type=Path, help="Optional audio-feature JSON output")
    video.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    video.add_argument("--comfy-workflow", type=Path, help="API-format workflow, used to generate a source image")
    video.add_argument("--comfy-set", action="append", default=[], metavar="NODE.FIELD=VALUE", help="Override exported node input")
    video.add_argument("--comfy-timeout", type=float, default=600)
    video.add_argument("--keep-comfy-source", type=Path, help="Save generated source image to a chosen path")
    video.add_argument("--comfy-resubmit", action="store_true", help="Ignore the saved ComfyUI job record and queue a new prompt")
    video.add_argument("--model-license", action="append", default=[], metavar="FILE=LICENSE",
                       help="Record the license you verified for a model file referenced by the workflow")
    video.add_argument("--max-seconds", type=float, help="Abort the render (keeping no partial output) after this wall-clock time")
    video.add_argument("--manifest", type=Path, help="Manifest path (default: <output>.manifest.json)")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "doctor":
        for executable in ("ffmpeg", "ffprobe"):
            path = shutil.which(executable)
            print(f"{executable}: {path or 'MISSING'}")
        if args.comfy_url:
            print(f"ComfyUI: checking {args.comfy_url}")
            print(f"node classes: {len(ComfyClient(args.comfy_url)._get('/object_info').json())}")
        return
    if not args.audio.is_file():
        raise SystemExit(f"Audio file does not exist: {args.audio}")
    preset = None
    if args.command == "render":
        try:  # fail before analysis or any expensive ComfyUI job
            preset = load_preset(args.preset)
        except PresetError as exc:
            raise SystemExit(f"Invalid preset: {exc}")
    duration = None if args.duration == 0 else args.duration
    features = analyze(args.audio, fps=args.fps, duration=duration)
    if args.command == "analyze":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(features.as_json(), indent=2) + "\n")
        print(f"Wrote audio features: {args.output}")
        return
    source = args.source
    comfy_record = None
    if args.comfy_workflow:
        if source:
            raise SystemExit("Choose either --source or --comfy-workflow, not both")
        licenses = {}
        for item in args.model_license:
            name, sep, text = item.partition("=")
            if not sep or not name or not text:
                raise SystemExit(f"Invalid --model-license {item!r}; use FILE=LICENSE")
            licenses[name] = text
        workflow = override_workflow(load_workflow(args.comfy_workflow), args.comfy_set)
        source = args.keep_comfy_source or args.output.with_suffix(".source.png")
        job_path = source.with_name(source.name + ".job.json")
        try:
            result = ComfyClient(args.comfy_url, timeout=args.comfy_timeout).generate(
                workflow, source, job_path=job_path, resubmit=args.comfy_resubmit)
        except ComfyError as exc:
            raise SystemExit(f"ComfyUI: {exc}")
        state = "reused saved image" if result.reused_file else "resumed earlier job" if result.resumed else "submitted new job"
        print(f"ComfyUI source: {source} ({state}, prompt_id={result.prompt_id})")
        comfy_record = {
            "url": args.comfy_url, "prompt_id": result.prompt_id, "resumed": result.resumed,
            "reused_file": result.reused_file, "workflow_file": str(args.comfy_workflow),
            "workflow_sha256": result.workflow_sha256, "overrides": args.comfy_set,
            "workflow": workflow, "models": comfy_models(workflow, licenses), "job_file": str(job_path),
        }
    elif args.comfy_set or args.keep_comfy_source or args.comfy_resubmit or args.model_license:
        raise SystemExit("--comfy-set, --keep-comfy-source, --comfy-resubmit and --model-license require --comfy-workflow")
    if args.features_output:
        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        args.features_output.write_text(json.dumps(features.as_json(), indent=2) + "\n")
    try:
        info = render(audio=args.audio, output=args.output, analysis=features, source=source,
                      preset=preset, width=args.width, height=args.height, seed=args.seed, crf=args.crf,
                      max_seconds=args.max_seconds)
    except RenderTimeout as exc:
        raise SystemExit(f"Timed out: {exc}")
    manifest_path = args.manifest or args.output.with_name(args.output.name + ".manifest.json")
    write_manifest(manifest_path, build_manifest(
        audio=args.audio, output=args.output, source=source if args.comfy_workflow or args.source else None,
        preset=preset, seed=args.seed, fps=args.fps, width=args.width, height=args.height,
        duration=features.duration, crf=args.crf, ffmpeg_command=info.ffmpeg_command, frames=info.frames,
        cli_args=sys.argv[1:], comfy=comfy_record))
    print(f"Rendered {args.output} ({features.duration:.2f}s, {args.fps} fps, seed={args.seed})")
    print(f"Manifest: {manifest_path}")


def _terminate(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def run() -> None:
    signal.signal(signal.SIGTERM, _terminate)
    try:
        main()
    except KeyboardInterrupt:
        # Partial video is already removed; a recorded ComfyUI job can be resumed by rerunning.
        print("Cancelled; no partial output kept.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    run()
