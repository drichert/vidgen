"""Command-line interface for reproducible audio-reactive rendering."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .audio import analyze
from .comfy import ComfyClient, load_workflow, override_workflow
from .renderer import PRESETS, render


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
    video.add_argument("--preset", choices=PRESETS, default="analog-feedback")
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
    duration = None if args.duration == 0 else args.duration
    features = analyze(args.audio, fps=args.fps, duration=duration)
    if args.command == "analyze":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(features.as_json(), indent=2) + "\n")
        print(f"Wrote audio features: {args.output}")
        return
    source = args.source
    if args.comfy_workflow:
        if source:
            raise SystemExit("Choose either --source or --comfy-workflow, not both")
        workflow = override_workflow(load_workflow(args.comfy_workflow), args.comfy_set)
        source = args.keep_comfy_source or args.output.with_suffix(".source.png")
        ComfyClient(args.comfy_url, timeout=args.comfy_timeout).generate(workflow, source)
        print(f"ComfyUI source: {source}")
    elif args.comfy_set or args.keep_comfy_source:
        raise SystemExit("--comfy-set and --keep-comfy-source require --comfy-workflow")
    if args.features_output:
        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        args.features_output.write_text(json.dumps(features.as_json(), indent=2) + "\n")
    render(audio=args.audio, output=args.output, analysis=features, source=source,
           preset=args.preset, width=args.width, height=args.height, seed=args.seed, crf=args.crf)
    print(f"Rendered {args.output} ({features.duration:.2f}s, {args.fps} fps, seed={args.seed})")


if __name__ == "__main__":
    main()
