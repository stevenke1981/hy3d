#!/usr/bin/env python3
"""Shape-only Hunyuan3D generation backend for hy3d.exe.

This script wraps the official Tencent Hunyuan3D Python pipeline so the C++
CLI can produce a usable GLB while the native C++ runtime is still being built.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a shape GLB with the official Hunyuan3D pipeline.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--out", required=True, help="Output mesh path, usually .glb.")
    parser.add_argument("--model-path", default="", help="Local model repo path. Defaults to models/Hunyuan3D-2.1.")
    parser.add_argument("--source-root", default="", help="Official Hunyuan3D source checkout.")
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-1", help="Shape model subfolder.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Inference device.")
    parser.add_argument("--quality", default="normal", choices=("smoke", "draft", "normal"), help="Step preset.")
    parser.add_argument("--steps", type=int, default=None, help="Diffusion inference steps. Overrides --quality.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--low-vram", action="store_true", help="Enable model CPU offload when available.")
    parser.add_argument("--no-rembg", action="store_true", help="Do not run background removal for RGB images.")
    parser.add_argument("--dry-run", action="store_true", help="Validate imports and paths without loading model weights.")
    parser.add_argument("--log", default="", help="Log path. Defaults to <out>.log.txt.")
    parser.add_argument("--metadata", default="", help="Metadata JSON path. Defaults to <out>.json.")
    return parser.parse_args()


def steps_for_quality(quality: str) -> int:
    return {"smoke": 5, "draft": 10, "normal": 30}[quality]


def resolve_paths(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    root = pathlib.Path(__file__).resolve().parents[1]
    source_root = pathlib.Path(args.source_root) if args.source_root else root / "third_party" / "Hunyuan3D-2.1"
    model_path = pathlib.Path(args.model_path) if args.model_path else root / "models" / "Hunyuan3D-2.1"
    return source_root.resolve(), model_path.resolve()


def add_source_paths(source_root: pathlib.Path) -> None:
    hy3dshape_root = source_root / "hy3dshape"
    hy3dpaint_root = source_root / "hy3dpaint"
    sys.path.insert(0, str(hy3dshape_root))
    if hy3dpaint_root.exists():
        sys.path.insert(0, str(hy3dpaint_root))


class Tee:
    def __init__(self, stream, log_file) -> None:
        self.stream = stream
        self.log_file = log_file

    def write(self, data: str) -> int:
        self.stream.write(data)
        written = self.log_file.write(data)
        self.flush()
        return written

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()


def sidecar_path(output_path: pathlib.Path, suffix: str) -> pathlib.Path:
    return pathlib.Path(str(output_path) + suffix)


def write_metadata(path: pathlib.Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    image_path = pathlib.Path(args.image).resolve()
    output_path = pathlib.Path(args.out).resolve()
    source_root, model_path = resolve_paths(args)
    steps = args.steps if args.steps is not None else steps_for_quality(args.quality)
    log_path = pathlib.Path(args.log).resolve() if args.log else sidecar_path(output_path, ".log.txt")
    metadata_path = pathlib.Path(args.metadata).resolve() if args.metadata else sidecar_path(output_path, ".json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = Tee(stdout, log_file)
    sys.stderr = Tee(stderr, log_file)

    metadata = {
        "command": "generate",
        "status": "started",
        "python": sys.executable,
        "image": str(image_path),
        "out": str(output_path),
        "source_root": str(source_root),
        "model_path": str(model_path),
        "subfolder": args.subfolder,
        "device": args.device,
        "quality": args.quality,
        "steps": steps,
        "seed": args.seed,
        "low_vram": args.low_vram,
        "no_rembg": args.no_rembg,
        "dry_run": args.dry_run,
        "log": str(log_path),
    }

    def finish(status: str, code: int, error: str = "") -> int:
        elapsed = time.perf_counter() - started
        metadata["status"] = status
        metadata["exit_code"] = code
        metadata["elapsed_seconds"] = round(elapsed, 3)
        if error:
            metadata["error"] = error
        if output_path.exists():
            metadata["output_size"] = output_path.stat().st_size
        write_metadata(metadata_path, metadata)
        print(f"metadata: {metadata_path}")
        return code

    try:
        if not image_path.exists():
            message = f"image not found: {image_path}"
            print(f"error: {message}", file=sys.stderr)
            return finish("error", 2, message)
        if not source_root.exists():
            message = f"Hunyuan3D source checkout not found: {source_root}"
            print(f"error: {message}", file=sys.stderr)
            return finish("error", 3, message)
        if not (model_path / args.subfolder / "model.fp16.ckpt").exists():
            message = f"shape checkpoint not found under: {model_path / args.subfolder}"
            print(f"error: {message}", file=sys.stderr)
            return finish("error", 4, message)

        add_source_paths(source_root)
        os.environ.setdefault("HY3DGEN_MODELS", str(model_path.parent))

        try:
            import torch
            from PIL import Image
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        except Exception as exc:
            message = f"failed to import Hunyuan3D dependencies: {exc}"
            print(f"error: {message}", file=sys.stderr)
            return finish("error", 5, message)

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            metadata["cuda_device"] = torch.cuda.get_device_name(0)

        print(f"python: {sys.executable}")
        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device: {torch.cuda.get_device_name(0)}")
        print(f"source_root: {source_root}")
        print(f"model_path: {model_path}")
        print(f"image: {image_path}")
        print(f"out: {output_path}")
        print(f"quality: {args.quality}")
        print(f"steps: {steps}")
        print(f"log: {log_path}")

        if args.dry_run:
            print("dry_run: ok")
            return finish("dry_run", 0)

        if args.device == "cuda" and not torch.cuda.is_available():
            message = "--device cuda requested but torch.cuda.is_available() is false"
            print(f"error: {message}", file=sys.stderr)
            return finish("error", 6, message)

        dtype = torch.float16 if args.device == "cuda" else torch.float32
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(model_path),
            subfolder=args.subfolder,
            device=args.device,
            dtype=dtype,
            use_safetensors=False,
            variant="fp16",
        )
        if args.low_vram:
            if not hasattr(pipeline, "components"):
                pipeline.components = {
                    "conditioner": pipeline.conditioner,
                    "model": pipeline.model,
                    "vae": pipeline.vae,
                }
            pipeline.enable_model_cpu_offload(device=args.device)

        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        image_arg: str | Image.Image
        if args.no_rembg:
            image_arg = str(image_path)
        else:
            image = Image.open(image_path).convert("RGBA")
            image_arg = image

        with torch.inference_mode():
            mesh = pipeline(image=image_arg, num_inference_steps=steps)[0]
        mesh.export(str(output_path))

        elapsed = time.perf_counter() - started
        print(f"done: {output_path}")
        print(f"elapsed_seconds: {elapsed:.2f}")
        return finish("ok", 0)
    finally:
        sys.stdout = stdout
        sys.stderr = stderr
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
