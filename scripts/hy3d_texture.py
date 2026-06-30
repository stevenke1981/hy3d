#!/usr/bin/env python3
"""PBR texture backend for hy3d.exe.

This script wraps the official Hunyuan3D-Paint pipeline. The paint model is
heavy; dry-run mode verifies paths and imports without loading diffusion
weights.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import types

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hy3d_run_context import RunContext, partial_output_path, sidecar_path

REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PBR texture for a mesh with Hunyuan3D-Paint.")
    parser.add_argument("--mesh", required=True, help="Input mesh path, usually a shape GLB.")
    parser.add_argument("--image", required=True, help="Reference image path.")
    parser.add_argument("--out", required=True, help="Output textured GLB path.")
    parser.add_argument("--model-path", default="", help="Local model repo path. Defaults to models/Hunyuan3D-2.1.")
    parser.add_argument("--source-root", default="", help="Official Hunyuan3D source checkout.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Inference device.")
    parser.add_argument("--max-views", type=int, default=6, help="Selected views, between 6 and 12.")
    parser.add_argument("--resolution", type=int, default=512, choices=(512, 768), help="Paint resolution.")
    parser.add_argument("--no-remesh", action="store_true", help="Skip the official remesh step.")
    parser.add_argument("--dry-run", action="store_true", help="Validate imports and paths without loading model weights.")
    parser.add_argument("--log", default="", help="Log path. Defaults to <out>.log.txt.")
    parser.add_argument("--metadata", default="", help="Metadata JSON path. Defaults to <out>.json.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    root = pathlib.Path(__file__).resolve().parents[1]
    source_root = pathlib.Path(args.source_root) if args.source_root else root / "third_party" / "Hunyuan3D-2.1"
    model_path = pathlib.Path(args.model_path) if args.model_path else root / "models" / "Hunyuan3D-2.1"
    return source_root.resolve(), model_path.resolve()


def add_source_paths(source_root: pathlib.Path) -> None:
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "hy3dpaint"))


def patch_snapshot_download(model_path: pathlib.Path) -> None:
    import huggingface_hub

    real_snapshot_download = huggingface_hub.snapshot_download

    def snapshot_download(repo_id: str, *args, **kwargs):
        candidate = pathlib.Path(repo_id)
        if candidate.exists():
            return str(candidate.resolve())
        if repo_id == "tencent/Hunyuan3D-2.1" and (model_path / "hunyuan3d-paintpbr-v2-1").exists():
            return str(model_path.resolve())
        return real_snapshot_download(repo_id, *args, **kwargs)

    huggingface_hub.snapshot_download = snapshot_download


def install_bpy_stub_if_needed(metadata: dict) -> None:
    try:
        import bpy  # noqa: F401

        metadata["bpy"] = "available"
    except Exception:
        sys.modules["bpy"] = types.ModuleType("bpy")
        metadata["bpy"] = "stubbed"


def add_windows_dll_dirs(metadata: dict) -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    added: list[str] = []
    candidates = [
        pathlib.Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            os.add_dll_directory(str(candidate))
            added.append(str(candidate))
    metadata["dll_dirs"] = added


def validate_preflight(
    args: argparse.Namespace,
    mesh_path: pathlib.Path,
    image_path: pathlib.Path,
    source_root: pathlib.Path,
    model_path: pathlib.Path,
) -> tuple[int, str] | None:
    if args.max_views < 6 or args.max_views > 12:
        return 2, "--max-views must be between 6 and 12"
    if not mesh_path.exists():
        return 3, f"mesh not found: {mesh_path}"
    if not image_path.exists():
        return 4, f"image not found: {image_path}"
    if not source_root.exists():
        return 5, f"Hunyuan3D source checkout not found: {source_root}"
    paint_model = model_path / "hunyuan3d-paintpbr-v2-1"
    if not paint_model.exists():
        return 6, f"paint model not found: {paint_model}"
    realesrgan_ckpt = source_root / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    if not realesrgan_ckpt.exists():
        return 7, f"RealESRGAN checkpoint not found: {realesrgan_ckpt}; download from {REALESRGAN_URL}"
    return None


def load_dependencies():
    import torch
    import trimesh
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    return torch, trimesh, Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline


def create_pipeline(
    args: argparse.Namespace,
    source_root: pathlib.Path,
    model_path: pathlib.Path,
    realesrgan_ckpt: pathlib.Path,
    config_class,
    pipeline_class,
):
    config = config_class(args.max_views, args.resolution)
    config.device = args.device
    config.multiview_pretrained_path = str(model_path)
    config.realesrgan_ckpt_path = str(realesrgan_ckpt)
    config.multiview_cfg_path = str(source_root / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
    config.custom_pipeline = str(source_root / "hy3dpaint" / "hunyuanpaintpbr")
    return pipeline_class(config)


def run_inference(
    args: argparse.Namespace,
    mesh_path: pathlib.Path,
    image_path: pathlib.Path,
    output_path: pathlib.Path,
    partial_output_path: pathlib.Path,
    trimesh,
    pipeline,
) -> tuple[int, str] | None:
    output_obj = output_path.with_suffix(".obj")
    produced_obj = pathlib.Path(
        pipeline(
            mesh_path=str(mesh_path),
            image_path=str(image_path),
            output_mesh_path=str(output_obj),
            use_remesh=not args.no_remesh,
            save_glb=False,
        )
    )
    if not produced_obj.exists():
        return 10, f"texture pipeline did not produce expected OBJ: {produced_obj}"
    scene = trimesh.load(str(produced_obj), force="scene")
    partial_output_path.unlink(missing_ok=True)
    scene.export(str(partial_output_path))
    if not partial_output_path.exists() or partial_output_path.stat().st_size == 0:
        return 11, f"failed to export textured GLB: {partial_output_path}"
    partial_output_path.replace(output_path)
    return None


def main() -> int:
    args = parse_args()

    mesh_path = pathlib.Path(args.mesh).resolve()
    image_path = pathlib.Path(args.image).resolve()
    output_path = pathlib.Path(args.out).resolve()
    source_root, model_path = resolve_paths(args)
    log_path = pathlib.Path(args.log).resolve() if args.log else sidecar_path(output_path, ".log.txt")
    metadata_path = pathlib.Path(args.metadata).resolve() if args.metadata else sidecar_path(output_path, ".json")
    partial_path = partial_output_path(output_path)

    metadata = {
        "command": "texture",
        "status": "started",
        "python": sys.executable,
        "mesh": str(mesh_path),
        "image": str(image_path),
        "out": str(output_path),
        "source_root": str(source_root),
        "model_path": str(model_path),
        "device": args.device,
        "max_views": args.max_views,
        "resolution": args.resolution,
        "no_remesh": args.no_remesh,
        "dry_run": args.dry_run,
        "log": str(log_path),
    }

    run = RunContext(output_path, log_path, metadata_path, metadata)
    run.__enter__()
    finish = run.finish

    try:
        preflight = validate_preflight(args, mesh_path, image_path, source_root, model_path)
        if preflight is not None:
            code, message = preflight
            print(f"error: {message}", file=sys.stderr)
            return finish("error", code, message)
        realesrgan_ckpt = source_root / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"

        add_source_paths(source_root)
        os.environ.setdefault("HY3DGEN_MODELS", str(model_path.parent))
        patch_snapshot_download(model_path)
        install_bpy_stub_if_needed(metadata)
        add_windows_dll_dirs(metadata)
        try:
            from torchvision_fix import apply_fix

            metadata["torchvision_fix"] = bool(apply_fix())
        except Exception as exc:
            metadata["torchvision_fix"] = f"failed: {exc}"

        try:
            torch, trimesh, config_class, pipeline_class = load_dependencies()
        except Exception as exc:
            message = f"failed to import Hunyuan3D-Paint dependencies: {exc}"
            print(f"error: {message}", file=sys.stderr)
            return finish("error", 8, message)

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
        print(f"mesh: {mesh_path}")
        print(f"image: {image_path}")
        print(f"out: {output_path}")
        print(f"log: {log_path}")

        if args.dry_run:
            print("dry_run: ok")
            return finish("dry_run", 0)

        if args.device == "cuda" and not torch.cuda.is_available():
            message = "--device cuda requested but torch.cuda.is_available() is false"
            print(f"error: {message}", file=sys.stderr)
            return finish("error", 9, message)

        pipeline = create_pipeline(
            args,
            source_root,
            model_path,
            realesrgan_ckpt,
            config_class,
            pipeline_class,
        )
        inference_error = run_inference(
            args,
            mesh_path,
            image_path,
            output_path,
            partial_path,
            trimesh,
            pipeline,
        )
        if inference_error is not None:
            code, message = inference_error
            print(f"error: {message}", file=sys.stderr)
            return finish("error", code, message)

        elapsed = run.elapsed_seconds()
        print(f"done: {output_path}")
        print(f"elapsed_seconds: {elapsed:.2f}")
        return finish("ok", 0)
    except Exception as exc:
        partial_path.unlink(missing_ok=True)
        message = f"unhandled texture error: {exc}"
        print(f"error: {message}", file=sys.stderr)
        return finish("error", 99, message)
    finally:
        run.close()


if __name__ == "__main__":
    raise SystemExit(main())
