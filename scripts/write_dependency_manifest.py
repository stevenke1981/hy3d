#!/usr/bin/env python3
"""Write the installed Python environment and pinned model revisions as JSON."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pathlib
import platform
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--model-revision", default="")
    return parser.parse_args()


def installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    return dict(sorted(packages.items()))


def runtime_details() -> dict[str, object]:
    details: dict[str, object] = {}
    try:
        import torch

        details["torch"] = torch.__version__
        details["cuda_available"] = torch.cuda.is_available()
        details["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            details["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        details["torch_error"] = str(exc)
    return details


def main() -> int:
    args = parse_args()
    output_path = pathlib.Path(args.out).resolve()
    temporary_path = pathlib.Path(str(output_path) + ".partial")
    manifest = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "source_revision": args.source_revision,
        "model_revision": args.model_revision,
        "packages": installed_packages(),
        "runtime": runtime_details(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(f"dependency manifest: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
