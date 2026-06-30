#!/usr/bin/env python3
"""Apply the pinned Hunyuan3D-Paint Windows compiler compatibility patch."""

import argparse
import pathlib
import re
import subprocess
import sys


def git_head(source_root: pathlib.Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"source root is not a git checkout: {source_root}")
    return result.stdout.strip()


def update_file(path: pathlib.Path, transform) -> bool:
    if not path.is_file():
        raise RuntimeError(f"required upstream file is missing: {path}")
    original = path.read_text(encoding="utf-8")
    updated = transform(original)
    if updated == original:
        return False
    temporary = path.with_name(path.name + ".hy3d-patch.tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(path)
    return True


def patch_grid_neighbor(text: str) -> str:
    text = re.sub(
        r"(?<!static_cast<int64_t>\()(\b(?:seq2pos\.size\(\) / 3|seq2feat\.size\(\) / feat_channel))",
        r"static_cast<int64_t>(\1)",
        text,
    )
    text = re.sub(
        r"(?<!static_cast<int64_t>\()(\bgrids\[i\]\.(?:seq2grid|seq2evencorner|seq2oddcorner|downsample_seq)\.size\(\))",
        r"static_cast<int64_t>(\1)",
        text,
    )
    text = re.sub(
        r"(?<!static_cast<int64_t>\()feat_channel(?=}, float_options)",
        r"static_cast<int64_t>(feat_channel)",
        text,
    )
    text = re.sub(r"\blong\* ([nd]ptr)\b", r"int64_t* \1", text)
    text = text.replace("data_ptr<long>()", "data_ptr<int64_t>()")
    return text


def patch_rasterizer(text: str) -> str:
    return text.replace(
        "(long)maxint", "static_cast<int64_t>(maxint)"
    ).replace(
        "(INT64*)z_min.data_ptr<long>()",
        "reinterpret_cast<INT64*>(z_min.data_ptr<int64_t>())",
    )


def patch_setup(text: str) -> str:
    if "-allow-unsupported-compiler" in text:
        return text
    marker = '        "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",\n    ],\n)'
    replacement = (
        '        "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",\n'
        '    ],\n'
        '    extra_compile_args={"nvcc": ["-allow-unsupported-compiler"]},\n'
        ')'
    )
    if marker not in text:
        raise RuntimeError("upstream custom-rasterizer setup.py layout changed")
    return text.replace(marker, replacement, 1)


def apply_patch(source_root: pathlib.Path, expected_revision: str) -> int:
    actual_revision = git_head(source_root)
    if actual_revision.lower() != expected_revision.lower():
        raise RuntimeError(
            f"source revision mismatch: expected {expected_revision}, got {actual_revision}"
        )

    rasterizer_root = source_root / "hy3dpaint" / "custom_rasterizer"
    kernel_root = rasterizer_root / "lib" / "custom_rasterizer_kernel"
    changed = 0
    changed += update_file(kernel_root / "grid_neighbor.cpp", patch_grid_neighbor)
    changed += update_file(kernel_root / "rasterizer.cpp", patch_rasterizer)
    changed += update_file(kernel_root / "rasterizer_gpu.cu", patch_rasterizer)
    changed += update_file(rasterizer_root / "setup.py", patch_setup)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    try:
        changed = apply_patch(args.source_root.resolve(), args.expected_revision)
    except (OSError, RuntimeError) as error:
        print(f"Windows rasterizer patch failed: {error}", file=sys.stderr)
        return 1
    print(f"Windows rasterizer patch ready: {changed} file(s) updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
