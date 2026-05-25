#!/usr/bin/env python3
"""Convert Hunyuan3D shape checkpoints to a GGUF container.

This converter intentionally writes a Hunyuan3D-specific GGUF file without
depending on the Python ``gguf`` package. It stores PyTorch tensor names and
shapes so the native C++ runtime can define its own mapping in the next phase.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import struct
import sys
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

GGUF_VALUE_UINT8 = 0
GGUF_VALUE_INT8 = 1
GGUF_VALUE_UINT16 = 2
GGUF_VALUE_INT16 = 3
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_INT32 = 5
GGUF_VALUE_FLOAT32 = 6
GGUF_VALUE_BOOL = 7
GGUF_VALUE_STRING = 8
GGUF_VALUE_ARRAY = 9
GGUF_VALUE_UINT64 = 10
GGUF_VALUE_INT64 = 11
GGUF_VALUE_FLOAT64 = 12

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


@dataclasses.dataclass(frozen=True)
class TensorRecord:
    name: str
    array: Any


@dataclasses.dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    data_offset: int
    nbytes: int


def align_offset(offset: int, alignment: int) -> int:
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + (alignment - remainder)


def write_padding(handle, alignment: int) -> None:
    pos = handle.tell()
    aligned = align_offset(pos, alignment)
    if aligned > pos:
        handle.write(b"\0" * (aligned - pos))


def encode_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def write_string(handle, value: str) -> None:
    handle.write(encode_string(value))


def metadata_type_and_payload(value: Any) -> tuple[int, bytes]:
    if isinstance(value, bool):
        return GGUF_VALUE_BOOL, struct.pack("<?", value)
    if isinstance(value, int):
        if value >= 0:
            return GGUF_VALUE_UINT64, struct.pack("<Q", value)
        return GGUF_VALUE_INT64, struct.pack("<q", value)
    if isinstance(value, float):
        return GGUF_VALUE_FLOAT64, struct.pack("<d", value)
    if isinstance(value, str):
        return GGUF_VALUE_STRING, encode_string(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return GGUF_VALUE_ARRAY, struct.pack("<IQ", GGUF_VALUE_STRING, 0)
        child_type, first_payload = metadata_type_and_payload(value[0])
        if child_type == GGUF_VALUE_ARRAY:
            raise ValueError("nested metadata arrays are not supported")
        payload = bytearray(struct.pack("<IQ", child_type, len(value)))
        payload.extend(first_payload)
        for item in value[1:]:
            item_type, item_payload = metadata_type_and_payload(item)
            if item_type != child_type:
                raise ValueError("metadata arrays must contain one value type")
            payload.extend(item_payload)
        return GGUF_VALUE_ARRAY, bytes(payload)
    raise TypeError(f"unsupported metadata value type for {value!r}")


def dtype_name(value: Any) -> str:
    return str(value.dtype).replace("torch.", "")


def tensor_type_for_value(value: Any, outtype: str | None = None) -> int:
    if outtype == "f16":
        return GGML_TYPE_F16
    if outtype == "f32":
        return GGML_TYPE_F32

    dtype = dtype_name(value)
    if dtype == "float16":
        return GGML_TYPE_F16
    if dtype == "float32":
        return GGML_TYPE_F32
    raise TypeError(f"unsupported tensor dtype: {dtype}")


def shape_for_value(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def nbytes_for_value(value: Any, outtype: str | None = None) -> int:
    shape = shape_for_value(value)
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if outtype == "f16":
        return count * 2
    if outtype == "f32":
        return count * 4
    if hasattr(value, "element_size"):
        return count * int(value.element_size())
    return int(np.ascontiguousarray(value).nbytes)


def normalize_array(array: np.ndarray, outtype: str) -> np.ndarray:
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"unsupported non-floating tensor dtype: {array.dtype}")
    if outtype == "f16":
        return np.ascontiguousarray(array.astype(np.float16, copy=False))
    if outtype == "f32":
        return np.ascontiguousarray(array.astype(np.float32, copy=False))
    raise ValueError(f"unsupported outtype: {outtype}")


def map_tensor_name(name: str) -> str:
    prefixes = (
        "state_dict.model.",
        "state_dict.module.",
        "state_dict.",
        "module.model.",
        "module.",
        "model.",
    )
    mapped = name
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if mapped.startswith(prefix):
                mapped = mapped[len(prefix) :]
                changed = True
    return mapped


def build_tensor_infos(tensors: list[TensorRecord], alignment: int, outtype: str | None = None) -> list[TensorInfo]:
    infos: list[TensorInfo] = []
    offset = 0
    for tensor in tensors:
        offset = align_offset(offset, alignment)
        infos.append(
            TensorInfo(
                name=tensor.name,
                shape=shape_for_value(tensor.array),
                ggml_type=tensor_type_for_value(tensor.array, outtype),
                data_offset=offset,
                nbytes=nbytes_for_value(tensor.array, outtype),
            )
        )
        offset += infos[-1].nbytes
    return infos


def write_gguf(
    output_path: os.PathLike[str] | str,
    tensors: Iterable[TensorRecord],
    metadata: Mapping[str, Any],
    alignment: int = DEFAULT_ALIGNMENT,
) -> None:
    tensor_list = [TensorRecord(t.name, np.ascontiguousarray(t.array)) for t in tensors]
    tensor_infos = build_tensor_infos(tensor_list, alignment)

    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<IQQ", GGUF_VERSION, len(tensor_infos), len(metadata)))

        for key, value in metadata.items():
            write_string(handle, key)
            value_type, payload = metadata_type_and_payload(value)
            handle.write(struct.pack("<I", value_type))
            handle.write(payload)

        for info in tensor_infos:
            write_string(handle, info.name)
            handle.write(struct.pack("<I", len(info.shape)))
            for dim in info.shape:
                handle.write(struct.pack("<Q", dim))
            handle.write(struct.pack("<IQ", info.ggml_type, info.data_offset))

        write_padding(handle, alignment)

        for tensor, _info in zip(tensor_list, tensor_infos, strict=True):
            write_padding(handle, alignment)
            handle.write(np.ascontiguousarray(tensor.array).tobytes(order="C"))


def write_checkpoint_gguf(
    output_path: os.PathLike[str] | str,
    tensors: Iterable[TensorRecord],
    metadata: Mapping[str, Any],
    outtype: str,
    alignment: int = DEFAULT_ALIGNMENT,
) -> None:
    tensor_list = list(tensors)
    tensor_infos = build_tensor_infos(tensor_list, alignment, outtype)

    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<IQQ", GGUF_VERSION, len(tensor_infos), len(metadata)))

        for key, value in metadata.items():
            write_string(handle, key)
            value_type, payload = metadata_type_and_payload(value)
            handle.write(struct.pack("<I", value_type))
            handle.write(payload)

        for info in tensor_infos:
            write_string(handle, info.name)
            handle.write(struct.pack("<I", len(info.shape)))
            for dim in info.shape:
                handle.write(struct.pack("<Q", dim))
            handle.write(struct.pack("<IQ", info.ggml_type, info.data_offset))

        write_padding(handle, alignment)

        for index, tensor in enumerate(tensor_list, start=1):
            write_padding(handle, alignment)
            array = tensor_to_numpy(tensor.array, outtype)
            handle.write(array.tobytes(order="C"))
            if index == 1 or index % 100 == 0 or index == len(tensor_list):
                print(f"wrote tensor {index}/{len(tensor_list)}: {tensor.name}")


def import_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required to read .ckpt files. Install it with: "
            "uv pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch


def is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy")


def select_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "module"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint
    raise TypeError("checkpoint does not contain a recognizable tensor state dict")


def tensor_to_numpy(value: Any, outtype: str) -> np.ndarray:
    if not is_torch_tensor(value):
        raise TypeError("value is not a PyTorch tensor")
    tensor = value.detach().cpu()
    if tensor.layout != import_torch().strided:
        tensor = tensor.to_dense()
    return normalize_array(tensor.contiguous().numpy(), outtype)


def iter_checkpoint_tensors(checkpoint_path: pathlib.Path, outtype: str) -> Iterable[TensorRecord]:
    torch = import_torch()
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, mmap=True, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, weights_only=False, **load_kwargs)
    state_dict = select_state_dict(checkpoint)

    seen: set[str] = set()
    for raw_name in sorted(state_dict.keys()):
        value = state_dict[raw_name]
        if not is_torch_tensor(value):
            continue
        if not value.is_floating_point():
            continue
        name = map_tensor_name(raw_name)
        if name in seen:
            raise ValueError(f"duplicate mapped tensor name: {name}")
        seen.add(name)
        yield TensorRecord(name=name, array=value.detach().cpu())


def build_metadata(args: argparse.Namespace, tensor_count: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "general.architecture": "hunyuan3d",
        "general.name": args.name,
        "general.alignment": args.alignment,
        "hy3d.version": args.hy3d_version,
        "hy3d.model_part": args.model_part,
        "hy3d.source_format": "pytorch_checkpoint",
        "hy3d.source_checkpoint": pathlib.Path(args.input).name,
        "hy3d.tensor_layout": "pytorch",
        "hy3d.tensor_count": tensor_count,
        "hy3d.outtype": args.outtype,
        "hy3d.converter": "convert-hy3d-shape-to-gguf.py",
    }
    if args.config:
        metadata["hy3d.config_path"] = str(pathlib.Path(args.config))
    return metadata


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Hunyuan3D shape .ckpt checkpoints to GGUF.")
    parser.add_argument("--input", required=True, help="Path to Hunyuan3D .ckpt checkpoint.")
    parser.add_argument("--output", required=True, help="Path to output .gguf.")
    parser.add_argument("--config", default="", help="Optional config.yaml path to record in metadata.")
    parser.add_argument("--outtype", choices=("f16", "f32"), default="f16", help="Output tensor dtype.")
    parser.add_argument("--model-part", choices=("shape", "vae"), default="shape", help="Hunyuan3D model part label.")
    parser.add_argument("--hy3d-version", default="2.1", help="Hunyuan3D version metadata.")
    parser.add_argument("--name", default="Hunyuan3D-2.1 Shape", help="GGUF model name metadata.")
    parser.add_argument("--alignment", type=int, default=DEFAULT_ALIGNMENT, help="GGUF tensor data alignment.")
    parser.add_argument("--dry-run", action="store_true", help="Load checkpoint and print tensor summary without writing.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    if not input_path.exists():
        print(f"error: input checkpoint not found: {input_path}", file=sys.stderr)
        return 2

    print(f"loading checkpoint: {input_path}")
    tensors = list(iter_checkpoint_tensors(input_path, args.outtype))
    if not tensors:
        print("error: checkpoint did not contain floating point tensors", file=sys.stderr)
        return 3

    total_bytes = sum(nbytes_for_value(t.array, args.outtype) for t in tensors)
    print(f"tensors: {len(tensors)}")
    print(f"tensor_bytes: {total_bytes}")
    print(f"outtype: {args.outtype}")

    if args.dry_run:
        for tensor in tensors[:20]:
            print(f"{tensor.name}: shape={tuple(tensor.array.shape)} dtype={tensor.array.dtype}")
        if len(tensors) > 20:
            print(f"... {len(tensors) - 20} more tensors")
        return 0

    metadata = build_metadata(args, len(tensors))
    print(f"writing GGUF: {output_path}")
    write_checkpoint_gguf(output_path, tensors, metadata, args.outtype, alignment=args.alignment)
    print(f"done: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
