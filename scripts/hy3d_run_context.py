"""Shared logging and metadata lifecycle for Hunyuan3D Python backends."""

from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any, TextIO


class Tee:
    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
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


def write_metadata(path: pathlib.Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(str(path) + ".partial")
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class RunContext:
    def __init__(
        self,
        output_path: pathlib.Path,
        log_path: pathlib.Path,
        metadata_path: pathlib.Path,
        metadata: dict[str, Any],
    ) -> None:
        self.output_path = output_path
        self.log_path = log_path
        self.metadata_path = metadata_path
        self.metadata = metadata
        self.started = time.perf_counter()
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None
        self._log_file: TextIO | None = None

    def __enter__(self) -> "RunContext":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = Tee(self._stdout, self._log_file)
        sys.stderr = Tee(self._stderr, self._log_file)
        return self

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started

    def finish(self, status: str, code: int, error: str = "") -> int:
        self.metadata["status"] = status
        self.metadata["exit_code"] = code
        self.metadata["elapsed_seconds"] = round(self.elapsed_seconds(), 3)
        if error:
            self.metadata["error"] = error
        if status == "ok" and self.output_path.exists():
            self.metadata["output_size"] = self.output_path.stat().st_size
        write_metadata(self.metadata_path, self.metadata)
        print(f"metadata: {self.metadata_path}")
        return code

    def close(self) -> None:
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        if self._log_file is not None:
            self._log_file.close()
        self._stdout = None
        self._stderr = None
        self._log_file = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
