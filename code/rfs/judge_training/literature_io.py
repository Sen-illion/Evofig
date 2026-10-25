"""Atomic checkpoint I/O for resumable literature collection."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Replace one UTF-8 checkpoint atomically, including on Windows."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        ) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
    return target


__all__ = ["atomic_write_text"]
