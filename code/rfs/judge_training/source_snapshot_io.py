"""Strict, stable and content-addressed file I/O for source snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from rfs.contracts.source_snapshot import snapshot_relative_path
from rfs.utils import sanitize_surrogates, write_text


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def stable_bytes(path: str | Path) -> bytes:
    source = Path(path)
    before = source.stat()
    data = source.read_bytes()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(data) != after.st_size:
        raise RuntimeError(f"Source changed while reading: {source}")
    return data


def strict_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows = []
    for number, line in enumerate(stable_bytes(source).decode("utf-8-sig").split("\n"), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {source}:{number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Non-object JSONL at {source}:{number}")
        rows.append(value)
    return rows


def streamed_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON values that are concatenated instead of split one per line.

    Frozen ``text_locations.jsonl`` files are written by appending compact JSON
    objects, so one line can hold many records and one record can span lines.
    ``json.JSONDecoder.raw_decode`` is the only reader that yields every record.
    """
    source = Path(path)
    text = stable_bytes(source).decode("utf-8-sig")
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index] in " \r\n\t":
            index += 1
        if index >= len(text):
            break
        value, index = decoder.raw_decode(text, index)
        if not isinstance(value, dict):
            raise ValueError(f"Non-object JSON record at {source}:{index}")
        records.append(value)
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(sanitize_surrogates(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def contained_path(root: str | Path, relative: str) -> Path:
    base = Path(root).resolve()
    result = base / snapshot_relative_path(relative)
    if not result.resolve().is_relative_to(base):
        raise ValueError(f"Snapshot link escapes root: {relative}")
    return result


def copy_verified(source: str | Path, target: str | Path, expected_sha256: str) -> dict:
    """Copy to a new file, detect source changes, and never overwrite content."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or sha256_file(target) != expected_sha256:
            raise ValueError(f"Existing destination has different content: {target}")
        return {"sha256": expected_sha256, "size_bytes": target.stat().st_size}
    before = source.stat()
    temporary = None
    try:
        digest = hashlib.sha256()
        with source.open("rb") as original, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            for chunk in iter(lambda: original.read(1024 * 1024), b""):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed while copying: {source}")
        if digest.hexdigest() != expected_sha256 or sha256_file(temporary) != expected_sha256:
            raise RuntimeError(f"Source/target digest mismatch: {source}")
        # Link only our fully written temporary copy, never the live source.
        # Removing the temporary name leaves an independent immutable object.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or sha256_file(target) != expected_sha256:
                raise ValueError(f"Concurrent destination content conflict: {target}")
        if sha256_file(target) != expected_sha256:
            raise RuntimeError(f"Published copy digest mismatch: {target}")
        return {"sha256": expected_sha256, "size_bytes": after.st_size}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def archive_path(digest: str) -> str:
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid archive digest")
    return f"sha256/{digest[:2]}/{digest}"
