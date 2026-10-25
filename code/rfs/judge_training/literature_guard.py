"""Process-safe coordination between canonical collection and source capture."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Iterator


class LiteratureBusyError(RuntimeError):
    """Another writer owns the batch or an older unguarded worker is active."""


@contextmanager
def literature_batch_guard(batch_root: str | Path, *, purpose: str) -> Iterator[None]:
    """Hold an OS advisory lock; crashed processes release it automatically."""
    lock = Path(batch_root).resolve() / "manifests" / "source_capture.guard"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise LiteratureBusyError(f"Batch writer/capture active: {lock}") from exc
        try:
            stream.seek(1)
            stream.truncate()
            stream.write(json.dumps({"pid": os.getpid(), "purpose": purpose, "started_at": time.time()}).encode())
            stream.flush()
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _python_processes() -> list[dict]:
    if os.name == "nt":
        command = (
            "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
            "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|rfs' } "
            "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, timeout=30, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        raw = result.stdout.decode("utf-8-sig", errors="replace").strip()
        values = json.loads(raw) if raw else []
        return [values] if isinstance(values, dict) else values
    values = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                values.append({"ProcessId": int(entry.name), "CommandLine": command})
            except (OSError, ProcessLookupError):
                continue
    return values


def active_legacy_writers(batch_root: str | Path) -> list[dict]:
    """Find workers launched before lock support, without exposing arguments."""
    batch = Path(batch_root).resolve().as_posix().casefold()
    markers = (
        "collect-papers", "probe_fresh_arxiv_sources.py", "extract_fresh_fulltext.py",
        "freeze_incremental_10k.py", "run_incremental_10k.py", "run_independent_10k.py",
        "verify_fresh_official_landings.py", "freeze_fresh_verified.py",
    )
    active = []
    for item in _python_processes():
        pid = int(item.get("ProcessId") or 0)
        command = str(item.get("CommandLine") or "").replace("\\", "/").casefold()
        marker = next((value for value in markers if value in command), None)
        same_collection = batch in command or Path(batch).parent.as_posix() in command or Path(batch).name in command
        if pid != os.getpid() and marker and (same_collection or "--batch-root" not in command):
            active.append({"pid": pid, "worker": marker})
    return active


def guarded_collection(collector, batch_root: str | Path, **kwargs) -> dict:
    """Enter the coordination protocol without changing the collector itself."""
    try:
        with literature_batch_guard(batch_root, purpose="collect-papers"):
            if (Path(batch_root) / "manifests/source_binding_repair_pending.json").exists():
                return {"ok": False, "status": "waiting_for_binding_repair_recovery", "retryable": True}
            return collector(batch_root, **kwargs)
    except LiteratureBusyError as exc:
        return {"ok": False, "status": "waiting_for_batch", "retryable": True, "error": str(exc)}
