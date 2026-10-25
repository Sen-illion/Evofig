"""Small append-only progress reporter for long-running workflows."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[dict[str, Any]], None]


def build_progress_reporter(out: str | Path, *, echo: bool = True) -> ProgressCallback:
    """Return a callback that persists events and optionally mirrors them to stderr."""
    path = Path(out) / "progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sequence = 0

    def report(event: dict[str, Any]) -> None:
        nonlocal sequence
        sequence += 1
        record = {"seq": sequence, "elapsed_seconds": round(time.monotonic() - started, 3), **event}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if echo:
            stage = str(record.get("stage") or "workflow")
            status = str(record.get("status") or "info")
            detail = str(record.get("message") or "")
            print(f"[paper-evolve-editable] {stage}: {status}{' - ' + detail if detail else ''}", file=sys.stderr, flush=True)

    return report
