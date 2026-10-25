"""Run planned scientific image corruptions with resumable bounded concurrency."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from rfs.providers.image_edit import edit_reference_image


EditFunction = Callable[..., dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _output_path(images_root: Path, request: dict[str, Any]) -> Path:
    return images_root / str(request["paper_id"]) / f"{request['corruption_id']}.png"


def run_corruptions(
    requests_path: str | Path,
    results_path: str | Path,
    images_dir: str | Path,
    *,
    workers: int = 2,
    retries: int = 2,
    model: str | None = None,
    limit: int | None = None,
    corruption_types: set[str] | None = None,
    edit_function: EditFunction = edit_reference_image,
) -> dict[str, Any]:
    """Generate pending corruption images and persist every completed attempt."""
    requests = _read_jsonl(Path(requests_path))
    result_file = Path(results_path)
    images_root = Path(images_dir)
    existing = {str(row["corruption_id"]): row for row in _read_jsonl(result_file)}
    pending = [
        row for row in requests
        if (not corruption_types or str(row.get("corruption_type")) in corruption_types)
        and existing.get(str(row["corruption_id"]), {}).get("status") not in {"generated_unverified", "verified"}
    ]
    if limit is not None:
        pending = pending[:max(0, int(limit))]
    lock = threading.Lock()

    def persist(result: dict[str, Any]) -> None:
        with lock:
            existing[str(result["corruption_id"])] = result
            _write_jsonl(result_file, sorted(existing.values(), key=lambda row: str(row["corruption_id"])))

    def generate(request: dict[str, Any]) -> dict[str, Any]:
        output = _output_path(images_root, request)
        output.parent.mkdir(parents=True, exist_ok=True)
        result = dict(request)
        try:
            provider = edit_function(
                request["source_image"], request["edit_instruction"], output,
                model=model, retries=retries,
            )
            result.update({
                "status": "generated_unverified",
                "output_image": str(output.resolve()),
                "output_sha256": _sha256(output),
                "provider_model": provider.get("model"),
                "provider_endpoint": provider.get("endpoint"),
                "attempts": provider.get("attempts", 1),
                "error": None,
            })
        except Exception as exc:
            output.unlink(missing_ok=True)
            result.update({"status": "failed", "output_image": None, "output_sha256": None, "attempts": retries + 1, "error": f"{type(exc).__name__}: {exc}"})
        persist(result)
        return result

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(8, int(workers)))) as executor:
        futures = [executor.submit(generate, request) for request in pending]
        for future in as_completed(futures):
            completed.append(future.result())
    all_rows = list(existing.values())
    return {
        "requested": len(requests),
        "selected_pending": len(pending),
        "completed_this_run": len(completed),
        "generated": sum(row.get("status") == "generated_unverified" for row in all_rows),
        "failed": sum(row.get("status") == "failed" for row in all_rows),
        "remaining": len(requests) - len(all_rows),
        "results_path": str(result_file.resolve()),
    }
