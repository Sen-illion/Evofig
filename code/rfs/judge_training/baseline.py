"""Run a resumable Bootstrap VLM baseline over Judge JSONL data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from rfs.providers.client import call_vlm_json, resolve_vlm_model

from .prompts import PROMPT_VERSION, build_prompt
from .evaluator import evaluate_predictions


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _image_paths(row: dict[str, Any]) -> list[str]:
    return [str(row[key]) for key in ("candidate_a", "candidate_b") if row.get(key)]


def run_baseline(dataset: str | Path, out: str | Path, *, limit: int | None = None, model: str | None = None, timeout: int = 180, retries: int = 2, adapter: Callable[..., dict[str, Any]] = call_vlm_json) -> dict[str, Any]:
    source = Path(dataset); source = source / "validation.jsonl" if source.is_dir() else source
    root = Path(out); root.mkdir(parents=True, exist_ok=True)
    prediction_path, failure_path = root / "predictions.jsonl", root / "failures.jsonl"
    existing = {}
    if prediction_path.exists():
        for row in _rows(prediction_path):
            if row.get("example_id"): existing[str(row["example_id"])] = row
    rows = _rows(source)
    if limit is not None:
        wanted = max(0, int(limit)); grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows: grouped.setdefault(str(row.get("task_type")), []).append(row)
        rows = []
        while len(rows) < wanted and any(grouped.values()):
            for task in sorted(grouped):
                if grouped[task] and len(rows) < wanted: rows.append(grouped[task].pop(0))
    mode = model or resolve_vlm_model("RFS_ONLINE_JUDGE_MODEL", "RFS_CRITIC_MODEL")
    evaluation_dataset = root / "evaluation_dataset.jsonl"
    evaluation_dataset.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    failures: list[dict[str, Any]] = []
    with prediction_path.open("a", encoding="utf-8") as pred_handle, failure_path.open("a", encoding="utf-8") as fail_handle:
        for row in rows:
            example_id = str(row.get("example_id") or "")
            if not example_id or example_id in existing:
                continue
            images = _image_paths(row)
            missing = [p for p in images if not Path(p).exists()]
            if missing:
                failure = {"example_id": example_id, "task_type": row.get("task_type"), "error": "missing_image", "missing_images": missing}
                fail_handle.write(json.dumps(failure, ensure_ascii=False) + "\n"); failures.append(failure); continue
            metadata: dict[str, Any] = {}
            try:
                output = adapter(build_prompt(row), images, model=mode, timeout=timeout, retries=retries, call_metadata=metadata)
                record = {"example_id": example_id, "task_type": row.get("task_type"), "prediction": output, "raw_output": output, "normalized_prediction": output, "provider": {"model": mode, "metadata": metadata}, "prompt_version": PROMPT_VERSION, "error": None}
                pred_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"); pred_handle.flush()
            except Exception as exc:
                failure = {"example_id": example_id, "task_type": row.get("task_type"), "error": str(exc), "provider": {"model": mode, "metadata": metadata}}
                fail_handle.write(json.dumps(failure, ensure_ascii=False, separators=(",", ":")) + "\n"); fail_handle.flush(); failures.append(failure)
    def _count(path: Path) -> int:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    result = {"dataset": str(source.resolve()), "output": str(root.resolve()), "model": mode, "prompt_version": PROMPT_VERSION, "requested": len(rows), "already_completed": len(existing), "prediction_records": _count(prediction_path), "failure_records": _count(failure_path), "new_failures": len(failures), "api_configured": bool(os.getenv("API_BASE") and (os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")))}
    (root / "config.json").write_text(json.dumps({"dataset": str(source.resolve()), "model": mode, "timeout": timeout, "retries": retries, "prompt_version": PROMPT_VERSION, "limit": limit}, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluation = evaluate_predictions(evaluation_dataset, prediction_path, checkpoint=mode, suite="validation", report_out=root)
    (root / "metrics.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Judge Bootstrap Baseline", "", f"- Model: `{mode}`", f"- Prompt: `{PROMPT_VERSION}`", f"- Requested: `{len(rows)}`", f"- Predictions: `{result['prediction_records']}`", f"- Failures: `{result['failure_records']}`", "", "## Metrics", ""]
    for name, item in evaluation.get("metrics", {}).items(): lines.append(f"- `{name}`: {item.get('value')} (n={item.get('sample_count')})")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result["evaluation"] = {"gate_pass": evaluation.get("gate_pass"), "matched_records": evaluation.get("matched_records"), "metrics": evaluation.get("metrics")}
    return result
