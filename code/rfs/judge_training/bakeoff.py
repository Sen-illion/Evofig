"""Run a reproducible small validation comparison across VLM backbones."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .baseline import run_baseline


DEFAULT_MODELS = ["Qwen2.5-VL-7B-Instruct", "InternVL2.5-8B", "MiniCPM-V-2.6"]


def run_bakeoff(dataset: str | Path, out: str | Path, *, models: list[str] | None = None, limit: int = 40, timeout: int = 180, retries: int = 2) -> dict[str, Any]:
    root = Path(out); root.mkdir(parents=True, exist_ok=True)
    selected = models or DEFAULT_MODELS
    runs = []
    for model in selected:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
        result = run_baseline(dataset, root / safe, limit=limit, model=model, timeout=timeout, retries=retries)
        runs.append(result)
    payload = {"schema_version": "judge-backbone-bakeoff-v1", "dataset": str(Path(dataset).resolve()), "limit": limit, "models": selected, "runs": runs, "selection_status": "pending_metrics_review"}
    (root / "bakeoff_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

