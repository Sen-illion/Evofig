"""Reproducibly select a positive framework-paper reference from the dataset."""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any

def select_positive_framework_reference(dataset_root: str | Path, *, seed: int = 20260827) -> dict[str, Any]:
    """Select one usable figure labeled 好（科研） with deterministic sampling."""
    root = Path(dataset_root)
    positive_label = "\u597d\uff08\u79d1\u7814\uff09"
    papers = {item["paper_id"]: item for item in (json.loads(line) for line in (root / "papers.jsonl").read_text(encoding="utf-8").splitlines()) if isinstance(item, dict) and item.get("paper_id")}
    figures = [item for item in (json.loads(line) for line in (root / "figures.jsonl").read_text(encoding="utf-8").splitlines()) if isinstance(item, dict) and item.get("final_label") == positive_label]
    grouped: dict[str, dict[str, Any]] = {}
    for figure in sorted(figures, key=lambda item: (str(item.get("paper_id")), int(item.get("candidate_rank") or 999), str(item.get("figure_id")))):
        paper_id = str(figure.get("paper_id")); paper = papers.get(paper_id, {})
        paper_path, figure_path = Path(str(paper.get("paper_path") or "")), Path(str(figure.get("figure_path") or ""))
        if paper_path.exists() and figure_path.exists() and paper_id not in grouped:
            grouped[paper_id] = {"paper_id": paper_id, "title": paper.get("title"), "paper_path": str(paper_path.resolve()), "figure_path": str(figure_path.resolve()), "figure_id": figure.get("figure_id"), "final_label": figure.get("final_label")}
    eligible = sorted(grouped.values(), key=lambda item: item["paper_id"])
    if not eligible:
        raise ValueError(f"No usable {positive_label} framework references found")
    selected = dict(random.Random(int(seed)).choice(eligible))
    return {"seed": int(seed), "label_filter": positive_label, "eligible_count": len(eligible), "selected": selected, "eligible_paper_ids": [item["paper_id"] for item in eligible]}
