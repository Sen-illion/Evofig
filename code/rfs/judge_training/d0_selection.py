"""Select a deterministic, label-stratified D0 protocol set from a frozen manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def select_d0(papers_path: str | Path, out_path: str | Path, *, size: int = 45, split: str = "validation") -> dict[str, Any]:
    """Select D0 papers without touching the hidden test split."""
    records = [json.loads(line) for line in Path(papers_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible = [item for item in records if item.get("split") == split]
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in eligible:
        groups.setdefault(str(item.get("final_label") or "unknown"), []).append(item)
    for items in groups.values():
        items.sort(key=lambda item: hashlib.sha256(str(item.get("arxiv_id")).encode("utf-8")).hexdigest())
    selected: list[dict[str, Any]] = []
    labels = sorted(groups)
    while len(selected) < min(size, len(eligible)):
        progressed = False
        for label in labels:
            if groups[label] and len(selected) < size:
                item = groups[label].pop(0)
                selected.append({"arxiv_id": item["arxiv_id"], "split": item["split"], "final_label": item["final_label"]})
                progressed = True
        if not progressed:
            break
    payload = {"dataset_version": "judge-framework-d0-v001", "source_split": split, "size": len(selected), "records": selected, "label_counts": {label: sum(item["final_label"] == label for item in selected) for label in labels}}
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
