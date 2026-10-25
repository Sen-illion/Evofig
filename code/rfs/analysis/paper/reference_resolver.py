"""Resolve curated positive framework references for paper inputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_POSITIVE = {"优", "优秀", "正向", "positive", "good", "excellent", "selected", "pass", "浼?", "浼樼", "姝ｅ悜"}
_NEGATIVE = {"差", "较差", "负向", "negative", "bad", "rejected", "fail", "坏", "鍧?", "璐熷悜"}
_ID_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?!\d)")


def normalize_quality_label(value: Any) -> str:
    """Normalize multilingual and legacy quality labels."""
    label = str(value or "").strip().casefold()
    if label in {token.casefold() for token in _POSITIVE}:
        return "positive"
    if label in {token.casefold() for token in _NEGATIVE}:
        return "negative"
    return label


def _values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _values(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _values(nested)]
    return [str(value).strip()] if value is not None else []


def _paper_ids(paper: str | Path) -> list[str]:
    ids = _ID_RE.findall(str(paper))
    source = Path(paper)
    if source.suffix.lower() == ".pdf" and source.exists():
        try:
            import fitz  # type: ignore
            with fitz.open(source) as document:
                metadata = document.metadata or {}
            ids.extend(_ID_RE.findall(" ".join(str(value or "") for value in metadata.values())))
        except Exception:
            pass
    return list(dict.fromkeys(ids))


def _candidate_images(folder: Path) -> list[Path]:
    preferred = [folder / "framework.png", folder / "framework.jpg", folder / "framework.jpeg"]
    others = sorted((p for p in folder.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}), key=lambda p: p.name.casefold())
    return list(dict.fromkeys([p for p in preferred if p.exists()] + others))


def resolve_positive_reference(paper: str | Path, *, dataset_root: str | Path | None = None) -> dict[str, Any]:
    """Return the highest-ranked verified positive reference for ``paper``."""
    root = Path(dataset_root) if dataset_root else Path(__file__).resolve().parents[3] / "datasets" / "judge-d0" / "figures"
    paper_ids = _paper_ids(paper)
    candidates: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        folder = root / paper_id
        metadata_path = folder / "meta.json"
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            labels = {normalize_quality_label(value) for value in _values(raw)}
            reason = None
        except Exception as exc:
            labels, reason = set(), str(exc)
        quality = "negative" if "negative" in labels else "positive" if "positive" in labels else "unknown"
        for image in _candidate_images(folder):
            candidates.append({"paper_id": paper_id, "image": str(image.resolve()), "metadata": str(metadata_path.resolve()) if metadata_path.exists() else None, "quality_label": quality, "reason": reason, "_rank": (2 if quality == "positive" else 0 if quality == "negative" else 1, 1 if image.name.casefold().startswith("framework") else 0)})
    candidates.sort(key=lambda item: item["_rank"], reverse=True)
    for item in candidates:
        item.pop("_rank", None)
    selected = next((item for item in candidates if item["quality_label"] == "positive"), None)
    if selected:
        return {"status": "selected", "selection_source": "dataset_paper_id_match", "candidates": candidates, **selected}
    if candidates:
        best = candidates[0]
        return {"status": "rejected" if best["quality_label"] == "negative" else "unverified", "candidates": candidates, **best}
    return {"status": "not_found", "paper_id": paper_ids[0] if paper_ids else None, "image": None, "quality_label": None, "candidates": []}
