"""Cosine similarity between a whole preview and the paper's method section.

This is the paper-figure CLIP column. A missing preview, a missing method
section, or a missing embedder is null, never zero. The embedder is injected
so tests do not download AltCLIP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence


class ClipEmbedder(Protocol):
    def embed_image(self, path: Path) -> Sequence[float]:
        """Return one image vector."""

    def embed_text(self, text: str) -> Sequence[float]:
        """Return one text vector."""


def cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    dot = sum(float(x) * float(y) for x, y in zip(left, right))
    left_norm = sum(float(x) * float(x) for x in left) ** 0.5
    right_norm = sum(float(y) * float(y) for y in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return None
    return dot / (left_norm * right_norm)


def score_paper_figure_clip(
    preview_path: str | Path | None,
    method_text: str | None,
    *,
    embedder: ClipEmbedder | None = None,
    model: str | None = None,
    render_error: str | None = None,
) -> dict[str, Any]:
    """Score one preview against one method section."""
    recorded_model = model or getattr(embedder, "model_name", None)
    base = {
        "summary": "Cosine between the whole preview and the method section.",
        "available": False,
        "cosine": None,
        "model": recorded_model,
        "reason": "",
    }
    if render_error:
        base["reason"] = "render_failed: " + render_error
        return base
    if not str(method_text or "").strip():
        base["reason"] = "missing method section"
        return base
    path = Path(preview_path) if preview_path else Path()
    if not path.is_file():
        base["reason"] = "missing preview.png"
        return base
    if embedder is None:
        base["reason"] = "embedder_unavailable"
        return base
    try:
        value = cosine(embedder.embed_image(path), embedder.embed_text(str(method_text)))
    except Exception as exc:
        base["reason"] = f"{type(exc).__name__}: {exc}"[:200]
        return base
    if value is None:
        base["reason"] = "empty embedding"
        return base
    base["available"] = True
    base["cosine"] = round(value, 6)
    return base
