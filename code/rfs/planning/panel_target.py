"""The paper's own figure panels, as the layout target of a benchmark run.

A product run draws its own overview of the method. A public benchmark scores the
figure the paper printed, down to its panel granularity, so a run measured by one
has to be aimed at that figure instead. The paper contract already carries those
panels: :mod:`rfs.analysis.paper.preparation` normalises ``panel_graphs`` into one
entry per sub-figure, each with the labels a reader can see in it.

Everything here is deterministic and calls nothing. The target is an instruction
and a measurement, never a hard constraint: a design that misses a label is
reported, not refused, because the paper's own figure is a reference and not a
contract this system can always satisfy.
"""

from __future__ import annotations

from typing import Any
import json
import re


#: Where a target came from, so a report reader never has to guess.
PANEL_TARGET_SOURCE = "figure_specification.panel_graphs"

#: The reason a target is unusable: the paper declares no panel structure at all
#: (no ``panel_graphs``, or none of them carries a labelled node).
NO_PANELS_REASON = "no_panel_graphs"

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)


def _normalize_label(value: Any) -> str:
    """Fold a label to the form two spellings of the same term can be compared in."""

    text = _WHITESPACE.sub(" ", str(value or "")).strip().casefold()
    return _PUNCTUATION.sub(" ", text).strip()


def panel_target_from_specification(
    specification: dict[str, Any] | None,
    *,
    max_items_per_panel: int = 16,
) -> dict[str, Any]:
    """Read the paper contract's panels into the target a benchmark run draws to.

    ``available`` is false when the paper declares no usable panel structure. That
    is a fact about the paper, not a failure: the caller falls back to the overview
    path and records the reason.
    """

    spec = specification if isinstance(specification, dict) else {}
    raw_panels = spec.get("panel_graphs") if isinstance(spec.get("panel_graphs"), list) else []
    panels: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_panels):
        if not isinstance(raw, dict):
            continue
        labels: list[str] = []
        seen: set[str] = set()
        for node in raw.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            label = str(node.get("name") or node.get("label") or node.get("instance_id") or "").strip()
            folded = _normalize_label(label)
            if not folded or folded in seen:
                continue
            seen.add(folded)
            labels.append(label)
        panels.append(
            {
                "panel_id": str(raw.get("id") or f"panel_{index + 1}").strip(),
                "panel_label": str(raw.get("panel_label") or "").strip(),
                "title": str(raw.get("label") or raw.get("description") or "").strip(),
                "labels": labels[:max_items_per_panel],
                "node_count": len(labels),
            }
        )
    label_count = sum(panel["node_count"] for panel in panels)
    available = bool(panels) and label_count > 0
    return {
        "available": available,
        "reason": "" if available else NO_PANELS_REASON,
        "source": PANEL_TARGET_SOURCE,
        "panel_count": len(panels),
        "label_count": label_count,
        "panels": panels if available else [],
    }


def compact_panel_target(
    target: dict[str, Any] | None,
    *,
    max_panels: int = 8,
    max_labels_per_panel: int = 12,
) -> str:
    """The target in prompt-sized form, or an empty string when there is none."""

    source = target if isinstance(target, dict) else {}
    if not source.get("available"):
        return ""
    rows = [
        {
            "panel": panel.get("panel_label") or panel.get("panel_id") or "",
            "title": panel.get("title") or "",
            "labels": list(panel.get("labels") or [])[:max_labels_per_panel],
        }
        for panel in (source.get("panels") or [])[:max_panels]
        if isinstance(panel, dict)
    ]
    if not rows:
        return ""
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def check_design_against_panel_target(
    design: dict[str, Any] | None,
    target: dict[str, Any] | None,
) -> dict[str, Any]:
    """Measure how much of the paper's own panel structure a design carries.

    Two counts are reported for the labels, because they answer different
    questions: ``labels_matched_count`` is the strict one (the design spells the
    label the way the paper does), and ``labels_contained_count`` also credits a
    label the design wrapped in a longer phrase. Reporting only the lenient one
    would overstate the alignment.
    """

    source = target if isinstance(target, dict) else {}
    if not source.get("available"):
        return {"enabled": False, "reason": str(source.get("reason") or NO_PANELS_REASON)}
    plan = design if isinstance(design, dict) else {}
    design_panels = plan.get("panels") if isinstance(plan.get("panels"), list) else []
    design_labels = [
        _normalize_label(node.get("label") or node.get("name"))
        for node in (plan.get("nodes") or [])
        if isinstance(node, dict)
    ]
    design_labels = [label for label in design_labels if label]
    matched: list[str] = []
    contained: list[str] = []
    missing: list[str] = []
    for panel in source.get("panels") or []:
        if not isinstance(panel, dict):
            continue
        for label in panel.get("labels") or []:
            folded = _normalize_label(label)
            if not folded:
                continue
            if folded in design_labels:
                matched.append(label)
                contained.append(label)
            elif any(folded in candidate for candidate in design_labels):
                contained.append(label)
            else:
                missing.append(label)
    total = max(1, len(matched) + len(missing))
    return {
        "enabled": True,
        "panels_target_count": int(source.get("panel_count") or 0),
        "panels_produced_count": len(design_panels),
        "labels_target_count": len(matched) + len(missing),
        "labels_matched_count": len(matched),
        "labels_contained_count": len(contained),
        "labels_missing_count": len(missing),
        "label_coverage": round(len(matched) / total, 4),
        "container_coverage": round(len(contained) / total, 4),
        "missing_labels": missing[:32],
    }


__all__ = [
    "NO_PANELS_REASON",
    "PANEL_TARGET_SOURCE",
    "check_design_against_panel_target",
    "compact_panel_target",
    "panel_target_from_specification",
]
