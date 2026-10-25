"""Deterministic structural motif classification for graph neighborhoods.

``repeated`` is the one motif that is not read off the neighborhood: a reading that
saw the same object drawn more than once records the group on the object itself, and
that statement outranks the shape of its arrows.
"""

from __future__ import annotations

from typing import Any


def classify_node_motif(
    node_id: str,
    edges: list[dict[str, Any]],
    *,
    node: dict[str, Any] | None = None,
    nodes_by_id: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Classify the most informative supported motif around one node."""

    properties = (node or {}).get("properties")
    if isinstance(properties, dict) and str(properties.get("repeat_group") or "").strip():
        return "repeated"
    incoming = [item for item in edges if str(item.get("target")) == node_id]
    outgoing = [item for item in edges if str(item.get("source")) == node_id]
    if any(str(item.get("relation_type") or "").casefold() in {"feedback", "feedback_loop", "iteration", "iterate", "return_flow"} for item in [*incoming, *outgoing]):
        return "feedback"
    phases = {
        str((nodes_by_id or {}).get(endpoint, {}).get("properties", {}).get("phase") or "")
        for item in [*incoming, *outgoing]
        for endpoint in (str(item.get("source") or ""), str(item.get("target") or ""))
    }
    if {"training", "inference"} <= phases:
        return "training_inference"
    panels = {
        value
        for item in [node or {}, *((nodes_by_id or {}).get(str(edge.get("source") or ""), {}) for edge in incoming), *((nodes_by_id or {}).get(str(edge.get("target") or ""), {}) for edge in outgoing)]
        for value in item.get("panel_ids", [])
    }
    if len(panels) > 1:
        return "cross_panel"
    if len(incoming) > 1 and len(outgoing) > 1:
        return "multi_input" if len(incoming) >= len(outgoing) else "multi_output"
    if len(incoming) > 1:
        return "merge"
    if len(outgoing) > 1:
        return "branch"
    if len(incoming) == 1 and len(outgoing) == 1:
        return "linear"
    if len(incoming) > 0:
        return "multi_input"
    if len(outgoing) > 0:
        return "multi_output"
    return "unknown"
