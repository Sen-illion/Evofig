"""Deterministic visibility checks for editable connector paths."""

from __future__ import annotations

from typing import Any


def _inside(point: list[float], box: dict[str, Any], margin: float = 0.0) -> bool:
    return (
        float(box.get("x", 0)) - margin <= float(point[0]) <= float(box.get("x", 0)) + float(box.get("w", 0)) + margin
        and float(box.get("y", 0)) - margin <= float(point[1]) <= float(box.get("y", 0)) + float(box.get("h", 0)) + margin
    )


def _segment_hits_box(a: list[float], b: list[float], box: dict[str, Any], margin: float = 0.004) -> bool:
    for index in range(21):
        t = index / 20
        point = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
        if _inside(point, box, margin):
            return True
    return False


def _container_ids(objects: dict[str, dict[str, Any]], object_id: str) -> set[str]:
    """Collect the containers an endpoint already lives in.

    A card is not an obstacle to a link between its own children: the child sits
    inside the parent box, so every path to it has to enter that box. Corpus
    figures nest submodules this way, and reading the parent as an unrelated
    object flagged 13 of 15 links of a figure whose own route review found no
    collision at all.
    """
    chain: set[str] = set()
    current = str(object_id)
    while True:
        parent = str((objects.get(current) or {}).get("parent_id") or "")
        if not parent or parent in chain or parent == object_id:
            return chain
        chain.add(parent)
        current = parent


def evaluate_connector_visibility(program: dict[str, Any]) -> dict[str, Any]:
    """Report connectors hidden by unrelated nodes or slots without rewriting them."""
    objects = {
        str(item.get("id")): item
        for key in ("semantic_nodes", "slots")
        for item in (program.get(key) or [])
        if isinstance(item, dict) and item.get("id") and isinstance(item.get("bbox_percent"), dict)
    }
    issues: list[dict[str, Any]] = []
    for arrow in program.get("arrows", []) or []:
        if not isinstance(arrow, dict):
            continue
        path = arrow.get("path_percent") if isinstance(arrow.get("path_percent"), list) else []
        if len(path) < 2:
            issues.append({"severity": "blocker", "arrow_id": arrow.get("id"), "reason": "connector has no usable path"})
            continue
        source_id = str(arrow.get("source") or arrow.get("source_id") or "")
        target_id = str(arrow.get("target") or arrow.get("target_id") or "")
        # The endpoint's own slot stays in scope on purpose: a link that starts
        # inside its own card and runs across its Icon is a real defect.
        related = {source_id, target_id} | _container_ids(objects, source_id) | _container_ids(objects, target_id)
        hits = []
        for object_id, obj in objects.items():
            if object_id in related:
                continue
            if any(_segment_hits_box(a, b, obj["bbox_percent"]) for a, b in zip(path[:-1], path[1:])):
                hits.append(object_id)
        if hits:
            issues.append({"severity": "major", "arrow_id": arrow.get("id"), "reason": "connector intersects unrelated editable object", "covered_by_object_ids": hits, "requested_action": "reroute"})
    status = "blocked" if any(item["severity"] == "blocker" for item in issues) else "needs_reroute" if issues else "pass"
    return {"summary": "Deterministic connector visibility report.", "status": status, "issue_count": len(issues), "issues": issues, "connector_count": len([x for x in program.get("arrows", []) or [] if isinstance(x, dict)])}


__all__ = ["evaluate_connector_visibility"]
