"""Immutable state snapshots, hashes, diffs, and rollback for direct evolution."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def state_hash(state: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(state).encode("utf-8")).hexdigest()


def create_state_snapshot(state: dict[str, Any], *, label: str, round_index: int) -> dict[str, Any]:
    """Capture a deep immutable copy with a verifiable content hash."""

    payload = deepcopy(state)
    return {
        "contract": "rfs.direct_state_snapshot",
        "version": 1,
        "label": str(label),
        "round": int(round_index),
        "state_hash": state_hash(payload),
        "state": payload,
    }


def verify_state_snapshot(snapshot: dict[str, Any]) -> bool:
    state = snapshot.get("state") if isinstance(snapshot.get("state"), dict) else {}
    return bool(snapshot.get("state_hash")) and str(snapshot["state_hash"]) == state_hash(state)


def restore_state_snapshot(snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore only verified snapshots and report the rollback hash."""

    if not verify_state_snapshot(snapshot):
        raise ValueError("cannot restore an invalid or modified direct evolution snapshot")
    restored = deepcopy(snapshot["state"])
    return restored, {
        "contract": "rfs.direct_rollback_report",
        "version": 1,
        "ok": True,
        "restored_hash": snapshot["state_hash"],
        "source_label": snapshot.get("label"),
        "round": snapshot.get("round"),
    }


def diff_state_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Report changed top-level objects and preserve snapshot provenance."""

    if not verify_state_snapshot(before) or not verify_state_snapshot(after):
        raise ValueError("snapshot diff requires two valid snapshots")
    left, right = before["state"], after["state"]
    keys = sorted(set(left) | set(right))
    changes = [
        {"field": key, "before": left.get(key), "after": right.get(key)}
        for key in keys if _canonical(left.get(key)) != _canonical(right.get(key))
    ]
    return {
        "contract": "rfs.direct_state_diff",
        "version": 1,
        "before_hash": before["state_hash"],
        "after_hash": after["state_hash"],
        "changed_fields": [item["field"] for item in changes],
        "changes": changes,
        "changed": bool(changes),
    }


SKELETON_GEOMETRY_PRECISION = 2


def _quantized(value: Any) -> float | None:
    """Quantize one canvas coordinate to a hundredth, i.e. one percent of the canvas."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), SKELETON_GEOMETRY_PRECISION)
    return None


def _box_key(item: dict[str, Any]) -> list[Any]:
    box = item.get("bbox_percent") if isinstance(item.get("bbox_percent"), dict) else item.get("bbox") if isinstance(item.get("bbox"), dict) else {}
    return [_quantized(box.get(axis)) for axis in ("x", "y", "w", "h")]


def skeleton_state_hash(state: dict[str, Any] | None) -> str:
    """Hash the figure's skeleton: its panel, node and connector counts, how the nodes connect,
    and roughly where they sit -- never their ids, names, colours, fonts or text.

    The paired protocol asks whether both arms started from the same figure. Read literally,
    they cannot: one stage plans its own colours and writes its own names, so two runs of the
    same paper differ byte for byte without a box moving, and a hash that counts those
    differences refuses every honest pair. What a reader sees as the start state is the
    skeleton, so that is what the audit compares.

    Nodes are numbered by their place in the figure rather than by the id a rebuild rewrites:
    a node's slot is its panel's box plus its own, and connectors are recorded as pairs of those
    slots. Two runs that laid the same boxes therefore sign the same adjacency, whatever they
    called the nodes. An empty layout hashes to the empty string, so a state that carries no
    figure reads as unmeasured instead of as a figure that happens to be empty.
    """

    source = state if isinstance(state, dict) else {}
    layout = source.get("layout") if isinstance(source.get("layout"), dict) else {}
    panels = [item for item in (layout.get("panels") or []) if isinstance(item, dict)]
    nodes = [item for item in (layout.get("nodes") or []) if isinstance(item, dict)]
    connectors = [item for item in (layout.get("connectors") or []) if isinstance(item, dict)]
    if not panels and not nodes and not connectors:
        return ""
    panel_box = {str(panel.get("id") or ""): _canonical(_box_key(panel)) for panel in panels if str(panel.get("id") or "")}
    panel_slots = sorted(set(panel_box.values()))
    panel_slot = {key: index for index, key in enumerate(panel_slots)}

    def node_slot_key(node: dict[str, Any]) -> str:
        owner = panel_slot.get(panel_box.get(str(node.get("panel_id") or ""), ""), -1)
        return _canonical([owner, _box_key(node)])

    node_keys = {str(node.get("id") or ""): node_slot_key(node) for node in nodes if str(node.get("id") or "")}
    node_slots = sorted(set(node_keys.values()))
    node_rank = {key: index for index, key in enumerate(node_slots)}
    edges = []
    for connector in connectors:
        source_rank = node_rank.get(node_keys.get(str(connector.get("source") or ""), ""))
        target_rank = node_rank.get(node_keys.get(str(connector.get("target") or ""), ""))
        if source_rank is None or target_rank is None:
            continue
        edges.append([source_rank, target_rank])
    body = {
        "panel_count": len(panels),
        "node_count": len(nodes),
        "connector_count": len(connectors),
        "panel_boxes": panel_slots,
        "node_slots": node_slots,
        "edges": sorted(edges),
    }
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def graph_layer_digest(graph: dict[str, Any] | None) -> str:
    """Hash the nodes and edges of one layer, ignoring ids that a rebuild rewrites."""

    source = graph if isinstance(graph, dict) else {}
    body = {"nodes": source.get("nodes") or [], "edges": source.get("edges") or []}
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def layer_change(previous: dict[str, str] | None, current: dict[str, str], *, before_score: float | None, after_score: float | None, accepted: bool) -> dict[str, Any]:
    """Record whether each layer moved relative to the figure the loop has kept."""

    prior = previous if isinstance(previous, dict) and previous else None
    changed = {
        name: None if prior is None else prior.get(name) != current.get(name)
        for name in ("scientific", "visual", "editable")
    }
    return {
        "contract": "rfs.layer_change",
        "version": 1,
        "scientific_changed": changed["scientific"],
        "visual_changed": changed["visual"],
        "editable_changed": changed["editable"],
        "digests": dict(current),
        "before_score": before_score,
        "after_score": after_score,
        "score_delta": None if before_score is None or after_score is None else round(after_score - before_score, 4),
        "accepted": bool(accepted),
    }


FIGURE_PROGRAM_OBJECT_KEYS = ("panels", "semantic_nodes", "arrows")


def diff_figure_programs(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    """Report which figure objects a round changed, so a round that moved nothing cannot pass
    as a repair -- a reroute the router resolves to the same paths, or an escalation the
    workflow declined, both leave the figure exactly as it was.

    A state diff over a compiled candidate always reports a change, because the candidate
    snapshot carries the program, the graph ids and the Gate result the round-before state
    does not. The objects below are the ones a reader would actually see, so a round that
    re-emitted the same figure reports ``changed: False``.
    """

    left = before if isinstance(before, dict) else {}
    right = after if isinstance(after, dict) else {}
    available = bool(left) and bool(right)
    changed: list[str] = []
    for key in FIGURE_PROGRAM_OBJECT_KEYS:
        before_items = {str(item.get("id") or ""): item for item in (left.get(key) or []) if isinstance(item, dict)}
        after_items = {str(item.get("id") or ""): item for item in (right.get(key) or []) if isinstance(item, dict)}
        for object_id in sorted(set(before_items) | set(after_items)):
            if _canonical(before_items.get(object_id)) != _canonical(after_items.get(object_id)):
                changed.append(object_id)
    return {
        "contract": "rfs.direct_figure_diff",
        "version": 1,
        "available": available,
        "changed": bool(changed) if available else None,
        "changed_object_ids": changed if available else [],
    }
