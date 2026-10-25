"""Require post-binding text and strict completion at a collector batch boundary."""

from pathlib import Path

from .source_snapshot_io import canonical_hash, strict_jsonl


def collection_boundary(batch: Path) -> dict:
    """Combine legacy ordered events with the caller's OS guard and process check."""
    path = batch / "manifests/multisource_collection_events.jsonl"
    if not path.exists():
        return {"ready": True, "event_evidence": "no_collector_events"}
    events = strict_jsonl(path)
    starts = [i for i, e in enumerate(events) if e.get("stage") == "asset-bind" and e.get("status") == "started"]
    if not starts:
        return {"ready": True, "event_evidence": "no_asset_batch_started", "events_sha256": canonical_hash(events)}
    start = starts[-1]
    tail = [(i, e) for i, e in enumerate(events) if i > start]
    terminal = next(((i, e) for i, e in tail if e.get("stage") == "asset-bind" and e.get("status") in {"completed", "failed"}), None)
    if terminal is None:
        return {"ready": False, "reason": "asset_batch_has_no_terminal_event", "started_event": events[start]}
    if terminal[1]["status"] == "failed":
        return {"ready": True, "event_evidence": "failed_batch_exited_without_active_writers",
                "terminal_event": terminal[1], "events_sha256": canonical_hash(events)}
    text = next((i for i, e in tail if i > terminal[0] and e.get("stage") == "fulltext" and e.get("status") == "completed"), None)
    frozen = next((i for i, e in tail if text is not None and i > text and e.get("stage") == "freeze" and e.get("status") == "completed"), None)
    if frozen is None:
        return {"ready": False, "reason": "waiting_for_post_asset_fulltext_and_freeze", "asset_completed_event": terminal[1]}
    return {"ready": True, "event_evidence": "asset_then_fulltext_then_freeze_completed",
            "terminal_event": events[frozen], "events_sha256": canonical_hash(events),
            "run_identity": "legacy_events_without_run_id_guard_and_process_evidence_required"}
