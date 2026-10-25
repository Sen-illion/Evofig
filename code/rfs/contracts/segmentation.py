"""Stable contracts for semantic-to-visual slot segmentation plans."""

from __future__ import annotations

from typing import Any


SEGMENTATION_CONTRACT_VERSION = 1


def normalize_segmentation_plan(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a stable segmentation plan with explicit asset ownership."""

    source = raw if isinstance(raw, dict) else {}
    slots = source.get("slots") if isinstance(source.get("slots"), list) else []
    normalized_slots: list[dict[str, Any]] = []
    for index, item in enumerate(slots, start=1):
        if not isinstance(item, dict):
            continue
        slot_id = str(item.get("slot_id") or item.get("id") or f"slot_{index:02d}").strip()
        normalized_slots.append({
            **item,
            "slot_id": slot_id,
            "semantic_node_ids": [str(value) for value in item.get("semantic_node_ids", []) if str(value).strip()],
            "editable_object_ids": [str(value) for value in item.get("editable_object_ids", []) if str(value).strip()],
            "input_ports": [str(value) for value in item.get("input_ports", []) if str(value).strip()],
            "output_ports": [str(value) for value in item.get("output_ports", []) if str(value).strip()],
            "visual_asset_scope": [str(value) for value in item.get("visual_asset_scope", []) if str(value).strip()],
            "forbidden_baked_content": [str(value) for value in item.get("forbidden_baked_content", ["labels", "formulas", "cross_slot_connectors"]) if str(value).strip()],
            "asset_source_policy": str(item.get("asset_source_policy") or "api_generate"),
            "asset_policy": str(item.get("asset_policy") or "visual_only_no_text_no_arrows"),
            "text_policy": str(item.get("text_policy") or "native_ppt"),
            "connector_policy": str(item.get("connector_policy") or "native_ppt"),
        })
    return {
        "contract": "rfs.segmentation_plan",
        "version": int(source.get("version") or SEGMENTATION_CONTRACT_VERSION),
        "plan_id": str(source.get("plan_id") or "segmentation_default"),
        "summary": str(source.get("summary") or "Semantic-to-visual segmentation plan."),
        "slots": normalized_slots,
        "rationale": [str(value) for value in source.get("rationale", []) if str(value).strip()],
        "provenance": source.get("provenance") if isinstance(source.get("provenance"), dict) else {},
    }


def validate_segmentation_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate slot identity and explicit ownership policies."""

    normalized = normalize_segmentation_plan(plan)
    errors: list[str] = []
    slot_ids = [str(item.get("slot_id") or "") for item in normalized["slots"]]
    if len(slot_ids) != len(set(slot_ids)):
        errors.append("segmentation slot IDs must be unique")
    for index, slot in enumerate(normalized["slots"]):
        if not slot["slot_id"]:
            errors.append(f"slot[{index}] requires slot_id")
        if not slot["semantic_node_ids"]:
            errors.append(f"slot[{index}] requires semantic_node_ids")
        bbox = slot.get("bbox_percent")
        if bbox is not None and not isinstance(bbox, dict):
            errors.append(f"slot[{index}] bbox_percent must be an object")
        if slot["asset_policy"] != "visual_only_no_text_no_arrows":
            errors.append(f"slot[{index}] must use visual_only_no_text_no_arrows asset policy")
        if slot["text_policy"] != "native_ppt":
            errors.append(f"slot[{index}] must use native_ppt text policy")
        if slot["connector_policy"] != "native_ppt":
            errors.append(f"slot[{index}] must use native_ppt connector policy")
    return {"ok": not errors, "errors": errors, "slot_count": len(normalized["slots"])}
