"""VLM planner for native editable figure text."""

from __future__ import annotations
import json
import re
from typing import Any
from ..contracts import normalize_figure_text_plan, validate_figure_text_plan
from ..providers import call_vlm_json, resolve_vlm_model
from ..contracts.vlm_context import compact_method, compact_reference, context_stats, json_compact


#: A single token that is one of the planner's own ids rather than words the paper
#: printed. The design hands the model connector records that carry an id and, when the
#: paper wrote nothing on the edge, no label; the model then echoes the id back as the
#: label and it is drawn. A label with a space in it is always words, so only a single
#: token has to be checked.
_IDENTIFIER_LABEL = re.compile(
    r"^(?:panel_?rel|panel|stage|node|edge|sedge|snode|nose|connector|conn|relation|rel)[._\-][A-Za-z0-9._\-]+$",
    re.IGNORECASE,
)


def _identifier_label(text: str) -> bool:
    """True when a string is one of the planner's own ids rather than printed words."""

    value = str(text or "").strip()
    if not value or " " in value:
        return False
    return bool(_IDENTIFIER_LABEL.match(value))


def _drop_identifier_labels(plan: dict[str, Any], design: dict[str, Any]) -> tuple[dict[str, Any], list]:
    """Remove text items whose text is an internal id, and name the ones that went."""

    known: set[str] = set()
    for group in ("panels", "nodes", "connectors"):
        for item in design.get(group) or []:
            if not isinstance(item, dict):
                continue
            for key in ("id", "semantic_source_id"):
                value = str(item.get(key) or "").strip()
                if value:
                    known.add(value)
    kept, dropped = [], []
    for item in plan.get("items") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if text and (text in known or _identifier_label(text)):
            dropped.append(text)
            continue
        kept.append(item)
    if not dropped:
        return plan, []
    updated = dict(plan)
    updated["items"] = kept
    provenance = dict(updated.get("provenance") or {})
    provenance["identifier_labels_dropped"] = sorted(set(dropped))
    updated["provenance"] = provenance
    return updated, sorted(set(dropped))


def plan_figure_text_vlm(
    method: dict[str, Any],
    design: dict[str, Any],
    *,
    exemplar_style_brief: dict[str, Any] | None = None,
    model: str | None = None,
    timeout: int = 180,
    retries: int = 1,
):
    prompt = (
        "Return exactly one non-empty JSON object matching {items:[{id,text,text_role,semantic_source_id,panel_id,bbox_percent:{x,y,w,h},priority,editable,font_size_role,font_size_pt,font_weight,color_role,max_lines,min_readable_size_pt}],text_policy:{native_ppt_only:true}}. Every bbox_percent uses normalized x,y,width,height values in 0..1, never an array or x1,y1,x2,y2. Include at least one item per design node and panel titles when available. A node with render_policy=panel_only or visual_role=semantic_container is a PPT-native parent container: place its text in a compact header lane inside the top of that node and do not reserve a centered icon-label region; keep each child label inside its own child node. Use panel_title 14-20pt, primary node 12-16pt, secondary node 10-13pt, connector label 8-10pt; preserve hierarchy and never request below 8pt. Prefer a short two-line label over tiny text. Preserve exact scientific terminology, provide short text for long labels, and keep all text native PPT. Do not return markdown or an empty items array. METHOD="
        + json_compact(compact_method(method))
        + " EXEMPLAR_STYLE_BRIEF="
        + json_compact(compact_reference({}, exemplar_style_brief))
        + " DESIGN="
        + json_compact({"canvas": design.get("canvas"), "panels": design.get("panels"), "nodes": design.get("nodes"), "connectors": design.get("connectors")})
    )
    raw = None
    provider_metadata: dict[str, Any] = {}
    try:
        raw = call_vlm_json(
            prompt,
            [],
            model=resolve_vlm_model("RFS_TEXT_PLAN_MODEL", explicit_model=model),
            timeout=timeout,
            retries=retries,
            call_metadata=provider_metadata,
        )
        plan = normalize_figure_text_plan(raw)
        plan, dropped_identifiers = _drop_identifier_labels(plan, design)
        validation = validate_figure_text_plan(plan)
        if not validation["ok"]:
            raise ValueError("invalid VLM text plan: " + "; ".join(validation["errors"]))
        return plan, {
            **validation,
            "mode": "vlm",
            "fallback_used": False,
            "identifier_labels_dropped": dropped_identifiers,
            "raw_response": raw,
            "context": context_stats(prompt, field_names=["compact_method", "text_rules", "design_geometry", "style_rules"]),
            "provider": provider_metadata,
        }
    except Exception as exc:
        nodes=[n for n in design.get("nodes",[]) if isinstance(n,dict)]
        # Only a node the design actually named is labelled; the id is kept only when
        # nothing at all was named, so the fallback plan is never empty.
        chosen=[n for n in nodes if str(n.get("label") or "").strip()] or nodes
        items=[{"id":f"text_{n.get('id')}","text":str(n.get("label") or n.get("id") or ""),"text_role":"node_label","semantic_source_id":n.get("semantic_source_id"),"bbox_percent":n.get("bbox_percent"),"editable":True} for n in chosen]
        plan=normalize_figure_text_plan({"items":items,"text_policy":{"native_ppt_only":True},"provenance":{"mode":"semantic_fallback"}})
        return plan, {**validate_figure_text_plan(plan), "mode": "semantic_fallback", "fallback_used": True, "fallback_reason": str(exc), "raw_response": raw, "context": context_stats(prompt, field_names=["compact_method", "text_rules", "design_geometry", "style_rules"]), "provider": provider_metadata}
