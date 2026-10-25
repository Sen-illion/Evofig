"""Compact, stage-specific context builders for VLM requests.

The builders keep evidence IDs and short quotes for traceability while
preventing later planning stages from re-sending full paper contracts.
"""

from __future__ import annotations

import json
from typing import Any


def _short(value: Any, limit: int = 600) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def compact_document_index(index: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only headings and figure/table captions needed for review."""
    source = index if isinstance(index, dict) else {}
    sections = [
        {"title": _short(item.get("title"), 120), "page": item.get("page")}
        for item in source.get("sections", [])
        if isinstance(item, dict) and item.get("title")
    ]
    figures = [
        {"page": item.get("page"), "text": _short(item.get("text"), 700)}
        for item in source.get("figures", [])
        if isinstance(item, dict)
    ]
    tables = [
        {"page": item.get("page"), "text": _short(item.get("text"), 400)}
        for item in source.get("tables", [])
        if isinstance(item, dict)
    ]
    return {"sections": sections[:80], "figures": figures[:24], "tables": tables[:16]}


def compact_evidence(parsed: dict[str, Any] | None, *, max_chars: int = 14000) -> str:
    """Return prioritized evidence without duplicating full document text."""
    source = parsed if isinstance(parsed, dict) else {}
    evidence = [item for item in source.get("evidence", []) if isinstance(item, dict)]

    def priority(item: dict[str, Any]) -> tuple[int, int, str]:
        kind = str(item.get("kind") or "").casefold()
        section = str(item.get("section_hint") or "").casefold()
        text = str(item.get("text") or "").casefold()
        score = 0
        if kind == "caption" and any(term in text for term in ("framework", "architecture", "pipeline", "overview")):
            score -= 40
        if any(term in section for term in ("method", "approach", "architecture", "framework", "abstract")):
            score -= 20
        return score, int(item.get("page") or 0), str(item.get("block_id") or "")

    blocks, used = [], 0
    for item in sorted(evidence, key=priority):
        remaining = max_chars - used
        if remaining < 80:
            break
        header = f"[{item.get('id')} | page {item.get('page')} | {item.get('section_hint') or 'unknown section'}]\n"
        block = header + _short(item.get("text"), max(40, min(1200, remaining - len(header))))
        if used + len(block) > max_chars:
            continue
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def compact_review(review: dict[str, Any] | None, *, max_items: int = 24) -> dict[str, Any]:
    """Reduce a review to fields consumed by method and visual planners."""
    source = review if isinstance(review, dict) else {}

    def items(name: str) -> list[dict[str, Any]]:
        values = source.get(name) if isinstance(source.get(name), list) else []
        result = []
        for item in values[:max_items]:
            if not isinstance(item, dict):
                continue
            result.append({
                "id": item.get("id"),
                "statement": _short(item.get("statement"), 360),
                "visible_label": _short(item.get("visible_label") or item.get("name"), 100),
                "role": item.get("visual_role") or item.get("role"),
                "evidence_ids": list(item.get("evidence_ids") or [])[:6],
                "must_appear_in_figure": item.get("must_appear_in_figure"),
            })
        return result

    return {
        "paper_identity": source.get("paper_identity") or {},
        "research_questions": items("research_questions")[:4],
        "central_claims": items("central_claims")[:4],
        "inputs": items("inputs"),
        "outputs": items("outputs"),
        "modules": items("modules"),
        "relations": items("relations"),
        "innovations": items("innovations")[:8],
        "terminology": items("terminology")[:16],
    }


def compact_method(method: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only fields required by downstream visual planning."""
    source = method if isinstance(method, dict) else {}
    result: dict[str, Any] = {"figure_goal": _short(source.get("figure_goal"), 500)}
    for field in ("inputs", "modules", "outputs", "stages", "innovations", "optional_details", "relations", "uncertainties"):
        values = source.get(field) if isinstance(source.get(field), list) else []
        result[field] = []
        # A dense framework figure states one relation per printed arrow -- the
        # panels of a multi-panel figure carry dozens -- while the other fields
        # are objects. Cutting relations at the object budget told the design
        # stage about a fraction of the paper's own wiring.
        limit = 64 if field == "relations" else 32
        for item in values[:limit]:
            if not isinstance(item, dict):
                continue
            if field == "relations":
                result[field].append({k: item.get(k) for k in ("id", "source", "target", "type", "label", "evidence_ids")})
            else:
                result[field].append({k: item.get(k) for k in ("id", "name", "label", "role", "parent_id", "detail_level", "scientific_importance", "visualization_value", "must_show", "should_render", "render_priority", "evidence_ids", "input_ids", "output_ids")})
    result["panel_graphs"] = _compact_panel_graphs(source.get("panel_graphs"))
    return result


def _compact_panel_graphs(panels: Any, *, max_panels: int = 8, max_nodes: int = 40) -> list[dict[str, Any]]:
    """Keep the paper's own panel structure and the exact labels inside each panel.

    The panel graphs are the only place where a paper's inner modules are named
    one by one, so dropping them here collapsed every dense framework figure into
    a handful of coarse boxes downstream.
    """
    rows: list[dict[str, Any]] = []
    for panel in panels if isinstance(panels, list) else []:
        if not isinstance(panel, dict):
            continue
        nodes = []
        for node in panel.get("nodes") if isinstance(panel.get("nodes"), list) else []:
            if not isinstance(node, dict):
                continue
            name = str(node.get("name") or node.get("label") or "").strip()
            if not name:
                continue
            nodes.append({
                "instance_id": str(node.get("instance_id") or node.get("id") or ""),
                "name": _short(name, 120),
                "role": str(node.get("role") or "module"),
                "evidence_ids": list(node.get("evidence_ids") or [])[:4],
            })
        if not nodes:
            continue
        rows.append({
            "panel_label": str(panel.get("panel_label") or ""),
            "label": _short(panel.get("label") or panel.get("title"), 160),
            "nodes": nodes[:max_nodes],
        })
        if len(rows) >= max_panels:
            break
    return rows


def advisory_rule_rows(brief: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the advisory rule directives a brief carries, in a prompt-sized form."""

    source = brief if isinstance(brief, dict) else {}
    container = source.get("creator_rules") if isinstance(source.get("creator_rules"), dict) else {}
    rows: list[dict[str, Any]] = []
    for rule in container.get("rules") or []:
        if not isinstance(rule, dict) or str(rule.get("class") or "") != "advisory":
            continue
        directive = str(rule.get("directive") or "").strip()
        if not directive:
            continue
        provenance = rule.get("provenance") if isinstance(rule.get("provenance"), dict) else {}
        rows.append(
            {
                "rule_id": str(rule.get("rule_id") or ""),
                "source": str(provenance.get("source") or ""),
                "directive": directive,
            }
        )
    return rows[:12]


def compact_reference(profile: dict[str, Any] | None, exemplar: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return reusable visual tokens, excluding paper semantics."""
    source = profile if isinstance(profile, dict) else {}
    layout = source.get("layout") if isinstance(source.get("layout"), dict) else {}
    style = source.get("style") if isinstance(source.get("style"), dict) else {}
    template = source.get("reference_layout_template") if isinstance(source.get("reference_layout_template"), dict) else {}
    brief = exemplar if isinstance(exemplar, dict) else {}
    reference = {
        "layout": {k: layout.get(k) for k in ("aspect_ratio", "macro_pattern", "topology_hint")},
        "style": {k: style.get(k) for k in ("background_rgb", "palette", "line_style", "node_style", "icon_style")},
        "template": template,
        "adopt": list(brief.get("layout_rules") or [])[:8],
        "structure_rules": list(brief.get("structure_rules") or [])[:8],
        "color_rules": list(brief.get("color_rules") or [])[:8],
    }
    # Rule text used to stop at the density advisory, which reads only the numeric sub-keys, so
    # a rule list could be "injected" and still never reach a planning prompt. The directives
    # travel here now, and the key is left out entirely when there are none, so a run without
    # rules keeps exactly the prompt it had.
    rules = advisory_rule_rows(brief)
    if rules:
        reference["creator_rules"] = rules
    return reference


def compact_color_experience(exemplar: dict[str, Any] | None, *, max_recipes: int = 3) -> dict[str, Any]:
    """Return only the recipe evidence consumed by the color planner."""
    brief = exemplar if isinstance(exemplar, dict) else {}
    experience = brief.get("color_experience") if isinstance(brief.get("color_experience"), dict) else {}
    recipes = experience.get("selected_recipes") if isinstance(experience.get("selected_recipes"), list) else []
    return {
        "selected_recipes": [item for item in recipes[: max(0, min(3, int(max_recipes)))] if isinstance(item, dict)],
        "shared_patterns": list(experience.get("shared_patterns") or [])[:8],
        "candidate_harmonies": list(experience.get("candidate_harmonies") or [])[:3],
    }


def context_stats(prompt: str, *, image_count: int = 0, field_names: list[str] | None = None) -> dict[str, Any]:
    """Build common request telemetry without inspecting secret values."""
    chars = len(prompt or "")
    return {
        "prompt_chars": chars,
        "estimated_tokens": max(1, round(chars / 4)),
        "image_count": int(image_count),
        "context_fields": list(field_names or []),
    }


def json_compact(value: Any) -> str:
    """Stable compact JSON for prompt interpolation."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


__all__ = ["advisory_rule_rows", "compact_document_index", "compact_evidence", "compact_review", "compact_method", "compact_reference", "compact_color_experience", "context_stats", "json_compact"]
