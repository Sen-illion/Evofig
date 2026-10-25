"""Plan direct paper-to-PPTX layout and local Image2 assets."""

from __future__ import annotations

from typing import Any

from ..contracts.color_theme import resolve_role_style
from ..contracts.priority import priority_number

from ..contracts import (
    normalize_figure_layout_plan,
    normalize_icon_generation_plan,
    validate_figure_layout_plan,
    validate_icon_generation_plan,
)
from .global_structure import build_global_structure_plan
from .icon_style import reference_icon_rules, resolve_primary_object
from .panel_style import build_panel_style_plan


def _priority(value: Any) -> float:
    """Read one planned priority through the shared word-to-number table."""

    return priority_number(value)


def _preferred_area(value: Any) -> float:
    try:
        return max(0.24, min(0.70, float(value)))
    except (TypeError, ValueError):
        return 0.34


def structure_from_design_plan(
    design: dict[str, Any],
    *,
    aspect_ratio: str = "16:9",
    reference: str | None = None,
    reference_profile: dict[str, Any] | None = None,
    exemplar_brief: dict[str, Any] | None = None,
    color_theme: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a validated semantic design into existing layout/icon contracts."""
    nodes, icons = [], []
    panel_style = build_panel_style_plan(design.get("panels", []), reference_profile, exemplar_brief, color_theme)
    panel_colors = {item["panel_id"]: item for item in panel_style["panels"]}
    icon_rules = reference_icon_rules(exemplar_brief)
    for node in design.get("nodes", []):
        node_id = str(node.get("id"))
        is_container = str(node.get("kind") or "").casefold() == "container" or str(node.get("render_policy") or "").casefold() in {"panel_only", "ppt_native_container"}
        nodes.append(
            {
                **node,
                "label": node.get("label") or node_id,
                "object_type": "container_anchor" if is_container else "icon_with_label",
                "semantic_node_id": node.get("semantic_source_id"),
                "asset_id": None if is_container else f"icon_{node_id}",
                "panel_id": node.get("panel_id"),
                "panel_background_color": panel_colors.get(str(node.get("panel_id")), {}).get("fill_color_hex"),
                "background_policy": "transparent_panel_inherit",
                "visual_role": str(node.get("visual_role") or "standard_icon"),
                "visual_priority": _priority(node.get("visual_priority", node.get("importance", 0.5))),
                "complexity": str(node.get("complexity") or "medium"),
            }
        )
        if is_container:
            continue
        subject = str(node.get("label") or node_id)
        role_theme = resolve_role_style(color_theme or {}, node.get("role"))
        complexity_kind = "hero_visual" if str(node.get("visual_role") or "").casefold() in {"hero_visual", "large_visual", "wide_visual"} else "standard_icon"
        primary_object = resolve_primary_object(node, subject)
        icons.append(
            {
                "asset_id": f"icon_{node_id}",
                "semantic_node_ids": [node_id],
                "semantic_source_id": str(node.get("semantic_source_id") or node_id),
                "subject": subject,
                "visual_metaphor": primary_object,
                "primary_object": primary_object,
                "icon_grammar": "bounded_mechanism_visual" if complexity_kind == "hero_visual" else "academic_flat_symbol",
                "complexity_kind": complexity_kind,
                "supporting_object_limit": 3 if complexity_kind == "hero_visual" else 1,
                "prompt": f"academic framework-diagram icon for {subject}; primary object: {primary_object}; flat, minimal, publication-style, no text, formulas, arrows, or frame",
                "negative_prompt": "text, labels, formulas, arrows, connectors, watermark, logo, frame, full illustration, multi-object scene, poster, dashboard, photorealism, 3D render, decorative clutter",
                "must_not_include": [
                    "text",
                    "formulas",
                    "arrows",
                    "connectors",
                    "logo",
                    "watermark",
                ],
                "aspect_ratio": "1:1",
                "panel_id": node.get("panel_id"),
                "panel_background_color": panel_colors.get(str(node.get("panel_id")), {}).get("fill_color_hex"),
                "background_policy": "transparent_panel_inherit",
                "content_fill_target": 0.70 if complexity_kind == "standard_icon" else 0.84,
                "visual_role": str(node.get("visual_role") or "standard_icon"),
                "visual_priority": _priority(node.get("visual_priority", node.get("importance", 0.5))),
                "complexity": str(node.get("complexity") or "medium"),
                "preferred_aspect_ratio": str(node.get("preferred_aspect_ratio") or "1:1"),
                "preferred_area": _preferred_area(node.get("preferred_area")),
                "layout_mode": str(node.get("layout_mode") or "auto"),
                "prompt_fingerprint": str(node.get("prompt_fingerprint") or ""),
                "color_guidance": role_theme,
                "theme_name": (color_theme or {}).get("theme_name"),
                "stroke_style": "clean uniform medium outline",
                "fill_style": "flat limited pastel fills",
                "detail_policy": "recognizable at small size; one semantic symbol with minimal supporting marks",
                "reference_icon_rules": icon_rules,
            }
        )
    canvas = dict(design.get("canvas") or {})
    canvas["aspect_ratio"] = str(aspect_ratio)
    layout = normalize_figure_layout_plan(
        {
            "canvas": canvas or {"aspect_ratio": aspect_ratio, "margin": 0.04},
            "topology": design.get("topology") or "reference_guided_multiframe",
            "reading_order": [str(n.get("id")) for n in nodes],
            "panels": design.get("panels", []),
            "nodes": nodes,
            "connectors": design.get("connectors", []),
            "style_tokens": (reference_profile or {}).get("style", {}),
            "provenance": {"source": "vlm_design_plan", "reference": reference},
        }
    )
    icon_plan = normalize_icon_generation_plan(
        {
            "global_style": {
                "style": "reference-consistent scientific illustration",
                "background": "transparent_panel_inherit",
                "panel_style_plan": panel_style,
                "reference_icon_rules": icon_rules,
            },
            "icons": icons,
            "provenance": {"source": "design_plan", "reference": reference},
        }
    )
    return {
        "global_structure_plan": design,
        "segmentation_plan": {
            "contract": "rfs.segmentation_plan",
            "version": 1,
            "slots": [],
        },
        "figure_layout_plan": layout,
        "structure_candidates": [
            {"candidate_id": "candidate_00", "source": "vlm_design", "plan": layout}
        ],
        "icon_generation_plan": icon_plan,
        "validation": {
            "layout": validate_figure_layout_plan(layout),
            "icons": validate_icon_generation_plan(icon_plan),
        },
        "panel_style_plan": panel_style,
    }


def plan_paper_figure_structure(
    figure_specification: dict[str, Any],
    *,
    aspect_ratio: str = "16:9",
    reference: str | None = None,
    reference_profile: dict[str, Any] | None = None,
    exemplar_brief: dict[str, Any] | None = None,
) -> dict[str, Any]:
    global_plan, segmentation, upstream_validation = build_global_structure_plan(
        figure_specification, None, source="paper_evolve_editable"
    )
    nodes = []
    icons = []
    panel_style = build_panel_style_plan([], reference_profile, exemplar_brief)
    icon_rules = reference_icon_rules(exemplar_brief)
    for node in global_plan["nodes"]:
        node_id = str(node["id"])
        nodes.append(
            {
                **node,
                "object_type": "icon_with_label",
                "semantic_node_id": node_id,
                "asset_id": f"icon_{node_id}",
            }
        )
        style = ((reference_profile or {}).get("style") or {}).get(
            "icon_style"
        ) or "clean scientific illustration"
        icons.append(
            {
                "asset_id": f"icon_{node_id}",
                "semantic_node_ids": [node_id],
                "semantic_source_id": str(node.get("semantic_source_id") or node_id),
                "subject": node["label"],
                "visual_metaphor": resolve_primary_object(node, str(node["label"])),
                "primary_object": resolve_primary_object(node, str(node["label"])),
                "icon_grammar": "academic_flat_symbol",
                "complexity_kind": "standard_icon",
                "supporting_object_limit": 1,
                "prompt": f"publication-quality academic framework-diagram icon for {node['label']}; one simple primary symbol, flat fills, clean outline, no text or arrows",
                "negative_prompt": "text, labels, formulas, arrows, connectors, watermark, logo, frame, full illustration, multi-object scene, poster, dashboard, photorealism, 3D render, decorative clutter",
                "must_not_include": [
                    "text",
                    "formulas",
                    "arrows",
                    "connectors",
                    "logo",
                    "watermark",
                ],
                "aspect_ratio": "1:1",
                "background_policy": "transparent_panel_inherit",
                "content_fill_target": 0.70,
                "visual_role": str(node.get("visual_role") or "standard_icon"),
                "visual_priority": _priority(node.get("visual_priority", node.get("importance", 0.5))),
                "complexity": str(node.get("complexity") or "medium"),
                "preferred_aspect_ratio": str(node.get("preferred_aspect_ratio") or "1:1"),
                "preferred_area": _preferred_area(node.get("preferred_area")),
                "layout_mode": str(node.get("layout_mode") or "auto"),
                "prompt_fingerprint": str(node.get("prompt_fingerprint") or ""),
                "stroke_style": "clean uniform medium outline",
                "fill_style": "flat limited pastel fills",
                "detail_policy": "recognizable at small size; one semantic symbol with minimal supporting marks",
                "reference_icon_rules": icon_rules,
            }
        )
    reference_ratio = ((reference_profile or {}).get("layout") or {}).get(
        "aspect_ratio"
    )
    effective_ratio = (
        "16:9" if reference_ratio and float(reference_ratio) >= 1.35 else aspect_ratio
    )
    layout = normalize_figure_layout_plan(
        {
            "canvas": {"aspect_ratio": effective_ratio, "margin": 0.04},
            "topology": global_plan["topology"],
            "reading_order": global_plan["reading_order"],
            "nodes": nodes,
            "connectors": [
                {
                    "id": item["id"],
                    "source": item["source"],
                    "target": item["target"],
                    "type": item["type"],
                    "route": "orthogonal",
                    "source_port": "right",
                    "target_port": "left",
                }
                for item in global_plan["relations"]
            ],
            "style_tokens": (reference_profile or {}).get("style", {}),
            "provenance": {
                "reference": reference,
                "reference_status": (reference_profile or {}).get("status"),
                "source": "paper_evolve_editable",
            },
        }
    )
    icon_plan = normalize_icon_generation_plan(
        {
            "global_style": {
                "style": ((reference_profile or {}).get("style") or {}).get(
                    "icon_style", "clean scientific illustration"
                ),
                "background": "transparent_panel_inherit",
                "panel_style_plan": panel_style,
                "palette": ((reference_profile or {}).get("style") or {}).get(
                    "palette", []
                ),
                "reference_icon_rules": icon_rules,
            },
            "icons": icons,
            "provenance": {"source": "figure_layout_plan", "reference": reference},
        }
    )
    alternative = create_layout_candidate(layout, 2)
    candidates = [
        {"candidate_id": "candidate_00", "source": "reference_guided", "plan": layout},
        {
            "candidate_id": "candidate_01",
            "source": "semantic_alternative",
            "plan": alternative,
        },
    ]
    return {
        "global_structure_plan": global_plan,
        "segmentation_plan": segmentation,
        "figure_layout_plan": layout,
        "structure_candidates": candidates,
        "icon_generation_plan": icon_plan,
        "panel_style_plan": panel_style,
        "validation": {
            "upstream": upstream_validation,
            "layout": validate_figure_layout_plan(layout),
            "icons": validate_icon_generation_plan(icon_plan),
        },
    }


def create_layout_candidate(base: dict[str, Any], variant: int) -> dict[str, Any]:
    """Create a bounded deterministic layout alternative without changing semantics."""
    candidate = normalize_figure_layout_plan(base)
    nodes = candidate["nodes"]
    count = len(nodes)
    if count <= 1 or variant % 3 == 0:
        return candidate
    columns = 1 if variant % 3 == 1 and count <= 5 else min(3, count)
    rows = (count + columns - 1) // columns
    gap_x, gap_y = 0.035, 0.045
    width = (0.88 - gap_x * (columns - 1)) / columns
    height = min(0.24, (0.70 - gap_y * (rows - 1)) / max(rows, 1))
    for index, node in enumerate(nodes):
        column, row = index % columns, index // columns
        node["bbox_percent"] = {
            "x": round(0.06 + column * (width + gap_x), 4),
            "y": round(0.16 + row * (height + gap_y), 4),
            "w": round(width, 4),
            "h": round(height, 4),
        }
    candidate["topology"] = (
        "top_to_bottom_pipeline" if columns == 1 else "grid_pipeline"
    )
    candidate["provenance"] = {
        **candidate.get("provenance", {}),
        "candidate_variant": int(variant),
    }
    return candidate
