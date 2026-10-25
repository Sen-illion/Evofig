"""VLM planner for one role-aware color theme shared by icon and PPT stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts.color_theme import normalize_color_theme, validate_color_theme
from ..contracts.vlm_context import compact_color_experience, compact_method, compact_reference, context_stats, json_compact
from ..providers import call_vlm_json, resolve_vlm_model
from ..evaluation.color_plan_validation import validate_color_plan
from .prompt_assembly import render_patch_block


COLOR_THEME_PROMPT = """You are the color-system designer for an editable scientific framework figure.

Return exactly one non-empty JSON object matching the supplied rfs.color_theme schema. Return JSON only—no Markdown or commentary.

INPUTS
- METHOD: scientific content and semantic relationships.
- DESIGN: canvas, panels, nodes, their semantic roles, hierarchy and geometry.
- TEXT: editable text roles and placement.
- REFERENCE_STYLE: reusable color and hierarchy principles extracted from excellent scientific figures. It is guidance, not a palette to copy.
- COLOR_EXPERIENCE: 2–3 evidence-backed color recipes. Every supplied recipe must be explicitly adopted or rejected.
- PREVIOUS_THEME: the previous theme, if this is a targeted revision.
- COLOR_FEEDBACK: visual-review feedback, if available.

OBJECTIVE
Create a coherent, aesthetically pleasing and publication-appropriate color system that improves semantic hierarchy, stage distinction and readability. “Academic” and “restrained” do not mean that every panel should be gray. Avoid both monotonous all-neutral results and decorative rainbow coloring.

SEMANTIC COLOR PLANNING
1. Inspect every panel and node role appearing in DESIGN.
2. Build semantic color groups from the actual roles in this figure. Do not assume a fixed input/module/output role list.
3. Return a color specification for every actual node role. No role may silently fall back to a generic panel color.
4. Roles with the same scientific meaning should normally share a color family.
5. Roles with different meanings—such as input, generation, intermediate artifact, evaluation, filtering, feedback and output—should be visually distinguishable when that distinction improves scientific interpretation.
6. Do not assign different colors merely because objects have different IDs.

PANEL COLOR PLANNING
1. Return fill, border, header, header_text and body_text for every supplied panel ID.
2. Panel colors must follow their semantic function and relationship.
3. Semantically related panels may share a color family.
4. Major panels representing different scientific stages must remain distinguishable through hue, lightness, border or header treatment.
5. If multiple semantically different panels receive identical colors, provide an explicit rationale; otherwise revise the palette.
6. Large panel surfaces should normally use pale coordinated colors. Stronger saturation should be reserved for headers, borders and focal nodes.

PALETTE HARMONY
1. Select one dominant color family and at most two supporting families, plus limited semantic accents where scientifically useful.
2. State the intended harmony: analogous, complementary, split-complementary or neutral-with-accents.
3. Keep saturation and temperature consistent across panels and nodes.
4. Do not mix unrelated pastel colors merely to increase variety.
5. Avoid excessive gray unless the scientific hierarchy genuinely requires a neutral background.
6. Avoid pure black, pure red and highly saturated colors on large surfaces.

REFERENCE USAGE
Learn transferable principles from REFERENCE_STYLE: dominant/supporting/accent color proportions; warm–cool balance; lightness hierarchy; semantic emphasis; panel-versus-node contrast; and connector prominence. Do not mechanically copy exact colors, labels or paper-specific semantics.

COLOR EXPERIENCE ADOPTION
1. Inspect every recipe supplied in COLOR_EXPERIENCE.
2. Adopt only transferable palette relationships and role mappings compatible with the current DESIGN.
3. For every supplied recipe, record either an adopted entry with the applied current roles, or a rejected entry with a concrete incompatibility reason.
4. Exact reference HEX values may be reused when appropriate, but never copy paper-specific labels or semantic meanings.

CONNECTORS AND TEXT
1. Primary flow, secondary flow and feedback connectors must be distinguishable without competing with nodes.
2. Connector colors must remain visible against every crossed panel.
3. Normal text contrast must be at least 4.5:1.
4. Large text contrast must be at least 3:1.
5. Header text must remain readable against the header color.

TARGETED REVISION
When PREVIOUS_THEME and COLOR_FEEDBACK are provided, preserve color roles not implicated by the feedback; revise only the affected semantic groups; keep the overall palette identity stable; and do not replace the complete theme unless feedback explicitly requires it.

REQUIRED OUTPUT
{
  "theme_name": "...",
  "theme_intent": "...",
  "palette_strategy": {
    "dominant_family": "...",
    "supporting_families": ["..."],
    "harmony": "...",
    "rationale": "..."
  },
  "canvas_background": "#RRGGBB",
  "text": {"primary": "#RRGGBB", "secondary": "#RRGGBB", "muted": "#RRGGBB", "inverse": "#RRGGBB"},
  "connectors": {"primary": "#RRGGBB", "secondary": "#RRGGBB", "feedback": "#RRGGBB"},
  "panels": {
    "<every actual panel id>": {
      "fill": "#RRGGBB", "border": "#RRGGBB", "header": "#RRGGBB",
      "header_text": "#RRGGBB", "body_text": "#RRGGBB", "semantic_group": "..."
    }
  },
  "role_mapping": {"<every actual node role>": "<semantic color group>"},
  "roles": {
    "<every semantic color group>": {"fill": "#RRGGBB", "border": "#RRGGBB", "text": "#RRGGBB"}
  },
  "accents": ["#RRGGBB"],
  "usage_policy": {"rules": ["..."], "max_distinct_hues": 4},
  "experience_adoption": {
    "adopted": [{"recipe_id": "...", "principle": "...", "applied_to": ["panel:<id>", "role:<role>"]}],
    "rejected": [{"recipe_id": "...", "reason": "..."}]
  },
  "self_check": {
    "all_panel_ids_covered": true,
    "all_node_roles_covered": true,
    "semantically_distinct_panels_are_visible": true,
    "palette_is_harmonious": true,
    "not_all_neutral_without_reason": true,
    "contrast_requirements_met": true
  },
  "provenance": {"source": "vlm", "reference_style_used": true, "exemplar_rules_used": true}
}

Before returning, verify that every actual panel ID and every actual node role from DESIGN is covered.
"""


def _mix_with_white(value: str, ratio: float = 0.84) -> str:
    raw = str(value or "").strip().lstrip("#")
    if len(raw) != 6 or not all(char in "0123456789abcdefABCDEF" for char in raw):
        return "#F4F6F8"
    channels = [int(raw[i:i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(channel * (1 - ratio) + 255 * ratio) for channel in channels]
    return "#%02X%02X%02X" % tuple(mixed)


def _adapt_vlm_theme(raw: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    """Bind common generic VLM color output to concrete panel ids and roles."""
    adapted = dict(raw)
    generic_panel = raw.get("panel") if isinstance(raw.get("panel"), dict) else {}
    semantic_roles = raw.get("semantic_roles") if isinstance(raw.get("semantic_roles"), dict) else {}
    raw_panels = raw.get("panels")
    panel_keys = set(raw_panels) if isinstance(raw_panels, dict) else set()
    design_ids = {str(item.get("id")) for item in design.get("panels", []) if isinstance(item, dict) and item.get("id")}
    if generic_panel or (panel_keys and not panel_keys.intersection(design_ids)):
        generic_panel = generic_panel or (raw_panels.get("module") if isinstance(raw_panels, dict) else {}) or {}
        panels = {}
        for panel in design.get("panels", []):
            if not isinstance(panel, dict) or not panel.get("id"):
                continue
            role = str(panel.get("role") or "module")
            role_spec = semantic_roles.get(role) if isinstance(semantic_roles.get(role), dict) else {}
            source = role_spec or (raw_panels.get(role) if isinstance(raw_panels, dict) and isinstance(raw_panels.get(role), dict) else generic_panel)
            accent = str(role_spec.get("border") or semantic_roles.get(role) or source.get("border") or "#64748B")
            panels[str(panel["id"])] = {"fill": str(source.get("fill") or _mix_with_white(accent)), "border": accent, "header": str(source.get("header") or accent), "header_text": str(source.get("header_text") or "#FFFFFF"), "body_text": str(source.get("body_text") or (raw.get("text") or {}).get("primary") or "#193247")}
        adapted["panels"] = panels
    if not isinstance(raw.get("roles"), dict):
        primary_text = str((raw.get("text") or {}).get("primary") or "#193247")
        role_panels = raw_panels if isinstance(raw_panels, dict) else {}
        adapted_roles = {}
        for role in ("input", "module", "output"):
            role_spec = semantic_roles.get(role) if isinstance(semantic_roles.get(role), dict) else {}
            role_color = semantic_roles.get(role) if isinstance(semantic_roles.get(role), str) else None
            adapted_roles[role] = {"fill": str(role_spec.get("fill") or (role_panels.get(role) or {}).get("fill") or _mix_with_white(str(role_color or "#64748B"), 0.82)), "border": str(role_spec.get("border") or role_color or (role_panels.get(role) or {}).get("border") or "#64748B"), "text": str(role_spec.get("text") or primary_text)}
        adapted["roles"] = adapted_roles
    if isinstance(raw.get("accents"), dict):
        adapted["accents"] = [{"id": key, "color": value} for key, value in raw["accents"].items() if isinstance(value, str)]
    if isinstance(raw.get("usage_policy"), str):
        adapted["usage_policy"] = {"rules": [raw["usage_policy"]]}
    if isinstance(raw.get("provenance"), str):
        adapted["provenance"] = {"source": "vlm", "rationale": raw["provenance"]}
    return adapted


#: The deterministic palette used when the color planner is disabled or its
#: answer is rejected. It is measured from the pixels of the curated
#: excellent-figure library (``reference-figures/excellent``): the chromatic
#: mass there is a light blue (#DCEAF7/#BCD7EA/#9EC6D9), an amber
#: (#FEE28C/#FFEFCE), a soft green (#E3F0D8/#C3DEC5) and a rose
#: (#EFCBD6/#FFC6C6), over near-white paper with muted slate structure.
#: An earlier palette was derived from ``output/reference_learning_color_v1``
#: instead; those recipes kept only two to five colors per figure and were
#: then averaged across figures, which washed the saturation out, so this one
#: is built from the measured colors themselves. Each family keeps a
#: chromatic large surface (fill), a mid-tone border and a darker header whose
#: white title text clears the 3:1 large-text gate.
#: The library extractor has since been fixed to hold on to small saturated
#: accents (``color_recipe.EXTRACTOR_ID``), which feeds the colour planner; this
#: fallback stays independently measured so it cannot inherit a stale library.
REFERENCE_FALLBACK_FAMILIES: tuple[dict[str, str], ...] = (
    {"fill": "#DCEAF7", "border": "#9EC6D9", "header": "#63849A"},  # light blue (dominant)
    {"fill": "#E3F0D8", "border": "#A8C9A6", "header": "#668C67"},  # soft green
    {"fill": "#FCEFD2", "border": "#F0CE7E", "header": "#AC7D2C"},  # amber
    {"fill": "#F6DCE3", "border": "#E4B0BC", "header": "#AB6E7A"},  # rose
)

#: Body text shared by every fallback panel and node surface.
REFERENCE_FALLBACK_TEXT = "#2F3E48"

#: Node-role surfaces: near-white tints so a node reads as an object on the
#: tinted panel behind it, with the family tone as its outline.
REFERENCE_FALLBACK_ROLES: dict[str, dict[str, str]] = {
    "input": {"fill": "#EBF4FB", "border": "#63849A", "text": REFERENCE_FALLBACK_TEXT},
    "module": {"fill": "#EDF5EA", "border": "#668C67", "text": REFERENCE_FALLBACK_TEXT},
    "output": {"fill": "#FDF6E6", "border": "#AC7D2C", "text": REFERENCE_FALLBACK_TEXT},
}


def _fallback(panel_ids: list[str], node_roles: set[str] | None = None) -> dict[str, Any]:
    """Return the deterministic reference-measured palette for a design."""
    families = REFERENCE_FALLBACK_FAMILIES
    panels = {
        pid: {
            **families[i % len(families)],
            "header_text": "#FFFFFF",
            "body_text": REFERENCE_FALLBACK_TEXT,
        }
        for i, pid in enumerate(panel_ids)
    }
    role_mapping = {role: (role if role in REFERENCE_FALLBACK_ROLES else "module") for role in (node_roles or set())}
    return normalize_color_theme({
        "theme_name": "reference_measured_fallback",
        "theme_intent": "reference-measured palette for an editable scientific framework figure",
        "palette_strategy": {
            "dominant_family": "light blue",
            "supporting_families": ["soft green", "amber", "rose"],
            "harmony": "split-complementary",
            "rationale": "Measured from the pixels of the curated excellent-figure library: near-white paper, chromatic panel tints, muted slate structure.",
        },
        "canvas_background": "#F6F7F8",
        "text": {"primary": "#2F3E48", "secondary": "#5C6B74", "muted": "#8B959C", "inverse": "#FFFFFF"},
        "connectors": {"primary": "#66889D", "secondary": "#A8C2D2", "feedback": "#C08F3C"},
        "panels": panels,
        "role_mapping": role_mapping,
        "roles": {name: dict(style) for name, style in REFERENCE_FALLBACK_ROLES.items()},
        "accents": [],
        "usage_policy": {"surface_ratio_max": 0.75, "accent_ratio_max": 0.15, "max_distinct_hues": 6},
        "self_check": {
            "all_panel_ids_covered": True,
            "all_node_roles_covered": True,
            "semantically_distinct_panels_are_visible": True,
            "palette_is_harmonious": True,
            "not_all_neutral_without_reason": True,
            "contrast_requirements_met": True,
        },
    })


def plan_color_theme_vlm(method: dict[str, Any], design: dict[str, Any], text_plan: dict[str, Any], reference_profile: dict[str, Any] | None = None, exemplar_style_brief: dict[str, Any] | None = None, *, mode: str = "vlm", model: str | None = None, timeout: int = 180, retries: int = 1, previous_theme: dict[str, Any] | None = None, color_feedback: dict[str, Any] | None = None, prompt_patches: list[dict[str, Any]] | None = None):
    panel_ids = [str(item.get("id")) for item in design.get("panels", []) if isinstance(item, dict) and item.get("id")]
    node_roles = {str(item.get("role")).strip() for item in design.get("nodes", []) if isinstance(item, dict) and str(item.get("role") or "").strip()}
    color_experience = compact_color_experience(exemplar_style_brief)
    recipe_ids = {str(item.get("recipe_id")) for item in color_experience["selected_recipes"] if item.get("recipe_id")}
    if mode != "vlm":
        theme = _fallback(panel_ids, node_roles)
        return theme, {**validate_color_theme(theme, set(panel_ids), node_roles), "mode": "fallback", "fallback_used": True, "fallback_reason": "color theme VLM disabled", "raw_response": {}}
    base_prompt = (COLOR_THEME_PROMPT + "\nMETHOD=" + json_compact(compact_method(method)) + "\nDESIGN=" + json_compact({"canvas": design.get("canvas"), "panels": design.get("panels"), "nodes": design.get("nodes")}) + "\nTEXT=" + json_compact({"items": text_plan.get("items", [])}) + render_patch_block(prompt_patches) + "\nREFERENCE_STYLE=" + json_compact(compact_reference(reference_profile or {}, exemplar_style_brief)) + "\nCOLOR_EXPERIENCE=" + json_compact(color_experience) + "\nPREVIOUS_THEME=" + json_compact(previous_theme or {}) + "\nCOLOR_FEEDBACK=" + json_compact(color_feedback or {}))
    raw: dict[str, Any] | None = None
    theme = normalize_color_theme({})
    metadata: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = []
    validation: dict[str, Any] = {"ok": False, "errors": ["color VLM was not called"], "theme": theme}
    for attempt_index in range(3):
        prompt = base_prompt
        if attempt_index:
            prompt += "\nVALIDATION_FEEDBACK=" + json_compact({
                "instruction": "Return a complete corrected theme. Change only fields implicated by these failures.",
                "failed_items": validation.get("errors", [])[:12],
            })
        metadata = {}
        try:
            raw = call_vlm_json(prompt, [], model=resolve_vlm_model("RFS_COLOR_THEME_MODEL", explicit_model=model), timeout=timeout, retries=retries, call_metadata=metadata)
            theme = normalize_color_theme(_adapt_vlm_theme(raw, design))
            validation = validate_color_plan(
                theme,
                expected_panel_ids=set(panel_ids),
                expected_node_roles=node_roles,
                expected_recipe_ids=recipe_ids or None,
            )
            attempts.append({"attempt": attempt_index + 1, "ok": validation["ok"], "errors": validation["errors"], "provider": metadata})
            if validation["ok"]:
                return theme, {**validation, "mode": "vlm", "fallback_used": False, "raw_response": raw, "context": context_stats(prompt, image_count=0, field_names=["compact_method", "design_geometry", "text_geometry", "reference_style", "color_experience"]), "provider": metadata, "supplied_recipe_ids": sorted(recipe_ids), "attempts": attempts}
        except Exception as exc:
            validation = {"ok": False, "errors": [f"provider_or_parse_error: {exc}"], "theme": theme}
            attempts.append({"attempt": attempt_index + 1, "ok": False, "errors": validation["errors"], "provider": metadata})
    return theme, {
        **validation,
        "mode": "vlm",
        "fallback_used": False,
        "failure_reason": "; ".join(validation.get("errors") or ["unknown color planning failure"]),
        "raw_response": raw or {},
        "context": context_stats(base_prompt, image_count=0, field_names=["compact_method", "design_geometry", "text_geometry", "reference_style", "color_experience"]),
        "provider": metadata,
        "supplied_recipe_ids": sorted(recipe_ids),
        "attempts": attempts,
    }
