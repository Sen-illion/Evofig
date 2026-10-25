"""VLM semantic design planner returning a validated JSON contract."""

from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from ..contracts import design_plan_coverage, method_object_index, normalize_design_plan, validate_design_plan
from ..providers import call_vlm_json, resolve_vlm_model
from ..contracts.vlm_context import compact_method, compact_reference, context_stats, json_compact
from .density_advisory import build_density_advisory
from .structure_wireframe import render_structure_wireframe
from .canvas_constraints import fit_design_to_vertical_safe_area
from .panel_adaptation import adapt_design_panels
from .prompt_assembly import assemble_prompt, render_patch_block
from .graph_constraints import check_design_against_graph, compact_graph_context, graph_context as build_graph_context
from .panel_target import compact_panel_target


def _ground_connectors(plan: dict[str, Any], method: dict[str, Any]) -> dict[str, Any]:
    """Rebind VLM connector geometry to paper-authoritative relation content."""
    semantic_by_node = {str(item.get("id")): str(item.get("semantic_source_id") or item.get("id")) for item in plan.get("nodes", []) if isinstance(item, dict)}
    relation_by_pair = {
        (str(item.get("source")), str(item.get("target"))): item
        for item in method.get("relations", []) if isinstance(item, dict)
    }
    grounded = []
    for connector in plan.get("connectors", []):
        if not isinstance(connector, dict):
            continue
        pair = (semantic_by_node.get(str(connector.get("source")), str(connector.get("source"))), semantic_by_node.get(str(connector.get("target")), str(connector.get("target"))))
        relation = relation_by_pair.get(pair)
        grounded.append({**connector, **({"type": relation.get("type") or connector.get("type"), "label": relation.get("label") or "", "evidence_ids": relation.get("evidence_ids") or [], "label_evidence_ids": relation.get("label_evidence_ids") or relation.get("evidence_ids") or [], "paper_relation_id": relation.get("id"), "explanation_source": relation.get("explanation_source")} if relation else {})})
    return {**plan, "connectors": grounded}


def _complete_connectors_from_method(
    plan: dict[str, Any],
    method: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Add the paper's own relations that the design stage left unwired.

    The design stage reads the contract in contract-id space and states the
    connectors it decided to draw.  A dense framework figure names dozens of
    relations, and until now a plan that drew four of them was accepted: the
    grounding pass only rebinds the connectors that exist, and the required-edge
    list is built from the design itself, so nothing downstream could tell "the
    paper states four arrows" apart from "the model drew four of sixty-nine".

    Every relation whose both endpoints the design did place is therefore added
    here, deterministically, and stamped with the contract relation id so the
    grounding report can say which arrows the paper stated and the figure would
    otherwise not have drawn.
    """

    if not isinstance(plan, dict):
        return plan, []
    node_id_by_semantic: dict[str, str] = {}
    semantic_by_node: dict[str, str] = {}
    for item in plan.get("nodes", []) or []:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("id") or "").strip()
        semantic_id = str(item.get("semantic_source_id") or "").strip()
        if not node_id:
            continue
        semantic_by_node[node_id] = semantic_id or node_id
        if semantic_id:
            node_id_by_semantic.setdefault(semantic_id, node_id)
    existing: set[tuple[str, str]] = set()
    for connector in plan.get("connectors", []) or []:
        if not isinstance(connector, dict):
            continue
        source = semantic_by_node.get(str(connector.get("source")), str(connector.get("source")))
        target = semantic_by_node.get(str(connector.get("target")), str(connector.get("target")))
        if source and target:
            existing.add((source, target))
    connectors = list(plan.get("connectors") or [])
    added: list[dict[str, Any]] = []
    for index, relation in enumerate(method.get("relations") or []):
        if not isinstance(relation, dict):
            continue
        source = str(relation.get("source") or relation.get("source_id") or "").strip()
        target = str(relation.get("target") or relation.get("target_id") or "").strip()
        if not source or not target or source == target:
            continue
        if (source, target) in existing:
            continue
        source_node, target_node = node_id_by_semantic.get(source), node_id_by_semantic.get(target)
        if not source_node or not target_node or source_node == target_node:
            continue
        existing.add((source, target))
        connector = {
            "id": "connector_paper_" + str(index + 1).zfill(3),
            "source": source_node,
            "target": target_node,
            "type": str(relation.get("type") or relation.get("relation_type") or "data_flow"),
            "label": str(relation.get("label") or ""),
            "evidence_ids": [str(value) for value in (relation.get("evidence_ids") or []) if value],
            "paper_relation_id": str(relation.get("id") or ""),
            "paper_source": source,
            "paper_target": target,
            "added_by": "contract_relation_completion",
        }
        connectors.append(connector)
        added.append({
            "id": connector["id"],
            "source": source_node,
            "target": target_node,
            "paper_source": source,
            "paper_target": target,
            "paper_relation_id": connector["paper_relation_id"],
            "type": connector["type"],
        })
    if not added:
        return plan, []
    return {**plan, "connectors": connectors}, added


def _ground_and_complete_connectors(
    plan: dict[str, Any],
    method: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rebind a design's connectors to the contract, then wire the ones it omitted."""

    return _complete_connectors_from_method(_ground_connectors(plan, method), method)


def _lock_canvas_aspect_ratio(plan: dict[str, Any], aspect_ratio: str) -> dict[str, Any]:
    """Keep page geometry authoritative while preserving VLM margin choices."""
    canvas = dict(plan.get("canvas") or {})
    canvas["aspect_ratio"] = str(aspect_ratio)
    return {**plan, "canvas": canvas}


def _total_area(plan: dict[str, Any], field: str) -> float:
    """Return summed normalized area for a planned object collection."""
    total = 0.0
    for item in plan.get(field, []):
        box = item.get("bbox_percent") if isinstance(item, dict) else None
        if not isinstance(box, dict):
            continue
        try:
            total += max(0.0, float(box.get("w", 0))) * max(0.0, float(box.get("h", 0)))
        except (TypeError, ValueError):
            continue
    return total


def _is_substantive_density_replan(
    before: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Reject density gains created only by enlarging existing node boxes."""
    before_ids = {str(item.get("id")) for item in before.get("nodes", []) if isinstance(item, dict)}
    candidate_nodes = [item for item in candidate.get("nodes", []) if isinstance(item, dict)]
    evidenced_added_ids = [
        str(item.get("id"))
        for item in candidate_nodes
        if str(item.get("id")) not in before_ids and item.get("evidence_ids")
    ]
    before_panel_area = _total_area(before, "panels")
    candidate_panel_area = _total_area(candidate, "panels")
    panel_area_ratio = candidate_panel_area / max(before_panel_area, 1e-9)
    panel_compacted = panel_area_ratio <= 0.97
    substantive = panel_compacted or bool(evidenced_added_ids)
    return substantive, {
        "before_panel_area": round(before_panel_area, 4),
        "candidate_panel_area": round(candidate_panel_area, 4),
        "panel_area_ratio": round(panel_area_ratio, 4),
        "panel_compacted": panel_compacted,
        "evidenced_added_node_ids": evidenced_added_ids,
        "policy": "compact_sparse_panels_or_add_evidenced_payload",
    }


def _build_structure_prompt(
    method: dict[str, Any],
    scientific_hierarchy: dict[str, Any] | None,
    reference_profile: dict[str, Any],
    exemplar_style_brief: dict[str, Any] | None,
    *,
    aspect_ratio: str,
    content_top: float,
    content_bottom: float,
    prompt_patches: list[dict[str, Any]] | None = None,
    replan_feedback: str = "",
    graph_block: str = "",
    panel_block: str = "",
    evidence_structure: bool = True,
) -> str:
    """Build the compact, auditable structure-planning request."""
    instructions = (
        "You are the scientific figure structure designer. Return exactly one non-empty JSON object matching this schema: "
        "{canvas:{aspect_ratio,margin}, panels:[{id,title,role,bbox_percent:{x,y,w,h},node_ids}], "
        "nodes:[{id,semantic_source_id,label,panel_id:string|null,role,importance,detail_level,parent_id,scientific_importance,visualization_value,display_policy,bbox_percent:{x,y,w,h},evidence_ids}], "
        "connectors:[{id,source,target,type,label,evidence_ids}], groups:[], "
        "reference_adoption:{adopted:[],rejected:[]}, "
        "density_targets:{nodes,connectors,text_items,canvas_occupancy_min,panel_occupancy_min,max_empty_region_ratio,icon_area_ratio_min,icon_area_ratio_max}}. "
        "The canvas aspect_ratio is fixed by FIXED_CANVAS_ASPECT_RATIO and must match it exactly. "
        "FIXED_CONTENT_VERTICAL_BAND is a containment boundary, not a target panel height: keep every panel, node, group and connector path inside it, "
        "but never stretch a sparse panel merely to fill the band. The space above the band belongs to the editable global title and must remain empty. "
        "Size each panel from its actual scientific payload, title/header allowance and necessary connector lane. A panel containing only one or two compact nodes "
        "should normally be a compact card or band unless its evidenced hierarchy or relations require more space. Keep payload visually substantial inside its panel; "
        "unused safe-area space is not a reason to inflate a container. Every bbox_percent uses normalized x,y,width,height values in 0..1, never an array or x1,y1,x2,y2. "
        "The panels array must contain at least 2 panels and nodes at least 3 nodes. Set panel_id to an actual panel id for every node whose center lies inside that panel. "
        "A root-level boundary input, output, shared context or bridge may use panel_id:null only when it has no parent and its center lies outside every panel; never omit panel_id accidentally for a panel-contained node. "
        "Every METHOD object with must_show=true or should_render=true must be represented either by a node whose semantic_source_id is that object id, or by a same-id semantic panel when the object is an enclosing stage/framework. "
    )
    if evidence_structure:
        # The hierarchy is the paper's own candidate inventory, so this sentence
        # belongs with the block that carries it. A row that answers "does the
        # explicit structure earn its keep" never planned a hierarchy, and a
        # sentence pointing at an empty block is not the prompt that row means.
        instructions += (
            "Treat SCIENTIFIC_HIERARCHY as the authoritative candidate inventory: use evidenced sections as macro regions and evidenced submodules as nested nodes when their scientific_importance or visualization_value warrants display. "
        )
    instructions += (
        "Do not collapse a complex method into a few coarse boxes when renderable evidenced submodules exist. Keep parent_id/detail_level/display_policy on every selected node. "
        "Treat REFERENCE.reference_layout_template as an executable design prior: borrow useful zone proportions, internal patterns, hierarchy, density and flow types only when compatible with METHOD; never copy scientific labels or force its topology. "
        "Prefer compact nested layouts over oversized containers; target 0.65-0.88 meaningful canvas occupancy without inventing science. "
        "Use only semantic_source_id values present in METHOD. Do not return an empty object, markdown, prose or PPT code."
    )
    if graph_block:
        # The graph is the paper's own topology in contract-id space. It is an
        # evidence layer, so it belongs with SCIENTIFIC_HIERARCHY/METHOD and not
        # with the promoted patches, which are rules measured by earlier runs.
        instructions += (
            " SCIENTIFIC_GRAPH states this paper's own topology in the same contract-id space as METHOD: follow it when you place and connect nodes. "
            "Draw a feedback motif as a visible loop, a training_inference motif as separated training and inference stages, and a branch or merge motif "
            "with its fan-out or fan-in kept explicit. Every node or relation the graph marks required must appear. Never invent topology the graph does not state."
        )
    if panel_block:
        # A benchmark run is scored against the figure the paper printed, so the
        # target is an evidence layer too: this paper's own panels and their visible
        # labels, not a rule measured by earlier runs.
        instructions += (
            " FIGURE_TARGET_PANELS states the panels of the paper's own framework figure in the paper's own order, "
            "with the labels a reader can see in each. Follow it: keep its panel partitioning and its granularity, "
            "one design panel per target panel, and keep every label it lists. When it names a single panel, that "
            "panel is the whole guide and the canvas still carries its own panel grid inside it. Never invent a "
            "panel or a label the target does not state."
        )
    # The promoted patches sit below the schema and above the reference prior, which is
    # exactly where the assembly order puts them.
    return assemble_prompt(
        instructions
        + " FIXED_CANVAS_ASPECT_RATIO=" + str(aspect_ratio)
        + " FIXED_CONTENT_VERTICAL_BAND=" + json_compact([content_top, content_bottom])
        + (" SCIENTIFIC_HIERARCHY=" + json_compact(scientific_hierarchy or {}) if evidence_structure else "")
        + (" SCIENTIFIC_GRAPH=" + graph_block if graph_block else "")
        + (" FIGURE_TARGET_PANELS=" + panel_block if panel_block else "")
        + " METHOD=" + json_compact(compact_method(method))
        # A second attempt at the same paper is asked about the objects the first attempt
        # got wrong. It sits with the evidence rather than with the patches, because it
        # describes this run's own measurement and not a rule measured by earlier runs.
        + (" REPLAN_FEEDBACK=" + str(replan_feedback) if str(replan_feedback or "").strip() else ""),
        prompt_patches,
        reference_prior="VISUAL_PRIOR=" + json_compact(compact_reference(reference_profile, exemplar_style_brief)),
    )


_EMPTY_PLAN_RETRY_LIMIT = 1
_PLAN_REPAIR_LIMIT = 1
_PLAN_KEYS = ("panels", "nodes", "connectors")


def _plan_is_empty(raw: Any) -> bool:
    """Report whether the provider answered without any plan structure at all."""
    if not isinstance(raw, dict):
        return True
    return not any(key in raw for key in _PLAN_KEYS)


def _plan_content_ids(plan: dict[str, Any]) -> tuple[set[str], set[tuple[str, str]]]:
    """Collect the node ids and connector endpoint pairs a design actually carries."""
    node_ids = {
        str(item.get("id"))
        for item in plan.get("nodes", [])
        if isinstance(item, dict) and item.get("id")
    }
    connector_pairs = {
        (str(item.get("source")), str(item.get("target")))
        for item in plan.get("connectors", [])
        if isinstance(item, dict)
    }
    return node_ids, connector_pairs


def _preserves_plan_content(before: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Reject a repair that silently dropped scientific nodes or relations."""
    before_nodes, before_edges = _plan_content_ids(before)
    candidate_nodes, candidate_edges = _plan_content_ids(candidate)
    return before_nodes.issubset(candidate_nodes) and before_edges.issubset(candidate_edges)


def _build_plan_repair_prompt(
    plan: dict[str, Any],
    errors: list[str],
    *,
    aspect_ratio: str,
    content_top: float,
    content_bottom: float,
    prompt_patches: list[dict[str, Any]] | None = None,
    graph_requirements: str = "",
    missing_objects: str = "",
) -> str:
    """Send an invalid plan back to the model together with its own error list."""
    keep_rule = (
        "Keep every existing node id, semantic_source_id, panel id, connector source/target and scientific relation exactly as it is; add or delete no scientific content. "
    )
    # The default rule forbids every addition, which would make a missing required
    # node impossible to repair. Opening exactly the doors that are named -- the
    # scientific graph and the method contract -- keeps the rest of the plan frozen.
    additions: list[str] = []
    if graph_requirements:
        additions.append("a node or connector GRAPH_REQUIREMENTS names")
    if missing_objects:
        additions.append("a node MISSING_REQUIRED_METHOD_OBJECTS names")
    if additions:
        keep_rule = (
            "Keep every existing node id, semantic_source_id, panel id, connector source/target and scientific relation exactly as it is. "
            "The only content you may add is " + " and ".join(additions) + " and the design is missing; add nothing else and delete nothing. "
        )
    return (
        "You are repairing a scientific figure design plan you already produced. Return exactly one JSON object with the same schema. "
        + keep_rule
        + "Change only the geometry the problems in VALIDATION_ERRORS point at. Every bbox_percent x,y,w,h must be a normalized number with 0 <= x < 1, 0 <= y < 1, "
        "w > 0, h > 0, x + w <= 1 and y + h <= 1, while staying inside FIXED_CONTENT_VERTICAL_BAND and keeping panels readable and non-overlapping. "
        "Do not return an empty object, markdown, prose or PPT code."
        + render_patch_block(prompt_patches)
        + " FIXED_CANVAS_ASPECT_RATIO=" + str(aspect_ratio)
        + " FIXED_CONTENT_VERTICAL_BAND=" + json_compact([content_top, content_bottom])
        + (" GRAPH_REQUIREMENTS=" + graph_requirements if graph_requirements else "")
        + (" MISSING_REQUIRED_METHOD_OBJECTS=" + missing_objects if missing_objects else "")
        + " VALIDATION_ERRORS=" + json_compact(errors)
        + " CURRENT_PLAN=" + json_compact(plan)
    )


def _missing_required_objects_block(plan: dict[str, Any], method: dict[str, Any] | None) -> str:
    """Name the method objects a design still owes, in the repair prompt's own words.

    An empty string means the design already carries every required object, so a
    prompt built without a shortfall stays byte-identical to one a fully covered
    design sends.
    """
    coverage = design_plan_coverage(plan, method)
    if not coverage["missing_ids"]:
        return ""
    return json_compact([
        {"id": object_id, "label": label}
        for object_id, label in zip(coverage["missing_ids"], coverage["missing_labels"])
    ])


def _covers_more_required_content(
    before: dict[str, Any],
    candidate: dict[str, Any],
    method: dict[str, Any] | None,
) -> bool:
    """True when a repair drew more of the required method content, even if not all.

    A single plan repair cannot be asked to close every gap in one response, so a
    candidate that drew more required objects is an improvement the caller keeps
    instead of a guess the caller throws away for not being complete.
    """
    if method is None:
        return False
    return design_plan_coverage(candidate, method)["coverage"] > design_plan_coverage(before, method)["coverage"]



#: The margin an object the contract draws keeps from its panel edge. It is the margin the drawn
#: nodes keep and the separation a gutter rule asks for, so an added node lands in the room a
#: separation pass would leave it anyway.
COMPLETION_MARGIN = 0.012
#: Sizes an added node is tried at, as shares of the size its siblings were drawn at. A panel with
#: little room left takes the object at a smaller size rather than not at all.
COMPLETION_SIZE_STEPS = (1.0, 0.85, 0.7, 0.55, 0.45)
#: The size an added node takes in a panel whose own nodes carry no readable box.
COMPLETION_DEFAULT_SIZE = (0.10, 0.07)
#: The grid a free position inside a panel is looked for on, in canvas percent.
COMPLETION_GRID_STEP = 0.01


def _readable_box(box: Any) -> bool:
    """Whether a box carries the four numbers a design plan rectangle is read from."""

    if not isinstance(box, dict):
        return False
    for key in ("x", "y", "w", "h"):
        value = box.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
    return float(box["w"]) > 0 and float(box["h"]) > 0


def _apart(a: dict[str, Any], b: dict[str, Any], gap: float) -> bool:
    """Whether two boxes stand at least ``gap`` apart on some axis."""

    return (
        float(a["x"]) + float(a["w"]) + gap <= float(b["x"])
        or float(b["x"]) + float(b["w"]) + gap <= float(a["x"])
        or float(a["y"]) + float(a["h"]) + gap <= float(b["y"])
        or float(b["y"]) + float(b["h"]) + gap <= float(a["y"])
    )


def _free_slot(
    panel: dict[str, Any],
    occupied: list[dict[str, Any]],
    width: float,
    height: float,
) -> dict[str, float] | None:
    """The first position on a grid inside a panel where a box of that size stands clear.

    The scan runs row-major, so the same panel and the same boxes always give the same position,
    and every candidate keeps the panel margin on all four sides: the margin the drawn nodes keep,
    which is also the separation a gutter rule asks for.
    """

    left = float(panel["x"]) + COMPLETION_MARGIN
    top = float(panel["y"]) + COMPLETION_MARGIN
    last_x = float(panel["x"]) + float(panel["w"]) - COMPLETION_MARGIN - width
    last_y = float(panel["y"]) + float(panel["h"]) - COMPLETION_MARGIN - height
    if last_x < left or last_y < top:
        return None
    rows = int(round((last_y - top) / COMPLETION_GRID_STEP))
    columns = int(round((last_x - left) / COMPLETION_GRID_STEP))
    for row in range(rows + 1):
        for column in range(columns + 1):
            candidate = {
                "x": round(left + column * COMPLETION_GRID_STEP, 4),
                "y": round(top + row * COMPLETION_GRID_STEP, 4),
                "w": round(width, 4),
                "h": round(height, 4),
            }
            if all(_apart(candidate, other, COMPLETION_MARGIN) for other in occupied):
                return candidate
    return None


def _completion_size(sizes: list[tuple[float, float]], panel: dict[str, Any]) -> tuple[float, float]:
    """The size a panel's own nodes were drawn at, kept inside a third of the panel."""

    if sizes:
        widths = sorted(size[0] for size in sizes)
        heights = sorted(size[1] for size in sizes)
        width, height = widths[len(widths) // 2], heights[len(heights) // 2]
    else:
        width, height = COMPLETION_DEFAULT_SIZE
    return (
        max(0.03, min(float(width), float(panel["w"]) / 3)),
        max(0.02, min(float(height), float(panel["h"]) / 3)),
    )


def _completion_panel(
    panel_ids: list[str],
    fields_by_panel: dict[str, list[str]],
    field: str,
) -> str:
    """The panel a missing object belongs in: the one that draws its kind, else the emptiest."""

    best_id, best_score = panel_ids[0], None
    for panel_id in panel_ids:
        drawn = fields_by_panel.get(panel_id, [])
        same = sum(1 for item in drawn if item and item == field)
        score = (same, -len(drawn))
        if best_score is None or score > best_score:
            best_id, best_score = panel_id, score
    return best_id


def _complete_nodes_from_method(
    plan: dict[str, Any],
    method: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Draw the required method objects the design stage never drew, as nodes of their own.

    The coverage measurement could already say a design carried 24 of the 40 objects a paper marks
    must_show or should_render, but nothing downstream could act on what it said: the missing ids
    were only ever quoted back at the model in a repair prompt, and a run that plans in bulk -- and
    the frozen layout of a controlled pair -- deliver the plan they started from either way, so a
    figure carrying 60% of the method was indistinguishable from one carrying all of it.

    Every object still missing once the repairs have had their turn is therefore drawn here: named
    and evidenced from the contract, carrying the object's own id as both node id and
    semantic_source_id, parented to the panel that already draws objects of its field, and placed in
    the room that panel has left. The nodes the model drew keep their geometry and their relations
    untouched; an object no free slot could be found for is still drawn, at the smallest size tried,
    and reported, so a shortfall is visible instead of silent.
    """

    if not isinstance(plan, dict):
        return plan, [], []
    coverage = design_plan_coverage(plan, method)
    if not coverage["missing_ids"]:
        return plan, [], []
    index = method_object_index(method)
    nodes = [item for item in plan.get("nodes", []) or [] if isinstance(item, dict)]
    panel_boxes = {
        str(item.get("id")): item.get("bbox_percent")
        for item in plan.get("panels", []) or []
        if isinstance(item, dict) and _readable_box(item.get("bbox_percent"))
    }
    if not panel_boxes:
        return plan, [], list(coverage["missing_ids"])
    fields_by_panel: dict[str, list[str]] = {panel_id: [] for panel_id in panel_boxes}
    boxes_by_panel: dict[str, list[dict[str, Any]]] = {panel_id: [] for panel_id in panel_boxes}
    sizes_by_panel: dict[str, list[tuple[float, float]]] = {panel_id: [] for panel_id in panel_boxes}
    for node in nodes:
        panel_id = str(node.get("panel_id") or "")
        if panel_id not in panel_boxes:
            continue
        drawn = index.get(str(node.get("semantic_source_id") or "").strip())
        fields_by_panel[panel_id].append(str(drawn.get("field") or "") if drawn else "")
        box = node.get("bbox_percent")
        if _readable_box(box):
            boxes_by_panel[panel_id].append(box)
            sizes_by_panel[panel_id].append((float(box["w"]), float(box["h"])))
    additions: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    crowded: list[str] = []
    for object_id in coverage["missing_ids"]:
        info = index.get(object_id) or {"field": "", "label": object_id, "role": "module", "evidence_ids": []}
        field = str(info.get("field") or "")
        panel_id = _completion_panel(list(panel_boxes), fields_by_panel, field)
        panel = panel_boxes[panel_id]
        base_width, base_height = _completion_size(sizes_by_panel[panel_id], panel)
        slot = None
        share = COMPLETION_SIZE_STEPS[-1]
        for candidate_share in COMPLETION_SIZE_STEPS:
            slot = _free_slot(
                panel,
                boxes_by_panel[panel_id],
                round(base_width * candidate_share, 4),
                round(base_height * candidate_share, 4),
            )
            if slot is not None:
                share = candidate_share
                break
        if slot is None:
            slot = {
                "x": round(float(panel["x"]) + COMPLETION_MARGIN, 4),
                "y": round(float(panel["y"]) + COMPLETION_MARGIN, 4),
                "w": round(base_width * COMPLETION_SIZE_STEPS[-1], 4),
                "h": round(base_height * COMPLETION_SIZE_STEPS[-1], 4),
            }
            crowded.append(object_id)
        boxes_by_panel[panel_id].append(slot)
        fields_by_panel[panel_id].append(field)
        node = {
            "id": object_id,
            "semantic_source_id": object_id,
            "label": str(info.get("label") or object_id),
            "panel_id": panel_id,
            "role": str(info.get("role") or "module"),
            "importance": "medium",
            "detail_level": "module",
            "parent_id": None,
            "scientific_importance": 0.6,
            "visualization_value": 0.6,
            "display_policy": "solid",
            "bbox_percent": slot,
            "evidence_ids": list(info.get("evidence_ids") or []),
            "added_by": "contract_node_completion",
        }
        additions.append(node)
        added.append({
            "id": object_id,
            "label": node["label"],
            "panel_id": panel_id,
            "field": field,
            "size_share": share,
            "crowded": object_id in crowded,
        })
    return {**plan, "nodes": [*nodes, *additions]}, added, crowded


def _attempt_density_replan(
    plan: dict[str, Any],
    density: dict[str, Any],
    method: dict[str, Any],
    reference_profile: dict[str, Any],
    exemplar_style_brief: dict[str, Any] | None,
    reference: str | Path | None,
    wireframe_out: str | Path | None,
    model: str | None,
    timeout: int,
    retries: int,
    aspect_ratio: str,
    content_top: float,
    content_bottom: float,
    scientific_hierarchy: dict[str, Any] | None = None,
    prompt_patches: list[dict[str, Any]] | None = None,
    graph_block: str = "",
    panel_block: str = "",
    evidence_structure: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str | None]:
    """Run bounded density reflow for the VLM design plan."""
    if density.get("status") != "replan_required":
        return plan, density, [], None
    wireframe_path = str(wireframe_out or (Path.cwd() / "structure_wireframe.png"))
    render_structure_wireframe(plan, wireframe_path)
    base_ids = {str(n.get("id")) for n in plan.get("nodes", [])}
    base_edges = {(str(e.get("source")), str(e.get("target"))) for e in plan.get("connectors", [])}
    method_objects = {
        str(item.get("id"))
        for field in ("inputs", "outputs", "modules", "stages", "innovations", "optional_details")
        for item in method.get(field, []) if isinstance(method.get(field), list)
        if isinstance(item, dict) and item.get("id")
    }
    add_rule = (
        "You may add only evidenced submodules from SCIENTIFIC_HIERARCHY/METHOD "
        if evidence_structure
        else "You may add only evidenced submodules from METHOD "
    )
    missing_objects_block = _missing_required_objects_block(plan, method)
    repair_prompt = (
        "You are compacting an existing scientific figure structure. Return exactly one JSON object with the same schema. "
        "Preserve every existing node id, semantic_source_id, connector source/target and scientific relation. "
        + add_rule
        + "when they have high scientific and visualization value; never invent modules or delete required relations. "
        "Resolve every DENSITY.sparse_panels item by fitting its panel around the union of its real payload, header and necessary connector lane. Tighten accidental gutters and enlarge a node only when its content or hierarchy warrants that size; "
        "do not blindly scale all nodes or fill space with decoration. Preserve readable text and connector lanes. CURRENT_PLAN="
        + json_compact(plan) + " DENSITY=" + json_compact(density)
        + (" SCIENTIFIC_HIERARCHY=" + json_compact(scientific_hierarchy or {}) if evidence_structure else "")
        + " METHOD_OBJECT_IDS=" + json_compact(sorted(method_objects))
        + (" MISSING_REQUIRED_METHOD_OBJECTS=" + missing_objects_block if missing_objects_block else "")
        + (" SCIENTIFIC_GRAPH=" + graph_block if graph_block else "")
        + (" FIGURE_TARGET_PANELS=" + panel_block if panel_block else "")
        + render_patch_block(prompt_patches)
        + " VISUAL_PRIOR=" + json_compact(compact_reference(reference_profile, exemplar_style_brief))
    )
    attempts: list[dict[str, Any]] = []
    for attempt in range(2):
        try:
            candidate_raw = call_vlm_json(
                repair_prompt,
                [wireframe_path] if Path(wireframe_path).exists() else ([reference] if reference else []),
                model=resolve_vlm_model("RFS_STRUCTURE_MODEL", explicit_model=model),
                timeout=timeout,
                retries=retries,
                call_metadata={},
            )
            candidate, safe_area = fit_design_to_vertical_safe_area(
                _lock_canvas_aspect_ratio(normalize_design_plan(candidate_raw), aspect_ratio),
                top=content_top,
                bottom=content_bottom,
            )
            candidate, _candidate_completed = _ground_and_complete_connectors(candidate, method)
            candidate_validation = validate_design_plan(candidate, method)
            candidate_adaptation: dict[str, Any] = {}
            if candidate_validation.get("ok"):
                candidate, candidate_adaptation = adapt_design_panels(candidate)
                candidate_validation = validate_design_plan(candidate, method)
            stage4_manifest = (exemplar_style_brief or {}).get("stage4_manifest") if isinstance(exemplar_style_brief, dict) else None
            candidate_density = build_density_advisory(candidate, stage4_manifest)
            candidate_ids = {str(n.get("id")) for n in candidate.get("nodes", [])}
            invalid_added = [
                str(n.get("semantic_source_id") or n.get("id"))
                for n in candidate.get("nodes", [])
                if str(n.get("id")) not in base_ids and str(n.get("semantic_source_id") or n.get("id")) not in method_objects
            ]
            missing_added_evidence = [
                str(n.get("id")) for n in candidate.get("nodes", [])
                if str(n.get("id")) not in base_ids and not n.get("evidence_ids")
            ]
            candidate_edges = {(str(e.get("source")), str(e.get("target"))) for e in candidate.get("connectors", [])}
            substantive, substantive_report = _is_substantive_density_replan(plan, candidate)
            # A candidate is denser when it cleared the sparse-panel gate or when it
            # genuinely raised occupancy. Asking for both at once threw away every
            # reflow that drew more content but still read sparse, which is how a
            # thirteen-object plan was delivered while two paid repairs were discarded.
            denser = (
                candidate_density.get("status") != "replan_required"
                or candidate_density.get("node_occupancy_estimate", 0)
                > density.get("node_occupancy_estimate", 0)
            )
            accepted = (
                bool(candidate_validation.get("ok"))
                and not invalid_added
                and not missing_added_evidence
                and base_ids.issubset(candidate_ids)
                and base_edges.issubset(candidate_edges)
                and denser
                and substantive
            )
            attempts.append({"attempt": attempt + 1, "validation": candidate_validation, "density": candidate_density, "canvas_safe_area": safe_area, "panel_adaptation": candidate_adaptation, "substantive_change": substantive_report, "accepted": accepted, "invalid_added_object_ids": invalid_added, "missing_added_evidence_ids": missing_added_evidence})
            if accepted:
                return candidate, candidate_density, attempts, wireframe_path
        except Exception as repair_exc:
            attempts.append({"attempt": attempt + 1, "accepted": False, "error": str(repair_exc), "error_type": type(repair_exc).__name__})
    return plan, density, attempts, wireframe_path


def _graph_requirements(design: dict[str, Any], context: dict[str, Any], *, enabled: bool) -> str:
    """Name the required nodes and relations a design is missing, for the repair prompt.

    An empty string means either the switch is off or the design already carries
    every required item, so the repair prompt stays byte-identical to the one a
    run without the switch sends.
    """

    if not enabled or not context:
        return ""
    check = check_design_against_graph(design, context)
    if check["ok"]:
        return ""
    return json_compact({
        "missing_nodes": check["missing_required_node_ids"],
        "missing_edges": check["missing_required_edge_ids"],
    })


def plan_structure_vlm(
    method: dict[str, Any],
    reference_profile: dict[str, Any],
    reference: str | Path | None,
    *,
    exemplar_style_brief: dict[str, Any] | None = None,
    model: str | None = None,
    timeout: int = 180,
    retries: int = 1,
    wireframe_out: str | Path | None = None,
    scientific_hierarchy: dict[str, Any] | None = None,
    aspect_ratio: str = "16:9",
    content_top: float = 0.11,
    content_bottom: float = 0.96,
    initial_plan: dict[str, Any] | None = None,
    frozen_layout: bool = False,
    bulk_figures: bool = False,
    prompt_patches: list[dict[str, Any]] | None = None,
    replan_feedback: str = "",
    graph_context: bool = False,
    evidence_structure: bool = True,
    figure_target: str = "overview",
    panel_target: dict[str, Any] | None = None,
):
    """Plan one paper's figure structure, optionally reading its own graph.

    ``graph_context`` is computed from the method being planned and not handed in
    from upstream: an upstream retry plans a different method, and a context
    built from the earlier one would describe a paper this call is not drawing.

    ``evidence_structure`` is the switch a row that answers "does the explicit
    structure earn its keep" turns off. It drops the hierarchy and the graph
    together, because the row that never built either has neither to give.

    ``figure_target`` names which figure the design is aimed at. ``overview`` is
    the product's own reading of the method; ``paper`` follows the panels the
    paper printed, which is what a public benchmark scores. The target is an
    instruction and never a gate: a design that misses a label is measured by the
    caller and delivered anyway.
    """

    if not evidence_structure:
        graph_context = False
    resolved_graph_context = build_graph_context(method) if graph_context else {}
    # A dense framework figure carries one relation per printed arrow, so the block
    # has to be able to state them: the budget grows with the graph's own edge count
    # and is capped so a pathological contract cannot blow up the request.
    graph_edges = int(((resolved_graph_context.get("counts") or {}).get("edge_count") or 0))
    graph_block = (
        compact_graph_context(resolved_graph_context, max_chars=min(8000, max(2400, 90 * graph_edges)))
        if graph_context
        else ""
    )
    panel_block = compact_panel_target(panel_target or {}) if figure_target == "paper" else ""
    prompt = _build_structure_prompt(
        method,
        scientific_hierarchy,
        reference_profile,
        exemplar_style_brief,
        aspect_ratio=aspect_ratio,
        content_top=content_top,
        content_bottom=content_bottom,
        prompt_patches=prompt_patches,
        replan_feedback=replan_feedback,
        graph_block=graph_block,
        panel_block=panel_block,
        evidence_structure=evidence_structure,
    )
    raw = None
    provider_metadata: dict[str, Any] = {}
    replayed = initial_plan is not None
    empty_plan_retry_count = 0
    if replayed:
        raw = deepcopy(initial_plan)
        provider_metadata.update({"attempts": 0, "success": True, "replayed": True})
    else:
        for attempt in range(_EMPTY_PLAN_RETRY_LIMIT + 1):
            raw = call_vlm_json(
                prompt,
                [reference] if reference else [],
                model=resolve_vlm_model("RFS_STRUCTURE_MODEL", explicit_model=model),
                timeout=timeout,
                retries=retries,
                call_metadata=provider_metadata,
            )
            if not _plan_is_empty(raw):
                break
            empty_plan_retry_count = attempt + 1
        if _plan_is_empty(raw):
            raise RuntimeError(
                "VLM structure planning returned an empty design plan after "
                + str(_EMPTY_PLAN_RETRY_LIMIT + 1) + " attempts; no figure can be built"
            )
    # The response the figure is actually built from. A plan repair replaces the first response
    # with the one the model returned when it was shown the contract violations, and every artifact
    # below is a deterministic function of whichever response won, so a replay that re-derived the
    # figure from the first response alone would start from a figure this arm never drew.
    seed_response = raw
    plan, initial_safe_area = fit_design_to_vertical_safe_area(
        _lock_canvas_aspect_ratio(normalize_design_plan(raw), aspect_ratio),
        top=content_top,
        bottom=content_bottom,
    )
    plan, _completed = _ground_and_complete_connectors(plan, method)
    validation = validate_design_plan(plan, method)

    plan_repair_used = False
    plan_repair_attempts: list[dict[str, Any]] = []
    plan_repair_reason: list[str] = []
    repaired_safe_area: dict[str, Any] | None = None
    plan_repair_skipped = bool(bulk_figures and not validation.get("ok") and not replayed)
    if not bulk_figures and not validation.get("ok") and not replayed:
        plan_repair_reason = list(validation.get("errors") or [])
        repair_prompt = _build_plan_repair_prompt(
            plan,
            plan_repair_reason,
            aspect_ratio=aspect_ratio,
            content_top=content_top,
            content_bottom=content_bottom,
            prompt_patches=prompt_patches,
            graph_requirements=_graph_requirements(plan, resolved_graph_context, enabled=graph_context),
            missing_objects=_missing_required_objects_block(plan, method),
        )
        for attempt in range(_PLAN_REPAIR_LIMIT):
            plan_repair_used = True
            try:
                candidate_raw = call_vlm_json(
                    repair_prompt,
                    [reference] if reference else [],
                    model=resolve_vlm_model("RFS_STRUCTURE_MODEL", explicit_model=model),
                    timeout=timeout,
                    retries=retries,
                    call_metadata={},
                )
            except Exception as repair_exc:
                plan_repair_attempts.append({"attempt": attempt + 1, "accepted": False, "error": str(repair_exc), "error_type": type(repair_exc).__name__})
                break
            if _plan_is_empty(candidate_raw):
                plan_repair_attempts.append({"attempt": attempt + 1, "accepted": False, "error": "repair response carried no design plan", "error_type": "EmptyRepairPlan"})
                break
            candidate, candidate_safe_area = fit_design_to_vertical_safe_area(
                _lock_canvas_aspect_ratio(normalize_design_plan(candidate_raw), aspect_ratio),
                top=content_top,
                bottom=content_bottom,
            )
            candidate, _candidate_completed = _ground_and_complete_connectors(candidate, method)
            candidate_validation = validate_design_plan(candidate, method)
            preserved = _preserves_plan_content(plan, candidate)
            candidate_errors = list(candidate_validation.get("errors") or [])
            repaired = bool(candidate_validation.get("ok")) and preserved
            fewer_errors = preserved and len(candidate_errors) < len(plan_repair_reason)
            added_coverage = preserved and _covers_more_required_content(plan, candidate, method)
            plan_repair_attempts.append({
                "attempt": attempt + 1,
                "validation": candidate_validation,
                "preserves_plan_content": preserved,
                "error_count": len(candidate_errors),
                "covered_more_required_content": bool(added_coverage),
                "accepted": bool(repaired or fewer_errors or added_coverage),
                "canvas_safe_area": candidate_safe_area,
            })
            if repaired or fewer_errors or added_coverage:
                plan = candidate
                validation = candidate_validation
                repaired_safe_area = candidate_safe_area
                seed_response = candidate_raw
                if repaired:
                    break
    # A plan that still violates the contract is carried forward with its real
    # scientific content instead of being replaced by invented geometry; the
    # report states exactly what stayed unresolved.
    plan, initial_adaptation = adapt_design_panels(plan)
    density = build_density_advisory(plan, (exemplar_style_brief or {}).get("stage4_manifest") if isinstance(exemplar_style_brief, dict) else None)
    if bulk_figures or frozen_layout:
        # A bulk corpus run wants a figure per paper, not a second opinion about
        # a thin panel: this is up to two paid calls that a rejected replan
        # throws away, which the accepted run of 2025.emnlp-main.956 spent while
        # still delivering the plan it started from. The local panel adaptation
        # above already fits the stages to their payload.
        #
        # A frozen layout wants the same thing for a stronger reason: a fresh
        # model call would hand the two arms of a controlled comparison two
        # different starting layouts, and their difference would then read as
        # one the experience caused.
        replan_attempts: list[dict[str, Any]] = []
        wireframe_path = None
        density_replan_skipped = True
    else:
        plan, density, replan_attempts, wireframe_path = _attempt_density_replan(
            plan, density, method, reference_profile, exemplar_style_brief, reference,
            wireframe_out, model, timeout, retries, aspect_ratio, content_top, content_bottom,
            scientific_hierarchy, prompt_patches, graph_block,
            panel_block, evidence_structure,
        )
        density_replan_skipped = False
    plan, final_safe_area = fit_design_to_vertical_safe_area(plan, top=content_top, bottom=content_bottom)
    # The model gets the first say about which objects the figure carries, and the coverage check
    # has already asked it for every one it left out. What the repairs could not close is drawn
    # here rather than delivered short: carrying every required object is the contract, not an
    # aspiration the report apologises for.
    plan, completed_objects, crowded_objects = _complete_nodes_from_method(plan, method)
    accepted_attempt = next((item for item in replan_attempts if item.get("accepted")), None)
    final_adaptation = (accepted_attempt or {}).get("panel_adaptation") or initial_adaptation
    validation = validate_design_plan(plan, method)
    continued_with_invalid_plan = not bool(validation.get("ok"))
    unrepaired_validation_errors = list(validation.get("errors") or [])
    method_coverage = design_plan_coverage(plan, method)
    return plan, {
        **validation,
        "mode": "replay_seed" if replayed else "vlm",
        "initial_plan_replayed": replayed,
        "fallback_used": continued_with_invalid_plan,
        "empty_plan_retry_count": empty_plan_retry_count,
        "plan_repair_used": plan_repair_used,
        "plan_repair_attempts": plan_repair_attempts,
        "plan_repair_reason": plan_repair_reason,
        "bulk_figures": bool(bulk_figures),
        "plan_repair_skipped": plan_repair_skipped,
        "density_replan_skipped": density_replan_skipped,
        "unrepaired_validation_errors": unrepaired_validation_errors,
        "continued_with_invalid_plan": continued_with_invalid_plan,
        "method_coverage": method_coverage,
        "method_coverage_ok": bool(method_coverage["coverage"] >= 1.0),
        "method_object_completion": {
            "added_count": len(completed_objects),
            "added": completed_objects,
            "crowded_ids": crowded_objects,
        },
        "raw_response": raw,
        "seed_response": seed_response,
        "context": context_stats(prompt, image_count=1 if reference else 0, field_names=["compact_method", "compact_reference"] + (["scientific_graph"] if graph_context else []) + (["figure_target_panels"] if panel_block else [])),
        "figure_target": figure_target,
        "connector_completion": {
            "method_relation_count": len(method.get("relations") or []),
            "added_count": sum(
                1 for item in plan.get("connectors", [])
                if isinstance(item, dict) and item.get("added_by") == "contract_relation_completion"
            ),
            "added": [
                {
                    "id": item.get("id"),
                    "source": item.get("source"),
                    "target": item.get("target"),
                    "paper_source": item.get("paper_source"),
                    "paper_target": item.get("paper_target"),
                    "paper_relation_id": item.get("paper_relation_id"),
                    "type": item.get("type"),
                }
                for item in plan.get("connectors", [])
                if isinstance(item, dict) and item.get("added_by") == "contract_relation_completion"
            ],
            "planned_connector_count": len(plan.get("connectors") or []),
        },
        "graph_constraint": (
            {"enabled": True, **check_design_against_graph(plan, resolved_graph_context)}
            if graph_context
            else {"enabled": False}
        ),
        "provider": provider_metadata,
        "density": density,
        "replan_attempts": replan_attempts,
        "wireframe_path": wireframe_path,
        "canvas_safe_area": {"initial": initial_safe_area, "repaired": repaired_safe_area, "final": final_safe_area},
        "panel_adaptation": final_adaptation,
    }
