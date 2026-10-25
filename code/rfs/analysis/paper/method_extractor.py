"""Extract a compact, evidence-aware method contract from the paper plan."""

from __future__ import annotations

from typing import Any
import json
import re

from ...contracts import normalize_method_figure_content, validate_method_figure_content
from ...providers import call_vlm_json, resolve_vlm_model
from ...contracts.vlm_context import compact_method, compact_review, context_stats, json_compact
from ...planning.panel_target import compact_panel_target

DEFAULT_METHOD_TIMEOUT_SECONDS = 240


def _repair_relation_endpoints(content: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    declared = {
        str(item.get("id"))
        for field in ("inputs", "outputs", "modules", "stages", "innovations")
        for item in content.get(field, [])
        if isinstance(item, dict) and item.get("id")
    }
    kept, removed = [], []
    for relation in content.get("relations", []):
        source, target = str(relation.get("source") or ""), str(relation.get("target") or "")
        if source in declared and target in declared:
            kept.append(relation)
        else:
            removed.append({"id": relation.get("id"), "source": source, "target": target, "reason": "undeclared_endpoint"})
    return {**content, "relations": kept}, removed


def _fallback(figure_specification: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    source = figure_specification if isinstance(figure_specification, dict) else {}
    payload = {key: source.get(key, []) for key in ("inputs", "outputs", "modules", "relations", "stages", "innovations", "optional_details", "forbidden_inventions", "uncertainties")}
    payload["figure_goal"] = source.get("figure_goal") or source.get("summary") or "Explain the paper method."
    payload["provenance"] = provenance
    return normalize_method_figure_content(payload)


def _panel_node_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _panel_node_role(value: Any) -> str:
    role = str(value or "").strip().casefold()
    return {"input": "input", "output": "output", "objective": "module", "conditioning": "module", "group": "module"}.get(role, "module")


def _panel_node_refs(node: dict[str, Any]) -> list[str]:
    """Every contract-side spelling a panel edge may use for one panel node."""

    refs: list[str] = []
    for value in (node.get("instance_id"), node.get("id"), node.get("name"), node.get("label")):
        text = str(value or "").strip()
        if not text:
            continue
        refs.append(text)
        refs.append(_panel_node_key(text))
    return list(dict.fromkeys(refs))


def _merge_panel_nodes_into_content(
    content: dict[str, Any],
    figure_specification: dict[str, Any],
    *,
    max_panels: int = 8,
    max_nodes_per_panel: int = 16,
    max_relations: int = 160,
) -> dict[str, Any]:
    """Declare the paper's own panel nodes and panel edges the reader left out.

    The contract already carries the panels, the exact labels printed inside them and
    the edges drawn between those labels, but the reader is a small model that keeps
    collapsing a dense framework figure into a handful of coarse boxes and a few
    global relations.  A benchmark run is scored against the figure the paper printed,
    so both the nodes and the panel's own wiring are recovered deterministically:
    every label becomes a declared object under a stage for its panel and every panel
    edge becomes a relation between those objects, with the contract's own evidence
    ids, and nothing else about the plan is rewritten.

    Wiring is the half that decides whether the figure reads as a method.  Without it
    the design stage is asked to lay out eighteen objects while being told about four
    connections, which is exactly how a dense figure ends up carrying four arrows.
    """

    panels = figure_specification.get("panel_graphs") if isinstance(figure_specification, dict) else None
    if not isinstance(panels, list) or not panels:
        return {"applied": False, "added": [], "panels": 0, "skipped": [], "relations_added": 0, "relation_pairs": []}
    object_fields = ("inputs", "outputs", "modules", "stages", "innovations", "optional_details")
    declared: set[str] = set()
    id_by_ref: dict[str, str] = {}
    for field in object_fields:
        for item in content.get(field) or []:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("id") or "").strip()
            if not object_id:
                continue
            for value in (object_id, item.get("name"), item.get("label")):
                text = str(value or "").strip()
                if text:
                    id_by_ref.setdefault(text, object_id)
                    id_by_ref.setdefault(_panel_node_key(text), object_id)
            name_key = _panel_node_key(item.get("name") or item.get("label"))
            if name_key:
                declared.add(name_key)
    added: list[str] = []
    skipped: list[str] = []
    relation_pairs: list[str] = []
    seen_pairs = {
        (str(item.get("source") or item.get("source_id") or ""), str(item.get("target") or item.get("target_id") or ""))
        for item in (content.get("relations") or [])
        if isinstance(item, dict)
    }
    for panel_index, panel in enumerate(panels[:max_panels]):
        if not isinstance(panel, dict):
            continue
        nodes = [
            node for node in (panel.get("nodes") or [])
            if isinstance(node, dict) and str(node.get("name") or node.get("label") or "").strip()
        ]
        if not nodes:
            continue
        label = str(panel.get("panel_label") or panel_index + 1)
        panel_key = _panel_node_key(label) or str(panel_index + 1)
        fresh = [node for node in nodes if _panel_node_key(node.get("name") or node.get("label")) not in declared]
        if fresh:
            stage_evidence = [
                evidence for node in fresh for evidence in (node.get("evidence_ids") or [])
            ][:4]
            content.setdefault("stages", []).append({
                "id": "stage_panel_" + panel_key,
                "label": str(panel.get("label") or panel.get("title") or f"Panel {label}").strip() or f"Panel {label}",
                "role": "stage",
                "parent_id": None,
                "detail_level": "stage",
                "scientific_importance": 0.9,
                "visualization_value": 0.8,
                "must_show": bool(stage_evidence),
                "should_render": False,
                "render_priority": 2,
                "evidence_ids": list(dict.fromkeys(stage_evidence)),
                "input_ids": [],
                "output_ids": [],
            })
            declared.add(_panel_node_key(f"Panel {label}"))
        declared_ids = {id(node) for node in fresh[:max_nodes_per_panel]}
        for node in nodes:
            name = str(node.get("name") or node.get("label")).strip()
            name_key = _panel_node_key(name)
            existing = id_by_ref.get(name) or id_by_ref.get(name_key)
            if existing:
                for ref in _panel_node_refs(node):
                    id_by_ref.setdefault(ref, existing)
                continue
            if id(node) not in declared_ids:
                continue
            evidence_ids = list(dict.fromkeys(node.get("evidence_ids") or []))[:4]
            field = {"input": "inputs", "output": "outputs"}.get(_panel_node_role(node.get("role")), "modules")
            content_id = "panelnode_" + (name_key[:40] or "node")
            content.setdefault(field, []).append({
                "id": content_id,
                "label": name,
                "role": _panel_node_role(node.get("role")),
                "parent_id": "stage_panel_" + panel_key,
                "detail_level": "mechanism",
                "scientific_importance": 0.7,
                "visualization_value": 0.9,
                "must_show": bool(evidence_ids),
                "should_render": True,
                "render_priority": 2,
                "evidence_ids": evidence_ids,
                "input_ids": [],
                "output_ids": [],
            })
            declared.add(name_key)
            id_by_ref.setdefault(name, content_id)
            id_by_ref.setdefault(name_key, content_id)
            for ref in _panel_node_refs(node):
                id_by_ref.setdefault(ref, content_id)
            added.append(name)
        for node in fresh[max_nodes_per_panel:]:
            skipped.append(str(node.get("name") or node.get("label")))
        for relation_index, relation in enumerate(panel.get("relations") if isinstance(panel.get("relations"), list) else []):
            if not isinstance(relation, dict):
                continue
            source_ref = str(relation.get("source") or relation.get("source_id") or "").strip()
            target_ref = str(relation.get("target") or relation.get("target_id") or "").strip()
            source = id_by_ref.get(source_ref) or id_by_ref.get(_panel_node_key(source_ref))
            target = id_by_ref.get(target_ref) or id_by_ref.get(_panel_node_key(target_ref))
            if not source or not target or source == target or (source, target) in seen_pairs:
                continue
            if len(content.get("relations") or []) >= max_relations:
                continue
            seen_pairs.add((source, target))
            content.setdefault("relations", []).append({
                "id": "panelrel_" + panel_key + "_" + str(relation_index + 1).zfill(2),
                "source": source,
                "target": target,
                "type": str(relation.get("type") or relation.get("relation_type") or "data_flow"),
                "label": str(relation.get("label") or ""),
                "evidence_ids": [str(value) for value in (relation.get("evidence_ids") or []) if value],
            })
            relation_pairs.append(source + "->" + target)
    return {"applied": bool(added), "added": added, "panels": len(panels), "skipped": skipped, "relations_added": len(relation_pairs), "relation_pairs": relation_pairs}


def extract_method_figure_content(figure_specification: dict[str, Any], *, paper_review: dict[str, Any] | None = None, mode: str = "heuristic", model: str | None = None, timeout_seconds: int = DEFAULT_METHOD_TIMEOUT_SECONDS, retries: int = 1, provenance: dict[str, Any] | None = None, feedback: str = "", figure_target: str = "overview", panel_target: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract only paper-grounded content that belongs in the method figure."""
    base_provenance = provenance or {"source": "figure_specification"}
    fallback_used = False
    error = None
    recovery_feedback = ""
    #: The panel target in prompt form; empty unless a benchmark run asked for the
    #: paper's own panels and the contract declared any.
    panel_rows = ""
    if mode == "vlm":
        compact_spec = compact_method(figure_specification)
        compact_paper_review = compact_review(paper_review)
        # A benchmark run is scored against the figure the paper printed, down to
        # its panel granularity, so the target reaches this prompt and helps decide
        # what the figure is asked to show. The overview path sends nothing and
        # keeps exactly the prompt it had.
        panel_rows = compact_panel_target(panel_target or {}) if figure_target == "paper" else ""
        panel_block = (
            "Figure target: the paper's own framework figure, panel by panel, in the paper's own order.\n"
            "Match its granularity: split the content the way its panels split it, and keep every label it lists, "
            "spelled the way the paper spells it. When it names a single panel, that panel is the whole guide. "
            "Never add a panel or a label the target does not state.\n"
            "PANEL_TARGET=" + panel_rows + "\n"
        ) if panel_rows else ""
        # A second extraction of the same paper is asked about the objects a measurement
        # could not tell apart. It reads the same specification and the same review, so the
        # note may only re-decide ownership and selection -- never add a module the paper
        # does not state, which the prompt above already forbids.
        # The specification may already carry the paper's own panels and the exact
        # labels inside them.  That inventory is the only per-node evidence for a
        # dense framework figure, so the reader is told to keep every one of those
        # names instead of re-deciding the granularity from prose alone.
        declared_panels = figure_specification.get("panel_graphs") if isinstance(figure_specification, dict) else None
        panel_contract_block = ""
        if isinstance(declared_panels, list) and declared_panels:
            required_rows: list[dict[str, Any]] = []
            supporting_names: list[str] = []
            for panel in declared_panels[:8]:
                if not isinstance(panel, dict):
                    continue
                panel_label = str(panel.get("panel_label") or "")
                nodes = [node for node in (panel.get("nodes") or []) if isinstance(node, dict) and str(node.get("name") or "").strip()]
                for node in nodes[:24]:
                    required_rows.append({
                        "panel": panel_label,
                        "name": str(node.get("name")).strip(),
                        "evidence_ids": list(node.get("evidence_ids") or [])[:2],
                    })
                for node in nodes[24:40]:
                    supporting_names.append(str(node.get("name")).strip())
            if required_rows:
                panel_contract_block = (
                    "\nREQUIRED_PANEL_NODES: the specification's panel_graphs were recovered verbatim from the figure the paper printed. "
                    "Declare every name in the list below as its own method object, exactly once, with the same spelling and the same evidence_ids, "
                    "and group each panel's objects under one stage object for that panel via parent_id. A panel that lists N names must produce N objects: "
                    "merging two names into one object, or leaving one out, fails this contract. Set should_render=true for the names printed as boxes in "
                    "the figure; a name that is only a value or a triple may be detail_level=artifact with should_render=false.\n"
                    "REQUIRED_PANEL_NODES=" + json_compact(required_rows) + "\n"
                    + ("Also evidenced, lower priority: " + ", ".join(supporting_names) + "\n" if supporting_names else "")
                )
        if str(feedback or "").strip():
            recovery_feedback = f"\nRecovery feedback:\n{feedback}\n"
        prompt = f"""Extract a detailed but selective, evidence-grounded scientific method hierarchy for a paper framework figure. Return JSON only.
First identify the actual method stages, core modules, valuable internal mechanisms, intermediate artifacts, constraints, outputs and feedback relations. Then decide which details deserve visualization. Do not collapse a complex paper into only 3-7 coarse boxes when evidence supports meaningful submodules. Do not include ordinary implementation trivia, parameter lists, repeated prose, background motivation, or isolated nouns merely to increase density.
Keep exact paper terminology. Every renderable node and every relation must preserve evidence_ids from the supplied contracts. Declare every concept exactly once: a module or mechanism that is already listed must not be repeated as an innovation, because repeated entries are drawn as duplicate cards; innovations carry only additional, separately evidenced contributions. Separate training, inference and evaluation stages when supported. Put uncertain content in uncertainties. Never invent modules or use visual content from a reference image.
Schema keys: figure_goal, inputs, outputs, modules, relations, stages, innovations, optional_details, forbidden_inventions, uncertainties.
Each method object: id, label, role, parent_id, detail_level(stage|module|mechanism|artifact), scientific_importance(0..1), visualization_value(0..1), must_show, should_render, render_priority(1 highest), evidence_ids, input_ids, output_ids. input_ids and output_ids may only repeat ids declared as method objects in this same JSON: a handoff you name there (a tensor, table, assignment, reward signal, memory) must first be declared as its own object, drawn with detail_level=artifact and should_render=false when the figure should not show it. Never write an undeclared product name in these two lists; when a handoff is only carried by an arrow, keep it in that relation's label instead. Use parent_id to express real hierarchy. Put evidenced but non-visual details in optional_details with should_render=false. Each relation: id, source, target, type, label, evidence_ids.
{panel_block}{panel_contract_block}Figure specification:
{json_compact(compact_spec)}
Paper review:
{json_compact(compact_paper_review)}{recovery_feedback}"""
        provider_metadata: dict[str, Any] = {}
        try:
            raw = call_vlm_json(prompt, [], model=resolve_vlm_model("RFS_PAPER_REVIEW_MODEL", explicit_model=model), timeout=max(10, int(timeout_seconds)), retries=max(0, int(retries)), call_metadata=provider_metadata)
            raw["provenance"] = {**base_provenance, "mode": "vlm", "model": resolve_vlm_model("RFS_PAPER_REVIEW_MODEL", explicit_model=model)}
            content = normalize_method_figure_content(raw)
        except Exception as exc:
            fallback_used, error = True, str(exc)
            content = _fallback(figure_specification, {**base_provenance, "mode": "heuristic_fallback", "fallback_reason": error})
    else:
        content = _fallback(figure_specification, {**base_provenance, "mode": "heuristic"})
    panel_merge = {"applied": False, "added": [], "panels": 0, "skipped": [], "relations_added": 0, "relation_pairs": []}
    if figure_target == "paper":
        panel_merge = _merge_panel_nodes_into_content(content, figure_specification)
        content = normalize_method_figure_content(content)
    content, removed_relations = _repair_relation_endpoints(content)
    validation = validate_method_figure_content(content)
    missing_evidence = []
    for section in ("inputs", "outputs", "modules", "innovations", "relations"):
        for item in content.get(section, []):
            if item.get("must_show", section != "innovations") and not item.get("evidence_ids"):
                missing_evidence.append(str(item.get("id") or item.get("label") or item.get("name") or "unknown"))
    report = {**validation, "mode": mode, "fallback_used": fallback_used, "fallback_reason": error, "missing_evidence_ids": missing_evidence, "removed_relations": removed_relations, "removed_relation_count": len(removed_relations), "figure_target": figure_target, "panel_target_supplied": bool(panel_rows), "panel_nodes_added": panel_merge["added"], "panel_nodes_skipped": panel_merge["skipped"], "panel_node_merge_applied": panel_merge["applied"], "panel_relations_added": panel_merge["relations_added"], "context": context_stats(prompt, field_names=["compact_figure_specification", "compact_paper_review"] + (["panel_target"] if mode == "vlm" and panel_rows else []) + (["recovery_feedback"] if recovery_feedback else [])) if mode == "vlm" else {}, "provider": provider_metadata if mode == "vlm" else {}, "grounding_status": "pass" if validation["ok"] and not missing_evidence else "needs_review"}
    return content, report
