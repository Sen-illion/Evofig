"""Evidence-grounded scientific hierarchy planning for paper figures."""

from __future__ import annotations

from typing import Any

from ..contracts.vlm_context import compact_method, json_compact, context_stats
from ..providers import call_vlm_json, resolve_vlm_model


def _items(method: dict[str, Any], field: str) -> list[dict[str, Any]]:
    values = method.get(field) if isinstance(method.get(field), list) else []
    return [item for item in values if isinstance(item, dict) and item.get("id")]


def _method_entity_evidence(method: dict[str, Any] | None) -> dict[str, list[str]]:
    """Map every entity the method contract declares to the evidence it carries."""

    table: dict[str, list[str]] = {}
    for field in ("inputs", "modules", "outputs", "stages", "innovations"):
        for item in _items(method or {}, field):
            table[str(item["id"])] = [str(x) for x in (item.get("evidence_ids") or [])]
    return table


def repair_scientific_hierarchy(plan: dict[str, Any] | None, method: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fix the model's bookkeeping while keeping the science it wrote.

    A section the model named differently from the section list, an entity id
    copied out of the wrong field, or an evidence list it left empty are all
    repairable. Rejecting the whole plan for one of them used to replace the
    paper's own hierarchy with generic buckets, so the figure lost the very
    structure it was meant to show.
    """

    source = plan if isinstance(plan, dict) else {}
    evidence_by_entity = _method_entity_evidence(method)
    sections = [dict(item) for item in (source.get("sections") or []) if isinstance(item, dict)]
    submodules = [dict(item) for item in (source.get("submodules") or []) if isinstance(item, dict)]
    repairs: list[dict[str, str]] = []
    for index, section in enumerate(sections):
        if not str(section.get("id") or "").strip():
            section["id"] = f"section_recovered_{index + 1}"
            repairs.append({"kind": "section_id_generated", "target": str(section["id"])})
        if not str(section.get("title") or "").strip():
            section["title"] = str(section.get("id") or "")
            repairs.append({"kind": "section_title_from_id", "target": str(section["id"])})
    section_ids = {str(section.get("id") or "") for section in sections}
    declared_owner = {str(sub): str(section.get("id") or "") for section in sections for sub in (section.get("submodule_ids") or [])}
    for item in submodules:
        sub_id = str(item.get("id") or "")
        parent = str(item.get("parent_section_id") or "")
        if sections and parent not in section_ids:
            adopted = declared_owner.get(sub_id) or str(sections[0].get("id") or "")
            item["parent_section_id"] = adopted
            repairs.append({"kind": "parent_section_adopted", "target": sub_id, "from": parent or "(none)", "to": adopted})
        entity_ids = [str(x) for x in (item.get("entity_ids") or [])]
        if evidence_by_entity and entity_ids:
            known = [x for x in entity_ids if x in evidence_by_entity]
            unknown = [x for x in entity_ids if x not in evidence_by_entity]
            if unknown and known:
                item["entity_ids"] = known
                repairs.append({"kind": "unknown_entities_dropped", "target": sub_id, "dropped": ", ".join(unknown)})
            elif unknown:
                repairs.append({"kind": "all_entities_unknown", "target": sub_id, "dropped": ", ".join(unknown)})
        if not [x for x in (item.get("evidence_ids") or []) if str(x).strip()]:
            inherited = sorted({str(e) for x in (item.get("entity_ids") or []) for e in evidence_by_entity.get(str(x), []) if str(e).strip()})
            if inherited:
                item["evidence_ids"] = inherited
                repairs.append({"kind": "submodule_evidence_inherited", "target": sub_id, "count": str(len(inherited))})
    grounded_by_parent: dict[str, list[str]] = {}
    for item in submodules:
        grounded_by_parent.setdefault(str(item.get("parent_section_id") or ""), []).extend(str(e) for e in (item.get("evidence_ids") or []) if str(e).strip())
    for section in sections:
        if not [x for x in (section.get("evidence_ids") or []) if str(x).strip()]:
            aggregated = sorted(set(grounded_by_parent.get(str(section.get("id") or ""), [])))
            if aggregated:
                section["evidence_ids"] = aggregated
                repairs.append({"kind": "section_evidence_aggregated", "target": str(section.get("id") or ""), "count": str(len(aggregated))})
    return {**source, "sections": sections, "submodules": submodules, "repaired": bool(repairs), "repairs": repairs}


def normalize_scientific_hierarchy(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Accept equivalent VLM field spellings while emitting one stable contract."""
    source = raw if isinstance(raw, dict) else {}
    sections: list[dict[str, Any]] = []
    submodules: list[dict[str, Any]] = []
    for section in source.get("sections", []) if isinstance(source.get("sections"), list) else []:
        if not isinstance(section, dict):
            continue
        sid = str(section.get("id") or "")
        nested = section.get("submodules") if isinstance(section.get("submodules"), list) else []
        ids = [str(x.get("id")) for x in nested if isinstance(x, dict) and x.get("id")]
        ids.extend(str(x) for x in (section.get("submodule_ids") or []) if str(x) not in ids)
        sections.append({**section, "id": sid, "title": str(section.get("title") or section.get("name") or sid), "submodule_ids": ids})
        for item in nested:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            submodules.append({**item, "id": str(item["id"]), "parent_section_id": str(item.get("parent_section_id") or item.get("section_id") or sid), "title": str(item.get("title") or item.get("name") or item["id"]), "entity_ids": list(item.get("entity_ids") or ([item.get("id")] if item.get("id") else [])), "evidence_ids": list(item.get("evidence_ids") or [])})
    for item in source.get("submodules", []) if isinstance(source.get("submodules"), list) else []:
        if not isinstance(item, dict) or not item.get("id") or any(str(x.get("id")) == str(item.get("id")) for x in submodules):
            continue
        submodules.append({**item, "id": str(item["id"]), "parent_section_id": str(item.get("parent_section_id") or item.get("section_id") or ""), "title": str(item.get("title") or item.get("name") or item["id"]), "entity_ids": list(item.get("entity_ids") or ([item.get("id")] if item.get("id") else [])), "evidence_ids": list(item.get("evidence_ids") or [])})
    return {"contract": "rfs.scientific_hierarchy", "version": 1, "sections": sections, "submodules": submodules, "primary_flow": source.get("primary_flow") if isinstance(source.get("primary_flow"), list) else [], "supporting_flows": source.get("supporting_flows") if isinstance(source.get("supporting_flows"), list) else [], "feedback_flows": source.get("feedback_flows") if isinstance(source.get("feedback_flows"), list) else [], "complexity": source.get("complexity") if isinstance(source.get("complexity"), dict) else {}}


def validate_scientific_hierarchy(plan: dict[str, Any] | None, method: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate hierarchy IDs and evidence without enforcing a fixed section count."""
    source = plan if isinstance(plan, dict) else {}
    sections = [x for x in source.get("sections", []) if isinstance(x, dict)]
    submodules = [x for x in source.get("submodules", []) if isinstance(x, dict)]
    method_ids = {str(x.get("id")) for field in ("inputs", "modules", "outputs", "stages", "innovations") for x in _items(method or {}, field)}
    errors: list[str] = []
    section_ids = {str(x.get("id")) for x in sections if x.get("id")}
    sub_ids = {str(x.get("id")) for x in submodules if x.get("id")}
    for i, section in enumerate(sections):
        if not section.get("id") or not section.get("title"):
            errors.append(f"section[{i}] requires id and title")
        if not section.get("evidence_ids"):
            errors.append(f"section[{i}] requires evidence_ids")
        for sub_id in section.get("submodule_ids") or []:
            if str(sub_id) not in sub_ids:
                errors.append(f"section[{i}] references unknown submodule")
    for i, item in enumerate(submodules):
        if not item.get("id") or not item.get("title"):
            errors.append(f"submodule[{i}] requires id and title")
        if str(item.get("parent_section_id")) not in section_ids:
            errors.append(f"submodule[{i}] references unknown parent section")
        entity_ids = [str(x) for x in (item.get("entity_ids") or [])]
        if method_ids and entity_ids and not set(entity_ids).issubset(method_ids):
            errors.append(f"submodule[{i}] references entities absent from method")
        if not item.get("evidence_ids"):
            errors.append(f"submodule[{i}] requires evidence_ids")
    return {
        "ok": not errors,
        "errors": errors,
        "section_count": len(sections),
        "submodule_count": len(submodules),
        "warning": "complex method has fewer than three evidence-grounded sections" if len(sections) < 3 and len(submodules) >= 5 else None,
    }


def _heuristic(method: dict[str, Any]) -> dict[str, Any]:
    """Build a conservative hierarchy when VLM is unavailable."""
    buckets = [("inputs", "Input and Grounding"), ("modules", "Core Method"), ("stages", "Workflow Stages"), ("outputs", "Outputs and Evaluation")]
    sections: list[dict[str, Any]] = []
    submodules: list[dict[str, Any]] = []
    for field, title in buckets:
        values = _items(method, field)
        if not values:
            continue
        sid = "section_" + field
        sub_ids = []
        evidence = []
        for index, item in enumerate(values[:8]):
            mid = f"{sid}_module_{index + 1}"
            sub_ids.append(mid)
            evidence.extend(item.get("evidence_ids") or [])
            submodules.append({"id": mid, "parent_section_id": sid, "title": str(item.get("label") or item.get("name") or item["id"]), "role": item.get("role") or field, "entity_ids": [str(item["id"])], "evidence_ids": list(item.get("evidence_ids") or [])})
        sections.append({"id": sid, "title": title, "role": field, "submodule_ids": sub_ids, "evidence_ids": sorted(set(evidence))})
    if not sections:
        sections = [{"id": "section_core", "title": "Core Method", "role": "method", "submodule_ids": [], "evidence_ids": []}]
    return {"contract": "rfs.scientific_hierarchy", "version": 1, "sections": sections, "submodules": submodules, "primary_flow": [], "supporting_flows": [], "feedback_flows": [], "complexity": {"workflow_stage_count": len(_items(method, "stages")), "section_count": len(sections), "max_depth": 2 if submodules else 1, "relation_density": "high" if len(_items(method, "relations")) >= 8 else "medium" if len(_items(method, "relations")) >= 4 else "low"}}


def plan_scientific_hierarchy(method: dict[str, Any], exemplar_style_brief: dict[str, Any] | None = None, *, model: str | None = None, timeout: int = 180, retries: int = 1) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask VLM for evidence-grounded section/submodule hierarchy with safe fallback."""
    prompt = (
        "Return exactly one JSON object matching rfs.scientific_hierarchy with sections, submodules, primary_flow, supporting_flows, feedback_flows, complexity. "
        "Derive sections from the paper's actual method/workflow/training/inference logic, not canvas geometry. Do not force three sections: use two when scientifically justified, and use more when the paper has distinct evidence-grounded stages. "
        "Every section and submodule must cite evidence_ids and only use entity IDs from METHOD. Split coarse modules into evidenced submodules when the method supports it; never invent science. "
        "REFERENCE_EXPERIENCE=" + json_compact({"structure_rules": (exemplar_style_brief or {}).get("structure_rules", [])[:8], "layout_rules": (exemplar_style_brief or {}).get("layout_rules", [])[:8]})
        + " METHOD=" + json_compact(compact_method(method))
    )
    raw = None
    try:
        raw = call_vlm_json(prompt, [], model=resolve_vlm_model("RFS_STRUCTURE_MODEL", explicit_model=model), timeout=timeout, retries=retries, call_metadata={})
        plan = repair_scientific_hierarchy(normalize_scientific_hierarchy(raw), method)
        validation = validate_scientific_hierarchy(plan, method)
        repair_rounds = 0
        if not validation["ok"]:
            # Hand the errors straight back to the model: the alternative is a
            # generic bucket hierarchy that says nothing about this paper, and
            # one more question is cheaper than that figure.
            try:
                repaired_raw = call_vlm_json(_repair_prompt(prompt, plan, validation), [], model=resolve_vlm_model("RFS_STRUCTURE_MODEL", explicit_model=model), timeout=timeout, retries=retries, call_metadata={})
                repaired_plan = repair_scientific_hierarchy(normalize_scientific_hierarchy(repaired_raw), method)
                repaired_validation = validate_scientific_hierarchy(repaired_plan, method)
                repair_rounds = 1
                if repaired_validation["ok"] or len(repaired_validation["errors"]) < len(validation["errors"]):
                    plan, validation, raw = repaired_plan, repaired_validation, repaired_raw
            except Exception:
                # The question was asked and the answer never arrived. Keep the
                # plan we have; the report says it is still short of evidence.
                repair_rounds = 1
        report = {**validation, "mode": "vlm", "fallback_used": False, "repaired": bool(plan.get("repaired")), "repairs": plan.get("repairs") or [], "repair_rounds": repair_rounds, "continued_with_invalid_plan": not validation["ok"], "unrepaired_errors": [] if validation["ok"] else list(validation["errors"]), "raw_response": raw, "context": context_stats(prompt, field_names=["compact_method", "exemplar_structure_rules"])}
        return plan, report
    except Exception as exc:
        plan = _heuristic(method)
        return plan, {**validate_scientific_hierarchy(plan, method), "mode": "heuristic_fallback", "fallback_used": True, "fallback_reason": str(exc), "raw_response": raw, "context": context_stats(prompt, field_names=["compact_method", "exemplar_structure_rules"])}


def _repair_prompt(prompt: str, plan: dict[str, Any], validation: dict[str, Any]) -> str:
    """Ask again with the validation errors, keeping the science untouched."""

    return (
        "The object you returned for rfs.scientific_hierarchy did not pass validation. Return the corrected object and change nothing else: keep every section id, submodule id, title, entity id and every scientific statement exactly as it is, and fix only these errors: "
        + json_compact({"errors": list(validation.get("errors") or [])})
        + " Every section and submodule must carry evidence_ids taken from its own entities in METHOD. "
        "ORIGINAL_PLAN=" + json_compact(plan) + " " + prompt
    )
