from __future__ import annotations

import json
import re
from typing import Any

from ...providers import call_vlm_json, resolve_vlm_model, vlm_credentials_available
from .document import evidence_excerpt
from ...contracts.vlm_context import compact_document_index, compact_evidence, context_stats


DOMAIN_PROFILES: dict[str, dict[str, Any]] = {
    "general": {
        "summary": "General research-paper review profile.",
        "required_sections": ["research_questions", "central_claims", "contributions", "modules", "limitations"],
        "keywords": [],
    },
    "ai-ml-method": {
        "summary": "AI/ML method-paper review profile.",
        "required_sections": ["research_questions", "central_claims", "inputs", "outputs", "modules", "relations", "innovations", "experiments", "terminology"],
        "keywords": ["model", "training", "inference", "loss", "encoder", "decoder", "benchmark", "neural"],
    },
    "system-platform": {
        "summary": "System and platform paper review profile.",
        "required_sections": ["research_questions", "central_claims", "inputs", "outputs", "modules", "relations", "workflows", "experiments", "limitations"],
        "keywords": ["system", "platform", "architecture", "service", "deployment", "latency", "throughput", "pipeline"],
    },
    "dataset-benchmark": {
        "summary": "Dataset and benchmark paper review profile.",
        "required_sections": ["research_questions", "central_claims", "inputs", "outputs", "contributions", "experiments", "limitations"],
        "keywords": ["dataset", "benchmark", "annotation", "corpus", "collection", "split", "baseline"],
    },
    "empirical-science": {
        "summary": "Experimental natural-science paper review profile.",
        "required_sections": ["research_questions", "central_claims", "inputs", "outputs", "assumptions", "experiments", "results", "limitations"],
        "keywords": ["experiment", "sample", "specimen", "clinical", "assay", "measurement", "hypothesis"],
    },
    "survey-review": {
        "summary": "Survey and review paper profile.",
        "required_sections": ["research_questions", "central_claims", "contributions", "concepts", "figure_candidates", "limitations"],
        "keywords": ["survey", "review", "taxonomy", "systematic review", "meta-analysis", "landscape"],
    },
}


FACT_DEFAULTS = {
    "status": "required",
    "importance": "medium",
    "confidence": 0.5,
    "must_appear_in_figure": False,
    "visual_role": "supporting_detail",
}


def detect_domain_profile(parsed: dict, explicit: str = "auto") -> dict:
    if explicit != "auto":
        profile = DOMAIN_PROFILES.get(explicit)
        if not profile:
            raise ValueError(f"Unknown domain profile: {explicit}")
        return {"summary": profile["summary"], "id": explicit, "selection": "explicit", **profile}
    sample = " ".join(item.get("text", "") for item in parsed.get("evidence", [])[:20]).lower()
    scores = {}
    for name, profile in DOMAIN_PROFILES.items():
        scores[name] = sum(sample.count(keyword) for keyword in profile.get("keywords", []))
    selected = max((name for name in scores if name != "general"), key=lambda item: scores[item], default="general")
    if scores.get(selected, 0) <= 0:
        selected = "general"
    profile = DOMAIN_PROFILES[selected]
    return {"summary": profile["summary"], "id": selected, "selection": "automatic", "scores": scores, **profile}


def _review_prompt(parsed: dict, domain_profile: dict, evidence_max_chars: int = 65000) -> str:
    evidence_limit = min(max(4000, int(evidence_max_chars)), 14000)
    return f"""
# Summary

You are a rigorous cross-domain scientific paper reviewer. Return JSON only. Extract facts from evidence, never from generic expectations. Do not create a figure layout yet.

Domain profile:
{json.dumps(domain_profile, ensure_ascii=False, indent=2)}

Document structure index:
{json.dumps(compact_document_index(parsed.get('document_index')), ensure_ascii=False, separators=(',', ':'))}

Every fact must use this object shape:
{{
  "id": "stable_snake_case_id",
  "statement": "paper-grounded statement",
  "status": "required|optional|forbidden|unknown",
  "importance": "critical|high|medium|low",
  "confidence": 0.0,
  "evidence_ids": ["E0001"],
  "must_appear_in_figure": true,
  "visual_role": "input|output|module|relation|innovation|experiment|constraint|supporting_detail"
}}

Return:
{{
  "summary": "Universal structured paper review.",
  "schema_version": "2.0",
  "paper_identity": {{"title": "...", "paper_type": "...", "field": "..."}},
  "domain_profile": "{domain_profile['id']}",
  "research_questions": [],
  "central_claims": [],
  "inputs": [],
  "outputs": [],
  "research_objects": [],
  "concepts": [],
  "modules": [],
  "relations": [{{
    "id": "relation_id",
    "source_id": "module_or_object_id",
    "target_id": "module_or_object_id",
    "relation_type": "data_flow|control_flow|causal|comparison|feedback|contains|depends_on",
    "statement": "...",
    "status": "required|optional|forbidden|unknown",
    "importance": "critical|high|medium|low",
    "confidence": 0.0,
    "evidence_ids": ["E0001"],
    "must_appear_in_figure": true,
    "visual_role": "relation"
  }}],
  "workflows": {{"training": [], "inference": [], "offline": [], "online": [], "feedback": []}},
  "contributions": [],
  "innovations": [],
  "assumptions": [],
  "limitations": [],
  "experiments": {{"datasets": [], "settings": [], "metrics": [], "baselines": [], "ablations": []}},
  "results": [],
  "terminology": [{{"id": "term_id", "statement": "Exact Term", "visible_label": "Exact Term", "status": "required", "importance": "critical", "confidence": 1.0, "evidence_ids": ["E0001"], "must_appear_in_figure": true, "visual_role": "module"}}],
  "forbidden_inventions": [],
  "unknowns": [],
  "contradictions": [],
  "ambiguities": [],
  "human_review_required": [],
  "figure_candidates": [{{"id": "figure_overview", "figure_type": "method_overview|system_pipeline|data_construction|conceptual_framework|experiment_design", "purpose": "...", "information_payload_ids": [], "recommended": true}}]
}}

Rules:
- Bind critical claims, modules, relations, innovations, results, and terminology to evidence IDs.
- Do not infer a relation merely because two section headings are adjacent.
- Separate training from inference and offline from online.
- Put unsupported but tempting mechanisms into forbidden_inventions.
- Use unknown status instead of guessing.
- Short visible labels should normally be at most 32 characters.

Paper evidence:
{compact_evidence(parsed, max_chars=evidence_limit)}
""".strip()


def _fact(identifier: str, statement: str, evidence_ids: list[str], **overrides: Any) -> dict:
    result = {"id": identifier, "statement": statement, "evidence_ids": evidence_ids, **FACT_DEFAULTS}
    result.update(overrides)
    return result


def _heuristic_review(parsed: dict, domain_profile: dict) -> dict:
    evidence = parsed.get("evidence", [])
    first = evidence[0]
    lines = [line.strip() for line in first.get("text", "").splitlines() if line.strip()]
    title = next((line for line in lines if 8 <= len(line) <= 180), parsed.get("source_name", "Untitled paper"))
    headings = [item for item in parsed.get("headings", []) if item.lower() not in {"abstract", "introduction", "method", "methods", "experiments", "results", "conclusion", "references"}]
    modules = []
    for index, heading in enumerate(headings[:8]):
        evidence_id = evidence[min(index, len(evidence) - 1)]["id"]
        modules.append(_fact(f"module_{index + 1:02d}", heading, [evidence_id], importance="high", must_appear_in_figure=True, visual_role="module"))
    research_question = _fact("research_question_01", first.get("text", "")[:600].strip(), [first["id"]], importance="critical", must_appear_in_figure=True, visual_role="constraint")
    return {
        "summary": "Heuristic universal paper review for engineering validation; use VLM mode for production.",
        "schema_version": "2.0",
        "paper_identity": {"title": title, "paper_type": "unknown", "field": "unknown"},
        "domain_profile": domain_profile["id"],
        "research_questions": [research_question],
        "central_claims": [],
        "inputs": [],
        "outputs": [],
        "research_objects": [],
        "concepts": [],
        "modules": modules,
        "relations": [],
        "workflows": {"training": [], "inference": [], "offline": [], "online": [], "feedback": []},
        "contributions": [],
        "innovations": [],
        "assumptions": [],
        "limitations": [],
        "experiments": {"datasets": [], "settings": [], "metrics": [], "baselines": [], "ablations": []},
        "results": [],
        "terminology": [{**item, "visible_label": item["statement"]} for item in modules],
        "forbidden_inventions": [_fact("forbidden_generic_modules", "Do not add unsupported generic AI modules.", [], status="forbidden", importance="critical", confidence=1.0, visual_role="constraint")],
        "unknowns": [_fact("unknown_claims", "Central claims and innovations require VLM review.", [], status="unknown", importance="high", confidence=0.0, visual_role="constraint")],
        "contradictions": [],
        "ambiguities": [],
        "human_review_required": ["Production generation requires VLM paper review."],
        "figure_candidates": [{"id": "figure_overview", "figure_type": "method_overview", "purpose": "Overview of paper-derived modules.", "information_payload_ids": [item["id"] for item in modules], "recommended": True}],
    }


def _normalize_fact(item: Any, fallback_id: str) -> dict:
    raw = item if isinstance(item, dict) else {"statement": str(item or "")}
    result = dict(FACT_DEFAULTS)
    result.update(raw)
    result["id"] = str(result.get("id") or fallback_id)
    result["statement"] = str(result.get("statement") or result.get("name") or "unknown").strip()
    result["status"] = str(result.get("status") or "unknown")
    if result["status"] not in {"required", "optional", "forbidden", "unknown"}:
        result["status"] = "unknown"
    result["importance"] = str(result.get("importance") or "medium")
    try:
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
    except Exception:
        result["confidence"] = 0.5
    result["evidence_ids"] = [str(value) for value in result.get("evidence_ids", []) if str(value).strip()]
    result["must_appear_in_figure"] = bool(result.get("must_appear_in_figure", False))
    result["visual_role"] = str(result.get("visual_role") or "supporting_detail")
    return result


def normalize_review(raw: dict, domain_profile: dict) -> dict:
    result = raw if isinstance(raw, dict) else {}
    result.setdefault("summary", "Universal structured paper review.")
    result["schema_version"] = "2.0"
    result.setdefault("paper_identity", {"title": "unknown", "paper_type": "unknown", "field": "unknown"})
    result["domain_profile"] = domain_profile["id"]
    list_fields = ["research_questions", "central_claims", "inputs", "outputs", "research_objects", "concepts", "modules", "relations", "contributions", "innovations", "assumptions", "limitations", "results", "terminology", "forbidden_inventions", "unknowns"]
    for field in list_fields:
        values = result.get(field) if isinstance(result.get(field), list) else []
        result[field] = [_normalize_fact(item, f"{field}_{index + 1:02d}") for index, item in enumerate(values)]
    for field in ["contradictions", "ambiguities", "human_review_required", "figure_candidates"]:
        result[field] = result.get(field) if isinstance(result.get(field), list) else []
    workflows = result.get("workflows") if isinstance(result.get("workflows"), dict) else {}
    result["workflows"] = {name: [_normalize_fact(item, f"workflow_{name}_{index + 1:02d}") for index, item in enumerate(workflows.get(name, []) if isinstance(workflows.get(name), list) else [])] for name in ["training", "inference", "offline", "online", "feedback"]}
    experiments = result.get("experiments") if isinstance(result.get("experiments"), dict) else {}
    result["experiments"] = {name: [_normalize_fact(item, f"experiment_{name}_{index + 1:02d}") for index, item in enumerate(experiments.get(name, []) if isinstance(experiments.get(name), list) else [])] for name in ["datasets", "settings", "metrics", "baselines", "ablations"]}
    return result


def _expand_evidence(review: dict, parsed: dict) -> None:
    evidence = {item["id"]: item for item in parsed.get("evidence", [])}
    aliases = {item.get("legacy_id"): item["id"] for item in parsed.get("evidence", []) if item.get("legacy_id")}
    def expand(item: dict) -> None:
        item["evidence_ids"] = list(dict.fromkeys(aliases.get(str(value), str(value)) for value in item.get("evidence_ids", [])))
        item["evidence_refs"] = [{"evidence_id": value, "page": evidence[value].get("page"), "section": evidence[value].get("section_hint"), "quote": evidence[value].get("text", "")[:600]} for value in item.get("evidence_ids", []) if value in evidence]
    for field in ["research_questions", "central_claims", "inputs", "outputs", "research_objects", "concepts", "modules", "relations", "contributions", "innovations", "assumptions", "limitations", "results", "terminology", "forbidden_inventions", "unknowns"]:
        for item in review.get(field, []):
            expand(item)
    for items in review.get("workflows", {}).values():
        for item in items:
            expand(item)
    for items in review.get("experiments", {}).values():
        for item in items:
            expand(item)


def expand_review_evidence(review: dict, parsed: dict) -> None:
    """Rebind every fact's evidence refs to the ids it cites in this run's own parse.

    Folding two reviews into one unions the evidence ids each fact cites, so the refs a
    provider or a frozen snapshot wrote no longer describe the union. Rebuilding them
    from the ids keeps every ref pointing at a block of this run's parse, which is what
    ``build_paper_review`` does for the reviews it reads itself.
    """

    _expand_evidence(review, parsed)


def validate_review_coverage(review: dict, parsed: dict, domain_profile: dict, strict: bool) -> dict:
    valid_evidence = {item["id"] for item in parsed.get("evidence", [])}
    errors: list[str] = []
    warnings: list[str] = []
    checked = 0
    grounded = 0
    fact_fields = ["research_questions", "central_claims", "inputs", "outputs", "research_objects", "concepts", "modules", "relations", "contributions", "innovations", "assumptions", "limitations", "results", "terminology"]
    grouped_facts: list[tuple[str, dict]] = []
    for field in fact_fields:
        grouped_facts.extend((field, item) for item in review.get(field, []))
    for workflow_name, items in review.get("workflows", {}).items():
        grouped_facts.extend((f"workflows.{workflow_name}", item) for item in items)
    for experiment_name, items in review.get("experiments", {}).items():
        grouped_facts.extend((f"experiments.{experiment_name}", item) for item in items)
    for field, item in grouped_facts:
            checked += 1
            evidence_ids = item.get("evidence_ids", [])
            invalid = [value for value in evidence_ids if value not in valid_evidence]
            if invalid:
                errors.append(f"{field}:{item.get('id')} references unknown evidence {invalid}")
            if evidence_ids:
                grounded += 1
            elif item.get("status") in {"required", "optional"}:
                errors.append(f"{field}:{item.get('id')} lacks evidence")
    endpoint_fields = ("inputs", "outputs", "research_objects", "concepts", "modules", "innovations")
    module_ids = {
        item.get("id")
        for field in endpoint_fields
        for item in review.get(field, [])
        if isinstance(item, dict) and item.get("id")
    }
    for relation in review.get("relations", []):
        source = relation.get("source_id") or relation.get("source")
        target = relation.get("target_id") or relation.get("target")
        if source not in module_ids or target not in module_ids:
            errors.append(f"relation {relation.get('id')} has unknown endpoint {source}->{target}")
        if not relation.get("evidence_ids"):
            errors.append(f"relation {relation.get('id')} lacks evidence")
    train_statements = {re.sub(r"\s+", " ", item.get("statement", "").lower()).strip() for item in review.get("workflows", {}).get("training", [])}
    infer_statements = {re.sub(r"\s+", " ", item.get("statement", "").lower()).strip() for item in review.get("workflows", {}).get("inference", [])}
    overlap = sorted(value for value in train_statements & infer_statements if value)
    if overlap:
        message = f"training/inference duplicated steps: {overlap}"
        if strict:
            errors.append(message)
        else:
            warnings.append(message)
    populated = {}
    for field in domain_profile.get("required_sections", []):
        value = review.get(field)
        if field == "experiments":
            present = any(review.get("experiments", {}).values())
        elif field == "workflows":
            present = any(review.get("workflows", {}).values())
        else:
            present = bool(value)
        populated[field] = present
        if not present:
            errors.append(f"required review section is empty: {field}")
    if strict and review.get("human_review_required"):
        warnings.append("paper review contains human_review_required items")
    effective_errors = errors if strict else []
    if not strict:
        warnings.extend(errors)
    return {
        "summary": "Universal paper-review coverage and evidence validation.",
        "ok": not effective_errors,
        "strict": strict,
        "domain_profile": domain_profile["id"],
        "required_section_coverage": populated,
        "facts_checked": checked,
        "grounded_facts": grounded,
        "grounding_rate": round(grounded / max(checked, 1), 4),
        "errors": effective_errors,
        "warnings": warnings,
    }


def build_paper_review(parsed: dict, domain_profile: dict, mode: str = "vlm", model: str | None = None, timeout_seconds: int = 300, retries: int = 1, evidence_max_chars: int = 65000, deadline_at: float | None = None) -> tuple[dict, dict]:
    prompt = _review_prompt(parsed, domain_profile, evidence_max_chars=evidence_max_chars)
    metadata = {"summary": "Paper-review execution metadata.", "requested_mode": mode, "mode": mode, "model": None, "warning": None, "prompt": prompt, "provider": {}}
    metadata["context"] = context_stats(prompt, field_names=["domain_profile", "compact_document_index", "prioritized_evidence"])
    if mode == "vlm" and vlm_credentials_available():
        resolved = resolve_vlm_model("RFS_PAPER_REVIEW_MODEL", "RFS_PAPER_TO_IMAGE_MODEL", explicit_model=model)
        try:
            raw = call_vlm_json(prompt, [], model=resolved, timeout=max(10, int(timeout_seconds)), retries=max(0, int(retries)), call_metadata=metadata["provider"], deadline_at=deadline_at)
            metadata["model"] = resolved
            review = normalize_review(raw, domain_profile)
            _expand_evidence(review, parsed)
            return review, metadata
        except Exception as exc:
            metadata.update({"mode": "heuristic_after_vlm_failure", "model": resolved, "warning": str(exc)})
    elif mode == "vlm":
        metadata.update({"mode": "heuristic_without_credentials", "warning": "VLM credentials unavailable"})
    review = normalize_review(_heuristic_review(parsed, domain_profile), domain_profile)
    _expand_evidence(review, parsed)
    return review, metadata
