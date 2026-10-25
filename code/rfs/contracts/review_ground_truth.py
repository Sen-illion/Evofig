from __future__ import annotations

import re
from typing import Any


def _text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("text", "statement", "name", "label", "title", "description", "claim", "step"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug[:48] or "item"


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.casefold())


def _excerpt(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}…"


def _tokens(value: Any) -> set[str]:
    stop = {"the", "and", "for", "with", "from", "into", "that", "this", "are", "was", "were", "has", "have", "via", "only", "shown", "show", "high", "level"}
    return {token for token in re.findall(r"[a-z0-9\u4e00-\u9fff]+", str(value or "").casefold()) if len(token) >= 3 and token not in stop}


def _repair_owner(issue_type: str, instruction: str) -> str:
    structural_prefixes = (
        "ocr_",
        "topology_",
        "scientific_missing_relations",
        "scientific_reversed_relations",
        "scientific_containment_mismatches",
    )
    if issue_type.startswith(structural_prefixes):
        return "overlay_compiler"
    normalized = instruction.casefold()
    structural_terms = (
        "arrow",
        "connector",
        "edge direction",
        "reversed relation",
        "missing relation",
        "missing label",
        "misspelled",
        "duplicate label",
        "forbidden label",
        "unexpected label",
        "ocr",
        "text spelling",
    )
    return "overlay_compiler" if any(term in normalized for term in structural_terms) else "image2_visual_substrate"


def compile_review_ground_truth(plan: dict, evidence_records: list[dict] | None = None) -> dict:
    """Compile a compact, evidence-bearing contract used only for image review.

    The figure specification remains the semantic source of truth. This view
    adds the original evidence excerpts needed by a visual critic to check the
    candidate against the paper instead of judging only a summarized plan.
    """

    specification = plan.get("figure_specification") if isinstance(plan.get("figure_specification"), dict) else {}
    evidence_records = [item for item in (evidence_records or []) if isinstance(item, dict)]
    evidence_index: dict[str, dict] = {}
    for item in evidence_records:
        for key in (item.get("id"), item.get("legacy_id")):
            if key:
                evidence_index[str(key)] = item
    searchable_evidence = [
        (item, _normalized(str(item.get("text") or "")))
        for item in evidence_records
        if _normalized(str(item.get("text") or ""))
    ]

    entity_labels: dict[str, str] = {}
    for field in ("inputs", "modules", "outputs", "innovations"):
        for item in _list(specification.get(field)):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip()
            label = _text(item)
            if item_id and label:
                entity_labels[item_id] = label

    priorities: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add_priority(kind: str, value: Any, *, importance: str = "required", visual_expectation: str | None = None) -> None:
        source_id = ""
        target_id = ""
        if isinstance(value, dict):
            label = _text(value)
            evidence_ids = [str(item) for item in _list(value.get("evidence_ids")) if str(item).strip()]
            source_id = str(value.get("source") or "").strip()
            target_id = str(value.get("target") or "").strip()
            if kind in {"relation", "feedback_loop", "training_flow", "inference_flow"} and (source_id or target_id):
                source = entity_labels.get(source_id, source_id)
                target = entity_labels.get(target_id, target_id)
                relation_type = str(value.get("type") or value.get("relation") or "directed relation").strip()
                label = f"{source} -> {target} [{relation_type}]"
            item_id = str(value.get("id") or "").strip()
            status = str(value.get("status") or importance).strip().casefold()
            explicitly_required = bool(value.get("required") is True or value.get("must_appear_in_figure") or value.get("visible_label_required"))
            explicitly_optional = bool(value.get("required") is False or value.get("optional") or status in {"optional", "unknown", "uncertain"})
            required = explicitly_required or (not explicitly_optional and importance != "optional")
        else:
            label = _text(value)
            evidence_ids = []
            item_id = ""
            required = importance != "optional"
        if not label:
            return
        semantic_kind = "relation" if kind in {"relation", "feedback_loop"} else "flow" if kind in {"training_flow", "inference_flow"} and (source_id or target_id) else "concept"
        key = (semantic_kind, _normalized(label))
        if key in seen:
            return
        seen.add(key)
        excerpts = []
        for evidence_id in evidence_ids:
            record = evidence_index.get(evidence_id)
            if not record:
                continue
            excerpts.append({
                "evidence_id": str(record.get("id") or evidence_id),
                "page": record.get("page"),
                "section": record.get("section_hint") or record.get("section"),
                "text": _excerpt(record.get("text"), 240),
                "source": record.get("source"),
                "confidence": record.get("confidence"),
            })
        link_method = "explicit" if excerpts else "none"
        if not excerpts:
            needle = _normalized(label)
            if len(needle) >= 3:
                matches = []
                needle_tokens = _tokens(label)
                for item, haystack in searchable_evidence:
                    reverse_fragment = len(haystack) >= 8 and len(haystack) / len(needle) >= 0.45 and haystack in needle
                    evidence_tokens = _tokens(item.get("text"))
                    token_coverage = len(needle_tokens & evidence_tokens) / max(1, len(needle_tokens))
                    if needle == haystack:
                        rank = 0
                    elif needle in haystack:
                        rank = 1
                    elif reverse_fragment:
                        rank = 2
                    elif len(needle_tokens) >= 3 and token_coverage >= 0.8:
                        rank = 3
                    else:
                        continue
                    matches.append((rank, len(str(item.get("text") or "")), int(item.get("page") or 0), item))
                matches.sort(key=lambda match: match[:3])
                for _, _, _, record in matches[:2]:
                    record_id = str(record.get("id") or record.get("legacy_id") or "").strip()
                    if record_id and record_id not in evidence_ids:
                        evidence_ids.append(record_id)
                    excerpts.append({
                        "evidence_id": record_id,
                        "page": record.get("page"),
                        "section": record.get("section_hint") or record.get("section"),
                        "text": _excerpt(record.get("text"), 240),
                        "source": record.get("source"),
                        "confidence": record.get("confidence"),
                    })
                if excerpts:
                    link_method = "deterministic_text_match"
        priorities.append({
            "id": item_id or f"GT_{kind.upper()}_{_slug(label)}",
            "kind": kind,
            "label": label,
            "importance": "critical" if importance == "critical" else "required" if required else "optional",
            "required": required,
            "visual_expectation": visual_expectation or ("visible and scientifically correct" if required else "may be visible when useful"),
            "evidence_ids": evidence_ids,
            "evidence": excerpts,
            "evidence_link_method": link_method,
        })

    add_priority("research_problem", specification.get("research_problem"), importance="critical", visual_expectation="the problem context is visually recognizable")
    add_priority("central_claim", specification.get("central_claim"), importance="critical", visual_expectation="the figure's main visual story supports this claim")
    for field, kind, expectation in (
        ("inputs", "input", "shown with the correct semantic role as an input or source"),
        ("modules", "module", "shown as the named paper-specific component"),
        ("relations", "relation", "shown with the correct source, target, and direction"),
        ("feedback_loops", "feedback_loop", "shown as a closed and correctly directed loop"),
        ("outputs", "output", "shown as an output rather than an internal module"),
        ("innovations", "innovation", "the paper's novelty is visually identifiable"),
        ("must_show", "must_show", "explicitly visible in the figure"),
    ):
        for item in _list(specification.get(field)):
            add_priority(kind, item, importance="critical" if kind in {"module", "relation", "feedback_loop"} else "required", visual_expectation=expectation)
    for item in _list(specification.get("training_flow")):
        add_priority("training_flow", item, importance="optional", visual_expectation="when visible, shown only in the training path")
    for item in _list(specification.get("inference_flow")):
        add_priority("inference_flow", item, importance="optional", visual_expectation="when visible, shown only in the inference path")

    for label in _list(specification.get("required_labels")):
        add_priority("required_label", label, importance="required", visual_expectation="appears as exact visible text")
    for label in _list(specification.get("optional_visible_labels")):
        add_priority("optional_label", label, importance="optional", visual_expectation="allowed but not required")

    forbidden = []
    for index, item in enumerate(_list(specification.get("forbidden_inventions")), start=1):
        label = _text(item)
        if label:
            forbidden.append({"id": f"FORBIDDEN_{index:03d}", "label": label, "rule": "must not be asserted or visually implied"})

    unknown = []
    for index, item in enumerate(_list(specification.get("uncertainties")), start=1):
        label = _text(item)
        if label:
            unknown.append({"id": f"UNKNOWN_{index:03d}", "label": label, "rule": "do not resolve by invention"})

    evidence_catalog: dict[str, dict] = {}
    for priority in priorities:
        for item in priority.get("evidence", []):
            evidence_id = str(item.get("evidence_id") or "").strip()
            record = evidence_index.get(evidence_id)
            if evidence_id and record and evidence_id not in evidence_catalog:
                evidence_catalog[evidence_id] = {
                    "evidence_id": evidence_id,
                    "page": record.get("page"),
                    "section": record.get("section_hint") or record.get("section"),
                    "text": _excerpt(record.get("text"), 900),
                    "source": record.get("source"),
                    "confidence": record.get("confidence"),
                }
    required_priorities = [item for item in priorities if item["required"]]
    grounded_required = [item for item in required_priorities if item["evidence"]]
    return {
        "summary": "Paper-evidence ground truth for visual candidate review.",
        "source": "figure_specification + key_evidence",
        "evidence_available": bool(grounded_required),
        "strict_evidence_required": bool(evidence_records and required_priorities),
        "required_priority_count": len(required_priorities),
        "grounded_required_priority_count": len(grounded_required),
        "priorities": priorities,
        "evidence_catalog": list(evidence_catalog.values()),
        "forbidden": forbidden,
        "unknown": unknown,
    }


def build_repair_plan(review: dict, source_candidate_id: str | None = None, round_index: int | None = None) -> dict:
    """Turn critic output into a deterministic, auditable Image2 edit plan."""

    items: list[dict] = []
    seen: set[tuple[str, str]] = set()
    preserve = [str(item).strip() for item in review.get("preserve", []) if str(item).strip()]

    def add(issue_type: str, issue: Any, severity: str, expected_check: str, evidence_ids: list[str] | None = None) -> None:
        if isinstance(issue, dict):
            instruction = str(issue.get("repair") or issue.get("instruction") or issue.get("reason") or issue.get("label") or issue).strip()
            region = issue.get("region") or issue.get("visible_region") or issue.get("target") or "auto-localize"
            ids = [str(item) for item in issue.get("evidence_ids", []) if str(item).strip()]
        else:
            instruction = str(issue).strip()
            region = "auto-localize"
            ids = []
        if not instruction:
            return
        key = (issue_type, instruction.casefold())
        if key in seen:
            return
        seen.add(key)
        items.append({
            "id": f"RP_{len(items) + 1:03d}",
            "region": region,
            "issue_type": issue_type,
            "severity": severity,
            "owner": _repair_owner(issue_type, instruction),
            "evidence_ids": list(dict.fromkeys([*(evidence_ids or []), *ids])),
            "preserve": preserve,
            "instruction": instruction,
            "expected_check": expected_check,
        })

    alignment = review.get("paper_alignment", {}) if isinstance(review.get("paper_alignment"), dict) else {}
    for issue in alignment.get("missing_priorities", []):
        ids = issue.get("evidence_ids", []) if isinstance(issue, dict) else []
        add("paper_missing_priority", issue, "critical", "paper_alignment.missing_priorities no longer contains this priority", ids)
    for issue in alignment.get("evidence_conflicts", []):
        ids = issue.get("evidence_ids", []) if isinstance(issue, dict) else []
        add("paper_evidence_conflict", issue, "critical", "paper_alignment.evidence_conflicts is empty for this claim", ids)
    for issue in alignment.get("unsupported_visual_claims", []):
        add("unsupported_visual_claim", issue, "critical", "paper_alignment.unsupported_visual_claims no longer contains this claim")

    scientific = review.get("scientific", {}) if isinstance(review.get("scientific"), dict) else {}
    for field in ("missing_modules", "missing_relations", "reversed_relations", "invented_items", "role_mismatches", "containment_mismatches"):
        for issue in scientific.get(field, []):
            add(f"scientific_{field}", issue, "critical", f"scientific.{field} is empty for this issue")

    topology = review.get("topology", {}) if isinstance(review.get("topology"), dict) else {}
    for field in ("missing_relations", "reversed_relations", "bypassed_relations", "invented_relations"):
        for issue in topology.get(field, []):
            add(f"topology_{field}", issue, "critical", f"topology.{field} is empty for this connector")

    ocr = review.get("ocr", {}) if isinstance(review.get("ocr"), dict) else {}
    unexpected_field = "unsupported_unexpected_labels" if "unsupported_unexpected_labels" in ocr else "unexpected_labels"
    for field in ("missing_labels", "misspelled_labels", "duplicate_labels", "forbidden_labels_found", unexpected_field):
        for issue in ocr.get(field, []):
            add(f"ocr_{field}", issue, "high", f"ocr.{field} is empty for this label")

    for instruction in review.get("repair", []):
        add("critic_instruction", instruction, "medium", "the originating critic issue is resolved")
    for issue in review.get("hard_errors", []):
        add("hard_error", issue, "critical", "the hard error is absent after full re-review")

    template = review.get("template", {}) if isinstance(review.get("template"), dict) else {}
    if not template.get("passed", False):
        add("template_alignment", "Improve macro-panel alignment, reading order, connector rhythm, and density without changing scientific content.", "medium", "template.passed is true")
    aesthetic = review.get("aesthetic", {}) if isinstance(review.get("aesthetic"), dict) else {}
    if not aesthetic.get("passed", False):
        add("aesthetic_quality", "Improve hierarchy, balance, whitespace, color consistency, mechanism visualization, and publication polish while preserving every correct scientific element.", "medium", "aesthetic.passed is true")
    basic = review.get("basic", {}) if isinstance(review.get("basic"), dict) else {}
    if not basic.get("visual_enrichment_passed", True):
        add("visual_enrichment", "Add paper-grounded mechanism cues and scientific visual detail inside existing regions; do not alter labels or connector topology.", "high", "basic.visual_enrichment_passed is true")

    return {
        "summary": "Machine-executable repair plan compiled from all candidate critics.",
        "source_candidate_id": source_candidate_id,
        "round_index": round_index,
        "item_count": len(items),
        "preserve": preserve,
        "items": items,
    }
