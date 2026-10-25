"""Evidence-grounded short explanations for paper method relations."""

from __future__ import annotations

import json
from typing import Any

from ..providers import call_vlm_json, resolve_vlm_model


def plan_relation_explanations_vlm(
    method: dict[str, Any], *, model: str | None = None, timeout: int = 180, retries: int = 1
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enrich unlabeled relations without allowing unsupported semantics."""

    relations = [item for item in method.get("relations", []) if isinstance(item, dict)]
    known = {str(item.get("id")): item for item in relations if item.get("id")}
    eligible = [item for item in relations if not str(item.get("label") or "").strip() and item.get("evidence_ids")]
    if not eligible:
        return method, {"summary": "Relation explanation planning skipped.", "status": "pass", "relation_count": len(relations), "eligible_count": 0, "explained_count": sum(bool(item.get("label")) for item in relations), "evidence_coverage": 1.0}
    prompt = (
        "Return JSON only as {relations:[{id,label,evidence_ids,confidence}]}. For each supplied unlabeled paper relation, write a concise 2-6 word scientific operation or transferred-information phrase. Use only the supplied source, target, relation type, and evidence IDs. Never invent algorithms, variables, datasets, causality, or numeric claims. Preserve evidence_ids exactly as a subset of the relation evidence. If evidence is insufficient, return an empty label. RELATIONS="
        + json.dumps(eligible, ensure_ascii=False)
        + " METHOD_NODES="
        + json.dumps(method.get("nodes", []), ensure_ascii=False)
    )
    raw: dict[str, Any] | None = None
    try:
        raw = call_vlm_json(prompt, [], model=resolve_vlm_model("RFS_RELATION_EXPLANATION_MODEL", explicit_model=model), timeout=timeout, retries=retries)
        proposed = {str(item.get("id")): item for item in raw.get("relations", []) if isinstance(item, dict) and item.get("id")}
        enriched = []
        accepted = []
        for relation in relations:
            relation_id = str(relation.get("id") or "")
            candidate = proposed.get(relation_id)
            if not candidate or str(relation.get("label") or "").strip():
                enriched.append(relation)
                continue
            label = " ".join(str(candidate.get("label") or "").split())
            allowed_evidence = {str(value) for value in relation.get("evidence_ids", []) if value}
            evidence = [str(value) for value in candidate.get("evidence_ids", []) if str(value) in allowed_evidence]
            if not label or len(label.split()) > 8 or not evidence:
                enriched.append(relation)
                continue
            updated = {**relation, "label": label, "label_evidence_ids": evidence, "explanation_source": "vlm_evidence_grounded", "explanation_confidence": float(candidate.get("confidence") or 0.0)}
            enriched.append(updated)
            accepted.append({"id": relation_id, "label": label, "evidence_ids": evidence})
        output = {**method, "relations": enriched}
        return output, {"summary": "Evidence-grounded relation explanation plan.", "status": "pass", "mode": "vlm", "relation_count": len(relations), "eligible_count": len(eligible), "explained_count": sum(bool(item.get("label")) for item in enriched), "generated_count": len(accepted), "unexplained_count": sum(not bool(item.get("label")) for item in enriched), "evidence_coverage": round(sum(bool(item.get("label")) and bool(item.get("evidence_ids")) for item in enriched) / max(sum(bool(item.get("label")) for item in enriched), 1), 4), "accepted": accepted, "raw_response": raw}
    except Exception as exc:
        return method, {"summary": "Relation explanation planning failed without inventing fallback labels.", "status": "warning", "mode": "vlm", "relation_count": len(relations), "eligible_count": len(eligible), "generated_count": 0, "error": str(exc), "raw_response": raw}

