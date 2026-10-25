"""Schemas and deterministic validators for paper evidence contracts."""

from __future__ import annotations

from typing import Any
import re


EVIDENCE_SCHEMA_VERSION = "judge-paper-evidence-v001"


def normalize_structured_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize harmless provider field aliases without inventing content."""
    normalized = dict(record)
    spans = []
    for item in normalized.get("evidence_spans", []) if isinstance(normalized.get("evidence_spans"), list) else []:
        value = dict(item) if isinstance(item, dict) else {}
        if not value.get("quote") and value.get("text"):
            value["quote"] = value["text"]
        spans.append(value)
    normalized["evidence_spans"] = spans
    relations = []
    for item in normalized.get("relations", []) if isinstance(normalized.get("relations"), list) else []:
        value = dict(item) if isinstance(item, dict) else {}
        if not value.get("source") and value.get("source_entity_id"):
            value["source"] = value["source_entity_id"]
        if not value.get("target") and value.get("target_entity_id"):
            value["target"] = value["target_entity_id"]
        relations.append(value)
    normalized["relations"] = relations
    return normalized


def repair_unique_evidence_pages(record: dict[str, Any], raw: dict[str, Any]) -> list[str]:
    """Correct a uniquely locatable quote's page without changing its text."""
    pages = {int(item["page"]): str(item.get("text") or "") for item in raw.get("pages", []) if isinstance(item, dict) and item.get("page")}
    compact = lambda value: re.sub(r"\s+", "", value).replace("\u00ad", "").casefold()
    corrections: list[str] = []
    for span in record.get("evidence_spans", []) if isinstance(record.get("evidence_spans"), list) else []:
        if not isinstance(span, dict) or not str(span.get("quote") or "").strip():
            continue
        quote = compact(str(span["quote"]))
        matches = [page for page, text in pages.items() if quote in compact(text)]
        if len(matches) == 1 and span.get("page") != matches[0]:
            old = span.get("page")
            span["page"] = matches[0]
            corrections.append(f"{span.get('evidence_id')}: page {old} -> {matches[0]}")
    return corrections


def empty_structured_evidence(paper_id: str) -> dict[str, Any]:
    """Return the stable empty contract used by VLM extraction and validation."""
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "paper_id": paper_id,
        "method_summary": "",
        "core_contributions": [],
        "entities": [],
        "relations": [],
        "training_flow": [],
        "inference_flow": [],
        "expected_figure_content": {"required_entities": [], "required_relations": [], "optional_entities": [], "must_not_invent": []},
        "evidence_spans": [],
        "uncertainties": [],
        "extraction": {},
    }


def validate_raw_evidence(record: dict[str, Any]) -> list[str]:
    """Validate a local PDF extraction before it is sent to a VLM."""
    errors: list[str] = []
    for field in ("paper_id", "pdf_sha256", "pages", "status"):
        if not record.get(field):
            errors.append(f"missing {field}")
    pages = record.get("pages") if isinstance(record.get("pages"), list) else []
    if pages and any(not isinstance(item, dict) or not item.get("page") for item in pages):
        errors.append("invalid pages")
    if record.get("status") != "success":
        errors.append("raw extraction not successful")
    return errors


def validate_structured_evidence(record: dict[str, Any], raw: dict[str, Any]) -> list[str]:
    """Check entity/relation references and verify quoted evidence against raw pages."""
    errors: list[str] = []
    if record.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        errors.append("invalid schema_version")
    if str(record.get("paper_id") or "") != str(raw.get("paper_id") or ""):
        errors.append("paper_id mismatch")
    spans = record.get("evidence_spans") if isinstance(record.get("evidence_spans"), list) else []
    span_ids = {str(item.get("evidence_id")) for item in spans if isinstance(item, dict) and item.get("evidence_id")}
    def compact(value: str) -> str:
        return re.sub(r"\s+", "", value).replace("\u00ad", "")

    page_text = {int(item["page"]): str(item.get("text") or "") for item in raw.get("pages", []) if isinstance(item, dict) and item.get("page")}
    for span in spans:
        if not isinstance(span, dict):
            errors.append("invalid evidence span")
            continue
        page = span.get("page")
        quote = str(span.get("quote") or "").strip()
        if not quote or not isinstance(page, int) or page not in page_text:
            errors.append(f"invalid evidence span {span.get('evidence_id')}")
        elif compact(quote).casefold() not in compact(page_text[page]).casefold():
            errors.append(f"quote not found: {span.get('evidence_id')}")
    entities = record.get("entities") if isinstance(record.get("entities"), list) else []
    entity_ids = {str(item.get("entity_id")) for item in entities if isinstance(item, dict) and item.get("entity_id")}
    for collection in (record.get("core_contributions", []), entities, record.get("relations", []), record.get("training_flow", []), record.get("inference_flow", [])):
        for item in collection if isinstance(collection, list) else []:
            refs = item.get("evidence_ids", []) if isinstance(item, dict) else []
            if not refs or any(str(ref) not in span_ids for ref in refs):
                errors.append("missing or invalid evidence_ids")
    for relation in record.get("relations", []) if isinstance(record.get("relations"), list) else []:
        if str(relation.get("source") or "") not in entity_ids or str(relation.get("target") or "") not in entity_ids:
            errors.append("relation references unknown entity")
    return sorted(set(errors))
