"""Generate grounded structured paper evidence contracts with a canonical VLM provider."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rfs.providers.client import call_vlm_json, resolve_vlm_model

from .evidence_schema import EVIDENCE_SCHEMA_VERSION, empty_structured_evidence, normalize_structured_evidence, repair_unique_evidence_pages, validate_structured_evidence


PROMPT_VERSION = "judge-paper-evidence-prompt-v003-typed-claims"


def build_source_catalog(raw: dict[str, Any], *, max_chars: int = 60000) -> list[dict[str, Any]]:
    """Build exact page-aware source chunks that a VLM can reference by ID."""
    catalog: list[dict[str, Any]] = []
    used = 0
    for page_item in raw.get("pages", []):
        if not isinstance(page_item, dict) or not page_item.get("page"):
            continue
        page = int(page_item["page"])
        text = str(page_item.get("text") or "")
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        chunk_index = 0
        for paragraph in paragraphs:
            for start in range(0, len(paragraph), 900):
                chunk = paragraph[start : start + 900].strip()
                if not chunk:
                    continue
                if used + len(chunk) > max_chars:
                    return catalog
                chunk_index += 1
                catalog.append({"source_id": f"p{page}_c{chunk_index}", "page": page, "text": chunk})
                used += len(chunk)
    return catalog


def materialize_source_spans(record: dict[str, Any], catalog: list[dict[str, Any]]) -> list[str]:
    """Replace VLM span text and page with the exact selected source chunk."""
    sources = {str(item["source_id"]): item for item in catalog}
    errors: list[str] = []
    spans = record.get("evidence_spans", []) if isinstance(record.get("evidence_spans"), list) else []
    for span in spans:
        if not isinstance(span, dict):
            continue
        source_id = str(span.get("source_id") or "")
        # Some providers copy the catalog ID into evidence_id and omit the
        # explicit source_id field. Accept this unambiguous alias only when it
        # exactly names a catalog entry; never infer from arbitrary text.
        if not source_id and str(span.get("evidence_id") or "") in sources:
            source_id = str(span.get("evidence_id"))
            span["source_id"] = source_id
        source = sources.get(source_id)
        if not source:
            errors.append(f"unknown source_id: {source_id or '<missing>'}")
            continue
        span["page"] = source["page"]
        span["quote"] = source["text"]
    return errors


def build_prompt(raw: dict[str, Any]) -> str:
    """Compile bounded paper text and the strict evidence JSON contract."""
    catalog = build_source_catalog(raw)
    context = "\n".join(json.dumps(item, ensure_ascii=False) for item in catalog)
    template = empty_structured_evidence(str(raw.get("paper_id") or ""))
    template["core_contributions"] = [{"contribution": "supported contribution statement", "evidence_ids": ["ev_example"]}]
    template["evidence_spans"] = [{"evidence_id": "ev_example", "source_id": "p3_c2", "page": None, "quote": ""}]
    return f"""You extract auditable scientific method evidence for a framework-figure Judge.
Return exactly one JSON object, no markdown. Schema version: {EVIDENCE_SCHEMA_VERSION}.

    Rules:
1. Every contribution, entity, relation, training-flow step, and inference-flow step must have non-empty evidence_ids.
2. Every evidence_id must point to evidence_spans. For each span, select exactly one source_id from SOURCE CATALOG. Do not write or alter quote/page: leave quote empty and page null; the program copies exact text and page from source_id.
3. Never invent modules, relations, training stages, inference stages, or claims.
4. If evidence is ambiguous, omit the claim and record it in uncertainties.
5. entity_id values must be stable snake_case. Relation source and target must reference declared entity_id values.
6. Use exactly these relation keys: source, target, relation_type, evidence_ids. Use exactly evidence span keys: evidence_id, source_id, page, quote.
7. expected_figure_content.required_entities and required_relations contain only well-supported central method content.
8. core_contributions must be JSON objects with exactly contribution and evidence_ids; never output contribution strings.

JSON shape:
{json.dumps(template, ensure_ascii=False, indent=2)}

Paper title: {raw.get('title', '')}
Paper abstract: {raw.get('abstract', '')}

SOURCE CATALOG (each line is one selectable JSON source; never invent source_id):
{context}
"""


def generate_d0_structured_evidence(raw_dir: str | Path, selection_path: str | Path, out_dir: str | Path, *, model: str | None = None, limit: int | None = None, workers: int = 1) -> dict[str, Any]:
    """Generate structured contracts with resume, validation, and bounded concurrency."""
    raw_root = Path(raw_dir)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    records = selection.get("records", []) if isinstance(selection, dict) else []
    if limit:
        records = records[:limit]
    resolved_model = resolve_vlm_model("RFS_JUDGE_EVIDENCE_MODEL", explicit_model=model)
    def process(item: dict[str, Any]) -> dict[str, Any]:
        aid = str(item.get("arxiv_id") or "")
        raw_path = raw_root / f"{aid}.json"
        out_path = target / f"{aid}.json"
        validation_path = target / f"{aid}.validation.json"
        if out_path.exists() and validation_path.exists():
            prior = json.loads(validation_path.read_text(encoding="utf-8"))
            if prior.get("ok") is True:
                return {"paper_id": aid, "status": "valid", "skipped": True}
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        metadata: dict[str, Any] = {}
        try:
            result = call_vlm_json(build_prompt(raw), [], model=resolved_model, timeout=240, retries=2, call_metadata=metadata)
            result = normalize_structured_evidence(result)
            source_errors = materialize_source_spans(result, build_source_catalog(raw))
            corrections = repair_unique_evidence_pages(result, raw)
            result["schema_version"] = EVIDENCE_SCHEMA_VERSION
            result["paper_id"] = aid
            result["extraction"] = {"model": resolved_model, "prompt_version": PROMPT_VERSION, "provider_call": metadata, "page_corrections": corrections}
            errors = source_errors + validate_structured_evidence(result, raw)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            validation = {"paper_id": aid, "ok": not errors, "errors": errors}
            validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
            if errors:
                return {"paper_id": aid, "status": "failed", "failure": validation}
            return {"paper_id": aid, "status": "valid", "skipped": False}
        except Exception as exc:  # noqa: BLE001
            failure = {"paper_id": aid, "ok": False, "errors": [str(exc)], "provider_call": metadata}
            return {"paper_id": aid, "status": "failed", "failure": failure}

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(process, item) for item in records]
        for future in as_completed(futures):
            results.append(future.result())
    completed = sum(item["status"] == "valid" for item in results)
    skipped = sum(item.get("skipped") is True for item in results)
    failures = sorted((item["failure"] for item in results if item["status"] == "failed"), key=lambda item: item["paper_id"])
    report = {"selection": str(Path(selection_path).resolve()), "requested": len(records), "completed": completed, "skipped": skipped, "failed_or_invalid": len(failures), "model": resolved_model, "prompt_version": PROMPT_VERSION, "failures": failures}
    (target / "_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
