"""Independent semantic review and controlled correction of Judge evidence packs."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from rfs.providers.client import call_vlm_json, resolve_vlm_model

from .evidence_schema import EVIDENCE_SCHEMA_VERSION, normalize_structured_evidence, repair_unique_evidence_pages, validate_structured_evidence
from .structured_evidence import build_source_catalog, materialize_source_spans


REVIEW_PROMPT_VERSION = "judge-paper-evidence-semantic-review-v001"


def build_review_prompt(raw: dict[str, Any], current: dict[str, Any]) -> str:
    """Build a paper-grounded semantic audit prompt for one evidence pack."""
    catalog = build_source_catalog(raw)
    sources = "\n".join(json.dumps(item, ensure_ascii=False) for item in catalog)
    return f"""Act as an independent scientific-method evidence reviewer. Audit the CURRENT CONTRACT against the SOURCE CATALOG.
Return exactly one JSON object, no markdown.

Evaluate semantic correctness, not merely JSON validity:
1. method_summary must faithfully describe the paper's proposed method, not background or experiments.
2. core_contributions and entities must include the central method components without invented or peripheral items.
3. relations must have correct endpoints, direction, and meaning.
4. training_flow and inference_flow must not be confused; omit a flow when the paper does not define it.
5. Each claim's evidence_ids must point to source text that actually supports that claim, not just mention similar words.
6. Detect important omissions that would make a framework-figure Judge accept a scientifically incomplete diagram.

Output shape:
{{
  "decision": "pass|revise|reject",
  "summary": "concise audit conclusion",
  "issues": [{{"severity":"major|minor", "category":"summary|contribution|entity|relation|training_flow|inference_flow|evidence|omission", "path":"contract field/item", "explanation":"specific reason", "source_ids":["existing source IDs supporting the review"]}}],
  "corrected_contract": null
}}

If decision is revise, corrected_contract must contain the complete corrected contract. For every evidence span in a corrected contract, use an existing source_id from SOURCE CATALOG and leave page null and quote empty; the program will copy the exact source text and page. Every contribution, entity, relation, training step, and inference step must have non-empty evidence_ids. If a claim cannot be supported, delete it rather than guessing. Relation source/target must reference declared entity_id values. Preserve paper_id {raw.get('paper_id')} and schema_version {EVIDENCE_SCHEMA_VERSION}.
If decision is pass, corrected_contract must be null. Use reject only when the available paper text is insufficient for a trustworthy method contract.

CURRENT CONTRACT:
{json.dumps(current, ensure_ascii=False, indent=2)}

SOURCE CATALOG:
{sources}
"""


def review_one_evidence_pack(
    raw: dict[str, Any],
    current: dict[str, Any],
    *,
    model: str,
    adapter: Callable[..., dict[str, Any]] = call_vlm_json,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    """Review one contract and return the audit, optional valid correction, and errors."""
    metadata: dict[str, Any] = {}
    audit = adapter(build_review_prompt(raw, current), [], model=model, timeout=300, retries=2, call_metadata=metadata)
    if not isinstance(audit, dict):
        return {"decision": "invalid", "provider_call": metadata}, None, ["review output is not an object"]
    audit["provider_call"] = metadata
    audit["model"] = model
    audit["prompt_version"] = REVIEW_PROMPT_VERSION
    decision = str(audit.get("decision") or "").lower()
    if decision == "pass":
        return audit, None, []
    if decision != "revise":
        return audit, None, [f"review decision requires human handling: {decision or '<missing>'}"]
    corrected = audit.get("corrected_contract")
    if not isinstance(corrected, dict):
        return audit, None, ["revise decision missing corrected_contract"]
    corrected = normalize_structured_evidence(corrected)
    source_errors = materialize_source_spans(corrected, build_source_catalog(raw))
    repair_unique_evidence_pages(corrected, raw)
    corrected["schema_version"] = EVIDENCE_SCHEMA_VERSION
    corrected["paper_id"] = str(raw.get("paper_id") or "")
    corrected["extraction"] = {
        **(current.get("extraction") if isinstance(current.get("extraction"), dict) else {}),
        "semantic_review": {"model": model, "prompt_version": REVIEW_PROMPT_VERSION},
    }
    errors = source_errors + validate_structured_evidence(corrected, raw)
    return audit, corrected if not errors else None, sorted(set(errors))


def review_evidence_directory(
    raw_dir: str | Path,
    structured_dir: str | Path,
    *,
    model: str | None = None,
    workers: int = 3,
    adapter: Callable[..., dict[str, Any]] = call_vlm_json,
) -> dict[str, Any]:
    """Audit every valid evidence pack, applying only corrections that revalidate."""
    raw_root = Path(raw_dir)
    root = Path(structured_dir)
    review_root = root.parent / "semantic_reviews"
    archive_root = root.parent / "pre_semantic_review"
    review_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    resolved_model = resolve_vlm_model("RFS_JUDGE_EVIDENCE_REVIEW_MODEL", explicit_model=model)
    paper_ids = []
    for validation_path in sorted(root.glob("*.validation.json")):
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("ok") is True:
            paper_ids.append(validation_path.name.removesuffix(".validation.json"))
    results: list[dict[str, Any]] = []

    def process(aid: str) -> dict[str, Any]:
        review_path = review_root / f"{aid}.json"
        if review_path.exists():
            prior = json.loads(review_path.read_text(encoding="utf-8"))
            if prior.get("status") in {"pass", "revised"}:
                return {"paper_id": aid, "status": prior["status"], "skipped": True}
        raw = json.loads((raw_root / f"{aid}.json").read_text(encoding="utf-8"))
        contract_path = root / f"{aid}.json"
        current = json.loads(contract_path.read_text(encoding="utf-8"))
        try:
            audit, corrected, errors = review_one_evidence_pack(raw, current, model=resolved_model, adapter=adapter)
            decision = str(audit.get("decision") or "invalid")
            status = "pass" if decision == "pass" and not errors else "revised" if corrected is not None else "failed"
            if corrected is not None:
                archive_path = archive_root / f"{aid}.json"
                if not archive_path.exists():
                    shutil.copy2(contract_path, archive_path)
                contract_path.write_text(json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8")
                (root / f"{aid}.validation.json").write_text(json.dumps({"paper_id": aid, "ok": True, "errors": []}, ensure_ascii=False, indent=2), encoding="utf-8")
            record = {"paper_id": aid, "status": status, "decision": decision, "errors": errors, "audit": audit}
        except Exception as exc:  # noqa: BLE001
            record = {"paper_id": aid, "status": "failed", "decision": "error", "errors": [str(exc)]}
        review_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(process, aid): aid for aid in paper_ids}
        for future in as_completed(futures):
            results.append(future.result())
    counts: dict[str, int] = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    report = {"requested": len(paper_ids), "model": resolved_model, "prompt_version": REVIEW_PROMPT_VERSION, "counts": counts, "results": sorted(results, key=lambda item: item["paper_id"])}
    (review_root / "_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
