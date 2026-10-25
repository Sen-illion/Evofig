"""Rebuild the paper document view of a frozen source snapshot.

Phase 3 starts from the immutable D0 snapshot, so a paper must never be parsed or
OCR'd again. This module rebuilds the same ``parsed`` document shape that
``rfs.analysis.paper.document.parse_paper`` returns, reading only the snapshot's
own ``full_text.txt``, ``text_locations.jsonl`` and ``captions.json``.

The rebuild is deterministic: identical frozen inputs always produce identical
evidence ids, section hints and figures, on any machine.
"""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path
from typing import Any

from rfs.analysis.paper.document import evidence_id
from rfs.judge_training.source_snapshot_io import contained_path, sha256_file, streamed_json_records

FROZEN_DOCUMENT_VERSION = "frozen-paper-v1"
CAPTION_KIND = "caption"
PARAGRAPH_KIND = "paragraph"
MAX_HEADING_CHARS = 80
SECTION_HEADINGS = (
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "approach",
    "system",
    "architecture",
    "framework",
    "experiment",
    "experiments",
    "evaluation",
    "results",
    "discussion",
    "conclusion",
    "references",
    "acknowledg",
)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _record_page(record: dict[str, Any]) -> int:
    """Read a page number from either frozen record layout, defaulting to zero."""
    for key in ("page", "pdf_page"):
        value = record.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        text = str(value).strip()
        if text.isdigit():
            return int(text)
    return 0


def looks_like_heading(line: str) -> str | None:
    """Return the heading text when a line is a section heading, else None.

    Frozen records are coarse text blocks, so the heading test stays conservative:
    short standalone lines that start with a known section word and carry no
    sentence punctuation.
    """
    raw = _clean(line)
    if not raw or len(raw) > MAX_HEADING_CHARS or raw.endswith((".", ",", ";", "?")):
        return None
    stripped = raw.strip("0123456789. :()-")
    folded = stripped.casefold()
    if not folded:
        return None
    for heading in SECTION_HEADINGS:
        if folded == heading or folded.startswith(heading + " ") or folded.startswith(heading + ":"):
            return stripped
    return None


FLAT_BLOCK_CHARS = 1200
FLAT_SECTION_WORDS = (
    "Abstract",
    "Introduction",
    "Related work",
    "Background",
    "Results",
    "Discussion",
    "Methods",
    "Materials and methods",
    "Conclusion",
    "Data availability",
    "Acknowledgements",
    "Acknowledgments",
    "References",
)


def _offset(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _heading_offsets(text: str) -> list[tuple[int, str]]:
    """Return the usable section starts of a newline-free frozen document.

    Publisher-flat text has no line breaks, so headings are found by matching
    known section words at word boundaries, ignoring cross references such as
    "see Methods", and keeping only the first occurrence of each word.
    """
    found: list[tuple[int, str]] = []
    for word in FLAT_SECTION_WORDS:
        pattern = re.compile("(?<![A-Za-z])" + re.escape(word) + "(?![a-z])", re.IGNORECASE)
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 5): match.start()].casefold()
            if prefix.endswith("see "):
                continue
            found.append((match.start(), word))
            break
    found.sort()
    references = next((offset for offset, word in found if word.casefold() == "references"), None)
    if references is not None:
        found = [(offset, word) for offset, word in found if offset <= references]
    return found


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split a long paragraph at sentence boundaries so prompts keep real detail."""
    pieces: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        cut = remaining.rfind(". ", 0, max_chars)
        if cut < max_chars // 3:
            cut = remaining.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        pieces.append(remaining[: cut + 1].strip())
        remaining = remaining[cut + 1:].strip()
    if remaining:
        pieces.append(remaining)
    return [piece for piece in pieces if piece]


def split_flat_document(text: str, *, max_chars: int = FLAT_BLOCK_CHARS) -> list[dict[str, Any]]:
    """Split a newline-free frozen document into heading-anchored blocks."""
    offsets = _heading_offsets(text)
    starts = [0] + [offset for offset, _ in offsets]
    titles: list[str | None] = [None] + [word for _, word in offsets]
    blocks: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        for piece in _split_long_text(body, max_chars):
            blocks.append({"section": titles[index], "text": piece})
    return blocks


def paper_directory(snapshot: str | Path, paper: dict[str, Any]) -> Path:
    """Resolve the frozen paper directory and refuse paths that leave the snapshot."""
    pointer = str(paper.get("captions_path") or paper.get("bindings_path") or "")
    if not pointer:
        raise ValueError(f"Paper record has no captions_path or bindings_path: {paper.get('paper_id')!r}")
    return contained_path(snapshot, posixpath.dirname(pointer))


def _evidence_record(page: int, text: str, kind: str, section: str | None, block_id: str) -> dict[str, Any]:
    return {
        "id": evidence_id(page, text, None, kind),
        "page": page,
        "block_id": block_id,
        "bbox": None,
        "section_hint": section,
        "kind": kind,
        "source": "frozen-snapshot",
        "confidence": 1.0,
        "text": text,
        "char_count": len(text),
    }


def _unique_evidence_id(base: str, used: dict[str, int], issued: set[str]) -> str:
    """Keep the first id as-is and number repeated ids (``_2``, ``_3``, ...).

    Flat publisher documents repeat the same paragraph (for example under
    "Data availability" and "References"), so the content hash collides. The
    repeated record is kept, not dropped: it simply gets a stable suffix.
    """
    candidate = base
    index = used.get(base, 1)
    while candidate in issued:
        index += 1
        candidate = "{0}_{1}".format(base, index)
    used[base] = index
    issued.add(candidate)
    return candidate


def build_frozen_document(snapshot: str | Path, paper: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one frozen paper document without re-parsing a PDF or running OCR."""
    directory = paper_directory(snapshot, paper)
    full_text_path = directory / "full_text.txt"
    text_locations_path = directory / "text_locations.jsonl"
    captions_path = directory / "captions.json"
    for required in (full_text_path, text_locations_path, captions_path):
        if not required.is_file():
            raise FileNotFoundError(f"Frozen paper is missing {required.name}: {required}")

    full_text = full_text_path.read_text(encoding="utf-8")
    records = streamed_json_records(text_locations_path)
    captions = json.loads(captions_path.read_text(encoding="utf-8"))
    figures = [item for item in (captions.get("figures") or []) if isinstance(item, dict)] if isinstance(captions, dict) else []

    warnings: list[str] = []
    evidence: list[dict[str, Any]] = []
    headings: list[dict[str, Any]] = []
    seen_headings: set[str] = set()
    page_blocks: dict[int, list[str]] = {}
    section: str | None = None
    used_evidence_ids: dict[str, int] = {}
    issued_evidence_ids: set[str] = set()

    def append_evidence(page: int, text: str, kind: str, section_hint: str | None, block_id: str) -> None:
        record = _evidence_record(page, text, kind, section_hint, block_id)
        record["id"] = _unique_evidence_id(str(record["id"]), used_evidence_ids, issued_evidence_ids)
        evidence.append(record)

    flat_mode = bool(records) and not any(_clean(record.get("text")) for record in records)
    if flat_mode:
        for record in records:
            start = _offset(record.get("char_start"), 0)
            end = _offset(record.get("char_end"), len(full_text))
            block_id = str(record.get("text_asset_id") or "")
            page = _record_page(record)
            for block in split_flat_document(full_text[start:end]):
                heading = block["section"]
                if heading and heading not in seen_headings:
                    seen_headings.add(heading)
                    headings.append({"title": heading, "page": page, "block_id": block_id})
                section = heading or section
                append_evidence(page, block["text"], PARAGRAPH_KIND, heading or section, block_id)
                page_blocks.setdefault(page, []).append(block["text"])
        warnings.append("frozen text locations carry no text; the flat source document was split by section headings")
    else:
        for record in records:
            text = _clean(record.get("text"))
            if not text:
                continue
            page = _record_page(record)
            kind = str(record.get("kind") or PARAGRAPH_KIND)
            block_id = str(record.get("text_asset_id") or record.get("source_location") or "")
            opened = None
            for line in str(record.get("text") or "").splitlines():
                heading = looks_like_heading(line)
                if not heading:
                    continue
                section = heading
                if heading not in seen_headings:
                    seen_headings.add(heading)
                    headings.append({"title": heading, "page": page, "block_id": block_id})
                if opened is None:
                    opened = heading
            append_evidence(page, text, kind, opened or section, block_id)
            page_blocks.setdefault(page, []).append(text)

    figure_index: list[dict[str, Any]] = []
    for figure in figures:
        caption = _clean(figure.get("caption"))
        if not caption:
            continue
        page = _record_page(figure)
        figure_index.append({
            "page": page,
            "text": caption,
            "caption": caption,
            "figure_id": str(figure.get("figure_id") or ""),
            "figure_label": figure.get("figure_label"),
        })
        append_evidence(page, caption, CAPTION_KIND, section, str(figure.get("figure_id") or ""))
        page_blocks.setdefault(page, []).append(caption)

    if not headings:
        warnings.append("no section headings were detected in the frozen text")
    if not figure_index:
        warnings.append("the frozen snapshot carries no figure captions")
    if not evidence:
        warnings.append("the frozen snapshot carries no text records")

    document_index = {"sections": headings, "figures": figure_index, "tables": []}
    evidence_pages = {item["page"] for item in evidence if item["page"]}
    extraction_report = {
        "summary": "Frozen snapshot extraction facts; no PDF parsing or OCR was re-run.",
        "source": "frozen_snapshot",
        "document_version": FROZEN_DOCUMENT_VERSION,
        "readers": {"full_text": "text", "text_locations": "streamed-json", "captions": "json"},
        "warnings": warnings,
        "ocr_pages": [],
        "evidence_char_count": sum(int(item["char_count"]) for item in evidence),
        "evidence_page_count": len(evidence_pages),
        "evidence_page_coverage_ratio": round(len(evidence_pages) / max(1, len(page_blocks)), 4),
    }
    return {
        "summary": "Frozen snapshot paper document rebuilt without re-parsing the PDF.",
        "source_path": str(directory),
        "source_name": str(paper.get("title") or paper.get("paper_id") or directory.name),
        "source_type": ".frozen",
        "source_sha256": sha256_file(full_text_path),
        "loader": "frozen-snapshot",
        "page_count": len(page_blocks),
        "char_count": len(full_text),
        "headings": [item["title"] for item in headings],
        "sections": headings,
        "pages": [{"page": number, "text": " ".join(page_blocks[number])} for number in sorted(page_blocks)],
        "document_index": document_index,
        "evidence": evidence,
        "extraction_report": extraction_report,
        "frozen": {
            "document_version": FROZEN_DOCUMENT_VERSION,
            "paper_id": str(paper.get("paper_id") or ""),
            "captions_path": str(paper.get("captions_path") or ""),
            "bindings_path": str(paper.get("bindings_path") or ""),
        },
    }


def frozen_document_stats(parsed: dict[str, Any]) -> dict[str, Any]:
    """Summarize a rebuilt document for the Pilot graph ledger and reports."""
    evidence = [item for item in (parsed.get("evidence") or []) if isinstance(item, dict)]
    identifiers = [str(item.get("id") or "") for item in evidence]
    return {
        "evidence_count": len(evidence),
        "unique_evidence_ids": len(set(identifiers)),
        "duplicate_evidence_ids": len(identifiers) - len(set(identifiers)),
        "caption_count": sum(1 for item in evidence if item.get("kind") == CAPTION_KIND),
        "heading_count": len(parsed.get("headings") or []),
        "figure_count": len((parsed.get("document_index") or {}).get("figures") or []),
        "page_count": int(parsed.get("page_count") or 0),
        "char_count": int(parsed.get("char_count") or 0),
        "missing_section_hint": sum(1 for item in evidence if not item.get("section_hint")),
        "warnings": list((parsed.get("extraction_report") or {}).get("warnings") or []),
    }


FROZEN_SPECIFICATION_VERSION = "frozen-figure-specification-v1"
SPEC_ENTITY_FIELDS = ("inputs", "outputs", "modules", "innovations")
SPEC_DETAIL_LEVELS = {
    "inputs": "artifact",
    "outputs": "artifact",
    "modules": "module",
    "innovations": "mechanism",
}
SPEC_IMPORTANCE_SCORES = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}
SPEC_WORKFLOW_BUCKETS = ("training", "inference", "offline", "online", "feedback")
SPEC_UNKNOWN_STATUS = "unknown"


def build_frozen_evidence_map(parsed: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the canonical evidence_map shape from a frozen document."""
    source = parsed if isinstance(parsed, dict) else {}
    return {
        "contract": "evidence_map",
        "loader": "frozen-snapshot",
        "source_path": str(source.get("source_path") or ""),
        "source_name": str(source.get("source_name") or ""),
        "source_sha256": str(source.get("source_sha256") or ""),
        "evidence": [dict(item) for item in (source.get("evidence") or []) if isinstance(item, dict)],
    }


def _spec_evidence_ids(item: dict[str, Any]) -> list[str]:
    values = item.get("evidence_ids")
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _spec_entity(item: dict[str, Any], field: str, index: int) -> dict[str, Any]:
    """Project one review fact into the entity shape the method plan uses."""
    identifier = _clean(item.get("id") or item.get("name") or "") or "{field}_{index}".format(field=field, index=index)
    evidence_ids = _spec_evidence_ids(item)
    must_show = bool(item.get("must_appear_in_figure")) and bool(evidence_ids)
    return {
        "id": identifier,
        "label": _clean(item.get("visible_label") or item.get("name") or item.get("statement") or identifier)[:160],
        "role": _clean(item.get("visual_role") or item.get("role") or field),
        "detail_level": SPEC_DETAIL_LEVELS.get(field, "module"),
        "scientific_importance": SPEC_IMPORTANCE_SCORES.get(str(item.get("importance") or "").strip().casefold(), 0.5),
        "visualization_value": 1.0 if must_show else 0.5,
        "must_show": must_show,
        "should_render": must_show,
        "render_priority": 1 if must_show else 2,
        "evidence_ids": evidence_ids,
    }


def _spec_stage(bucket: str, items: list[Any]) -> dict[str, Any]:
    """Turn one review workflow bucket into a stage-level seed entity."""
    evidence_ids: list[str] = []
    statements: list[str] = []
    for item in items:
        if isinstance(item, dict):
            evidence_ids.extend(_spec_evidence_ids(item))
            statements.append(_clean(item.get("visible_label") or item.get("statement") or item.get("id") or ""))
        else:
            statements.append(_clean(item))
    return {
        "id": "stage_{bucket}".format(bucket=bucket),
        "label": bucket.replace("_", " ").title(),
        "role": "stage",
        "detail_level": "stage",
        "scientific_importance": 0.5,
        "visualization_value": 0.5,
        "must_show": False,
        "should_render": False,
        "render_priority": 2,
        "evidence_ids": list(dict.fromkeys(evidence_ids)),
        "seed_statements": [value for value in statements if value][:8],
    }


def _spec_relation(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": _clean(item.get("id") or "") or "relation_{index}".format(index=index),
        "source": _clean(item.get("source_id") or item.get("source") or ""),
        "target": _clean(item.get("target_id") or item.get("target") or ""),
        "type": _clean(item.get("relation_type") or item.get("type") or "related_to"),
        "label": _clean(item.get("statement") or item.get("label") or "")[:200],
        "evidence_ids": _spec_evidence_ids(item),
    }


def build_frozen_figure_specification(parsed: dict[str, Any], review: dict[str, Any] | None, *, paper_id: str = "", domain: str = "", review_mode: str = "") -> dict[str, Any]:
    """Rebuild the method-plan input shape from the stored paper review alone.

    The shipped generation path builds its figure specification with a full
    planning pass over the paper plan. Phase 3 deliberately re-plans nothing: the
    stored review is the only grounded source, so the specification is a
    deterministic seed that keeps every review fact, its id and its evidence ids
    in the shape the method extractor expects. The extractor refines that seed
    with the same review in its prompt.
    """
    source = review if isinstance(review, dict) else {}
    specification: dict[str, Any] = {}
    declared: list[str] = []
    for field in SPEC_ENTITY_FIELDS:
        entities = [
            _spec_entity(item, field, index)
            for index, item in enumerate(source.get(field) or [])
            if isinstance(item, dict)
        ]
        specification[field] = entities
        declared.extend(entity["id"] for entity in entities)
    workflows = source.get("workflows") if isinstance(source.get("workflows"), dict) else {}
    stages = [
        _spec_stage(bucket, list(workflows.get(bucket) or []))
        for bucket in SPEC_WORKFLOW_BUCKETS
        if workflows.get(bucket)
    ]
    specification["stages"] = stages
    declared.extend(stage["id"] for stage in stages)
    declared_ids = {value for value in declared if value}
    relations: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    for index, item in enumerate(source.get("relations") or []):
        if not isinstance(item, dict):
            continue
        relation = _spec_relation(item, index)
        if relation["source"] in declared_ids and relation["target"] in declared_ids:
            relations.append(relation)
        else:
            dropped.append({
                "id": relation["id"],
                "source": relation["source"],
                "target": relation["target"],
                "reason": "undeclared_endpoint",
            })
    specification["relations"] = relations
    claims = [item for item in (source.get("central_claims") or []) if isinstance(item, dict)]
    identity = source.get("paper_identity") if isinstance(source.get("paper_identity"), dict) else {}
    goal = _clean(claims[0].get("statement")) if claims else ""
    if not goal:
        goal = _clean(identity.get("title"))
    specification["figure_goal"] = goal[:400] or "Explain the paper method."
    forbidden: list[str] = []
    for item in source.get("forbidden_inventions") or []:
        text = _clean(item.get("statement") if isinstance(item, dict) else item)
        if text:
            forbidden.append(text)
    specification["forbidden_inventions"] = forbidden
    uncertainties: list[str] = []
    for field in ("inputs", "outputs", "modules", "relations", "innovations", "central_claims"):
        for item in source.get(field) or []:
            if isinstance(item, dict) and str(item.get("status") or "").strip().casefold() == SPEC_UNKNOWN_STATUS:
                uncertainties.append(_clean(item.get("statement") or item.get("id"))[:200])
    specification["uncertainties"] = [value for value in dict.fromkeys(uncertainties) if value]
    specification["overview_panels"] = []
    specification["optional_details"] = []
    specification["seed"] = {
        "specification_version": FROZEN_SPECIFICATION_VERSION,
        "source": "pilot-review-seed",
        "paper_id": str(paper_id or ""),
        "domain": str(domain or ""),
        "review_mode": str(review_mode or ""),
        "review_schema_version": str(source.get("schema_version") or ""),
        "declared_entities": len(declared_ids),
        "dropped_relations": dropped,
        "evidence_count": len((parsed.get("evidence") if isinstance(parsed, dict) else []) or []),
        "document_version": FROZEN_DOCUMENT_VERSION,
        "note": "Phase 3 rebuilds no page-layout plan: this seed only carries the stored review facts, their ids and their evidence ids.",
    }
    return specification


__all__ = [
    "CAPTION_KIND",
    "FROZEN_DOCUMENT_VERSION",
    "FROZEN_SPECIFICATION_VERSION",
    "PARAGRAPH_KIND",
    "build_frozen_document",
    "build_frozen_evidence_map",
    "build_frozen_figure_specification",
    "frozen_document_stats",
    "looks_like_heading",
    "split_flat_document",
    "paper_directory",
]
