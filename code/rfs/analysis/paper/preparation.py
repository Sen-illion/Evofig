from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from ...utils import ensure_dir, read_json, write_json, write_text
from .document import overview_figure_panel_excerpt, paper_markdown, parse_paper
from ...contracts.paper_completion import augment_contract_from_evidence
from ...contracts.semantic import merge_upstream_inventory, merge_upstream_review, reground_frozen_contract
from .document_cache import DOCUMENT_CACHE_VERSION, read_document_cache, write_document_cache
from ...planning.paper_contract import build_review_grounded_plan, compile_image_prompt, merge_preferences, plan_fast_paper_contract, plan_paper_image, repair_plan_evidence, validate_plan_grounding
from .review import build_paper_review, detect_domain_profile, expand_review_evidence, validate_review_coverage
from ...contracts.review_ground_truth import compile_review_ground_truth


def _load_preferences(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Preferences file does not exist: {source}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Preferences file must contain a JSON object")
    return raw


def _archive_file(source: str | Path, target_dir: Path, target_name: str | None = None) -> str:
    path = Path(source).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (target_name or path.name)
    if target.resolve() != path:
        shutil.copyfile(path, target)
    return str(target)


def _section_summary_markdown(parsed: dict[str, Any]) -> str:
    grouped: dict[str, list[str]] = {}
    for item in parsed.get("evidence", []):
        section = str(item.get("section_hint") or "Unclassified")
        grouped.setdefault(section, []).append(str(item.get("text") or ""))
    lines = ["# Section Summary", ""]
    for section, values in grouped.items():
        combined = " ".join(value.replace("\n", " ") for value in values)
        sentences = [value.strip() for value in combined.split(". ") if value.strip()]
        summary = ". ".join(sentences[:3])[:900].strip()
        lines.extend([f"## {section}", "", summary or "No reliable summary extracted.", ""])
    return "\n".join(lines).strip() + "\n"


def _collect_evidence_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "evidence_ids" and isinstance(child, list):
                found.update(str(item) for item in child if str(item).strip())
            else:
                found.update(_collect_evidence_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_evidence_ids(child))
    return found


def _review_evidence_map(value: Any) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if isinstance(value, dict):
        item_id = str(value.get("id") or "").strip()
        evidence_ids = [str(item) for item in value.get("evidence_ids", []) if str(item).startswith("E")]
        if item_id and evidence_ids:
            mapping[item_id] = evidence_ids
        for child in value.values():
            mapping.update(_review_evidence_map(child))
    elif isinstance(value, list):
        for child in value:
            mapping.update(_review_evidence_map(child))
    return mapping


def expand_plan_evidence(plan: dict[str, Any], paper_review: dict[str, Any], parsed: dict[str, Any]) -> None:
    mapping = _review_evidence_map(paper_review)
    valid = {item["id"] for item in parsed.get("evidence", [])}
    legacy = {item.get("legacy_id"): item["id"] for item in parsed.get("evidence", []) if item.get("legacy_id")}
    def expand(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("evidence_ids"), list):
                expanded = []
                for item in value["evidence_ids"]:
                    key = str(item)
                    if key in valid:
                        expanded.append(key)
                    elif key in legacy:
                        expanded.append(legacy[key])
                    else:
                        expanded.extend(mapping.get(key, []))
                value["evidence_ids"] = list(dict.fromkeys(expanded))
            for child in value.values():
                expand(child)
        elif isinstance(value, list):
            for child in value:
                expand(child)
    expand(plan)


def _item_id(item: dict[str, Any], field: str, index: int) -> str:
    return str(item.get("id") or item.get("name") or item.get("visible_label") or item.get("text") or item.get("statement") or f"{field}_{index}").strip()


def _item_label(item: dict[str, Any]) -> str:
    return str(item.get("visible_label") or item.get("name") or item.get("text") or item.get("statement") or item.get("label") or item.get("id") or "").strip()


def _normalized_label(value: str) -> str:
    return "".join(char for char in value.casefold() if char.isalnum())


def _script_counts(value: str) -> tuple[int, int]:
    cjk = sum(
        "\u3400" <= char <= "\u9fff"
        or "\u3040" <= char <= "\u30ff"
        or "\uac00" <= char <= "\ud7af"
        for char in str(value or "")
    )
    latin = sum(("a" <= char.casefold() <= "z") for char in str(value or ""))
    return cjk, latin


def _repair_cross_script_terminology(spec: dict[str, Any], parsed: dict[str, Any]) -> None:
    terminology = spec.get("terminology") if isinstance(spec.get("terminology"), dict) else {}
    if not terminology:
        return
    corpus = " ".join(str(item.get("text") or "") for item in parsed.get("evidence", []) if isinstance(item, dict))
    normalized_corpus = re.sub(r"\s+", " ", corpus.casefold())
    repaired: dict[str, str] = {}
    for source_term, visible_label in terminology.items():
        source = str(source_term or "").strip()
        visible = str(visible_label or "").strip()
        source_present = bool(source and re.sub(r"\s+", " ", source.casefold()) in normalized_corpus)
        visible_present = bool(visible and re.sub(r"\s+", " ", visible.casefold()) in normalized_corpus)
        source_cjk, source_latin = _script_counts(source)
        visible_cjk, visible_latin = _script_counts(visible)
        changed_script = (source_cjk >= 2 and visible_cjk == 0 and visible_latin >= 2) or (source_latin >= 4 and source_cjk == 0 and visible_cjk >= 2 and visible_latin == 0)
        repaired[source] = source if source_present and not visible_present and changed_script else visible
    spec["terminology"] = repaired


def _repair_cross_script_entity_labels(spec: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    evidence_by_id = {
        str(item.get("id")): str(item.get("text") or "")
        for item in parsed.get("evidence", [])
        if isinstance(item, dict) and item.get("id")
    }
    repaired: list[str] = []
    collections: list[tuple[str, list[Any]]] = [
        (field, spec.get(field, []) if isinstance(spec.get(field), list) else [])
        for field in ("inputs", "modules", "outputs", "innovations")
    ]
    collections.extend(
        (f"panel_graphs.{panel.get('id')}.nodes", panel.get("nodes", []) if isinstance(panel.get("nodes"), list) else [])
        for panel in spec.get("panel_graphs", []) if isinstance(panel, dict)
    )
    for field, items in collections:
        for item in items:
            if not isinstance(item, dict):
                continue
            label = _item_label(item)
            label_cjk, label_latin = _script_counts(label)
            records = [evidence_by_id[str(value)] for value in item.get("evidence_ids", []) if str(value) in evidence_by_id]
            source = " ".join(records)
            source_cjk, source_latin = _script_counts(source)
            if not (label_cjk >= 1 and label_latin >= 2 and source_cjk == 0 and source_latin >= 4):
                continue
            label_without_cjk = re.sub(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+", "", label)
            label_key = _normalized_label(label_without_cjk)
            exact_record = next(
                (
                    str(record).strip()
                    for record in sorted(records, key=len)
                    if str(record).strip()
                    and len(str(record).strip()) <= max(120, len(label) + 24)
                    and _normalized_label(str(record)) == label_key
                ),
                None,
            )
            if exact_record:
                item["name"] = exact_record
                item.pop("visible_label", None)
                repaired.append(f"{item.get('id') or item.get('instance_id')}:{label}->{exact_record}")
                continue
            tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", label)
            if len(tokens) < 2:
                continue
            pattern = r"\b" + r"[^A-Za-z0-9]+".join(re.escape(token) for token in tokens) + r"\b"
            match = re.search(pattern, source, re.IGNORECASE)
            if not match:
                continue
            exact = re.sub(r"\s+", " ", match.group(0)).strip(" .,:;")
            if not exact:
                continue
            item["name"] = exact
            item.pop("visible_label", None)
            repaired.append(f"{item.get('id') or item.get('instance_id')}:{label}->{exact}")
    return repaired


def _ground_exact_visible_entities(spec: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    evidence = [item for item in parsed.get("evidence", []) if isinstance(item, dict) and item.get("id") and item.get("text")]
    novelty_pattern = re.compile(r"\b(?:we\s+(?:introduce|propose|present|develop)|novel|new\s+(?:method|model|system|framework)|contribution)\b|\u672c\u6587\u63d0\u51fa|\u6211\u4eec\u63d0\u51fa|\u9996\u6b21|\u521b\u65b0", re.IGNORECASE)
    overview_pattern = re.compile(r"overview|framework|architecture|pipeline|procedure|pre-training|fine-tuning|system|model|\u6846\u67b6|\u67b6\u6784|\u7cfb\u7edf", re.IGNORECASE)
    overview_pages = {
        int(item.get("page") or 0)
        for item in parsed.get("document_index", {}).get("figures", [])
        if isinstance(item, dict) and overview_pattern.search(str(item.get("caption") or ""))
    }
    grounded: list[str] = []
    for field in ("inputs", "modules", "outputs", "innovations"):
        for index, item in enumerate(spec.get(field, []) if isinstance(spec.get(field), list) else []):
            if not isinstance(item, dict) or item.get("evidence_ids"):
                continue
            label = _item_label(item)
            compact = _normalized_label(label)
            if not compact:
                continue
            matches = []
            for record in evidence:
                record_compact = _normalized_label(str(record.get("text") or ""))
                short_diagram_label = len(compact) < 3 and int(record.get("page") or 0) in overview_pages and record_compact == compact
                if not short_diagram_label and (len(compact) < 3 or compact not in record_compact):
                    continue
                if field == "innovations" and not novelty_pattern.search(str(record.get("text") or "")):
                    continue
                matches.append(str(record["id"]))
            if not matches:
                continue
            item["evidence_ids"] = list(dict.fromkeys(matches))[:3]
            grounded.append(f"{field}[{index}]")
    return grounded


def _repair_relation_evidence_after_grounding(spec: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    """Attach conservative endpoint evidence after late exact-label grounding."""
    endpoints = {
        str(item.get("id")): item
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict) and item.get("id")
    }
    evidence_by_id = {
        str(item.get("id")): item
        for item in parsed.get("evidence", [])
        if isinstance(item, dict) and item.get("id")
    }
    repaired: list[str] = []
    for relation in spec.get("relations", []) if isinstance(spec.get("relations"), list) else []:
        if not isinstance(relation, dict) or relation.get("evidence_ids"):
            continue
        source = endpoints.get(str(relation.get("source") or ""))
        target = endpoints.get(str(relation.get("target") or ""))
        if not source or not target:
            continue
        source_ids = [str(value) for value in source.get("evidence_ids", []) if str(value) in evidence_by_id]
        target_ids = [str(value) for value in target.get("evidence_ids", []) if str(value) in evidence_by_id]
        candidate_ids = list(dict.fromkeys([*source_ids, *target_ids]))
        if not candidate_ids:
            continue
        source_pages = {int(evidence_by_id[value].get("page") or 0) for value in source_ids}
        target_pages = {int(evidence_by_id[value].get("page") or 0) for value in target_ids}
        nearby = bool(source_pages and target_pages and min(abs(left - right) for left in source_pages for right in target_pages) <= 1)
        source_label = _normalized_label(_item_label(source))
        target_label = _normalized_label(_item_label(target))
        bridge_ids = [
            str(item["id"])
            for item in parsed.get("evidence", [])
            if isinstance(item, dict)
            and item.get("id")
            and source_label
            and target_label
            and source_label in _normalized_label(str(item.get("text") or ""))
            and target_label in _normalized_label(str(item.get("text") or ""))
        ][:2]
        if not nearby and not bridge_ids:
            continue
        relation["evidence_ids"] = list(dict.fromkeys([*candidate_ids, *bridge_ids]))
        repaired.append(f"{relation.get('source')}->{relation.get('target')}")
    return repaired


def _prune_out_of_scope_entities(spec: dict[str, Any], parsed: dict[str, Any], completion_report: dict[str, Any]) -> list[str]:
    selected_ids = {str(value) for value in completion_report.get("selected_overview_evidence_ids", []) if str(value)}
    if not selected_ids:
        return []
    evidence_by_id = {
        str(item.get("id")): item
        for item in parsed.get("evidence", [])
        if isinstance(item, dict) and item.get("id")
    }
    anchor_text = " ".join(
        [
            str(spec.get("figure_goal") or ""),
            str(spec.get("research_problem", {}).get("text") or "") if isinstance(spec.get("research_problem"), dict) else str(spec.get("research_problem") or ""),
            str(spec.get("central_claim", {}).get("text") or "") if isinstance(spec.get("central_claim"), dict) else str(spec.get("central_claim") or ""),
            *(str(evidence_by_id[value].get("text") or "") for value in selected_ids if value in evidence_by_id),
        ]
    )
    normalized_anchor = _normalized_label(anchor_text)
    blocked_sections = ("appendix", "experiment", "result", "analysis", "visualization", "ablation", "attention map", "reference", "acknowledg")
    removed_ids: set[str] = set()
    removed_labels: list[str] = []
    for field in ("inputs", "modules", "outputs"):
        kept = []
        for item in spec.get(field, []) if isinstance(spec.get(field), list) else []:
            if not isinstance(item, dict):
                continue
            evidence_ids = [str(value) for value in item.get("evidence_ids", []) if str(value)]
            records = [evidence_by_id[value] for value in evidence_ids if value in evidence_by_id]
            label = _item_label(item)
            label_in_anchor = bool(_normalized_label(label) and _normalized_label(label) in normalized_anchor)
            selected_grounded = any(value in selected_ids for value in evidence_ids)
            only_blocked = bool(records) and all(any(term in str(record.get("section_hint") or "").casefold() for term in blocked_sections) for record in records)
            weak_isolated_label = bool(records) and all(_normalized_label(str(record.get("text") or "")) == _normalized_label(label) for record in records)
            if not selected_grounded and not label_in_anchor and only_blocked and weak_isolated_label:
                removed_ids.add(str(item.get("id") or ""))
                removed_labels.append(label)
                continue
            kept.append(item)
        spec[field] = kept
    if removed_ids:
        spec["relations"] = [
            item for item in spec.get("relations", [])
            if isinstance(item, dict) and str(item.get("source")) not in removed_ids and str(item.get("target")) not in removed_ids
        ]
    return removed_labels


def _remove_ungrounded_innovations(spec: dict[str, Any]) -> list[str]:
    removed_ids: set[str] = set()
    removed_labels: list[str] = []
    kept = []
    for item in spec.get("innovations", []) if isinstance(spec.get("innovations"), list) else []:
        if isinstance(item, dict) and item.get("evidence_ids"):
            kept.append(item)
            continue
        if isinstance(item, dict):
            removed_ids.add(str(item.get("id") or ""))
            removed_labels.append(_item_label(item))
    spec["innovations"] = kept
    if removed_ids:
        spec["relations"] = [
            item for item in spec.get("relations", [])
            if isinstance(item, dict) and str(item.get("source")) not in removed_ids and str(item.get("target")) not in removed_ids
        ]
    return removed_labels


def _remove_input_shortcuts_through_intermediates(spec: dict[str, Any]) -> list[str]:
    inputs = {str(item.get("id")) for item in spec.get("inputs", []) if isinstance(item, dict) and item.get("id")}
    entities = {
        str(item.get("id")): item
        for field in ("modules", "outputs")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict) and item.get("id")
    }
    relations = [item for item in spec.get("relations", []) if isinstance(item, dict)]
    outgoing: dict[str, list[dict[str, Any]]] = {}
    for relation in relations:
        outgoing.setdefault(str(relation.get("source") or ""), []).append(relation)
    removed: list[str] = []
    remove_ids: set[int] = set()
    flow_types = {"data_flow", "encoding", "prediction", "generation_input"}
    for index, direct in enumerate(relations):
        source = str(direct.get("source") or "")
        target = str(direct.get("target") or "")
        if source not in inputs or str(direct.get("type") or "data_flow") not in flow_types:
            continue
        for first in outgoing.get(source, []):
            middle = str(first.get("target") or "")
            if middle == target or middle not in entities:
                continue
            role_text = f"{entities[middle].get('role', '')} {_item_label(entities[middle])}".casefold()
            if not any(term in role_text for term in ("intermediate", "artifact", "patch", "token sequence", "representation")):
                continue
            if any(str(second.get("target") or "") == target and str(second.get("type") or "data_flow") in flow_types for second in outgoing.get(middle, [])):
                remove_ids.add(index)
                removed.append(f"{source}->{target}")
                break
    if remove_ids:
        spec["relations"] = [item for index, item in enumerate(relations) if index not in remove_ids]
    return removed


def _stable_id(prefix: str, value: str, index: int) -> str:
    slug = "_".join(part for part in re.sub(r"[^a-z0-9]+", " ", value.casefold()).split() if part)[:48]
    return f"{prefix}_{slug or index}"


def _statement_object(value: Any, fallback: Any = None) -> dict[str, Any]:
    candidate = value if value is not None else fallback
    if isinstance(candidate, dict):
        item = dict(candidate)
        text = str(item.get("text") or item.get("statement") or "unknown").strip() or "unknown"
        item["text"] = text
        item["evidence_ids"] = list(item.get("evidence_ids")) if isinstance(item.get("evidence_ids"), list) else []
        return item
    if isinstance(candidate, str) and candidate.strip():
        return {"text": candidate.strip(), "evidence_ids": []}
    return {"text": "unknown", "evidence_ids": [], "status": "unknown"}


def _infer_topology(spec: dict[str, Any]) -> str:
    relations = [item for item in spec.get("relations", []) if isinstance(item, dict)]
    if any("feedback" in str(item.get("type") or "").casefold() for item in relations):
        return "feedback"
    inputs = [item for item in spec.get("inputs", []) if isinstance(item, dict)]
    labels = " ".join(_item_label(item).casefold() for item in inputs)
    modalities = sum(term in labels for term in ("image", "text", "audio", "video", "depth", "thermal", "imu"))
    modules = [item for item in spec.get("modules", []) if isinstance(item, dict)]
    encoder_count = sum("encoder" in _item_label(item).casefold() for item in modules)
    if modalities >= 3 or (modalities >= 2 and encoder_count >= 2):
        return "multimodal"
    outgoing: dict[str, int] = {}
    for relation in relations:
        source = str(relation.get("source") or relation.get("source_id") or "")
        outgoing[source] = outgoing.get(source, 0) + 1
    if any(value > 1 for value in outgoing.values()):
        return "branch"
    overview_panels = [item for item in spec.get("modules", []) if isinstance(item, dict) and str(item.get("role") or "") == "overview_panel"]
    if len(overview_panels) >= 3:
        return "dense_multiframe"
    return "linear" if relations else "unknown"


def _detect_dense_overview_panels(parsed: dict[str, Any]) -> dict[str, Any]:
    figures = parsed.get("document_index", {}).get("figures", [])
    overview_pattern = re.compile(r"\b(?:overview|architecture|framework|pipeline|model)\b", re.IGNORECASE)
    figure = next(
        (
            item
            for item in figures
            if isinstance(item, dict)
            and re.match(r"^(?:figure|fig\.)\s*1\b", str(item.get("caption") or ""), re.IGNORECASE)
            and overview_pattern.search(str(item.get("caption") or ""))
        ),
        None,
    )
    if not figure:
        return {"detected": False, "page": None, "panel_labels": [], "panel_count": 0}

    page_number = int(figure.get("page") or 0)
    page = next(
        (
            item
            for item in parsed.get("pages", [])
            if isinstance(item, dict) and int(item.get("page") or 0) == page_number
        ),
        None,
    )
    if not page:
        return {"detected": False, "page": page_number, "panel_labels": [], "panel_count": 0}

    labels: set[str] = set()
    for block in page.get("blocks", []):
        if not isinstance(block, dict):
            continue
        text = re.sub(r"\s+", " ", str(block.get("text") or "")).strip()
        marker = re.fullmatch(r"([a-h])", text, re.IGNORECASE)
        description = re.match(r"^([a-h])\s*[,.:]\s+", text, re.IGNORECASE)
        if marker:
            labels.add(marker.group(1).casefold())
        if description:
            labels.add(description.group(1).casefold())
    caption = re.sub(r"\s+", " ", str(figure.get("caption") or ""))
    labels.update(match.group(1).casefold() for match in re.finditer(r"(?:^|[.;])\s*([a-h])\s*[,.:]\s+", caption, re.IGNORECASE))
    ordered = sorted(labels)
    return {
        "detected": len(ordered) >= 2,
        "page": page_number,
        "panel_labels": ordered,
        "panel_count": len(ordered),
    }


def _overview_caption_blocks(parsed: dict[str, Any], figure: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover a caption that continues across columns on the same page."""
    page_number = int(figure.get("page") or 0)
    page = next(
        (
            item
            for item in parsed.get("pages", [])
            if isinstance(item, dict) and int(item.get("page") or 0) == page_number
        ),
        None,
    )
    if not page:
        return []
    blocks = [item for item in page.get("blocks", []) if isinstance(item, dict)]
    start_id = str(figure.get("block_id") or "")
    start = next((item for item in blocks if str(item.get("id") or "") == start_id), None)
    if not start:
        return []
    start_bbox = start.get("bbox") if isinstance(start.get("bbox"), list) else [0, 0, 0, 0]
    start_y = float(start_bbox[1] or 0)
    page_height = max(1.0, float(page.get("height") or 1))
    max_y = start_y + max(42.0, page_height * 0.09)
    start_font = float(start.get("font_size") or 0)
    start_order = int(start.get("reading_order") or 0)
    caption_blocks = []
    for block in blocks:
        bbox = block.get("bbox") if isinstance(block.get("bbox"), list) else [0, 0, 0, 0]
        block_y = float(bbox[1] or 0)
        block_font = float(block.get("font_size") or 0)
        if int(block.get("reading_order") or 0) < start_order:
            continue
        if not (start_y - 2.0 <= block_y <= max_y):
            continue
        if start_font and block_font and abs(block_font - start_font) > 0.8:
            continue
        caption_blocks.append(block)
    return sorted(caption_blocks, key=lambda item: int(item.get("reading_order") or 0))


def _clean_overview_panel_label(description: str) -> str:
    first_sentence = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", description).strip())[0].strip(" .")
    first_sentence = re.sub(
        r"^(?:flow\s+chart|diagram|schematic|illustration)\s+(?:showing|of)\s+(?:the\s+)?",
        "",
        first_sentence,
        flags=re.IGNORECASE,
    ).strip()
    if len(first_sentence) > 110:
        first_sentence = re.split(r"\s+(?:that|which|where|by|to)\s+", first_sentence, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return first_sentence[:1].upper() + first_sentence[1:] if first_sentence else ""


def _extract_overview_panels(parsed: dict[str, Any], spec: dict[str, Any]) -> list[dict[str, Any]]:
    figures = parsed.get("document_index", {}).get("figures", [])
    figure = next(
        (
            item
            for item in figures
            if isinstance(item, dict)
            and re.match(r"^(?:figure|fig\.)\s*1\b", str(item.get("caption") or ""), re.IGNORECASE)
            and re.search(r"\b(?:overview|architecture|framework|pipeline|model)\b", str(item.get("caption") or ""), re.IGNORECASE)
        ),
        None,
    )
    if not figure:
        return []
    blocks = _overview_caption_blocks(parsed, figure)
    if not blocks:
        return []
    evidence_by_block = {
        str(item.get("block_id") or ""): str(item.get("id") or "")
        for item in parsed.get("evidence", [])
        if isinstance(item, dict) and item.get("block_id") and item.get("id")
    }
    marker_pattern = re.compile(r"(?:^|[.;])\s*([a-h])\s*[,.:]\s*", re.IGNORECASE)
    descriptions: dict[str, list[str]] = {}
    evidence_ids: dict[str, list[str]] = {}
    current: str | None = None
    for block in blocks:
        text = re.sub(r"\s+", " ", str(block.get("text") or "")).strip()
        if not text:
            continue
        matches = list(marker_pattern.finditer(text))
        cursor = 0
        for match in matches:
            prefix = text[cursor:match.start()].strip(" ;,")
            if prefix and match.group(0).lstrip().startswith((".", ";")) and not prefix.endswith((".", ";")):
                prefix += match.group(0).lstrip()[0]
            if current and prefix:
                descriptions.setdefault(current, []).append(prefix)
                evidence_id = evidence_by_block.get(str(block.get("id") or ""))
                if evidence_id:
                    evidence_ids.setdefault(current, []).append(evidence_id)
            current = match.group(1).casefold()
            descriptions.setdefault(current, [])
            evidence_ids.setdefault(current, [])
            cursor = match.end()
        suffix = text[cursor:].strip(" ;,")
        if current and suffix:
            descriptions.setdefault(current, []).append(suffix)
            evidence_id = evidence_by_block.get(str(block.get("id") or ""))
            if evidence_id:
                evidence_ids.setdefault(current, []).append(evidence_id)
    if len(descriptions) < 2:
        return []

    entities = [
        (field, item)
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict) and item.get("id")
    ]
    panels: list[dict[str, Any]] = []
    for label in sorted(descriptions):
        description = re.sub(r"\s+", " ", " ".join(descriptions[label])).strip(" .")
        visible_label = _clean_overview_panel_label(description)
        if not visible_label:
            continue
        panel_evidence = list(dict.fromkeys(evidence_ids.get(label, [])))
        panel_evidence_set = set(panel_evidence)
        entity_ids = [
            str(item.get("id"))
            for _, item in entities
            if panel_evidence_set.intersection(str(value) for value in item.get("evidence_ids", []) if value)
        ]
        panels.append({
            "id": f"panel_{label}",
            "panel_label": label,
            "label": visible_label,
            "description": description,
            "page": int(figure.get("page") or 0),
            "evidence_ids": panel_evidence,
            "entity_ids": list(dict.fromkeys(entity_ids)),
        })
    if len(panels) < 2:
        return []

    # Caption evidence and exact entity evidence often use different blocks.
    # Assign otherwise-unbound entities by semantic panel role so a two-panel
    # overview does not place the complete architecture into the high-level
    # training/usage panel by default.
    assigned_ids = {entity_id for panel in panels for entity_id in panel.get("entity_ids", [])}
    architecture_terms = ("architecture", "internal", "layer", "encoder", "decoder", "network", "model structure")
    high_level_terms = ("overview", "workflow", "pipeline", "pre-training", "pretraining", "training", "usage", "inference", "application")
    training_terms = ("train", "pretrain", "loss", "objective", "synthetic", "supervision")
    inference_terms = ("usage", "inference", "prediction", "application", "forward pass", "test")
    for field, item in entities:
        item_id = str(item.get("id") or "")
        if not item_id or item_id in assigned_ids:
            continue
        label = f"{_item_label(item)} {item.get('role', '')}".casefold()
        item_evidence = {str(value) for value in item.get("evidence_ids", []) if value}
        scored: list[tuple[int, int]] = []
        for index, panel in enumerate(panels):
            panel_text = f"{panel.get('label', '')} {panel.get('description', '')}".casefold()
            score = 0
            if item_evidence.intersection(str(value) for value in panel.get("evidence_ids", []) if value):
                score += 100
            normalized_label = _normalized_label(_item_label(item))
            if normalized_label and normalized_label in _normalized_label(panel_text):
                score += 12
            if field == "modules":
                if any(term in panel_text for term in architecture_terms):
                    score += 5
                if any(term in label for term in training_terms) and any(term in panel_text for term in training_terms):
                    score += 7
                if any(term in label for term in inference_terms) and any(term in panel_text for term in inference_terms):
                    score += 7
            elif field == "inputs":
                if any(term in panel_text for term in high_level_terms):
                    score += 4
                if any(term in panel_text for term in ("input", "data", "train", "pretrain")):
                    score += 3
            elif field == "outputs":
                if any(term in panel_text for term in high_level_terms):
                    score += 4
                if any(term in panel_text for term in inference_terms):
                    score += 3
            scored.append((score, index))
        best_score, best_index = max(scored, default=(0, 0))
        if best_score > 0:
            panels[best_index]["entity_ids"].append(item_id)
            assigned_ids.add(item_id)
    for panel in panels:
        panel["entity_ids"] = list(dict.fromkeys(panel.get("entity_ids", [])))
    return panels


def _merge_isolated_architecture_entities(spec: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    modules = [item for item in spec.get("modules", []) if isinstance(item, dict)]
    relations = [item for item in spec.get("relations", []) if isinstance(item, dict)]
    connected_ids = {
        str(value)
        for relation in relations
        for value in (relation.get("source"), relation.get("target"))
        if str(value or "").strip()
    }
    evidence_by_id = {
        str(item.get("id")): item
        for item in parsed.get("evidence", [])
        if isinstance(item, dict) and item.get("id")
    }
    overview_pages = {
        int(item.get("page") or 0)
        for item in parsed.get("document_index", {}).get("figures", [])
        if isinstance(item, dict) and re.search(r"\b(?:overview|architecture|framework|pipeline)\b", str(item.get("caption") or ""), re.IGNORECASE)
    }
    removed_ids: set[str] = set()
    merged: list[str] = []
    for isolated in modules:
        isolated_id = str(isolated.get("id") or "")
        isolated_label = _item_label(isolated)
        isolated_normalized = _normalized_label(isolated_label)
        if not isolated_id or isolated_id in connected_ids or len(isolated_normalized) < 3:
            continue
        for target in modules:
            target_id = str(target.get("id") or "")
            target_label = _item_label(target)
            target_normalized = _normalized_label(target_label)
            if not target_id or target_id not in connected_ids or target_id == isolated_id or len(target_normalized) < 3:
                continue
            shared_ids = set(str(value) for value in isolated.get("evidence_ids", []) if value) & set(str(value) for value in target.get("evidence_ids", []) if value)
            records = [evidence_by_id[value] for value in shared_ids if value in evidence_by_id]
            relation_record = next(
                (
                    record
                    for record in records
                    if isolated_normalized in _normalized_label(str(record.get("text") or ""))
                    and target_normalized in _normalized_label(str(record.get("text") or ""))
                    and re.search(r"\b(?:based on|built on|using|uses|powered by|adapts?)\b", str(record.get("text") or ""), re.IGNORECASE)
                ),
                None,
            )
            if not relation_record:
                continue
            exact_label_candidates = [
                record
                for record in parsed.get("evidence", [])
                if isinstance(record, dict)
                and int(record.get("page") or 0) in overview_pages
                and 3 <= len(str(record.get("text") or "").strip()) <= 90
                and isolated_normalized in _normalized_label(str(record.get("text") or ""))
                and target_normalized in _normalized_label(str(record.get("text") or ""))
            ]
            exact_label_record = min(
                exact_label_candidates,
                key=lambda record: len(str(record.get("text") or "").strip()),
                default=None,
            )
            if exact_label_record:
                target["name"] = re.sub(r"\s+", " ", str(exact_label_record.get("text") or "")).strip(" .")
                target["evidence_ids"] = list(dict.fromkeys([*target.get("evidence_ids", []), str(exact_label_record.get("id"))]))
            else:
                target["name"] = f"{target_label} ({isolated_label})"
            target["evidence_ids"] = list(dict.fromkeys([*target.get("evidence_ids", []), *isolated.get("evidence_ids", [])]))
            removed_ids.add(isolated_id)
            merged.append(f"{isolated_id}->{target_id}")
            break
    if removed_ids:
        spec["modules"] = [item for item in modules if str(item.get("id") or "") not in removed_ids]
        for relation in relations:
            if str(relation.get("source") or "") in removed_ids:
                relation["source"] = next(value.split("->", 1)[1] for value in merged if value.startswith(f"{relation.get('source')}->"))
            if str(relation.get("target") or "") in removed_ids:
                relation["target"] = next(value.split("->", 1)[1] for value in merged if value.startswith(f"{relation.get('target')}->"))
        spec["relations"] = relations
    return merged


def _supporting_visual_cue(label: str) -> str:
    normalized = _normalized_label(label)
    if "contrastive" in normalized:
        return "paired augmented views converging on an unlabeled comparison objective"
    if "mask" in normalized and ("embedding" in normalized or "token" in normalized):
        return "a compact sequence of feature blocks with several visibly masked positions"
    if "classificationtoken" in normalized or "classtoken" in normalized or normalized in {"cls", "clstoken"}:
        return "one distinct highlighted token block among repeated token blocks"
    if "patch" in normalized or "tile" in normalized or "crop" in normalized:
        return "a compact group of image patches or multi-scale views"
    if "embedding" in normalized:
        return "a compact sequence of feature blocks"
    if "loss" in normalized or "objective" in normalized:
        return "an unlabeled objective or comparison cue"
    return "one unlabeled panel-local scientific mechanism detail"


def _label_like_evidence_text(value: str) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip(".;:")
    if not 2 <= len(text) <= 64 or len(text.split()) > 9:
        return None
    if re.match(r"^(?:to|we|our|the|this|these|those|it|they)\b", text, re.IGNORECASE):
        return None
    if re.search(r"\b(?:which|that|can|could|would|should|serve as|results? in)\b", text, re.IGNORECASE):
        return None
    return text


def _defined_symbol_for_supporting_detail(
    label: str,
    evidence_ids: list[str],
    parsed: dict[str, Any] | None,
) -> str | None:
    if not parsed:
        return None
    evidence = [item for item in parsed.get("evidence", []) if isinstance(item, dict)]
    page_by_id = {str(item.get("id")): int(item.get("page") or 0) for item in evidence if item.get("id")}
    pages = {page_by_id.get(str(value), 0) for value in evidence_ids}
    pages.discard(0)
    normalized_label = _normalized_label(label)
    if not pages or not normalized_label:
        return None
    for page in sorted(pages):
        page_text = " ".join(
            re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
            for item in evidence
            if int(item.get("page") or 0) == page
        )
        for match in re.finditer(
            r"(?P<symbol>\[[A-Za-z0-9_+\-]{1,16}\]|<[A-Za-z0-9_+\-]{1,16}>)\s+(?:is|denotes|represents)\s+(?:the\s+|an?\s+)?(?P<definition>[^.;]{2,80})",
            page_text,
            re.IGNORECASE,
        ):
            definition = _normalized_label(match.group("definition"))
            if normalized_label in definition or definition in normalized_label:
                return match.group("symbol")
    return None


def _demote_unconnected_completion_entities(
    spec: dict[str, Any],
    completion_report: dict[str, Any],
    parsed: dict[str, Any] | None = None,
) -> list[str]:
    added_ids = {str(value) for value in completion_report.get("added_entities", []) if str(value)}
    if not added_ids:
        return []
    connected_ids = {
        str(value)
        for relation in spec.get("relations", []) if isinstance(relation, dict)
        for value in (relation.get("source"), relation.get("target"))
        if str(value or "").strip()
    }
    demoted: list[str] = []
    evidence_by_id = {
        str(item.get("id")): item
        for item in (parsed or {}).get("evidence", [])
        if isinstance(item, dict) and item.get("id")
    }
    supporting = [dict(item) for item in spec.get("supporting_details", []) if isinstance(item, dict)]
    for field in ("inputs", "modules", "outputs", "innovations"):
        kept = []
        for item in spec.get(field, []) if isinstance(spec.get(field), list) else []:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "")
            if item_id in added_ids and item_id not in connected_ids:
                supporting.append({
                    **item,
                    "original_field": field,
                    "original_role": item.get("role"),
                    "role": "panel_local_supporting_detail",
                    "visibility": "unlabeled_visual_detail_not_global_node",
                })
                demoted.append(item_id)
                continue
            kept.append(item)
        spec[field] = kept
    normalized_supporting: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for item in supporting:
        exact_candidates = [
            _label_like_evidence_text(str(evidence_by_id[evidence_id].get("text") or ""))
            for evidence_id in (str(value) for value in item.get("evidence_ids", []) if value)
            if evidence_id in evidence_by_id
        ]
        exact_candidates = [value for value in exact_candidates if value]
        if evidence_by_id and not exact_candidates:
            continue
        if exact_candidates:
            item["name"] = min(exact_candidates, key=len)
        label = _item_label(item)
        if not label:
            continue
        item["visual_cue"] = _supporting_visual_cue(label)
        item["visibility"] = "unlabeled_visual_detail_not_global_node"
        optional_symbol = _defined_symbol_for_supporting_detail(
            label,
            [str(value) for value in item.get("evidence_ids", []) if value],
            parsed,
        )
        if optional_symbol:
            item["optional_visible_label"] = optional_symbol
        key = str(item.get("id") or "").strip() or _normalized_label(label)
        label_key = _normalized_label(label)
        existing = by_key.get(key) or by_key.get(label_key)
        if existing:
            existing["evidence_ids"] = list(dict.fromkeys([
                *existing.get("evidence_ids", []),
                *item.get("evidence_ids", []),
            ]))
            continue
        normalized_supporting.append(item)
        by_key[key] = item
        if label_key:
            by_key[label_key] = item
    spec["supporting_details"] = normalized_supporting
    return demoted


def _complete_from_overview_caption(spec: dict[str, Any], parsed: dict[str, Any]) -> None:
    figures = parsed.get("document_index", {}).get("figures", [])
    figure = next((item for item in figures if re.match(r"^(figure|fig\.)\s*1\b", str(item.get("caption") or ""), re.IGNORECASE) and re.search(r"\b(components?|overview|framework|architecture|pipeline)\b", str(item.get("caption") or ""), re.IGNORECASE)), None)
    figure = figure or next((item for item in figures if re.search(r"\b(overview|framework|architecture|pipeline)\b", str(item.get("caption") or ""), re.IGNORECASE)), None)
    figure = figure or next((item for item in figures if re.match(r"^(figure|fig\.)\s*1\b", str(item.get("caption") or ""), re.IGNORECASE)), None)
    if not figure:
        return
    caption = str(figure.get("caption") or "")
    caption_evidence = next((item["id"] for item in parsed.get("evidence", []) if item.get("kind") == "caption" and item.get("page") == figure.get("page") and str(item.get("text") or "").startswith(str(caption)[:60])), None)
    evidence_ids = [caption_evidence] if caption_evidence else []
    match = re.search(r"(?:interconnected\s+)?components\s*:\s*(.+)", caption, re.IGNORECASE)
    component_ids: dict[str, str] = {}
    if match:
        raw_components = re.split(r",\s+(?=(?:and\s+)?(?:a|an)\s+)", match.group(1))
        existing = {_normalized_label(_item_label(item)) for field in ("modules", "inputs", "outputs", "innovations") for item in (spec.get(field, []) if isinstance(spec.get(field), list) else []) if isinstance(item, dict)}
        for index, raw in enumerate(raw_components):
            phrase = re.sub(r"^(?:and\s+)?(?:a|an)\s+", "", raw.strip(), flags=re.IGNORECASE)
            phrase = re.split(r"\s+(?:that|which|for|by|to)\s+", phrase, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .")
            if not 3 <= len(phrase) <= 80:
                continue
            label = phrase[0].upper() + phrase[1:]
            normalized = _normalized_label(label)
            existing_match = next((item for item in spec.get("modules", []) if isinstance(item, dict) and normalized and (normalized in _normalized_label(_item_label(item)) or _normalized_label(_item_label(item)) in normalized)), None)
            if existing_match:
                component_ids[normalized] = str(existing_match.get("id"))
                continue
            if normalized not in existing:
                item_id = _stable_id("overview", label, index)
                spec.setdefault("modules", []).append({"id": item_id, "name": label, "role": "overview_panel", "evidence_ids": evidence_ids})
                existing.add(normalized)
                component_ids[normalized] = item_id
    dataset_match = re.search(r"\b([A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*)\b", caption) if re.search(r"\bdataset\b|data engine|collecting", caption, re.IGNORECASE) else None
    dataset_item = None
    if dataset_match:
        dataset_label = dataset_match.group(1)
        dataset_item = next((item for item in spec.get("outputs", []) if isinstance(item, dict) and dataset_label.casefold() in _item_label(item).casefold()), None)
        if not dataset_item:
            dataset_item = {"id": _stable_id("dataset", dataset_label, 0), "name": dataset_label, "role": "dataset", "evidence_ids": evidence_ids}
            spec.setdefault("outputs", []).append(dataset_item)
    model_item = next((item for item in spec.get("modules", []) if isinstance(item, dict) and ("segmentation model" in _item_label(item).casefold() or "segment anything model" in _item_label(item).casefold())), None)
    model_item = model_item or next((item for item in spec.get("modules", []) if isinstance(item, dict) and ("model" in _item_label(item).casefold() or "sam" in _item_label(item).casefold())), None)
    engine_item = next((item for item in spec.get("modules", []) if isinstance(item, dict) and "data engine" in _item_label(item).casefold()), None)
    relations = spec.setdefault("relations", [])
    pairs = {(str(item.get("source")), str(item.get("target"))) for item in relations if isinstance(item, dict)}
    if model_item and engine_item and (str(model_item.get("id")), str(engine_item.get("id"))) not in pairs and re.search(r"powers?\s+data\s+annotation|data\s+engine", caption, re.IGNORECASE):
        relations.append({"source": str(model_item.get("id")), "target": str(engine_item.get("id")), "type": "annotation_support", "label": "", "evidence_ids": evidence_ids})
        pairs.add((str(model_item.get("id")), str(engine_item.get("id"))))
    evidence = parsed.get("evidence", [])
    for index, item in enumerate(evidence):
        if not re.search(r"has\s+(?:three|3)\s+stages", str(item.get("text") or ""), re.IGNORECASE):
            continue
        combined = " ".join(str(value.get("text") or "") for value in evidence[index:index + 3])
        stage_match = re.search(r"([A-Za-z]+(?:-[A-Za-z]+)?)\s*,\s*([A-Za-z]+(?:-[A-Za-z]+)?)\s*,\s*(?:and\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)?)\s*\.", combined)
        if not stage_match:
            continue
        stage_evidence = [value["id"] for value in evidence[index:index + 3]]
        stage_items = []
        for stage_index, value in enumerate(stage_match.groups()):
            label = value.strip().title().replace("-", "-")
            existing_stage = next((module for module in spec.get("modules", []) if isinstance(module, dict) and _normalized_label(label) == _normalized_label(_item_label(module))), None)
            if existing_stage:
                stage_items.append(existing_stage)
            else:
                stage = {"id": _stable_id("stage", label, stage_index), "name": label, "role": "data_engine_stage", "evidence_ids": stage_evidence}
                spec.setdefault("modules", []).append(stage)
                stage_items.append(stage)
        for source, target in zip(stage_items, stage_items[1:]):
            pair = (str(source.get("id")), str(target.get("id")))
            if pair not in pairs:
                relations.append({"source": pair[0], "target": pair[1], "type": "stage_transition", "label": "", "evidence_ids": stage_evidence})
                pairs.add(pair)
        if dataset_item and stage_items:
            pair = (str(stage_items[-1].get("id")), str(dataset_item.get("id")))
            if pair not in pairs:
                relations.append({"source": pair[0], "target": pair[1], "type": "data_generation", "label": "", "evidence_ids": stage_evidence + evidence_ids})
        break


def _normalize_contract_entities(raw_items: Any, field: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_items if isinstance(raw_items, list) else []):
        if isinstance(raw, str) and raw.strip():
            item = {"name": raw.strip(), "evidence_ids": []}
        elif isinstance(raw, dict):
            item = dict(raw)
        else:
            continue
        label = _item_label(item)
        if not label:
            continue
        item["id"] = str(item.get("id") or _stable_id(field.rstrip("s"), label, index)).strip()
        item["evidence_ids"] = list(item.get("evidence_ids")) if isinstance(item.get("evidence_ids"), list) else []
        normalized.append(item)
    return normalized


def _normalize_panel_graphs(spec: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    evidence_ids = {
        str(item.get("id"))
        for item in parsed.get("evidence", [])
        if isinstance(item, dict) and item.get("id")
    }
    normalized_panels: list[dict[str, Any]] = []
    removed_nodes: list[str] = []
    removed_relations: list[str] = []
    for panel_index, raw_panel in enumerate(spec.get("panel_graphs", []) if isinstance(spec.get("panel_graphs"), list) else []):
        if not isinstance(raw_panel, dict):
            continue
        panel_id = str(raw_panel.get("id") or f"panel_{panel_index + 1}").strip()
        panel_label = str(raw_panel.get("panel_label") or chr(ord("a") + panel_index)).strip()
        label = str(raw_panel.get("label") or raw_panel.get("description") or panel_label).strip()
        if not panel_id or not label:
            continue
        nodes: list[dict[str, Any]] = []
        seen_instance_ids: set[str] = set()
        for node_index, raw_node in enumerate(raw_panel.get("nodes", []) if isinstance(raw_panel.get("nodes"), list) else []):
            if not isinstance(raw_node, dict):
                continue
            node = dict(raw_node)
            node_label = _item_label(node)
            instance_id = str(node.get("instance_id") or node.get("id") or _stable_id(f"{panel_id}_instance", node_label, node_index)).strip()
            if not node_label or not instance_id or instance_id in seen_instance_ids:
                removed_nodes.append(f"{panel_id}:{instance_id or node_index}")
                continue
            node["instance_id"] = instance_id
            node["name"] = node_label
            node["role"] = str(node.get("role") or "module").strip()
            node["evidence_ids"] = [
                str(value) for value in node.get("evidence_ids", [])
                if str(value) in evidence_ids
            ]
            nodes.append(node)
            seen_instance_ids.add(instance_id)
        node_by_id = {str(item.get("instance_id")): item for item in nodes}
        relations: list[dict[str, Any]] = []
        seen_relations: set[tuple[str, str, str]] = set()
        for raw_relation in raw_panel.get("relations", []) if isinstance(raw_panel.get("relations"), list) else []:
            if not isinstance(raw_relation, dict):
                continue
            source = str(raw_relation.get("source") or raw_relation.get("source_id") or "").strip()
            target = str(raw_relation.get("target") or raw_relation.get("target_id") or "").strip()
            relation_type = str(raw_relation.get("type") or "data_flow").strip()
            key = (source, target, relation_type)
            if not source or not target or source == target or source not in node_by_id or target not in node_by_id or key in seen_relations:
                removed_relations.append(f"{panel_id}:{source}->{target}:{relation_type}")
                continue
            relation = dict(raw_relation)
            relation["source"] = source
            relation["target"] = target
            relation["type"] = relation_type
            relation["label"] = str(relation.get("label") or "").strip()
            relation["evidence_ids"] = [
                str(value) for value in relation.get("evidence_ids", [])
                if str(value) in evidence_ids
            ]
            if not relation["evidence_ids"]:
                relation["evidence_ids"] = list(dict.fromkeys([
                    *node_by_id[source].get("evidence_ids", []),
                    *node_by_id[target].get("evidence_ids", []),
                ]))
            relations.append(relation)
            seen_relations.add(key)
        normalized_panels.append({
            "id": panel_id,
            "panel_label": panel_label,
            "label": label,
            "description": str(raw_panel.get("description") or label).strip(),
            "evidence_ids": [str(value) for value in raw_panel.get("evidence_ids", []) if str(value) in evidence_ids],
            "nodes": nodes,
            "relations": relations,
        })
    spec["panel_graphs"] = normalized_panels
    return {
        "panel_count": len(normalized_panels),
        "node_count": sum(len(item["nodes"]) for item in normalized_panels),
        "relation_count": sum(len(item["relations"]) for item in normalized_panels),
        "removed_nodes": removed_nodes,
        "removed_relations": removed_relations,
    }


def _complete_panel_graphs_from_inventory(spec: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    try:
        inventory = json.loads(overview_figure_panel_excerpt(parsed))
    except (TypeError, ValueError, json.JSONDecodeError):
        inventory = {}
    graphs = [item for item in spec.get("panel_graphs", []) if isinstance(item, dict)] if isinstance(spec.get("panel_graphs"), list) else []
    panels = [item for item in inventory.get("panels", []) if isinstance(item, dict)] if isinstance(inventory, dict) else []
    if len(graphs) < 2 or not panels:
        return {"applied": False, "added_nodes": [], "rebuilt_panels": [], "required_candidate_coverage": None}

    added_nodes: list[str] = []
    removed_nodes: list[str] = []
    rebuilt_panels: list[str] = []
    required_total = 0
    covered_total = 0

    def center(node: dict[str, Any]) -> tuple[float, float]:
        bbox = node.get("bbox_percent") if isinstance(node.get("bbox_percent"), list) and len(node.get("bbox_percent")) == 4 else [0.0, 0.0, 0.0, 0.0]
        return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0

    def relation(source: dict[str, Any], target: dict[str, Any], relation_type: str) -> dict[str, Any]:
        return {
            "source": str(source.get("instance_id")),
            "target": str(target.get("instance_id")),
            "type": relation_type,
            "label": "",
            "evidence_ids": list(dict.fromkeys([*source.get("evidence_ids", []), *target.get("evidence_ids", [])])),
        }

    inventory_by_label = {str(item.get("panel_label") or "").casefold(): item for item in panels}
    for graph in graphs:
        panel_label = str(graph.get("panel_label") or "").casefold()
        panel_inventory = inventory_by_label.get(panel_label)
        if not panel_inventory:
            continue
        required_items = [item for item in panel_inventory.get("label_inventory", []) if isinstance(item, dict) and item.get("required_candidate")]
        required_evidence_ids = {str(item.get("evidence_id") or "") for item in required_items if item.get("evidence_id")}
        nodes = []
        for item in graph.get("nodes", []) if isinstance(graph.get("nodes"), list) else []:
            if not isinstance(item, dict):
                continue
            source_evidence_id = str(item.get("source_evidence_id") or "")
            if source_evidence_id and source_evidence_id not in required_evidence_ids:
                removed_nodes.append(f"{graph.get('id')}:{item.get('instance_id')}")
                continue
            nodes.append(item)
        by_evidence = {
            str(evidence_id): node
            for node in nodes
            for evidence_id in node.get("evidence_ids", [])
            if evidence_id
        }
        required_total += len(required_items)
        occurrence_counts: dict[str, int] = {}
        for item in required_items:
            evidence_id = str(item.get("evidence_id") or "")
            text_value = str(item.get("text") or "").strip()
            role = str(item.get("role_hint") or "module").strip()
            existing = by_evidence.get(evidence_id)
            if existing:
                existing["name"] = text_value
                existing["role"] = role
                existing["bbox_percent"] = item.get("bbox_percent")
                existing["source_evidence_id"] = evidence_id
                covered_total += 1
                continue
            key = _normalized_label(text_value) or role
            occurrence_counts[key] = occurrence_counts.get(key, 0) + 1
            instance_id = f"{graph.get('id')}_{key[:32]}_{occurrence_counts[key]}"
            while any(str(node.get("instance_id")) == instance_id for node in nodes):
                occurrence_counts[key] += 1
                instance_id = f"{graph.get('id')}_{key[:32]}_{occurrence_counts[key]}"
            node = {
                "instance_id": instance_id,
                "name": text_value,
                "role": role,
                "evidence_ids": [evidence_id] if evidence_id else [],
                "bbox_percent": item.get("bbox_percent"),
                "source_evidence_id": evidence_id,
            }
            nodes.append(node)
            if evidence_id:
                by_evidence[evidence_id] = node
            added_nodes.append(f"{graph.get('id')}:{instance_id}")
            covered_total += 1

        objectives = [item for item in nodes if str(item.get("role") or "").casefold() == "objective"]
        if objectives:
            objective_x = min(center(item)[0] for item in objectives)
            for node in nodes:
                if _normalized_label(_item_label(node)) == "ytest" and center(node)[0] < objective_x:
                    node["role"] = "target"

        groups = [item for item in nodes if str(item.get("role") or "").casefold() == "group"]
        modules = [item for item in nodes if str(item.get("role") or "").casefold() == "module"]
        inputs = [item for item in nodes if str(item.get("role") or "").casefold() == "input"]
        targets = [item for item in nodes if str(item.get("role") or "").casefold() in {"target", "conditioning"}]
        outputs = [item for item in nodes if str(item.get("role") or "").casefold() == "output"]
        inferred: list[dict[str, Any]] = []
        for input_node in inputs:
            input_x, input_y = center(input_node)
            candidates = [item for item in modules if center(item)[0] >= input_x - 0.02]
            candidates = candidates or modules
            if candidates:
                target_node = min(candidates, key=lambda item: abs(center(item)[0] - input_x) + 0.45 * abs(center(item)[1] - input_y))
                inferred.append(relation(input_node, target_node, "data_flow"))
        ordered_modules = sorted(modules, key=lambda item: (center(item)[0], center(item)[1]))
        for source, target in zip(ordered_modules, ordered_modules[1:]):
            source_label = _normalized_label(_item_label(source))
            target_label = _normalized_label(_item_label(target))
            source_x, source_y = center(source)
            target_x, target_y = center(target)
            if source_label == target_label or target_x <= source_x or abs(target_y - source_y) > 0.22:
                continue
            inferred.append(relation(source, target, "data_flow"))
        for output_node in outputs:
            output_x, output_y = center(output_node)
            candidates = [item for item in modules if center(item)[0] <= output_x + 0.02]
            candidates = candidates or modules
            if candidates:
                source_node = min(candidates, key=lambda item: abs(output_x - center(item)[0]) + 0.45 * abs(output_y - center(item)[1]))
                inferred.append(relation(source_node, output_node, "prediction"))
        for objective in objectives:
            objective_x, objective_y = center(objective)
            sources = [
                item for item in [*outputs, *targets]
                if abs(center(item)[0] - objective_x) + abs(center(item)[1] - objective_y) <= 0.38
            ]
            if not sources and modules:
                sources = [min(modules, key=lambda item: abs(center(item)[0] - objective_x) + abs(center(item)[1] - objective_y))]
            for source in sources:
                inferred.append(relation(source, objective, "training_objective"))
        seen: set[tuple[str, str, str]] = set()
        deduplicated = []
        for item in inferred:
            key = (str(item.get("source")), str(item.get("target")), str(item.get("type")))
            if key in seen or key[0] == key[1]:
                continue
            seen.add(key)
            deduplicated.append(item)
        if deduplicated:
            graph["relations"] = deduplicated
            rebuilt_panels.append(str(graph.get("id")))
        graph["nodes"] = nodes
        graph["group_instance_ids"] = [str(item.get("instance_id")) for item in groups]

    return {
        "applied": bool(rebuilt_panels or added_nodes),
        "added_nodes": added_nodes,
        "removed_nodes": removed_nodes,
        "rebuilt_panels": rebuilt_panels,
        "required_candidate_count": required_total,
        "covered_required_candidate_count": covered_total,
        "required_candidate_coverage": round(covered_total / max(1, required_total), 4) if required_total else None,
    }


def _deduplicate_contract_entities(spec: dict[str, Any]) -> dict[str, list[str]]:
    replacements: dict[str, str] = {}
    removed: list[str] = []
    for field in ("inputs", "modules", "outputs", "innovations"):
        deduplicated: list[dict[str, Any]] = []
        local_by_label: dict[str, dict[str, Any]] = {}
        for item in spec.get(field, []) if isinstance(spec.get(field), list) else []:
            if not isinstance(item, dict):
                continue
            label = _normalized_label(_item_label(item))
            existing = local_by_label.get(label) if label else None
            if existing:
                existing["evidence_ids"] = list(dict.fromkeys(list(existing.get("evidence_ids", [])) + list(item.get("evidence_ids", []))))
                replacements[str(item.get("id") or "")] = str(existing.get("id") or "")
                removed.append(f"{field}:{item.get('id')}->{existing.get('id')}")
                continue
            deduplicated.append(item)
            if label:
                local_by_label[label] = item
        spec[field] = deduplicated

    for relation in spec.get("relations", []) if isinstance(spec.get("relations"), list) else []:
        if not isinstance(relation, dict):
            continue
        for key in ("source", "source_id", "target", "target_id"):
            value = str(relation.get(key) or "")
            if value in replacements:
                relation[key] = replacements[value]
    return {"removed": removed, "replacements": [f"{source}->{target}" for source, target in replacements.items() if source and target]}


def _simplify_branch_contract(spec: dict[str, Any]) -> dict[str, list[str]]:
    if str(spec.get("topology") or "") != "branch":
        return {"merged_entities": [], "removed_cross_branch_relations": [], "removed_branch_shortcuts": []}

    def group_for(item: dict[str, Any]) -> str | None:
        label = _normalized_label(_item_label(item))
        if label in {"rpn", "regionproposals", "regionproposalnetwork", "regionproposalnetworkrpn", "proposals"}:
            return "proposals"
        if any(value in label for value in ("classificationbranch", "classificationhead", "classprediction", "classpredictions")) or label == "classification":
            return "classification"
        if any(value in label for value in ("boundingboxregression", "boxregression", "boxbranch", "boundingboxoffset", "boundingboxpredictions")):
            return "box"
        if label in {"maskbranch", "binarymask", "validsegmentationmask", "maskprediction"}:
            return "mask"
        return None

    preference = {
        "proposals": ("rpn", "regionproposals"),
        "classification": ("classification", "classificationbranch", "classprediction"),
        "box": ("boundingboxregression", "boxbranch", "boundingboxpredictions", "boundingboxoffset"),
        "mask": ("maskbranch", "validsegmentationmask", "binarymask"),
    }
    records: list[tuple[str, dict[str, Any], str]] = []
    for field in ("inputs", "modules", "outputs"):
        for item in spec.get(field, []) if isinstance(spec.get(field), list) else []:
            if isinstance(item, dict) and (group := group_for(item)):
                records.append((field, item, group))

    replacements: dict[str, str] = {}
    removed_ids: set[str] = set()
    merged_entities: list[str] = []
    for group in ("proposals", "classification", "box", "mask"):
        members = [(field, item) for field, item, item_group in records if item_group == group]
        if len(members) <= 1:
            continue

        def rank(record: tuple[str, dict[str, Any]]) -> tuple[int, int]:
            field, item = record
            label = _normalized_label(_item_label(item))
            preferred = preference[group]
            label_rank = next((index for index, value in enumerate(preferred) if label == value), len(preferred))
            field_rank = 0 if (group == "classification" and field == "outputs") or (group != "classification" and field == "modules") else 1
            return label_rank, field_rank

        canonical_field, canonical = min(members, key=rank)
        canonical_id = str(canonical.get("id") or "")
        for field, item in members:
            item_id = str(item.get("id") or "")
            if item is canonical:
                continue
            canonical["evidence_ids"] = list(dict.fromkeys([*canonical.get("evidence_ids", []), *item.get("evidence_ids", [])]))
            if item_id and canonical_id:
                replacements[item_id] = canonical_id
                removed_ids.add(item_id)
                merged_entities.append(f"{item_id}->{canonical_id}")

    if removed_ids:
        for field in ("inputs", "modules", "outputs"):
            spec[field] = [item for item in spec.get(field, []) if not (isinstance(item, dict) and str(item.get("id") or "") in removed_ids)]

    removed_cross_branch_relations: list[str] = []
    deduplicated: list[dict[str, Any]] = []
    seen_relations: dict[tuple[str, str, str], dict[str, Any]] = {}
    endpoints = {
        str(item.get("id")): item
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict) and item.get("id")
    }
    for relation in spec.get("relations", []) if isinstance(spec.get("relations"), list) else []:
        if not isinstance(relation, dict):
            continue
        source = replacements.get(str(relation.get("source") or ""), str(relation.get("source") or ""))
        target = replacements.get(str(relation.get("target") or ""), str(relation.get("target") or ""))
        if not source or not target or source == target or source not in endpoints or target not in endpoints:
            continue
        source_group = group_for(endpoints[source])
        target_group = group_for(endpoints[target])
        branch_groups = {"classification", "box", "mask"}
        if source_group in branch_groups and target_group in branch_groups and source_group != target_group:
            removed_cross_branch_relations.append(f"{source}->{target}")
            continue
        relation["source"], relation["target"] = source, target
        key = (source, target, str(relation.get("type") or "data_flow"))
        if key in seen_relations:
            existing = seen_relations[key]
            existing["evidence_ids"] = list(dict.fromkeys([*existing.get("evidence_ids", []), *relation.get("evidence_ids", [])]))
            continue
        seen_relations[key] = relation
        deduplicated.append(relation)
    branch_head_ids = {
        str(item.get("id"))
        for item in spec.get("outputs", []) if isinstance(spec.get("outputs"), list)
        if isinstance(item, dict)
        and item.get("id")
        and (group_for(item) in branch_groups or "outputhead" in _normalized_label(str(item.get("role") or "")))
    }
    source_head_counts: dict[str, set[str]] = {}
    for relation in deduplicated:
        source, target = str(relation.get("source") or ""), str(relation.get("target") or "")
        if target in branch_head_ids:
            source_head_counts.setdefault(source, set()).add(target)
    branch_source = next(
        (
            source for source, targets in sorted(source_head_counts.items(), key=lambda item: (-len(item[1]), item[0]))
            if len(targets) >= 2
        ),
        None,
    )
    removed_branch_shortcuts: list[str] = []
    if branch_source:
        adjacency: dict[str, set[str]] = {}
        for relation in deduplicated:
            source, target = str(relation.get("source") or ""), str(relation.get("target") or "")
            if source and target and target not in branch_head_ids:
                adjacency.setdefault(source, set()).add(target)

        def reaches_branch_source(start: str) -> bool:
            pending = [start]
            visited: set[str] = set()
            while pending:
                current = pending.pop()
                if current == branch_source:
                    return True
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(adjacency.get(current, set()) - visited)
            return False

        direct_branch_pairs = {
            (str(item.get("source") or ""), str(item.get("target") or ""))
            for item in deduplicated
        }
        kept_relations: list[dict[str, Any]] = []
        for relation in deduplicated:
            source, target = str(relation.get("source") or ""), str(relation.get("target") or "")
            bypasses_branch_point = (
                target in branch_head_ids
                and source != branch_source
                and (branch_source, target) in direct_branch_pairs
                and reaches_branch_source(source)
            )
            if bypasses_branch_point:
                removed_branch_shortcuts.append(f"{source}->{target}")
                continue
            kept_relations.append(relation)
        deduplicated = kept_relations

    spec["relations"] = deduplicated
    return {
        "merged_entities": merged_entities,
        "removed_cross_branch_relations": removed_cross_branch_relations,
        "removed_branch_shortcuts": list(dict.fromkeys(removed_branch_shortcuts)),
    }


def _simplify_multimodal_contract(spec: dict[str, Any]) -> dict[str, Any]:
    modality_names = {"image", "images", "text", "audio", "video", "depth", "thermal", "imu"}
    modality_inputs = []
    modality_by_name: dict[str, dict[str, Any]] = {}
    for item in spec.get("inputs", []) if isinstance(spec.get("inputs"), list) else []:
        if not isinstance(item, dict):
            continue
        normalized = _normalized_label(_item_label(item))
        if normalized not in modality_names:
            continue
        key = "image" if normalized == "images" else normalized
        if key in modality_by_name:
            canonical = modality_by_name[key]
            canonical["evidence_ids"] = list(dict.fromkeys([*canonical.get("evidence_ids", []), *item.get("evidence_ids", [])]))
            continue
        modality_by_name[key] = item
        modality_inputs.append(item)
    all_entities = [
        (field, item)
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict)
    ]
    encoder_candidates = [
        (field, item) for field, item in all_entities
        if "modalityencoder" in _normalized_label(_item_label(item)) or "encodersforeachmodality" in _normalized_label(_item_label(item))
    ]
    joint_candidates = [
        (field, item) for field, item in all_entities
        if "embedding" in _normalized_label(_item_label(item))
        and any(term in _normalized_label(_item_label(item)) for term in ("joint", "common", "shared"))
    ]
    emergent_candidates = [
        (field, item) for field, item in all_entities
        if ("emergent" in _normalized_label(_item_label(item)) and "align" in _normalized_label(_item_label(item)))
        or "bindingproperty" in _normalized_label(_item_label(item))
        or "bindingagent" in _normalized_label(_item_label(item))
    ]
    if len(modality_inputs) < 3 or not encoder_candidates or not joint_candidates or not emergent_candidates:
        return {"applied": False, "removed_entities": [], "rebuilt_relations": []}

    def choose(candidates: list[tuple[str, dict[str, Any]]], preferred: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
        return min(
            candidates,
            key=lambda record: next(
                (index for index, value in enumerate(preferred) if _normalized_label(_item_label(record[1])) == value),
                len(preferred),
            ),
        )

    _, encoder = choose(encoder_candidates, ("modalityencoders", "encodersforeachmodality"))
    _, joint = choose(joint_candidates, ("jointembeddingspace", "singlejointembeddingspace", "commonembeddingspace", "sharedrepresentationspace"))
    _, emergent = choose(emergent_candidates, ("emergentcrossmodalalignment", "emergentalignment", "bindingproperty", "bindingagent"))
    encoder["name"] = "Modality Encoders"
    encoder["role"] = "module group"
    joint["name"] = "Joint Embedding Space"
    joint["role"] = "shared representation"
    emergent["name"] = "Emergent Cross-modal Alignment"
    emergent["role"] = "output"

    for canonical, candidates in ((encoder, encoder_candidates), (joint, joint_candidates), (emergent, emergent_candidates)):
        canonical["evidence_ids"] = list(dict.fromkeys(
            value for _, item in candidates for value in item.get("evidence_ids", [])
        ))

    keep_ids = {str(item.get("id")) for item in modality_inputs + [encoder, joint, emergent] if item.get("id")}
    removed_entities = [
        str(item.get("id")) for _, item in all_entities if item.get("id") and str(item.get("id")) not in keep_ids
    ]
    spec["inputs"] = modality_inputs
    spec["modules"] = [encoder, joint]
    spec["outputs"] = [emergent]
    spec["innovations"] = []

    encoder_id = str(encoder.get("id"))
    joint_id = str(joint.get("id"))
    emergent_id = str(emergent.get("id"))
    relations = []
    for item in modality_inputs:
        relations.append({
            "source": str(item.get("id")),
            "target": encoder_id,
            "type": "encoding",
            "label": "",
            "evidence_ids": list(dict.fromkeys([*item.get("evidence_ids", []), *encoder.get("evidence_ids", [])])),
        })
    relations.extend([
        {
            "source": encoder_id,
            "target": joint_id,
            "type": "alignment",
            "label": "",
            "evidence_ids": list(dict.fromkeys([*encoder.get("evidence_ids", []), *joint.get("evidence_ids", [])])),
        },
        {
            "source": joint_id,
            "target": emergent_id,
            "type": "enables",
            "label": "",
            "evidence_ids": list(dict.fromkeys([*joint.get("evidence_ids", []), *emergent.get("evidence_ids", [])])),
        },
    ])
    spec["relations"] = relations
    spec["topology"] = "multimodal"
    return {
        "applied": True,
        "removed_entities": removed_entities,
        "rebuilt_relations": [f"{item['source']}->{item['target']}" for item in relations],
    }


def _simplify_dense_multiframe_contract(spec: dict[str, Any]) -> dict[str, Any]:
    all_entities = [
        (field, item)
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict)
    ]

    def candidates(predicate: Callable[[str, dict[str, Any]], bool]) -> list[tuple[str, dict[str, Any]]]:
        return [(field, item) for field, item in all_entities if predicate(_normalized_label(_item_label(item)), item)]

    promptable = candidates(lambda label, item: "segmentationtask" in label and "prompt" in label)
    sam_model = candidates(lambda label, item: "segmentanythingmodel" in label or "segmentationmodelsam" in label or label == "sam")
    data_engine = candidates(lambda label, item: label == "dataengine")
    image = candidates(lambda label, item: label in {"image", "images", "inputimage"} and item in spec.get("inputs", []))
    prompt = candidates(lambda label, item: label in {"prompt", "inputprompts", "segmentationprompt", "sparseprompts", "denseprompts"} and item in spec.get("inputs", []))
    image_encoder = candidates(lambda label, item: label.endswith("imageencoder") and "prompt" not in label)
    prompt_encoder = candidates(lambda label, item: label.endswith("promptencoder"))
    mask_decoder = candidates(lambda label, item: label.endswith("maskdecoder"))
    valid_mask = candidates(lambda label, item: label in {"validsegmentationmask", "validmask", "validmasks"})
    assisted = candidates(lambda label, item: label == "assistedmanual")
    semi = candidates(lambda label, item: label == "semiautomatic")
    fully = candidates(lambda label, item: label == "fullyautomatic")
    sa1b = candidates(lambda label, item: label == "sa1b")
    groups = (promptable, sam_model, data_engine, image, prompt, image_encoder, prompt_encoder, mask_decoder, valid_mask, assisted, semi, fully, sa1b)
    if any(not group for group in groups):
        return {"applied": False, "removed_entities": [], "rebuilt_relations": []}

    def merge(group: list[tuple[str, dict[str, Any]]], name: str, role: str) -> dict[str, Any]:
        canonical = group[0][1]
        canonical["name"] = name
        canonical["role"] = role
        canonical["evidence_ids"] = list(dict.fromkeys(value for _, item in group for value in item.get("evidence_ids", [])))
        return canonical

    promptable_item = merge(promptable, "Promptable Segmentation Task", "overview_panel")
    sam_item = merge(sam_model, "Segment Anything Model", "overview_panel")
    data_engine_item = merge(data_engine, "Data Engine", "overview_panel")
    image_item = merge(image, "Image", "input")
    prompt_item = merge(prompt, "Prompt", "input")
    image_encoder_item = merge(image_encoder, "Image Encoder", "module")
    prompt_encoder_item = merge(prompt_encoder, "Prompt Encoder", "module")
    mask_decoder_item = merge(mask_decoder, "Mask Decoder", "module")
    valid_mask_item = merge(valid_mask, "Valid Segmentation Mask", "output")
    assisted_item = merge(assisted, "Assisted-manual", "data_engine_stage")
    semi_item = merge(semi, "Semi-automatic", "data_engine_stage")
    fully_item = merge(fully, "Fully Automatic", "data_engine_stage")
    sa1b_item = merge(sa1b, "SA-1B", "dataset")

    kept = [
        promptable_item, sam_item, data_engine_item, image_item, prompt_item,
        image_encoder_item, prompt_encoder_item, mask_decoder_item, valid_mask_item,
        assisted_item, semi_item, fully_item, sa1b_item,
    ]
    keep_ids = {str(item.get("id")) for item in kept if item.get("id")}
    removed_entities = [str(item.get("id")) for _, item in all_entities if item.get("id") and str(item.get("id")) not in keep_ids]
    spec["inputs"] = [image_item, prompt_item]
    spec["modules"] = [
        promptable_item, sam_item, data_engine_item,
        image_encoder_item, prompt_encoder_item, mask_decoder_item,
        assisted_item, semi_item, fully_item,
    ]
    spec["outputs"] = [valid_mask_item, sa1b_item]
    spec["innovations"] = []

    def relation(source: dict[str, Any], target: dict[str, Any], relation_type: str) -> dict[str, Any]:
        return {
            "source": str(source.get("id")),
            "target": str(target.get("id")),
            "type": relation_type,
            "label": "",
            "evidence_ids": list(dict.fromkeys([*source.get("evidence_ids", []), *target.get("evidence_ids", [])])),
        }

    relations = [
        relation(image_item, image_encoder_item, "data_flow"),
        relation(prompt_item, prompt_encoder_item, "data_flow"),
        relation(image_encoder_item, mask_decoder_item, "feature_flow"),
        relation(prompt_encoder_item, mask_decoder_item, "conditioning"),
        relation(mask_decoder_item, valid_mask_item, "data_flow"),
        relation(assisted_item, semi_item, "stage_transition"),
        relation(semi_item, fully_item, "stage_transition"),
        relation(fully_item, sa1b_item, "data_generation"),
        relation(sam_item, data_engine_item, "annotation_support"),
    ]
    spec["relations"] = relations
    spec["topology"] = "dense_multiframe"
    return {
        "applied": True,
        "removed_entities": removed_entities,
        "rebuilt_relations": [f"{item['source']}->{item['target']}" for item in relations],
    }


def _simplify_feedback_contract(spec: dict[str, Any]) -> dict[str, Any]:
    if str(spec.get("topology") or "") != "feedback":
        return {"applied": False, "removed_entities": [], "rebuilt_relations": []}
    all_entities = [
        (field, item)
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict)
    ]

    def find(labels: set[str]) -> list[tuple[str, dict[str, Any]]]:
        return [(field, item) for field, item in all_entities if _normalized_label(_item_label(item)) in labels]

    groups = {
        "input": find({"input", "inputx", "taskprompt", "taskinstruction"}),
        "generate": find({"generate", "init", "generator"}),
        "initial": find({"initialoutput"}),
        "feedback": find({"feedback", "feedbackprovider"}),
        "self_feedback": find({"selffeedback", "selffeedbacknl"}),
        "refine": find({"refine", "iteraterefine", "refiner"}),
        "refined": find({"refinedoutput"}),
        "model": find({"modelm"}),
    }
    if any(not groups[key] for key in ("input", "generate", "initial", "feedback", "self_feedback", "refine", "refined")):
        return {"applied": False, "removed_entities": [], "rebuilt_relations": []}

    def merge(group: list[tuple[str, dict[str, Any]]], name: str, role: str) -> dict[str, Any]:
        canonical = group[0][1]
        canonical["name"] = name
        canonical["role"] = role
        canonical["evidence_ids"] = list(dict.fromkeys(value for _, item in group for value in item.get("evidence_ids", [])))
        return canonical

    input_item = merge(groups["input"], "input x", "input")
    generate_item = merge(groups["generate"], "Generate", "generator")
    initial_item = merge(groups["initial"], "Initial Output", "artifact")
    feedback_item = merge(groups["feedback"], "FEEDBACK", "feedback module")
    self_feedback_item = merge(groups["self_feedback"], "Self-Feedback", "feedback artifact")
    refine_item = merge(groups["refine"], "REFINE", "refiner")
    refined_item = merge(groups["refined"], "Refined output", "output")
    model_item = merge(groups["model"], "Model M", "shared model") if groups["model"] else None
    kept = [input_item, generate_item, initial_item, feedback_item, self_feedback_item, refine_item, refined_item]
    if model_item:
        kept.append(model_item)
    keep_ids = {str(item.get("id")) for item in kept if item.get("id")}
    removed_entities = [str(item.get("id")) for _, item in all_entities if item.get("id") and str(item.get("id")) not in keep_ids]
    spec["inputs"] = [input_item]
    spec["modules"] = [item for item in (model_item, generate_item, initial_item, feedback_item, self_feedback_item, refine_item) if item]
    spec["outputs"] = [refined_item]
    spec["innovations"] = []

    def relation(source: dict[str, Any], target: dict[str, Any], relation_type: str) -> dict[str, Any]:
        return {
            "source": str(source.get("id")),
            "target": str(target.get("id")),
            "type": relation_type,
            "label": "iterate" if relation_type == "feedback_loop" else "",
            "evidence_ids": list(dict.fromkeys([*source.get("evidence_ids", []), *target.get("evidence_ids", [])])),
        }

    relations = [
        relation(input_item, generate_item, "data_flow"),
        relation(generate_item, initial_item, "data_flow"),
        relation(initial_item, feedback_item, "evaluation"),
        relation(feedback_item, self_feedback_item, "feedback"),
        relation(initial_item, refine_item, "revision_input"),
        relation(self_feedback_item, refine_item, "feedback"),
        relation(refine_item, refined_item, "data_flow"),
        relation(refined_item, feedback_item, "feedback_loop"),
    ]
    spec["relations"] = relations
    spec["feedback_loops"] = [relations[-1]]
    spec["topology"] = "feedback"
    return {
        "applied": True,
        "removed_entities": removed_entities,
        "rebuilt_relations": [f"{item['source']}->{item['target']}" for item in relations],
    }


def _normalize_contract_relations(spec: dict[str, Any]) -> list[dict[str, Any]]:
    aliases: dict[str, set[str]] = {}
    endpoint_ids: set[str] = set()
    for field in ("inputs", "modules", "outputs", "innovations"):
        for item in spec.get(field, []) if isinstance(spec.get(field), list) else []:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            item_id = str(item["id"])
            endpoint_ids.add(item_id)
            for value in (item_id, _item_label(item)):
                normalized = _normalized_label(str(value))
                if normalized:
                    aliases.setdefault(normalized, set()).add(item_id)

    def resolve(value: Any) -> str:
        raw = str(value or "").strip()
        if raw in endpoint_ids:
            return raw
        matches = aliases.get(_normalized_label(raw), set())
        return next(iter(matches)) if len(matches) == 1 else raw

    normalized_relations: list[dict[str, Any]] = []
    for raw in spec.get("relations", []) if isinstance(spec.get("relations"), list) else []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        source = resolve(item.get("source") or item.get("source_id"))
        target = resolve(item.get("target") or item.get("target_id"))
        if not source or not target:
            continue
        item["source"] = source
        item["target"] = target
        item["type"] = str(item.get("type") or item.get("relation_type") or "data_flow")
        item["label"] = str(item.get("label") or "")
        item["evidence_ids"] = list(item.get("evidence_ids")) if isinstance(item.get("evidence_ids"), list) else []
        normalized_relations.append(item)
    return normalized_relations


def _repair_contract_relation_endpoints(spec: dict[str, Any]) -> dict[str, list[str]]:
    endpoints = {
        str(item.get("id")): item
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict) and item.get("id")
    }
    label_matches: dict[str, set[str]] = {}
    for item_id, item in endpoints.items():
        normalized = _normalized_label(_item_label(item))
        if normalized:
            label_matches.setdefault(normalized, set()).add(item_id)

    replacements: dict[str, str] = {}
    repaired: list[str] = []
    relations = [item for item in spec.get("relations", []) if isinstance(item, dict)]
    for relation in relations:
        target = str(relation.get("target") or "")
        if target in endpoints:
            continue
        label = _normalized_label(str(relation.get("label") or ""))
        matches = label_matches.get(label, set())
        if target and label and len(matches) == 1:
            replacement = next(iter(matches))
            replacements[target] = replacement
            relation["target"] = replacement
            repaired.append(f"{target}->{replacement}")

    for relation in relations:
        source = str(relation.get("source") or "")
        target = str(relation.get("target") or "")
        if source in replacements:
            relation["source"] = replacements[source]
        if target in replacements:
            relation["target"] = replacements[target]

    deduplicated: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    removed_duplicates: list[str] = []
    removed_unresolved: list[str] = []
    removed_self_relations: list[str] = []
    for relation in relations:
        source = str(relation.get("source") or "")
        target = str(relation.get("target") or "")
        relation_type = str(relation.get("type") or "data_flow")
        if source not in endpoints or target not in endpoints:
            removed_unresolved.append(f"{source}->{target}")
            continue
        if source == target:
            removed_self_relations.append(f"{source}->{target}:{relation_type}")
            continue
        key = (source, target, relation_type)
        existing = by_key.get(key)
        if existing:
            existing["evidence_ids"] = list(dict.fromkeys(list(existing.get("evidence_ids", [])) + list(relation.get("evidence_ids", []))))
            if not existing.get("label") and relation.get("label"):
                existing["label"] = relation["label"]
            removed_duplicates.append(f"{source}->{target}:{relation_type}")
            continue
        by_key[key] = relation
        deduplicated.append(relation)
    spec["relations"] = deduplicated
    return {
        "repaired": list(dict.fromkeys(repaired)),
        "removed_duplicates": list(dict.fromkeys(removed_duplicates)),
        "removed_unresolved": list(dict.fromkeys(removed_unresolved)),
        "removed_self_relations": list(dict.fromkeys(removed_self_relations)),
    }


def _clear_unverbatim_relation_labels(spec: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    evidence_by_id = {
        str(item.get("id")): str(item.get("text") or "")
        for item in parsed.get("evidence", [])
        if isinstance(item, dict) and item.get("id")
    }
    removed: list[str] = []
    for relation in spec.get("relations", []) if isinstance(spec.get("relations"), list) else []:
        if not isinstance(relation, dict):
            continue
        label = str(relation.get("label") or "").strip()
        if not label:
            continue
        cited_text = " ".join(evidence_by_id.get(str(value), "") for value in relation.get("evidence_ids", []))
        if _normalized_label(label) and _normalized_label(label) in _normalized_label(cited_text):
            continue
        removed.append(f"{relation.get('source')}->{relation.get('target')}:{label}")
        relation["label"] = ""
    return removed


def _repair_isolated_input_relations(spec: dict[str, Any], parsed: dict[str, Any]) -> dict[str, list[str]]:
    """Recover explicit data-source -> modality provenance from overview text.

    A common planner error is to flatten ``CFP and OCT from MEH-MIDAS and
    public datasets`` into three peer inputs.  That preserves all words while
    changing the scientific meaning.  This repair only fires for an explicit
    ``using/with ... from ...`` clause in cited evidence.  It reclassifies the
    right-hand entities as data sources, creates a missing generic dataset
    source when it is named verbatim, and connects sources to the left-hand
    modalities before the existing modality-to-method flow.
    """
    inputs = [item for item in spec.get("inputs", []) if isinstance(item, dict) and item.get("id")]
    relations = spec.get("relations", []) if isinstance(spec.get("relations"), list) else []
    by_normalized = {_normalized_label(_item_label(item)): item for item in inputs if _normalized_label(_item_label(item))}
    repaired: list[str] = []
    added_sources: list[str] = []
    reclassified: list[str] = []
    removed_shortcuts: list[str] = []
    removed_orphan_sources: list[str] = []
    collapsed_source_entities: list[str] = []
    existing_pairs = {(str(item.get("source")), str(item.get("target"))) for item in relations if isinstance(item, dict)}
    provenance_pattern = re.compile(r"\b(?:using|uses?|with)\s+(.{1,140}?)\s+from\s+(.{1,140}?)(?=[.;]|\bstage\s+(?:two|2)\b|$)", re.IGNORECASE)
    source_term_pattern = re.compile(r"\b(?:datasets?|data|corpus|corpora|cohort|database|registry|biobank|repository)\b", re.IGNORECASE)

    for evidence in parsed.get("evidence", []):
        if not isinstance(evidence, dict) or not evidence.get("id"):
            continue
        text = str(evidence.get("text") or "")
        match = provenance_pattern.search(text)
        if not match:
            continue
        modality_phrase, source_phrase = match.groups()
        normalized_modality_phrase = _normalized_label(modality_phrase)
        normalized_source_phrase = _normalized_label(source_phrase)
        modalities = [
            item for item in inputs
            if _normalized_label(_item_label(item))
            and _normalized_label(_item_label(item)) in normalized_modality_phrase
        ]
        if not modalities:
            continue
        sources = [
            item for item in inputs
            if _normalized_label(_item_label(item))
            and _normalized_label(_item_label(item)) in normalized_source_phrase
        ]
        component_norms: list[str] = []
        component_labels: list[str] = []
        for raw_chunk in re.split(r"\s*(?:,|\band\b|\+)\s*", source_phrase, flags=re.IGNORECASE):
            chunk = re.sub(r"^(?:the|a|an)\s+", "", raw_chunk.strip(" ()"), flags=re.IGNORECASE).strip()
            normalized_chunk = _normalized_label(chunk)
            if not chunk or not normalized_chunk:
                continue
            component_norms.append(normalized_chunk)
            component_labels.append(chunk)
            exact_existing = by_normalized.get(normalized_chunk)
            if exact_existing:
                if exact_existing not in sources:
                    sources.append(exact_existing)
                continue
            acronym_source = bool(re.fullmatch(r"[A-Z][A-Z0-9-]{2,}", chunk))
            if (not source_term_pattern.search(chunk) and not acronym_source) or len(chunk) > 50:
                continue
            source = {
                "id": _stable_id("source", chunk, len(inputs)),
                "name": chunk,
                "role": "data_source",
                "evidence_ids": [str(evidence["id"])],
            }
            inputs.append(source)
            spec.setdefault("inputs", []).append(source)
            by_normalized[normalized_chunk] = source
            sources.append(source)
            added_sources.append(str(source["id"]))
        component_norms = list(dict.fromkeys(component_norms))
        if len(component_norms) >= 2:
            composite = next(
                (
                    item for item in inputs
                    if isinstance(item, dict)
                    and _normalized_label(_item_label(item)) not in component_norms
                    and all(value in _normalized_label(_item_label(item)) for value in component_norms)
                ),
                None,
            )
            if not composite:
                composite_label = " + ".join(component_labels)
                composite = {
                    "id": _stable_id("source", composite_label, len(inputs)),
                    "name": composite_label,
                    "role": "data_source",
                    "evidence_ids": [str(evidence["id"])],
                }
                inputs.append(composite)
                spec.setdefault("inputs", []).append(composite)
                by_normalized[_normalized_label(composite_label)] = composite
                added_sources.append(str(composite["id"]))
            component_ids = {
                str(item.get("id"))
                for item in inputs
                if isinstance(item, dict) and _normalized_label(_item_label(item)) in component_norms
            }
            if component_ids:
                collapsed_source_entities.extend(sorted(component_ids))
                inputs = [item for item in inputs if str(item.get("id")) not in component_ids]
                spec["inputs"] = [item for item in spec.get("inputs", []) if not (isinstance(item, dict) and str(item.get("id")) in component_ids)]
                relations = [
                    item for item in relations
                    if not (
                        isinstance(item, dict)
                        and (str(item.get("source")) in component_ids or str(item.get("target")) in component_ids)
                    )
                ]
                existing_pairs = {(str(item.get("source")), str(item.get("target"))) for item in relations if isinstance(item, dict)}
                added_sources = [value for value in added_sources if value not in component_ids]
                for value in component_norms:
                    by_normalized.pop(value, None)
            sources = [composite]
        if not sources:
            continue

        evidence_id = str(evidence["id"])
        modality_ids = {str(item.get("id")) for item in modalities}
        source_ids = {str(item.get("id")) for item in sources}
        downstream_targets = {
            str(relation.get("target"))
            for relation in relations
            if isinstance(relation, dict) and str(relation.get("source")) in modality_ids
        }
        kept_relations = []
        for relation in relations:
            if not isinstance(relation, dict):
                kept_relations.append(relation)
                continue
            pair = (str(relation.get("source")), str(relation.get("target")))
            if pair[0] in source_ids and pair[1] in downstream_targets:
                removed_shortcuts.append(f"{pair[0]}->{pair[1]}")
                existing_pairs.discard(pair)
                continue
            kept_relations.append(relation)
        relations = kept_relations

        for modality in modalities:
            if str(modality.get("role") or "").casefold() != "input modality":
                modality["role"] = "input modality"
                reclassified.append(str(modality.get("id")))
        for source in sources:
            if str(source.get("role") or "").casefold() != "data_source":
                source["role"] = "data_source"
                reclassified.append(str(source.get("id")))
            source["evidence_ids"] = list(dict.fromkeys([*source.get("evidence_ids", []), evidence_id]))
            for modality in modalities:
                pair = (str(source.get("id")), str(modality.get("id")))
                if pair in existing_pairs:
                    continue
                relations.append({
                    "source": pair[0],
                    "target": pair[1],
                    "type": "data_source",
                    "label": "",
                    "evidence_ids": [evidence_id],
                })
                existing_pairs.add(pair)
                repaired.append(f"{pair[0]}->{pair[1]}")
    connected_sources = {
        str(item.get("source"))
        for item in relations
        if isinstance(item, dict) and str(item.get("type") or "").casefold() == "data_source"
    }
    retained_inputs = []
    for item in spec.get("inputs", []) if isinstance(spec.get("inputs"), list) else []:
        item_id = str(item.get("id") or "") if isinstance(item, dict) else ""
        role = str(item.get("role") or "").casefold() if isinstance(item, dict) else ""
        if item_id.startswith("source_") and role == "data_source" and item_id not in connected_sources:
            removed_orphan_sources.append(item_id)
            continue
        retained_inputs.append(item)
    spec["inputs"] = retained_inputs
    spec["relations"] = relations
    return {
        "repaired_relations": list(dict.fromkeys(repaired)),
        "added_source_entities": list(dict.fromkeys(added_sources)),
        "reclassified_entities": list(dict.fromkeys(reclassified)),
        "removed_source_shortcuts": list(dict.fromkeys(removed_shortcuts)),
        "removed_orphan_sources": list(dict.fromkeys(removed_orphan_sources)),
        "collapsed_source_entities": list(dict.fromkeys(collapsed_source_entities)),
    }


def _repair_explicit_evaluation_outputs(spec: dict[str, Any], parsed: dict[str, Any]) -> dict[str, list[str]]:
    modules = [item for item in spec.get("modules", []) if isinstance(item, dict) and item.get("id")]
    if not modules:
        return {"added_outputs": [], "added_relations": []}
    source = next(
        (
            item for item in reversed(modules)
            if any(term in _item_label(item).casefold() for term in ("supervised", "fine-tun", "adapt"))
        ),
        modules[-1],
    )
    outputs = spec.setdefault("outputs", [])
    by_normalized = {
        _normalized_label(_item_label(item)): item
        for item in outputs if isinstance(item, dict) and _normalized_label(_item_label(item))
    }
    relations = spec.setdefault("relations", [])
    existing_pairs = {(str(item.get("source")), str(item.get("target"))) for item in relations if isinstance(item, dict)}
    added_outputs: list[str] = []
    added_relations: list[str] = []
    for evidence in parsed.get("evidence", []):
        if not isinstance(evidence, dict) or not evidence.get("id"):
            continue
        text = str(evidence.get("text") or "")
        section = str(evidence.get("section_hint") or "").casefold()
        if str(evidence.get("kind") or "").casefold() != "caption" and "figure" not in section:
            continue
        if not re.search(r"\binternal\s+and\s+external\s+evaluation\b", text, re.IGNORECASE):
            continue
        evidence_id = str(evidence["id"])
        for label in ("internal evaluation", "external evaluation"):
            normalized = _normalized_label(label)
            output = by_normalized.get(normalized)
            if not output:
                output = {
                    "id": _stable_id("output", label, len(outputs)),
                    "name": label,
                    "role": "evaluation output",
                    "evidence_ids": [evidence_id],
                }
                outputs.append(output)
                by_normalized[normalized] = output
                added_outputs.append(str(output["id"]))
            else:
                output["evidence_ids"] = list(dict.fromkeys([*output.get("evidence_ids", []), evidence_id]))
            pair = (str(source.get("id")), str(output.get("id")))
            if pair not in existing_pairs:
                relations.append({"source": pair[0], "target": pair[1], "type": "evaluation", "label": "", "evidence_ids": [evidence_id]})
                existing_pairs.add(pair)
                added_relations.append(f"{pair[0]}->{pair[1]}")
        break
    return {"added_outputs": added_outputs, "added_relations": added_relations}


def _repair_generic_named_modules(spec: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    generic_names = {
        "model", "foundationmodel", "proposedmodel", "ourmodel", "method", "proposedmethod", "framework", "proposedframework",
    }
    evidence_by_id = {
        str(item.get("id")): item
        for item in parsed.get("evidence", [])
        if isinstance(item, dict) and item.get("id")
    }
    repaired: list[str] = []
    patterns = (
        re.compile(r"\bfoundation models?\s*\(\s*([A-Za-z][A-Za-z0-9_-]{2,40})\s*\)", re.IGNORECASE),
        re.compile(r"\b(?:model|method|framework)\s+(?:called|named|denoted as)\s+([A-Za-z][A-Za-z0-9_-]{2,40})\b", re.IGNORECASE),
    )
    for item in spec.get("modules", []) if isinstance(spec.get("modules"), list) else []:
        if not isinstance(item, dict) or _normalized_label(_item_label(item)) not in generic_names:
            continue
        cited = [evidence_by_id.get(str(value)) for value in item.get("evidence_ids", [])]
        candidates = [record for record in cited if record]
        candidates.extend(
            record for record in parsed.get("evidence", [])
            if isinstance(record, dict)
            and record not in candidates
            and (str(record.get("kind") or "").casefold() == "caption" or int(record.get("page") or 9999) <= 3)
        )
        replacement = None
        replacement_evidence = None
        for record in candidates:
            text = str(record.get("text") or "")
            match = next((pattern.search(text) for pattern in patterns if pattern.search(text)), None)
            if not match:
                continue
            candidate = match.group(1).strip()
            if _normalized_label(candidate) in {"ssl", "cnn", "transformer", "network"}:
                continue
            replacement = candidate
            replacement_evidence = str(record.get("id") or "")
            break
        if not replacement:
            continue
        old_label = _item_label(item)
        item["name"] = replacement
        terminology = spec.get("terminology") if isinstance(spec.get("terminology"), dict) else {}
        for key, value in list(terminology.items()):
            if _normalized_label(value) == _normalized_label(old_label) and _normalized_label(key) == _normalized_label(replacement):
                terminology[key] = replacement
            elif _normalized_label(key) == _normalized_label(old_label):
                terminology.pop(key, None)
                terminology[replacement] = replacement
        if replacement_evidence:
            item["evidence_ids"] = list(dict.fromkeys([*item.get("evidence_ids", []), replacement_evidence]))
        repaired.append(f"{old_label}->{replacement}")
    return repaired


def _repair_explicit_named_techniques(spec: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    existing = {
        _normalized_label(_item_label(item)): item
        for item in spec.get("innovations", []) if isinstance(item, dict) and _normalized_label(_item_label(item))
    }
    modules = [item for item in spec.get("modules", []) if isinstance(item, dict) and item.get("id")]
    ssl_module = next((item for item in modules if _normalized_label(_item_label(item)) in {"ssl", "selfsupervisedlearning", "sslpretraining"}), None)
    if not ssl_module:
        return []
    repaired: list[str] = []
    pattern = re.compile(r"\b(?:ssl\s+)?(?:technique|method|approach)\s*\(\s*([A-Za-z][A-Za-z -]{2,50}?)\s*(?:\d+)?\s*\)", re.IGNORECASE)
    for evidence in parsed.get("evidence", []):
        if not isinstance(evidence, dict) or not evidence.get("id"):
            continue
        match = pattern.search(str(evidence.get("text") or ""))
        if not match:
            continue
        label = re.sub(r"\s+", " ", match.group(1)).strip()
        normalized = _normalized_label(label)
        if not normalized or normalized in {"ssl", "learning", "training"}:
            continue
        innovation = existing.get(normalized)
        if innovation:
            innovation["evidence_ids"] = list(dict.fromkeys([*innovation.get("evidence_ids", []), str(evidence["id"])]))
            innovation["must_appear_in_figure"] = True
            innovation["visible_label_required"] = True
            continue
        innovation = {
            "id": _stable_id("innovation", label, len(existing)),
            "name": label,
            "role": f"technique used by {_item_label(ssl_module)}",
            "evidence_ids": [str(evidence["id"])],
            "must_appear_in_figure": True,
            "visible_label_required": True,
        }
        spec.setdefault("innovations", []).append(innovation)
        existing[normalized] = innovation
        repaired.append(str(innovation["id"]))
        break
    return repaired


def _repair_parallel_output_heads(spec: dict[str, Any], parsed: dict[str, Any]) -> dict[str, list[str]]:
    """Promote evidence-backed leaf branches to output-head roles.

    Branch planners often place classification, regression, mask, score, or
    prediction heads in ``modules`` while placing only one sibling in
    ``outputs``.  That makes visually equivalent parallel heads receive
    incompatible role instructions.  Promotion is limited to leaf entities
    sharing one upstream branch point, with at least two output-like siblings
    and explicit branch/head/parallel evidence.
    """
    if str(spec.get("topology") or "").casefold() != "branch":
        return {"promoted_outputs": [], "reclassified_outputs": []}
    modules = [item for item in spec.get("modules", []) if isinstance(item, dict) and item.get("id")]
    outputs = [item for item in spec.get("outputs", []) if isinstance(item, dict) and item.get("id")]
    relations = [item for item in spec.get("relations", []) if isinstance(item, dict)]
    if not modules or not relations:
        return {"promoted_outputs": [], "reclassified_outputs": []}

    entity_by_id = {str(item["id"]): item for item in [*modules, *outputs]}
    outgoing = {str(item.get("source")) for item in relations if item.get("source")}
    incoming_by_source: dict[str, set[str]] = {}
    relation_evidence: dict[tuple[str, str], list[str]] = {}
    for relation in relations:
        source, target = str(relation.get("source") or ""), str(relation.get("target") or "")
        if not source or not target:
            continue
        incoming_by_source.setdefault(source, set()).add(target)
        relation_evidence.setdefault((source, target), []).extend(str(value) for value in relation.get("evidence_ids", []) if value)

    evidence_by_id = {
        str(item.get("id")): str(item.get("text") or "")
        for item in parsed.get("evidence", [])
        if isinstance(item, dict) and item.get("id")
    }
    output_name_pattern = re.compile(
        r"\b(?:class(?:ification|ifier)?|regression|mask|segmentation|detection|prediction|score|label|output|head)\b",
        re.IGNORECASE,
    )
    branch_evidence_pattern = re.compile(r"\b(?:parallel|branch(?:es)?|heads?|multi[- ]?task)\b", re.IGNORECASE)
    promoted: list[str] = []
    reclassified: list[str] = []

    for source_id, target_ids in incoming_by_source.items():
        leaf_targets = [
            entity_by_id[target_id]
            for target_id in target_ids
            if target_id in entity_by_id and target_id not in outgoing
        ]
        output_like = [item for item in leaf_targets if output_name_pattern.search(_item_label(item))]
        if len(output_like) < 2:
            continue
        evidence_ids = list(dict.fromkeys(
            value
            for item in output_like
            for value in [
                *item.get("evidence_ids", []),
                *relation_evidence.get((source_id, str(item.get("id"))), []),
            ]
            if value
        ))
        evidence_text = " ".join(evidence_by_id.get(str(value), "") for value in evidence_ids)
        if not branch_evidence_pattern.search(evidence_text):
            continue
        output_ids = {str(item.get("id")) for item in outputs}
        for item in output_like:
            item_id = str(item.get("id"))
            if str(item.get("role") or "").casefold() != "output head":
                item["role"] = "output head"
                reclassified.append(item_id)
            if item_id not in output_ids:
                outputs.append(item)
                output_ids.add(item_id)
                promoted.append(item_id)

    if promoted:
        promoted_set = set(promoted)
        spec["modules"] = [item for item in modules if str(item.get("id")) not in promoted_set]
        spec["outputs"] = outputs
    return {
        "promoted_outputs": list(dict.fromkeys(promoted)),
        "reclassified_outputs": list(dict.fromkeys(reclassified)),
    }

def normalize_figure_contract(plan: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    summary = plan.get("paper_summary") if isinstance(plan.get("paper_summary"), dict) else {}
    spec = plan.get("figure_specification") if isinstance(plan.get("figure_specification"), dict) else {}
    spec["research_problem"] = _statement_object(spec.get("research_problem"), summary.get("research_problem"))
    spec["central_claim"] = _statement_object(spec.get("central_claim"), summary.get("central_claim"))
    if not isinstance(spec.get("inputs"), list) or not spec.get("inputs"):
        spec["inputs"] = summary.get("inputs") if isinstance(summary.get("inputs"), list) else []
    if not isinstance(spec.get("outputs"), list) or not spec.get("outputs"):
        spec["outputs"] = summary.get("outputs") if isinstance(summary.get("outputs"), list) else []
    if not isinstance(spec.get("modules"), list) or not spec.get("modules"):
        summary_modules = summary.get("core_modules") if isinstance(summary.get("core_modules"), list) else summary.get("modules")
        spec["modules"] = summary_modules if isinstance(summary_modules, list) else []
    for field in ("inputs", "modules", "outputs", "innovations"):
        spec[field] = _normalize_contract_entities(spec.get(field), field)
    entity_deduplication = _deduplicate_contract_entities(spec)
    spec["relations"] = _normalize_contract_relations(spec)
    panel_graph_report = _normalize_panel_graphs(spec, parsed)
    completion_report = augment_contract_from_evidence(spec, parsed)
    completion_report["panel_graphs"] = panel_graph_report
    completion_report["panel_graph_completion"] = _complete_panel_graphs_from_inventory(spec, parsed)
    completion_report["removed_duplicate_entities"] = entity_deduplication["removed"]
    completion_report["duplicate_entity_replacements"] = entity_deduplication["replacements"]
    plan["contract_completion_report"] = completion_report
    if len(completion_report.get("added_entities", [])) >= 3:
        fallback_ids = {
            str(item.get("id"))
            for item in (spec.get("modules", []) if isinstance(spec.get("modules"), list) else [])
            if isinstance(item, dict) and str(item.get("role") or "") == "paper-derived stage requiring VLM verification"
        }
        if fallback_ids:
            spec["modules"] = [item for item in spec.get("modules", []) if not (isinstance(item, dict) and str(item.get("id")) in fallback_ids)]
            spec["relations"] = [
                item for item in spec.get("relations", []) if not (
                    isinstance(item, dict)
                    and (str(item.get("source")) in fallback_ids or str(item.get("target")) in fallback_ids)
                )
            ]
            completion_report["removed_fallback_entities"] = sorted(fallback_ids)
    _complete_from_overview_caption(spec, parsed)
    endpoint_report = _repair_contract_relation_endpoints(spec)
    completion_report["relation_endpoint_repairs"] = endpoint_report["repaired"]
    completion_report["removed_duplicate_relations"] = endpoint_report["removed_duplicates"]
    completion_report["removed_unresolved_relations"] = endpoint_report["removed_unresolved"]
    completion_report["removed_self_relations"] = endpoint_report["removed_self_relations"]
    provenance_report = _repair_isolated_input_relations(spec, parsed)
    completion_report["repaired_isolated_inputs"] = provenance_report["repaired_relations"]
    completion_report["repaired_data_provenance"] = provenance_report["repaired_relations"]
    completion_report["added_data_sources"] = provenance_report["added_source_entities"]
    completion_report["reclassified_data_entities"] = provenance_report["reclassified_entities"]
    completion_report["removed_data_source_shortcuts"] = provenance_report["removed_source_shortcuts"]
    completion_report["removed_orphan_data_sources"] = provenance_report["removed_orphan_sources"]
    completion_report["collapsed_data_source_entities"] = provenance_report["collapsed_source_entities"]
    evaluation_report = _repair_explicit_evaluation_outputs(spec, parsed)
    completion_report["added_explicit_evaluation_outputs"] = evaluation_report["added_outputs"]
    completion_report["added_explicit_evaluation_relations"] = evaluation_report["added_relations"]
    completion_report["repaired_generic_named_modules"] = _repair_generic_named_modules(spec, parsed)
    completion_report["added_explicit_named_techniques"] = _repair_explicit_named_techniques(spec, parsed)
    output_head_report = _repair_parallel_output_heads(spec, parsed)
    completion_report["promoted_parallel_output_heads"] = output_head_report["promoted_outputs"]
    completion_report["reclassified_parallel_output_heads"] = output_head_report["reclassified_outputs"]
    if endpoint_report["repaired"]:
        secondary_completion = augment_contract_from_evidence(spec, parsed)
        for key in ("added_entities", "upgraded_entities", "adopted_entities", "added_relations", "repaired_relations", "grounded_entities"):
            completion_report[key] = list(dict.fromkeys(list(completion_report.get(key, [])) + list(secondary_completion.get(key, []))))
    modules = spec.get("modules") if isinstance(spec.get("modules"), list) else []
    encoder_modules = [item for item in modules if isinstance(item, dict) and "encoder" in _item_label(item).casefold() and "modality encoder" not in _item_label(item).casefold()]
    if len(encoder_modules) >= 3:
        group_id = "modality_encoders_group"
        evidence_ids = list(dict.fromkeys(value for item in encoder_modules for value in item.get("evidence_ids", [])))
        if not any(str(item.get("id")) == group_id for item in modules if isinstance(item, dict)):
            modules.append({"id": group_id, "name": "Modality Encoders", "role": "group", "evidence_ids": evidence_ids})
        relations = spec.get("relations") if isinstance(spec.get("relations"), list) else []
        encoder_ids = {str(item.get("id")) for item in encoder_modules}
        endpoints = modules + list(spec.get("outputs", []) if isinstance(spec.get("outputs"), list) else []) + list(spec.get("innovations", []) if isinstance(spec.get("innovations"), list) else [])
        joint_items = [item for item in endpoints if isinstance(item, dict) and "joint" in _item_label(item).casefold() and ("embedding" in _item_label(item).casefold() or "representation" in _item_label(item).casefold())]
        existing = {(str(item.get("source")), str(item.get("target"))) for item in relations if isinstance(item, dict)}
        for relation in list(relations):
            if not isinstance(relation, dict) or str(relation.get("target")) not in encoder_ids:
                continue
            pair = (str(relation.get("source")), group_id)
            if pair not in existing:
                relations.append({"source": pair[0], "target": pair[1], "type": "encoding", "label": "", "evidence_ids": list(relation.get("evidence_ids", []))})
                existing.add(pair)
        modality_terms = ("image", "text", "audio", "depth", "thermal", "imu", "video", "modalit")
        for item in spec.get("inputs", []) if isinstance(spec.get("inputs"), list) else []:
            if not isinstance(item, dict) or not any(term in _item_label(item).casefold() for term in modality_terms):
                continue
            pair = (str(item.get("id")), group_id)
            if pair not in existing:
                relations.append({"source": pair[0], "target": pair[1], "type": "encoding", "label": "", "evidence_ids": list(item.get("evidence_ids", []))})
                existing.add(pair)
        for joint in joint_items:
            if any(str(item.get("source")) in encoder_ids and str(item.get("target")) == str(joint.get("id")) for item in relations if isinstance(item, dict)):
                pair = (group_id, str(joint.get("id")))
                if pair not in existing:
                    relations.append({"source": pair[0], "target": pair[1], "type": "alignment", "label": "", "evidence_ids": evidence_ids})
                    existing.add(pair)
        emergent_items = [
            item for item in endpoints
            if isinstance(item, dict)
            and (("emergent" in _item_label(item).casefold() and "align" in _item_label(item).casefold()) or "binding" in _item_label(item).casefold())
        ]
        if not emergent_items:
            emergent_relation = next((item for item in relations if isinstance(item, dict) and "emergent" in str(item.get("label") or "").casefold() and "align" in str(item.get("label") or "").casefold()), None)
            if emergent_relation:
                emergent = {"id": "emergent_alignment_concept", "statement": "Emergent Alignment", "role": "innovation", "evidence_ids": list(emergent_relation.get("evidence_ids", []))}
                spec.setdefault("innovations", []).append(emergent)
                emergent_items.append(emergent)
        for joint in joint_items:
            for emergent in emergent_items:
                pair = (str(joint.get("id")), str(emergent.get("id")))
                if pair not in existing:
                    relations.append({"source": pair[0], "target": pair[1], "type": "enables", "label": "emergent alignment", "evidence_ids": list(emergent.get("evidence_ids", []))})
                    existing.add(pair)
        spec["relations"] = relations
        spec["modules"] = modules
    modules = spec.get("modules") if isinstance(spec.get("modules"), list) else []
    relations = spec.get("relations") if isinstance(spec.get("relations"), list) else []
    existing_labels = [_normalized_label(_item_label(item)) for field in ("inputs", "modules", "outputs", "innovations") for item in (spec.get(field, []) if isinstance(spec.get(field), list) else []) if isinstance(item, dict)]
    component_terms = ("token", "embedding", "encoder", "decoder", "head", "branch", "prompt", "proposal", "mask", "engine", "dataset")
    for index, item in enumerate(spec.get("must_show", []) if isinstance(spec.get("must_show"), list) else []):
        if not isinstance(item, dict):
            continue
        text = _item_label(item)
        normalized = _normalized_label(text)
        if not text or len(text) > 80 or not any(term in text.casefold() for term in component_terms):
            continue
        if any(normalized and (normalized in label or label in normalized) for label in existing_labels if label):
            continue
        module = {"id": _stable_id("required", text, index), "name": text, "role": "required_component", "evidence_ids": list(item.get("evidence_ids", []))}
        modules.append(module)
        existing_labels.append(normalized)
    spec["modules"] = modules
    existing_pairs = {(str(item.get("source")), str(item.get("target"))) for item in relations if isinstance(item, dict)}
    transformer_target = next((item for item in modules if "transformer encoder" in _item_label(item).casefold()), None)
    for item in modules:
        label = _item_label(item).casefold()
        normalized_component = _normalized_label(label)
        if transformer_target and str(item.get("id")) != str(transformer_target.get("id")) and any(term in normalized_component for term in ("classtoken", "classificationtoken", "positionembedding")):
            pair = (str(item.get("id")), str(transformer_target.get("id")))
            if pair not in existing_pairs:
                relations.append({"source": pair[0], "target": pair[1], "type": "conditioning", "label": "", "evidence_ids": list(item.get("evidence_ids", []))})
                existing_pairs.add(pair)
    outgoing = {str(item.get("source")) for item in relations if isinstance(item, dict)}
    incoming = {str(item.get("target")) for item in relations if isinstance(item, dict)}
    def choose_input_target(label: str) -> dict[str, Any] | None:
        preferences = []
        low = label.casefold()
        if any(term in low for term in ("prompt", "提示", "プロンプト", "프롬프트")):
            preferences = ["prompt encoder", "提示编码器", "提示編碼器", "プロンプトエンコーダ", "프롬프트 인코더", "encoder", "decoder"]
        elif any(term in low for term in ("image", "video", "图像", "圖像", "影像", "视频", "影片", "画像", "動画", "이미지", "영상")):
            preferences = ["patch", "image encoder", "图像编码器", "圖像編碼器", "画像エンコーダ", "이미지 인코더", "backbone", "encoder"]
        elif any(term in low for term in ("text", "文本", "文字", "テキスト", "텍스트")):
            preferences = ["text encoder", "文本编码器", "文本編碼器", "テキストエンコーダ", "텍스트 인코더", "modality encoders", "encoder"]
        elif any(term in low for term in ("audio", "音频", "音訊", "音声", "오디오")):
            preferences = ["audio encoder", "音频编码器", "音訊編碼器", "音声エンコーダ", "오디오 인코더", "modality encoders", "encoder"]
        elif any(term in low for term in ("depth", "thermal", "imu", "深度", "热成像", "熱成像")):
            preferences = ["modality encoders", "encoder", "编码器", "編碼器", "エンコーダ", "인코더"]
        for preference in preferences:
            match = next((module for module in modules if preference and preference in _item_label(module).casefold()), None)
            if match:
                return match
        return None
    for item in spec.get("inputs", []) if isinstance(spec.get("inputs"), list) else []:
        source = str(item.get("id"))
        if not source or source in outgoing:
            continue
        target = choose_input_target(_item_label(item))
        if target:
            pair = (source, str(target.get("id")))
            if pair not in existing_pairs:
                relations.append({"source": pair[0], "target": pair[1], "type": "data_flow", "label": "", "evidence_ids": list(dict.fromkeys(list(item.get("evidence_ids", [])) + list(target.get("evidence_ids", []))))})
                existing_pairs.add(pair)
                outgoing.add(source)
                incoming.add(pair[1])

    # When several peer inputs already converge on the same first module,
    # connect any remaining declared input to that same target. This covers
    # generic tensor/table inputs such as X_test without inventing a new stage.
    input_ids = {
        str(item.get("id"))
        for item in spec.get("inputs", []) if isinstance(item, dict) and item.get("id")
    }
    input_target_counts: dict[str, int] = {}
    for relation in relations:
        if not isinstance(relation, dict) or str(relation.get("source") or "") not in input_ids:
            continue
        target_id = str(relation.get("target") or "")
        if target_id:
            input_target_counts[target_id] = input_target_counts.get(target_id, 0) + 1
    dominant_input_target = next(
        (target_id for target_id, count in sorted(input_target_counts.items(), key=lambda item: (-item[1], item[0])) if count >= 2),
        None,
    )
    dominant_target_item = next((item for item in modules if str(item.get("id") or "") == dominant_input_target), None)
    if dominant_target_item:
        for item in spec.get("inputs", []) if isinstance(spec.get("inputs"), list) else []:
            if not isinstance(item, dict):
                continue
            source = str(item.get("id") or "")
            if not source or source in outgoing:
                continue
            pair = (source, str(dominant_target_item.get("id")))
            evidence_ids = list(dict.fromkeys([*item.get("evidence_ids", []), *dominant_target_item.get("evidence_ids", [])]))
            if pair not in existing_pairs and evidence_ids:
                relations.append({"source": pair[0], "target": pair[1], "type": "encoding", "label": "", "evidence_ids": evidence_ids})
                existing_pairs.add(pair)
                outgoing.add(source)
                incoming.add(pair[1])
    module_sources = {str(item.get("source")) for item in relations if isinstance(item, dict)}
    completion_added_ids = {str(value) for value in completion_report.get("added_entities", []) if str(value)}
    eligible_output_modules = [item for item in modules if str(item.get("id") or "") not in completion_added_ids]
    sink_modules = [item for item in eligible_output_modules if str(item.get("id")) not in module_sources]
    for item in spec.get("outputs", []) if isinstance(spec.get("outputs"), list) else []:
        target_id = str(item.get("id"))
        if not target_id or target_id in incoming:
            continue
        output_label = _item_label(item).casefold()
        source = next((module for module in reversed(eligible_output_modules) if any(term in _item_label(module).casefold() for term in ("head", "decoder", "refine", "classifier", "predict"))), None)
        if "mask" in output_label:
            source = next((module for module in eligible_output_modules if "mask decoder" in _item_label(module).casefold() or "mask branch" in _item_label(module).casefold()), source)
        elif "box" in output_label or "bounding" in output_label or "class" in output_label or "category" in output_label:
            source = next((module for module in eligible_output_modules if "box branch" in _item_label(module).casefold() or "classification head" in _item_label(module).casefold()), source)
        source = source or (sink_modules[-1] if sink_modules else eligible_output_modules[-1] if eligible_output_modules else None)
        if source:
            pair = (str(source.get("id")), target_id)
            if pair not in existing_pairs:
                relations.append({"source": pair[0], "target": pair[1], "type": "data_flow", "label": "", "evidence_ids": list(dict.fromkeys(list(source.get("evidence_ids", [])) + list(item.get("evidence_ids", []))))})
                existing_pairs.add(pair)
                incoming.add(target_id)
    spec["relations"] = relations
    completion_report["merged_architecture_entities"] = _merge_isolated_architecture_entities(spec, parsed)
    completion_report["demoted_unconnected_completion_entities"] = _demote_unconnected_completion_entities(spec, completion_report, parsed)
    feedback_report = _simplify_feedback_contract(spec)
    completion_report["feedback_simplification_applied"] = feedback_report["applied"]
    completion_report["feedback_removed_entities"] = feedback_report["removed_entities"]
    completion_report["feedback_rebuilt_relations"] = feedback_report["rebuilt_relations"]
    dense_report = _simplify_dense_multiframe_contract(spec)
    completion_report["dense_multiframe_simplification_applied"] = dense_report["applied"]
    completion_report["dense_multiframe_removed_entities"] = dense_report["removed_entities"]
    completion_report["dense_multiframe_rebuilt_relations"] = dense_report["rebuilt_relations"]
    multimodal_report = _simplify_multimodal_contract(spec)
    completion_report["multimodal_simplification_applied"] = multimodal_report["applied"]
    completion_report["multimodal_removed_entities"] = multimodal_report["removed_entities"]
    completion_report["multimodal_rebuilt_relations"] = multimodal_report["rebuilt_relations"]
    branch_report = _simplify_branch_contract(spec)
    completion_report["branch_merged_entities"] = branch_report["merged_entities"]
    completion_report["removed_cross_branch_relations"] = branch_report["removed_cross_branch_relations"]
    completion_report["removed_branch_shortcuts"] = branch_report["removed_branch_shortcuts"]
    spec.setdefault("training_flow", summary.get("training_flow") if isinstance(summary.get("training_flow"), list) else [])
    spec.setdefault("inference_flow", summary.get("inference_flow") if isinstance(summary.get("inference_flow"), list) else [])
    spec.setdefault("feedback_loops", [item for item in spec.get("relations", []) if isinstance(item, dict) and "feedback" in str(item.get("type") or "").casefold()])
    allowed_topologies = {"linear", "branch", "feedback", "multimodal", "dense_multiframe", "unknown"}
    inferred_topology = _infer_topology(spec)
    dense_overview = _detect_dense_overview_panels(parsed)
    completion_report["dense_overview_detection"] = dense_overview
    if dense_overview["detected"] and inferred_topology not in {"feedback", "multimodal"}:
        inferred_topology = "dense_multiframe"
    if str(spec.get("topology") or "") not in allowed_topologies or inferred_topology in {"feedback", "multimodal", "dense_multiframe"}:
        spec["topology"] = inferred_topology
    completion_report["exact_visible_entity_groundings"] = _ground_exact_visible_entities(spec, parsed)
    completion_report["late_repaired_relation_evidence"] = _repair_relation_evidence_after_grounding(spec, parsed)
    completion_report["removed_out_of_scope_entities"] = _prune_out_of_scope_entities(spec, parsed, completion_report)
    completion_report["removed_ungrounded_innovations"] = _remove_ungrounded_innovations(spec)
    completion_report["removed_input_shortcuts"] = _remove_input_shortcuts_through_intermediates(spec)
    _repair_cross_script_terminology(spec, parsed)
    completion_report["repaired_cross_script_entity_labels"] = _repair_cross_script_entity_labels(spec, parsed)
    completion_report["removed_unverbatim_relation_labels"] = _clear_unverbatim_relation_labels(spec, parsed)
    overview_panels = _extract_overview_panels(parsed, spec)
    if overview_panels:
        spec["overview_panels"] = overview_panels
    elif not isinstance(spec.get("overview_panels"), list):
        spec["overview_panels"] = []
    completion_report["overview_panels"] = spec["overview_panels"]
    optional_labels: list[str] = []
    optional_normalized: set[str] = set()
    def add_optional_label(value: object) -> None:
        label = str(value or "").strip()
        normalized = _normalized_label(label)
        if label and normalized and normalized not in optional_normalized:
            optional_labels.append(label)
            optional_normalized.add(normalized)
    for value in spec.get("optional_visible_labels", []) if isinstance(spec.get("optional_visible_labels"), list) else []:
        add_optional_label(value)
    for panel in spec.get("overview_panels", []) if isinstance(spec.get("overview_panels"), list) else []:
        if isinstance(panel, dict):
            add_optional_label(panel.get("panel_label"))
    for item in spec.get("supporting_details", []) if isinstance(spec.get("supporting_details"), list) else []:
        if isinstance(item, dict):
            add_optional_label(item.get("optional_visible_label"))
    spec["optional_visible_labels"] = optional_labels
    labels = []
    normalized_labels: set[str] = set()
    def add_required_label(value: object) -> None:
        label = str(value or "").strip()
        normalized = _normalized_label(label)
        if label and normalized and normalized not in normalized_labels:
            labels.append(label)
            normalized_labels.add(normalized)
    terminology = spec.get("terminology") if isinstance(spec.get("terminology"), dict) else {}
    active_panel_graphs = [item for item in spec.get("panel_graphs", []) if isinstance(item, dict) and item.get("nodes")] if isinstance(spec.get("panel_graphs"), list) else []
    if len(active_panel_graphs) >= 2:
        for panel in active_panel_graphs:
            add_required_label(panel.get("label"))
            for node in panel.get("nodes", []) if isinstance(panel.get("nodes"), list) else []:
                if isinstance(node, dict):
                    add_required_label(_item_label(node))
    else:
        for field in ("inputs", "modules", "outputs"):
            for item in spec.get(field, []) if isinstance(spec.get(field), list) else []:
                if isinstance(item, dict) and _item_label(item):
                    add_required_label(_item_label(item))
        for panel in spec.get("overview_panels", []) if isinstance(spec.get("overview_panels"), list) else []:
            if isinstance(panel, dict):
                add_required_label(panel.get("label"))
    for value in terminology.values():
        add_required_label(value)
    spec["required_labels"] = labels
    repeatable_labels = [str(value).strip() for value in spec.get("repeatable_labels", []) if str(value).strip()] if isinstance(spec.get("repeatable_labels"), list) else []
    entity_by_id = {
        str(item.get("id")): item
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict) and item.get("id")
    }
    panel_membership_counts: dict[str, int] = {}
    for panel in spec.get("overview_panels", []) if isinstance(spec.get("overview_panels"), list) else []:
        if not isinstance(panel, dict):
            continue
        for item_id in {str(value) for value in panel.get("entity_ids", []) if value}:
            panel_membership_counts[item_id] = panel_membership_counts.get(item_id, 0) + 1
    for item_id, count in panel_membership_counts.items():
        item = entity_by_id.get(item_id)
        label = _item_label(item) if item else ""
        if count >= 2 and label and label not in repeatable_labels:
            repeatable_labels.append(label)
    panel_instance_label_counts: dict[str, tuple[str, int]] = {}
    for panel in spec.get("panel_graphs", []) if isinstance(spec.get("panel_graphs"), list) else []:
        if not isinstance(panel, dict):
            continue
        for node in panel.get("nodes", []) if isinstance(panel.get("nodes"), list) else []:
            if not isinstance(node, dict):
                continue
            label = _item_label(node)
            key = _normalized_label(label)
            if not key:
                continue
            original, count = panel_instance_label_counts.get(key, (label, 0))
            panel_instance_label_counts[key] = (original, count + 1)
    for label, count in panel_instance_label_counts.values():
        if count >= 2 and label not in repeatable_labels:
            repeatable_labels.append(label)
    central_claim_text = str(spec.get("central_claim", {}).get("text") or "") if isinstance(spec.get("central_claim"), dict) else str(spec.get("central_claim") or "")
    shared_component_signal = bool(re.search(r"\b(?:same|shared|reused?|single underlying)\b", central_claim_text, re.IGNORECASE))
    if shared_component_signal:
        for item in spec.get("modules", []) if isinstance(spec.get("modules"), list) else []:
            if not isinstance(item, dict):
                continue
            label = _item_label(item)
            role = str(item.get("role") or "").casefold()
            if label and ("model" in label.casefold() or "shared" in role or "reused" in role) and label not in repeatable_labels:
                repeatable_labels.append(label)
    if len(active_panel_graphs) >= 2:
        required_normalized = {_normalized_label(value) for value in labels if _normalized_label(value)}
        repeatable_labels = [value for value in repeatable_labels if _normalized_label(value) in required_normalized]
    spec["repeatable_labels"] = repeatable_labels
    # Several repair steps above can add a relation of their own, so the endpoint
    # sweep runs again last, after every one of them. A relation that names an
    # endpoint no entity declares -- the model wrote source_s_emphasize_performance
    # -> S where nothing is called either -- used to survive into grounding
    # validation, fail it, and throw the paper away after it had already been read
    # and paid for. It is dropped and named instead.
    final_endpoint_report = _repair_contract_relation_endpoints(spec)
    if final_endpoint_report["removed_unresolved"] or final_endpoint_report["removed_duplicates"]:
        completion_report["removed_unresolved_relations"] = list(dict.fromkeys(
            list(completion_report.get("removed_unresolved_relations", []))
            + list(final_endpoint_report["removed_unresolved"])
        ))
        completion_report["removed_duplicate_relations"] = list(dict.fromkeys(
            list(completion_report.get("removed_duplicate_relations", []))
            + list(final_endpoint_report["removed_duplicates"])
        ))
        plan["contract_completion_report"] = completion_report
    raw_uncertainties = list(summary.get("unknowns", []) if isinstance(summary.get("unknowns"), list) else [])
    raw_uncertainties.extend(
        f"Dropped unresolved relation {value} because its endpoint was not declared."
        for value in completion_report.get("removed_unresolved_relations", [])
    )
    uncertainties = [
        str(item.get("statement") or item.get("text") or item.get("id") or "unknown") if isinstance(item, dict) else str(item)
        for item in raw_uncertainties
    ]
    evidence_by_id = {item["id"]: item for item in parsed.get("evidence", [])}
    for field in ("inputs", "modules", "outputs", "relations", "innovations"):
        for item in spec.get(field, []) if isinstance(spec.get(field), list) else []:
            if not isinstance(item, dict):
                continue
            evidence = [evidence_by_id.get(value) for value in item.get("evidence_ids", [])]
            valid_records = [record for record in evidence if record]
            if valid_records and all(float(record.get("confidence", 1.0)) < 0.75 and str(record.get("source") or "").casefold() in {"easyocr", "paddle", "adapter"} for record in valid_records):
                uncertainties.append(f"{_item_id(item, field, 0)} relies only on low-confidence OCR evidence")
    if parsed.get("extraction_report", {}).get("semantic_scope") == "sampled_pages_only":
        uncertainties.append("The source is a long scanned document; only scheduled sample pages were OCRed, so unprocessed pages may contain additional scientific details.")
    spec["uncertainties"] = list(dict.fromkeys(str(item) for item in uncertainties if str(item).strip()))
    spec.setdefault("forbidden_inventions", [])
    plan["figure_specification"] = spec
    return spec


def build_overlay_spec(plan: dict[str, Any]) -> dict[str, Any]:
    spec = plan.get("figure_specification", {})
    panel_graphs = [item for item in spec.get("panel_graphs", []) if isinstance(item, dict) and item.get("nodes")] if isinstance(spec.get("panel_graphs"), list) else []
    if len(panel_graphs) >= 2:
        labels = []
        connectors = []
        groups = []
        reading_order = []
        for panel in panel_graphs:
            panel_id = str(panel.get("id") or "")
            groups.append({
                "id": panel_id,
                "panel_label": str(panel.get("panel_label") or ""),
                "label": str(panel.get("label") or ""),
                "member_ids": [str(item.get("instance_id") or item.get("id") or "") for item in panel.get("nodes", []) if isinstance(item, dict)],
            })
            node_ids = set()
            for node in panel.get("nodes", []) if isinstance(panel.get("nodes"), list) else []:
                if not isinstance(node, dict):
                    continue
                instance_id = str(node.get("instance_id") or node.get("id") or "")
                text_value = _item_label(node)
                if not instance_id or not text_value:
                    continue
                node_ids.add(instance_id)
                reading_order.append(instance_id)
                labels.append({
                    "target_id": instance_id,
                    "text": text_value,
                    "role": str(node.get("role") or "module"),
                    "panel_id": panel_id,
                    "bbox_percent": node.get("bbox_percent"),
                })
            for item in panel.get("relations", []) if isinstance(panel.get("relations"), list) else []:
                if not isinstance(item, dict):
                    continue
                source = str(item.get("source") or "")
                target = str(item.get("target") or "")
                if source not in node_ids or target not in node_ids or source == target:
                    continue
                connectors.append({
                    "source": source,
                    "target": target,
                    "type": str(item.get("type") or "data_flow"),
                    "label": str(item.get("label") or ""),
                    "panel_id": panel_id,
                })
        return {
            "summary": "Exact editable panel-instance labels and local directed connectors derived from the paper contract.",
            "labels": labels,
            "connectors": connectors,
            "groups": groups,
            "reading_order": reading_order,
        }
    labels = []
    visible_targets: dict[str, str] = {}
    target_aliases: dict[str, str] = {}
    for field in ("inputs", "modules", "outputs", "innovations"):
        for index, item in enumerate(spec.get(field, []) if isinstance(spec.get(field), list) else []):
            if not isinstance(item, dict):
                continue
            text = _item_label(item)
            normalized = _normalized_label(text)
            target_id = _item_id(item, field, index)
            if text and normalized not in visible_targets:
                labels.append({"target_id": target_id, "text": text, "role": field.rstrip("s")})
                visible_targets[normalized] = target_id
            elif text and normalized:
                target_aliases[target_id] = visible_targets[normalized]
    connectors = []
    connector_keys: set[tuple[str, str, str, str]] = set()
    for item in spec.get("relations", []) if isinstance(spec.get("relations"), list) else []:
        if isinstance(item, dict):
            source = target_aliases.get(str(item.get("source") or item.get("source_id") or ""), str(item.get("source") or item.get("source_id") or ""))
            target = target_aliases.get(str(item.get("target") or item.get("target_id") or ""), str(item.get("target") or item.get("target_id") or ""))
            relation_type = str(item.get("type") or item.get("relation_type") or "data_flow")
            label = str(item.get("label") or "")
            key = (source, target, relation_type, label)
            if not source or not target or source == target or key in connector_keys:
                continue
            connectors.append({"source": source, "target": target, "type": relation_type, "label": label})
            connector_keys.add(key)
    return {
        "summary": "Exact editable labels and directed connectors derived from the paper contract.",
        "labels": labels,
        "connectors": connectors,
        "groups": list(plan.get("design_plan", {}).get("groups", []) if isinstance(plan.get("design_plan"), dict) else []),
        "reading_order": list(plan.get("design_plan", {}).get("reading_order", []) if isinstance(plan.get("design_plan"), dict) else []),
    }


def synchronize_plan_to_contract(plan: dict[str, Any]) -> None:
    spec = plan.get("figure_specification") if isinstance(plan.get("figure_specification"), dict) else {}
    panel_graphs = [item for item in spec.get("panel_graphs", []) if isinstance(item, dict) and item.get("nodes")] if isinstance(spec.get("panel_graphs"), list) else []
    entities = [
        item
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict)
    ]
    required_labels = [str(value).strip() for value in spec.get("required_labels", []) if str(value).strip()]
    allowed = {_normalized_label(value) for value in [*required_labels, *spec.get("repeatable_labels", [])] if _normalized_label(value)}
    visible_entities = [item for item in entities if _normalized_label(_item_label(item)) in allowed] if allowed else entities
    panel_nodes = [node for panel in panel_graphs for node in panel.get("nodes", []) if isinstance(node, dict)]
    entity_ids = (
        [str(item.get("instance_id") or item.get("id") or "") for item in panel_nodes if str(item.get("instance_id") or item.get("id") or "")]
        if len(panel_graphs) >= 2 else
        [str(item.get("id") or "") for item in visible_entities if str(item.get("id") or "")]
    )
    labels = required_labels or [_item_label(item) for item in visible_entities if _item_label(item)]
    spec["storyline"] = labels
    spec["must_show"] = labels
    topology = str(spec.get("topology") or "unknown")
    if topology == "multimodal":
        spec["training_flow"] = []
        spec["inference_flow"] = []
    plan["figure_specification"] = spec
    plan["design_plan"] = {
        "summary": "Information narrative synchronized to the normalized scientific contract.",
        "reading_order": entity_ids,
        "groups": [],
        "innovation_emphasis": [_item_label(item) for item in spec.get("innovations", []) if isinstance(item, dict) and _normalized_label(_item_label(item)) in allowed],
        "preserve": labels,
        "remove": list(spec.get("forbidden_inventions", []) if isinstance(spec.get("forbidden_inventions"), list) else []),
    }
    layout = plan.get("layout_intent") if isinstance(plan.get("layout_intent"), dict) else {}
    layout.update({
        "summary": "Layout intent synchronized to the normalized scientific topology.",
        "pattern": {"feedback": "loop", "multimodal": "hub_and_spoke", "dense_multiframe": "stacked", "branch": "two_stage"}.get(topology, "left_to_right"),
        "flow_description": "Render only declared entities and directed relations; do not add implementation details outside the visible-label whitelist.",
    })
    plan["layout_intent"] = layout
    plan["visual_metaphors"] = {
        "summary": "Paper-grounded visual metaphors synchronized to the normalized contract.",
        "items": [
            {"module_id": str(item.get("instance_id") or item.get("id") or ""), "metaphor": _item_label(item), "must_show": [], "avoid_showing": list(spec.get("forbidden_inventions", []))}
            for item in (panel_nodes if len(panel_graphs) >= 2 else visible_entities)
        ],
    }


def merge_review_grounded_contract(primary: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    primary_spec = primary.get("figure_specification") if isinstance(primary.get("figure_specification"), dict) else {}
    fallback_spec = fallback.get("figure_specification") if isinstance(fallback.get("figure_specification"), dict) else {}
    endpoint_remap: dict[str, str] = {}
    for field in ("inputs", "modules", "outputs", "innovations", "must_show", "relations"):
        combined = list(primary_spec.get(field, []) if isinstance(primary_spec.get(field), list) else [])
        if field == "relations":
            keys = {(str(item.get("source")), str(item.get("target")), str(item.get("type"))) for item in combined if isinstance(item, dict)}
            for item in fallback_spec.get(field, []) if isinstance(fallback_spec.get(field), list) else []:
                if not isinstance(item, dict):
                    continue
                merged_relation = dict(item)
                merged_relation["source"] = endpoint_remap.get(str(item.get("source")), str(item.get("source")))
                merged_relation["target"] = endpoint_remap.get(str(item.get("target")), str(item.get("target")))
                key = (str(merged_relation.get("source")), str(merged_relation.get("target")), str(merged_relation.get("type")))
                if key and key not in keys:
                    combined.append(merged_relation)
                    keys.add(key)
        else:
            by_id = {_item_id(item, field, index).casefold(): _item_id(item, field, index) for index, item in enumerate(combined) if isinstance(item, dict)}
            by_label = {_item_label(item).casefold(): _item_id(item, field, index) for index, item in enumerate(combined) if isinstance(item, dict) and _item_label(item)}
            for index, item in enumerate(fallback_spec.get(field, []) if isinstance(fallback_spec.get(field), list) else []):
                if not isinstance(item, dict):
                    continue
                raw_item_id = _item_id(item, field, index)
                item_id = raw_item_id.casefold()
                label = _item_label(item).casefold()
                if item_id in by_id:
                    endpoint_remap[raw_item_id] = by_id[item_id]
                elif label and label in by_label:
                    endpoint_remap[raw_item_id] = by_label[label]
                else:
                    combined.append(item)
                    endpoint_remap[raw_item_id] = raw_item_id
                    by_id[item_id] = raw_item_id
                    if label:
                        by_label[label] = raw_item_id
        primary_spec[field] = combined
    terminology = dict(fallback_spec.get("terminology") if isinstance(fallback_spec.get("terminology"), dict) else {})
    terminology.update(primary_spec.get("terminology") if isinstance(primary_spec.get("terminology"), dict) else {})
    primary_spec["terminology"] = terminology
    primary_spec["forbidden_inventions"] = list(dict.fromkeys(
        list(primary_spec.get("forbidden_inventions", []) if isinstance(primary_spec.get("forbidden_inventions"), list) else [])
        + list(fallback_spec.get("forbidden_inventions", []) if isinstance(fallback_spec.get("forbidden_inventions"), list) else [])
    ))
    primary["figure_specification"] = primary_spec
    primary_design = primary.get("design_plan") if isinstance(primary.get("design_plan"), dict) else {}
    fallback_order = fallback.get("design_plan", {}).get("reading_order", []) if isinstance(fallback.get("design_plan"), dict) else []
    primary_design["reading_order"] = list(dict.fromkeys(list(primary_design.get("reading_order", [])) + list(fallback_order)))
    primary["design_plan"] = primary_design
    return primary


def _paper_review_from_plan(plan: dict[str, Any], selected_domain: dict[str, Any]) -> dict[str, Any]:
    summary = plan.get("paper_summary") if isinstance(plan.get("paper_summary"), dict) else {}
    spec = plan.get("figure_specification") if isinstance(plan.get("figure_specification"), dict) else {}
    def fact(item: Any, item_id: str) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {"id": item_id, "statement": str(item or "unknown"), "evidence_ids": [], "status": "unknown"}
        return {
            "id": str(item.get("id") or item_id),
            "statement": str(item.get("statement") or item.get("text") or item.get("name") or "unknown"),
            "visible_label": str(item.get("visible_label") or item.get("name") or item.get("text") or item.get("statement") or ""),
            "evidence_ids": list(item.get("evidence_ids", [])),
            "status": str(item.get("status") or "required"),
            "importance": str(item.get("importance") or "high"),
            "confidence": float(item.get("confidence", 1.0) or 0.0),
            "must_appear_in_figure": True,
            "visual_role": str(item.get("role") or "module"),
        }
    return {
        "summary": "Compact paper review derived from the fast VLM figure contract.",
        "schema_version": "2.0-fast",
        "paper_identity": {"title": summary.get("title"), "paper_type": summary.get("paper_type", "unknown")},
        "domain_profile": selected_domain.get("id"),
        "research_questions": [fact(spec.get("research_problem") or summary.get("research_problem"), "research_problem")],
        "central_claims": [fact(spec.get("central_claim") or summary.get("central_claim"), "central_claim")],
        "inputs": [fact(item, f"input_{index}") for index, item in enumerate(spec.get("inputs", [])) if isinstance(item, dict)],
        "outputs": [fact(item, f"output_{index}") for index, item in enumerate(spec.get("outputs", [])) if isinstance(item, dict)],
        "research_objects": [],
        "concepts": [],
        "modules": [fact(item, f"module_{index}") for index, item in enumerate(spec.get("modules", [])) if isinstance(item, dict)],
        "relations": [
            {**fact(item, f"relation_{index}"), "source_id": item.get("source"), "target_id": item.get("target"), "relation_type": item.get("type", "data_flow")}
            for index, item in enumerate(spec.get("relations", [])) if isinstance(item, dict)
        ],
        "contributions": [],
        "innovations": [fact(item, f"innovation_{index}") for index, item in enumerate(spec.get("innovations", [])) if isinstance(item, dict)],
        "workflows": {
            "training": [fact(item, f"training_{index}") for index, item in enumerate(spec.get("training_flow", []))],
            "inference": [fact(item, f"inference_{index}") for index, item in enumerate(spec.get("inference_flow", []))],
        },
        "experiments": {},
        "assumptions": [],
        "limitations": [],
        "results": [],
        "terminology": [],
        "forbidden_inventions": [{"id": f"forbidden_{index}", "statement": str(value), "evidence_ids": []} for index, value in enumerate(spec.get("forbidden_inventions", []))],
        "unknowns": list(spec.get("uncertainties", [])),
    }


def _fast_cache_path(parsed: dict[str, Any], model: str, domain_profile: str) -> Path:
    signature = json.dumps({"version": 41, "document_cache_version": DOCUMENT_CACHE_VERSION, "model": model, "domain_profile": domain_profile}, sort_keys=True).encode("utf-8")
    variant = hashlib.sha256(signature).hexdigest()[:16]
    root = Path(os.getenv("RFS_CACHE_DIR", "").strip() or (Path.home() / ".cache" / "research-figure-studio"))
    return root / "paper_contracts" / str(parsed.get("source_sha256")) / variant / "fast_plan.json"


def _semantic_cache_safe(parsed: dict[str, Any]) -> bool:
    report = parsed.get("extraction_report", {})
    return bool(report.get("scientific_scope_complete", True) and report.get("ocr_run_complete", True))


#: How long a paper's read may take, per reading mode. ``fast`` is the single call the
#: product's own runs make and is the 180 seconds it has always been given. ``full`` pays for
#: a separate review pass and for a planning call that is handed the reference images, so it
#: is given the wall clock those calls need instead of being cut off at the single call's.
READ_DEADLINE_SECONDS = {"fast": 180, "full": 900}


def prepare_paper_figure_contract(
    paper: str | Path,
    out: str | Path,
    deadline_seconds: int | None = None,
    planner_mode: str = "vlm",
    planner_model: str | None = None,
    ocr_engine: str = "auto",
    ocr_lang: str = "en_ch",
    preferences_path: str | Path | None = None,
    positive_references: list[str] | None = None,
    negative_references: list[str] | None = None,
    aspect_ratio: str | None = None,
    language: str | None = None,
    domain_profile: str = "auto",
    ocr_adapter: Callable | None = None,
    read_mode: str = "full",
    frozen_contract: dict[str, Any] | None = None,
    frozen_reground_min_ratio: float | None = None,
) -> dict[str, Any]:
    requested_read_mode = str(read_mode or "").strip().casefold() or "full"
    if requested_read_mode not in READ_DEADLINE_SECONDS:
        raise ValueError(
            "read_mode must be one of %s, got %r" % (sorted(READ_DEADLINE_SECONDS), read_mode)
        )
    fast_mode = requested_read_mode == "fast"
    # One VLM call's ceiling, the same number the fast path has always read. The full read
    # takes two such calls, so it is bounded by this and not by the 35 seconds the two steps
    # used to be capped at, which a review of a long paper does not fit in.
    configured_vlm_timeout = max(35, int(os.getenv("RFS_VLM_TIMEOUT", "90") or 90))
    started = time.monotonic()
    root = ensure_dir(out).resolve()
    inputs = ensure_dir(root / "inputs")
    positive_dir = ensure_dir(inputs / "positive_references")
    negative_dir = ensure_dir(inputs / "negative_references")
    archived_paper = _archive_file(paper, inputs, f"paper{Path(paper).suffix.lower()}")
    archived_positive = [_archive_file(path, positive_dir) for path in (positive_references or [])]
    archived_negative = [_archive_file(path, negative_dir) for path in (negative_references or [])]
    raw_preferences = _load_preferences(preferences_path)
    preferences = merge_preferences(raw_preferences, aspect_ratio=aspect_ratio, language=language)
    preferences["positive_references"] = archived_positive
    preferences["negative_references"] = archived_negative
    write_json(root / "preferences.json", preferences)
    write_json(root / "input_manifest.json", {
        "summary": "Archived inputs for paper figure contract preparation.",
        "paper_original": str(Path(paper).resolve()),
        "paper_archived": archived_paper,
        "preferences_original": str(Path(preferences_path).resolve()) if preferences_path else None,
        "positive_references": archived_positive,
        "negative_references": archived_negative,
    })

    deadline = (
        READ_DEADLINE_SECONDS[requested_read_mode]
        if deadline_seconds is None
        else max(30, int(deadline_seconds))
    )
    deadline_at = started + deadline
    provider_deadline_at = deadline_at - 15.0
    ocr_rescue_min_remaining = 90.0 if fast_mode and planner_mode == "vlm" and deadline <= 240 else 45.0
    parsed = None if ocr_adapter else read_document_cache(archived_paper, ocr_engine=ocr_engine, ocr_lang=ocr_lang)
    document_cache_hit = parsed is not None
    if parsed is None:
        parsed = parse_paper(archived_paper, deadline_at=deadline_at, ocr_engine=ocr_engine, ocr_lang=ocr_lang, ocr_adapter=ocr_adapter, ocr_rescue_min_remaining=ocr_rescue_min_remaining)
        if not ocr_adapter:
            write_document_cache(archived_paper, parsed, ocr_engine=ocr_engine, ocr_lang=ocr_lang)
    parsed["document_cache_version"] = DOCUMENT_CACHE_VERSION
    document_preparation_seconds = round(time.monotonic() - started, 3)
    write_json(root / "document_model.json", parsed)
    write_json(root / "extraction_report.json", parsed["extraction_report"])
    write_json(root / "document_index.json", parsed["document_index"])
    write_json(root / "section_index.json", parsed["document_index"])
    write_json(root / "evidence_map.json", {"summary": "Page-aware evidence map.", "source_path": parsed["source_path"], "page_count": parsed["page_count"], "char_count": parsed["char_count"], "headings": parsed["headings"], "evidence": parsed["evidence"]})
    write_text(root / "paper.md", paper_markdown(parsed))
    write_text(root / "section_summary.md", _section_summary_markdown(parsed))

    if parsed["extraction_report"].get("status") == "fail":
        result = {
            "summary": "Paper extraction failed the minimum readable-page gate.",
            "ok": False,
            "status": "extraction_failed",
            "out_dir": str(root),
            "paper": archived_paper,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "errors": parsed["extraction_report"].get("warnings", []),
        }
        write_json(root / "run_report.json", result)
        return {**result, "root": root, "parsed": parsed, "preferences": preferences}

    selected_domain = detect_domain_profile(parsed, explicit=domain_profile)
    write_json(root / "domain_profile.json", selected_domain)
    frozen = (
        frozen_contract
        if isinstance(frozen_contract, dict)
        and isinstance(frozen_contract.get("paper_review"), dict)
        and isinstance(frozen_contract.get("figure_specification"), dict)
        else None
    )
    frozen_grounding: dict[str, Any] | None = None
    frozen_method: dict[str, Any] = {}
    upstream_merge: dict[str, Any] = {}
    if frozen is not None:
        # The middle layer read this paper and this run read it too. Neither reading
        # is thrown away: the middleware's review is the deeper of the two, so it is
        # the base, and the review this run compiles for itself is folded into it --
        # the plan built here from the same review is this run's own contract for the
        # paper, and the review behind that plan is read back out of it. The frozen
        # evidence ids belong to the middleware's snapshot, so they are rebound onto
        # this parse before the two readings are folded. A contract that does not
        # land here at all is refused by the rebinding.
        frozen, frozen_grounding = reground_frozen_contract(frozen, parsed, min_ratio=frozen_reground_min_ratio)
        write_json(root / "upstream_contract_grounding.json", frozen_grounding)
        frozen_method = frozen.get("method_figure_content") if isinstance(frozen.get("method_figure_content"), dict) else {}
        own_review = _paper_review_from_plan(
            build_review_grounded_plan(parsed, preferences, frozen["paper_review"]),
            selected_domain,
        )
        paper_review, review_merge = merge_upstream_review(
            own_review,
            frozen["paper_review"],
            source=str(frozen.get("source") or ""),
        )
        expand_review_evidence(paper_review, parsed)
        write_json(root / "upstream_review_merge.json", review_merge)
        review_metadata = {
            "summary": "Paper review read from the middle layer's frozen contract, folded together with this run's own.",
            "requested_mode": planner_mode,
            "mode": "frozen_upstream_merged",
            "model": None,
            "warning": None,
            "prompt": "",
            "source": str(frozen.get("source") or ""),
        }
        plan = build_review_grounded_plan(parsed, preferences, paper_review)
        # The middle layer's specification is not adopted in place of the plan this
        # run compiled from the merged review. Its objects are merged into that plan
        # instead, so a figure keeps everything this run read while gaining what the
        # middle layer found, and the panel structure the contract carries -- the
        # panels of the figure the paper printed -- is kept rather than replaced by
        # a plan that has none.
        plan["figure_specification"], upstream_merge = merge_upstream_inventory(
            plan.get("figure_specification"),
            frozen["figure_specification"],
            source=str(frozen.get("source") or ""),
        )
        write_json(root / "upstream_inventory_merge.json", upstream_merge)
        planner_metadata = {
            "requested_mode": planner_mode,
            "mode": "frozen_upstream_merged",
            "model": None,
            "warning": None,
            "prompt": "",
            "source": str(frozen.get("source") or ""),
        }
    elif fast_mode:
        remaining = max(0, int(provider_deadline_at - time.monotonic()))
        effective_mode = planner_mode if remaining >= 25 else "heuristic"
        cache_path = _fast_cache_path(parsed, str(planner_model or ""), str(selected_domain.get("id") or "general"))
        provider_timeout = max(1, min(configured_vlm_timeout, remaining))
        if cache_path.exists() and effective_mode == "vlm" and _semantic_cache_safe(parsed):
            plan = read_json(cache_path)
            planner_metadata = {"requested_mode": "vlm", "mode": "vlm", "model": planner_model, "warning": None, "prompt": "", "cached": True}
        else:
            plan, planner_metadata = plan_fast_paper_contract(
                parsed,
                preferences,
                mode=effective_mode,
                model=planner_model,
                timeout_seconds=provider_timeout,
                retries=2,
                evidence_max_chars=36000,
                deadline_at=provider_deadline_at,
            )
        if planner_metadata.get("mode") == "vlm":
            paper_review = _paper_review_from_plan(plan, selected_domain)
            review_metadata = {"summary": "Paper review derived from the successful fast VLM plan.", "requested_mode": planner_mode, "mode": "derived_from_vlm_plan", "model": planner_metadata.get("model"), "warning": None, "prompt": ""}
        else:
            remaining = max(0, int(provider_deadline_at - time.monotonic()))
            review_mode = planner_mode if remaining >= 25 else "heuristic"
            paper_review, review_metadata = build_paper_review(
                parsed,
                selected_domain,
                mode=review_mode,
                model=planner_model,
                timeout_seconds=max(1, min(configured_vlm_timeout, remaining)),
                retries=1,
                evidence_max_chars=36000,
                deadline_at=provider_deadline_at,
            )
            if review_metadata.get("mode") == "vlm":
                plan = build_review_grounded_plan(parsed, preferences, paper_review)
                planner_metadata = {**planner_metadata, "mode": "review_grounded_vlm", "model": review_metadata.get("model"), "warning": planner_metadata.get("warning")}
    else:
        remaining = max(0, int(provider_deadline_at - time.monotonic()))
        effective_mode = planner_mode if remaining >= 30 else "heuristic"
        paper_review, review_metadata = build_paper_review(
            parsed,
            selected_domain,
            mode=effective_mode,
            model=planner_model,
            timeout_seconds=max(1, min(configured_vlm_timeout, remaining)),
            retries=1,
            deadline_at=provider_deadline_at,
        )
        remaining = max(0, int(provider_deadline_at - time.monotonic()))
        planning_mode = effective_mode if remaining >= 25 else "heuristic"
        plan, planner_metadata = plan_paper_image(
            parsed,
            preferences,
            mode=planning_mode,
            model=planner_model,
            reference_images=archived_positive + archived_negative,
            paper_review=paper_review,
            timeout_seconds=max(1, min(configured_vlm_timeout, remaining)),
            retries=1,
            deadline_at=provider_deadline_at,
        )
        fallback_plan = build_review_grounded_plan(parsed, preferences, paper_review)
        if review_metadata.get("mode") == "vlm":
            plan = merge_review_grounded_contract(plan, fallback_plan)
    review_prompt = review_metadata.pop("prompt", "")
    write_text(root / "prompts" / "paper_review_prompt.txt", review_prompt)
    write_json(root / "paper_review_metadata.json", review_metadata)
    write_json(root / "paper_review.json", paper_review)
    coverage = validate_review_coverage(paper_review, parsed, selected_domain, strict=False)
    write_json(root / "review_coverage_report.json", coverage)
    planning_prompt = planner_metadata.pop("prompt")
    write_text(root / "prompts" / "planning_prompt.txt", planning_prompt)
    write_json(root / "planning_metadata.json", {"summary": "Planner execution metadata.", **planner_metadata})
    expand_plan_evidence(plan, paper_review, parsed)
    normalize_figure_contract(plan, parsed)
    synchronize_plan_to_contract(plan)
    completion_report = plan.get("contract_completion_report") if isinstance(plan.get("contract_completion_report"), dict) else {}
    write_json(root / "contract_completion_report.json", completion_report)
    for name in ("paper_summary", "figure_specification", "design_plan", "layout_intent", "visual_metaphors", "style_plan"):
        write_json(root / f"{name}.json", plan[name])
    planning_validation = validate_plan_grounding(plan, parsed)
    evidence_repair: dict[str, Any] = {}
    if (not planning_validation.get("ok") or planning_validation.get("grounding_gaps")) and planner_metadata.get("mode") in {"vlm", "review_grounded_vlm"}:
        remaining_for_repair = max(0, int(provider_deadline_at - time.monotonic()))
        if remaining_for_repair >= 20:
            plan, evidence_repair = repair_plan_evidence(
                plan,
                parsed,
                [*list(planning_validation.get("errors") or []), *list(planning_validation.get("grounding_gaps") or [])],
                model=planner_model,
                timeout_seconds=max(10, min(int(os.getenv("RFS_VLM_TIMEOUT", "90") or 90), remaining_for_repair)),
                retries=1,
                deadline_at=provider_deadline_at,
            )
            if evidence_repair.get("applied"):
                for name in ("paper_summary", "figure_specification", "design_plan", "layout_intent", "visual_metaphors", "style_plan"):
                    write_json(root / f"{name}.json", plan[name])
                planning_validation = validate_plan_grounding(plan, parsed)
    repair_prompt = str(evidence_repair.pop("prompt", "") or "")
    if repair_prompt:
        write_text(root / "prompts" / "evidence_repair_prompt.txt", repair_prompt)
    write_json(root / "evidence_repair_report.json", evidence_repair)
    write_json(root / "planning_validation_report.json", planning_validation)
    if fast_mode and planning_validation.get("ok") and planner_metadata.get("mode") in {"vlm", "review_grounded_vlm"} and _semantic_cache_safe(parsed):
        write_json(cache_path, plan)
    overlay = build_overlay_spec(plan)
    write_json(root / "overlay_spec.json", overlay)
    final_prompt = compile_image_prompt(plan, preferences, candidate_variant=1, deterministic_overlay=True)
    write_text(root / "image_prompt.md", final_prompt)
    write_text(root / "image_prompt.txt", final_prompt)
    referenced_ids = _collect_evidence_ids({"paper_review": paper_review, "figure_specification": plan["figure_specification"]})
    provisional_ground_truth = compile_review_ground_truth(plan, parsed["evidence"])
    review_evidence_ids = {
        str(evidence_id)
        for priority in provisional_ground_truth.get("priorities", [])
        for evidence_id in priority.get("evidence_ids", [])
        if str(evidence_id).strip()
    }
    referenced_ids.update(review_evidence_ids)
    key_evidence = [
        item for item in parsed["evidence"]
        if str(item.get("id") or "") in referenced_ids or str(item.get("legacy_id") or "") in referenced_ids
    ]
    write_json(root / "key_evidence.json", {
        "summary": "Evidence records referenced by the paper review and figure contract.",
        "evidence": key_evidence,
    })
    review_ground_truth = compile_review_ground_truth(plan, key_evidence)
    plan["review_ground_truth"] = review_ground_truth
    write_json(root / "review_ground_truth.json", review_ground_truth)

    elapsed = round(time.monotonic() - started, 3)
    deadline_reached = time.monotonic() >= deadline_at
    scientific_scope_complete = bool(parsed["extraction_report"].get("scientific_scope_complete", True))
    grounded_review_or_plan = bool(
        {"vlm", "frozen_upstream", "frozen_upstream_merged"}
        & {review_metadata.get("mode"), planner_metadata.get("mode")}
    )
    production_ready = bool(grounded_review_or_plan and planning_validation.get("ok") and not deadline_reached and scientific_scope_complete)
    has_warning = bool(planner_metadata.get("warning") or review_metadata.get("warning") or parsed["extraction_report"].get("status") == "warning")
    status = "complete" if production_ready and not has_warning else "completed_with_warnings"
    contract_source = (
        "frozen_upstream_contract" if frozen is not None
        else "cache" if planner_metadata.get("cached")
        else "vlm" if planner_metadata.get("mode") == "vlm"
        else "vlm_review_deterministic_compile" if planner_metadata.get("mode") == "review_grounded_vlm"
        else "deterministic_evidence_rules"
    )
    source_extraction_seconds = float(parsed["extraction_report"].get("elapsed_seconds") or 0.0)
    extraction_seconds = 0.0 if document_cache_hit else source_extraction_seconds
    planner_provider = planner_metadata.get("provider") if isinstance(planner_metadata.get("provider"), dict) else {}
    review_provider = review_metadata.get("provider") if isinstance(review_metadata.get("provider"), dict) else {}
    provider_calls = [item for item in (planner_provider, review_provider) if item.get("attempts")]
    provider_summary = {
        "planner": planner_provider,
        "review": review_provider,
        "attempts": sum(int(item.get("attempts") or 0) for item in provider_calls),
        "retries_used": sum(int(item.get("retries_used") or 0) for item in provider_calls),
        "success": any(bool(item.get("success")) for item in provider_calls) if provider_calls else None,
        "successful_call_count": sum(bool(item.get("success")) for item in provider_calls),
        "call_count": len(provider_calls),
        "elapsed_seconds": round(sum(float(item.get("elapsed_seconds") or 0.0) for item in provider_calls), 3),
        "failure_categories": list(dict.fromkeys(category for item in provider_calls for category in (item.get("failure_categories") or []))),
    }
    result = {
        "summary": "Fast paper-to-framework contract preparation completed.",
        "ok": bool(planning_validation.get("ok")),
        "status": status,
        "production_ready": production_ready,
        "engineering_only": not production_ready,
        "frozen_contract": frozen_grounding,
        "out_dir": str(root),
        "paper": archived_paper,
        "source_sha256": parsed["source_sha256"],
        "deadline_seconds": deadline,
        # The reading mode that was actually used, resolved rather than echoed: the caller that
        # left it unset would otherwise report nothing, and "how much of the paper was read" is
        # the first thing a thin figure has to be read against.
        "read_mode": requested_read_mode,
        "deadline_reached": deadline_reached,
        "elapsed_seconds": elapsed,
        "planner_mode": planner_metadata.get("mode"),
        "paper_review_mode": review_metadata.get("mode"),
        "contract_source": contract_source,
        "method_source": "middle_layer" if frozen_method else "planned_by_this_run",
        "cache_hit": bool(planner_metadata.get("cached")),
        "document_cache_hit": document_cache_hit,
        "extraction_quality": {
            "pdf_type": parsed["extraction_report"].get("pdf_type"),
            "page_count": parsed["extraction_report"].get("page_count"),
            "readable_page_ratio": parsed["extraction_report"].get("readable_page_ratio"),
            "semantic_scope": parsed["extraction_report"].get("semantic_scope", "full_document"),
            "scientific_scope_complete": scientific_scope_complete,
            "sampled_scan_ready": parsed["extraction_report"].get("sampled_scan_ready", False),
            "evidence_page_coverage_ratio": parsed["extraction_report"].get("evidence_page_coverage_ratio"),
            "evidence_char_count": parsed["extraction_report"].get("evidence_char_count"),
            "max_column_count": parsed["extraction_report"].get("max_column_count"),
            "multi_column_page_count": parsed["extraction_report"].get("multi_column_page_count"),
            "rotated_pages": parsed["extraction_report"].get("rotated_pages", []),
            "section_count": parsed["extraction_report"].get("section_count", 0),
            "typographic_heading_count": parsed["extraction_report"].get("typographic_heading_count", 0),
            "merged_heading_line_count": parsed["extraction_report"].get("merged_heading_line_count", 0),
            "figure_caption_count": parsed["extraction_report"].get("figure_caption_count", 0),
            "table_caption_count": parsed["extraction_report"].get("table_caption_count", 0),
            "formula_count": parsed["extraction_report"].get("formula_count", 0),
            "section_coverage": parsed["extraction_report"].get("section_coverage", {}),
            "missing_priority_sections": [name for name, present in parsed["extraction_report"].get("section_coverage", {}).items() if not present],
            "ocr_candidate_count": len(parsed["extraction_report"].get("ocr_candidate_pages", [])),
            "ocr_scheduled_count": len(parsed["extraction_report"].get("ocr_priority_pages", [])),
            "ocr_completed_count": len(parsed["extraction_report"].get("ocr_pages", [])),
            "ocr_attempted_count": len(parsed["extraction_report"].get("ocr_attempted_pages", [])),
            "ocr_schedule_complete": parsed["extraction_report"].get("ocr_schedule_complete", True),
            "ocr_run_complete": parsed["extraction_report"].get("ocr_run_complete", True),
            "ocr_worker_count": parsed["extraction_report"].get("ocr_worker_count", 1),
            "ocr_rescue_min_remaining_seconds": parsed["extraction_report"].get("ocr_rescue_min_remaining_seconds", 45.0),
            "repeated_margin_noise_removed_count": parsed["extraction_report"].get("repeated_margin_noise_removed_count", 0),
            "native_hyphenation_repair_count": parsed["extraction_report"].get("native_hyphenation_repair_count", 0),
            "ocr_margin_noise_removed_count": parsed["extraction_report"].get("ocr_margin_noise_removed_count", 0),
            "ocr_spacing_repair_count": parsed["extraction_report"].get("ocr_spacing_repair_count", 0),
            "ocr_latin_lexical_gate_applied": parsed["extraction_report"].get("ocr_latin_lexical_gate_applied", False),
            "ocr_latin_known_word_ratio": parsed["extraction_report"].get("ocr_latin_known_word_ratio"),
        },
        "provider": provider_summary,
        "contract_completion": {
            "overview_term_coverage": completion_report.get("overview_term_coverage"),
            "added_entity_count": len(completion_report.get("added_entities", [])),
            "added_relation_count": len(completion_report.get("added_relations", [])),
        },
        "planner_warning": planner_metadata.get("warning"),
        "review_warning": review_metadata.get("warning"),
        "topology": plan["figure_specification"].get("topology"),
        "module_count": len(plan["figure_specification"].get("modules", [])),
        "relation_count": len(plan["figure_specification"].get("relations", [])),
        "stage_timings": {
            "document_preparation_seconds": document_preparation_seconds,
            "document_extraction_seconds": extraction_seconds,
            "cached_source_extraction_seconds": source_extraction_seconds if document_cache_hit else None,
            "semantic_compilation_seconds": round(max(0.0, elapsed - document_preparation_seconds), 3),
            "total_seconds": elapsed,
        },
        "uncertainties": plan["figure_specification"].get("uncertainties", []),
        "artifacts": ["paper.md", "document_model.json", "extraction_report.json", "section_index.json", "section_summary.md", "key_evidence.json", "review_ground_truth.json", "paper_review.json", "figure_specification.json", "contract_completion_report.json", "planning_validation_report.json", "image_prompt.md", "overlay_spec.json", "run_report.json"],
    }
    if frozen_grounding is not None:
        result["artifacts"] = [
            *result["artifacts"],
            "upstream_contract_grounding.json",
            "upstream_review_merge.json",
            "upstream_inventory_merge.json",
        ]
    write_json(root / "run_report.json", result)
    return {
        **result,
        "root": root,
        "parsed": parsed,
        "preferences": preferences,
        "paper_review": paper_review,
        "frozen_method": frozen_method,
        "frozen_method_report": (frozen or {}).get("method_extraction_report") or {},
        "review_metadata": review_metadata,
        "selected_domain": selected_domain,
        "plan": plan,
        "planner_metadata": planner_metadata,
        "archived_positive": archived_positive,
        "archived_negative": archived_negative,
        "planning_validation": planning_validation,
        "contract_evidence_repair": evidence_repair,
    }


def run_fast_framework_prompt(**kwargs: Any) -> dict[str, Any]:
    editable_ppt = bool(kwargs.pop("editable_ppt", False))
    verify_ppt_roundtrip = bool(kwargs.pop("verify_ppt_roundtrip", False))
    editable_ppt = editable_ppt or verify_ppt_roundtrip
    if not kwargs.get("planner_model"):
        kwargs["planner_model"] = os.getenv("RFS_FAST_FRAMEWORK_MODEL", "").strip() or "gemini-2.5-flash"
    kwargs["read_mode"] = "fast"
    prepared = prepare_paper_figure_contract(**kwargs)
    if editable_ppt and prepared.get("ok"):
        from .editable_ppt import compile_semantic_ppt

        ppt_started = time.monotonic()
        paper_summary = prepared.get("plan", {}).get("paper_summary", {})
        title = str(paper_summary.get("title") or paper_summary.get("title_guess") or "").strip() or None
        ppt_result = compile_semantic_ppt(
            prepared["plan"]["figure_specification"],
            prepared["root"],
            aspect_ratio=prepared.get("preferences", {}).get("aspect_ratio") or "16:9",
            title=title,
        )
        prepared.update({
            "pptx_generated": bool(ppt_result.get("ok")),
            "pptx": ppt_result.get("pptx"),
            "semantic_ppt": ppt_result,
        })
        ppt_seconds = round(time.monotonic() - ppt_started, 3)
        prepared["ppt_compile_seconds"] = ppt_seconds
        prepared["elapsed_seconds"] = round(float(prepared.get("elapsed_seconds") or 0.0) + ppt_seconds, 3)
        if isinstance(prepared.get("stage_timings"), dict):
            prepared["stage_timings"]["ppt_compile_seconds"] = ppt_seconds
            prepared["stage_timings"]["total_seconds"] = prepared["elapsed_seconds"]
        prepared["artifacts"] = [*prepared.get("artifacts", []), "figure_program.json", "editable_composition.pptx", "semantic_ppt_report.json", "composition_quality_report.json"]
        if verify_ppt_roundtrip and ppt_result.get("ok"):
            from .ppt_roundtrip import evaluate_ppt_roundtrip

            roundtrip_started = time.monotonic()
            roundtrip = evaluate_ppt_roundtrip(
                read_json(Path(prepared["root"]) / "figure_program.json"),
                str(ppt_result["pptx"]),
                prepared["root"],
                render=True,
            )
            roundtrip_seconds = round(time.monotonic() - roundtrip_started, 3)
            prepared["ppt_roundtrip"] = roundtrip
            prepared["ppt_roundtrip_seconds"] = roundtrip_seconds
            prepared["ok"] = bool(prepared.get("ok") and roundtrip.get("ok"))
            prepared["production_ready"] = bool(prepared.get("production_ready") and roundtrip.get("ok"))
            if not roundtrip.get("ok"):
                prepared["status"] = "completed_with_warnings"
            prepared["elapsed_seconds"] = round(float(prepared.get("elapsed_seconds") or 0.0) + roundtrip_seconds, 3)
            if isinstance(prepared.get("stage_timings"), dict):
                prepared["stage_timings"]["ppt_roundtrip_seconds"] = roundtrip_seconds
                prepared["stage_timings"]["total_seconds"] = prepared["elapsed_seconds"]
            prepared["artifacts"] = [*prepared.get("artifacts", []), "ppt_roundtrip_report.json", "review.pdf", "final_600dpi.png"]
        write_json(Path(prepared["root"]) / "run_report.json", {key: value for key, value in prepared.items() if key not in {"root", "parsed", "preferences", "paper_review", "review_metadata", "selected_domain", "plan", "planner_metadata", "archived_positive", "archived_negative", "planning_validation"}})
    internal = {"root", "parsed", "preferences", "paper_review", "review_metadata", "selected_domain", "plan", "planner_metadata", "archived_positive", "archived_negative", "planning_validation"}
    return {key: value for key, value in prepared.items() if key not in internal}
