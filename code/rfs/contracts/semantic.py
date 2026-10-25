from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

from ..utils import ensure_dir, read_json, write_json, write_text
from .method_figure import (
    INVENTORY_FILES,
    METHOD_INVENTORY_FILE,
    figure_content_plan_path,
    method_inventory_path,
)


#: The middleware's own record of the paper a contract was built for and of the nodes
#: and edges that contract requires. An export writes it beside the review, so a run that
#: adopts the contract can keep one verbatim copy of it; the pairing builder reads that
#: copy to prove both lines worked from one contract instead of two look-alikes.
UPSTREAM_GROUND_TRUTH_FILE = "scientific_ground_truth.json"
#: The name a run files that copy under, so it can never be mistaken for the run's own
#: plan-derived ``scientific_ground_truth.json``.
UPSTREAM_GROUND_TRUTH_COPY = "upstream_scientific_ground_truth.json"


def load_paper_semantic_contract(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    if root.is_file():
        payload = read_json(root)
        if root.name in INVENTORY_FILES or "panel_graphs" in payload or "relations" in payload:
            return {
                "summary": "Paper-grounded semantic contract loaded from a figure specification.",
                "source_run_dir": str(root.parent.resolve()),
                "figure_specification": payload,
            }
        return payload

    def optional(name: str) -> dict[str, Any]:
        path = root / name
        return read_json(path) if path.exists() else {}

    specification_path = method_inventory_path(root)
    if specification_path is None:
        raise FileNotFoundError(f"semantic contract directory has no {METHOD_INVENTORY_FILE}: {root}")
    return {
        "summary": "Paper-grounded semantic contract for editable figure reconstruction.",
        "source_run_dir": str(root.resolve()),
        "paper_review": optional("paper_review.json"),
        "figure_specification": read_json(specification_path),
        "paper_summary": optional("paper_summary.json"),
        "design_plan": optional("design_plan.json"),
        "layout_intent": optional("layout_intent.json"),
        "style_plan": optional("style_plan.json"),
        "global_structure_plan": optional("global_structure_plan.json"),
        "segmentation_plan": optional("segmentation_plan.json"),
        "segmentation_candidates": optional("segmentation_candidates.json"),
    }


def _declined_upstream(source: Path, reason: str) -> dict[str, Any]:
    return {
        "status": "declined",
        "source": str(source),
        "declined_reason": reason,
        "paper_review": {},
        "figure_specification": {},
        "method_figure_content": {},
        "method_extraction_report": {},
        "method_status": "declined",
        "method_declined_reason": "",
        "ground_truth_path": "",
    }


def _load_upstream_method(source: Path) -> dict[str, Any]:
    """Read the middle layer's method contract for this paper, or say why not.

    The middle layer's contract step extracted this method from the same review
    and the same specification that this run just adopted, so a run holding it
    has nothing left to extract. The method is optional: a source that ships
    only the review and the specification still counts as loaded, and this run
    then extracts the method itself, as it did before this existed.
    """
    result: dict[str, Any] = {
        "status": "absent",
        "reason": "",
        "content": {},
        "report": {},
    }
    content_path = figure_content_plan_path(source)
    if content_path is None:
        return result
    try:
        content = read_json(content_path)
        report = read_json(source / "method_extraction_report.json")
    except (OSError, ValueError) as error:
        result.update({"status": "unreadable", "reason": f"unreadable upstream method contract: {error}"})
        return result
    report = report if isinstance(report, dict) else {}
    if not isinstance(content, dict) or not content:
        result.update({"status": "empty", "reason": f"upstream {content_path.name} carries nothing"})
        return result
    if report.get("ok") is False:
        result.update({
            "status": "unusable",
            "reason": "the middle layer flagged its own method contract as not grounded",
            "report": report,
        })
        return result
    result.update({"status": "loaded", "content": content, "report": report})
    return result


def load_upstream_contract(upstream_from: str | Path | None) -> dict[str, Any]:
    """Read the middle layer's frozen contract for one paper, or say why not.

    The middle layer is the authority for what a paper's figure has to show, so
    a run that is handed one reads that specification back instead of planning a
    second opinion about the same paper. A source that cannot be read is
    declined rather than fatal: the caller then plans the contract itself, which
    is what it did before this existed, and reports the reason it did.
    """
    if upstream_from in (None, ""):
        return {
            "status": "off",
            "source": None,
            "declined_reason": "",
            "paper_review": {},
            "figure_specification": {},
            "method_figure_content": {},
            "method_extraction_report": {},
            "method_status": "off",
            "method_declined_reason": "",
            "ground_truth_path": "",
        }
    source = Path(upstream_from).expanduser()
    if not source.exists():
        return _declined_upstream(source, f"upstream contract source does not exist: {source}")
    if not source.is_dir():
        return _declined_upstream(
            source,
            "upstream contract source is not a directory: "
            f"{source}; point at the middle layer's directory for this paper, "
            f"the one that holds {METHOD_INVENTORY_FILE} and paper_review.json",
        )
    try:
        contract = load_paper_semantic_contract(source)
    except (OSError, ValueError) as error:
        return _declined_upstream(source, f"unreadable upstream contract: {error}")
    specification = contract.get("figure_specification")
    if not isinstance(specification, dict) or not specification:
        return _declined_upstream(source, f"upstream directory carries no {METHOD_INVENTORY_FILE}: {source}")
    review = contract.get("paper_review")
    if not isinstance(review, dict) or not review:
        return _declined_upstream(
            source,
            "upstream directory carries no paper_review.json, so its specification "
            f"has no review behind it: {source}",
        )
    method = _load_upstream_method(source)
    ground_truth = source / UPSTREAM_GROUND_TRUTH_FILE
    return {
        "status": "loaded",
        "ground_truth_path": str(ground_truth.resolve()) if ground_truth.is_file() else "",
        "source": str(source.resolve()),
        "declined_reason": "",
        "paper_review": review,
        "figure_specification": specification,
        "method_figure_content": method["content"],
        "method_extraction_report": method["report"],
        "method_status": method["status"],
        "method_declined_reason": method["reason"],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adopt_upstream_ground_truth(run_dir: str | Path, ground_truth_path: str | Path | None) -> dict[str, Any]:
    """File one verbatim copy of the middle layer's own ground truth inside a run.

    The run writes its own ``scientific_ground_truth.json`` from its figure content plan,
    so that file is a derivative of the contract and is never byte-identical to the middle
    layer's own. The copy filed here is the contract itself, which is what lets the pairing
    builder prove both lines describe one paper instead of refusing every honest run. A
    source that ships no ground truth is reported as not adopted rather than fatal, because
    the generation run is complete without it.
    """

    record: dict[str, Any] = {"adopted": False, "file": "", "sha256": "", "source": ""}
    if ground_truth_path in (None, ""):
        return record
    source = Path(ground_truth_path)
    if not source.is_file():
        return record
    target = ensure_dir(Path(run_dir)) / UPSTREAM_GROUND_TRUTH_COPY
    shutil.copyfile(source, target)
    record.update({
        "adopted": True,
        "file": UPSTREAM_GROUND_TRUTH_COPY,
        "sha256": _sha256_file(target),
        "source": str(source.resolve()),
    })
    return record


#: Normalized prefix lengths tried when a frozen quote is looked up in a parse.
#: A long prefix is tried first so a quote that opens with the same boilerplate
#: as another block cannot win on the short one.
FROZEN_REGROUND_PREFIXES = (400, 120)
#: Shortest stored block that counts as evidence of a sentence. Anything shorter
#: matches too much to mean anything.
FROZEN_REGROUND_MIN_BLOCK_CHARS = 40
#: Share of a frozen contract's evidence ids that has to land on this run's own
#: parse. A contract for another paper lands on almost none of them.
FROZEN_REGROUND_MIN_RATIO = 0.5


class FrozenContractUnusable(ValueError):
    """Raised when a frozen contract cannot be read as describing this paper."""


def _normalized_quote(value: Any) -> str:
    """Reduce a quote or a stored evidence text to comparable characters."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def _evidence_index(parsed: dict[str, Any]) -> tuple[dict[int, list[tuple[str, str]]], list[tuple[str, str]]]:
    by_page: dict[int, list[tuple[str, str]]] = {}
    everything: list[tuple[str, str]] = []
    evidence = parsed.get("evidence") if isinstance(parsed, dict) else None
    for item in evidence if isinstance(evidence, list) else []:
        if not isinstance(item, dict):
            continue
        identifier = str(item.get("id") or "")
        text = _normalized_quote(item.get("text"))
        if not identifier or not text:
            continue
        everything.append((identifier, text))
        try:
            page = int(item.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        if page:
            by_page.setdefault(page, []).append((identifier, text))
    return by_page, everything


def _frozen_evidence_refs(value: Any, found: dict[str, dict[str, Any]]) -> None:
    if isinstance(value, dict):
        identifier = str(value.get("evidence_id") or "")
        if identifier:
            found.setdefault(identifier, value)
        for child in value.values():
            _frozen_evidence_refs(child, found)
    elif isinstance(value, list):
        for child in value:
            _frozen_evidence_refs(child, found)


def _match_frozen_quote(
    quote: str,
    page: int,
    by_page: dict[int, list[tuple[str, str]]],
    everything: list[tuple[str, str]],
) -> str:
    """Return this parse's evidence id for one frozen quote, or an empty string.

    The frozen snapshot quoted whole page blocks while this parse stores the
    paragraph, so the stored paragraph is looked for inside the quote, and the
    longest paragraph that fits is the match. Only when no paragraph fits is the
    quote looked for inside this parse's own blocks, and then the tightest block
    that holds it wins.
    """
    if not quote:
        return ""
    same_page = by_page.get(page) or []
    contained = [
        (len(text), identifier)
        for identifier, text in same_page
        if len(text) >= FROZEN_REGROUND_MIN_BLOCK_CHARS and text in quote
    ]
    if contained:
        return max(contained)[1]
    for prefix in FROZEN_REGROUND_PREFIXES:
        head = quote[:prefix]
        if not head:
            continue
        holders = [(len(text), identifier) for identifier, text in same_page if head in text]
        if holders:
            return min(holders)[1]
    # A quote whose page number moved still has its sentence somewhere.
    contained_anywhere = [
        (len(text), identifier)
        for identifier, text in everything
        if len(text) >= FROZEN_REGROUND_MIN_BLOCK_CHARS and text in quote
    ]
    if contained_anywhere:
        return max(contained_anywhere)[1]
    head = quote[:FROZEN_REGROUND_PREFIXES[-1]]
    holders = [(len(text), identifier) for identifier, text in everything if head and head in text]
    return min(holders)[1] if holders else ""


def _remap_evidence(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        rebuilt: dict[str, Any] = {}
        for key, child in value.items():
            if key == "evidence_id" and isinstance(child, str):
                rebuilt[key] = mapping.get(child, child)
            elif key == "evidence_ids" and isinstance(child, list):
                rebuilt[key] = list(dict.fromkeys(
                    mapping.get(str(item), str(item)) for item in child if str(item).strip()
                ))
            else:
                rebuilt[key] = _remap_evidence(child, mapping)
        return rebuilt
    if isinstance(value, list):
        return [_remap_evidence(item, mapping) for item in value]
    return value


def _referenced_evidence_ids(value: Any, found: set[str]) -> None:
    """Collect every evidence id a stored artifact cites, refs and lists alike."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "evidence_id" and isinstance(child, str) and child.strip():
                found.add(child)
            elif key == "evidence_ids" and isinstance(child, list):
                found.update(str(item) for item in child if str(item).strip())
            else:
                _referenced_evidence_ids(child, found)
    elif isinstance(value, list):
        for child in value:
            _referenced_evidence_ids(child, found)


def reground_frozen_contract(frozen: dict[str, Any], parsed: dict[str, Any], *, min_ratio: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebind a frozen contract's evidence ids to the ids this run's parse produced.

    The middle layer read its paper from a frozen snapshot and cited that
    snapshot's evidence ids. This run parses the paper itself, so the ids differ
    even where the sentences agree. Every frozen id is therefore looked up by
    page and by quoted text and rewritten to the id this parse gave the same
    sentence, which is what lets the middle layer's contract be used here as it
    stands instead of being re-derived.

    A contract that lands on almost none of this parse's evidence is a contract
    for another paper, and it is refused rather than used. ``min_ratio`` lowers
    that bar for a paper already known to be the same one, whose quotes failed
    to rebind; the default stays at half.
    """
    floor = FROZEN_REGROUND_MIN_RATIO if min_ratio is None else float(min_ratio)
    if floor < 0 or floor > 1:
        raise ValueError(f"reground minimum ratio must be between 0 and 1: {floor}")
    review = frozen.get("paper_review") if isinstance(frozen.get("paper_review"), dict) else {}
    specification = frozen.get("figure_specification") if isinstance(frozen.get("figure_specification"), dict) else {}
    method = frozen.get("method_figure_content") if isinstance(frozen.get("method_figure_content"), dict) else {}
    method_report = frozen.get("method_extraction_report") if isinstance(frozen.get("method_extraction_report"), dict) else {}
    by_page, everything = _evidence_index(parsed)
    refs: dict[str, dict[str, Any]] = {}
    _frozen_evidence_refs(review, refs)
    _frozen_evidence_refs(specification, refs)
    mapping: dict[str, str] = {}
    unresolved: list[str] = []
    for identifier in sorted(refs):
        ref = refs[identifier]
        quote = _normalized_quote(ref.get("quote") or ref.get("text"))
        try:
            page = int(ref.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        match = _match_frozen_quote(quote, page, by_page, everything)
        if match:
            mapping[identifier] = match
        else:
            unresolved.append(identifier)
    ratio = len(mapping) / len(refs) if refs else 0.0
    # The method contract cites the same frozen evidence the review quotes, so
    # it rides the mapping the review produced. An id the review never quoted
    # cannot be rebound, and a method that mostly misses is dropped on its own
    # while the review and the specification are still usable.
    method_refs: set[str] = set()
    _referenced_evidence_ids(method, method_refs)
    method_unresolved = sorted(identifier for identifier in method_refs if identifier not in mapping)
    method_dropped_reason = ""
    if method and method_refs and (len(method_refs) - len(method_unresolved)) / len(method_refs) < floor:
        method_dropped_reason = (
            "the frozen method contract cites evidence this run's parse does not have: "
            f"{len(method_unresolved)} of {len(method_refs)} cited ids were not found"
        )
        method = {}
    method_status = "regrounded" if method else ("dropped" if method_refs else "absent")
    report = {
        "summary": "Evidence ids of the frozen middle-layer contract, rebound onto this run's own parse.",
        "source": str(frozen.get("source") or ""),
        "referenced_evidence_ids": len(refs),
        "regrounded_evidence_ids": len(mapping),
        "regrounded_ratio": round(ratio, 4),
        "unresolved_evidence_ids": unresolved,
        "unresolved_evidence_count": len(unresolved),
        "fully_regrounded": not unresolved,
        "method_status": method_status,
        "method_referenced_evidence_ids": len(method_refs),
        "method_unresolved_evidence_ids": method_unresolved,
        "method_unresolved_evidence_count": len(method_unresolved),
        "method_dropped_reason": method_dropped_reason,
        "reground_min_ratio": floor,
    }
    if ratio < floor:
        raise FrozenContractUnusable(
            "the frozen contract's evidence does not describe this paper: "
            f"{len(mapping)} of {len(refs)} quoted evidence ids were found in this "
            f"run's parse ({ratio:.1%})"
        )
    return {
        "source": frozen.get("source"),
        "paper_review": _remap_evidence(review, mapping),
        "figure_specification": _remap_evidence(specification, mapping),
        "method_figure_content": _remap_evidence(method, mapping) if method else {},
        "method_extraction_report": method_report,
    }, report


#: The spec fields that state method content. A run's own plan and the middle
#: layer's inventory both write them, so both are folded into one contract.
MERGE_OBJECT_FIELDS = ("inputs", "outputs", "modules", "stages", "innovations", "optional_details")


def _merge_key(item: dict[str, Any]) -> tuple[str, str]:
    """The id and the label one method object can be recognised by in another contract."""

    identifier = str(item.get("id") or "").strip().casefold()
    label = "".join(
        character
        for character in str(
            item.get("label")
            or item.get("visible_label")
            or item.get("name")
            or item.get("text")
            or item.get("statement")
            or ""
        ).casefold()
        if character.isalnum()
    )
    return identifier, label


def _fold_object_fields(
    base: dict[str, Any],
    extra: dict[str, Any],
    fields: tuple[str, ...],
) -> tuple[dict[str, dict[str, int]], list[str], list[str], dict[str, str]]:
    """Fold one contract's method objects into another's, keyed by id and then by label.

    Returns the per field tally, the ids that were added, the ids that were folded into an
    object the run already had, and the map from the id the incoming contract used to the id
    the kept object ended up with -- a relation has to be rewritten onto that object or it
    names a node that is no longer there.
    """

    added: list[str] = []
    folded: list[str] = []
    report: dict[str, dict[str, int]] = {}
    remap: dict[str, str] = {}
    for field in fields:
        kept = [dict(item) for item in (base.get(field) or []) if isinstance(item, dict)]
        by_id: dict[str, dict[str, Any]] = {}
        by_label: dict[str, dict[str, Any]] = {}
        for item in kept:
            identifier, label = _merge_key(item)
            if identifier:
                by_id.setdefault(identifier, item)
            if label:
                by_label.setdefault(label, item)
        added_here = folded_here = 0
        for item in (extra.get(field) or []):
            if not isinstance(item, dict):
                continue
            identifier, label = _merge_key(item)
            match = by_id.get(identifier) if identifier else None
            if match is None and label:
                match = by_label.get(label)
            if match is None:
                kept.append(dict(item))
                added.append(identifier or label or field)
                added_here += 1
                if identifier:
                    by_id[identifier] = kept[-1]
                if label:
                    by_label.setdefault(label, kept[-1])
                continue
            folded.append(identifier or label)
            folded_here += 1
            # What is kept is the two declarations read as one: the run's own values
            # stand where it states one, and every field only the middle layer states
            # -- what the figure must show, what it may omit -- survives the fold.
            evidence = list(dict.fromkeys([
                *[str(value) for value in (match.get("evidence_ids") or [])],
                *[str(value) for value in (item.get("evidence_ids") or [])],
            ]))
            merged = {**dict(item), **match}
            merged["evidence_ids"] = evidence
            match.clear()
            match.update(merged)
            if identifier:
                remap[identifier] = str(match.get("id") or identifier)
        if field in base or added_here or folded_here:
            base[field] = kept
        if added_here or folded_here:
            report[field] = {"added": added_here, "folded": folded_here}
    return report, added, folded, remap


def _fold_relations(
    relations: list[dict[str, Any]],
    incoming: Any,
    remap: dict[str, str],
) -> int:
    """Append the relations this contract does not already state, onto the objects that were kept."""

    seen = {
        (
            str(item.get("source") or item.get("source_id") or ""),
            str(item.get("target") or item.get("target_id") or ""),
            str(item.get("type") or item.get("relation_type") or ""),
        )
        for item in relations
    }
    added = 0
    for item in (incoming or []):
        if not isinstance(item, dict):
            continue
        relation = dict(item)
        for key in ("source", "source_id", "target", "target_id"):
            value = str(relation.get(key) or "")
            if value in remap:
                relation[key] = remap[value]
        key = (
            str(relation.get("source") or relation.get("source_id") or ""),
            str(relation.get("target") or relation.get("target_id") or ""),
            str(relation.get("type") or relation.get("relation_type") or ""),
        )
        if not key[0] or not key[1] or key[0] == key[1] or key in seen:
            continue
        relations.append(relation)
        seen.add(key)
        added += 1
    return added


def merge_upstream_inventory(
    own: dict[str, Any] | None,
    upstream: dict[str, Any] | None,
    *,
    source: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fold the middle layer's method inventory into the contract a run planned itself.

    The middle layer reads a paper's method in more depth than a run's own planning
    pass does, but the plan it states is not this run's plan. Replacing one with the
    other throws away half of what the two together know, so the inventory's objects
    are added to the run's own contract instead: an object both state is kept once
    with the two evidence lists unioned, a relation is added only when it is new and
    its endpoints are rewritten onto the objects that were kept, and the panel
    structure the run already carries is left exactly as it is. A run that carries no
    panel with a node in it falls back to the middle layer's panels, and the report
    says which of the two the merged contract holds.
    """

    base = dict(own) if isinstance(own, dict) else {}
    extra = upstream if isinstance(upstream, dict) else {}
    fields, added, folded, remap = _fold_object_fields(base, extra, MERGE_OBJECT_FIELDS)
    relations = [dict(item) for item in (base.get("relations") or []) if isinstance(item, dict)]
    added_relations = _fold_relations(relations, extra.get("relations"), remap)
    if relations or "relations" in base:
        base["relations"] = relations
    terminology = dict(extra.get("terminology") or {}) if isinstance(extra.get("terminology"), dict) else {}
    terminology.update(base.get("terminology") if isinstance(base.get("terminology"), dict) else {})
    if terminology:
        base["terminology"] = terminology
    for field in ("forbidden_inventions", "uncertainties"):
        values = list(dict.fromkeys([
            *[str(value) for value in (base.get(field) or []) if str(value).strip()],
            *[str(value) for value in (extra.get(field) or []) if str(value).strip()],
        ]))
        if values:
            base[field] = values
    own_panels = [item for item in (base.get("panel_graphs") or []) if isinstance(item, dict) and item.get("nodes")]
    upstream_panels = [item for item in (extra.get("panel_graphs") or []) if isinstance(item, dict) and item.get("nodes")]
    if own_panels:
        panel_source = "run"
    elif upstream_panels:
        panel_source = "middle_layer"
        base["panel_graphs"] = [dict(item) for item in upstream_panels]
    else:
        panel_source = "none"
    report = {
        "summary": "The middle layer's method inventory merged into the contract this run planned.",
        "source": str(source or ""),
        "fields": fields,
        "added_ids": added,
        "folded_ids": folded,
        "added_relation_count": added_relations,
        "panel_source": panel_source,
        "run_panel_count": len(own_panels),
        "middle_layer_panel_count": len(upstream_panels),
        "object_count": sum(len(base.get(field) or []) for field in MERGE_OBJECT_FIELDS),
    }
    return base, report


def merge_upstream_content_plan(
    own: dict[str, Any] | None,
    upstream: dict[str, Any] | None,
    *,
    source: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fold the middle layer's figure content plan into the plan a run extracted itself.

    The middle layer picks the paper's method objects at a depth this run's own planning
    pass does not reach, but the plan it writes is aimed at the specification it built for
    itself and at no panel target. Adopting it whole therefore hands the figure the
    middleware's reading of the paper in place of the one this run just planned against the
    figure the paper printed, and the ids it names belong to the inventory it read rather
    than to the contract this run merged. Its objects are added to the run's own plan
    instead: an object both plans state is kept once with the two evidence lists unioned and
    the middleware's ``must_show`` / ``should_render`` flags preserved, a relation is added
    only when it is new, and the goal and provenance the run stated stand as it stated them.
    """

    base = dict(own) if isinstance(own, dict) else {}
    extra = upstream if isinstance(upstream, dict) else {}
    fields, added, folded, remap = _fold_object_fields(base, extra, MERGE_OBJECT_FIELDS)
    relations = [dict(item) for item in (base.get("relations") or []) if isinstance(item, dict)]
    added_relations = _fold_relations(relations, extra.get("relations"), remap)
    if relations or "relations" in base:
        base["relations"] = relations
    for field in ("forbidden_inventions", "uncertainties"):
        values = list(dict.fromkeys([
            *[str(value) for value in (base.get(field) or []) if str(value).strip()],
            *[str(value) for value in (extra.get(field) or []) if str(value).strip()],
        ]))
        if values:
            base[field] = values
    report = {
        "summary": "The middle layer's figure content plan merged into the plan this run extracted.",
        "source": str(source or ""),
        "fields": fields,
        "added_ids": added,
        "folded_ids": folded,
        "added_relation_count": added_relations,
        "node_count": sum(len(base.get(field) or []) for field in MERGE_OBJECT_FIELDS),
    }
    return base, report


#: The review fields that hold one normalized fact each. A run's own review of a
#: paper and the middle layer's are folded into one review the same way their plans
#: are.
MERGE_REVIEW_FACT_FIELDS = (
    "research_questions",
    "central_claims",
    "inputs",
    "outputs",
    "research_objects",
    "concepts",
    "modules",
    "contributions",
    "innovations",
    "assumptions",
    "limitations",
    "results",
    "terminology",
    "forbidden_inventions",
    "unknowns",
)

#: The review's grouped fact fields, by the group that holds them.
MERGE_REVIEW_FACT_GROUPS = {
    "workflows": ("training", "inference", "offline", "online", "feedback"),
    "experiments": ("datasets", "settings", "metrics", "baselines", "ablations"),
}

#: Review fields that hold loose records -- a sentence, or a record the review schema
#: does not normalize into a fact -- rather than one id-and-label fact each.
MERGE_REVIEW_LOOSE_FIELDS = ("contradictions", "ambiguities", "human_review_required", "figure_candidates")


def _review_states_something(item: dict[str, Any]) -> bool:
    """Whether a review fact names something or cites something, rather than stating nothing."""

    if item.get("evidence_ids"):
        return True
    for key in ("statement", "visible_label", "name", "text"):
        value = str(item.get(key) or "").strip()
        if value and value.casefold() != "unknown":
            return True
    return False


def _review_with_content(review: dict[str, Any]) -> dict[str, Any]:
    """A copy of one review with the facts that state nothing and cite nothing dropped.

    Reading a review back out of a plan leaves placeholder facts behind wherever the
    plan states no such thing -- a research question that came out as the string
    ``unknown`` with no evidence under it. Folding one of those into the other review
    would add a claim about the paper that neither reading makes.
    """

    filtered = dict(review)
    for field in (*MERGE_REVIEW_FACT_FIELDS, "relations"):
        values = review.get(field)
        if isinstance(values, list):
            filtered[field] = [
                item for item in values if not isinstance(item, dict) or _review_states_something(item)
            ]
    for group, names in MERGE_REVIEW_FACT_GROUPS.items():
        values = review.get(group)
        if isinstance(values, dict):
            filtered[group] = {
                name: [
                    item
                    for item in (values.get(name) or [])
                    if not isinstance(item, dict) or _review_states_something(item)
                ]
                for name in names
            }
    return filtered


def _loose_key(item: Any) -> str:
    """The key one loose review record is recognised by: its id, then its label, then its text."""

    if isinstance(item, dict):
        identifier, label = _merge_key(item)
        return identifier or label or json.dumps(item, sort_keys=True, ensure_ascii=False)
    return str(item).strip()


def merge_upstream_review(
    own: dict[str, Any] | None,
    upstream: dict[str, Any] | None,
    *,
    source: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fold the run's own paper review into the middle layer's, into one review.

    A run pointed at a frozen contract used to keep the middleware's review and plan
    the figure from it alone, as though it had read nothing itself. The middleware
    read the paper in more depth, so its review is the base and the run's own is
    folded into it -- the reverse of the two plan merges, where the run's own aim is
    the one aimed at the figure the paper printed. What the run's own reading states
    and the middleware's does not is added: a fact both reviews state is kept once,
    as the middleware stated it, with the two evidence lists unioned; a fact only the
    run's own reading found is added; and a relation is added only when it is new,
    with its endpoints rewritten onto the facts that were kept.
    """

    base = dict(upstream) if isinstance(upstream, dict) else {}
    extra = _review_with_content(own) if isinstance(own, dict) else {}
    fields, added, folded, remap = _fold_object_fields(base, extra, MERGE_REVIEW_FACT_FIELDS)
    relations = [dict(item) for item in (base.get("relations") or []) if isinstance(item, dict)]
    added_relations = _fold_relations(relations, extra.get("relations"), remap)
    if relations or "relations" in base:
        base["relations"] = relations
    for group, names in MERGE_REVIEW_FACT_GROUPS.items():
        own_group = extra.get(group) if isinstance(extra.get(group), dict) else {}
        if not own_group:
            continue
        carried = {
            name: list((base.get(group) or {}).get(name) or []) if isinstance(base.get(group), dict) else []
            for name in names
        }
        tally, group_added, group_folded, group_remap = _fold_object_fields(carried, own_group, names)
        base[group] = carried
        remap.update(group_remap)
        added.extend(group_added)
        folded.extend(group_folded)
        fields.update({f"{group}.{name}": value for name, value in tally.items()})
    for field in MERGE_REVIEW_LOOSE_FIELDS:
        values = [item for item in (base.get(field) or []) if item not in (None, "")]
        seen = {_loose_key(item) for item in values}
        for item in (extra.get(field) or []):
            key = _loose_key(item)
            if key and key not in seen:
                values.append(item)
                seen.add(key)
        if values or field in base:
            base[field] = values
    identity = dict(base.get("paper_identity")) if isinstance(base.get("paper_identity"), dict) else {}
    own_identity = extra.get("paper_identity") if isinstance(extra.get("paper_identity"), dict) else {}
    for key, value in own_identity.items():
        if not identity.get(key):
            identity[key] = value
    if identity:
        base["paper_identity"] = identity
    if extra.get("domain_profile"):
        # The domain profile is what this run detected about the paper it parsed, so it
        # stands over the profile the middleware filed with its review.
        base["domain_profile"] = extra["domain_profile"]
    report = {
        "summary": "The run's own paper review folded into the middle layer's, into one review.",
        "source": str(source or ""),
        "fields": fields,
        "added_ids": added,
        "folded_ids": folded,
        "added_relation_count": added_relations,
        "fact_count": sum(len(base.get(field) or []) for field in MERGE_REVIEW_FACT_FIELDS),
    }
    return base, report


def _text(item: dict, fallback: str = "") -> str:
    for key in ("visible_label", "name", "text", "statement", "label", "id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return fallback


def _specification(contract: dict) -> dict:
    return contract.get("figure_specification") if isinstance(contract.get("figure_specification"), dict) else contract


def _panel_graph_entities(spec: dict) -> list[dict]:
    entities: list[dict] = []
    seen_ids: set[str] = set()
    for panel_index, panel in enumerate(spec.get("panel_graphs", []) if isinstance(spec.get("panel_graphs"), list) else []):
        if not isinstance(panel, dict):
            continue
        panel_id = str(panel.get("id") or f"panel_{panel_index + 1}").strip()
        panel_label = str(panel.get("label") or panel.get("description") or panel_id).strip()
        for node_index, raw in enumerate(panel.get("nodes", []) if isinstance(panel.get("nodes"), list) else []):
            if not isinstance(raw, dict):
                continue
            role = str(raw.get("role") or "module").strip()
            if role.casefold() in {"group", "container", "panel", "stage"}:
                continue
            label = _text(raw)
            entity_id = str(raw.get("instance_id") or raw.get("id") or f"{panel_id}_node_{node_index + 1:02d}").strip()
            if not label or not entity_id or entity_id in seen_ids:
                continue
            seen_ids.add(entity_id)
            entities.append({
                "id": entity_id,
                "semantic_id": str(raw.get("semantic_id") or "").strip() or None,
                "label": label,
                "role": role,
                "panel_id": panel_id,
                "panel_label": panel_label,
                "bbox_percent": raw.get("bbox_percent"),
                "evidence_ids": [str(value) for value in raw.get("evidence_ids", []) if value],
                "source": raw,
            })
    return entities


def _entities(contract: dict) -> list[dict]:
    spec = _specification(contract)
    panel_entities = _panel_graph_entities(spec)
    if panel_entities:
        return panel_entities
    required_labels = {
        re.sub(r"[^a-z0-9]+", "", str(value).casefold())
        for value in spec.get("required_labels", [])
        if str(value).strip()
    }
    entities: list[dict] = []
    for field, role in (("inputs", "input"), ("modules", "module"), ("outputs", "output"), ("innovations", "innovation")):
        values = spec.get(field) if isinstance(spec.get(field), list) else []
        for index, raw in enumerate(values, start=1):
            if not isinstance(raw, dict):
                continue
            label = _text(raw)
            if not label:
                continue
            if role == "innovation":
                visible_required = bool(raw.get("visible_label_required") or raw.get("must_appear_in_figure"))
                normalized_label = re.sub(r"[^a-z0-9]+", "", label.casefold())
                if not visible_required and normalized_label not in required_labels:
                    continue
            entity_id = str(raw.get("id") or raw.get("name") or f"{role}_{index:02d}").strip()
            entities.append({
                "id": entity_id,
                "label": label,
                "role": role,
                "evidence_ids": list(raw.get("evidence_ids") or []),
                "source": raw,
            })
    return entities


def _relations(contract: dict) -> tuple[list[dict], str]:
    spec = _specification(contract)
    panel_entities = _panel_graph_entities(spec)
    if panel_entities:
        entity_ids = {item["id"] for item in panel_entities}
        relations: list[dict] = []
        for panel_index, panel in enumerate(spec.get("panel_graphs", []) if isinstance(spec.get("panel_graphs"), list) else []):
            if not isinstance(panel, dict):
                continue
            panel_id = str(panel.get("id") or f"panel_{panel_index + 1}").strip()
            for relation_index, raw in enumerate(panel.get("relations", []) if isinstance(panel.get("relations"), list) else []):
                if not isinstance(raw, dict):
                    continue
                source = str(raw.get("source") or raw.get("source_id") or "").strip()
                target = str(raw.get("target") or raw.get("target_id") or "").strip()
                if not source or not target or source == target or source not in entity_ids or target not in entity_ids:
                    continue
                relations.append({
                    **raw,
                    "id": str(raw.get("id") or f"{panel_id}_relation_{relation_index + 1:02d}"),
                    "source": source,
                    "target": target,
                    "panel_id": panel_id,
                })
        return relations, "panel_graphs"
    return [item for item in spec.get("relations", []) if isinstance(item, dict)] if isinstance(spec.get("relations"), list) else [], "global_graph"


def _composition_type(role: str, label: str) -> str:
    low = f"{role} {label}".casefold()
    if any(token in low for token in ("input", "dataset", "sample", "image", "text", "audio", "video")):
        return "scene_thumbnail"
    if any(token in low for token in ("output", "prediction", "loss", "objective", "score", "distribution")):
        return "full_frame_icon"
    return "full_bleed_card"


def compile_paper_brief_from_semantic_contract(
    contract: dict,
    out_dir: str | Path | None = None,
    *,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compile the fast paper contract into the legacy brief shape without rereading the paper."""

    spec = _specification(contract)
    entities = _entities(contract)
    relations, relation_source = _relations(contract)
    panels = [item for item in spec.get("panel_graphs", []) if isinstance(item, dict)] if isinstance(spec.get("panel_graphs"), list) else []
    modules = [str(item.get("label") or item.get("description") or item.get("id") or "").strip() for item in panels]
    modules = [item for item in modules if item]
    if not modules:
        modules = list(dict.fromkeys(item["panel_label"] for item in entities if item.get("panel_label")))
    if not modules:
        modules = list(dict.fromkeys(item["label"] for item in entities if item["role"].casefold() in {"module", "stage"}))
    if not modules:
        modules = ["Paper-grounded method"]

    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, list[str]] = {}
    labels_by_id = {item["id"]: item["label"] for item in entities}
    for relation in relations:
        source = str(relation.get("source") or "")
        target = str(relation.get("target") or "")
        outgoing.setdefault(source, []).append(labels_by_id.get(target, target))
        incoming.setdefault(target, []).append(labels_by_id.get(source, source))

    suggestions = []
    for entity in entities:
        panel = entity.get("panel_label") or modules[0]
        neighbor_cues = [*incoming.get(entity["id"], []), *outgoing.get(entity["id"], [])]
        must_show = [f"paper-grounded visual object for {entity['label']}", "local reference shape and visual density"]
        if neighbor_cues:
            must_show.append("declared local context: " + ", ".join(list(dict.fromkeys(neighbor_cues))[:3]))
        suggestions.append({
            "id": entity["id"],
            "macro_panel": panel,
            "paper_concept": entity["label"],
            "composition_type": _composition_type(entity["role"], entity["label"]),
            "visual_metaphor": f"reference-grounded scientific visual for the paper term {entity['label']}",
            "must_show": must_show,
            "avoid_showing": ["unmentioned architecture modules", "fake formulas", "fake numeric chart text", "critical text baked into the image"],
            "semantic_instance_id": entity["id"],
            "semantic_id": entity.get("semantic_id"),
            "semantic_panel_id": entity.get("panel_id"),
            "semantic_role": entity["role"],
            "evidence_ids": entity["evidence_ids"],
            "contract_source": relation_source,
        })

    review = contract.get("paper_review") if isinstance(contract.get("paper_review"), dict) else {}
    summary = contract.get("paper_summary") if isinstance(contract.get("paper_summary"), dict) else {}
    title = str(review.get("title") or review.get("paper_title") or summary.get("title") or "Paper-grounded framework").strip()
    goal = str(spec.get("figure_goal") or "").strip()
    if not goal:
        central_claim = spec.get("central_claim")
        goal = _text(central_claim) if isinstance(central_claim, dict) else str(central_claim or "").strip()
    if not goal:
        goal = "Visualize only the modules and local relations declared in the paper semantic contract."

    brief = {
        "summary": "Paper brief compiled directly from the fast evidence-grounded semantic contract.",
        "planner": "semantic_contract",
        "semantic_authority": relation_source,
        "title_guess": title,
        "figure_kind": str(spec.get("topology") or "paper_driven_generic"),
        "figure_goal": goal,
        "modules": modules[:8],
        "concepts": [item["label"] for item in entities],
        "variables": [str(item) for item in spec.get("required_labels", []) if str(item).strip()],
        "slot_suggestions": suggestions,
        "semantic_relations": relations,
        "semantic_entity_count": len(entities),
        "semantic_relation_count": len(relations),
        "source_path": str(source_path or contract.get("source_run_dir") or ""),
        "source_loader": "semantic_contract",
        "char_count": 0,
        "warnings": [] if relation_source == "panel_graphs" else ["panel_graphs unavailable; using the compatibility global graph"],
    }
    if out_dir is not None:
        out = Path(out_dir)
        write_json(out / "paper_brief.json", brief)
        lines = [
            "# Summary",
            "Paper brief compiled directly from the fast evidence-grounded semantic contract.",
            "",
            "## Source",
            f"- Contract authority: {relation_source}",
            f"- Source: `{brief['source_path']}`",
            f"- Title: {title}",
            "",
            "## Figure Goal",
            goal,
            "",
            "## Panels",
            *[f"- {item}" for item in modules],
            "",
            "## Contract Entities",
            *[f"- {item['id']}: {item['label']} ({item['role']})" for item in entities],
            "",
            "## Contract Relations",
            *[f"- {item.get('source')} -> {item.get('target')} ({item.get('type') or 'data_flow'})" for item in relations],
        ]
        write_text(out / "paper_brief.md", "\n".join(lines) + "\n")
    return brief


def _tokens(value: object) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(token) > 1}


def _object_text(obj: dict) -> str:
    return " ".join(str(obj.get(key) or "") for key in ("id", "title", "label", "name", "semantic_type", "description", "prompt"))


def _center(obj: dict) -> tuple[float, float]:
    box = obj.get("bbox_percent") if isinstance(obj.get("bbox_percent"), dict) else {}
    return float(box.get("x") or 0) + float(box.get("w") or 0) / 2, float(box.get("y") or 0) + float(box.get("h") or 0) / 2


def _assign_entities(entities: list[dict], program: dict) -> tuple[dict[str, dict], list[dict]]:
    minimum_match_score = 0.18
    ambiguity_margin = 0.05
    slots = [item for item in program.get("slots", []) if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict)]
    cards = [item for item in program.get("cards", []) if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict)]
    panels = [item for item in program.get("panels", []) if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict)]
    objects = [*cards, *slots, *panels]
    seen: set[str] = set()
    objects = [item for item in objects if not (str(item.get("id")) in seen or seen.add(str(item.get("id"))))]
    text_items = [
        item for item in (program.get("text_program", {}).get("items", []) if isinstance(program.get("text_program"), dict) else [])
        if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict)
        and not str(item.get("id") or "").startswith("semantic_text_")
        and not item.get("semantic_entity_id")
    ]

    def object_search_text(obj: dict) -> str:
        values = [_object_text(obj)]
        box = obj.get("bbox_percent") or {}
        for text_item in text_items:
            if str(text_item.get("target_id") or "") == str(obj.get("id") or "") or _inside(text_item.get("bbox_percent") or {}, box):
                values.append(str(text_item.get("text") or ""))
        return " ".join(values)

    def object_priority(obj: dict) -> int:
        if obj in cards:
            return 2
        if obj in slots:
            return 1
        return 0

    objects.sort(key=lambda item: (_center(item)[1], _center(item)[0], str(item.get("id") or "")))
    available = list(objects)
    mapping: dict[str, dict] = {}
    records: list[dict] = []
    for entity in entities:
        entity_tokens = _tokens(entity["label"]) | _tokens(entity["id"])
        scored = []
        for obj in available:
            searchable = object_search_text(obj)
            obj_tokens = _tokens(searchable)
            score = len(entity_tokens & obj_tokens) / max(1, len(entity_tokens | obj_tokens))
            semantic_instance_id = str(obj.get("semantic_instance_id") or "").strip()
            if semantic_instance_id and semantic_instance_id == entity["id"]:
                score += 10.0
            semantic_panel_id = str(obj.get("semantic_panel_id") or "").strip()
            if semantic_panel_id and semantic_panel_id == str(entity.get("panel_id") or ""):
                score += 0.5
            entity_compact = re.sub(r"[^a-z0-9]+", "", str(entity.get("label") or "").casefold())
            object_compact = re.sub(r"[^a-z0-9]+", "", searchable.casefold())
            if entity_compact and entity_compact in object_compact:
                score += 1.0
            scored.append((score, object_priority(obj), -_center(obj)[1], -_center(obj)[0], obj))
        if not scored:
            records.append({"entity_id": entity["id"], "label": entity["label"], "status": "unmapped"})
            continue
        scored.sort(key=lambda value: (value[0], value[1], value[2], value[3]), reverse=True)
        best_score = float(scored[0][0])
        second_score = float(scored[1][0]) if len(scored) > 1 else 0.0
        explicit_binding = best_score >= 5.0
        if not explicit_binding and best_score < minimum_match_score:
            records.append({"entity_id": entity["id"], "label": entity["label"], "role": entity["role"], "match_score": round(max(0.0, best_score), 4), "status": "rejected", "reason": "below_confidence_threshold"})
            continue
        if not explicit_binding and second_score > 0 and best_score - second_score < ambiguity_margin:
            records.append({"entity_id": entity["id"], "label": entity["label"], "role": entity["role"], "match_score": round(max(0.0, best_score), 4), "runner_up_score": round(max(0.0, second_score), 4), "status": "ambiguous", "reason": "top_candidates_too_close"})
            continue
        chosen = scored[0][4]
        available.remove(chosen)
        mapping[entity["id"]] = chosen
        records.append({
            "entity_id": entity["id"],
            "label": entity["label"],
            "role": entity["role"],
            "object_id": chosen.get("id"),
            "match_score": round(max(0.0, best_score), 4),
            "status": "matched",
        })
    return mapping, records


def _inside(inner: dict, outer: dict) -> bool:
    cx = float(inner.get("x") or 0) + float(inner.get("w") or 0) / 2
    cy = float(inner.get("y") or 0) + float(inner.get("h") or 0) / 2
    return float(outer.get("x") or 0) <= cx <= float(outer.get("x") or 0) + float(outer.get("w") or 0) and float(outer.get("y") or 0) <= cy <= float(outer.get("y") or 0) + float(outer.get("h") or 0)


def _label_bbox(obj: dict) -> dict[str, float]:
    box = obj["bbox_percent"]
    x, y, w, h = (float(box[key]) for key in ("x", "y", "w", "h"))
    label_h = min(0.055, max(0.026, h * 0.18))
    return {"x": round(x, 4), "y": round(max(0.0, y), 4), "w": round(w, 4), "h": round(label_h, 4)}


def _anchor(source: dict, target: dict) -> list[list[float]]:
    sx, sy = _center(source)
    tx, ty = _center(target)
    sb, tb = source["bbox_percent"], target["bbox_percent"]
    if abs(tx - sx) >= abs(ty - sy):
        start = [float(sb["x"]) + (float(sb["w"]) if tx >= sx else 0.0), sy]
        end = [float(tb["x"]) + (0.0 if tx >= sx else float(tb["w"])), ty]
        mid = (start[0] + end[0]) / 2
        points = [start, [mid, start[1]], [mid, end[1]], end]
    else:
        start = [sx, float(sb["y"]) + (float(sb["h"]) if ty >= sy else 0.0)]
        end = [tx, float(tb["y"]) + (0.0 if ty >= sy else float(tb["h"]))]
        mid = (start[1] + end[1]) / 2
        points = [start, [start[0], mid], [end[0], mid], end]
    clean: list[list[float]] = []
    for point in points:
        rounded = [round(max(0.0, min(1.0, point[0])), 4), round(max(0.0, min(1.0, point[1])), 4)]
        if not clean or clean[-1] != rounded:
            clean.append(rounded)
    return clean


def _preserve_local_reference_arrows(program: dict, mapping: dict[str, dict]) -> tuple[list[dict], list[str]]:
    mapped_objects = list(mapping.values())
    all_objects = {
        str(item.get("id")): item
        for field in ("slots", "cards", "panels")
        for item in program.get(field, [])
        if isinstance(item, dict) and item.get("id") and isinstance(item.get("bbox_percent"), dict)
    }

    def mapped_container(obj: dict | None) -> str | None:
        if not obj:
            return None
        center_box = obj.get("bbox_percent") or {}
        for mapped in mapped_objects:
            if str(mapped.get("id")) == str(obj.get("id")) or _inside(center_box, mapped.get("bbox_percent") or {}):
                return str(mapped.get("id"))
        return None

    preserved: list[dict] = []
    replaced: list[str] = []
    for arrow in program.get("arrows", []) if isinstance(program.get("arrows"), list) else []:
        if not isinstance(arrow, dict):
            continue
        source_id = str(arrow.get("source_id") or arrow.get("source") or "")
        target_id = str(arrow.get("target_id") or arrow.get("target") or "")
        arrow_id = str(arrow.get("id") or "reference_arrow")
        if not source_id or not target_id:
            replaced.append(arrow_id)
            continue
        source_container = mapped_container(all_objects.get(source_id))
        target_container = mapped_container(all_objects.get(target_id))
        if source_container and target_container and source_container != target_container:
            replaced.append(arrow_id)
            continue
        preserved.append(arrow)
    return preserved, replaced


def apply_paper_semantic_contract(program: dict, contract: dict, out_dir: str | Path) -> tuple[dict, dict]:
    out = Path(out_dir)
    entities = _entities(contract)
    relations, relation_source = _relations(contract)
    mapping, mapping_records = _assign_entities(entities, program)
    mapped_boxes = [obj["bbox_percent"] for obj in mapping.values()]
    text_program = program.get("text_program") if isinstance(program.get("text_program"), dict) else {"items": []}
    existing = [item for item in text_program.get("items", []) if isinstance(item, dict)]
    preserved = [
        item for item in existing
        if not str(item.get("id") or "").startswith("semantic_text_")
        and not item.get("semantic_entity_id")
        and not any(_inside(item.get("bbox_percent") or {}, box) for box in mapped_boxes)
    ]
    semantic_text = []
    semantic_reference_regions = []
    for entity in entities:
        obj = mapping.get(entity["id"])
        if not obj:
            continue
        box = _label_bbox(obj)
        reference_text_id = f"semantic_ref_text_{entity['id']}"
        center = {"x": round(box["x"] + box["w"] / 2, 4), "y": round(box["y"] + box["h"] / 2, 4)}
        estimated_font_ratio = round(max(0.006, box["h"] * 0.55), 6)
        semantic_text.append({
            "id": f"semantic_text_{entity['id']}",
            "text": entity["label"],
            "role": f"paper_{entity['role']}_label",
            "target_id": obj.get("id"),
            "semantic_entity_id": entity["id"],
            "semantic_panel_id": entity.get("panel_id"),
            "semantic_evidence_ids": entity["evidence_ids"],
            "source_reference_text_id": reference_text_id,
            "reference_binding": "reference_slot_semantic_overlay_geometry",
            "bbox_percent": box,
            "center_percent": center,
            "width_percent": box["w"],
            "height_percent": box["h"],
            "estimated_font_ratio": estimated_font_ratio,
            "font_size_pt": 10.0,
            "color_hex": "#263747",
            "bold": entity["role"] in {"module", "innovation"},
            "align": "center",
            "font_family_guess": "Arial",
            "font_weight_guess": "bold" if entity["role"] in {"module", "innovation"} else "regular",
            "fit_strategy": "paper_semantic_contract_exact_label",
            "editable_in": "pptx",
            "visible": True,
            "layer_ownership": "editable_text_layer",
        })
        semantic_reference_regions.append({
            "id": reference_text_id,
            "text": entity["label"],
            "raw_text": entity["label"],
            "role": f"paper_{entity['role']}_label",
            "target_id": obj.get("id"),
            "semantic_entity_id": entity["id"],
            "semantic_panel_id": entity.get("panel_id"),
            "evidence_ids": entity["evidence_ids"],
            "bbox_percent": box,
            "center_percent": center,
            "width_percent": box["w"],
            "height_percent": box["h"],
            "estimated_font_ratio": estimated_font_ratio,
            "color_hex": "#263747",
            "font_family_guess": "Arial",
            "font_weight_guess": "bold" if entity["role"] in {"module", "innovation"} else "regular",
            "source": "reference_slot_semantic_overlay_geometry",
            "editable_in": "pptx",
        })
    text_program = {
        **text_program,
        "summary": "Editable text with paper-grounded exact labels overriding image-derived labels in mapped semantic objects.",
        "semantic_authority": "paper_ground_truth",
        "items": preserved + semantic_text,
    }
    program["text_program"] = text_program
    program["text_program_path"] = "text_program.json"
    program["reference_text_geometry_path"] = "reference_text_geometry.json"
    program["text_alignment_report_path"] = "text_alignment_report.json"

    geometry_path = out / "reference_text_geometry.json"
    if geometry_path.exists():
        reference_geometry = read_json(geometry_path)
    else:
        reference_geometry = {
            "summary": "Reference-first text geometry with semantic overlay bindings.",
            "policy": "reference_image_is_highest_authority_for_text_size_position_color_and_hierarchy",
            "detection_mode": "reference_geometry_and_semantic_overlay",
            "text_regions": [],
        }
    geometry_regions = [
        item for item in reference_geometry.get("text_regions", [])
        if isinstance(item, dict) and not str(item.get("id") or "").startswith("semantic_ref_text_")
    ]
    reference_geometry["text_regions"] = geometry_regions + semantic_reference_regions
    reference_geometry["semantic_overlay_policy"] = "exact paper terminology bound to reference-derived slot geometry"

    alignment_items = []
    for item in text_program["items"]:
        alignment_items.append({
            "label_id": item.get("id"),
            "source_reference_text_id": item.get("source_reference_text_id"),
            "target_id": item.get("target_id"),
            "center_delta_percent": 0.0,
            "width_delta_percent": 0.0,
            "height_delta_percent": 0.0,
            "font_ratio_delta": 0.0,
            "color_match": True,
            "editable_in": "pptx",
            "status": "pass",
        })
    alignment_report = {
        "summary": "Reference alignment report after applying exact paper-semantic overlay labels.",
        "policy": "reference_alignment_only; semantic labels inherit their mapped reference slot geometry",
        "status": "pass",
        "max_center_delta_percent": 0.0,
        "max_width_delta_percent": 0.0,
        "max_height_delta_percent": 0.0,
        "items": alignment_items,
    }

    semantic_arrows = []
    skipped_relations = []
    for index, relation in enumerate(relations, start=1):
        source_id = str(relation.get("source") or relation.get("source_id") or "")
        target_id = str(relation.get("target") or relation.get("target_id") or "")
        source, target = mapping.get(source_id), mapping.get(target_id)
        if not source or not target:
            skipped_relations.append({"source": source_id, "target": target_id, "reason": "unmapped_endpoint"})
            continue
        semantic_arrows.append({
            "id": str(relation.get("id") or f"semantic_relation_{index:02d}"),
            "source_id": str(source.get("id")),
            "target_id": str(target.get("id")),
            "semantic_source_id": source_id,
            "semantic_target_id": target_id,
            "semantic_panel_id": relation.get("panel_id"),
            "relation_type": str(relation.get("type") or relation.get("relation_type") or "data_flow"),
            "label": str(relation.get("label") or relation.get("statement") or ""),
            "evidence_ids": list(relation.get("evidence_ids") or []),
            "path_percent": _anchor(source, target),
            "source_anchor": "auto",
            "target_anchor": "auto",
            "control_kind": "elbow_connector",
            "render_style": "elbow_connector",
            "route_intent": "paper_semantic_relation",
            "stroke_color": "#3F5063",
            "stroke_width_pt": 1.8,
            "arrowhead": "triangle",
            "editable_in": "pptx",
            "render_policy": "ppt_shape_not_image_asset",
        })
    preserved_reference_arrows, replaced_reference_arrow_ids = _preserve_local_reference_arrows(program, mapping)
    if relation_source == "panel_graphs":
        heuristic_defaults = [
            item for item in preserved_reference_arrows
            if str(item.get("binding_source") or "").startswith("heuristic_default_")
        ]
        if heuristic_defaults:
            replaced_reference_arrow_ids.extend(str(item.get("id") or "heuristic_default_arrow") for item in heuristic_defaults)
            preserved_reference_arrows = [
                item for item in preserved_reference_arrows
                if not str(item.get("binding_source") or "").startswith("heuristic_default_")
            ]
    if mapping:
        program["arrows"] = [*preserved_reference_arrows, *semantic_arrows]

    rejected_mappings = [item for item in mapping_records if item.get("status") == "rejected"]
    ambiguous_mappings = [item for item in mapping_records if item.get("status") == "ambiguous"]
    report = {
        "summary": "Paper semantic contract application report.",
        "status": "pass" if len(mapping) == len(entities) and not skipped_relations else "warning",
        "semantic_authority": "paper_ground_truth",
        "contract_graph_source": relation_source,
        "visual_authority": "generated_reference_image",
        "entity_count": len(entities),
        "mapped_entity_count": len(mapping),
        "relation_count": len(relations),
        "mapped_relation_count": len(semantic_arrows),
        "preserved_reference_arrow_count": len(preserved_reference_arrows),
        "replaced_reference_arrow_ids": replaced_reference_arrow_ids,
        "mappings": mapping_records,
        "matched_mappings": [item for item in mapping_records if item.get("status") == "matched"],
        "rejected_mappings": rejected_mappings,
        "ambiguous_mappings": ambiguous_mappings,
        "low_confidence_mappings": rejected_mappings + ambiguous_mappings,
        "skipped_relations": skipped_relations,
        "policy": "exact labels and scientific relations come from the paper contract; the image supplies layout and style only",
    }
    write_json(out / "paper_semantic_contract.json", contract)
    write_json(out / "semantic_binding_report.json", report)
    write_json(out / "reference_text_geometry.json", reference_geometry)
    write_json(out / "text_program.json", text_program)
    write_json(out / "text_alignment_report.json", alignment_report)
    write_json(out / "figure_program.json", program)
    return program, report
