"""Instance freezing, metric orchestration, and score aggregation for EditableFigureBench.

EditableFigureBench is the benchmark this repository releases. It is a separate
artefact from 'benchmarks/paper-evolve-editable', which only holds product
acceptance cases. This module owns the three benchmark actions:

- 'prepare': freeze gold instances from the frozen, human-reviewed corpus.
- 'run': score one or more systems against those instances.
- 'score': aggregate per-instance metrics into the four-layer report.

It never mutates the paper pipeline and never converts a hard gate into a score.
The three gates are reported as counts and the four layers are reported
separately, so that a figure which is wrong but pretty cannot hide behind an
aggregate. Scoring is offline by default. Two optional network calls stay off unless asked:
the L1 embedding level (`RFS_BINDING_EMBEDDER`) and the L3 aesthetic judge
(`RFS_AESTHETIC_JUDGE`). A missing credential is recorded, not hidden.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..utils import read_json, write_json, write_text
from .connector_visibility import evaluate_connector_visibility
from .editable_figure_domains import (
    DEFAULT_CATEGORIES_PATH,
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    DOMAIN_SCHEMA_VERSION,
    domain_fields,
    load_arxiv_categories,
    summarize_domains,
)
from .editable_figure_matching import (
    EMBEDDING_ENV_VAR,
    match_entities,
    match_relations,
    module_coverage,
    normalize_tokens,
    relation_triple,
    rendered_entity_recall,
    split_notation,
    unbound_nodes,
)
from .aesthetic_judge import judge_aesthetics
from .geometry_quality import evaluate_density_quality, evaluate_geometry_quality
from .layout_alignment import measure_layout_alignment
from .external_benchmark_scores import figurebench_record, sciflow_record
from .livefigure_nine import score_nine
from .native_editability import editability_verdict, extract_shape_labels, measure_native_editability
from .paper_figure_clip import score_paper_figure_clip
from .paper_quiz import score_paper_quiz
from .rendered_density import measure_rendered_density
from .text_readability import evaluate_text_readability

GOLD_DATASET_VERSION = "editable-figure-bench-gold-v001"

#: Instances the released benchmark targets. The freeze takes every held-out
#: "good" reference figure and fills the remainder from the "bad" stratum.
DEFAULT_INSTANCE_TARGET = 150

#: Reference-figure quality groups a freeze may draw from. A release that only
#: wants human-approved reference figures passes ("good",) and gets every
#: held-out good paper without padding the set with poor references.
DEFAULT_LABEL_GROUPS = ("good", "bad")

#: Held-out splits. The frozen training split stays reserved for Judge training.
HELD_OUT_SPLITS = ("test", "validation")
ALL_SPLITS = ("train", "test", "validation")

# Cohort of an instance relative to the Judge training corpus. Papers from the train split
# carry the reference figures the Judge was trained on, so a pipeline number on them may be
# slightly optimistic; they are reported as an extended stratum, never mixed into the
# held-out headline number.
HELD_OUT_COHORT = "held_out"
TRAIN_OVERLAP_COHORT = "judge_train_overlap"
COHORT_ORDER = (HELD_OUT_COHORT, TRAIN_OVERLAP_COHORT)
COHORT_NOTES = {
    HELD_OUT_COHORT: "test and validation split: no overlap with the Judge training corpus",
    TRAIN_OVERLAP_COHORT: "train split: reference figures may overlap the Judge training corpus",
}

#: Reference-figure quality strata from the frozen human review.
GOOD_LABELS = ("\u597d\uff08\u79d1\u7814\uff09", "\u597d\uff08\u751f\u52a8\uff09")
BAD_LABEL = "\u574f"

#: Preferred candidate pool when a paper has more than one surviving candidate.
SOURCE_PRIORITY = {"reselection_good": 0, "reselection": 1, "initial": 2}

HARD_GATES = ("must_not_invent_hit", "missing_editable_composition", "unreadable_powerpoint")

LAYERS = ("L1", "L2", "L3")

#: Instance files that must exist for a frozen instance to be usable.
INSTANCE_FILES = ("paper.pdf", "reference.png", "contract.json", "paper.json", "confirmation.json")

#: Numbers aggregated per layer, as (report name, path inside metrics.json).
L1_SUMMARY_PATHS = (
    ("entity_strict_f1", ("L1", "entity_strict", "f1")),
    ("entity_strict_precision", ("L1", "entity_strict", "precision")),
    ("entity_strict_recall", ("L1", "entity_strict", "recall")),
    ("entity_soft_f1", ("L1", "entity_soft", "f1")),
    ("relation_strict_f1", ("L1", "relation_strict", "f1")),
    ("relation_strict_recall", ("L1", "relation_strict", "recall")),
    ("relation_soft_f1", ("L1", "relation_soft", "f1")),
    ("rendered_entity_recall", ("L1", "rendered", "recall")),
    ("entity_embedding_f1", ("L1", "entity_embedding", "f1")),
    ("deck_entity_strict_f1", ("L1", "deck", "entity_strict", "f1")),
    ("deck_entity_soft_f1", ("L1", "deck", "entity_soft", "f1")),
    ("deck_entity_embedding_f1", ("L1", "deck", "entity_embedding", "f1")),
    ("module_coverage_strict", ("L1", "core_modules", "coverage", "strict", "coverage")),
    ("module_coverage_soft", ("L1", "core_modules", "coverage", "soft", "coverage")),
    ("module_coverage_embedding", ("L1", "core_modules", "coverage", "embedding", "coverage")),
    ("unbound_node_rate_strict", ("L1", "core_modules", "unbound", "strict", "rate")),
    ("unbound_node_rate_soft", ("L1", "core_modules", "unbound", "soft", "rate")),
    ("predicted_notation_ratio", ("L1", "core_modules", "notation_ratio")),
    ("deck_module_coverage_soft", ("L1", "deck", "core_modules", "coverage", "soft", "coverage")),
)
L2_SUMMARY_PATHS = (
    ("native_connector_count", ("L2", "native_connector_count")),
    ("line_shape_count", ("L2", "line_shape_count")),
    ("connector_realization_ratio", ("L2", "connector_realization_ratio")),
    ("connector_visibility_issue_count", ("L2", "connector_visibility", "issue_count")),
    ("editable_text_char_count", ("L2", "editable_text_char_count")),
    ("text_frame_count", ("L2", "text_frame_count")),
    ("grouping_ratio", ("L2", "grouping_ratio")),
    ("raster_area_ratio", ("L2", "raster_area_ratio")),
    ("no_rasterization_ratio", ("L2", "no_rasterization_ratio")),
    ("full_canvas_picture_count", ("L2", "full_canvas_picture_count")),
)
L3_SUMMARY_PATHS = (
    ("node_overlap_count", ("L3", "geometry", "node_overlap_count")),
    ("icon_text_overlap_count", ("L3", "geometry", "icon_text_overlap_count")),
    ("text_pass_count", ("L3", "geometry", "text_pass_count")),
    ("box_pass_count", ("L3", "geometry", "box_pass_count")),
    ("canvas_occupancy", ("L3", "geometry", "canvas_occupancy")),
    ("empty_region_ratio", ("L3", "geometry", "empty_region_ratio")),
    ("minimum_font_size_pt", ("L3", "readability", "minimum_font_size_pt")),
    ("text_failure_count", ("L3", "readability", "failure_count")),
    ("rendered_ink_coverage", ("L3", "rendered_density", "ink_coverage")),
    ("sparse_panel_count", ("L3", "density", "sparse_panel_count")),
    ("aligned_edge_ratio", ("L3", "layout", "aligned_edge_ratio")),
    ("gap_cv", ("L3", "layout", "gap_cv")),
    ("overflow_count", ("L3", "layout", "overflow_count")),
    ("tight_margin_count", ("L3", "layout", "tight_margin_count")),
    ("element_quality", ("L3", "aesthetics", "element_quality")),
    ("layout_balance", ("L3", "aesthetics", "layout_balance")),
    ("engagement", ("L3", "aesthetics", "engagement")),
)
SUMMARY_PATHS = {"L1": L1_SUMMARY_PATHS, "L2": L2_SUMMARY_PATHS, "L3": L3_SUMMARY_PATHS}

#: Instance checks a human annotator has to answer before the freeze is released.
CONFIRMATION_CHECKS = (
    "entities_complete",
    "relations_correct",
    "must_not_invent_reasonable",
    "reference_figure_is_framework",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_key(paper_id: str) -> str:
    """Stable, content-independent ordering key for deterministic sampling."""
    return hashlib.sha256(paper_id.encode("utf-8")).hexdigest()


def _cohort(split: str) -> str:
    """Which cohort a split belongs to, relative to the Judge training corpus."""
    return HELD_OUT_COHORT if str(split) in HELD_OUT_SPLITS else TRAIN_OVERLAP_COHORT


def _label_group(label: str) -> str | None:
    if label in GOOD_LABELS:
        return "good"
    if label == BAD_LABEL:
        return "bad"
    return None


def _tokens(value: Any) -> frozenset[str]:
    return frozenset(normalize_tokens(value))


def _covered(needle: frozenset[str], haystack: frozenset[str]) -> bool:
    return bool(needle) and needle <= haystack


def _dig(payload: Any, path: Sequence[str]) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _resolve_reference(records: Iterable[dict[str, Any]], root: Path) -> dict[str, Any] | None:
    """Pick the surviving reference figure for one paper.

    A few papers kept more than one human-selected candidate. Preference order is
    reselection_good, then reselection, then initial, then lowest candidate rank.
    """
    ordered = sorted(
        records,
        key=lambda item: (
            SOURCE_PRIORITY.get(str(item.get("candidate_source") or "initial"), 9),
            int(item.get("candidate_rank") or 0),
        ),
    )
    for record in ordered:
        relative = str(record.get("figure_path") or "")
        if relative and (root / relative).is_file():
            return record
    return None


def _relation_lookup(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the contract's own relation table by every id notation it uses."""
    lookup: dict[str, dict[str, Any]] = {}
    for relation in contract.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        source = str(relation.get("source") or "")
        target = str(relation.get("target") or "")
        if not source or not target:
            continue
        for key in (source + "_to_" + target, source + "->" + target, source + "_" + target):
            lookup.setdefault(key, relation)
    return lookup


def _known_entities(content: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    names = [str(value) for value in (content.get("required_entities") or [])]
    for entity in contract.get("entities") or []:
        if isinstance(entity, dict) and entity.get("entity_id"):
            names.append(str(entity["entity_id"]))
    ordered = sorted({name for name in names if name}, key=len, reverse=True)
    return ordered


def _resolve_relation_entry(
    entry: Any,
    lookup: dict[str, dict[str, Any]],
    known: Sequence[str],
) -> dict[str, Any] | None:
    if isinstance(entry, dict):
        try:
            source, target, relation_type = relation_triple(entry)
        except (ValueError, TypeError):
            return None
        if not source or not target:
            return None
        return {"source": source, "target": target, "relation_type": relation_type}
    if isinstance(entry, (list, tuple)):
        if len(entry) >= 3:
            return {"source": str(entry[0]), "target": str(entry[1]), "relation_type": str(entry[2])}
        if len(entry) == 2:
            return {"source": str(entry[0]), "target": str(entry[1]), "relation_type": ""}
        return None
    if not isinstance(entry, str) or not entry.strip():
        return None
    text_value = entry.strip()
    recorded = lookup.get(text_value)
    if recorded is not None:
        return {
            "source": str(recorded.get("source")),
            "target": str(recorded.get("target")),
            "relation_type": str(recorded.get("relation_type") or ""),
        }
    if "->" in text_value:
        source, _, target = text_value.partition("->")
        if source.strip() and target.strip():
            return {"source": source.strip(), "target": target.strip(), "relation_type": ""}
    for separator in (" ", "-"):
        for source in known:
            if not text_value.startswith(source + separator):
                continue
            for target in known:
                if target == source:
                    continue
                suffix = separator + target
                if not text_value.endswith(suffix):
                    continue
                middle = text_value[len(source) : len(text_value) - len(suffix)].strip().strip("-_ ")
                if middle:
                    return {"source": source, "target": target, "relation_type": middle}
    if "_to_" in text_value:
        source, _, target = text_value.partition("_to_")
        if source.strip() and target.strip():
            return {"source": source.strip(), "target": target.strip(), "relation_type": ""}
    return None


def _gold_relations(content: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Read the contract's required relations, which are encoded four ways.

    Frozen contracts store relations as objects, as two-item pairs, as relation
    ids in four notations, and as short free text of the form
    "<entity> <relation> <entity>". Resolution is deterministic, prefers the
    contract's own relation table, and reports whatever it could not read, which
    includes the contracts that listed bare entity names in this field.
    """
    entries = content.get("required_relations") or []
    lookup = _relation_lookup(contract)
    known = _known_entities(content, contract)
    relations: list[dict[str, Any]] = []
    unresolved = 0
    for entry in entries:
        resolved = _resolve_relation_entry(entry, lookup, known)
        if resolved is None:
            unresolved += 1
            continue
        relations.append(resolved)
    return {"relations": relations, "unresolved": unresolved, "declared": len(entries)}


def _copy_asset(source: Path, target: Path, *, force: bool) -> None:
    """Copy a frozen asset, refusing to overwrite a different file silently."""
    if target.is_file():
        if _sha256_file(target) == _sha256_file(source):
            return
        if not force:
            raise ValueError(f"frozen asset differs from source, rerun with force: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _confirmation_template(paper_id: str) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "reviewer": "",
        "decision": "pending",
        "checks": {name: None for name in CONFIRMATION_CHECKS},
        "corrections": [],
        "notes": "",
        "updated_at": "",
    }


def prepare_gold_instances(
    out_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    papers_path: str | Path = "datasets/judge-framework-v001/papers.jsonl",
    figures_path: str | Path = "datasets/judge-framework-v001/figures.jsonl",
    reviews_path: str | Path = "datasets/judge-framework-review/human_framework_reviews.jsonl",
    contracts_dir: str | Path = "datasets/judge-framework-all/evidence/structured",
    pdf_root: str | Path = "experiments/literature_review/papers/pdf",
    categories_path: str | Path = DEFAULT_CATEGORIES_PATH,
    target: int = DEFAULT_INSTANCE_TARGET,
    include_splits: Sequence[str] = HELD_OUT_SPLITS,
    label_groups: Sequence[str] = DEFAULT_LABEL_GROUPS,
    force: bool = False,
) -> dict[str, Any]:
    """Freeze gold instances for EditableFigureBench.

    Selection is deterministic: every held-out paper whose reference figure was
    kept by a human reviewer and whose contract is non-empty becomes a candidate,
    all 'good' candidates are taken in hash order, and the remaining slots are
    filled with 'bad' candidates in hash order. `label_groups` narrows which
    reference-figure quality groups may be drawn from, so a release can be built
    from human-approved references alone. Re-running is idempotent.
    """
    root = Path(repo_root).resolve()
    out = Path(out_dir)
    out = out if out.is_absolute() else root / out
    papers = {str(row.get("paper_id")): row for row in _read_jsonl(root / papers_path)}
    figures: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(root / figures_path):
        figures.setdefault(str(row.get("paper_id")), []).append(row)
    reviews = {str(row.get("arxiv_id")): row for row in _read_jsonl(root / reviews_path) if row.get("arxiv_id")}
    categories = load_arxiv_categories(root / categories_path)

    excluded: collections.Counter[str] = collections.Counter()
    candidates: list[dict[str, Any]] = []
    for paper_id in sorted(papers):
        paper = papers[paper_id]
        if str(paper.get("split") or "") not in include_splits:
            excluded["outside_held_out_splits"] += 1
            continue
        review = reviews.get(paper_id) or {}
        if str(review.get("notes") or "").strip():
            excluded["review_flagged"] += 1
            continue
        group = _label_group(str(paper.get("final_label") or ""))
        if group is None:
            excluded["unscreened_label"] += 1
            continue
        if group not in tuple(label_groups):
            excluded["label_group_excluded"] += 1
            continue
        contract = _load_json(root / contracts_dir / f"{paper_id}.json")
        content = contract.get("expected_figure_content")
        content = content if isinstance(content, dict) else {}
        if not (content.get("required_entities") or content.get("required_relations")):
            excluded["empty_contract"] += 1
            continue
        reference = _resolve_reference(figures.get(paper_id) or [], root)
        if reference is None:
            excluded["no_reference_figure"] += 1
            continue
        pdf_path = root / pdf_root / f"{paper_id}.pdf"
        if not pdf_path.is_file():
            excluded["missing_pdf"] += 1
            continue
        candidates.append(
            {
                "paper_id": paper_id,
                "split": str(paper.get("split") or ""),
                "final_label": str(paper.get("final_label") or ""),
                "label_group": group,
                "pdf_path": pdf_path,
                "reference_path": root / str(reference.get("figure_path") or ""),
                "reference_figure_id": str(reference.get("figure_id") or ""),
                "candidate_source": str(reference.get("candidate_source") or ""),
                "candidate_rank": reference.get("candidate_rank"),
                "content": content,
                "extraction": contract.get("extraction") or {},
                "paper": paper,
            }
        )

    good = sorted((item for item in candidates if item["label_group"] == "good"), key=lambda item: _selection_key(item["paper_id"]))
    bad = sorted((item for item in candidates if item["label_group"] == "bad"), key=lambda item: _selection_key(item["paper_id"]))
    selected = good[: max(0, min(len(good), int(target)))]
    selected = selected + bad[: max(0, int(target) - len(selected))]
    selected.sort(key=lambda item: item["paper_id"])

    instances_dir = out / "instances"
    entries: list[dict[str, Any]] = []
    for item in selected:
        instance_dir = instances_dir / item["paper_id"]
        instance_dir.mkdir(parents=True, exist_ok=True)
        _copy_asset(item["pdf_path"], instance_dir / "paper.pdf", force=force)
        _copy_asset(item["reference_path"], instance_dir / "reference.png", force=force)
        contract_record = {
            "paper_id": item["paper_id"],
            "schema_version": "editable-figure-bench-contract-v001",
            "expected_figure_content": item["content"],
            "provenance": {
                "extraction_model": (item["extraction"] or {}).get("model"),
                "prompt_version": (item["extraction"] or {}).get("prompt_version"),
                "source": "datasets/judge-framework-all/evidence/structured",
            },
        }
        write_json(instance_dir / "contract.json", contract_record)
        domain_record = domain_fields(item["paper_id"], categories)
        write_json(
            instance_dir / "paper.json",
            {
                "paper_id": item["paper_id"],
                "title": item["paper"].get("title") or "",
                "abstract": item["paper"].get("abstract") or "",
                "split": item["split"],
                "cohort": _cohort(item["split"]),
                "final_label": item["final_label"],
                "label_group": item["label_group"],
                "reference_figure_id": item["reference_figure_id"],
                "candidate_source": item["candidate_source"],
                "candidate_rank": item["candidate_rank"],
                **domain_record,
                "source": {
                    "papers": str(papers_path),
                    "figures": str(figures_path),
                    "reviews": str(reviews_path),
                    "categories": str(categories_path),
                },
            },
        )
        confirmation = instance_dir / "confirmation.json"
        if not confirmation.is_file():
            write_json(confirmation, _confirmation_template(item["paper_id"]))
        entries.append(
            {
                "paper_id": item["paper_id"],
                "split": item["split"],
                "cohort": _cohort(item["split"]),
                "final_label": item["final_label"],
                "label_group": item["label_group"],
                "reference_figure_id": item["reference_figure_id"],
                "domain": domain_record["domain"],
                "image_sha256": _sha256_file(instance_dir / "reference.png"),
                "pdf_sha256": _sha256_file(instance_dir / "paper.pdf"),
                "contract_sha256": _sha256_file(instance_dir / "contract.json"),
                "required_entity_count": len(item["content"].get("required_entities") or []),
                "required_relation_count": len(item["content"].get("required_relations") or []),
                "must_not_invent_count": len(item["content"].get("must_not_invent") or []),
            }
        )

    manifest = {
        "dataset_version": GOLD_DATASET_VERSION,
        "task": "paper_to_editable_framework_figure",
        "instance_count": len(entries),
        "label_counts": collections.Counter(entry["label_group"] for entry in entries),
        "split_counts": collections.Counter(entry["split"] for entry in entries),
        "candidate_pool": len(candidates),
        "excluded": dict(excluded),
        "selection": {
            "target": int(target),
            "rule": "all held-out good references in hash order, then remaining allowed groups in hash order to fill the target",
            "include_splits": list(include_splits),
            "label_groups": list(label_groups),
        },
        "source": {
            "papers": str(papers_path),
            "figures": str(figures_path),
            "reviews": str(reviews_path),
            "contracts": str(contracts_dir),
            "pdf_root": str(pdf_root),
            "categories": str(categories_path),
        },
        "cohorts": {
            "labels": dict(COHORT_NOTES),
            "counts": {
                name: sum(1 for entry in entries if entry["cohort"] == name)
                for name in COHORT_ORDER
                if any(entry["cohort"] == name for entry in entries)
            },
        },
        "domains": {
            "schema_version": DOMAIN_SCHEMA_VERSION,
            "available": bool(categories),
            "labels": dict(DOMAIN_LABELS),
            "counts": summarize_domains(entry["domain"] for entry in entries),
            "unlabeled": [entry["paper_id"] for entry in entries if not entry["domain"]],
        },
        "instances": entries,
    }
    manifest["label_counts"] = dict(manifest["label_counts"])
    manifest["split_counts"] = dict(manifest["split_counts"])
    write_json(out / "manifest.json", manifest)
    write_text(out / "COMPLETE", f"{GOLD_DATASET_VERSION}\n{len(entries)} instances\n")
    return {
        "summary": "EditableFigureBench gold instances frozen.",
        "ok": True,
        "dataset_version": GOLD_DATASET_VERSION,
        "out_dir": str(out),
        "instances_dir": str(instances_dir),
        "instance_count": len(entries),
        "label_counts": manifest["label_counts"],
        "split_counts": manifest["split_counts"],
        "cohort_counts": manifest["cohorts"]["counts"],
        "domain_counts": manifest["domains"]["counts"],
        "candidate_pool": len(candidates),
        "excluded": dict(excluded),
    }


def verify_gold_instances(gold_root: str | Path) -> dict[str, Any]:
    """Check a frozen gold set against its own manifest."""
    root = Path(gold_root)
    errors: list[str] = []
    warnings: list[str] = []
    manifest = _load_json(root / "manifest.json")
    if not manifest:
        return {"summary": "Gold verification.", "ok": False, "errors": [f"missing manifest: {root / 'manifest.json'}"], "warnings": []}
    if not (root / "COMPLETE").is_file():
        errors.append("missing COMPLETE marker")
    entries = manifest.get("instances")
    if not isinstance(entries, list):
        return {"summary": "Gold verification.", "ok": False, "errors": ["manifest has no instances list"], "warnings": []}
    for entry in entries:
        paper_id = str(entry.get("paper_id") or "")
        instance_dir = root / "instances" / paper_id
        if not instance_dir.is_dir():
            errors.append(f"missing instance directory: {paper_id}")
            continue
        for name in INSTANCE_FILES:
            if not (instance_dir / name).is_file():
                errors.append(f"missing instance file: {paper_id}/{name}")
        for name, key in (("reference.png", "image_sha256"), ("paper.pdf", "pdf_sha256")):
            path = instance_dir / name
            expected = entry.get(key)
            if path.is_file() and expected and _sha256_file(path) != expected:
                errors.append(f"frozen asset changed: {paper_id}/{name}")
        contract = _load_json(instance_dir / "contract.json")
        if not isinstance(contract.get("expected_figure_content"), dict):
            errors.append(f"contract is not a frozen figure contract: {paper_id}")
        if "domain" in entry or "cohort" in entry:
            paper_record = _load_json(instance_dir / "paper.json")
            if entry.get("domain") != paper_record.get("domain"):
                errors.append(f"domain label drifted between manifest and paper.json: {paper_id}")
            if "cohort" in entry and entry.get("cohort") != paper_record.get("cohort"):
                errors.append(f"cohort drifted between manifest and paper.json: {paper_id}")
    if entries and int(manifest.get("instance_count") or 0) != len(entries):
        errors.append("manifest instance_count does not match its instances list")
    declared_cohorts = manifest.get("cohorts")
    if isinstance(declared_cohorts, dict):
        expected = collections.Counter(entry.get("cohort") for entry in entries)
        if {name: expected[name] for name in COHORT_ORDER if expected[name]} != declared_cohorts.get("counts"):
            errors.append("manifest cohort counts do not match its instances")
    declared = manifest.get("domains")
    if isinstance(declared, dict) and declared.get("available"):
        if summarize_domains(entry.get("domain") for entry in entries) != declared.get("counts"):
            errors.append("manifest domain counts do not match its instances")
    else:
        warnings.append("gold set carries no domain strata")
    return {
        "summary": "Gold verification.",
        "ok": not errors,
        "gold_root": str(root),
        "instance_count": len(entries),
        "errors": errors,
        "warnings": warnings,
    }


def _render_text(run_dir: Path, program: dict[str, Any], deck_text: str) -> str:
    """Text a reader can see, from the deck and from the planned text layer."""
    parts = [deck_text]
    text_program = program.get("text_program")
    if isinstance(text_program, dict):
        for item in text_program.get("items") or []:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
    return "\n".join(part for part in parts if part)


def resolve_embedder(spec: str | None = None) -> tuple[Any, str, str]:
    """Resolve the optional L1 embedder from `EMBEDDING_ENV_VAR`.

    Returns `(embedder, spec, reason)`. The embedder is `None` when the level is
    switched off, when the spec is not an API spec, or when the provider cannot be
    imported. Nothing raises: scoring has to stay runnable offline, and the reason
    is recorded in the report instead of being hidden behind a silent fallback.
    """
    resolved = str(spec if spec is not None else os.getenv(EMBEDDING_ENV_VAR, "")).strip()
    if not resolved:
        return None, resolved, "disabled"
    try:
        from ..providers.api_embeddings import api_embedder, api_model_name, is_api_spec
    except Exception as exc:  # pragma: no cover - import failure is environment specific
        return None, resolved, "provider_import_failed: " + str(exc)[:120]
    if not is_api_spec(resolved):
        return None, resolved, "unsupported_spec"
    return api_embedder(api_model_name(resolved)), resolved, ""


def _embedder_provenance(embedder: Any, spec: str, reason: str) -> dict[str, Any]:
    """Describe the embedder that the L1 layer did or did not use."""
    block: dict[str, Any] = {"env_var": EMBEDDING_ENV_VAR, "spec": spec, "used": embedder is not None}
    if reason:
        block["reason"] = reason
    if embedder is not None:
        for name in ("fingerprint", "usage"):
            method = getattr(embedder, name, None)
            if callable(method):
                try:
                    block[name] = method()
                except Exception as exc:  # pragma: no cover - provider specific
                    block[name] = {"error": str(exc)[:120]}
    return block


def _core_modules_block(
    gold: Sequence[str],
    predicted: Sequence[str],
    *,
    embedder: Any = None,
) -> dict[str, Any]:
    """Module coverage and unbound drawn nodes, with notation split out.

    A generated figure legitimately draws more boxes than the frozen contract
    lists, and a planner can turn an equation into a node. Neither should be
    scored as a missing or an invented module, so this block reports coverage
    over named modules and keeps the notation and unbound labels for review. The
    `must_not_invent` hard gate, not this block, is the fabrication gate.
    """
    gold_modules, gold_notation = split_notation(gold)
    predicted_modules, predicted_notation = split_notation(predicted)
    coverage = {
        "strict": module_coverage(gold, predicted, mode="strict"),
        "soft": module_coverage(gold, predicted, mode="soft"),
    }
    unbound = {
        "strict": unbound_nodes(gold, predicted, mode="strict"),
        "soft": unbound_nodes(gold, predicted, mode="soft"),
    }
    if embedder is not None:
        try:
            coverage["embedding"] = module_coverage(gold, predicted, mode="embedding", embedder=embedder)
            unbound["embedding"] = unbound_nodes(gold, predicted, mode="embedding", embedder=embedder)
        except Exception as exc:  # pragma: no cover - provider specific
            coverage["embedding"] = {"available": False, "reason": str(exc)[:200]}
            unbound["embedding"] = {"available": False, "reason": str(exc)[:200]}
    return {
        "summary": "Module coverage and unbound drawn nodes; notation is split out, never scored as a module.",
        "gold_module_count": len(gold_modules),
        "gold_notation_count": len(gold_notation),
        "predicted_module_count": len(predicted_modules),
        "predicted_notation_count": len(predicted_notation),
        "notation_ratio": round(len(predicted_notation) / len(predicted), 4) if predicted else 0.0,
        "notation_modules_matched": {
            "strict": coverage["strict"].get("notation_modules_matched", 0),
            "soft": coverage["soft"].get("notation_modules_matched", 0),
        },
        "coverage": coverage,
        "unbound": unbound,
    }


def _deck_predictions(
    pptx_path: Path,
    gold_entities: Sequence[str],
    embedder: Any = None,
) -> dict[str, Any]:
    """Read entity predictions out of the delivered deck itself.

    Systems that never publish a semantic program (a text-to-PPTX baseline, for
    instance) can still be scored on content fidelity, because their node labels
    are the text frames they drew. Relations are reported unavailable rather than
    guessed: a deck-only connector carries no attachment metadata, so the
    endpoints are genuinely unknown.
    """
    if not pptx_path.is_file():
        return {"available": False, "reason": "missing editable_composition.pptx"}
    labels = extract_shape_labels(pptx_path)
    block: dict[str, Any] = {
        "available": True,
        "source": "deck_text_frames",
        "predicted_entity_count": len(labels),
        "entity_strict": match_entities(gold_entities, labels, mode="strict"),
        "entity_soft": match_entities(gold_entities, labels, mode="soft"),
        "core_modules": _core_modules_block(gold_entities, labels, embedder=embedder),
        "relations": {
            "available": False,
            "reason": "deck-only connectors expose no a:stCxn/a:endCxn attachment, so endpoints are unknown",
        },
    }
    if embedder is not None:
        try:
            block["entity_embedding"] = match_entities(gold_entities, labels, mode="embedding", embedder=embedder)
        except Exception as exc:
            block["entity_embedding"] = {"available": False, "reason": str(exc)[:200]}
            block["core_modules"]["coverage"]["embedding"] = {"available": False, "reason": str(exc)[:200]}
    return block


def _optional_text(instance_path: Path, names: Sequence[str]) -> str:
    for name in names:
        path = instance_path / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return ""


def _delivery_kind(run_path: Path, pptx_path: Path, deck_report: dict[str, Any]) -> str:
    if pptx_path.is_file() and deck_report.get("ooxml_valid"):
        return "pptx"
    if pptx_path.is_file():
        return "unreadable"
    note = _load_json(run_path / "delivery.json")
    if str(note.get("kind") or "") == "non_ooxml":
        return "non_ooxml"
    if any((run_path / name).is_file() for name in ("figure.tex", "figure.png", "preview.png")):
        return "non_ooxml"
    return "missing"


def _image_scoring_requested(paper_quiz_call: Any, nine_call: Any, clip_embedder: Any) -> bool:
    if paper_quiz_call is not None or nine_call is not None or clip_embedder is not None:
        return True
    quiz = os.getenv("RFS_PAPER_QUIZ", "").strip().lower() == "api"
    nine = os.getenv("RFS_NINE_JUDGE", "").strip().lower() == "api"
    return quiz or nine


def _ensure_preview(run_path: Path, pptx_path: Path) -> tuple[Path, str | None]:
    preview = run_path / "preview.png"
    if preview.is_file():
        return preview, None
    if not pptx_path.is_file():
        return preview, None
    try:
        from ..composition.pptx_render import render_pptx_page

        render_pptx_page(pptx_path, preview)
    except Exception as exc:
        return preview, f"{type(exc).__name__}: {exc}"[:200]
    if not preview.is_file():
        return preview, "render produced no preview.png"
    return preview, None


def _main_table(
    *,
    instance_path: Path,
    run_path: Path,
    pptx_path: Path,
    deck_report: dict[str, Any],
    paper_quiz_call: Any = None,
    nine_call: Any = None,
    clip_embedder: Any = None,
) -> dict[str, Any]:
    """The four own-set columns. Contract coverage is not one of them."""
    kind = _delivery_kind(run_path, pptx_path, deck_report)
    if kind == "non_ooxml":
        editability = editability_verdict({}, delivery="non_ooxml")
    elif kind == "pptx":
        editability = deck_report.get("editability") or editability_verdict(deck_report)
    else:
        editability = editability_verdict({"ok": False, "ooxml_valid": False, "error": "missing editable_composition.pptx"})
    preview = run_path / "preview.png"
    render_error = None
    if _image_scoring_requested(paper_quiz_call, nine_call, clip_embedder):
        preview, render_error = _ensure_preview(run_path, pptx_path)
    paper_text = _optional_text(instance_path, ("paper.md", "paper.txt"))
    method_text = _optional_text(instance_path, ("method_section.txt", "method.txt"))
    return {
        "summary": "Own-set main table: PaperQuiz, CLIP, nine VLM scores, and editability.",
        "paper_quiz": score_paper_quiz(preview, paper_text, call=paper_quiz_call, render_error=render_error),
        "paper_figure_clip": score_paper_figure_clip(
            preview, method_text, embedder=clip_embedder, render_error=render_error
        ),
        "vlm_nine": score_nine(
            preview, method_text=method_text or paper_text, call=nine_call, render_error=render_error
        ),
        "editability": editability,
        "preview_render_error": render_error,
    }


def evaluate_instance(
    instance_dir: str | Path,
    run_dir: str | Path,
    *,
    system_id: str = "unknown",
    layers: Sequence[str] = LAYERS,
    paper_quiz_call: Any = None,
    nine_call: Any = None,
    clip_embedder: Any = None,
) -> dict[str, Any]:
    """Score one system output against one frozen instance."""
    instance_path = Path(instance_dir)
    run_path = Path(run_dir)
    contract = _load_json(instance_path / "contract.json")
    content = contract.get("expected_figure_content")
    content = content if isinstance(content, dict) else {}
    gold_entities = [str(value) for value in (content.get("required_entities") or [])]
    relation_reading = _gold_relations(content, contract)
    gold_relations = relation_reading["relations"]
    banned = [str(value) for value in (content.get("must_not_invent") or [])]

    pptx_path = run_path / "editable_composition.pptx"
    program = _load_json(run_path / "figure_program.json")
    preview_path = run_path / "preview.png"

    l2: dict[str, Any] = {}
    deck_text = ""
    if pptx_path.is_file():
        l2 = measure_native_editability(pptx_path, expected_connector_count=len(gold_relations) or None, include_text=True)
        deck_text = str(l2.pop("editable_text", "") or "")
    else:
        l2 = {"ok": False, "ooxml_valid": False, "error": "missing editable_composition.pptx"}

    report: dict[str, Any] = {
        "summary": "EditableFigureBench instance metrics.",
        "paper_id": instance_path.name,
        "system_id": system_id,
        "gold": {
            "split": "",
            "final_label": "",
            "label_group": "",
            "entity_count": len(gold_entities),
            "relation_count": len(gold_relations),
            "relation_count_declared": relation_reading["declared"],
            "relation_count_unresolved": relation_reading["unresolved"],
            "must_not_invent_count": len(banned),
        },
        "artifacts": {
            "editable_composition": pptx_path.is_file(),
            "figure_program": bool(program),
            "preview": preview_path.is_file(),
        },
        "layers": list(layers),
    }
    paper_record = _load_json(instance_path / "paper.json")
    report["gold"]["split"] = str(paper_record.get("split") or "")
    report["gold"]["final_label"] = str(paper_record.get("final_label") or "")
    report["gold"]["label_group"] = str(paper_record.get("label_group") or "")
    report["gold"]["domain"] = str(paper_record.get("domain") or "")
    report["gold"]["cohort"] = str(paper_record.get("cohort") or "")

    rendered_text = _render_text(run_path, program, deck_text)
    rendered_tokens = _tokens(rendered_text)

    nodes = [node for node in (program.get("semantic_nodes") or []) if isinstance(node, dict)]
    predicted_entities = [str(node.get("label") or node.get("id") or "") for node in nodes]
    predicted_entities = [value for value in predicted_entities if value]
    predicted_relations: list[tuple[str, str, str]] = []
    for arrow in program.get("arrows") or []:
        if not isinstance(arrow, dict):
            continue
        try:
            predicted_relations.append(relation_triple(arrow))
        except (ValueError, TypeError):
            continue

    hits: list[str] = []
    for term in banned:
        needle = _tokens(term)
        if _covered(needle, rendered_tokens) or any(_covered(needle, _tokens(label)) for label in predicted_entities):
            hits.append(term)
    gates = {
        "must_not_invent_hit": bool(hits),
        "missing_editable_composition": not pptx_path.is_file(),
        "unreadable_powerpoint": pptx_path.is_file() and not l2.get("ooxml_valid", False),
    }
    report["hard_gates"] = {
        "gates": dict(gates),
        "failed": any(gates.values()),
        "failed_gates": sorted(name for name, value in gates.items() if value),
        "must_not_invent_hits": hits,
    }

    if "L1" in layers:
        embedder, embedder_spec, embedder_reason = resolve_embedder()
        l1: dict[str, Any] = {
            "summary": "Content fidelity against the frozen contract.",
            "predicted_entity_count": len(predicted_entities),
            "predicted_relation_count": len(predicted_relations),
            "entity_strict": match_entities(gold_entities, predicted_entities, mode="strict"),
            "entity_soft": match_entities(gold_entities, predicted_entities, mode="soft"),
            "core_modules": _core_modules_block(gold_entities, predicted_entities, embedder=embedder),
            "relation_strict": match_relations(gold_relations, predicted_relations, mode="strict"),
            "relation_soft": match_relations(gold_relations, predicted_relations, mode="soft"),
            "rendered": rendered_entity_recall(gold_entities, rendered_text),
            "deck": _deck_predictions(pptx_path, gold_entities, embedder),
        }
        if embedder is not None:
            if predicted_entities:
                try:
                    l1["entity_embedding"] = match_entities(
                        gold_entities, predicted_entities, mode="embedding", embedder=embedder
                    )
                except Exception as exc:
                    embedder_reason = "embedding_failed: " + str(exc)[:160]
            else:
                l1["entity_embedding"] = {
                    "available": False,
                    "reason": "no program-derived entity predictions; see the deck block",
                }
            deck_error = (l1["deck"].get("entity_embedding") or {}).get("reason")
            if not embedder_reason and deck_error:
                embedder_reason = "embedding_failed: " + str(deck_error)[:160]
        l1["embedder"] = _embedder_provenance(None if embedder_reason else embedder, embedder_spec, embedder_reason)
        report["L1"] = l1
    if "L2" in layers:
        report["L2"] = dict(l2)
        report["L2"]["connector_visibility"] = evaluate_connector_visibility(program) if program else {"issue_count": None}
    if "L3" in layers:
        report["L3"] = {
            "summary": "Layout and readability of the delivered figure.",
            "geometry": evaluate_geometry_quality(program) if program else {},
            "density": evaluate_density_quality(program) if program else {},
            "readability": evaluate_text_readability(program) if program else {},
            "rendered_density": measure_rendered_density(preview_path) if preview_path.is_file() else {},
            "layout": measure_layout_alignment(pptx_path),
            "aesthetics": judge_aesthetics(preview_path),
        }
    report["main_table"] = _main_table(
        instance_path=instance_path,
        run_path=run_path,
        pptx_path=pptx_path,
        deck_report=l2,
        paper_quiz_call=paper_quiz_call,
        nine_call=nine_call,
        clip_embedder=clip_embedder,
    )
    return report


def _resolve_systems(systems_dir: str | Path, instance_ids: Sequence[str], system_id: str | None) -> list[tuple[str, Path]]:
    root = Path(systems_dir)
    if not root.is_dir():
        raise ValueError(f"systems directory does not exist: {root}")
    if system_id:
        nested = root / system_id
        return [(system_id, nested if nested.is_dir() else root)]
    if instance_ids and (root / instance_ids[0]).is_dir():
        return [(root.name, root)]
    systems = [(child.name, child) for child in sorted(root.iterdir()) if child.is_dir()]
    if not systems:
        raise ValueError(f"no systems found under {root}")
    return systems


def run_benchmark_instances(
    *,
    instances_dir: str | Path,
    systems_dir: str | Path,
    out_dir: str | Path,
    layers: Sequence[str] = LAYERS,
    system_id: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Score every system under systems_dir against every frozen instance."""
    instances_root = Path(instances_dir)
    if not instances_root.is_dir():
        raise ValueError(f"instances directory does not exist: {instances_root}")
    instance_ids = sorted(child.name for child in instances_root.iterdir() if child.is_dir() and (child / "contract.json").is_file())
    if limit is not None:
        instance_ids = instance_ids[: int(limit)]
    if not instance_ids:
        raise ValueError(f"no frozen instances found under {instances_root}")
    systems = _resolve_systems(systems_dir, instance_ids, system_id)
    out = Path(out_dir)
    summaries: list[dict[str, Any]] = []
    for identifier, root in systems:
        scored = 0
        failed = 0
        missing = 0
        for paper_id in instance_ids:
            run_path = root / paper_id
            if not run_path.is_dir():
                missing += 1
            metrics = evaluate_instance(instances_root / paper_id, run_path, system_id=identifier, layers=layers)
            write_json(out / identifier / paper_id / "metrics.json", metrics)
            scored += 1
            failed += int(metrics["hard_gates"]["failed"])
        summaries.append(
            {
                "system_id": identifier,
                "system_dir": str(root),
                "scored_instances": scored,
                "missing_run_directories": missing,
                "hard_gate_failures": failed,
            }
        )
    report = {
        "summary": "EditableFigureBench run report.",
        "ok": True,
        "instances_dir": str(instances_root),
        "systems_dir": str(systems_dir),
        "out_dir": str(out),
        "instance_count": len(instance_ids),
        "layers": list(layers),
        "systems": summaries,
    }
    write_json(out / "run_report.json", report)
    return report


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _layer_summary(records: Sequence[dict[str, Any]], layer: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, path in SUMMARY_PATHS[layer]:
        values = [value for value in (_dig(record, path) for record in records) if isinstance(value, (int, float)) and not isinstance(value, bool)]
        summary[name] = _aggregate(values)
    return summary


def _system_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gates = {name: sum(1 for record in records if _dig(record, ("hard_gates", "gates", name))) for name in HARD_GATES}
    passed = sum(1 for record in records if not _dig(record, ("hard_gates", "failed")))
    total = len(records)
    return {
        "instance_count": total,
        "hard_gate_passed": passed,
        "hard_gate_pass_rate": round(passed / total, 4) if total else None,
        "hard_gate_failures": gates,
        "layers": {layer: _layer_summary(records, layer) for layer in LAYERS if any(layer in record for record in records)},
        "by_reference_quality": {
            group: {
                layer: _layer_summary(
                    [record for record in records if _dig(record, ("gold", "label_group")) == group],
                    layer,
                )
                for layer in ("L1", "L2")
                if any(layer in record for record in records)
            }
            for group in ("good", "bad")
        },
        "by_cohort": {
            cohort: {
                layer: _layer_summary(
                    [record for record in records if _dig(record, ("gold", "cohort")) == cohort],
                    layer,
                )
                for layer in ("L1", "L2")
                if any(layer in record for record in records)
            }
            for cohort in COHORT_ORDER
            if any(_dig(record, ("gold", "cohort")) == cohort for record in records)
        },
        "main_table": _main_table_summary(records),
        "by_domain": {
            domain: {
                layer: _layer_summary(
                    [record for record in records if _dig(record, ("gold", "domain")) == domain],
                    layer,
                )
                for layer in ("L1", "L2")
                if any(layer in record for record in records)
            }
            for domain in DOMAIN_ORDER
            if any(_dig(record, ("gold", "domain")) == domain for record in records)
        },
    }


def _main_table_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate own-set editability. Module and edge coverage are not columns."""
    statuses: dict[str, int] = {}
    for record in records:
        status = str(((record.get("main_table") or {}).get("editability") or {}).get("status") or "")
        if status:
            statuses[status] = statuses.get(status, 0) + 1
    return {"editability_counts": statuses}


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def _score_table_markdown(report: dict[str, Any]) -> str:
    blocks = ["# EditableFigureBench score table", ""]
    blocks.append(f"Instances scored per system: {report.get('instance_count')}. Hard gates are reported as failures, never folded into a score.")
    blocks.append("")
    for system_id, summary in sorted(report["systems"].items()):
        blocks.append(f"## {system_id}")
        blocks.append("")
        blocks.append(
            _markdown_table(
                ("gate", "failed instances"),
                [(name, summary["hard_gate_failures"].get(name, 0)) for name in HARD_GATES],
            )
        )
        blocks.append("")
        blocks.append(f"Hard gate pass rate: {summary['hard_gate_pass_rate']} ({summary['hard_gate_passed']}/{summary['instance_count']})")
        blocks.append("")
        for layer, metrics in summary["layers"].items():
            blocks.append(f"### {system_id} {layer}")
            blocks.append("")
            blocks.append(
                _markdown_table(
                    ("metric", "n", "mean", "median", "min", "max"),
                    [(name, value["count"], value["mean"], value["median"], value["min"], value["max"]) for name, value in metrics.items()],
                )
            )
            blocks.append("")
        for group, group_layers in sorted(summary["by_reference_quality"].items()):
            for layer, metrics in sorted(group_layers.items()):
                blocks.append(f"### {system_id} reference quality: {group} {layer}")
                blocks.append("")
                blocks.append(
                    _markdown_table(
                        ("metric", "n", "mean", "median"),
                        [(name, value["count"], value["mean"], value["median"]) for name, value in metrics.items()],
                    )
                )
                blocks.append("")
        for cohort, cohort_layers in sorted(summary.get("by_cohort", {}).items()):
            for layer, metrics in sorted(cohort_layers.items()):
                blocks.append(f"### {system_id} cohort: {cohort} {layer}")
                blocks.append("")
                blocks.append(
                    _markdown_table(
                        ("metric", "n", "mean", "median"),
                        [(name, value["count"], value["mean"], value["median"]) for name, value in metrics.items()],
                    )
                )
                blocks.append("")
        for domain, domain_layers in sorted(summary.get("by_domain", {}).items()):
            for layer, metrics in sorted(domain_layers.items()):
                blocks.append(f"### {system_id} domain: {domain} {layer}")
                blocks.append("")
                blocks.append(
                    _markdown_table(
                        ("metric", "n", "mean", "median"),
                        [(name, value["count"], value["mean"], value["median"]) for name, value in metrics.items()],
                    )
                )
                blocks.append("")
    return "\n".join(blocks)


def score_benchmark(
    *,
    runs_dir: str | Path,
    out_dir: str | Path,
    layers: Sequence[str] = LAYERS,
) -> dict[str, Any]:
    """Aggregate per-instance metrics into the four-layer report."""
    runs_root = Path(runs_dir)
    if not runs_root.is_dir():
        raise ValueError(f"runs directory does not exist: {runs_root}")
    per_system: dict[str, list[dict[str, Any]]] = {}
    for system_dir in sorted(child for child in runs_root.iterdir() if child.is_dir()):
        records: list[dict[str, Any]] = []
        for metrics_path in sorted(system_dir.glob("*/metrics.json")):
            record = _load_json(metrics_path)
            if record:
                records.append(record)
        if records:
            per_system[system_dir.name] = records
    if not per_system:
        raise ValueError(f"no metrics.json found under {runs_root}")
    report = {
        "summary": "EditableFigureBench score report.",
        "ok": True,
        "runs_dir": str(runs_root),
        "instance_count": max(len(records) for records in per_system.values()),
        "layers": list(layers),
        "systems": {system_id: _system_summary(records) for system_id, records in sorted(per_system.items())},
        "external_benchmarks": {
            "sciflow": sciflow_record(),
            "figurebench": figurebench_record(),
        },
    }
    out = Path(out_dir)
    write_json(out / "score_report.json", report)
    write_text(out / "score_table.md", _score_table_markdown(report))
    return report


__all__ = [
    "BAD_LABEL",
    "CONFIRMATION_CHECKS",
    "DEFAULT_INSTANCE_TARGET",
    "GOLD_DATASET_VERSION",
    "GOOD_LABELS",
    "HARD_GATES",
    "HELD_OUT_SPLITS",
    "INSTANCE_FILES",
    "LAYERS",
    "SOURCE_PRIORITY",
    "evaluate_instance",
    "prepare_gold_instances",
    "run_benchmark_instances",
    "score_benchmark",
    "verify_gold_instances",
]
