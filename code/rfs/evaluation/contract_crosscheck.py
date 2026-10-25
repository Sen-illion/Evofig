"""Deterministic comparison of the two lines' figure specifications.

The middle layer decides what a paper's figure has to show and the generation
line draws it. When each line planned that answer for itself the two answers
differed, and that difference is a variable inside any paired data the Judge
learns from. This module measures it and nothing else: it reads two
specifications, reports what only one of them has, and never rewrites either.

Two uses. Verifying that a run which adopted the middle layer's contract really
carries it, and quantifying what a run that planned its own contract changed.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from ..contracts.method_figure import METHOD_INVENTORY_FILE, method_inventory_path
from ..utils import read_json, write_json

ENTITY_FIELDS = ("inputs", "modules", "outputs", "innovations")


def _label(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item or "").strip()
    for key in ("visible_label", "label", "name", "text", "statement", "id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""


def _label_key(value: Any) -> str:
    """Reduce a label to what two pipelines have to agree on to be the same object."""
    text = unicodedata.normalize("NFKC", _label(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _entities(specification: dict[str, Any], field: str) -> dict[str, dict[str, Any]]:
    entries = specification.get(field) if isinstance(specification, dict) else None
    found: dict[str, dict[str, Any]] = {}
    for item in entries if isinstance(entries, list) else []:
        key = _label_key(item)
        if key and key not in found:
            found[key] = {"label": _label(item), "must_show": bool((item or {}).get("must_show"))}
    return found


def _entity_ids(specification: dict[str, Any]) -> dict[str, str]:
    identifiers: dict[str, str] = {}
    for field in ENTITY_FIELDS:
        for item in (specification.get(field) or []) if isinstance(specification, dict) else []:
            if isinstance(item, dict):
                identifier = str(item.get("id") or item.get("instance_id") or "").strip()
                if identifier and identifier not in identifiers:
                    identifiers[identifier] = _label(item)
    return identifiers


def _relations(specification: dict[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    identifiers = _entity_ids(specification)
    entries = specification.get("relations") if isinstance(specification, dict) else None
    found: dict[tuple[str, str], dict[str, str]] = {}
    for item in entries if isinstance(entries, list) else []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or item.get("source_id") or "").strip()
        target = str(item.get("target") or item.get("target_id") or "").strip()
        key = (
            _label_key(identifiers.get(source, source)),
            _label_key(identifiers.get(target, target)),
        )
        if not any(key) or key in found:
            continue
        found[key] = {
            "relation": f"{identifiers.get(source, source)} -> {identifiers.get(target, target)}",
            "type": str(item.get("type") or item.get("relation_type") or "").strip(),
        }
    return found


def _agreement(shared: int, left_only: int, right_only: int) -> float:
    total = shared + left_only + right_only
    return 1.0 if not total else round(shared / total, 4)


def _side(specification: dict[str, Any]) -> dict[str, Any]:
    return {
        "topology": str((specification or {}).get("topology") or ""),
        "entity_counts": {field: len(_entities(specification, field)) for field in ENTITY_FIELDS},
        "relation_count": len(_relations(specification)),
        "panel_graph_count": len((specification or {}).get("panel_graphs") or []) if isinstance((specification or {}).get("panel_graphs"), list) else 0,
    }


def _difference(left: dict[str, Any], right: dict[str, Any]) -> tuple[list[Any], list[Any], list[Any]]:
    shared = sorted(set(left) & set(right))
    return shared, sorted(set(left) - set(right)), sorted(set(right) - set(left))


def compare_figure_specifications(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Report what one figure specification has that the other one does not.

    Entities are compared inside their own role, by visible label rather than by
    id, because the two lines name the same concept differently. Relations are
    compared by the labels of their endpoints, so a relation that survived both
    readings but changed its type is reported as a type mismatch rather than as
    two unrelated relations.
    """
    entities: dict[str, Any] = {}
    for field in ENTITY_FIELDS:
        left_items = _entities(left, field)
        right_items = _entities(right, field)
        shared, left_only, right_only = _difference(left_items, right_items)
        entities[field] = {
            "shared": [left_items[key]["label"] for key in shared],
            "only_left": [left_items[key]["label"] for key in left_only],
            "only_right": [right_items[key]["label"] for key in right_only],
            "agreement": _agreement(len(shared), len(left_only), len(right_only)),
            "must_show_mismatches": [
                left_items[key]["label"]
                for key in shared
                if left_items[key]["must_show"] != right_items[key]["must_show"]
            ],
        }
    left_relations = _relations(left)
    right_relations = _relations(right)
    shared_relations, left_only_relations, right_only_relations = _difference(left_relations, right_relations)
    relation_report = {
        "shared": [left_relations[key]["relation"] for key in shared_relations],
        "only_left": [left_relations[key]["relation"] for key in left_only_relations],
        "only_right": [right_relations[key]["relation"] for key in right_only_relations],
        "agreement": _agreement(len(shared_relations), len(left_only_relations), len(right_only_relations)),
        "type_mismatches": [
            {
                "relation": left_relations[key]["relation"],
                "left": left_relations[key]["type"],
                "right": right_relations[key]["type"],
            }
            for key in shared_relations
            if left_relations[key]["type"] != right_relations[key]["type"]
        ],
    }
    differing = (
        sum(len(entities[field][side]) for field in ENTITY_FIELDS for side in ("only_left", "only_right"))
        + sum(len(entities[field]["must_show_mismatches"]) for field in ENTITY_FIELDS)
        + len(relation_report["only_left"])
        + len(relation_report["only_right"])
        + len(relation_report["type_mismatches"])
    )
    left_topology = _side(left)["topology"]
    right_topology = _side(right)["topology"]
    return {
        "summary": "Deterministic comparison of two figure specifications for one paper.",
        "left": _side(left),
        "right": _side(right),
        "topology": {"left": left_topology, "right": right_topology, "agree": left_topology == right_topology},
        "entities": entities,
        "relations": relation_report,
        "differences": differing + (0 if left_topology == right_topology else 1),
        "agree": differing == 0 and left_topology == right_topology,
    }


def crosscheck_contract_directories(left_dir: str | Path, right_dir: str | Path) -> dict[str, Any]:
    """Compare the figure specification the two directories carry for one paper."""
    paths: dict[str, Path] = {}
    for side, directory in (("left", left_dir), ("right", right_dir)):
        path = method_inventory_path(directory)
        if path is None:
            raise FileNotFoundError(f"{side} directory carries no {METHOD_INVENTORY_FILE}: {Path(directory)}")
        paths[side] = path
    report = compare_figure_specifications(read_json(paths["left"]), read_json(paths["right"]))
    report["left"]["path"] = str(paths["left"].resolve())
    report["right"]["path"] = str(paths["right"].resolve())
    return report


def crosscheck_contract_pairs(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Compare every pair and summarise how far the two lines agree overall."""
    reports = []
    for left, right in pairs:
        report = crosscheck_contract_directories(left, right)
        reports.append({
            "left": report["left"]["path"],
            "right": report["right"]["path"],
            "agree": report["agree"],
            "differences": report["differences"],
            "entity_agreement": {field: report["entities"][field]["agreement"] for field in ENTITY_FIELDS},
            "relation_agreement": report["relations"]["agreement"],
            "topology_agree": report["topology"]["agree"],
            "report": report,
        })
    agreeing = sum(1 for report in reports if report["agree"])
    return {
        "summary": "Deterministic comparison of the middle layer's figure contracts against the generation line's.",
        "pairs": reports,
        "pair_count": len(reports),
        "agreeing_pairs": agreeing,
        "agreeing_ratio": round(agreeing / len(reports), 4) if reports else 1.0,
        "total_differences": sum(report["differences"] for report in reports),
    }


def run_contract_crosscheck(
    left_dirs: Iterable[str | Path] | None,
    right_dirs: Iterable[str | Path] | None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Pair both sides by position, compare every paper, and report the aggregate.

    The sides are called left and right rather than truth and candidate on
    purpose: this tool measures a difference, it does not decide which side is
    right. A report that lists differences is therefore still a passing run.
    """
    left = [str(item) for item in (left_dirs or [])]
    right = [str(item) for item in (right_dirs or [])]
    if not left or not right:
        raise ValueError("contract-crosscheck needs at least one --left and one --right directory")
    if len(left) != len(right):
        raise ValueError(
            "contract-crosscheck pairs directories by position, but was given "
            f"{len(left)} --left and {len(right)} --right"
        )
    report = crosscheck_contract_pairs(zip(left, right))
    if out:
        report_path = write_json(Path(out) / "contract_crosscheck.json", report)
        report["out_dir"] = str(report_path.parent)
        report["report"] = str(report_path)
    report["ok"] = True
    return report


__all__ = [
    "ENTITY_FIELDS",
    "compare_figure_specifications",
    "run_contract_crosscheck",
    "crosscheck_contract_directories",
    "crosscheck_contract_pairs",
]
