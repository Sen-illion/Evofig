#!/usr/bin/env python3
"""Fold a library's restated lessons into one row each, and keep every folded row as an archive.

The store folds a restatement at the moment it is appended, so the library the loop writes today
holds one row per lesson. What it cannot undo is the rows that were appended before folding existed:
the same sentence written out again with a new condition, one line per round. This script reads the
library, groups the rows that are the same lesson by the store's own key, keeps one row per group and
writes the rest to an archive file beside the library. Nothing is scored, nothing is measured and no
model is called: this is bookkeeping over rows that already exist.

Read-only unless ``--apply`` is passed. Without it the report says exactly what would move and where
to, which is the form to look at first.

    python scripts/fold_duplicate_experiences.py --library /path/to/selfevo_live/library
    python scripts/fold_duplicate_experiences.py --library ... --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rfs.evolution.direct.store import merge_rule_key, merge_rung_class  # noqa: E402 - path first
from rfs.utils import write_json  # noqa: E402 - path first


LIBRARY_FILE = "judge_experiences.jsonl"
MANIFEST_FILE = "manifest.json"
REPORT_CONTRACT = "rfs.experience_fold_report"
REPORT_VERSION = 1

#: The statuses that mean the ladder decided nothing about a row -- the measurement could not be
#: taken, or was taken and came out even. The live merge keeps them apart from every other rung,
#: because folding a row into one with a position would hand it a position it never earned. This
#: one-off fold is allowed to fold them together, and does so on one condition: every folded row is
#: written to the archive verbatim, so the record of "this was measured and came out even" survives
#: in a file rather than in forty lines of the library the loop reads from.
UNDECIDED_STATUSES = ("measurement_noise", "unverifiable", "inconclusive", "retrying")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _support(row: dict[str, Any]) -> int:
    try:
        return max(1, int(row.get("support") or 1))
    except (TypeError, ValueError):
        return 1


def _fold_key(row: dict[str, Any]) -> str:
    """The key two rows have to share to be one lesson, or ``""`` when the row is a lesson of its
    own. The rung class is part of it for the reason ``MERGE_RUNG_CLASSES`` gives: a validated
    lesson and a candidate saying the same thing are not the same row, and folding them would hand
    the newer measurement a rung it never earned.
    """

    if bool(row.get("carried_new_defect")):
        return ""
    rung = merge_rung_class(row.get("status"))
    if not rung and str(row.get("status") or "") in UNDECIDED_STATUSES:
        rung = "undecided"
    key = merge_rule_key(row)
    return f"{rung}\u0000{key}" if rung and key else ""


def _survivor(group: list[dict[str, Any]]) -> dict[str, Any]:
    """The row the others fold into: the one with the most support behind it, and on a tie the one
    that is already first, so a fold never changes which row came out of the ladder first."""

    best = group[0]
    for row in group[1:]:
        if _support(row) > _support(best):
            best = row
    return best


def fold(library: Path, *, apply: bool) -> dict[str, Any]:
    path = library / LIBRARY_FILE if library.is_dir() else library
    rows = _read_rows(path)
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    kept: list[dict[str, Any]] = []
    folded: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        key = _fold_key(row)
        if not key:
            kept.append(row)
            continue
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    for key in order:
        group = groups[key]
        if len(group) == 1:
            kept.append(group[0])
            continue
        survivor = _survivor(group)
        merged = dict(survivor)
        merged["support"] = sum(_support(row) for row in group)
        merged["folded_rows"] = len(group) - 1
        merged["folded_experience_ids"] = [str(row.get("experience_id") or "") for row in group if row is not survivor]
        kept.append(merged)
        for row in group:
            if row is not survivor:
                folded.append({"folded_into": str(survivor.get("experience_id") or ""), "row": row})
    report = {
        "contract": REPORT_CONTRACT,
        "version": REPORT_VERSION,
        "library": str(path),
        "rows_before": len(rows),
        "rows_after": len(kept),
        "folded": len(folded),
        "groups_folded": sum(1 for key in order if len(groups[key]) > 1),
        "applied": bool(apply),
    }
    if not apply:
        return report
    if not path.is_file():
        return {**report, "applied": False, "reason": "there is no library file to fold"}
    archive = path.with_name(f"{path.stem}.folded_{date.today().strftime('%Y%m%d')}.jsonl")
    if not archive.exists():
        shutil.copy2(path, archive)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with path.with_name(f"{path.stem}.folded.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in folded:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = path.with_name(MANIFEST_FILE)
    if manifest.is_file():
        try:
            current = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
        if isinstance(current, dict):
            write_json(manifest, {**current, "records": len(kept)})
    return {**report, "archive": str(archive), "folded_rows": str(path.with_name(f"{path.stem}.folded.jsonl"))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fold a library's restated lessons into one row each.")
    parser.add_argument("--library", required=True, help="the library directory, or the jsonl file itself")
    parser.add_argument("--apply", action="store_true", help="write the folded library; without it nothing is changed")
    args = parser.parse_args(argv)
    report = fold(Path(args.library).expanduser(), apply=bool(args.apply))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
