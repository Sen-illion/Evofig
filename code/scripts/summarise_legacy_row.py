#!/usr/bin/env python3
"""Summarise the 2026-10-04 rule-version batch as an experiment row.

The user decided to read the ablation's ``no experience library`` row out of the batch that was
already generated on 2026-10-04, instead of paying for a second 500-paper run. That batch is not
the product configuration: its own reports record ``judge_mode`` heuristic on both seats,
``candidates_per_round`` two, no experience library at all, and about four icon calls per paper
where the product configuration makes forty. It is a *rule-version* row, and whatever table it
appears in has to say so.

This script is arithmetic on artefacts that already exist: it reads each paper's ``metrics.json``
from the scored tree, drops the papers that also sit in the Judge's holdout lists so the row
counts the same papers as every other row, and reports the nine VLM items and the deterministic
editability numbers, with a paper-level paired bootstrap for the round-00 comparison.

    python scripts/summarise_legacy_row.py \
        --final <DATA_ROOT>/efb_score/own_final \
        --round00 <DATA_ROOT>/efb_score/own_round00 \
        --holdout <DATA_ROOT>/abl/holdout_overlap.txt \
        --out <DATA_ROOT>/abl/no_memory_legacy_row.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

NINE_ITEMS = (
    "accuracy",
    "completeness",
    "clarity",
    "flow",
    "legibility",
    "aesthetic",
    "expressiveness",
    "polish",
    "appropriateness",
)


def key_of(value: str) -> str:
    """One spelling for a paper id, so a DOI folder and a DOI list entry compare equal."""

    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")


def held_out(path: Path) -> set:
    """Every id named by the Judge holdout lists, or by the plain one-id-per-line overlap file."""

    ids: set = set()
    if not Path(path).is_file():
        return ids
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("{"):
            row = json.loads(text)
            for field in ("paper_id", "example_id"):
                if row.get(field):
                    ids.add(key_of(row[field]))
            continue
        ids.add(key_of(text.split()[0]))
    return ids


def read_rows(root: Path) -> dict:
    """paper key -> its nine-item scores and editability, for one scored tree."""

    rows: dict = {}
    for metrics in sorted(Path(root).glob("*/metrics.json")):
        try:
            body = json.loads(metrics.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        paper = key_of(body.get("paper_id") or metrics.parent.name)
        nine = body.get("vlm_nine") or {}
        editable = body.get("editability") or {}
        rows[paper] = {
            "nine": {item: nine.get(item) for item in NINE_ITEMS} if nine.get("available") else {},
            "nine_mean": nine.get("mean"),
            "editable": editable.get("editable"),
            "raster_area_ratio": editable.get("raster_area_ratio"),
            "native_object_count": editable.get("native_object_count"),
        }
    return rows


def mean(values: list) -> float | None:
    clean = [value for value in values if isinstance(value, (int, float))]
    return sum(clean) / len(clean) if clean else None


def bootstrap(values: list, seed: int, draws: int = 2000, lo: float = 2.5, hi: float = 97.5):
    """Paper-level percentile interval for the mean of a paired difference."""

    clean = [value for value in values if isinstance(value, (int, float))]
    if len(clean) < 2:
        return {"n": len(clean), "mean": mean(clean), "lo": None, "hi": None}
    rng = random.Random(seed)
    size = len(clean)
    means = []
    for _ in range(draws):
        means.append(sum(clean[rng.randrange(size)] for _ in range(size)) / size)
    means.sort()
    return {
        "n": size,
        "mean": sum(clean) / size,
        "lo": means[int(len(means) * lo / 100.0)],
        "hi": means[min(len(means) - 1, int(len(means) * hi / 100.0))],
    }


def summarise(rows: dict) -> dict:
    nine = {item: mean([row["nine"].get(item) for row in rows.values()]) for item in NINE_ITEMS}
    editable = [row["editable"] for row in rows.values() if isinstance(row["editable"], bool)]
    return {
        "papers": len(rows),
        "nine": nine,
        "nine_mean": mean([row["nine_mean"] for row in rows.values()]),
        "editability": {
            "scored": len(editable),
            "editable_rate": (sum(1 for value in editable if value) / len(editable)) if editable else None,
            "raster_area_ratio_mean": mean([row["raster_area_ratio"] for row in rows.values()]),
            "native_object_count_mean": mean([row["native_object_count"] for row in rows.values()]),
        },
    }


def paired(final: dict, round00: dict, seed: int) -> dict:
    """final minus round_00, per item, over the papers scored on both sides."""

    out: dict = {}
    for item in NINE_ITEMS:
        deltas = []
        for paper, row in final.items():
            other = round00.get(paper)
            if not other:
                continue
            before, after = other["nine"].get(item), row["nine"].get(item)
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                deltas.append(float(after) - float(before))
        out[item] = bootstrap(deltas, seed)
    return out


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise a rule-version batch as one row.")
    parser.add_argument("--final", default="<DATA_ROOT>/efb_score/own_final")
    parser.add_argument("--round00", default="<DATA_ROOT>/efb_score/own_round00")
    parser.add_argument("--holdout", default="<DATA_ROOT>/abl/holdout_overlap.txt")
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    blocked = held_out(Path(args.holdout))
    final_all = read_rows(Path(args.final))
    round_all = read_rows(Path(args.round00))
    final = {key: row for key, row in final_all.items() if key not in blocked}
    round00 = {key: row for key, row in round_all.items() if key not in blocked}

    report = {
        "row": "no_memory_legacy",
        "configuration": {
            "note": "the 2026-10-04 rule-version batch: both Judge seats heuristic (trained false), two candidates per round, no experience library, about four icon calls per paper",
            "final_tree": str(args.final),
            "round00_tree": str(args.round00),
            "holdout_list": str(args.holdout),
        },
        "excluded": {
            "holdout_named": len(blocked),
            "dropped_from_final": len(final_all) - len(final),
            "dropped_from_round00": len(round_all) - len(round00),
        },
        "final": summarise(final),
        "round00": summarise(round00),
        "delta_final_minus_round00": paired(final, round00, args.seed),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
