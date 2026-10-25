#!/usr/bin/env python3
"""Freeze the papers a case study is built from, before any figure is looked at.

A case study that picks its examples after seeing the outputs is not evidence, so this
script chooses them by a rule that reads only numbers a run wrote: contract size (modules,
relations), the editable-object audit (objects, connectors), the deterministic blockers the
quality gate recorded, and the per-stage cost. It never opens a preview, a PPTX, or a paper.

The run tree is the one the ablation driver writes, ``<run-root>/<domain>/<paper_id>/``.
Two files come out:

* ``candidate_table.json`` -- one row per eligible paper: the pool the picks came from, so a
  rule that says "the pool's P75" can be checked against the pool it meant;
* ``case_list.json`` -- the picks, each naming the rule that selected it, with the sha256 of
  the files the later steps read.

The picks are a proposal. Freezing them means committing the file; if a slot is later
replaced, keep the original list and record why (``docs/case-study-plan.md``, section 4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DOMAINS = ("agent", "biology", "cv", "materials", "rag")
PICKED_FILES = ("editable_composition.pptx", "preview.png", "figure_program.json")


def load_json(path: Path):
    """Read JSON that may sit behind log lines a library wrote to the same file."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            return json.loads(text[index:])
        except json.JSONDecodeError:
            continue
    return None


def dump_json(path: Path, payload) -> None:
    """Write UTF-8 JSON the way the rest of the repo writes it."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
    try:
        from rfs.utils import write_text

        write_text(path, text)
    except Exception:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction: float):
    """Linear-interpolated percentile; None when there is nothing to measure."""
    ordered = sorted(value for value in values if isinstance(value, (int, float)))
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def row_for(run_dir: Path, domain: str) -> dict:
    report = load_json(run_dir / "paper_evolve_editable_report.json") or {}
    gate = load_json(run_dir / "final_quality_gate.json") or {}
    graph = load_json(run_dir / "editable_graph_report.json") or {}
    run_report = load_json(run_dir / "run_report.json") or {}
    cost = load_json(run_dir / "cost_report.json") or {}
    rounds = load_json(run_dir / "evolution_rounds.json") or {}
    first_round = (rounds.get("rounds") or [{}])[0] if isinstance(rounds, dict) else {}
    judgement = first_round.get("online_judgement") or load_json(run_dir / "judgement.json") or {}

    blocking = list(judgement.get("blocking_issues") or [])
    generation_errors = list(gate.get("blocking_generation_errors") or [])
    graph_errors = list(graph.get("errors") or [])
    row = {
        "domain": domain,
        "paper_id": run_dir.name,
        "run_dir": str(run_dir),
        "has_pptx": (run_dir / "editable_composition.pptx").is_file(),
        "has_preview": (run_dir / "preview.png").is_file(),
        "generation_status": report.get("generation_status"),
        "quality_status": report.get("quality_status"),
        "quality_passed": report.get("quality_passed"),
        "delivery_status": report.get("delivery_status"),
        "repair_recommended": report.get("repair_recommended"),
        "module_count": run_report.get("module_count"),
        "relation_count": run_report.get("relation_count"),
        "object_count": graph.get("object_count"),
        "connector_count": graph.get("connector_count"),
        "judge_overall_score": judgement.get("overall_score"),
        "judge_status": judgement.get("status"),
        "judge_gate_only": judgement.get("gate_only"),
        "blocking_issue_count": len(blocking),
        "generation_error_count": len(generation_errors),
        "graph_error_count": len(graph_errors),
        "elapsed_seconds": cost.get("elapsed_seconds"),
    }
    row["blocker_total"] = (
        row["blocking_issue_count"] + row["generation_error_count"] + row["graph_error_count"]
    )
    row["eligible"] = bool(
        row["has_pptx"] and row["has_preview"] and row["generation_status"] == "completed"
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="a run tree, <run-root>/<domain>/<paper_id>/")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument(
        "--only-papers",
        help="a text file of paper ids; every other paper is dropped from the pool",
    )
    parser.add_argument(
        "--failure-rule",
        default="most_blockers",
        choices=("most_blockers", "lowest_score", "most_graph_errors"),
        help="how the one failure slot (A5) is chosen; the rule is written into the list",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    domains = [name.strip() for name in args.domains.split(",") if name.strip()]

    rows = []
    for domain in domains:
        domain_dir = run_root / domain
        if not domain_dir.is_dir():
            continue
        for run_dir in sorted(entry for entry in domain_dir.iterdir() if entry.is_dir()):
            rows.append(row_for(run_dir, domain))
    only = None
    if args.only_papers:
        only = {
            line.strip()
            for line in Path(args.only_papers).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
    eligible = [
        row for row in rows if row["eligible"] and (only is None or row["paper_id"] in only)
    ]
    if only is not None and not eligible:
        parser.error("--only-papers left no eligible paper")

    def numeric(rows_, field):
        return [row[field] for row in rows_ if isinstance(row[field], (int, float))]

    picks = []

    def closest(rows_, field, target):
        candidates = [row for row in rows_ if isinstance(row[field], (int, float))]
        if not candidates or target is None:
            return None
        return min(candidates, key=lambda row: (abs(row[field] - target), row["paper_id"]))

    def add(slot, case, rule, row, note):
        if row is None:
            return
        run_dir = Path(row["run_dir"])
        picks.append(
            {
                "slot": slot,
                "case": case,
                "rule": rule,
                "note": note,
                "domain": row["domain"],
                "paper_id": row["paper_id"],
                "run_dir": row["run_dir"],
                "measured": {
                    key: row[key]
                    for key in (
                        "module_count",
                        "relation_count",
                        "object_count",
                        "connector_count",
                        "quality_passed",
                        "blocker_total",
                        "judge_overall_score",
                    )
                },
                "sha256": {
                    name: sha256_file(run_dir / name)
                    for name in PICKED_FILES
                    if (run_dir / name).is_file()
                },
            }
        )

    # A1/A2: the hard, structure-heavy cases. Agent and retrieval papers describe pipelines,
    # vision papers pack the densest relations. The rule takes the pool's upper quartile of
    # relation count -- hard, but not the single densest paper in it.
    pipeline_pool = [row for row in eligible if row["domain"] in ("agent", "rag")]
    add(
        "A1",
        "cross-system comparison, multi-stage pipeline",
        "agent or rag, relation_count at that pool's P75",
        closest(
            pipeline_pool,
            "relation_count",
            percentile(numeric(pipeline_pool, "relation_count"), 0.75),
        ),
        "multi-stage procedural figure; hard case, not the hardest in the pool",
    )
    vision_pool = [row for row in eligible if row["domain"] == "cv"]
    add(
        "A2",
        "cross-system comparison, dense relations",
        "cv, relation_count at that pool's P75",
        closest(
            vision_pool,
            "relation_count",
            percentile(numeric(vision_pool, "relation_count"), 0.75),
        ),
        "relation-dense figure; same rule as A1 so both main-text cases are chosen the same way",
    )

    # A3: the editability audit should show a representative file, not the richest one.
    add(
        "A3",
        "native editability and object granularity",
        "all domains, object_count closest to the pool median",
        closest(eligible, "object_count", percentile(numeric(eligible, "object_count"), 0.5)),
        "representative object count; the audit row is the claim, not this paper's size",
    )

    # A4: one paper per non-AI domain at that domain's own median complexity.
    for domain in ("materials", "biology"):
        pool = [row for row in eligible if row["domain"] == domain]
        add(
            "A4." + domain,
            "cross-area generalisation, " + domain,
            domain + ", relation_count closest to that domain's median",
            closest(pool, "relation_count", percentile(numeric(pool, "relation_count"), 0.5)),
            "median complexity in the domain, so the case is what a typical paper there looks like",
        )

    # A5: the failure slot, chosen by one stated rule over papers that still delivered a figure.
    # A flag every paper carries is not a signal: the gate writes quality_passed=false for every
    # paper in a row that never repairs, so this slot reads blocker counts instead.
    if args.failure_rule == "most_blockers":
        pool = [row for row in eligible if row["blocker_total"] > 0] or eligible
        failure = min(pool, key=lambda row: (-row["blocker_total"], row["paper_id"])) if pool else None
        rule = "eligible, most recorded blockers (gate + generation + graph)"
    elif args.failure_rule == "lowest_score":
        pool = [row for row in eligible if isinstance(row["judge_overall_score"], (int, float))]
        failure = min(pool, key=lambda row: (row["judge_overall_score"], row["paper_id"])) if pool else None
        rule = "eligible, measured judge score, lowest first"
    else:
        failure = (
            min(eligible, key=lambda row: (-row["graph_error_count"], row["paper_id"]))
            if eligible
            else None
        )
        rule = "eligible, most editable-graph errors"
    add(
        "A5",
        "failure case, attributed to a stage",
        rule,
        failure,
        "a delivered figure that the run itself flagged",
    )

    out_dir = Path(args.out_dir)
    counts = {
        "papers": len(rows),
        "eligible": len(eligible),
        "quality_passed": sum(1 for row in eligible if row["quality_passed"]),
        "measured_judge_score": sum(
            1 for row in eligible if isinstance(row["judge_overall_score"], (int, float))
        ),
    }
    warnings = []
    for field in ("quality_passed", "generation_status", "delivery_status"):
        values = {json.dumps(row[field]) for row in rows}
        if len(values) == 1:
            warnings.append(
                field + " is constant across this tree (" + values.pop() + "): it cannot separate papers"
            )
    if rows and not any(isinstance(row["judge_overall_score"], (int, float)) for row in rows):
        warnings.append(
            "no paper here has a measured judge score; anything judge-based needs another run"
        )
    dump_json(
        out_dir / "candidate_table.json",
        {
            "schema": "rfs.case-study.candidate-table",
            "version": 1,
            "run_root": str(run_root),
            "domains": domains,
            "only_papers": args.only_papers,
            "counts": counts,
            "rows": eligible,
        },
    )
    dump_json(
        out_dir / "case_list.json",
        {
            "schema": "rfs.case-study.case-list",
            "version": 1,
            "status": "proposed",
            "run_root": str(run_root),
            "only_papers": args.only_papers,
            "failure_rule": args.failure_rule,
            "warnings": warnings,
            "picks": picks,
        },
    )

    print(
        json.dumps(
            {
                "counts": counts,
                "warnings": warnings,
                "picks": [
                    {
                        "slot": pick["slot"],
                        "domain": pick["domain"],
                        "paper_id": pick["paper_id"],
                        "rule": pick["rule"],
                    }
                    for pick in picks
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("wrote " + str(out_dir / "candidate_table.json") + " and " + str(out_dir / "case_list.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
