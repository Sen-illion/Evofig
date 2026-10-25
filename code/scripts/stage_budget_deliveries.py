#!/usr/bin/env python3
"""Lay one budget arm's deliveries out the way the own-set scorer expects to read them.

The driver writes each paper under `<run_root>/out/<arm>/<domain>/<paper_id>/`, but the
scorer looks for `<runs_root>/<paper_id>/delivery.pptx`. This copies nothing large: it
links the arm's `editable_composition.pptx` and `preview.png` into a staging tree that
carries the scorer's own filenames, leaving the run directory untouched so a re-run and
its costs stay attributable to the arm that paid for them.

    python scripts/stage_budget_deliveries.py \
        --arm-dir <DATA_ROOT>/abl/out/topk_5 \
        --staging <DATA_ROOT>/abl/staging/topk_5 \
        --manifest <DATA_ROOT>/abl/de25.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rfs.evaluation.own_set_score import safe_paper_id  # noqa: E402 - path first
from rfs.utils import write_json  # noqa: E402 - path first

#: What the arm leaves in its paper directory, and what the readers want.
#:
#: The scorer reads the two delivery names; the summariser reads the other two -- it looks for
#: the cost ledger beside the metrics it just wrote, so a staging folder that dropped it would
#: report every arm as free.
LINKED = (
    ("editable_composition.pptx", "delivery.pptx"),
    ("preview.png", "preview.png"),
    ("cost_report.json", "cost_report.json"),
    ("run.json", "run.json"),
)


def find_paper_dir(arm_dir: Path, safe: str) -> Path | None:
    """The paper's directory under whichever domain folder holds it.

    The scoring side names a paper's folder with ``safe_paper_id``, which cuts the name at 120
    characters, while the driver names the same paper with its own ``safe``, which does not. On
    the one long id in this list the two disagree by the last few words, and the paper then looks
    missing even though the arm delivered it -- a silent hole in the table, since the arm's own
    report counts it. An exact match is tried first and a prefix of exactly one folder second, so
    an id that merely starts like another paper's still resolves to nothing rather than the wrong
    figure.
    """

    direct = arm_dir / safe
    if direct.is_dir():
        return direct
    exact = [c for c in sorted(arm_dir.glob("*/%s" % safe)) if c.is_dir()]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    longer = [c for c in sorted(arm_dir.glob("*/%s*" % safe)) if c.is_dir()]
    return longer[0] if len(longer) == 1 else None


def stage_one(arm_dir: Path, staging: Path, safe: str) -> str:
    """Link one paper in. Returns 'staged', 'no_pptx' or 'not_found'."""

    paper_dir = find_paper_dir(arm_dir, safe)
    if paper_dir is None:
        return "not_found"
    source = paper_dir / LINKED[0][0]
    if not source.is_file():
        return "no_pptx"
    target = staging / safe
    target.mkdir(parents=True, exist_ok=True)
    for from_name, to_name in LINKED:
        origin = paper_dir / from_name
        dest = target / to_name
        if not origin.is_file():
            continue
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        os.symlink(origin, dest)
    return "staged"


def stage_candidates(arm_dir: Path, staging: Path, safe: str) -> dict[str, str]:
    """Link one paper's every candidate in, one tree per round and candidate.

    A rung pays for N candidates and delivers one of them. The other N-1 are what the spend
    actually bought, and they are only visible in the round's candidate directories -- which the
    scorer cannot be pointed at, because it reads exactly one delivery per paper id. So a
    candidate that never won gets a tree of its own, named after the round and the candidate it
    holds, and every tree carries the paper under its own id. One manifest then scores the whole
    matrix, and the winner's own record (``rounds[0].selected_variant``) says which column of it
    the delivered deck is.

    Only the deck is linked: the scorer renders the preview itself, so every candidate is read
    through one renderer instead of through whichever preview a losing candidate left behind.
    """

    paper_dir = find_paper_dir(arm_dir, safe)
    if paper_dir is None:
        return {"": "not_found"}
    rounds = sorted(item for item in paper_dir.glob("round_*") if item.is_dir())
    if not rounds:
        return {"": "no_candidate"}
    outcomes: dict[str, str] = {}
    for round_dir in rounds:
        candidates = sorted(item for item in round_dir.glob("candidate_*") if item.is_dir())
        for candidate in candidates:
            tree = "%s_%s" % (round_dir.name, candidate.name)
            source = candidate / LINKED[0][0]
            if not source.is_file():
                outcomes[tree] = "no_pptx"
                continue
            target = staging / tree / safe
            target.mkdir(parents=True, exist_ok=True)
            dest = target / LINKED[0][1]
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            os.symlink(source, dest)
            outcomes[tree] = "staged"
    return outcomes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm-dir", required=True, help="e.g. <run_root>/out/topk_5")
    ap.add_argument("--staging", required=True, help="Where the scorer should look.")
    ap.add_argument("--manifest", required=True, help="The paper list the arm ran on.")
    ap.add_argument(
        "--candidates",
        action="store_true",
        help="Stage every candidate of every round, one tree per candidate, instead of the delivered deck.",
    )
    ap.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = ap.parse_args()

    arm_dir = Path(args.arm_dir)
    staging = Path(args.staging)
    manifest = Path(args.manifest)
    if not arm_dir.is_dir():
        raise SystemExit("no such arm directory: %s" % arm_dir)

    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = {"staged": 0, "no_pptx": 0, "not_found": 0, "no_candidate": 0}
    trees: dict[str, dict[str, int]] = {}
    missing: list[dict[str, str]] = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            continue
        safe = safe_paper_id(paper_id)
        outcomes = stage_candidates(arm_dir, staging, safe) if args.candidates else {"": stage_one(arm_dir, staging, safe)}
        for tree, outcome in outcomes.items():
            counts[outcome] = counts.get(outcome, 0) + 1
            bucket = trees.setdefault(tree, {})
            bucket[outcome] = bucket.get(outcome, 0) + 1
            if outcome != "staged":
                missing.append({"paper_id": paper_id, "tree": tree, "reason": outcome})

    report = {
        "summary": "Symlink one arm's deliveries into the own-set scorer's layout.",
        "arm_dir": str(arm_dir),
        "staging": str(staging),
        "manifest": str(manifest),
        "considered": len(rows),
        "candidates": bool(args.candidates),
        "trees": {name: dict(sorted(bucket.items())) for name, bucket in sorted(trees.items())},
        "missing": missing,
        **counts,
    }
    write_json(staging / "_staging_report.json", report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            "staged=%d no_pptx=%d not_found=%d of %d papers in %d trees"
            % (counts["staged"], counts["no_pptx"], counts["not_found"], len(rows), len(trees))
        )
        for item in missing[:20]:
            print("  missing: %s (%s)" % (item["paper_id"], item["reason"]))
    return 0 if counts["not_found"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
