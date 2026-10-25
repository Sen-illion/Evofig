"""Put a list of real figure pairs to the acceptance seat and score its answers.

Two exams share this instrument, and neither one spends a round of the live loop:

* the pair question: whether the seat still reads a comparison correctly now that the question
  carries the whole-first rule and the loop's own readings, and
* the direction question: whether what it says about the round's layer matches what a person said
  about the same rounds.

The pairs are handed in as a file so the exam is about figures rather than about a run's layout:

    {"stages": ["content_planning", "segmentation"],
     "working_on": "segmentation",
     "pairs": [{"a": "A.png", "b": "B.png", "expect": "A", "label": "132 vs 133 r00"}]}

``expect`` is the figure a person judged worse; leave it out for a pair that is only being asked
about its layer. The report keeps every answer, so a wrong pair can be read rather than only
counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rfs.evaluation.direct_judges import ChoiceAcceptanceJudge, DirectJudgeInput, DirectPairJudgeInput

EXAM_CONTRACT = "rfs.acceptance_direction_exam"
EXAM_VERSION = 1


def _input(path: str, layer_plan: dict[str, Any] | None) -> DirectJudgeInput:
    return DirectJudgeInput(
        method={},
        layout={},
        icons={},
        program={},
        preview=Path(path),
        layer_plan=layer_plan,
    )


def run_exam(*, checkpoint: str, base_model: str, pairs: list[dict[str, Any]], working_on: str, stages: list[str], direction: str) -> dict[str, Any]:
    """Ask the seat about every pair and report how it answered."""

    plan = {"working_on": working_on, "layer_order": list(stages)} if working_on else None
    judge = ChoiceAcceptanceJudge(checkpoint=checkpoint, base_model=base_model, direction=direction)
    rows: list[dict[str, Any]] = []
    for item in pairs:
        result = judge.evaluate_pairwise(
            DirectPairJudgeInput(before=_input(str(item["a"]), plan), after=_input(str(item["b"]), plan))
        )
        rows.append(
            {
                "label": str(item.get("label") or ""),
                "expect": str(item.get("expect") or ""),
                "wrong_figure": str(result.get("wrong_figure") or ""),
                "defect": str(result.get("defect") or ""),
                "preferred": str(result.get("preferred_candidate") or ""),
                "direction": dict(result.get("direction") or {}),
                "direction_problem": str(result.get("direction_problem") or ""),
                "unavailable": result.get("pairwise_unavailable"),
            }
        )
    right = [row for row in rows if row["expect"] and row["wrong_figure"] == row["expect"]]
    asked = [row for row in rows if row["expect"]]
    legal = [row for row in rows if row["wrong_figure"] in {"A", "B"}]
    answered = [row for row in rows if row["direction"]]
    stated = [row for row in answered if str(row["direction"].get("settled")) in {"True", "False"}]
    return {
        "contract": EXAM_CONTRACT,
        "version": EXAM_VERSION,
        "checkpoint": checkpoint,
        "direction": direction,
        "pairs": len(rows),
        "read_the_pair_right": len(right),
        "scored_pairs": len(asked),
        "pair_accuracy": round(len(right) / len(asked), 4) if asked else None,
        "legal": len(legal),
        "legal_rate": round(len(legal) / len(rows), 4) if rows else None,
        "direction_answered": len(answered),
        "direction_settled_stated": len(stated),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the acceptance seat on a list of real figure pairs.")
    parser.add_argument("--pairs", required=True, help="The JSON file naming the pairs.")
    parser.add_argument("--checkpoint", default="", help="The local choice checkpoint to seat.")
    parser.add_argument("--base-model", default="", help="The base model the checkpoint was trained from.")
    parser.add_argument("--direction", default="record", choices=("off", "record", "authority"), help="How the layer question is put.")
    parser.add_argument("--out", default=None, help="Where to write the JSON report.")
    parser.add_argument("--json", action="store_true", help="Print the whole report instead of a summary.")
    args = parser.parse_args(argv)
    body = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    report = run_exam(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        pairs=list(body.get("pairs") or []),
        working_on=str(body.get("working_on") or ""),
        stages=[str(item) for item in (body.get("stages") or [])],
        direction=args.direction,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"pairs={report['pairs']} pair_accuracy={report['pair_accuracy']} legal_rate={report['legal_rate']} direction_answered={report['direction_answered']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())