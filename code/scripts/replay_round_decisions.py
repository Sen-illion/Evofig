"""Re-judge a run's recorded round decisions under the rule the loop uses now.

The loop used to let the mechanism overrule a judgement. If the model said the new figure was
better but a deterministic check named one more defect, the round was refused anyway; if a
reflection said the round's own measurement was noise, an accept was withdrawn. Both are gone:
the comparison decides, the gate only reports, and a noise flag is recorded and settled later
rather than spending the round. This script reads what those rounds recorded -- the comparison's
verdict, whether coverage rose, how the gate scored -- and says what the current rule would have
decided, without a model call and without re-running a round.

It is a measuring instrument for the change rather than a way to edit history: nothing on disk is
written except the report, and a record the current rule cannot re-judge (its inputs are not in
the file) is reported as unresolved instead of guessed at.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPLAY_CONTRACT = "rfs.round_decision_replay"
REPLAY_VERSION = 1
#: The reasons a reflection's noise claim used to write when it withdrew an accept. A round that
#: carries one was refused for a claim about the measurement, not about the figure.
NOISE_WITHDRAWAL_REASONS = ("measurement_flagged_randomness",)
#: How much a score comparison had to move; the loop's own default, kept so a replay of the
#: score path answers the same question the loop did.
DEFAULT_TOLERANCE = 0.01


def rejudge_decision(record: dict[str, Any] | None, *, tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    """What the current rule decides about one recorded round, and why.

    The record is the round's own ``repair_decision.json``: the comparison it took, the verdict
    that comparison gave, whether contract coverage rose, and the score reading when there was
    one. Read in this order because it mirrors the loop's own order: a paired comparison decides
    first, a round that moved nothing is refused before either instrument is consulted, and only
    a round with no comparison at all falls back to the score.
    """

    source = record if isinstance(record, dict) else {}
    recorded_reason = str(source.get("reason") or "")
    recorded_accepted = bool(source.get("accepted"))
    verdict = str(source.get("pairwise_verdict") or "")
    coverage = bool(source.get("coverage_improved"))
    comparison = str(source.get("comparison") or "")
    withdrawn = recorded_reason in NOISE_WITHDRAWAL_REASONS

    answer: dict[str, Any] = {
        "recorded_reason": recorded_reason,
        "recorded_accepted": recorded_accepted,
        "comparison": comparison,
        "pairwise_verdict": verdict or None,
        "coverage_improved": coverage,
        "withdrawn_by_noise": withdrawn,
        "unresolved": False,
    }
    if comparison == "pairwise" or verdict:
        preferred = verdict == "after"
        # One ruler under the current rule, the same one the loop uses: the comparison decides and
        # the coverage reading is recorded beside the decision. Where the two disagree the row
        # says so, so the old rule's extra accepts are visible rather than silently dropped.
        accepted = bool(preferred)
        reason = "pairwise_preferred" if preferred else "pairwise_not_preferred"
        answer["coverage_disagreed"] = bool(coverage) != preferred
    elif comparison == "figure_unchanged" or recorded_reason == "figure_unchanged":
        # Nothing moved. That is a question about the measurement, so it refuses the round under
        # the new rule exactly as it did under the old one.
        accepted, reason = False, "figure_unchanged"
    elif comparison == "compared":
        delta = source.get("score_delta")
        try:
            moved = float(delta) if delta is not None else None
        except (TypeError, ValueError):
            moved = None
        accepted = moved is not None and moved > tolerance
        reason = "target_improved" if accepted else "no_significant_improvement"
    elif comparison == "none":
        # No earlier state to compare against: the first candidate of a run is a starting point,
        # and a round whose accept a noise flag withdrew goes back to that accept, because the
        # flag no longer spends the round.
        accepted, reason = True, "initial_candidate"
        answer["recovered_from_noise"] = withdrawn
    else:
        # Nothing in the record says which decision this was: refuse to invent one.
        accepted, reason, answer["unresolved"] = None, "unresolved_no_comparison", True
    answer["accepted"] = accepted
    answer["reason"] = reason
    answer["changed"] = accepted is not None and accepted != recorded_accepted
    return answer


def _iter_decisions(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob("**/repair_decision.json"))


def replay_decisions(paths: list[Path], *, tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    """Re-judge every decision under a run directory and report what moved."""

    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            rows.append({"path": str(path), "error": f"{type(error).__name__}: {error}", "unresolved": True})
            continue
        answer = rejudge_decision(record, tolerance=tolerance)
        answer["path"] = str(path)
        answer["iteration"] = path.parent.parent.name if path.parent.parent else ""
        answer["round"] = path.parent.name
        rows.append(answer)

    flipped_to_accept = [row for row in rows if row.get("changed") and row.get("accepted")]
    flipped_to_refuse = [row for row in rows if row.get("changed") and row.get("accepted") is False]
    refused_to_accept = [row for row in rows if not row.get("recorded_accepted") and row.get("accepted")]
    accepted_to_refuse = [row for row in rows if row.get("recorded_accepted") and row.get("accepted") is False]
    model_called_better_refused = [
        row for row in rows if row.get("pairwise_verdict") == "after" and not row.get("recorded_accepted")
    ]
    noise_killed = [row for row in rows if row.get("withdrawn_by_noise")]
    noise_killed_after = [row for row in noise_killed if row.get("pairwise_verdict") == "after"]
    worse_still_refused = [row for row in rows if row.get("pairwise_verdict") == "before" and not row.get("accepted")]

    def _brief(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "iteration": row.get("iteration"),
            "round": row.get("round"),
            "recorded_reason": row.get("recorded_reason"),
            "pairwise_verdict": row.get("pairwise_verdict"),
            "new_reason": row.get("reason"),
        }

    checks = [
        {
            "name": "every round the comparison called better is accepted",
            "ok": all(row.get("accepted") for row in rows if row.get("pairwise_verdict") == "after"),
            "detail": f"{sum(1 for row in rows if row.get('pairwise_verdict') == 'after')} round(s) the comparison called better",
        },
        {
            "name": "no round is decided by a reading",
            "ok": all(row.get("accepted") == (row.get("pairwise_verdict") == "after") for row in rows if row.get("pairwise_verdict")),
            "detail": f"{sum(1 for row in rows if row.get('coverage_disagreed'))} round(s) where coverage read the other way from the comparison",
        },
        {
            "name": "no round the comparison called worse is accepted without a coverage gain",
            "ok": not any(row.get("accepted") for row in rows if row.get("pairwise_verdict") == "before" and not row.get("coverage_improved")),
            "detail": f"{len(worse_still_refused)} round(s) the comparison called worse and that gained no coverage stay refused",
        },
        {
            "name": "every record carries enough to be re-judged",
            "ok": not any(row.get("unresolved") for row in rows),
            "detail": f"{sum(1 for row in rows if row.get('unresolved'))} record(s) could not be re-judged",
        },
    ]
    return {
        "contract": REPLAY_CONTRACT,
        "version": REPLAY_VERSION,
        "decision_count": len(rows),
        "recorded_accepted_count": sum(1 for row in rows if row.get("recorded_accepted")),
        "rejudged_accepted_count": sum(1 for row in rows if row.get("accepted")),
        "flipped_to_accept_count": len(flipped_to_accept),
        "flipped_to_refuse_count": len(flipped_to_refuse),
        "refused_to_accept_count": len(refused_to_accept),
        "accepted_to_refuse_count": len(accepted_to_refuse),
        "model_called_better_refused_count": len(model_called_better_refused),
        "noise_killed_count": len(noise_killed),
        "noise_killed_after_count": len(noise_killed_after),
        "model_called_better_refused": [_brief(row) for row in model_called_better_refused],
        "noise_killed": [_brief(row) for row in noise_killed],
        "noise_killed_after": [_brief(row) for row in noise_killed_after],
        "flipped_to_accept": [_brief(row) for row in flipped_to_accept],
        "flipped_to_refuse": [_brief(row) for row in flipped_to_refuse],
        "refused_to_accept": [_brief(row) for row in refused_to_accept],
        "checks": checks,
        "ok": all(check["ok"] for check in checks),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-judge a run's recorded round decisions under the current rule.")
    parser.add_argument("--decisions", required=True, help="A run directory (recursively searched) or a single repair_decision.json.")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="How much a score comparison had to move.")
    parser.add_argument("--out", default=None, help="Where to write the JSON report.")
    parser.add_argument("--json", action="store_true", help="Print the whole report instead of a summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.decisions)
    paths = _iter_decisions(root)
    if not paths:
        print(f"no repair_decision.json under {root}", file=sys.stderr)
        return 2
    report = replay_decisions(paths, tolerance=float(args.tolerance))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps({key: value for key, value in report.items() if key != "checks"}, ensure_ascii=False, indent=2, sort_keys=True))
        for check in report["checks"]:
            print(("PASS " if check["ok"] else "FAIL ") + check["name"] + " -- " + check["detail"])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
