#!/usr/bin/env python3
"""Read a continuous run's own numbers off disk: how many rounds each layer cost, and the trend.

The loop writes two records per iteration -- the walk down the layers beside the iteration, and one
line of the same numbers in the run's trend ledger -- but no run prints them together, so "how many
rounds did this paper need" could only be answered by opening every iteration by hand. This script
opens them once and prints the answer. It generates nothing, calls no model and writes nothing but
its own report, so it costs nothing to run against a chain that is still moving.

The two numbers the direction change is judged on are per layer and per paper:

  * ``rounds_to_settle`` for a layer is the iteration that settled it minus the iteration that
    first moved it, plus one, and it is absent for a layer nobody has settled;
  * ``rounds_to_settle_all`` is the first iteration at which every layer stood settled at once, and
    it is absent until that has happened -- reopening a layer pushes it out by construction.

The trend comes from the ledger the loop appends to, one line per iteration, and is the share of
cross-round comparisons in which the current version beat the paper's own recent best.

    python scripts/summarize_evolution_metrics.py --out <PATH>

The per-layer table reads the newest layer ledger the run wrote, because that file carries both the
walk and the answers behind it and nothing rewrites it afterwards -- so this script and the round
that wrote it always agree, which recomputing off the live state would not. A paper that has not
finished an iteration yet falls back to its state. Each iteration's ``evolution_metrics.json`` is
read too, so a mismatch between what the iterations recorded and what the state now says is visible
rather than averaged away. Nothing is written anywhere unless ``--report`` names a path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rfs.contracts.evolution import STAGE_ORDER  # noqa: E402 - path first
from rfs.evolution.direct.metrics import layer_walk  # noqa: E402 - path first


STATE_FILE = "continuous_state.json"
LEDGER_FILE = "evolution_metrics.jsonl"
ROUND_METRICS_FILE = "evolution_metrics.json"
LAYER_LEDGER_FILE = "layer_ledger.json"
SUMMARY_CONTRACT = "rfs.evolution_metrics_summary"
SUMMARY_VERSION = 1


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def iteration_number(name: str) -> int | None:
    """The iteration a run directory's name carries, or ``None`` for anything else.

    The number is parsed rather than sorted as text, because ``iter_100`` sorts before ``iter_99``
    and a "newest" that is picked by string order is wrong on exactly the runs long enough to
    matter.
    """

    text = str(name or "")
    if not text.startswith("iter_"):
        return None
    try:
        return int(text[len("iter_") :])
    except ValueError:
        return None


def iteration_metrics(paper_dir: Path) -> list[dict[str, Any]]:
    """Every iteration's own metrics file, in iteration order, as the loop left them."""

    found: list[dict[str, Any]] = []
    for path in sorted(paper_dir.glob("iter_*"), key=lambda item: iteration_number(item.name) or 0):
        payload = _read_json(path / ROUND_METRICS_FILE)
        if isinstance(payload, dict) and payload:
            found.append({"iteration": iteration_number(path.name) or 0, **payload})
    return found


def newest_layer_ledger(paper_dir: Path) -> dict[str, Any]:
    """The last layer ledger this paper wrote, or an empty record if it has written none.

    The ledger is written once per iteration and never rewritten, so the newest one is the answer
    the round itself gave, and a reader asking "how many rounds did this layer cost" gets the same
    number whether it asks now or a month from now.
    """

    newest: dict[str, Any] = {}
    for path in sorted(paper_dir.glob("iter_*"), key=lambda item: iteration_number(item.name) or 0):
        payload = _read_json(path / LAYER_LEDGER_FILE)
        if isinstance(payload, dict) and payload:
            newest = payload
    return newest


def trend(rows: list[dict[str, Any]], *, paper: str = "") -> dict[str, Any]:
    """How this paper's versions stood against its own recent best, round by round.

    A comparison with no verdict is counted as undecided rather than as a tie: "nobody compared"
    and "the two were equal" are different facts, and averaging them would flatter the trend.
    """

    # Only rows the loop wrote carry a progress block. A run's own metrics line -- which lands in a
    # ledger of the same name when the library sits inside the run root -- is not a round of the
    # loop and would otherwise be counted as one nobody compared.
    mine = [
        row
        for row in rows
        if (not paper or str(row.get("paper") or "") == paper) and isinstance(row.get("progress"), dict)
    ]
    wins = losses = ties = undecided = 0
    streak = longest = 0
    for row in mine:
        progress = row.get("progress") if isinstance(row.get("progress"), dict) else {}
        counts = progress.get("cross_round") if isinstance(progress.get("cross_round"), dict) else {}
        wins += int(counts.get("wins") or 0)
        losses += int(counts.get("losses") or 0)
        ties += int(counts.get("ties") or 0)
        undecided += int(counts.get("undecided") or 0)
        if int(counts.get("wins") or 0) and not int(counts.get("losses") or 0):
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    decided = wins + losses + ties
    return {
        "rounds": len(mine),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "undecided": undecided,
        "win_rate": (round(wins / decided, 4) if decided else None),
        "longest_win_streak": longest,
    }


def summarize(out: Path) -> dict[str, Any]:
    state = _read_json(out / STATE_FILE)
    state = state if isinstance(state, dict) else {}
    ledger = _read_jsonl(out / LEDGER_FILE)
    papers: list[dict[str, Any]] = []
    for name, entry in sorted((state.get("papers") or {}).items()):
        if not isinstance(entry, dict):
            continue
        paper_dir = out / "papers" / str(name)
        ledger_record = newest_layer_ledger(paper_dir)
        walk = ledger_record.get("walk") if isinstance(ledger_record.get("walk"), dict) else None
        if not walk:
            walk = layer_walk(
                entry.get("layers"),
                stage_order=STAGE_ORDER,
                settle_events=entry.get("layer_settle_events") or (),
                reopen_events=ledger_record.get("reopen_events") or (),
            )
        repeats = [str(item.get("work_face") or "") for item in entry.get("history") or [] if isinstance(item, dict)]
        papers.append(
            {
                "paper": str(name),
                "iteration": int(entry.get("iteration") or 0),
                "work_face": str(entry.get("work_face") or ""),
                "layer_walk": walk,
                "layer_ledger": str(paper_dir) if ledger_record else "",
                "upstream": {
                    "requested": int(entry.get("upstream_requests") or 0),
                    "passed": int(entry.get("upstream_passed") or 0),
                },
                "claims": dict(entry.get("last_claims") or {}),
                "trend": trend(ledger, paper=str(name)),
                "iterations_recorded": len(iteration_metrics(paper_dir)),
                "work_faces_seen": sorted({value for value in repeats if value}),
            }
        )
    report = {
        "contract": SUMMARY_CONTRACT,
        "version": SUMMARY_VERSION,
        "out": str(out),
        "state": str(out / STATE_FILE),
        "ledger": str(out / LEDGER_FILE),
        "ledger_rows": len(ledger),
        "papers": papers,
        "trend": trend(ledger),
    }
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# How far the papers have walked",
        "",
        f"- Run: `{report['out']}`",
        f"- Trend ledger: {report['ledger_rows']} iteration(s) recorded",
        "",
        "| Paper | Iteration | Work face | Layers settled | Rounds to settle all | Reopens | Upstream asked / allowed | Win / loss / tie |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for paper in report["papers"]:
        walk = paper["layer_walk"]
        upstream = paper["upstream"]
        item = paper["trend"]
        lines.append(
            f"| {paper['paper']} | {paper['iteration']} | {paper['work_face'] or '-'} | "
            f"{walk['settled']}/{walk['total']} | {walk['rounds_to_settle_all'] if walk['rounds_to_settle_all'] is not None else '-'} | "
            f"{walk['reopen_events']} | {upstream['requested']} / {upstream['passed']} | "
            f"{item['wins']} / {item['losses']} / {item['ties']} |"
        )
    for paper in report["papers"]:
        lines.append("")
        lines.append(f"## {paper['paper']}")
        for stage, row in (paper["layer_walk"]["layers"] or {}).items():
            lines.append(
                f"- {stage}: first moved at {row['first_move_at'] or '-'}, settled at {row['settled_at'] or '-'} "
                f"({row['settled_by'] or 'nobody'}), cost {row['rounds_to_settle'] if row['rounds_to_settle'] is not None else '-'} round(s), "
                f"reopened {row['reopen_count']}x"
            )
        claims = paper.get("claims") or {}
        if claims:
            lines.append(f"- sentences checked later: {claims.get('settled')} settled, {claims.get('open')} open")
        item = paper["trend"]
        lines.append(
            f"- against its own recent best: {item['wins']} win(s), {item['losses']} loss(es), {item['ties']} tie(s), "
            f"{item['undecided']} not compared; win rate {item['win_rate']}; longest winning run {item['longest_win_streak']}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read a continuous run's layer walk and trend off disk.")
    parser.add_argument("--out", required=True, help="the continuous run root (the directory holding continuous_state.json)")
    parser.add_argument("--report", default="", help="where to write the JSON report (default: nowhere; --json prints it)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON instead of a table")
    args = parser.parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    report = summarize(out)
    if args.report:
        target = Path(args.report).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
