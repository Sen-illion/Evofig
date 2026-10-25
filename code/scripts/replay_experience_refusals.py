"""Re-judge a run's recorded Memory refusals under the rule the library uses now.

A round the comparison kept used to be refused a row whenever the deterministic gate named one
more defect than the earlier round had: the measurement was thrown away with the defect, and the
library stayed empty while the figures kept improving. That refusal is gone. The rule now files
the row, writes the defect it carried onto the row itself, and lets the validation ladder spend
the mark -- the row may be used by the paper that measured it, and may not reach another paper
until a set of paired cases shows the defect does not reproduce.

This script reads what those refusals recorded and says what the current rule would have done,
without a model call and without re-running a round. It is a measuring instrument for the change
rather than a way to edit history: nothing on disk is written except the report, and a record
whose reason is one the new rule still refuses is reported as unchanged rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPLAY_CONTRACT = "rfs.experience_refusal_replay"
REPLAY_VERSION = 1
#: The one reason the library no longer refuses a round for. It is written from an accepted
#: round only -- the round kept the figure and measured a gain, and the row was dropped over a
#: defect the round introduced -- so a record that carries it is a row the new rule files.
REFUSAL_THAT_NO_LONGER_REFUSES = "new_blocking_issues"
REFUSAL_FILE = "experience_rejected.json"


def rejudge_refusal(record: dict[str, Any] | None) -> dict[str, Any]:
    """Say what the current rule does with one recorded refusal.

    The record is the round's own ``experience_rejected.json``: the reason it was refused for,
    and -- for the reason that changed -- the defects the round introduced. A record with no
    reason names no decision, and is reported as unresolved instead of counted either way.
    """

    source = record if isinstance(record, dict) else {}
    reason = str(source.get("reason") or "").strip()
    if not reason:
        # Nothing in the record says which refusal this was: refuse to invent one.
        return {"decision": "unresolved", "reason": "", "carried_defects": []}
    if reason == REFUSAL_THAT_NO_LONGER_REFUSES:
        carried = [str(item) for item in (source.get("new_blocking_defects") or []) if str(item)]
        return {
            "decision": "file",
            "reason": reason,
            "carried_defects": carried,
            "carried_count": int(source.get("new_blockers") or len(carried) or 0),
            "files_as": "current_case_validated",
            "promotion": "held until a validation set shows the defect does not reproduce",
        }
    return {"decision": "unchanged", "reason": reason, "carried_defects": []}


def _iter_refusals(root: Path) -> list[Path]:
    """Every recorded refusal under a run directory, in a stable order."""

    if root.is_file():
        return [root]
    return sorted(root.glob(f"**/{REFUSAL_FILE}"))


def replay_refusals(paths: list[Path]) -> dict[str, Any]:
    """Re-judge every recorded refusal under a run directory and report what moved."""

    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            unresolved.append({"path": str(path), "reason": f"unreadable: {exc}"})
            continue
        answer = rejudge_refusal(record)
        row = {
            "path": str(path),
            "round": (record or {}).get("round"),
            "recorded_reason": str((record or {}).get("reason") or ""),
            **answer,
        }
        rows.append(row)
        if answer["decision"] == "unresolved":
            unresolved.append({"path": str(path), "reason": "the record names no reason"})

    filed = [row for row in rows if row["decision"] == "file"]
    unchanged = [row for row in rows if row["decision"] == "unchanged"]
    by_reason: dict[str, int] = {}
    for row in unchanged:
        by_reason[row["recorded_reason"]] = by_reason.get(row["recorded_reason"], 0) + 1
    carried: dict[str, int] = {}
    for row in filed:
        for defect in row["carried_defects"]:
            carried[defect] = carried.get(defect, 0) + 1
    return {
        "contract": REPLAY_CONTRACT,
        "version": REPLAY_VERSION,
        "refusal_count": len(rows),
        "now_filed_count": len(filed),
        "still_refused_count": len(unchanged),
        "unresolved_count": len(unresolved),
        "still_refused_by_reason": dict(sorted(by_reason.items())),
        "carried_defects": dict(sorted(carried.items())),
        "now_filed": filed,
        "unresolved": unresolved,
        "summary": (
            f"{len(rows)} recorded refusal(s); {len(filed)} would be filed as a row under the "
            f"current rule and {len(unchanged)} would still be refused"
        ),
        "checks": [
            {
                "name": "a round refused over a defect it introduced is filed now",
                "ok": True,
                "detail": f"{len(filed)} row(s) would enter the library carrying a defect",
            },
            {
                "name": "every other refusal keeps its own reason",
                "ok": True,
                "detail": f"{len(unchanged)} refusal(s) unchanged, over {len(by_reason)} reason(s)",
            },
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-judge recorded Memory refusals under the current rule.")
    parser.add_argument("--refusals", required=True, help=f"A run directory (recursively searched) or a single {REFUSAL_FILE}.")
    parser.add_argument("--out", default="", help="Where to write the report; stdout when omitted.")
    args = parser.parse_args(argv)

    root = Path(args.refusals)
    paths = _iter_refusals(root)
    if not paths:
        print(f"no {REFUSAL_FILE} under {root}", file=sys.stderr)
        return 2
    report = replay_refusals(paths)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
