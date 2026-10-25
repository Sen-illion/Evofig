"""Show a creator rule's refutation streak, and clear it when the rule deserves another run.

A rule is only written off after a run of losses, and the ledger that counts them is append-only:
a rule that was written off cannot simply be edited back in. This is the other end of that loop --
the operator's way to read which rules are standing at the limit and to record a release, so the
next run's manifest injects them again. Nothing here decides whether a rule is good; it reports
the tally and records the one act the ledger can be given.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts.creator_rules import CREATOR_RULE_REFUTATION_LIMIT
from ..evolution.direct import CreatorRuleLedger
from ..utils import write_json


CREATOR_RULE_LEDGER_CONTRACT = "rfs.creator_rule_ledger_report"
CREATOR_RULE_LEDGER_REPORT_FILE = "creator_rule_ledger_report.json"
RELEASE_REASON = "released by the operator"


def report_creator_rule_ledger(
    library: str | Path,
    *,
    release: str = "",
    release_all: bool = False,
    reason: str = "",
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Report every rule's refutation streak, recording a release when one was asked for.

    A release is written before the tally is read, so the report a caller sees is the state the
    next run will read rather than the one it replaced. ``out`` is a file path and nothing is
    written when it is empty: reading the tally must not leave a file inside the library.
    """

    ledger = CreatorRuleLedger(library)
    wanted_release = str(release or "").strip()
    released: list[str] = []
    if release_all:
        # Every rule the ledger has ever seen, because a rule written off many rounds ago is
        # exactly the one an operator is looking for.
        for rule_id in sorted(ledger.tally()):
            ledger.release(rule_id, reason=str(reason or RELEASE_REASON))
            released.append(rule_id)
    elif wanted_release:
        ledger.release(wanted_release, reason=str(reason or RELEASE_REASON))
        released.append(wanted_release)
    rules = [
        {
            "rule_id": rule_id,
            "refutations": int(row.get("refutations") or 0),
            "refuted": int(row.get("refuted") or 0),
            "confirmed": int(row.get("confirmed") or 0),
            "released": int(row.get("released") or 0),
            "disabled": int(row.get("refutations") or 0) >= CREATOR_RULE_REFUTATION_LIMIT,
        }
        for rule_id, row in sorted(ledger.tally().items())
    ]
    disabled_rule_ids = [row["rule_id"] for row in rules if row["disabled"]]
    report = {
        "contract": CREATOR_RULE_LEDGER_CONTRACT,
        "library": str(Path(library).resolve()),
        "events_path": str(ledger.events_path),
        "refutation_limit": CREATOR_RULE_REFUTATION_LIMIT,
        "rules": rules,
        "disabled_rule_ids": disabled_rule_ids,
        "released": released,
        "ok": True,
    }
    report["summary"] = (
        f"{len(rules)} rule(s) on the ledger, {len(disabled_rule_ids)} at the refutation limit"
        + (f"; released {len(released)}" if released else "")
        + "."
    )
    if out:
        write_json(Path(out), report)
    return report


__all__ = [
    "CREATOR_RULE_LEDGER_CONTRACT",
    "CREATOR_RULE_LEDGER_REPORT_FILE",
    "RELEASE_REASON",
    "report_creator_rule_ledger",
]
