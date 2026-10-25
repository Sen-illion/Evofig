"""Settle the claims a reflection writes, beside the library that measures them.

A reflection says why a round went the way it did and what it expects next time. Each substantive
sentence is registered as a claim, with the reading that would settle it, and a later round that
measures that reading settles it: confirmed when the prediction held, refuted when it did not. A
round that measured nothing settles nothing, and that is the whole point of recording a verdict
separately from the absence of one -- a claim must not be written off by a reading the loop itself
refused to trust.

The ledger is append-only and lives beside the experience library, so the settle history outlives
the run that wrote the claim. A claim's streak is read the way the creator-rule ledger reads one:
consecutive refutations count, one confirmation clears the streak, and the claim is written off
only after ``CLAIM_SETTLEMENT_LIMIT`` in a row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...contracts.evolution import CLAIM_VERDICTS, claim_settles, deterministic_blocker_counts

CLAIM_LEDGER_FILE = "claim_ledger.jsonl"
CLAIM_LEDGER_VERSION = "1.0"

#: How many refutations in a row write a claim off. Deliberately the same bar the creator-rule
#: ledger uses, because both answer "is this advice still earning its place".
CLAIM_SETTLEMENT_LIMIT = 2

CLAIM_REGISTERED_EVENT = "claim_registered"
CLAIM_CONFIRMED_EVENT = "claim_confirmed"
CLAIM_REFUTED_EVENT = "claim_refuted"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def measured_metrics(result: dict[str, Any] | None, *, extra: dict[str, Any] | None = None) -> dict[str, float]:
    """The numbers a claim may name, read only from a seat's own measurement.

    The deterministic defect counts come first, so a claim about a defect kind is settled by the
    count of that kind. The five dimension scores follow under ``dimension:<name>`` and the
    absolute reading under ``overall_score``. Anything the caller passes in ``extra`` is added
    last. Model prose is never read here: a number a model typed is not a measurement.
    """

    source = result if isinstance(result, dict) else {}
    metrics: dict[str, float] = {}
    for key, value in deterministic_blocker_counts(source.get("stage_feedback")).items():
        metrics[str(key)] = float(value)
    dimensions = source.get("dimension_scores") if isinstance(source.get("dimension_scores"), dict) else {}
    for key, value in dimensions.items():
        number = _number(value)
        if number is not None:
            metrics[f"dimension:{key}"] = number
    score = _number(source.get("overall_score"))
    if score is not None:
        metrics["overall_score"] = score
    for key, value in (extra or {}).items():
        number = _number(value)
        if number is not None:
            metrics[str(key)] = number
    return metrics


class ClaimLedger:
    """Record what a reflection claimed and how each claim was later settled."""

    def __init__(self, path: str | Path):
        self.root = Path(path).resolve()
        self.events_path = self.root / CLAIM_LEDGER_FILE

    def _events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        return record

    def register(self, claims: Any, *, round_index: int, paper: str = "") -> list[dict[str, Any]]:
        """Register the claims a round wrote, skipping any this ledger already holds."""

        known = {str(item.get("claim_id") or "") for item in self._events()}
        written: list[dict[str, Any]] = []
        for claim in claims if isinstance(claims, (list, tuple)) else []:
            if not isinstance(claim, dict):
                continue
            identifier = str(claim.get("claim_id") or "")
            if not identifier or identifier in known:
                continue
            known.add(identifier)
            written.append(
                self._append(
                    {
                        "event": CLAIM_REGISTERED_EVENT,
                        "version": CLAIM_LEDGER_VERSION,
                        "claim_id": identifier,
                        "claim": str(claim.get("claim") or ""),
                        "apply_at": str(claim.get("apply_at") or ""),
                        "prediction": str(claim.get("prediction") or ""),
                        "criterion": dict(claim.get("criterion") or {}),
                        "round": int(round_index),
                        "paper": str(paper or ""),
                    }
                )
            )
        return written

    def settle(self, claim_id: str, verdict: str, *, round_index: int, metric: str = "", observed: Any = None, source_run: str = "") -> dict[str, Any]:
        """Record a verdict on one claim. A verdict outside ``CLAIM_VERDICTS`` records nothing.

        A round that measured nothing has no verdict to give, so it writes no row: the claim
        stays open rather than being credited or blamed for a reading that was never taken.
        """

        if str(verdict) not in CLAIM_VERDICTS:
            return {}
        return self._append(
            {
                "event": CLAIM_CONFIRMED_EVENT if str(verdict) == "confirmed" else CLAIM_REFUTED_EVENT,
                "version": CLAIM_LEDGER_VERSION,
                "claim_id": str(claim_id),
                "round": int(round_index),
                "metric": str(metric or ""),
                "observed": observed,
                "source_run": str(source_run or ""),
            }
        )

    def verdicts(self, claim_id: str) -> list[str]:
        """Every verdict recorded against one claim, oldest first."""

        return [
            "confirmed" if str(item.get("event") or "") == CLAIM_CONFIRMED_EVENT else "refuted"
            for item in self._events()
            if str(item.get("claim_id") or "") == str(claim_id)
            and str(item.get("event") or "") in {CLAIM_CONFIRMED_EVENT, CLAIM_REFUTED_EVENT}
        ]

    def refutation_streak(self, claim_id: str) -> int:
        """The trailing refutations: a confirmation resets the streak, an open round does not."""

        streak = 0
        for verdict in self.verdicts(claim_id):
            streak = streak + 1 if verdict == "refuted" else 0
        return streak

    def disabled_ids(self, *, limit: int = CLAIM_SETTLEMENT_LIMIT) -> set[str]:
        known = {str(item.get("claim_id") or "") for item in self._events()} - {""}
        return {identifier for identifier in known if self.refutation_streak(identifier) >= max(1, int(limit))}

    def open_claims(self) -> list[dict[str, Any]]:
        """Every registered claim with no verdict yet, in registration order."""

        settled = {
            str(item.get("claim_id") or "")
            for item in self._events()
            if str(item.get("event") or "") in {CLAIM_CONFIRMED_EVENT, CLAIM_REFUTED_EVENT}
        }
        rows: list[dict[str, Any]] = []
        for item in self._events():
            if str(item.get("event") or "") != CLAIM_REGISTERED_EVENT:
                continue
            if str(item.get("claim_id") or "") in settled:
                continue
            rows.append(dict(item))
        return rows

    def settle_open(self, metrics: dict[str, Any] | None, *, round_index: int, source_run: str = "") -> list[dict[str, Any]]:
        """Settle every open claim this round's readings can, and leave the rest open."""

        outcomes: list[dict[str, Any]] = []
        for claim in self.open_claims():
            verdict = claim_settles(claim, metrics)
            if not verdict:
                continue
            criterion = claim.get("criterion") if isinstance(claim.get("criterion"), dict) else {}
            metric = str(criterion.get("metric") or "")
            record = self.settle(
                str(claim.get("claim_id") or ""),
                verdict,
                round_index=round_index,
                metric=metric,
                observed=(metrics or {}).get(metric),
                source_run=source_run,
            )
            if record:
                outcomes.append({**record, "claim": str(claim.get("claim") or "")})
        return outcomes

    def tally(self) -> dict[str, dict[str, Any]]:
        """Every claim the ledger knows: its words, its trailing streak and its verdict counts."""

        claims: dict[str, dict[str, Any]] = {}
        for item in self._events():
            identifier = str(item.get("claim_id") or "")
            if not identifier:
                continue
            row = claims.setdefault(
                identifier,
                {
                    "claim": str(item.get("claim") or ""),
                    "apply_at": str(item.get("apply_at") or ""),
                    "criterion": dict(item.get("criterion") or {}),
                    "round": item.get("round"),
                    "confirmed": 0,
                    "refuted": 0,
                    "refutations": 0,
                },
            )
            if str(item.get("claim") or "") and not row.get("claim"):
                row["claim"] = str(item["claim"])
            if str(item.get("event") or "") == CLAIM_CONFIRMED_EVENT:
                row["confirmed"] += 1
            elif str(item.get("event") or "") == CLAIM_REFUTED_EVENT:
                row["refuted"] += 1
        for identifier, row in claims.items():
            row["refutations"] = self.refutation_streak(identifier)
        return claims

    def report(self, *, limit: int = CLAIM_SETTLEMENT_LIMIT) -> dict[str, Any]:
        """The readable account: what each claim said and how it turned out."""

        claims = self.tally()
        disabled = self.disabled_ids(limit=limit)
        settled = sum(int(row["confirmed"]) + int(row["refuted"]) for row in claims.values())
        return {
            "contract": "rfs.claim_ledger",
            "version": CLAIM_LEDGER_VERSION,
            "settlement_limit": int(limit),
            "claims": [
                {**row, "claim_id": identifier, "disabled": identifier in disabled}
                for identifier, row in sorted(claims.items())
            ],
            "counts": {
                "registered": len(claims),
                "confirmed": sum(int(row["confirmed"]) for row in claims.values()),
                "refuted": sum(int(row["refuted"]) for row in claims.values()),
                "open": len(self.open_claims()),
                "settled": settled,
                "disabled": len(disabled),
            },
        }


__all__ = [
    "CLAIM_CONFIRMED_EVENT",
    "CLAIM_LEDGER_FILE",
    "CLAIM_LEDGER_VERSION",
    "CLAIM_REFUTED_EVENT",
    "CLAIM_REGISTERED_EVENT",
    "CLAIM_SETTLEMENT_LIMIT",
    "ClaimLedger",
    "measured_metrics",
]
