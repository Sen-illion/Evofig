"""Keep the advice a reflection stated, and count how many reflections stated it.

A reflection can now say how to advise a later paper, not only which bounded move to run next
(P9): "when a node fans out to four others, constrain the router" is a claim about drawing, and a
claim with nowhere to live is re-derived and forgotten every round. This module is where those
statements are kept, beside the experience library they belong to, as one row per statement plus a
support count. Only rules past the support bar, or explicitly validated, are selected for
injection. Selection is read-only here; the prompt side decides which role may see them, and that
is the acceptance side never.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...contracts.evolution import APPLY_AT_STAGES
from ...contracts.suggestion_rule import (
    MIN_SUPPORT,
    REFUSED_STATUSES,
    SUGGESTION_RULE_CONTRACT,
    SUGGESTION_RULE_SCHEMA_VERSION,
    SUGGESTION_RULE_STATUSES,
    effective_rule_status,
    normalize_rule_status,
    normalize_suggestion_rule,
    suggestion_rule_id,
    suggestion_rule_usable,
    validate_suggestion_rule,
)
from .actions import normalize_mechanism_rule
from .file_lock import append_jsonl, exclusive_lock, write_json_atomic
from .store import DIRECT_DOMAIN


RULE_LIBRARY_FILE = "suggestion_rules.jsonl"
RULE_EVENTS_FILE = "suggestion_rule_events.jsonl"
RULE_MANIFEST_FILE = "suggestion_rule_manifest.json"

#: Advice rules are about how the figure is compiled, which is the stage that draws it.
DEFAULT_APPLY_AT = "ppt_compilation"


class SuggestionRuleStore:
    """Persist advice rules and their support beside the experience library."""

    def __init__(self, path: str | Path):
        self.root = Path(path).resolve()
        self.domain = DIRECT_DOMAIN
        self.path = self.root / RULE_LIBRARY_FILE
        self.events_path = self.root / RULE_EVENTS_FILE
        self.manifest_path = self.root / RULE_MANIFEST_FILE

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def append_result(self, record: dict[str, Any]) -> dict[str, Any]:
        """Validate one statement and append it, so the support history stays readable."""

        validation = validate_suggestion_rule(record if isinstance(record, dict) else {})
        normalized = validation["rule"]
        rule_id = str(normalized["rule_id"])
        if not validation["ok"]:
            return {"ok": False, "reason": "contract_validation_failed", "rule_id": rule_id, "errors": list(validation["errors"])}
        if normalized["status"] not in SUGGESTION_RULE_STATUSES:
            return {"ok": False, "reason": "status_not_allowed", "rule_id": rule_id, "status": normalized["status"], "allowed": list(SUGGESTION_RULE_STATUSES)}
        self.root.mkdir(parents=True, exist_ok=True)
        # The append and the manifest are one step, for the same reason as the experience
        # library: a wave writes this file from several papers at once.
        with exclusive_lock(self.path):
            append_jsonl(self.path, normalized)
            rows = self._read()
            self._write_manifest(rows)
        return {"ok": True, "reason": "appended", "rule_id": rule_id, "support": normalized["support"], "records": len(rows)}

    def append(self, record: dict[str, Any]) -> bool:
        return bool(self.append_result(record)["ok"])

    def records(self) -> list[dict[str, Any]]:
        """Return every stored statement, which is the support history of each rule."""

        return self._read()

    def append_event(self, rule_id: str, event: str, **payload: Any) -> dict[str, Any]:
        record = {"rule_id": str(rule_id), "event": str(event), **payload}
        self.root.mkdir(parents=True, exist_ok=True)
        return append_jsonl(self.events_path, record)

    def events(self, rule_id: str = "") -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        want = str(rule_id or "")
        rows: list[dict[str, Any]] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            if want and str(value.get("rule_id") or "") != want:
                continue
            rows.append(value)
        return rows

    def latest(self) -> list[dict[str, Any]]:
        """One row per rule: the newest statement, with any lifecycle event applied."""

        streamed: dict[str, str] = {}
        for event in self.events():
            rule_id = str(event.get("rule_id") or "")
            if rule_id and event.get("status"):
                streamed[rule_id] = str(event["status"])
        newest: dict[str, dict[str, Any]] = {}
        for row in self._read():
            normalized = normalize_suggestion_rule(row)
            newest[normalized["rule_id"]] = normalized
        return [
            normalize_rule_status({**newest[rule_id], "status": streamed.get(rule_id, newest[rule_id]["status"])})
            for rule_id in sorted(newest)
        ]

    def status_of(self, rule_id: str, default: str = "candidate") -> str:
        status = str(default or "candidate")
        for row in self._read():
            if str(row.get("rule_id") or "") == str(rule_id):
                status = str(row.get("status") or status)
        for event in self.events(str(rule_id)):
            if event.get("status"):
                status = str(event["status"])
        # The count is applied to the status here too, so a caller reading one field and a caller
        # reading the other cannot be told different rungs about the same rule.
        return effective_rule_status({"rule_id": rule_id, "status": status, "support": self._support_of(rule_id)})

    def _support_of(self, rule_id: str) -> int:
        """How many statements of this rule the library holds."""

        want = str(rule_id)
        statements = 0
        for row in self._read():
            if str(row.get("rule_id") or "") == want:
                statements = max(statements, int(row.get("support") or 0))
        return statements

    def transition(self, rule_id: str, status: str, **payload: Any) -> dict[str, Any]:
        return self.append_event(rule_id, "status_changed", status=status, **payload)

    def disable(self, rule_id: str, *, reason: str = "") -> dict[str, Any]:
        """Refuse a rule for good: its statements stay readable but none of them is injected."""

        return self.append_event(rule_id, "disabled", status="disabled", reason=str(reason))

    def record_statement(
        self,
        *,
        rule: dict[str, Any] | None,
        apply_at: str = DEFAULT_APPLY_AT,
        defect_pattern: Any = (),
        source_run: str = "",
        round_index: Any = None,
        source: str = "reflection",
    ) -> dict[str, Any]:
        """Record one reflection's rule, raising the support count of an identical earlier one.

        The mechanism vocabulary is checked here, one layer above the contract, so a rule that
        names an object id or an unknown parameter is refused rather than stored as a lesson. An
        identical rule stated again is a second witness, not a duplicate: the support count is
        what separates a habit from an opinion.
        """

        normalized_rule = normalize_mechanism_rule(rule)
        if not normalized_rule:
            return {"ok": False, "reason": "the reflection stated no mechanism-level rule the compiler can run", "rule": None}
        resolved_apply = str(apply_at or normalized_rule.get("apply_at") or DEFAULT_APPLY_AT)
        if resolved_apply not in APPLY_AT_STAGES:
            resolved_apply = DEFAULT_APPLY_AT
        identifier = suggestion_rule_id(apply_at=resolved_apply, rule=normalized_rule)
        existing = next((row for row in self.latest() if str(row["rule_id"]) == identifier), None)
        support = int((existing or {}).get("support") or 0) + 1
        recorded_status = str((existing or {}).get("status") or "candidate")
        if recorded_status not in SUGGESTION_RULE_STATUSES:
            recorded_status = "candidate"
        # The status is written on the rung the count earns, so the two fields a reader can ask
        # "is this rule in use" of cannot disagree. A row already validated, refuted or disabled
        # keeps its own rung: only the count bar moves a rule that was still a candidate.
        status = effective_rule_status({"status": recorded_status, "support": support})
        patterns = sorted({str(item) for item in (*((existing or {}).get("defect_pattern") or []), *(defect_pattern or [])) if str(item)})
        record = normalize_suggestion_rule(
            {
                "rule_id": identifier,
                "apply_at": resolved_apply,
                "defect_pattern": patterns,
                "rule": normalized_rule,
                "support": support,
                "status": status,
                "provenance": {"source": source, "run": source_run, "round": round_index},
            }
        )
        stored = self.append_result(record)
        if not stored["ok"]:
            return {"ok": False, "reason": str(stored.get("reason") or "append_failed"), "rule": record, "errors": list(stored.get("errors") or [])}
        if support >= MIN_SUPPORT:
            self.append_event(identifier, "support_reached", support=support, status=status)
        if status != recorded_status:
            self.transition(identifier, status, support=support, reason="the support count reached the bar")
        return {"ok": True, "reason": "recorded", "rule": record, "support": support, "usable": suggestion_rule_usable({**record, "status": status})}

    def select(self, *, apply_at: str = "", defect_pattern: Any = (), statuses: Any = None) -> dict[str, Any]:
        """Choose the advice a prompt may carry, and say why each other rule lost.

        The bar belongs to the contract: stated by two reflections, or validated outright. A rule
        whose own defect pattern the figure does not have is not injected either, because advice
        about a fan-out shown to a figure without one is noise dressed as guidance.
        """

        wanted = {str(item) for item in (defect_pattern or []) if str(item)}
        allowed = {str(item) for item in statuses} if statuses else None
        selected: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        considered = 0
        for row in self.latest():
            considered += 1
            if apply_at and str(row["apply_at"]) != str(apply_at):
                rejected.append({"rule_id": row["rule_id"], "reason": "apply_at_mismatch"})
                continue
            if allowed is not None and str(row["status"]) not in allowed:
                rejected.append({"rule_id": row["rule_id"], "reason": "status_not_allowed"})
                continue
            if not suggestion_rule_usable(row):
                reason = "status_refused" if str(row["status"]) in REFUSED_STATUSES else "below_support"
                rejected.append({"rule_id": row["rule_id"], "support": row["support"], "reason": reason})
                continue
            if wanted and not (set(row["defect_pattern"]) & wanted):
                rejected.append({"rule_id": row["rule_id"], "reason": "defect_pattern_mismatch"})
                continue
            selected.append({key: row[key] for key in ("rule_id", "apply_at", "defect_pattern", "rule_text", "rule", "support", "status")})
        return {
            "contract": SUGGESTION_RULE_CONTRACT,
            "apply_at": str(apply_at),
            "defect_pattern": sorted(wanted),
            "rules": selected,
            "rejected": rejected,
            "considered": considered,
        }

    def _write_manifest(self, rows: list[dict[str, Any]]) -> None:
        write_json_atomic(
            self.manifest_path,
            {
                "schema_version": SUGGESTION_RULE_SCHEMA_VERSION,
                "records": len(rows),
                "rules": sorted({str(item.get("rule_id") or "") for item in rows}),
            },
        )


__all__ = [
    "DEFAULT_APPLY_AT",
    "RULE_EVENTS_FILE",
    "RULE_LIBRARY_FILE",
    "RULE_MANIFEST_FILE",
    "SuggestionRuleStore",
]
