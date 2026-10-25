"""The record a reflection leaves behind about how to advise, and the bar it must clear.

P9 let a reflection state a mechanism-level rule, but a rule with nowhere to live is forgotten by
the next round: the advisory Judge re-derives its advice from scratch every call. This contract is
where that advice is written down -- one row naming the defect pattern it answers, the mechanism
parameters it sets, and how many independent reflections have said it. Reading a rule back into a
prompt is gated on that count, so one round's opinion cannot become the next paper's standing
instruction.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


SUGGESTION_RULE_CONTRACT = "rfs.suggestion_rule"
SUGGESTION_RULE_SCHEMA_VERSION = "1.0"

#: The rungs an advice rule moves through. ``usable`` is the rung the support count earns and
#: ``validated`` is the one a measurement earns, which short-circuits the count.
SUGGESTION_RULE_STATUSES = ("candidate", "usable", "validated", "refuted", "disabled")

#: How many independent reflections have to state the same rule before a prompt may carry it.
MIN_SUPPORT = 2

#: The rung the support count itself earns. The status and the count used to be two calibers for
#: one question -- the stored status said ``candidate`` for ever while the count decided selection
#: -- so a reader asking "is this rule in use" got a different answer depending on which field it
#: read. One status, one count: reaching the count sets this status.
USABLE_STATUS = "usable"

#: The one status the support count does not have to be met for.
VALIDATED_STATUS = "validated"

#: The statuses that refuse injection however often they were stated.
REFUSED_STATUSES = ("refuted", "disabled")

#: Where a rule keeps the parameters it sets, whichever mechanism action it names.
RULE_PARAMETER_KEYS = ("constraints", "params")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def normalize_rule_block(rule: Any) -> dict[str, Any]:
    """Read a rule as a dict. The vocabulary it may use is checked one layer up, not here."""

    return dict(rule) if isinstance(rule, dict) else {}


def render_rule_text(rule: Any) -> str:
    """Render a rule as one stable line, so the stored text cannot be invented prose."""

    block = normalize_rule_block(rule)
    action_type = _text(block.get("type") or block.get("action"))
    if not action_type:
        return ""
    parameters: dict[str, Any] = {}
    for key in RULE_PARAMETER_KEYS:
        value = block.get(key)
        if isinstance(value, dict):
            parameters.update(value)
    if not parameters:
        return action_type
    body = ", ".join(f"{key}={_compact(value)}" for key, value in sorted(parameters.items()))
    return f"{action_type}({body})"


def suggestion_rule_id(*, apply_at: str, rule: Any) -> str:
    """Derive a stable id from what the rule says, so a rerun cannot fork it."""

    canonical = "|".join([_text(apply_at), render_rule_text(rule)])
    return "suggestion-rule-" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def _patterns(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted({str(item) for item in value if str(item)})


def normalize_suggestion_rule(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Read one advice rule, filling every field so a caller never branches on a missing key."""

    source = raw if isinstance(raw, dict) else {}
    provenance = source.get("provenance") if isinstance(source.get("provenance"), dict) else {}
    rule = normalize_rule_block(source.get("rule"))
    apply_at = _text(source.get("apply_at")) or _text(rule.get("apply_at"))
    status = _text(source.get("status")) or "candidate"
    if status not in SUGGESTION_RULE_STATUSES:
        status = "candidate"
    try:
        support = max(0, int(source.get("support") or 0))
    except (TypeError, ValueError):
        support = 0
    identifier = _text(source.get("rule_id")) or (suggestion_rule_id(apply_at=apply_at, rule=rule) if rule else "")
    return {
        "contract": SUGGESTION_RULE_CONTRACT,
        "schema_version": SUGGESTION_RULE_SCHEMA_VERSION,
        "rule_id": identifier,
        "apply_at": apply_at,
        "defect_pattern": _patterns(source.get("defect_pattern") or source.get("defect_patterns")),
        "rule_text": _text(source.get("rule_text")) or render_rule_text(rule),
        "rule": rule,
        "support": support,
        "status": status,
        "provenance": {
            "source": _text(provenance.get("source")),
            "run": _text(provenance.get("run")),
            "round": provenance.get("round"),
        },
    }


def validate_suggestion_rule(rule: dict[str, Any]) -> dict[str, Any]:
    """Check one advice rule against the contract, returning the normalized rule either way."""

    normalized = normalize_suggestion_rule(rule)
    errors: list[str] = []
    if not normalized["rule_id"]:
        errors.append("rule_id is required")
    if not normalized["rule"]:
        errors.append("a suggestion rule must carry the mechanism rule it states")
    if not normalized["rule_text"]:
        errors.append("rule_text is required")
    return {"ok": not errors, "errors": errors, "rule": normalized}


def suggestion_rule_usable(rule: dict[str, Any]) -> bool:
    """Whether a rule has earned its way into a prompt: stated twice, or validated outright."""

    normalized = normalize_suggestion_rule(rule)
    if normalized["status"] in REFUSED_STATUSES:
        return False
    if normalized["status"] == VALIDATED_STATUS:
        return True
    return normalized["support"] >= MIN_SUPPORT


def effective_rule_status(rule: dict[str, Any]) -> str:
    """The rung a rule actually stands on, with the support count applied to the status.

    The one place the two are reconciled, so every reader -- the store's own listing, the
    selection, a caller counting what is in use -- answers the same question the same way. A row
    written before the status followed the count is read here as the rung its support earns.
    """

    normalized = normalize_suggestion_rule(rule)
    if normalized["status"] in REFUSED_STATUSES or normalized["status"] == VALIDATED_STATUS:
        return normalized["status"]
    if normalized["support"] >= MIN_SUPPORT:
        return USABLE_STATUS
    return normalized["status"]


def normalize_rule_status(rule: dict[str, Any]) -> dict[str, Any]:
    """One rule with its status brought onto the rung its own support count earns."""

    normalized = normalize_suggestion_rule(rule)
    return {**normalized, "status": effective_rule_status(normalized)}


__all__ = [
    "MIN_SUPPORT",
    "REFUSED_STATUSES",
    "RULE_PARAMETER_KEYS",
    "SUGGESTION_RULE_CONTRACT",
    "SUGGESTION_RULE_SCHEMA_VERSION",
    "SUGGESTION_RULE_STATUSES",
    "USABLE_STATUS",
    "VALIDATED_STATUS",
    "effective_rule_status",
    "normalize_rule_block",
    "normalize_rule_status",
    "normalize_suggestion_rule",
    "render_rule_text",
    "suggestion_rule_id",
    "suggestion_rule_usable",
    "validate_suggestion_rule",
]
