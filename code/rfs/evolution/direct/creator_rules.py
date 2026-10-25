"""Aggregate measured Memory into the rule manifest that guides the Creator side.

This module reads; it never writes a rule of its own. Two libraries already hold measured
rows -- the experience store and the prompt-patch store -- and a creator rule is simply the
one row that survived the same precedence decision the Router uses when two rows compete
for one stage. A row whose support is thin stays out of the manifest and is reported as
withheld, so a single lucky case cannot become standing guidance.

The refutation ledger is the other half of the loop: a round that a rule guided and that
then rolled back counts against that rule, lowers its effective support, and after
``CREATOR_RULE_REFUTATION_LIMIT`` consecutive refutations the rule and the rows behind it
stop being injected at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from ...contracts.creator_rules import (
    CREATOR_RULE_MANIFEST_CONTRACT,
    CREATOR_RULE_MIN_SUPPORT,
    CREATOR_RULE_REFUTATION_LIMIT,
    CREATOR_RULE_VALIDATED_STATUSES,
    CREATOR_RULE_WITHDRAWN_STATUSES,
    creator_rule_id,
    normalize_creator_rule,
    normalize_creator_rule_manifest,
    rules_to_stage4_manifest,
    validate_creator_rule_manifest,
)
from ...contracts.prompt_patch import normalize_prompt_patch, render_patch_directive
from .prompt_patch import patch_class_for
from .routing import precedence_key
from .store import DIRECT_DOMAIN

CREATOR_RULE_EVENTS_FILE = "creator_rule_events.jsonl"
CREATOR_RULES_SOURCE = "measured_memory_aggregation"

#: The rollback reasons a guided round's own loss may be charged to the rule it ran. A reason
#: outside this list is not about the rule at all: a reading the round itself called unmeasurable
#: proved nothing, a new blocker says this version broke somewhere else, and a round that never
#: produced a figure has nothing to say about guidance. Every injected rule used to be charged for
#: every rollback alike, which is how ten rules that shared a run came to share one number that
#: described the run and none of the rules.
CREATOR_RULE_REFUTING_REASONS = ("pairwise_not_preferred", "no_significant_improvement")

#: The event that clears a rule's refutation streak. A rule is only written off after a run of
#: losses, so letting one back in has to be a recorded act rather than an edit of the event file.
CREATOR_RULE_RELEASED_EVENT = "creator_rule_released"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _texts(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _condition(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("condition", "context"):
        value = row.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _action(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("action")
    return value if isinstance(value, dict) else {}


def _action_type(row: dict[str, Any]) -> str:
    action = _action(row)
    return str(action.get("type") or action.get("action") or "")


def _support(row: dict[str, Any]) -> int:
    cases = row.get("validated_cases")
    counted = len(cases) if isinstance(cases, list) else 0
    measured = row.get("support")
    if isinstance(measured, int) and not isinstance(measured, bool):
        counted = max(counted, measured)
    return max(counted, 1)


def _row_status(row: dict[str, Any]) -> str:
    return str(row.get("status") or "")


def _is_supported(row: dict[str, Any], *, min_support: int) -> bool:
    """A row aggregates only when more than one case, or a validated rung, stands behind it."""

    return _support(row) >= max(1, int(min_support)) or _row_status(row) in CREATOR_RULE_VALIDATED_STATUSES


def _precedence_row(*, identifier: str, row: dict[str, Any], patch_class: str, condition: dict[str, Any]) -> dict[str, Any]:
    """Shape a row the way ``routing.precedence_key`` reads it.

    A prohibition declares the boundary the Router ranks first -- it names the action it
    refuses -- while an advisory row declares none, so a prohibition wins its group
    without a second, rule-specific tie-break.
    """

    action_type = _action_type(row)
    gain = _number(row.get("repair_gain"))
    if gain is None:
        gain = _number(row.get("median_repair_gain"))
    return {
        "evidence": {"experience_id": identifier, "support": _support(row), "repair_gain": gain},
        "repair_gain": gain,
        "action": _action(row),
        "condition": condition,
        "class": patch_class,
        "apply_at": str(row.get("apply_at") or ""),
        "forbidden_changes": [action_type] if patch_class == "prohibited" and action_type else [],
    }


def _agreed_applicability(ranked: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> tuple[str, str]:
    """Copy a scope only when every row that wrote one wrote the same one.

    A missing value does not vote, and a disagreement leaves the rule unlabeled.
    Motif overlap is not a vote.
    """

    kinds = {
        str(row.get("applicability") or "").strip().casefold()
        for _, row, _ in ranked
        if str(row.get("applicability") or "").strip()
    }
    if len(kinds) != 1:
        return "", ""
    kind = next(iter(kinds))
    if kind not in {"common", "domain"}:
        return "", ""
    domains = {
        str(row.get("applicability_domain") or "").strip()
        for _, row, _ in ranked
        if str(row.get("applicability") or "").strip().casefold() == "domain"
    }
    domains.discard("")
    if kind == "domain" and len(domains) != 1:
        return "", ""
    return kind, next(iter(domains)) if kind == "domain" else ""


def _rule_from_row(*, row: dict[str, Any], identifier: str, patch_class: str, condition: dict[str, Any], sources: dict[str, list[str]], applicability: str = "", applicability_domain: str = "") -> dict[str, Any]:
    action = _action(row)
    action_type = _action_type(row)
    directive = str(row.get("directive") or "").strip() or render_patch_directive(patch_class=patch_class, condition=condition, action=action if action else {"type": action_type})
    gains = [_number(row.get("repair_gain")), _number(row.get("median_repair_gain"))]
    measured = [value for value in gains if value is not None]
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    return normalize_creator_rule(
        {
            "rule_id": creator_rule_id(apply_at=str(row.get("apply_at") or ""), rule_class=patch_class, action_type=action_type, condition=condition),
            "class": patch_class,
            "apply_at": str(row.get("apply_at") or ""),
            "condition": condition,
            "action": action or {"type": action_type},
            "directive": directive,
            "applicability": applicability,
            "applicability_domain": applicability_domain,
            "support": _support(row),
            "median_repair_gain": round(float(median(measured)), 4) if measured else None,
            "excluded_conditions": row.get("excluded_conditions") if isinstance(row.get("excluded_conditions"), list) else [],
            "source_patch_ids": sources["patches"],
            "source_experience_ids": sources["experiences"],
            "provenance": {
                "source": str(provenance.get("source") or CREATOR_RULES_SOURCE),
                "run": str(provenance.get("run") or ""),
                "measured": bool(provenance.get("measured", True)),
                "round": provenance.get("round"),
            },
        }
    )


def build_creator_rule_manifest(
    rows: Iterable[dict[str, Any]],
    *,
    min_support: int = CREATOR_RULE_MIN_SUPPORT,
    experience_library: str | Path = "",
    domain: str = DIRECT_DOMAIN,
    refutations: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Aggregate measured rows into rules, one surviving row per stage-and-claim group."""

    counters = refutations if isinstance(refutations, dict) else {}
    groups: dict[tuple[str, str, str], list[tuple[str, dict[str, Any], dict[str, Any]]]] = {}
    withheld: list[dict[str, Any]] = []
    for item in rows if isinstance(rows, (list, tuple)) else []:
        row = item if isinstance(item, dict) else {}
        identifier = str(row.get("experience_id") or row.get("patch_id") or "")
        apply_at = str(row.get("apply_at") or "")
        patch_class = str(row.get("class") or patch_class_for(row))
        condition = _condition(row)
        if not identifier or not apply_at:
            continue
        if _row_status(row) in CREATOR_RULE_WITHDRAWN_STATUSES:
            withheld.append({"source_id": identifier, "apply_at": apply_at, "reason": "status_withdrawn", "status": _row_status(row)})
            continue
        if not _is_supported(row, min_support=min_support):
            withheld.append({"source_id": identifier, "apply_at": apply_at, "reason": "support_below_minimum", "support": _support(row)})
            continue
        precedence = _precedence_row(identifier=identifier, row=row, patch_class=patch_class, condition=condition)
        signature = json.dumps(condition, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        groups.setdefault((apply_at, _action_type(row), signature), []).append((identifier, row, precedence))

    rules: list[dict[str, Any]] = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda item: precedence_key(item[2]))
        winner_id, winner_row, _ = ranked[0]
        # A rule names the row it aggregated, every validated sibling that corroborates it,
        # and -- when the winner is a patch -- the experiences the patch was rendered from.
        experiences = {
            identifier
            for identifier, row, _ in ranked
            if str(row.get("status") or "") in CREATOR_RULE_VALIDATED_STATUSES and not row.get("patch_id")
        }
        patches = {identifier for identifier, row, _ in ranked if row.get("patch_id")}
        for _, row, _ in ranked:
            experiences.update(_texts(row.get("source_experience_ids")))
        if winner_row.get("patch_id"):
            patches.add(winner_id)
        else:
            experiences.add(winner_id)
        sources = {"experiences": sorted(experiences), "patches": sorted(patches)}
        applicability, applicability_domain = _agreed_applicability(ranked)
        rule = _rule_from_row(row=winner_row, identifier=winner_id, patch_class=str(winner_row.get("class") or patch_class_for(winner_row)), condition=winner_row.get("condition") if isinstance(winner_row.get("condition"), dict) else _condition(winner_row), sources=sources, applicability=applicability, applicability_domain=applicability_domain)
        refuted = int(counters.get(rule["rule_id"]) or 0)
        if refuted:
            rule = {
                **rule,
                "refutations": refuted,
                "status": "disabled" if refuted >= CREATOR_RULE_REFUTATION_LIMIT else "refuted",
                "excluded_conditions": rule["excluded_conditions"] or [{**rule["condition"], "reason": "guided_round_rolled_back", "refutations": refuted}],
            }
        rules.append(rule)

    manifest = normalize_creator_rule_manifest(
        {
            "rules": [rule for rule in rules if rule["status"] == "active"],
            "withheld": withheld,
            "min_support": int(min_support),
            "stage4_manifest": rules_to_stage4_manifest(rules),
            "provenance": {"source": CREATOR_RULES_SOURCE, "experience_library": str(experience_library), "measured": True, "domain": str(domain)},
        }
    )
    manifest["refuted_rules"] = [rule for rule in rules if rule["status"] != "active"]
    manifest["validation"] = validate_creator_rule_manifest(manifest)
    return manifest


def refuting_rule_ids(
    *,
    rule_stages: dict[str, str],
    reason: str,
    round_stages: Iterable[str],
) -> tuple[list[str], list[dict[str, str]]]:
    """Which rules one rolled-back round counts against, and why each of the others was spared.

    Two things have to hold before a rollback is charged to a rule. The reason has to say the
    guided figure itself came out worse (``CREATOR_RULE_REFUTING_REASONS``), and the rule has to
    guide a stage the round worked at: a round that redrew the panels says nothing against a rule
    that only ever spoke about gutters. When the round names no stage, or the rule names none, the
    round is charged to nothing, because a count that cannot be placed on one rule is not that
    rule's credibility.

    ``round_stages`` may hold more than one name for the same round: an action can be placed both
    at the stage it edits the figure at and at the stage its Memory is filed under, and a rule at
    either reading is implicated. The spared rules travel with the reason they were spared, so the
    round can record the whole decision rather than only the part of it that landed.
    """

    reason = str(reason or "").strip()
    wanted = {str(item or "").strip() for item in round_stages if str(item or "").strip()}
    charged: list[str] = []
    spared: list[dict[str, str]] = []
    for raw_id, raw_stage in rule_stages.items():
        rule_id = str(raw_id or "")
        if not rule_id:
            continue
        if reason not in CREATOR_RULE_REFUTING_REASONS:
            spared.append({"rule_id": rule_id, "reason": "the rollback reason is not this rule's own loss"})
            continue
        if not wanted or str(raw_stage or "").strip() not in wanted:
            spared.append({"rule_id": rule_id, "reason": "the round did not work at this rule's stage"})
            continue
        charged.append(rule_id)
    return charged, spared


class CreatorRuleLedger:
    """Record the rounds that refuted a creator rule, beside the library it guides."""

    def __init__(self, path: str | Path):
        self.root = Path(path).resolve()
        self.events_path = self.root / CREATOR_RULE_EVENTS_FILE

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

    def refutation_count(self, rule_id: str) -> int:
        """Count the trailing refutations: a confirmation or a release resets the streak."""

        streak = 0
        for item in self._events():
            if str(item.get("rule_id") or "") != str(rule_id):
                continue
            event = str(item.get("event") or "")
            if event == "creator_rule_refuted":
                streak += 1
            elif event in {"creator_rule_confirmed", CREATOR_RULE_RELEASED_EVENT}:
                streak = 0
        return streak

    def tally(self) -> dict[str, dict[str, int]]:
        """Every rule the ledger knows: its trailing streak and its lifetime event counts."""

        tally: dict[str, dict[str, int]] = {}
        for item in self._events():
            rule_id = str(item.get("rule_id") or "")
            if not rule_id:
                continue
            row = tally.setdefault(rule_id, {"refutations": 0, "refuted": 0, "confirmed": 0, "released": 0})
            event = str(item.get("event") or "")
            if event == "creator_rule_refuted":
                row["refuted"] += 1
            elif event == "creator_rule_confirmed":
                row["confirmed"] += 1
            elif event == CREATOR_RULE_RELEASED_EVENT:
                row["released"] += 1
        for rule_id, row in tally.items():
            row["refutations"] = self.refutation_count(rule_id)
        return tally

    def counters(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for key in {str(item.get("rule_id") or "") for item in self._events()} - {""}:
            streak = self.refutation_count(key)
            if streak:
                counts[key] = streak
        return counts

    def record_confirmation(self, rule_id: str, *, round_index: int) -> dict[str, Any]:
        """Record a guided round that survived the Gate, resetting the refutation streak."""

        record = {"rule_id": str(rule_id), "event": "creator_rule_confirmed", "round": int(round_index), "refutations": 0}
        self.root.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        return record

    def release(self, rule_id: str, *, round_index: int = 0, reason: str = "") -> dict[str, Any]:
        """Clear a rule's refutation streak, so a written-off rule may be injected again.

        The ledger is append-only, so this is the one way back: a rule that was written off on a
        run of losses is restored by an event of its own, and the run that restores it is on the
        record beside the run that wrote it off.
        """

        record = {
            "rule_id": str(rule_id),
            "event": CREATOR_RULE_RELEASED_EVENT,
            "round": int(round_index),
            "reason": str(reason),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        return record

    def record_refutation(self, rule_id: str, *, round_index: int, reason: str = "", source_ids: Iterable[str] = (), disabled: bool = False) -> dict[str, Any]:
        record = {
            "rule_id": str(rule_id),
            "event": "creator_rule_refuted",
            "round": int(round_index),
            "reason": str(reason),
            "source_ids": sorted(str(item) for item in source_ids if str(item)),
            "refutations": self.refutation_count(rule_id) + 1,
            "disabled": bool(disabled),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        return record


def rules_for_paper(manifest: dict[str, Any], domain_profile: dict[str, Any] | None) -> dict[str, Any]:
    """Drop domain rules that do not belong to this paper.

    A rule with no applicability still guides, because older rows never wrote one.
    A domain rule names the paper domain it was measured in and stays out of every
    other domain. A domain rule that names no domain can guide a paper that has one,
    and the experience written from that round records that paper's domain.
    """

    source = manifest if isinstance(manifest, dict) else {}
    profile = domain_profile if isinstance(domain_profile, dict) else {}
    paper_domain = str(profile.get("domain") or "").strip()
    missing = bool(profile.get("domain_missing")) or paper_domain.casefold() in {"", "unknown", "missing"}
    kept: list[dict[str, Any]] = []
    withheld = [dict(item) for item in source.get("withheld") or [] if isinstance(item, dict)]
    for rule in source.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        kind = str(rule.get("applicability") or "").strip().casefold()
        named = str(rule.get("applicability_domain") or "").strip()
        if kind == "domain" and (missing or (named and named != paper_domain)):
            withheld.append({"rule_id": str(rule.get("rule_id") or ""), "reason": "domain applicability does not match this paper", "applicability_domain": named})
            continue
        kept.append(rule)
    return {**source, "rules": kept, "withheld": withheld}


def read_creator_rule_manifest(
    *,
    experiences: Iterable[dict[str, Any]] = (),
    patches: Iterable[dict[str, Any]] = (),
    ledger: CreatorRuleLedger | None = None,
    min_support: int = CREATOR_RULE_MIN_SUPPORT,
    experience_library: str | Path = "",
    domain: str = DIRECT_DOMAIN,
) -> dict[str, Any]:
    """Read both libraries and the refutation ledger into one rule manifest."""

    rows: list[dict[str, Any]] = []
    for item in experiences:
        row = item if isinstance(item, dict) else {}
        if str(row.get("domain") or DIRECT_DOMAIN) != DIRECT_DOMAIN:
            continue
        rows.append(row)
    for item in patches:
        patch = normalize_prompt_patch(item if isinstance(item, dict) else {})
        if not patch["patch_id"]:
            continue
        rows.append(
            {
                "experience_id": patch["patch_id"],
                "patch_id": patch["patch_id"],
                "apply_at": patch["apply_at"],
                "class": patch["class"],
                "condition": patch["condition"],
                "action": {"type": patch["action_type"], **{key: value for key, value in (patch.get("parameters") or {}).items() if value is not None}},
                "directive": patch["directive"],
                "support": patch["support"],
                "median_repair_gain": patch["median_repair_gain"],
                "source_experience_ids": patch["source_experience_ids"],
                "status": patch["status"],
                "provenance": patch["provenance"],
            }
        )
    return build_creator_rule_manifest(
        rows,
        min_support=min_support,
        experience_library=experience_library,
        domain=domain,
        refutations=ledger.counters() if ledger is not None else None,
    )


__all__ = [
    "CREATOR_RULES_SOURCE",
    "CREATOR_RULE_EVENTS_FILE",
    "CREATOR_RULE_MANIFEST_CONTRACT",
    "CREATOR_RULE_REFUTING_REASONS",
    "CREATOR_RULE_RELEASED_EVENT",
    "CreatorRuleLedger",
    "build_creator_rule_manifest",
    "read_creator_rule_manifest",
    "refuting_rule_ids",
]
