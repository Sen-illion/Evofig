"""Stable contract for the rule manifest the Creator side is guided by.

A creator rule is the aggregated form of several measured rows: the same stage, the same
class, the same condition, and an action that survived the precedence decision between
them. It exists so the structure and composition planners can be guided once, by a list
that names its own evidence, instead of being handed a pile of raw experience.

The manifest carries the fields ``exemplar_style_brief.stage4_manifest`` already uses, so a
rule list can be injected through the same channel without the planning layer learning a
new vocabulary. A thin rule is not a rule: a row only aggregates when its measured support
or its validated lifecycle already says more than one case stands behind it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .evolution import APPLICABILITY_VALUES, APPLY_AT_STAGES


CREATOR_RULE_MANIFEST_CONTRACT = "rfs.creator_rule_manifest"
CREATOR_RULE_MANIFEST_VERSION = 1
CREATOR_RULE_CLASSES = ("advisory", "executable_geometry", "prohibited")
CREATOR_RULE_STATUSES = ("active", "refuted", "disabled")
CREATOR_RULE_MIN_SUPPORT = 2
CREATOR_RULE_REFUTATION_LIMIT = 2
# The lifecycle rungs a paired validation already walked, which are multi-case evidence on
# their own even when the record carries a single entry in ``validated_cases``. These are the
# same three rungs retrieval and patch assembly use, so a row cannot guide the Creator while
# staying invisible to every other reader.
CREATOR_RULE_VALIDATED_STATUSES = ("same_domain_validated", "cross_domain_validated", "domain_validated", "promoted")
# Rungs that mean the evidence behind a row was withdrawn: quarantined while it waits to be
# re-measured under a newer Judge, refuted, disabled, or failed validation. None of them may
# guide the Creator, however many cases the row once reported.
CREATOR_RULE_WITHDRAWN_STATUSES = ("retrying", "refuted", "disabled", "overfit_or_failed")

# A hand-written standard is not a measured finding: it guides the planners the same way an
# aggregated rule does, but it names no case and gains nothing from being validated, so it
# travels under its own source and is reported apart from the rules the loop earned.
CREATOR_RULE_CURATED_SOURCE = "curated"
# The one standard the loop starts with, written down so a figure is steered by an explicit
# idea of "good" instead of only by the defects a round happened to measure. Every directive
# names a mechanism parameter the compile already understands, so the guidance stays inside
# the vocabulary the rest of the pipeline reads.
CURATED_STANDARD_RULES: tuple[dict[str, Any], ...] = (
    {
        "rule_id": "curated-planning-direction",
        "class": "advisory",
        "apply_at": "global_planning",
        "directive": (
            "Drive the whole figure along one main direction (left to right, or top to bottom), "
            "with the entry modules at the top left and the outcome at the bottom right."
        ),
    },
    {
        "rule_id": "curated-segmentation-panels",
        "class": "advisory",
        "apply_at": "segmentation",
        "directive": (
            "Split the figure by scientific stage, one nameable stage per panel, and keep the "
            "panel count between four and seven; a panel holding a single low-information node "
            "should merge into its neighbour instead of standing alone."
        ),
    },
    {
        "rule_id": "curated-layout-occupancy",
        "class": "advisory",
        "apply_at": "ppt_compilation",
        "directive": (
            "Fill each panel with its members: keep panel_occupancy_min at or above 0.45 so a "
            "large panel is never mostly empty, and never let its members huddle in one corner."
        ),
    },
    {
        "rule_id": "curated-layout-gutter",
        "class": "advisory",
        "apply_at": "ppt_compilation",
        "directive": (
            "Keep at least gutter_min between neighbouring boxes inside a panel, and keep every "
            "label clear of box borders and of the arrow paths passing nearby."
        ),
    },
    {
        "rule_id": "curated-layout-grid",
        "class": "advisory",
        "apply_at": "ppt_compilation",
        "directive": (
            "Snap boxes onto one alignment grid (align_grid) so nodes on the same level share a "
            "baseline and a left edge instead of drifting a few pixels apart."
        ),
    },
    {
        "rule_id": "curated-route-shared-channel",
        "class": "advisory",
        "apply_at": "ppt_compilation",
        "directive": (
            "Give parallel long connectors one shared vertical channel (shared_vertical_channel) "
            "instead of a separate lane each, so the reader sees one bus rather than a bundle."
        ),
    },
    {
        "rule_id": "curated-route-no-crossing",
        "class": "advisory",
        "apply_at": "ppt_compilation",
        "directive": (
            "Prefer routes that do not cross (forbid_crossing); allow a crossing only inside a "
            "single panel, and only where the two paths meet near a right angle."
        ),
    },
    {
        "rule_id": "curated-route-simple",
        "class": "advisory",
        "apply_at": "ppt_compilation",
        "directive": (
            "Keep each bend no tighter than min_bend and each connector within max_segments "
            "straight runs, so a path reads as one clean elbow instead of a staircase."
        ),
    },
)
CREATOR_RULE_STAGE4_KEYS = ("composition_patterns", "global_geometry_rules", "panel_sizing_rules", "anti_patterns", "evidence_references")


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _applicability(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return text if text in APPLICABILITY_VALUES else ""


def _applicability_domain(applicability: Any, domain: Any) -> str:
    if _applicability(applicability) != "domain":
        return ""
    return str(domain or "").strip()


def _count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def normalize_creator_rule(raw: dict[str, Any] | None) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    provenance = source.get("provenance") if isinstance(source.get("provenance"), dict) else {}
    status = str(source.get("status") or "active")
    return {
        "contract": CREATOR_RULE_MANIFEST_CONTRACT,
        "version": CREATOR_RULE_MANIFEST_VERSION,
        "rule_id": str(source.get("rule_id") or ""),
        "class": str(source.get("class") or "advisory"),
        "apply_at": str(source.get("apply_at") or ""),
        "condition": source.get("condition") if isinstance(source.get("condition"), dict) else {},
        "action": source.get("action") if isinstance(source.get("action"), dict) else {},
        "directive": str(source.get("directive") or ""),
        "applicability": _applicability(source.get("applicability")),
        "applicability_domain": _applicability_domain(source.get("applicability"), source.get("applicability_domain")),
        "support": _count(source.get("support")),
        "median_repair_gain": _number(source.get("median_repair_gain")),
        "excluded_conditions": [dict(item) for item in source.get("excluded_conditions") if isinstance(item, dict)] if isinstance(source.get("excluded_conditions"), list) else [],
        "source_patch_ids": _list_of_strings(source.get("source_patch_ids")),
        "source_experience_ids": _list_of_strings(source.get("source_experience_ids")),
        "refutations": _count(source.get("refutations")),
        "status": status if status in CREATOR_RULE_STATUSES else "active",
        "provenance": {
            "source": str(provenance.get("source") or ""),
            "run": str(provenance.get("run") or ""),
            "measured": bool(provenance.get("measured")),
            "round": provenance.get("round"),
        },
    }


def validate_creator_rule(rule: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_creator_rule(rule)
    errors: list[str] = []
    if not normalized["rule_id"]:
        errors.append("rule_id is required")
    if normalized["class"] not in CREATOR_RULE_CLASSES:
        errors.append("rule class is unsupported")
    if normalized["apply_at"] not in APPLY_AT_STAGES:
        errors.append("rule apply_at stage is unsupported")
    if not normalized["directive"]:
        errors.append("rule directive is required")
    if not normalized["source_patch_ids"] and not normalized["source_experience_ids"]:
        errors.append("a rule must name the rows it aggregated")
    if normalized["median_repair_gain"] is None:
        errors.append("rule median_repair_gain must be a measured number")
    if normalized["status"] != "active" and not normalized["excluded_conditions"]:
        errors.append("an inactive rule must name the conditions it was refuted under")
    return {"ok": not errors, "errors": errors, "rule": normalized}


def rules_to_stage4_manifest(rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Render rules into the manifest shape the planning layer already reads.

    The numeric geometry keys stay empty on purpose: a rule may add guidance and an
    anti-pattern, but a measured threshold only ever moves through a parameter patch,
    where it stays one deterministic number.
    """

    active = [normalize_creator_rule(rule) for rule in rules if str((rule or {}).get("status") or "active") == "active"]
    return {
        "composition_patterns": [rule["directive"] for rule in active if rule["class"] == "advisory"],
        "global_geometry_rules": {},
        "panel_sizing_rules": {},
        "anti_patterns": [rule["directive"] for rule in active if rule["class"] == "prohibited"],
        "evidence_references": sorted({identifier for rule in active for identifier in (*rule["source_patch_ids"], *rule["source_experience_ids"])}),
        "creator_rules": [
            {
                "rule_id": rule["rule_id"],
                "class": rule["class"],
                "apply_at": rule["apply_at"],
                "directive": rule["directive"],
                "support": rule["support"],
                "median_repair_gain": rule["median_repair_gain"],
            }
            for rule in active
        ],
    }


def normalize_creator_rule_manifest(raw: dict[str, Any] | None) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    rules = [normalize_creator_rule(item) for item in source.get("rules", [])] if isinstance(source.get("rules"), list) else []
    buckets: dict[str, list[str]] = {}
    for rule in rules:
        buckets.setdefault(rule["apply_at"], []).append(rule["rule_id"])
    provenance = source.get("provenance") if isinstance(source.get("provenance"), dict) else {}
    return {
        "contract": CREATOR_RULE_MANIFEST_CONTRACT,
        "version": CREATOR_RULE_MANIFEST_VERSION,
        "rules": rules,
        "buckets": {key: sorted(value) for key, value in sorted(buckets.items())},
        "withheld": [dict(item) for item in source.get("withheld", []) if isinstance(item, dict)] if isinstance(source.get("withheld"), list) else [],
        "min_support": int(source.get("min_support") or CREATOR_RULE_MIN_SUPPORT),
        "stage4_manifest": source.get("stage4_manifest") if isinstance(source.get("stage4_manifest"), dict) else rules_to_stage4_manifest(rules),
        "stage4_execution_policy": {
            "enabled": bool(rules),
            "mode": "measured_memory_rules",
            "rule_classes": list(CREATOR_RULE_CLASSES),
            "source_manifest": str(provenance.get("source") or ""),
        },
        "provenance": {
            "source": str(provenance.get("source") or ""),
            "experience_library": str(provenance.get("experience_library") or ""),
            "measured": bool(provenance.get("measured")),
            "domain": str(provenance.get("domain") or ""),
        },
    }


def validate_creator_rule_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_creator_rule_manifest(manifest)
    errors: list[str] = []
    for index, rule in enumerate(normalized["rules"]):
        validation = validate_creator_rule(rule)
        errors.extend(f"rules[{index}]: {item}" for item in validation["errors"])
    if set(normalized["stage4_manifest"]) < set(CREATOR_RULE_STAGE4_KEYS):
        errors.append("the stage4 manifest is missing a key the planning layer reads")
    return {"ok": not errors, "errors": errors, "manifest": normalized}


def creator_rule_id(*, apply_at: str, rule_class: str, action_type: str, condition: Any) -> str:
    """Derive a stable id from what a rule claims, so a rerun cannot fork it."""

    canonical = "|".join([apply_at, rule_class, action_type, json.dumps(condition or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)])
    return "creator-rule-" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]



def _curated_rules(rules: Any) -> list[dict[str, Any]]:
    """Normalize hand-written rows without demanding the evidence a measured rule must name."""

    curated: list[dict[str, Any]] = []
    for raw in rules if isinstance(rules, (list, tuple)) else []:
        row = dict(raw) if isinstance(raw, dict) else {}
        row["class"] = "advisory"
        row.setdefault("applicability", "common")
        row["provenance"] = {"source": CREATOR_RULE_CURATED_SOURCE, "measured": False}
        normalized = normalize_creator_rule(row)
        if normalized["rule_id"] and normalized["directive"]:
            curated.append(normalized)
    return curated


def validate_curated_standard(manifest: dict[str, Any]) -> dict[str, Any]:
    """Check a curated manifest without demanding the case evidence a measured rule must name."""

    source = manifest if isinstance(manifest, dict) else {}
    errors: list[str] = []
    for index, raw in enumerate(source.get("rules") or []):
        rule = normalize_creator_rule(raw)
        if not rule["rule_id"]:
            errors.append(f"rules[{index}]: rule_id is required")
        if rule["class"] != "advisory":
            errors.append(f"rules[{index}]: a curated rule may only be advisory")
        if rule["apply_at"] not in APPLY_AT_STAGES:
            errors.append(f"rules[{index}]: rule apply_at stage is unsupported")
        if not rule["directive"]:
            errors.append(f"rules[{index}]: rule directive is required")
    return {"ok": not errors, "errors": errors, "count": len(source.get("rules") or [])}


def load_curated_standard(path: str | Path | None = None) -> dict[str, Any]:
    """Read the hand-written figure standard, from a file when one is named.

    The standard travels the same manifest shape an aggregated rule list does, so the planning
    layer reads one vocabulary. What it must never do is look measured: every row carries
    ``source: curated`` and ``measured: false`` and is reported apart from the earned rules.
    """

    if path:
        source = Path(path)
        body = json.loads(source.read_text(encoding="utf-8"))
        raw = body.get("rules") if isinstance(body, dict) else None
        provenance_path = str(source)
    else:
        raw = CURATED_STANDARD_RULES
        provenance_path = "builtin"
    rules = _curated_rules(raw)
    manifest = normalize_creator_rule_manifest({"rules": rules})
    # The standard is authored, not measured, so nothing stands behind it: an empty support is
    # the honest number, and it must not be rounded up to the minimum a measured row needs.
    manifest["min_support"] = 0
    manifest["stage4_manifest"] = rules_to_stage4_manifest(rules)
    manifest["provenance"] = {"source": CREATOR_RULE_CURATED_SOURCE, "measured": False, "path": provenance_path}
    manifest["validation"] = validate_curated_standard(manifest)
    return manifest


def merge_creator_rule_manifests(*manifests: dict[str, Any], refutations: dict[str, int] | None = None) -> dict[str, Any]:
    """Combine rule lists into one manifest, keeping each row's own source.

    A curated id and a measured id never collide, so the union is a concatenation; buckets and
    the stage-4 view are recomputed because both are derived, never stored.

    The refutation ledger is applied to the union when its counters are handed in. A measured rule
    arrives already carrying its own status, because the ledger was read where it was aggregated;
    a curated rule is built from the authored list instead, so without this pass the ledger could
    write a standard off and the standard would keep being injected under it. A written-off rule
    moves out of ``rules`` -- the list every stage injects -- and into ``refuted_rules``.
    """

    counters = refutations if isinstance(refutations, dict) else {}
    rules: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    refuted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest in manifests:
        source = manifest if isinstance(manifest, dict) else {}
        for raw in source.get("rules") or []:
            rule = normalize_creator_rule(raw)
            if not rule["rule_id"] or rule["rule_id"] in seen:
                continue
            seen.add(rule["rule_id"])
            rules.append(rule)
        withheld.extend(dict(item) for item in source.get("withheld") or [] if isinstance(item, dict))
        refuted.extend(dict(item) for item in source.get("refuted_rules") or [] if isinstance(item, dict))
    active: list[dict[str, Any]] = []
    for rule in rules:
        count = _count(counters.get(rule["rule_id"]))
        if not count:
            active.append(rule)
            continue
        refuted.append(
            {
                **rule,
                "refutations": count,
                "status": "disabled" if count >= CREATOR_RULE_REFUTATION_LIMIT else "refuted",
                "excluded_conditions": rule["excluded_conditions"]
                or [{**rule["condition"], "reason": "guided_round_rolled_back", "refutations": count}],
            }
        )
    merged = normalize_creator_rule_manifest({"rules": active, "withheld": withheld})
    merged["refuted_rules"] = refuted
    merged["stage4_manifest"] = rules_to_stage4_manifest(active)
    merged["curated_rule_ids"] = [rule["rule_id"] for rule in active if rule["provenance"]["source"] == CREATOR_RULE_CURATED_SOURCE]
    return merged


__all__ = [
    "CREATOR_RULE_CURATED_SOURCE",
    "CURATED_STANDARD_RULES",
    "load_curated_standard",
    "merge_creator_rule_manifests",
    "validate_curated_standard",
    "CREATOR_RULE_CLASSES",
    "CREATOR_RULE_MANIFEST_CONTRACT",
    "CREATOR_RULE_MANIFEST_VERSION",
    "CREATOR_RULE_MIN_SUPPORT",
    "CREATOR_RULE_REFUTATION_LIMIT",
    "CREATOR_RULE_STAGE4_KEYS",
    "CREATOR_RULE_STATUSES",
    "CREATOR_RULE_VALIDATED_STATUSES",
    "CREATOR_RULE_WITHDRAWN_STATUSES",
    "creator_rule_id",
    "normalize_creator_rule",
    "normalize_creator_rule_manifest",
    "rules_to_stage4_manifest",
    "validate_creator_rule",
    "validate_creator_rule_manifest",
]
