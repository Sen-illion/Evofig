"""Stable contract for versioned prompt patches derived from measured experience.

A prompt patch is a directive a run injects into one stage's provider prompt. The wording is
the seat's: the reflection is asked which requirement a stage's prompt failed to state and
writes the sentence, because the thing that has to change is a sentence and a template cannot
know which one. The standing is not the seat's. A patch earns its rung from the measurement
behind the row it was rendered from -- an accepted round that measured this repair improving
the figure, or a refused one that measured it worsening the figure under a bound a machine can
check -- and a patch with no measurement behind it stays a candidate and is injected nowhere.
So a patch can still always be traced back to the case that justified it, and rolled back to
the version before it.

The five-level assembly order below is the authority for where a patch may sit relative
to the prompt it joins. Safety and pipeline boundaries outrank the paper's own evidence,
which outranks the stage's output schema and action whitelist, which outranks a promoted
patch, which outranks the reference-image prior. A patch can therefore never widen the
action whitelist or overrule the scientific contract.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .evolution import APPLY_AT_STAGES, EXPERIENCE_STATUSES


PROMPT_PATCH_CONTRACT = "rfs.prompt_patch"
PROMPT_PATCH_VERSION = 2
PROMPT_PATCH_CLASSES = ("advisory", "executable_geometry", "prohibited")
# ``current_case_validated`` is the rung a patch earns from the paper that produced it: the
# measurement behind it is this run's own round, so the rung is worth enough to be injected back
# into that run and not enough to reach a paper that measured nothing.
PROMPT_PATCH_LIFECYCLE = ("candidate", "current_case_validated", "same_domain_validated", "cross_domain_validated", "promoted")
PROMPT_PATCH_TERMINAL = ("refuted", "disabled")
#: How many rounds in a row a patch may be measured and lose before it stops being injected.
#: A win clears the streak, so the count is consecutive rather than a lifetime tally: a patch
#: that loses one round in three is still a patch that helps sometimes.
PROMPT_PATCH_REFUTATION_LIMIT = 2
PROMPT_PATCH_PARAMETERS = ("factor", "dx", "dy")
# The comparison an enforceable prohibition may state over a measured metric.
PROHIBITION_OPERATORS = ("<=", "<", ">=", ">", "==")
# Every numeric a patch may carry: the geometry trio plus the deterministic segmentation
# thresholds a parameter-only patch is allowed to move.
PATCH_PARAMETER_KEYS = PROMPT_PATCH_PARAMETERS + (
    "min_panel_occupancy",
    "max_cross_panel_connectors",
    "max_node_slot_fanout",
    "min_panel_area",
)

# Where a patch may sit relative to the prompt it joins, strongest first.
PROMPT_ASSEMBLY_ORDER = (
    "safety_and_pipeline_boundary",
    "paper_evidence_and_scientific_contract",
    "stage_output_schema_and_action_whitelist",
    "promoted_patches",
    "reference_image_prior",
)

#: A condition value that reads as nothing. A bucket the pipeline could not resolve is not a
#: bound: a patch whose condition quotes one claims to apply under a condition that never holds,
#: and a directive that quotes it reads as noise in the prompt it is injected into. Such entries
#: are dropped from the condition rather than stored, and a condition that arrives with one is
#: reported as a contract error.
UNREADABLE_CONDITION_VALUES = frozenset({"", "unknown", "none", "null", "n/a", "na", "-", "?"})


def reads_as_nothing(value: Any) -> bool:
    """Whether a condition value carries no information at all."""

    if value is None:
        return True
    if isinstance(value, str) and value.strip().casefold() in UNREADABLE_CONDITION_VALUES:
        return True
    return False


#: The subjects that name how the run judges, and the asks that would move them. A patch is a
#: sentence a stage prompt carries: it may say what to draw, and it may never say how the run
#: decides. A directive that names one of these subjects *and* makes one of these asks is refused
#: outright, so a seat cannot buy a pass for its own figure by rewriting the check that scores it.
PATCH_INTENT_SUBJECTS = ("gate", "verdict", "judgement", "judgment", "whitelist", "white list", "allow list", "acceptance")
PATCH_INTENT_ASKS = ("change", "ignore", "bypass", "skip", "disable", "loosen", "relax", "override", "extend", "weaken", "remove", "turn off", "rewrite")
#: A directive that names a gate only to forbid moving it is the opposite of the intent refused
#: here, and is left alone.
PATCH_INTENT_NEGATIONS = ("do not", "don't", "never", "must not", "without", "avoid", "no need to")


def disallowed_patch_intent(directive: Any) -> str:
    """Why a directive may not be filed, or "" when it says something a prompt may carry.

    The boundary the patch lifecycle defends is that a patch changes a sentence and never a rule of
    the run. That boundary held while the sentences were a template's; now a seat writes them, it
    has to be checked rather than assumed. A directive that both names the gate, the verdict, the
    whitelist or the acceptance *and* asks for it to be moved is refused; a directive that names
    one of them only to forbid moving it is a sentence a prompt may carry.
    """

    text = " ".join(str(directive or "").split()).casefold()
    if not text:
        return ""
    subject = next((word for word in PATCH_INTENT_SUBJECTS if word in text), "")
    if not subject:
        return ""
    ask = next((word for word in PATCH_INTENT_ASKS if word in text), "")
    if not ask:
        return ""
    if any(word in text for word in PATCH_INTENT_NEGATIONS):
        return ""
    return f"a patch may not ask to change how the run judges ({ask} ... {subject})"


def readable_condition(condition: Any) -> dict[str, Any]:
    """Drop the entries of a condition that read as nothing, keeping the rest untouched."""

    source = condition if isinstance(condition, dict) else {}
    kept: dict[str, Any] = {}
    for key, value in source.items():
        name = str(key or "").strip()
        if not name or reads_as_nothing(value):
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            items = [item for item in value if not reads_as_nothing(item)]
            if not items:
                continue
            kept[name] = items
            continue
        kept[name] = value
    return kept


PATCH_DIRECTIVE_TEMPLATES = {
    "advisory": "When {condition}, prefer {action}{parameters}.",
    "executable_geometry": "When {condition}, apply {action}{parameters} inside its recorded bounds.",
    # ``check_prohibited_patches`` requires the measured metric to satisfy the bound and
    # blocks the round when it does not, so the sentence has to read the same way: the action
    # is allowed inside the bound and nowhere else. The earlier wording said "never when",
    # which a reader would take as the opposite of what the checker enforces.
    "prohibited": "Only apply {action} when {condition}.",
}


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _number(value: Any) -> float | None:
    """Read a measured number, keeping "not measured" distinct from a measured zero."""

    if isinstance(value, bool):
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _compact(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


#: A defect pattern rendered as the thing a stage can recognise. The condition used to be four
#: coarse buckets, which could not tell a four-way fan-out crossing two panels from a tidy
#: pipeline that happened to have as many boxes; these read as the shape itself.
DEFECT_PHRASES = {
    "fanout_1_to_n": "one node feeds four or more others",
    "cross_panel": "the connectors cross panels",
    "edge_crossings": "edges run across panels they do not belong to",
    "dashed_cycle": "a feedback loop runs back on itself",
    "long_edge": "a connector spans half the canvas",
    "port_collision": "two connectors leave one node for the same panel",
}


def render_condition_summary(condition: Any) -> str:
    """Render a condition as a stable, sorted phrase a patch directive can quote."""

    source = readable_condition(condition)
    metric = str(source.get("metric") or "").strip()
    op = str(source.get("op") or "").strip()
    if metric and op in PROHIBITION_OPERATORS:
        return f"{metric} {op} {_compact(source.get('value'))}"
    parts = []
    patterns = [str(item) for item in (source.get("defect_patterns") or []) if str(item)]
    if patterns:
        parts.append(" and ".join(DEFECT_PHRASES.get(item, item) for item in patterns))
    parts.extend(f"{key}={_compact(source[key])}" for key in sorted(source) if key != "defect_patterns")
    return ", ".join(parts) if parts else "the recorded condition"


def is_enforceable_prohibition(condition: Any) -> bool:
    """Return whether a condition states a measurable bound this pipeline can check."""

    source = condition if isinstance(condition, dict) else {}
    if str(source.get("op") or "").strip() not in PROHIBITION_OPERATORS:
        return False
    if not str(source.get("metric") or "").strip():
        return False
    return _number(source.get("value")) is not None


def render_parameter_clause(action: Any) -> str:
    """Render the bounded parameters an experience fixed, in a fixed order."""

    source = action if isinstance(action, dict) else {}
    clauses = []
    for key in PROMPT_PATCH_PARAMETERS:
        value = _number(source.get(key))
        if value is not None:
            clauses.append(f"{key}={value:+}")
    return (" with " + ", ".join(clauses)) if clauses else ""


def render_patch_directive(*, patch_class: str, condition: Any, action: Any) -> str:
    """Render a patch directive from experience fields alone, so no model writes it."""

    source = action if isinstance(action, dict) else {}
    action_type = str(source.get("type") or source.get("action") or "the recorded repair").strip() or "the recorded repair"
    template = PATCH_DIRECTIVE_TEMPLATES.get(str(patch_class), PATCH_DIRECTIVE_TEMPLATES["advisory"])
    return template.format(
        condition=render_condition_summary(condition),
        action=action_type.replace("_", " "),
        parameters=render_parameter_clause(source),
    )


def prompt_patch_id(*, experience_ids: list[str], apply_at: str, patch_class: str, directive: str) -> str:
    """Derive a stable patch id from what the patch says, so a rerun cannot fork it.

    The version is deliberately not part of the id: a rerender that keeps the same claim
    and the same condition is a new version of the same patch, which is what lets a
    rollback disable a version while the patch keeps its history.
    """

    canonical = "|".join([apply_at, patch_class, directive, *sorted(experience_ids)])
    return "prompt-patch-" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def normalize_prompt_patch(raw: dict[str, Any] | None) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    provenance = source.get("provenance") if isinstance(source.get("provenance"), dict) else {}
    status = str(source.get("status") or "candidate")
    if status not in EXPERIENCE_STATUSES:
        status = "candidate"
    try:
        version = int(source.get("version") or PROMPT_PATCH_VERSION)
    except (TypeError, ValueError):
        version = PROMPT_PATCH_VERSION
    try:
        support = max(0, int(source.get("support") or 0))
    except (TypeError, ValueError):
        support = 0
    return {
        "contract": PROMPT_PATCH_CONTRACT,
        "version": version,
        "patch_id": str(source.get("patch_id") or ""),
        "apply_at": str(source.get("apply_at") or ""),
        "class": str(source.get("class") or "advisory"),
        "action_type": str(source.get("action_type") or ""),
        "condition": readable_condition(source.get("condition")),
        # The bounds the source experience was measured under travel with the patch: a case set
        # can only hold a counter-example for a patch that still says what it does not cover.
        "excluded_conditions": [dict(item) for item in (source.get("excluded_conditions") or []) if isinstance(item, dict)],
        "graph_scope": dict(source.get("graph_scope") or {}) if isinstance(source.get("graph_scope"), dict) else {},
        "directive": str(source.get("directive") or ""),
        "parameters": {key: _number((source.get("parameters") or {}).get(key)) for key in PATCH_PARAMETER_KEYS} if isinstance(source.get("parameters"), dict) else {},
        "source_experience_ids": _list_of_strings(source.get("source_experience_ids")),
        "support": support,
        "median_repair_gain": _number(source.get("median_repair_gain")),
        # Who wrote the wording: the seat that measured the case, or the template that stands in
        # when the seat wrote nothing. The rung is earned by the measurement either way; this field
        # only says which of the two the sentence came from.
        "written_by": str(source.get("written_by") or "template"),
        "status": status,
        "provenance": {
            "source": str(provenance.get("source") or ""),
            "run": str(provenance.get("run") or ""),
            "measured": bool(provenance.get("measured")),
            "round": provenance.get("round"),
        },
    }


def validate_prompt_patch(patch: dict[str, Any]) -> dict[str, Any]:
    """Check one patch against the contract, returning the normalized patch either way."""

    normalized = normalize_prompt_patch(patch)
    errors: list[str] = []
    if not normalized["patch_id"]:
        errors.append("patch_id is required")
    if normalized["version"] < 1:
        errors.append("patch version must be a positive integer")
    if normalized["apply_at"] not in APPLY_AT_STAGES:
        errors.append("patch apply_at stage is unsupported")
    if normalized["class"] not in PROMPT_PATCH_CLASSES:
        errors.append("patch class is unsupported")
    if not normalized["directive"]:
        errors.append("patch directive is required")
    intent_problem = disallowed_patch_intent(normalized["directive"])
    if intent_problem:
        errors.append(intent_problem)
    if not normalized["source_experience_ids"]:
        errors.append("a patch must name the experience(s) it was rendered from")
    if normalized["median_repair_gain"] is None:
        errors.append("patch median_repair_gain must be a measured number")
    provenance = normalized["provenance"]
    if not provenance["measured"]:
        errors.append("patch provenance must record a measurement, not an expectation")
    if not provenance["source"]:
        errors.append("patch provenance must name the source that measured it")
    if normalized["class"] == "prohibited" and (normalized["median_repair_gain"] or 0.0) > 0:
        errors.append("a prohibited patch cannot record a positive median_repair_gain")
    if normalized["class"] == "prohibited" and not is_enforceable_prohibition(normalized["condition"]):
        errors.append("a prohibited patch must state a measurable condition (metric, op and value)")
    supplied = patch.get("condition") if isinstance(patch.get("condition"), dict) else {}
    dropped = sorted(str(key) for key in supplied if str(key) not in normalized["condition"])
    if dropped:
        # The normalisation drops these silently, which is right for a condition the builder
        # assembled from a measurement and wrong for one that arrived from outside: a caller
        # writing the contract directly has to hear that the bound it stated holds nowhere.
        errors.append("patch condition states a value that reads as nothing: " + ", ".join(dropped))
    return {"ok": not errors, "errors": errors, "patch": normalized}


__all__ = [
    "PATCH_PARAMETER_KEYS",
    "UNREADABLE_CONDITION_VALUES",
    "PATCH_INTENT_ASKS",
    "PATCH_INTENT_NEGATIONS",
    "PATCH_INTENT_SUBJECTS",
    "disallowed_patch_intent",
    "readable_condition",
    "reads_as_nothing",
    "PATCH_DIRECTIVE_TEMPLATES",
    "PROHIBITION_OPERATORS",
    "PROMPT_ASSEMBLY_ORDER",
    "PROMPT_PATCH_CLASSES",
    "PROMPT_PATCH_CONTRACT",
    "PROMPT_PATCH_LIFECYCLE",
    "PROMPT_PATCH_PARAMETERS",
    "PROMPT_PATCH_REFUTATION_LIMIT",
    "PROMPT_PATCH_TERMINAL",
    "PROMPT_PATCH_VERSION",
    "normalize_prompt_patch",
    "is_enforceable_prohibition",
    "prompt_patch_id",
    "render_condition_summary",
    "render_parameter_clause",
    "render_patch_directive",
    "validate_prompt_patch",
]
