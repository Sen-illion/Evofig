"""Stable contracts for direct paper-to-PPTX self-evolution."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .graphs import MOTIF_KINDS


EVOLUTION_CONTRACT_VERSION = "1.0"
#: The reflection's own contract version. Splitting ``outcome`` from the defects a round
#: cleared and introduced changed what a reflection says, so a reader can tell a row written
#: before the split from one written after it. A row that predates the split carries no
#: ``defects`` and is read as such rather than migrated.
REFLECTION_CONTRACT_VERSION = "1.2"

#: The two defect lists a reflection separates: what the round fixed and what it broke.
DEFECT_KEYS = ("cleared", "introduced")
#: How a round that lost explains itself. A rule that was carried out badly and a rule that
#: was wrong need two different answers, and only the second one is a lesson about how figures
#: should be drawn. The empty string is "the reflection said nothing", which is not the same as
#: any answer here and never earns another attempt.
ATTRIBUTION_VALUES = ("rule_is_wrong", "not_carried_out", "local_side_effect", "unmeasurable")
EXPERIENCE_SCHEMA_VERSION = "3.7"
#: The source marker for a score that was looked up in the defect table rather than
#: measured from the figure. The acceptance seat's pair answer names a defect, and the
#: defect table turns that name into a row of numbers; the same row is written onto
#: whichever figure was blamed, so the number says which defect was named, not how good
#: the figure is. A score carrying this marker is recorded and never decides a round, a
#: rollback or a ranking.
DERIVED_SCORE_SOURCE = "derived_from_defect"
MEMORY_TYPES = {
    "scientific",
    "layout",
    "segmentation",
    "connector",
    "theme",
    "icon",
    "density",
    "repair",
    # The author's own trade-off: which of the paper's concepts a figure draws and
    # which it leaves to prose. It applies at global_planning, where a figure's
    # content is chosen - the one stage a selection rule can act at.
    "selection",
}
#: The stages a repair action may apply at, in the order the pipeline runs them. Written once
#: so a planning prompt, a stage trace and an action table cannot disagree about the vocabulary.
#: The first two are not stages the loop repairs in place -- no pass here can put back a module
#: the reading never extracted, or ask for a figure the content plan never chose. They are in the
#: vocabulary because the seat is asked *where a defect entered*, and every content-level defect
#: enters there. Without the names the only honest answer to "the figure is missing half the
#: paper's modules" was "nowhere in particular", which reads to the loop exactly like a geometry
#: defect and is answered with a nudge to the figure already drawn.
STAGE_ORDER = (
    "paper_reading",
    "content_planning",
    "global_planning",
    "segmentation",
    "theme_planning",
    "icon_generation",
    "ppt_compilation",
    "repair",
)
APPLY_AT_STAGES = set(STAGE_ORDER)
EXPERIENCE_STATUSES = {
    "candidate",
    "current_case_validated",
    # One matched paper was measured under the quick profile and preferred this row. The rung is
    # its own on purpose: a row kept on a single pair must never be read as one the four-group
    # ladder promoted, and only a caller that named the quick profile may draw from it.
    "quick_validated",
    "same_domain_validated",
    "cross_domain_validated",
    "domain_validated",
    "promoted",
    "retrying",
    "overfit_or_failed",
    "disabled",
    "refuted",
    # Paired ON/OFF ran, but both arms executed the same move: not a rung, not a failure.
    "inconclusive",
    # The move the row names is the one the run would have made without any Memory: a fact about
    # the shipped default, not a finding, so it is recorded and never retrieved.
    "common_knowledge",
    # The pair was put to the Judge more than once and the readings disagreed: the case was not
    # measured, which is neither a pass nor a failure.
    "measurement_noise",
    # A row the ladder cannot examine because it never said where it applies. It is not left to
    # read as a plain ``candidate`` forever: the dead end is named, so a library that cannot
    # grow is explainable from the library itself.
    "unverifiable",
}
SCOPE_TYPES = {"global", "slot", "interface", "connector", "compile", "semantic"}
# Where an experience may be reused. This is not ``scope``, which names the part of the
# figure a repair touched. An empty value means no rule wrote one, so the row can be
# saved and cannot be promoted.
APPLICABILITY_VALUES = {"common", "domain"}
MISSING_PAPER_DOMAINS = {"", "unknown", "missing", "direct_editable"}
REPAIR_ACTIONS = {
    "reanalyze_semantics",
    "replan_structure",
    "replan_theme",
    "resegment",
    "relayout_global",
    "relayout_slots",
    "resize_nodes",
    "reroute",
    "repair_interfaces",
    "regen_assets",
    # The two mechanism-level rules. They name no object of the figure in front of them: a rule
    # rides on the action and the compile that draws a later figure honours it. They are in this
    # set because this set is the whitelist rendered into the advisory Judge's prompt, and a rule
    # the Judge cannot name is a channel with no way in -- which is how both stayed dead while
    # the executor, the stamp and the validators below already knew them. ``SUPPORTED_ACTIONS``
    # stays separate so ``action_is_repair`` keeps meaning "the Judge named a repair".
    "router_constraint",
    "layout_constraint",
}
# ``recompile`` was the eleventh action and is deliberately gone. Every round compiles its
# candidate anyway, and the compile is deterministic in the state it is handed, so asking for
# one over unchanged state re-emits the same figure by construction. It cost a round and a
# retry to reach a refusal that was knowable before the round started. This set is also the
# whitelist rendered into the Judge prompt, so keeping the name there taught the model to ask
# for a repair that cannot exist. A stored experience or a sealed run that still names it
# stays readable -- the contract is only checked when a new experience is written.

#: Actions that state a rule the next figure is drawn by, rather than a move on this one. A
#: Memory row can carry one, and the compile that draws the next paper honours it, so an
#: experience can say "route a one-to-many fan-out through a shared channel" and a figure it has
#: never seen obeys. A rule carries parameters and never object ids, so the name alone is not a
#: prescription: both readers that accept one refuse a name that arrived without its parameters.
MECHANISM_ACTIONS = {
    "router_constraint",
    "layout_constraint",
}
#: The record a compile-time rule leaves behind: what the compile applied and what it could
#: not reach. A reflection asked why a rule failed reads its attribution against this.
MECHANISM_RULE_REPORT_CONTRACT = "rfs.mechanism_rule_report"
#: The record a move on the figure itself leaves behind: whether the figure changed and which
#: objects it changed. It answers the same two attributions the compile report answers, so a
#: losing suggestion is refused for a reason a round can be read against either way.
FIGURE_MOVE_EVIDENCE_CONTRACT = "rfs.figure_move_evidence"
#: Every action a new experience may name. ``SUPPORTED_ACTIONS`` in the executor is the narrower
#: set that keeps ``action_is_repair`` meaning "the Judge named a repair on this figure".
EXPERIENCE_ACTIONS = REPAIR_ACTIONS | MECHANISM_ACTIONS

#: The loop asks two different questions, so one Judge role answers exactly one of them.
#: The advisory role names the defect and the bounded repair it wants; the acceptance role
#: scores the figure. Everything downstream reads the family, never the serving name.
ADVISORY_ROLE_FAMILY = "advisory"
ACCEPTANCE_ROLE_FAMILY = "acceptance"


def _list_of_strings(value: Any) -> list[str]:
    return [str(item) for item in value if str(item).strip()] if isinstance(value, list) else []


def _optional_number(value: Any) -> float | None:
    """Read a measured number, keeping "not measured" distinct from a measured zero."""

    if isinstance(value, bool):
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _optional_score(value: Any) -> float | None:
    """Read a Judge score in 0..1, keeping "not measured" distinct from a measured zero."""

    if isinstance(value, bool):
        return None
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return None


def judge_role_family(role: Any = "", *, frozen: bool = False) -> str:
    """Return which of the two questions a Judge role answers.

    The advisory role names the defect and the bounded repair it wants; the acceptance role
    scores the figure. A role is only allowed to answer its own question, so the rest of the
    pipeline reads this family rather than the serving role name.
    """

    name = str(role or "").strip().lower()
    if frozen or name.startswith("frozen") or name == ACCEPTANCE_ROLE_FAMILY:
        return ACCEPTANCE_ROLE_FAMILY
    return ADVISORY_ROLE_FAMILY


def normalize_judge_fingerprint(raw: Any) -> dict[str, Any]:
    """Normalize the Judge identity a Memory row was measured under.

    ``checkpoint`` names the weights; ``model`` names the endpoint that served them. Rows
    written before the registry binding existed carry no checkpoint and keep reading as the
    model they named.
    """

    source = raw if isinstance(raw, dict) else {}
    return {
        "role": str(source.get("role") or ""),
        "mode": str(source.get("mode") or ""),
        "trained": bool(source.get("trained")),
        "model": str(source.get("model") or ""),
        "checkpoint": str(source.get("checkpoint") or ""),
    }


def judge_fingerprint_identity(raw: Any) -> str:
    """Return the comparable identity behind a fingerprint.

    The checkpoint answers first, because the served model name does not change when the
    weights behind it do: a retrained Judge is normally published at the endpoint name the
    old one used, so a fingerprint built from that name reported "same Judge" across exactly
    the handover the sweep exists to catch. A row that names no checkpoint falls back to its
    model and then to its mode, so "recorded no fingerprint at all" is its own identity
    rather than a match for anything.
    """

    fingerprint = normalize_judge_fingerprint(raw)
    judge = "trained" if fingerprint["trained"] else "bootstrap"
    return f"{judge}:{fingerprint['checkpoint'] or fingerprint['model'] or fingerprint['mode'] or 'unknown'}"


def same_judge(left: Any, right: Any) -> bool:
    """Report whether two rows were measured under the same Judge."""

    return judge_fingerprint_identity(left) == judge_fingerprint_identity(right)


def _judge_actions(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the bounded repair actions one feedback entry asks for."""

    from .repair_actions import action_missing_required_targets, collected_target_ids

    actions = item.get("requested_actions") or item.get("repair") or []
    normalized: list[dict[str, Any]] = []
    for action in actions if isinstance(actions, list) else []:
        record = action if isinstance(action, dict) else {"action": action}
        action_name = str(record.get("action") or record.get("type") or "")
        if action_name not in REPAIR_ACTIONS:
            continue
        if action_missing_required_targets(action_name, record, item):
            continue
        ids = collected_target_ids(action_name, record, item)
        filled = {**record, "action": action_name}
        if ids:
            keys = ("semantic_node_ids", "object_ids", "slot_ids", "edge_ids")
            for key in keys:
                values = [str(item_id) for item_id in (record.get(key) or item.get(key) or []) if str(item_id).strip()]
                if values:
                    filled[key] = list(dict.fromkeys(values))
        normalized.append(filled)
    return normalized


#: The second question the loop asks about a round: it shows the two states together and asks
#: which one is better, because that is the question a model answers consistently. Asking for an
#: absolute number is not: of 74 numbers the served model produced in our own runs, every one but
#: a single 0.65 sat on a 0.1 grid, so the scale was never the limit -- the judgment is. The
#: choice vocabulary matches the training contract (``preferred_candidate`` in
#: :mod:`rfs.judge_training.judge_schema`), so the same answer can be supervised later.
PAIRWISE_CHOICES = ("A", "B", "TIE")
PAIRWISE_MARGINS = ("none", "slight", "clear")


def _pairwise_choice(value: Any) -> str:
    """Read one candidate name, accepting the words a model uses for a tie."""

    text = str(value or "").strip().upper()
    if text in {"TIE", "SAME", "EQUAL", "EQUIVALENT"}:
        return "TIE"
    return text if text in {"A", "B"} else ""


def _pairwise_margin(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return text if text in PAIRWISE_MARGINS else "none"


def normalize_pairwise_result(raw: dict[str, Any] | None, *, role: str = "", candidate_a: str = "", candidate_b: str = "") -> dict[str, Any]:
    """Normalize a two-state answer into the shared Judge result shape.

    The two states are always shown in the same order: ``A`` is the state the loop already
    accepted (the control), ``B`` is the state it just produced. The record keeps that mapping
    next to the answer, because "B was preferred" only means something to a reader who knows
    which state B was. A comparison measures neither state, so it carries no score at all.
    """

    source = raw if isinstance(raw, dict) else {}
    choice = _pairwise_choice(source.get("preferred_candidate")) or _pairwise_choice(source.get("pairwise_choice"))
    issues = [str(item) for item in source.get("blocking_issues") or [] if str(item).strip()] if isinstance(source.get("blocking_issues"), list) else []
    confidence = source.get("confidence")
    # A result that was normalized once already carries both names inside its own block, so the
    # validator can re-read a stored record without being handed the mapping again.
    stored = source.get("pairwise") if isinstance(source.get("pairwise"), dict) else {}
    pair = {
        "choice": choice,
        "margin": _pairwise_margin(source.get("margin")),
        "confidence": float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else 0.0,
        "candidate_a": str(candidate_a or source.get("candidate_a") or stored.get("candidate_a") or ""),
        "candidate_b": str(candidate_b or source.get("candidate_b") or stored.get("candidate_b") or ""),
    }
    return {
        **source,
        "contract": "rfs.judge_result",
        "version": EVOLUTION_CONTRACT_VERSION,
        "judgement_kind": "pairwise",
        "judge_role": str(role or source.get("judge_role") or ""),
        "judge_role_family": judge_role_family(role or source.get("judge_role"), frozen=bool(source.get("frozen"))),
        "preferred_candidate": choice,
        "pairwise": pair,
        "overall_score": None,
        "score_measured": False,
        "reported_score": None,
        "reported_actions": [],
        "dimension_scores": {},
        "status": str(source.get("status") or "fault"),
        "stage_feedback": [],
        "repair": [],
        "preserve": [],
        "blocking_issues": issues,
        "blocking_issue_count": len(issues),
    }


def validate_pairwise_result(result: dict[str, Any] | None) -> dict[str, Any]:
    """Check one comparison names both states it saw and answers with the vocabulary."""

    normalized = normalize_pairwise_result(result, role=str((result or {}).get("judge_role") or ""))
    pair = normalized["pairwise"]
    errors: list[str] = []
    if not pair["choice"]:
        errors.append("a comparison must prefer A, B or call it a tie")
    if pair["margin"] not in PAIRWISE_MARGINS:
        errors.append("margin must be none, slight, or clear")
    if not pair["candidate_a"] or not pair["candidate_b"]:
        errors.append("a comparison must name both states it compared")
    return {"ok": not errors, "errors": errors, "result": normalized}


def pairwise_verdict(pairwise: dict[str, Any] | None) -> str:
    """Read which state a comparison preferred: ``after``, ``before``, ``tie``, or nothing.

    ``B`` is always the state the loop just produced, so a comparison that prefers B is the
    only one that may be accepted as an improvement.
    """

    source = pairwise if isinstance(pairwise, dict) else {}
    if source.get("judgement_kind") != "pairwise":
        return ""
    block = source.get("pairwise") if isinstance(source.get("pairwise"), dict) else {}
    return {"B": "after", "A": "before", "TIE": "tie"}.get(_pairwise_choice(block.get("choice")), "")


def deterministic_blocker_ids(feedback: Any) -> list[str]:
    """Name the defect kinds a deterministic check found, without its measured numbers.

    The gate writes its measurements into the sentence: "Icon/text overlap=3, boundary
    violations=0". Two readings of one defect therefore never spell it the same way, and a
    repair that lowered the overlap from three to two would read as a different defect
    entirely. The stage the check belongs to and the repair it asks for do not move with the
    measurement, and no two of the gate's checks share that pair, so the two together name
    the defect and nothing else.
    """

    found: list[str] = []
    for item in feedback if isinstance(feedback, list) else []:
        if not isinstance(item, dict) or str(item.get("severity") or "") != "blocker":
            continue
        identity = _defect_identity(item)
        if identity and identity not in found:
            found.append(identity)
    return found


def _defect_identity(item: dict[str, Any]) -> str:
    """Name one deterministic defect by the check that found it, never by its prescription.

    ``stage:action`` was the old identity, and the action is the prescription: the same
    geometry break was answered with ``reroute``, ``relayout_slots`` or ``relayout_global``
    depending on which ids happened to be on hand, so one defect read as three and a repair
    that changed its mind looked like a defect that had just appeared. The gate now names the
    check itself (``interface:box_pass``) and that name is what identity means; a record
    written before the split carries only the prescription, and is read the old way rather
    than dropped.
    """

    stated = str(item.get("defect_key") or "").strip()
    if stated:
        return stated
    stage = str(item.get("stage") or item.get("scope_type") or item.get("scope") or "")
    action = ""
    for candidate in _judge_actions(item):
        action = str(candidate.get("action") or "")
        if action:
            break
    identity = f"{stage}:{action}"
    return "" if identity == ":" else identity


def deterministic_blocker_counts(feedback: Any) -> dict[str, int]:
    """How many times each named defect kind a rerun would find again was found.

    The comparison between two answers needs the counts, not only the names: a defect that
    fell from three occurrences to two is not a new defect, and reading only the set of names
    made a repair that reduced it look like one that introduced it.
    """

    counts: dict[str, int] = {}
    for item in feedback if isinstance(feedback, list) else []:
        if not isinstance(item, dict) or str(item.get("severity") or "") != "blocker":
            continue
        identity = _defect_identity(item)
        if identity:
            counts[identity] = counts.get(identity, 0) + 1
    return counts


def _separated_counts(result: Any) -> dict[str, int] | None:
    """How many times each reproducible defect kind was found, or None when unseparated.

    An empty mapping is "this answer separated nothing", not "this answer found nothing": the
    two readings are only comparable when both sides carried counts, and a side that answered
    with an empty map is a side that did not report them. Reading the empty map as a zero is
    how a round that plainly introduced a defect came out as one that introduced none.
    """

    value = (result or {}).get("deterministic_blocker_counts") if isinstance(result, dict) else None
    if not isinstance(value, dict) or not value:
        return None
    counts = {
        str(key): int(count)
        for key, count in value.items()
        if isinstance(count, (int, float)) and not isinstance(count, bool)
    }
    return counts or None


def _separated_ids(result: Any) -> list[str] | None:
    """The defect kinds a rerun of the same figure would find again, or None if unseparated."""

    value = (result or {}).get("deterministic_blockers") if isinstance(result, dict) else None
    if not isinstance(value, list):
        return None
    return [str(item) for item in value if str(item).strip()]


def _blocker_tally(result: Any) -> int:
    """How many defects the answer listed, counted the one way every writer counts them."""

    if not isinstance(result, dict):
        return 0
    value = result.get("blocking_issue_count")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    feedback = result.get("stage_feedback") or []
    return sum(
        1
        for item in feedback
        if isinstance(item, dict) and str(item.get("severity") or "") == "blocker"
    )


def introduced_defects(before: Any, after: Any) -> dict[str, Any]:
    """Name the defects one answer introduced over an earlier one, and say what that rests on.

    The round decision and the library both have to answer "what did this round introduce", and
    they used to answer it with two different instruments: one compared how many times each kind
    of defect was found, the other compared the set of defect names. Two rulers for one question
    meant a round could be accepted by one and refused by the other, and the two readings are
    kept in one place here so they cannot drift apart again.

    The rule is fixed, in this order:

    * both answers carried counts -> a defect kind is introduced when its count rose;
    * only one of them (or neither) carried counts, but both named their defects -> the kinds
      the later answer names and the earlier one does not are the ones introduced;
    * neither answer separated its defects at all -> only the size of the two lists can be
      compared, and the count is the difference.

    ``basis`` says which of the three was used, so a reader can tell a measurement that rested
    on counts from one that rested on names alone.
    """

    before_counts, after_counts = _separated_counts(before), _separated_counts(after)
    if before_counts is not None and after_counts is not None:
        defects = [
            str(key)
            for key, count in sorted(after_counts.items())
            if count > int(before_counts.get(str(key), 0))
        ]
        return {"defects": defects, "count": len(defects), "basis": "counts"}
    before_ids, after_ids = _separated_ids(before), _separated_ids(after)
    if before_ids is not None and after_ids is not None:
        known = set(before_ids)
        defects = [item for item in dict.fromkeys(after_ids) if item not in known]
        return {"defects": defects, "count": len(defects), "basis": "names"}
    return {
        "defects": [],
        "count": max(0, _blocker_tally(after) - _blocker_tally(before)),
        "basis": "totals",
    }


#: How a list index is spelled inside a field name, so a claim's ``feedback[0]`` and the
#: payload's own ``feedback[3]`` read as the same field instead of as two different ones.
_INDEXED = re.compile(r"\[\d*\]")


def normalize_field_name(name: Any) -> str:
    """A field name with its list indices folded together and its surrounding dots trimmed."""

    return _INDEXED.sub("[]", str(name or "")).strip().strip(".")


def citable_fields(payload: Any) -> set[str]:
    """Every field name a payload put in front of the seat: its dotted paths and its leaf keys.

    The seat is shown more than one block -- the trace's own numbers, the colour report it
    complains about, the grounding Gate's coverages, the advice the library offered -- and it
    cites them by the name it read: ``color_report.feedback[0].reason`` with its path, or
    ``edge_coverage`` without its parent. Holding only one block's keys meant a claim about a
    field the round really measured was thrown away as unreadable. A name the payload carries is
    checkable; a name no payload ever produced still is not, which is what the gate is for.

    The walker lives here rather than beside the gate that reads it because the two callers that
    need it are in different layers: the workflow checks a claim against what a round measured,
    and the seat records which fields the prompt handed it so its own answer can be checked
    against exactly those.
    """

    paths: set[str] = set()
    leaves: set[str] = set()

    def walk(value: Any, path: str) -> None:
        # The path is taken before descending, so a scalar keeps the name it was read under: an
        # earlier version added a path only on the way out of a container and silently dropped
        # every leaf that was not itself a container, which is why ``grounding_gate.checks.
        # required_edges_grounded`` -- the very field one live round cited -- was unreadable.
        if path:
            paths.add(path)
        if isinstance(value, dict):
            for key, item in value.items():
                name = str(key)
                leaves.add(name)
                walk(item, f"{path}.{name}" if path else name)
        elif isinstance(value, list):
            for item in value:
                walk(item, f"{path}[]" if path else "[]")

    walk(payload, "")
    return {normalize_field_name(name) for name in paths | leaves if str(name).strip()}


def normalize_judge_result(raw: dict[str, Any] | None, *, role: str = "") -> dict[str, Any]:
    """Normalize heuristic, bootstrap, and trained Judge outputs for one role.

    A role is only allowed to answer its own question, so the other role answer is kept as
    a reported value and never becomes the loop measurement: an advisory result carries no
    score (any number it volunteered stays in ``reported_score``), and an acceptance result
    carries no repair action (those stay in ``reported_actions``). A missing score is "not
    measured" -- ``score_measured`` false and ``overall_score`` null -- never a fabricated
    zero, because a zero is a real verdict and must not be invented.
    """

    source = raw if isinstance(raw, dict) else {}
    family = judge_role_family(role or source.get("judge_role"), frozen=bool(source.get("frozen")))
    advisory = family == ADVISORY_ROLE_FAMILY
    feedback = []
    reported_actions = [dict(item) for item in source.get("reported_actions") or [] if isinstance(item, dict)] if isinstance(source.get("reported_actions"), list) else []
    for item in source.get("stage_feedback", []):
        if not isinstance(item, dict):
            continue
        scope = str(item.get("scope_type") or item.get("scope") or item.get("stage") or "global")
        if scope not in SCOPE_TYPES:
            scope = "global"
        actions = _judge_actions(item)
        if not advisory:
            reported_actions.extend(actions)
        feedback.append(
            {
                **item,
                "scope_type": scope,
                "semantic_node_ids": _list_of_strings(item.get("semantic_node_ids")),
                "slot_ids": _list_of_strings(item.get("slot_ids")),
                "object_ids": _list_of_strings(item.get("object_ids")),
                "edge_ids": _list_of_strings(item.get("edge_ids")),
                "problem": str(item.get("problem") or ""),
                "evidence": item.get("evidence") if isinstance(item.get("evidence"), list) else [],
                "expected_effect": str(item.get("expected_effect") or ""),
                "forbidden_changes": _list_of_strings(item.get("forbidden_changes")),
                "requested_actions": actions if advisory else [],
            }
        )
    # The status the seat itself gave, read before the loose repairs below are folded in:
    # a repair named outside a stage block is a repair, not a verdict, and the fallback
    # ("no feedback means pass") must keep counting the entries the answer wrote.
    stated_status = str(source.get("status") or ("pass" if not feedback else "fault"))
    # The advisory menu is one flat list of bounded actions, so a seat may write it as one
    # flat list: the repairs go at the top level of the answer and ``stage_feedback`` carries
    # only the prose. Reading the nested spelling alone dropped them -- the repair was in the
    # answer and the loop ran no move, which is how a round the seat had already said how to
    # fix ended with nothing to do. The loose list becomes one more entry, so the picker, the
    # review's requested actions and the fork stage read it without a second spelling.
    if advisory:
        known = {
            json.dumps(action, sort_keys=True)
            for entry in feedback
            for action in entry["requested_actions"]
        }
        loose = [
            action
            for action in _judge_actions(
                {
                    "requested_actions": source.get("requested_actions") or source.get("repair"),
                    "problem": source.get("problem"),
                    "expected_effect": source.get("expected_effect"),
                    "forbidden_changes": source.get("forbidden_changes"),
                    "semantic_node_ids": source.get("semantic_node_ids"),
                    "object_ids": source.get("object_ids"),
                    "slot_ids": source.get("slot_ids"),
                    "edge_ids": source.get("edge_ids"),
                }
            )
            if json.dumps(action, sort_keys=True) not in known
        ]
        if loose:
            loose_scope = str(source.get("scope_type") or source.get("scope") or "global")
            feedback.append(
                {
                    "scope_type": loose_scope if loose_scope in SCOPE_TYPES else "global",
                    "semantic_node_ids": _list_of_strings(source.get("semantic_node_ids")),
                    "slot_ids": _list_of_strings(source.get("slot_ids")),
                    "object_ids": _list_of_strings(source.get("object_ids")),
                    "edge_ids": _list_of_strings(source.get("edge_ids")),
                    # A validator refuses an entry with no problem, and a repair the seat
                    # only named loosely still has to read as one, so the sentence says
                    # where it came from rather than being left empty.
                    "problem": str(source.get("problem") or "").strip()
                    or "the answer named this repair for the figure as a whole",
                    "evidence": [],
                    "expected_effect": str(source.get("expected_effect") or ""),
                    "forbidden_changes": _list_of_strings(source.get("forbidden_changes")),
                    "requested_actions": loose,
                    "stated_outside_a_stage": True,
                }
            )
    reported_score = _optional_score(source.get("overall_score"))
    if reported_score is None:
        reported_score = _optional_score(source.get("reported_score"))
    # A score the seat looked up in the defect table is derived from the defect it named,
    # not measured from the figure, so it is labelled and never counted as a measurement.
    derived = str(source.get("score_source") or "") == DERIVED_SCORE_SOURCE
    score_measured = bool(not advisory and reported_score is not None and not derived)
    dimensions = source.get("dimension_scores") if isinstance(source.get("dimension_scores"), dict) else {}
    blocking_issues = source.get("blocking_issues") if isinstance(source.get("blocking_issues"), list) else []
    # One writer for the blocker tally: the controller and the validators used to count it
    # differently, so "no new blocking issue" could hold while the Judge listed blockers.
    blocking_issue_count = len(blocking_issues) + sum(1 for entry in feedback if str(entry.get("severity") or "") == "blocker")
    # The advisory answer names where the lesson it just proposed should be examined.
    # An acceptance answer does not get to choose that, and a word outside the two
    # allowed values is dropped rather than stored as a guess.
    applicability = _applicability_value(source.get("applicability")) if advisory else ""
    return {
        **source,
        "contract": "rfs.judge_result",
        "version": EVOLUTION_CONTRACT_VERSION,
        "judge_role_family": family,
        "overall_score": reported_score if score_measured else None,
        "score_measured": score_measured,
        "reported_score": reported_score,
        "reported_actions": reported_actions,
        "dimension_scores": dimensions if not advisory else {},
        "status": stated_status,
        "stage_feedback": feedback,
        "repair": [action for item in feedback for action in item.get("requested_actions", [])],
        "preserve": source.get("preserve") if isinstance(source.get("preserve"), list) else [],
        "blocking_issues": blocking_issues,
        "blocking_issue_count": blocking_issue_count,
        "applicability": applicability,
        # Which of the blockers above a rerun of the same figure would find again. The model
        # rewrites its own list on every call, so only this part of it can be compared across
        # two answers; a result that never separated the two carries an empty list and the
        # reader falls back to the tally.
        "deterministic_blockers": _list_of_strings(source.get("deterministic_blockers")),
        "deterministic_blocker_counts": {
            str(key): int(value)
            for key, value in (source.get("deterministic_blocker_counts") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        } if isinstance(source.get("deterministic_blocker_counts"), dict) else {},
        # The deterministic gate's own reading of this same figure, on the one ruler every
        # state is measured by. The pair question answers a preference and its relative
        # number -- 1.0 for whichever side was not blamed -- is not a measurement of either
        # figure, so it is never allowed to stand in for this.
        "absolute_score": _optional_score(source.get("absolute_score")),
        "confidence": float(source.get("confidence") or 0.0),
    }


def validate_judge_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_judge_result(result)
    errors: list[str] = []
    if normalized["status"] not in {"pass", "fault", "blocked"}:
        errors.append("Judge status must be pass, fault, or blocked")
    for index, item in enumerate(normalized["stage_feedback"]):
        if not item["problem"]:
            errors.append(f"stage_feedback[{index}] requires problem")
        if not any(item[key] for key in ("semantic_node_ids", "slot_ids", "object_ids", "edge_ids")) and item["scope_type"] not in {"global", "compile", "semantic"}:
            errors.append(f"stage_feedback[{index}] requires object binding")
    return {"ok": not errors, "errors": errors, "result": normalized}


def _applicability_value(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return text if text in APPLICABILITY_VALUES else ""


def _paper_domain_missing(value: Any) -> bool:
    return str(value or "").strip().casefold() in MISSING_PAPER_DOMAINS


def _applicability_domain(applicability: Any, domain: Any) -> str:
    if _applicability_value(applicability) != "domain":
        return ""
    text = str(domain or "").strip()
    return "" if _paper_domain_missing(text) else text


def normalize_experience(raw: dict[str, Any] | None) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    status = str(source.get("status") or "candidate")
    if status not in EXPERIENCE_STATUSES:
        status = "candidate"
    graph_scope_source = source.get("graph_scope") if isinstance(source.get("graph_scope"), dict) else {}
    graph_scope = {
        "graph_ids": _list_of_strings(graph_scope_source.get("graph_ids")),
        "subgraph_ids": _list_of_strings(graph_scope_source.get("subgraph_ids")),
        "scientific_node_ids": _list_of_strings(graph_scope_source.get("scientific_node_ids")),
        "edge_ids": _list_of_strings(graph_scope_source.get("edge_ids")),
        "slot_ids": _list_of_strings(graph_scope_source.get("slot_ids")),
        "object_ids": _list_of_strings(graph_scope_source.get("object_ids")),
    }
    provenance_source = source.get("provenance") if isinstance(source.get("provenance"), dict) else {}
    return {
        **source,
        "contract": "rfs.evolution_experience",
        "version": EVOLUTION_CONTRACT_VERSION,
        "schema_version": EXPERIENCE_SCHEMA_VERSION,
        "experience_id": str(source.get("experience_id") or ""),
        "domain": str(source.get("domain") or "general"),
        "scope": str(source.get("scope") or "global"),
        "applicability": _applicability_value(source.get("applicability")),
        "applicability_domain": _applicability_domain(source.get("applicability"), source.get("applicability_domain")),
        "applicability_reason": str(source.get("applicability_reason") or ""),
        "memory_type": str(source.get("memory_type") or "repair"),
        "apply_at": str(source.get("apply_at") or "repair"),
        "problem_type": str(source.get("problem_type") or ""),
        "trigger": source.get("trigger") if isinstance(source.get("trigger"), dict) else {},
        "required_motifs": _list_of_strings(source.get("required_motifs")),
        "preserve": _list_of_strings(source.get("preserve")),
        "forbidden_changes": _list_of_strings(source.get("forbidden_changes")),
        "judge_fingerprint": normalize_judge_fingerprint(source.get("judge_fingerprint")),
        "condition": source.get("condition") if isinstance(source.get("condition"), dict) else {},
        "hypothesis": str(source.get("hypothesis") or ""),
        "action": source.get("action") if isinstance(source.get("action"), dict) else {},
        "expected_effect": source.get("expected_effect") if isinstance(source.get("expected_effect"), dict) else {},
        "validated_cases": _list_of_strings(source.get("validated_cases")),
        "failed_cases": _list_of_strings(source.get("failed_cases")),
        "side_effects": source.get("side_effects") if isinstance(source.get("side_effects"), list) else [],
        "graph_scope": graph_scope,
        # Where the binding half of this row lives. A rule may carry no object ids, so a row
        # written that way names the run-directory file that holds the ids it was measured on
        # instead of inlining them, and validation reads the pointer as its grounding.
        "binding_ref": str(source.get("binding_ref") or ""),
        "accepted": bool(source.get("accepted")),
        "repair_gain": _optional_number(source.get("repair_gain")),
        "comparison": str(source.get("comparison") or ""),
        "pairwise_verdict": str(source.get("pairwise_verdict") or ""),
        "object_diff": source.get("object_diff") if isinstance(source.get("object_diff"), dict) else {},
        "compiled_object_diff": source.get("compiled_object_diff") if isinstance(source.get("compiled_object_diff"), dict) else {},
        "reported_reflection": source.get("reported_reflection") if isinstance(source.get("reported_reflection"), dict) else {},
        "condition_evidence": source.get("condition_evidence") if isinstance(source.get("condition_evidence"), dict) else {},
        "excluded_conditions": [dict(item) for item in source.get("excluded_conditions") if isinstance(item, dict)] if isinstance(source.get("excluded_conditions"), list) else [],
        "provenance": {
            "source": str(provenance_source.get("source") or ""),
            "run": str(provenance_source.get("run") or ""),
            "round": int(provenance_source["round"]) if str(provenance_source.get("round") or "").lstrip("-").isdigit() else None,
            "measured": bool(provenance_source.get("measured")),
            "before_score": _optional_number(provenance_source.get("before_score")),
            "after_score": _optional_number(provenance_source.get("after_score")),
        },
        "status": status,
    }


def validate_experience(experience: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_experience(experience)
    errors: list[str] = []
    if not normalized["experience_id"]:
        errors.append("experience_id is required")
    if normalized["scope"] not in {"global", "slot", "interface", "connector"}:
        errors.append("experience scope is unsupported")
    if normalized["applicability"] == "domain" and _paper_domain_missing(normalized["applicability_domain"]):
        errors.append("a domain experience requires a named paper domain")
    if normalized["memory_type"] not in MEMORY_TYPES:
        errors.append("experience memory_type is unsupported")
    if normalized["apply_at"] not in APPLY_AT_STAGES:
        errors.append("experience apply_at stage is unsupported")
    unknown_motifs = [item for item in normalized["required_motifs"] if item not in MOTIF_KINDS]
    if unknown_motifs:
        errors.append("experience required_motifs are unsupported: " + ", ".join(sorted(unknown_motifs)))
    if not normalized["hypothesis"]:
        errors.append("experience hypothesis is required")
    action_type = str(normalized["action"].get("type") or normalized["action"].get("action") or "")
    if action_type not in EXPERIENCE_ACTIONS:
        errors.append("experience action is unsupported")
    graph_scope = normalized["graph_scope"]
    if not str(normalized.get("binding_ref") or "") and not any(graph_scope[key] for key in ("subgraph_ids", "scientific_node_ids", "edge_ids", "slot_ids", "object_ids")):
        # A rule that travels carries no object ids; it points at the binding that holds them.
        # A row written before the split still inlines its ids and is grounded the old way.
        errors.append("experience requires node/edge/slot/object or subgraph grounding, or a binding_ref to the file that holds them")
    legacy_keys = {
        "whole_" + "image",
        "approved_" + "image",
        "selected_" + "image",
        "whole_" + "image_evolution_report",
    }
    if legacy_keys.intersection(experience):
        errors.append("legacy whole-image experience fields are forbidden")
    gain = normalized["repair_gain"]
    provenance = normalized["provenance"]
    if gain is None:
        errors.append("experience repair_gain is required and must be a measured number")
    if not provenance["measured"]:
        errors.append("experience provenance must record a measurement, not an expectation")
    if not provenance["source"] or not provenance["run"]:
        errors.append("experience provenance must name the source and the run that measured it")
    measured = [value for value in (provenance["before_score"], provenance["after_score"]) if value is not None]
    if gain is not None and len(measured) == 2 and abs(gain - (measured[1] - measured[0])) > 0.0001:
        errors.append("experience repair_gain disagrees with its measured provenance")
    pairwise = str(normalized.get("comparison") or "") == "pairwise"
    if pairwise and normalized["accepted"] and str(normalized.get("pairwise_verdict") or "") != "after":
        errors.append("a pairwise-accepted experience must record pairwise_verdict=after")
    if normalized["status"] == "refuted":
        if not normalized["excluded_conditions"]:
            errors.append("a refuted experience must name the conditions it excludes")
        if not pairwise and gain is not None and gain > 0:
            errors.append("a refuted experience cannot record a positive repair_gain")
    if not pairwise and normalized["accepted"] and gain is not None and gain <= 0:
        errors.append("an accepted experience must record a positive measured repair_gain")
    evidence = normalized["condition_evidence"]
    if evidence:
        if not isinstance(evidence.get("changed_object_ids"), list):
            errors.append("condition_evidence must list the object ids the repair measurably changed")
        if not str(evidence.get("measured_by") or ""):
            errors.append("condition_evidence must name the measurement it came from")
    reflection = normalized["reported_reflection"]
    if reflection:
        if not isinstance(reflection.get("verified"), bool):
            errors.append("reported_reflection must state whether it was verified")
        if reflection.get("verified"):
            errors.append("reported_reflection cannot be stored as verified: only measured conditions are")
    for index, condition in enumerate(normalized["excluded_conditions"]):
        if "dimension" in condition and not str(condition.get("value") or ""):
            errors.append(f"excluded_conditions[{index}] names a dimension but no value")
    return {"ok": not errors, "errors": errors, "experience": normalized}


def normalize_authored_patch(raw: Any) -> dict[str, Any]:
    """Read the prompt patch a reflection wrote: where it applies, when, and what it says.

    Read as three plain fields rather than whatever shape the model returned, because the two
    halves are used apart: the stage is checked against the pipeline's own vocabulary before the
    sentence is allowed to travel, and the condition goes through the same "does this read as
    anything" filter as every other condition in the library.
    """

    source = raw if isinstance(raw, dict) else {}
    condition = source.get("condition") if isinstance(source.get("condition"), dict) else {}
    return {
        "apply_at": str(source.get("apply_at") or "").strip(),
        "condition": dict(condition),
        "directive": str(source.get("directive") or "").strip(),
    }


#: The comparison a claim may state over a measured metric. A claim that names no reading
#: cannot be settled by anyone, and a claim that names one is settled by the same numbers
#: every other reader uses, so the vocabulary is closed on purpose rather than left to prose.
CLAIM_OPERATORS = ("<=", "<", ">=", ">", "==", "!=")
#: The two verdicts a settlement reaches. A round that measured nothing settles nothing, so
#: "noise" is deliberately not a third verdict: it is the absence of one, and it neither
#: confirms a claim nor counts against it.
CLAIM_VERDICTS = ("confirmed", "refuted")


def _claim_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def claim_identifier(claim: str, apply_at: str, criterion: dict[str, Any]) -> str:
    """Name one claim by what it says and what would settle it.

    A seat that restates the same claim in the same words against the same reading is the
    same claim, so the row is not registered twice; a seat that changes the reading has said
    something new, and that is a new row.
    """

    raw = json.dumps(
        {
            "claim": str(claim or "").strip(),
            "apply_at": str(apply_at or "").strip(),
            "metric": str((criterion or {}).get("metric") or ""),
            "op": str((criterion or {}).get("op") or ""),
            "value": (criterion or {}).get("value"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return "claim-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def normalize_claim(raw: Any) -> dict[str, Any]:
    """Read one falsifiable claim: what it says, where it applies, and what would settle it.

    A claim with no sentence and no reading is not a claim and normalizes to an empty row the
    caller drops. A criterion whose operator or value does not read as one is dropped too,
    leaving the sentence on the record as an unfalsifiable note rather than a claim that
    something could have settled.
    """

    source = raw if isinstance(raw, dict) else {}
    stated = str(source.get("claim") or source.get("prediction") or "").strip()
    apply_at = str(source.get("apply_at") or source.get("check_at") or "").strip()
    criterion_source = source.get("criterion") if isinstance(source.get("criterion"), dict) else {}
    operator = str(criterion_source.get("op") or "").strip()
    if operator not in CLAIM_OPERATORS:
        operator = ""
    value = _claim_number(criterion_source.get("value"))
    metric = str(criterion_source.get("metric") or "").strip()
    criterion = {"metric": metric, "op": operator, "value": value} if metric and operator and value is not None else {}
    return {
        "claim_id": claim_identifier(stated, apply_at, criterion),
        "claim": stated,
        "apply_at": apply_at,
        "prediction": str(source.get("prediction") or "").strip(),
        "criterion": criterion,
        "settled_by": str(source.get("settled_by") or "").strip(),
    }


def normalize_claims(raw: Any) -> list[dict[str, Any]]:
    """Read a reflection's claim list, keeping one row per distinct claim."""

    if isinstance(raw, dict):
        raw = [raw]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, (list, tuple)) else []:
        claim = normalize_claim(item)
        if not claim["claim"] and not claim["criterion"]:
            continue
        if claim["claim_id"] in seen:
            continue
        seen.add(claim["claim_id"])
        rows.append(claim)
    return rows


def claim_settles(claim: dict[str, Any] | None, metrics: dict[str, Any] | None) -> str:
    """The verdict this reading gives a claim, or "" when the reading settles nothing.

    A claim is settled only by the number it named: if this round did not measure that metric,
    the claim stays open and no verdict is recorded. That is what keeps a round whose reading
    was withdrawn from counting against a claim, exactly as it counts against nothing else.
    """

    source = claim if isinstance(claim, dict) else {}
    criterion = source.get("criterion") if isinstance(source.get("criterion"), dict) else {}
    metric = str(criterion.get("metric") or "")
    operator = str(criterion.get("op") or "")
    value = _claim_number(criterion.get("value"))
    if not metric or not operator or value is None:
        return ""
    if not isinstance(metrics, dict) or metric not in metrics:
        return ""
    observed = _claim_number(metrics.get(metric))
    if observed is None:
        return ""
    if operator == "<=":
        holds = observed <= value
    elif operator == "<":
        holds = observed < value
    elif operator == ">=":
        holds = observed >= value
    elif operator == ">":
        holds = observed > value
    elif operator == "==":
        holds = observed == value
    else:
        holds = observed != value
    return "confirmed" if holds else "refuted"


def normalize_defects(raw: Any) -> dict[str, list[str]]:
    """Read a reflection's two defect lists, each as a list of non-empty strings."""

    source = raw if isinstance(raw, dict) else {}
    return {key: [str(item) for item in (source.get(key) or []) if str(item)] for key in DEFECT_KEYS}


def reflection_conflict(reflection: dict[str, Any] | None) -> str:
    """Name the contradiction a reflection states, or ``""`` when it states none.

    One schema used to carry both answers, so a row could say the round improved and name a
    regression in the same breath. ``defects`` now says what changed and ``failure_type`` says
    how the round failed, which makes the disagreement checkable: a row that claims an
    improvement and also names a failure is asked again rather than stored as a finding.

    ``measurement_noise_risk`` is the third way a row disagrees with itself. ``outcome`` reports
    the direction this round's own measurement gave, and that flag says the same measurement
    could come out the other way, so a row stating both is asked again instead of being left to
    withdraw its own improvement on an answer nobody can read. ``randomness_risk`` is deliberately
    not part of this check: it claims the drawing is stochastic, which an improved round can say
    honestly, and the consumption side already withholds it where a clear pair decided the round.
    """

    source = reflection if isinstance(reflection, dict) else {}
    outcome = str(source.get("outcome") or "").strip()
    failure_type = str(source.get("failure_type") or "").strip()
    if outcome == "improved" and failure_type and failure_type != "none":
        return f"outcome is improved but failure_type is {failure_type}"
    introduced = [str(item) for item in ((source.get("defects") or {}).get("introduced") or []) if str(item)]
    if outcome == "improved" and introduced:
        return "outcome is improved but defects.introduced is not empty"
    if outcome == "improved" and bool(source.get("measurement_noise_risk")):
        return "outcome is improved but measurement_noise_risk is set"
    return ""


def normalize_layer_settled(raw: Any) -> dict[str, str]:
    """The layer a reflection says it has finished with, and the reason it gives.

    Only a stage the pipeline has survives, and only the seat's own words: a reflection that names
    nothing readable has settled nothing, which is a different thing from a layer having been
    closed. The loop writes every settle down and works none of them out -- a reading says what the
    figure measures, never that a layer is done with.
    """

    source = raw if isinstance(raw, dict) else {}
    stage = str(source.get("stage") or source.get("layer") or "").strip()
    if stage not in STAGE_ORDER:
        return {}
    return {
        "stage": stage,
        "because": str(source.get("because") or source.get("reason") or "").strip(),
    }


def normalize_reflection(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Read one reflection. A row written before the outcome/defect split keeps its shape.

    A row that predates the split carries neither defect list and is read as an empty pair
    rather than migrated, and any contradiction it does carry is named on
    ``reflection_conflict`` so the caller can ask again instead of storing it as a finding.
    """

    source = raw if isinstance(raw, dict) else {}
    reflection = {
        **source,
        "contract": "rfs.judge_reflection",
        "version": REFLECTION_CONTRACT_VERSION,
        "outcome": str(source.get("outcome") or "unknown"),
        "defects": normalize_defects(source.get("defects")),
        "failure_type": str(source.get("failure_type") or "none"),
        "why_changed": str(source.get("why_changed") or ""),
        "scope_assessment": str(source.get("scope_assessment") or "appropriate"),
        "domain_mismatch": bool(source.get("domain_mismatch", False)),
        "randomness_risk": bool(source.get("randomness_risk", False)),
        "measurement_noise_risk": bool(source.get("measurement_noise_risk", False)),
        # Who wrote this reflection. The fallback the loop writes when the seat did not answer
        # used to set the noise flags on the model's behalf, which is what withdrew those
        # rounds; a reader has to be able to tell the two apart.
        "reflection_source": str(source.get("reflection_source") or "model"),
        "next_policy": str(source.get("next_policy") or "keep"),
        "revised_action": source.get("revised_action") if isinstance(source.get("revised_action"), dict) else {},
        # A reflection states the move it wants in two parts: the binding names this figure's
        # objects and the rule is what another figure could be drawn by. Both are read as dicts
        # so a malformed half is empty rather than whatever shape the model returned.
        "binding": source.get("binding") if isinstance(source.get("binding"), dict) else {},
        "rule": source.get("rule") if isinstance(source.get("rule"), dict) else {},
        # Three fields a losing round adds about itself, each read as an empty on a row written
        # before them: which of the four answers it gave, the same rule said more precisely, and
        # one repair aimed at the spot the round broke.
        "attribution": str(source.get("attribution") or ""),
        "refined_rule": source.get("refined_rule") if isinstance(source.get("refined_rule"), dict) else {},
        "follow_up": source.get("follow_up") if isinstance(source.get("follow_up"), dict) else {},
        # The instruction a later round's prompt should carry: which stage's prompt was wrong,
        # under what condition, and the sentence it should have said. The wording is the seat's;
        # whether it is ever injected is decided by the measurement behind the row, never here.
        "prompt_patch": normalize_authored_patch(source.get("prompt_patch")),
        # The layer this reflection says it has finished with. The loop records it and never
        # decides it: a reading is evidence about the figure, not a verdict about a layer.
        "layer_settled": normalize_layer_settled(source.get("layer_settled")),
        # Every substantive sentence a reflection writes about why the round went the way it
        # did is a claim somebody can check later: what it says, where it applies, and the
        # reading that would settle it. A claim that names no reading is kept as a note and
        # never settled, which is the difference between an opinion and a falsifiable claim.
        "claims": normalize_claims(source.get("claims")),
    }
    reflection["reflection_conflict"] = reflection_conflict(reflection)
    return reflection
