"""State, bounded repair, rollback, and reflection control for direct evolution."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from ...contracts.evolution import (
    ATTRIBUTION_VALUES,
    DERIVED_SCORE_SOURCE,
    FIGURE_MOVE_EVIDENCE_CONTRACT,
    introduced_defects,
    MECHANISM_ACTIONS,
    normalize_reflection,
    pairwise_verdict,
)
from ...contracts.repair_actions import ACTION_BUDGET_KEYS, action_missing_required_targets, action_needs_budget, budget_keys_for
from .action_policy import COMPILE_RULE_CLASS, FIGURE_PROGRAM_CLASS, action_class
from .actions import normalize_mechanism_rule
from .file_lock import append_jsonl_sequenced
from .rule_binding import case_object_evidence


ACTION_PRIORITY = {
    "reanalyze_semantics": 0,
    "replan_structure": 1,
    "resegment": 2,
    "relayout_global": 3,
    "relayout_slots": 4,
    "resize_nodes": 4,
    "repair_interfaces": 5,
    "reroute": 5,
    "regen_assets": 6,
    # A theme re-plan changes no geometry, so it waits until every object-level repair has
    # had its turn. The gap at 7 is where ``recompile`` used to sit; the numbers order the
    # actions and are not a count, so the ones already recorded keep their meaning.
    "replan_theme": 8,
    # The mechanism-level rules are not moves on this figure: they hand a rule to the compile
    # that draws the next one, so they are ordered after every object-level repair. They are in
    # this table at all because ``action_is_executable`` reads it: a name absent here can never
    # be selected, which is how the two mechanism actions stayed dead after they were written.
    "router_constraint": 9,
    "layout_constraint": 9,
}

# What a reflection may ask the loop to do next. Anything else is read as "keep".
REFLECTION_POLICIES = ("keep", "narrow_scope", "disable", "revise", "revise_rule", "follow_up")

#: How many cheap follow-ups one losing rule may be given before it is written off. A follow-up
#: edits the figure program or a compile-time rule, so it calls no model and costs seconds; three
#: of them is still minutes, which is cheaper than asking for a fresh rule, and worth spending
#: before a rule that may only have been carried out badly is thrown away.
FOLLOW_UP_ATTEMPTS = 3

#: The two attributions that earn another attempt. ``rule_is_wrong`` deliberately does not: it is
#: the claim that needs no evidence and buys nothing. ``unmeasurable`` is about the reading rather
#: than the rule, so there is nothing to try again.
FOLLOW_UP_ATTRIBUTIONS = ("not_carried_out", "local_side_effect")


def measured_score(result: dict[str, Any] | None) -> float | None:
    """Read the score this result measured, or None when it measured nothing.

    A result answering the advisory question carries no score at all, and a scoring role
    that was asked for one but gave nothing usable measured nothing either. Both read as
    None, never as zero: zero is the worst verdict, not a missing one, and a fabricated
    zero is what let an unmeasured round look like a regression.

    A number the acceptance seat looked up in the defect table is not a measurement of
    the figure either, so it is refused here the same way: the state falls back to the
    deterministic gate's own reading of that figure, which is the one ruler every state
    is measured on.
    """

    source = result if isinstance(result, dict) else {}
    if str(source.get("score_source") or "") == DERIVED_SCORE_SOURCE:
        value = source.get("absolute_score")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if source.get("score_measured") is False:
        return None
    value = source.get("overall_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # The absolute reading: the deterministic gate's own measurement of this figure, the
        # same ruler on every state. A seat that was asked the pair question carries no overall
        # score of its own, and the choice's relative number -- 1.0 for whichever side was not
        # blamed -- is not a measurement of either figure, so it never stands in for this.
        value = source.get("absolute_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def derived_score(result: dict[str, Any] | None) -> float | None:
    """Read the defect-table score a choice comparison wrote, or None.

    The number is derived from the defect the seat named, not measured from the figure:
    the same table row is written onto whichever figure was blamed, so two rounds that
    blamed different defects get different numbers for the same figure. It is reported
    under its own name, and it never decides.
    """

    source = result if isinstance(result, dict) else {}
    if str(source.get("score_source") or "") != DERIVED_SCORE_SOURCE:
        return None
    value = source.get("overall_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metric_score(result: dict[str, Any] | None, metric: str) -> float | None:
    """Read one comparable metric, falling back to the overall score."""

    if metric == "overall_score":
        return measured_score(result)
    source = result if isinstance(result, dict) else {}
    dimensions = source.get("dimension_scores") if isinstance(source.get("dimension_scores"), dict) else {}
    value = dimensions.get(metric, source.get(metric))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return measured_score(result)


def _blockers(result: dict[str, Any] | None) -> int:
    value = (result or {}).get("blocking_issue_count")
    if value is not None:
        return int(value or 0)
    feedback = (result or {}).get("stage_feedback") or []
    return sum(str(item.get("severity") or "") == "blocker" for item in feedback if isinstance(item, dict))


def _regression(before: dict[str, Any] | None, after: dict[str, Any]) -> tuple[list[str], int]:
    """Name the defects this round introduced, and how many they are.

    The tally of ``blocking_issues`` cannot answer this: that list is prose the model writes
    fresh on every call, and the two calls being compared saw two different images minutes
    apart, so it grew whenever the model happened to enumerate one more thing -- which is how
    a round the comparison preferred with a clear margin was refused for "introducing" a
    defect its own evidence showed in both states. Only the gate's half of the list is the same
    instrument twice, and the rule that reads it lives in one place (``introduced_defects``) so
    the round decision and the library cannot answer this question with two different rulers.
    """

    reading = introduced_defects(before, after)
    return list(reading["defects"]), int(reading["count"])


def _compact_name(value: Any) -> str:
    """Fold spacing and case so two spellings of one name can be compared."""

    return re.sub(r"[\s\-]+", "_", str(value or "").strip().lower()).strip("_")


def aliased_action_name(candidate: dict[str, Any]) -> tuple[str, str]:
    """Return the bounded action one proposed revision means, plus the name it came from.

    A model names the same action in shapes the loop does not spell the same way: ``Relayout``
    with scope ``slots``, or ``relayout-slots``. The alias below is deterministic and only ever
    lands on a name the loop may run, so it cannot widen what a reflection may ask for -- a name
    that matches nothing is still refused. The second element is the raw name, which is only
    worth recording when it differs from the name that matched.
    """

    source = candidate if isinstance(candidate, dict) else {}
    raw = str(source.get("type") or source.get("action") or "")
    compact = _compact_name(raw)
    if compact in ACTION_PRIORITY:
        return compact, raw
    scope = _compact_name(source.get("scope_type") or source.get("scope"))
    if not compact or not scope:
        return "", raw
    # ``slots`` and ``slot`` name the same scope, and a scope may be spelled plural in either
    # direction (``relayout_slots`` is the shipped name for scope ``slot``).
    for candidate_type in (f"{compact}_{scope}", f"{compact}_{scope.rstrip('s')}", f"{compact}_{scope}s"):
        if candidate_type in ACTION_PRIORITY:
            return candidate_type, raw
    return "", raw


def append_event(path: str | Path, event: str, **payload: Any) -> dict[str, Any]:
    """Append one immutable evolution event.

    The number is read from the row before it, under the file's own lock, because a continuous
    wave writes its events from more than one paper at a time and two writers that count the
    file at the same moment both write the same number.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return append_jsonl_sequenced(
        target,
        lambda previous, count: {"seq": int((previous or {}).get("seq") or count) + 1, "event": event, **payload},
    )


def select_bounded_action(judgement: dict[str, Any] | None) -> dict[str, Any] | None:
    """Select the earliest legal repair and retain its object bindings.

    Nothing legal to run is ``None``, not a synthesized ``relayout_global``. Inventing
    that name made an empty advisory answer indistinguishable from a real global-nudge
    prescription, and the experience it banked described a repair nobody asked for.
    """

    candidates = []
    for feedback in (judgement or {}).get("stage_feedback", []):
        if not isinstance(feedback, dict):
            continue
        for action in feedback.get("requested_actions") or feedback.get("repair") or []:
            action_record = action if isinstance(action, dict) else {"action": action}
            action_type = str(action_record.get("action") or action_record.get("type") or "")
            if action_type == "relayout":
                action_type = "relayout_slots" if feedback.get("slot_ids") or feedback.get("object_ids") else "relayout_global"
            if action_type not in ACTION_PRIORITY:
                continue
            record = {
                "type": action_type,
                "scope_type": feedback.get("scope_type") or feedback.get("stage") or "global",
                "semantic_node_ids": _merge_ids(action_record, feedback, "semantic_node_ids"),
                "slot_ids": _merge_ids(action_record, feedback, "slot_ids"),
                "object_ids": _merge_ids(action_record, feedback, "object_ids"),
                "edge_ids": _merge_ids(action_record, feedback, "edge_ids"),
                "problem": str(feedback.get("problem") or action_record.get("problem") or ""),
                # The half of the answer that is meant for another paper. It rides along on the
                # action so the Memory row can carry it as its rule; the problem sentence above
                # stays evidence about this figure and is split off into the binding.
                "transferable_advice": str(feedback.get("transferable_advice") or action_record.get("transferable_advice") or ""),
                "expected_effect": str(feedback.get("expected_effect") or ""),
                "forbidden_changes": list(feedback.get("forbidden_changes") or []),
            }
            for key in ACTION_BUDGET_KEYS:
                value = action_record.get(key)
                if isinstance(value, bool):
                    continue
                try:
                    record[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if action_needs_budget(action_type) and any(key not in record for key in budget_keys_for(action_type)):
                # A geometric move without magnitudes is the shipped default in disguise.
                # Skip it so an incomplete VLM answer becomes no executable repair.
                continue
            if action_type in MECHANISM_ACTIONS:
                # A mechanism-level rule is not a move on this figure: it is a rule the compile
                # that draws the next one honours, so its parameters are the whole prescription.
                # A bare name states nothing, and running it would spend a round, two Judge
                # answers and a compile on a rule the compiler cannot read. The one reader of
                # those parameters decides here and in the executor, so the two cannot disagree.
                rule = normalize_mechanism_rule({**action_record, "type": action_type})
                if not rule:
                    continue
                record.update(rule)
                candidates.append((ACTION_PRIORITY[action_type], record))
                continue
            if action_missing_required_targets(action_type, record, action_record, feedback):
                continue
            candidates.append((ACTION_PRIORITY[action_type], record))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    chosen = candidates[0][1]
    label = str((judgement or {}).get("applicability") or "").strip().casefold()
    if label in {"common", "domain"}:
        chosen["applicability"] = label
    return chosen


def _merge_ids(action_record: dict[str, Any], feedback: dict[str, Any], key: str) -> list[str]:
    """Keep object ids the action named even when the feedback entry only listed them once."""

    found: list[str] = []
    for source in (action_record, feedback):
        for item in source.get(key) or []:
            text = str(item).strip()
            if text and text not in found:
                found.append(text)
    return found


def action_is_executable(action: dict[str, Any] | None) -> bool:
    """Whether this dict names a bounded move and carries the keys the executor reads."""

    source = action if isinstance(action, dict) else {}
    action_type = str(source.get("type") or "")
    if action_type not in ACTION_PRIORITY:
        return False
    if action_needs_budget(action_type) and any(key not in source for key in budget_keys_for(action_type)):
        return False
    return True


def mechanism_rule_fingerprint(action: dict[str, Any] | None) -> str:
    """The identity of one mechanism-level rule, or empty when the action states no rule.

    Two rows that state the same parameters are the same rule however they were spelled, and a
    round that already spent its turn on it must not spend another. The normalized rule is that
    identity, so the comparison is over what the compiler would read rather than over the text.
    """

    rule = normalize_mechanism_rule(action)
    if not rule:
        return ""
    return json.dumps({key: rule[key] for key in sorted(rule) if key != "apply_at"}, sort_keys=True, ensure_ascii=False)


def next_rule_turn(
    rule_action: dict[str, Any] | None, *, tried: set[str] | frozenset[str] | None = None
) -> tuple[dict[str, Any], str]:
    """Offer a reflection's rule to the next round, at most once per iteration.

    A reflection's rule is a claim about how figures should be drawn, and a claim that was never
    put to the loop is only an opinion. So the next round tries it, once: one round is enough to
    measure it, a rule that did not help must not become every later round's proposal, and a rule
    that did help stops being a proposal anyway -- it becomes a Memory row. Returns the action to
    propose and its fingerprint, both empty when the rule is not offered this time.
    """

    action = rule_action if isinstance(rule_action, dict) else {}
    fingerprint = mechanism_rule_fingerprint(action)
    if not fingerprint or not action_is_executable(action):
        return {}, ""
    if fingerprint in {str(item) for item in (tried or ())}:
        return {}, ""
    return dict(action), fingerprint


def next_follow_up_turn(
    follow_up_action: dict[str, Any] | None,
    *,
    rule_fingerprint: str,
    tried: dict[str, int] | None = None,
) -> tuple[dict[str, Any], str]:
    """Offer the repair a losing rule earned, while that rule still has an attempt left.

    A rule that lost was measured once: the reading says this attempt did not land better, and
    the round that produced it says why. Asking again costs a compile and a judgement when the
    answer edits the figure program, so a rule is given ``FOLLOW_UP_ATTEMPTS`` cheap attempts
    before it is written off. The budget belongs to the *rule*, not to the wording of the repair:
    counting per fingerprint would let a reflection that rephrased itself every round run
    forever, which is the opposite of what a budget is for.
    """

    action = follow_up_action if isinstance(follow_up_action, dict) else {}
    key = str(rule_fingerprint or "").strip()
    if not key:
        # Nothing lost, so nothing is being repaired: a follow-up only exists as the answer to
        # a rule that was measured and did not win.
        return {}, ""
    if int((tried or {}).get(key, 0)) >= FOLLOW_UP_ATTEMPTS:
        return {}, ""
    fingerprint = action_execution_fingerprint(action)
    if not fingerprint or not action_is_executable(action):
        return {}, ""
    if action_class(action.get("type")) not in (FIGURE_PROGRAM_CLASS, COMPILE_RULE_CLASS):
        return {}, ""
    return dict(action), fingerprint


def verify_attribution(
    reflection: dict[str, Any] | None,
    implementation: dict[str, Any] | None,
    pairwise: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Weigh the reason a losing round gave against the records that round already holds.

    A reflection explains the round it just lost, and the two explanations that earn another
    attempt are checkable: ``not_carried_out`` has to answer to a round that could not reach where
    it aimed, and ``local_side_effect`` to a figure that changed under a comparison that named a
    defect. Checking the claim rather than trusting the model is the whole point -- the reflections
    this was written against read "the figure retained the same hash as the input, indicating no
    actual geometric changes" beside "a score drop from 1.0 to 0.6" in one answer, and a follow-up
    spent on that premise would have been aimed at nothing.

    Both halves are read off whatever the round recorded. A compile-time rule is answered by the
    compile's own report: what it applied, and where it could not reach. A move on the figure is
    answered by the figure's own diff: whether it changed at all, and the defect the comparison
    named. Either way the same two sentences are earned and refused for the same two reasons, so a
    losing suggestion is not written off merely because it was not a rule.

    ``rule_is_wrong`` needs no evidence because it spends nothing, and ``unmeasurable`` is about
    the reading rather than the rule, so neither is refused. An unstated attribution returns
    ``""``, which also buys nothing. The second value says why a claim was refused, so the round
    can record the refusal instead of quietly ignoring it.
    """

    source = reflection if isinstance(reflection, dict) else {}
    stated = str(source.get("attribution") or "").strip()
    if not stated:
        return "", ""
    if stated not in ATTRIBUTION_VALUES:
        return "", "is not one of " + "|".join(ATTRIBUTION_VALUES)
    if stated == "rule_is_wrong":
        return stated, ""
    report = implementation if isinstance(implementation, dict) else {}
    if not report:
        return "", "no implementation report was recorded for this round, so nothing about it can be checked"
    # A move on the figure is evidence about that move; a rule is evidence about the compile that
    # carried it. The two are told apart only so the refusal reads like what actually happened,
    # never to make one of them easier to earn: the tests below are the same for both.
    moved_the_figure = str(report.get("contract") or "") == FIGURE_MOVE_EVIDENCE_CONTRACT
    if stated == "not_carried_out":
        if list(report.get("compile_unmet") or []) or not bool(report.get("figure_changed")):
            return stated, ""
        if moved_the_figure:
            return "", "the round's own record shows the figure changed, so the move was carried out"
        return "", "the compile reported nothing unmet and the figure changed, so the rule was carried out"
    if stated == "local_side_effect":
        if not bool(report.get("figure_changed")):
            if moved_the_figure:
                return "", "the round's own record shows the figure did not change, so nothing was broken by it"
            return "", "the figure did not change, so this round cannot have broken anything"
        named = [item for item in ((pairwise or {}).get("diagnosis") or []) if isinstance(item, dict)]
        if named:
            return stated, ""
        return "", "the comparison named no defect, so nothing was broken that could be pointed at"
    margin = str((pairwise or {}).get("margin") or "").strip()
    if margin and margin != "clear":
        return stated, ""
    if bool(source.get("measurement_noise_risk")):
        return stated, ""
    return "", "the comparison called the difference clear and no measurement noise was claimed"


def color_feedback_proposal(pending_color_feedback: dict[str, Any] | None) -> dict[str, Any]:
    """Turn a measured color-review complaint into a theme re-plan, or nothing.

    In-candidate palette mutation stays disabled; this is the only color move the
    loop is allowed to schedule, and it is scheduled only when the review asked.
    """

    source = pending_color_feedback if isinstance(pending_color_feedback, dict) else {}
    if not source.get("repair_required"):
        return {}
    return {
        "type": "replan_theme",
        "scope_type": "global",
        "problem": "measured color review required a theme re-plan",
        "expected_effect": "re-plan the palette without changing scientific content",
        "source": "color_feedback",
    }


def _decomposition_is_hard(policy: dict[str, Any] | None) -> bool:
    """Whether the policy was driven by a hard decomposability break, not a soft occupancy one."""

    source = policy if isinstance(policy, dict) else {}
    violations = source.get("violations") if isinstance(source.get("violations"), list) else []
    return any(isinstance(item, dict) and str(item.get("kind") or "") == "hard" for item in violations)


def select_next_round_proposal(
    *,
    follow_up_action: dict[str, Any] | None = None,
    revised_action: dict[str, Any] | None = None,
    decomposition_action: dict[str, Any] | None = None,
    decomposition_proposal: dict[str, Any] | None = None,
    rule_action: dict[str, Any] | None = None,
    judgement: dict[str, Any] | None = None,
    pending_color_feedback: dict[str, Any] | None = None,
    skipped_action_types: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Pick the next bounded move without letting a soft occupancy break starve Judge or theme.

    Order is fixed: the repair a losing rule earned, then a reflection that already named a legal
    revision, then a hard decomposability break, then the rule that same reflection stated, then a
    measured color
    repair, then the advisory Judge, then a soft decomposability fallback. Soft occupancy used
    to be attached to the catalog nudge and therefore always executable, so the Judge never got
    a turn. A type that just rolled back is skipped once so the other axes still get a turn
    instead of repeating the same failed family until the round budget is gone.

    The rule sits below a hard break and above everything the figure was already being told: a
    figure whose structure does not hold has to be fixed before a drawing rule can be measured
    on it, and a rule the loop has never run cannot be left waiting behind a colour repair that
    was going to be proposed anyway. ``next_rule_turn`` decides whether there is a rule to offer
    at all, so this function is not the place that counts the turns.
    """

    order = round_proposal_order(
        follow_up_action=follow_up_action,
        revised_action=revised_action,
        decomposition_action=decomposition_action,
        decomposition_proposal=decomposition_proposal,
        rule_action=rule_action,
        judgement=judgement,
        pending_color_feedback=pending_color_feedback,
        skipped_action_types=skipped_action_types,
    )
    return dict(order[0]) if order else {}


def round_proposal_order(
    *,
    follow_up_action: dict[str, Any] | None = None,
    revised_action: dict[str, Any] | None = None,
    decomposition_action: dict[str, Any] | None = None,
    decomposition_proposal: dict[str, Any] | None = None,
    rule_action: dict[str, Any] | None = None,
    judgement: dict[str, Any] | None = None,
    pending_color_feedback: dict[str, Any] | None = None,
    skipped_action_types: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Every move this round may run, in the order the loop prefers them.

    The order used to be private to the picker, which returned only its first usable entry.
    A round that ran two moves needs to know what came second -- the suggestion the
    mechanism's own move displaced -- without asking the controller a second time and
    risking a different answer. The list is deduplicated by execution fingerprint so the
    soft decomposability fallback, which is offered twice, is offered once.

    The repair a losing rule earned goes ahead of everything: it continues the round that just
    failed, and a move that exists because the previous round was measured has to run before
    anything that was already waiting for a turn.
    """

    skipped = {str(item) for item in (skipped_action_types or ()) if str(item)}
    decomp = dict(decomposition_proposal) if isinstance(decomposition_proposal, dict) else {}
    hard = _decomposition_is_hard(decomposition_action)

    def _usable(candidate: dict[str, Any]) -> bool:
        return action_is_executable(candidate) and str(candidate.get("type") or "") not in skipped

    order: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in (
        dict(follow_up_action or {}),
        dict(revised_action or {}),
        decomp if hard else {},
        dict(rule_action or {}),
        color_feedback_proposal(pending_color_feedback),
        select_bounded_action(judgement) or {},
        decomp if not hard else {},
    ):
        if not _usable(candidate):
            continue
        fingerprint = action_execution_fingerprint(candidate)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        order.append(dict(candidate))
    return order


def _bounded_bbox(box: dict[str, Any], factor: float, dx: float = 0.0, dy: float = 0.0) -> dict[str, float]:
    x, y, w, h = (float(box.get(key) or 0.0) for key in ("x", "y", "w", "h"))
    nw, nh = min(0.95, max(0.04, w * factor)), min(0.95, max(0.04, h * factor))
    nx = min(1.0 - nw, max(0.0, x - (nw - w) / 2 + dx))
    ny = min(1.0 - nh, max(0.0, y - (nh - h) / 2 + dy))
    return {"x": round(nx, 4), "y": round(ny, 4), "w": round(nw, 4), "h": round(nh, 4)}


def _budget(action: dict[str, Any], key: str, default: float) -> float:
    """Read one unsigned magnitude from the prescription, else ``default``.

    ``default`` is the identity for that axis (factor 1, shift 0), not the catalog
    nudge. A missing VLM budget is refused before this pass; guessing the shipped
    numbers here would recreate the default policy the caller did not name.
    """

    value = action.get(key)
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def action_execution_fingerprint(action: dict[str, Any] | None) -> str:
    """Name the move a round actually executed, including the budget it ran under.

    Two ``relayout_global`` rounds that nudged by different amounts are two different
    moves. A comparison that only reads the action name reports them as identical, which
    is how a pair whose ON arm replayed a measured budget and whose OFF arm used the
    shipped default was logged as ``no_contrast_between_on_and_off``.
    """

    source = action if isinstance(action, dict) else {}
    parts = [str(source.get("type") or "")]
    for key in ACTION_BUDGET_KEYS:
        value = source.get(key)
        if isinstance(value, bool):
            continue
        try:
            parts.append(f"{key}={float(value)}")
        except (TypeError, ValueError):
            continue
    return "|".join(parts)


def apply_bounded_action(layout: dict[str, Any], icons: dict[str, Any], action: dict[str, Any], *, retry_index: int = 0, domain_profile: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Apply a deterministic, object-bound proposal without mutating inputs."""
    next_layout, next_icons = deepcopy(layout), deepcopy(icons)
    action_type = str(action.get("type") or "relayout_global")
    target_nodes = set(str(item) for item in action.get("semantic_node_ids", []) + action.get("object_ids", []))
    target_slots = set(str(item) for item in action.get("slot_ids", []))
    target_nodes.update(value.removeprefix("slot_") for value in target_slots)
    changed_nodes: list[str] = []
    changed_assets: list[str] = []
    applied_budget: dict[str, float] = {}
    identities = {"factor": 1.0, "dx": 0.0, "dy": 0.0}

    def resolve(key: str) -> float:
        """Read the unsigned magnitude already on the prescription.

        Missing keys stay identity (no scale, no shift). ``retry_index`` does not
        change them: a retry of the same prescription is the same move, and
        ``figure_unchanged`` is how a no-op retry is refused.
        """

        value = _budget(action, key, identities.get(key, 0.0))
        applied_budget.setdefault(key, value)
        return value

    if action_type in {"relayout_slots", "resize_nodes", "repair_interfaces"}:
        if not target_nodes and next_layout.get("nodes"):
            target_nodes.add(str(next_layout["nodes"][0].get("id") or ""))
        for index, node in enumerate(next_layout.get("nodes", [])):
            node_id = str(node.get("id") or "")
            if node_id not in target_nodes or not isinstance(node.get("bbox_percent"), dict):
                continue
            factor = resolve("factor")
            dy = resolve("dy")
            next_layout["nodes"][index] = {**node, "bbox_percent": _bounded_bbox(node["bbox_percent"], factor, dx=resolve("dx"), dy=dy)}
            changed_nodes.append(node_id)
    elif action_type == "relayout_global":
        nodes = next_layout.get("nodes", [])
        for index, node in enumerate(nodes):
            if not isinstance(node.get("bbox_percent"), dict):
                continue
            direction = -1 if index % 2 == 0 else 1
            dx = direction * resolve("dx")
            next_layout["nodes"][index] = {**node, "bbox_percent": _bounded_bbox(node["bbox_percent"], resolve("factor"), dx=dx, dy=resolve("dy"))}
            changed_nodes.append(str(node.get("id") or ""))
    if action_type == "regen_assets":
        style = str((domain_profile or {}).get("icon_style") or "domain-appropriate scientific illustration")
        constraint = action.get("icon_prompt_constraint") if isinstance(action.get("icon_prompt_constraint"), dict) else {}
        constraint_slots = {str(item) for item in (constraint.get("slot_ids") or []) if str(item)}
        if constraint_slots:
            target_slots.update(constraint_slots)
        if not target_slots and action.get("semantic_node_ids"):
            target_slots.update("slot_" + str(item) for item in action["semantic_node_ids"])
        revision_text = str(constraint.get("constraint") or action.get("problem") or "improve domain fitness and visual clarity")
        style_note = str(constraint.get("style") or style)
        for index, icon in enumerate(next_icons.get("icons", [])):
            asset_id = str(icon.get("asset_id") or "")
            semantic_ids = set(str(item) for item in icon.get("semantic_node_ids", []))
            if target_slots and asset_id not in target_slots and "slot_" + asset_id not in target_slots and not semantic_ids.intersection(target_nodes):
                continue
            revision = f" {style_note}. Repair objective: {revision_text}. Preserve scientific meaning; no text, arrows, formulas, frame, or watermark."
            next_icons["icons"][index] = {**icon, "prompt": str(icon.get("prompt") or "") + revision, "evolution_revision": retry_index + 1}
            changed_assets.append(asset_id)
    executed = dict(action)
    executed.update(applied_budget)
    return next_layout, next_icons, {"action": executed, "changed_node_ids": changed_nodes, "changed_asset_ids": changed_assets, "retry_index": retry_index, "applied_budget": applied_budget}


def feedback_object_ids(result: dict[str, Any] | None, *, severity: str = "") -> list[str]:
    """Collect the objects a judgement binds its feedback to, so a reason can name them."""

    found: list[str] = []
    for item in (result or {}).get("stage_feedback") or []:
        if not isinstance(item, dict):
            continue
        if severity and str(item.get("severity") or "") != severity:
            continue
        for key in ("semantic_node_ids", "slot_ids", "object_ids", "edge_ids"):
            for value in item.get(key) or []:
                text = str(value).strip()
                if text and text not in found:
                    found.append(text)
    return found


def build_object_binding(action: dict[str, Any] | None, bounded_change: dict[str, Any] | None, figure_diff: dict[str, Any] | None = None) -> dict[str, Any]:
    """Name the objects a round asked to change, touched, and actually moved.

    Diagnosis and Memory may not only say "the figure got better": the same record has to
    carry the node/edge/slot/PPT object it is about. The three lists stay apart on purpose,
    because a request that touched nothing must not read as a repair.
    """

    source = action if isinstance(action, dict) else {}
    change = bounded_change if isinstance(bounded_change, dict) else {}
    figure = figure_diff if isinstance(figure_diff, dict) else {}
    requested: list[str] = []
    for key in ("semantic_node_ids", "slot_ids", "object_ids", "edge_ids"):
        requested.extend(str(item) for item in (source.get(key) or []) if str(item))
    return {
        "action_type": str(source.get("type") or change.get("action_type") or ""),
        "requested_object_ids": list(dict.fromkeys(requested)),
        "touched_node_ids": [str(item) for item in (change.get("changed_node_ids") or []) if str(item)],
        "touched_asset_ids": [str(item) for item in (change.get("changed_asset_ids") or []) if str(item)],
        "program_change_types": [str(item.get("object_type") or "") for item in (change.get("program_changes") or []) if isinstance(item, dict) and str(item.get("object_type") or "")],
        "action_changed": bool(change.get("applied")),
        "figure_changed": figure.get("changed"),
        "figure_changed_object_ids": [str(item) for item in (figure.get("changed_object_ids") or []) if str(item)],
    }


def _coverage_pair(gate: dict[str, Any] | None) -> tuple[float, float] | None:
    """Read node/edge coverage the Grounding Gate measured, or nothing if either is missing."""

    source = gate if isinstance(gate, dict) else {}
    node, edge = source.get("node_coverage"), source.get("edge_coverage")
    if not isinstance(node, (int, float)) or isinstance(node, bool):
        return None
    if not isinstance(edge, (int, float)) or isinstance(edge, bool):
        return None
    return float(node), float(edge)


def _coverage_improved(before_gate: dict[str, Any] | None, after_gate: dict[str, Any] | None, *, threshold: float = 0.01) -> bool:
    """Whether contract coverage rose on at least one axis and fell on neither.

    Coverage is code reading the same graphs twice. It can overrule a pairwise preference
    that would keep the worse binding, but it cannot overrule a new deterministic blocker.
    """

    before, after = _coverage_pair(before_gate), _coverage_pair(after_gate)
    if before is None or after is None:
        return False
    before_nodes, before_edges = before
    after_nodes, after_edges = after
    if after_nodes + 1e-9 < before_nodes or after_edges + 1e-9 < before_edges:
        return False
    return (after_nodes - before_nodes) >= threshold or (after_edges - before_edges) >= threshold


def _item_ledger(before: dict[str, Any] | None, after: dict[str, Any]) -> list[dict[str, Any]]:
    """Every measured item that moved this round, and which way it moved.

    A round is judged as a whole, so a repair that improved four dimensions and worsened one is
    a round to keep and to keep working on. This is the account the next round reads: which
    item moved, which direction, and by how much.
    """

    left = (before or {}).get("dimension_scores") if isinstance((before or {}).get("dimension_scores"), dict) else {}
    right = after.get("dimension_scores") if isinstance(after.get("dimension_scores"), dict) else {}
    ledger: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        try:
            was, now = float(left.get(key) or 0.0), float(right.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
        if abs(now - was) <= 1e-9:
            continue
        ledger.append({
            "item": str(key),
            "before": round(was, 4),
            "after": round(now, 4),
            "delta": round(now - was, 4),
            "direction": "up" if now > was else "down",
        })
    return ledger


def _round_evidence(before: dict[str, Any] | None, after: dict[str, Any], introduced: list[str], preserve: list[str]) -> dict[str, Any]:
    """What the deterministic gate measured this round, kept apart from the verdict.

    The gate no longer refuses a round, so this is where its reading lives: the defect kinds
    that appeared, the items that moved and which way, and the blocker tally on each side.
    """

    return {
        "new_blocking_defects": list(introduced),
        "new_blocking_defect_count": len(introduced),
        "preserve_failures": list(preserve),
        "before_blockers": _blockers(before),
        "after_blockers": _blockers(after),
        "item_ledger": _item_ledger(before, after),
    }


def compare_candidate(before: dict[str, Any] | None, after: dict[str, Any], *, preserve_failures: list[str] | None = None, tolerance: float = 0.01, target_metric: str = "overall_score", action: dict[str, Any] | None = None, bounded_change: dict[str, Any] | None = None, figure_diff: dict[str, Any] | None = None, baseline_ready: bool = True, after_pairwise: dict[str, Any] | None = None, before_grounding: dict[str, Any] | None = None, after_grounding: dict[str, Any] | None = None) -> dict[str, Any]:
    """Accept only blocker-safe target improvement; initial candidates are accepted.

    `target_metric` may name a continuous entry of `dimension_scores`, which lets a
    caller measure repairs finer than the quantized overall score.

    A round is only compared when both sides carry a measurement of the same instrument.
    An unmeasured candidate cannot be an improvement and an unmeasured earlier state
    cannot be a baseline, so the record names which of the two is missing
    (`measurement_failed`, `baseline_unavailable`) instead of subtracting from a number
    nobody measured.

    When `after_pairwise` carries a two-state answer, that answer decides the round and the
    scores below are kept for the trend only: subtracting two numbers the model rounded to a
    0.1 grid measured its rounding, not the difference between the two figures. A comparison
    that could not be taken refuses the round (`pairwise_unavailable`) rather than falling back
    to the very subtraction it replaced. A round that moved nothing is refused before either
    instrument is consulted (`figure_unchanged`).

    Contract coverage (node/edge binding the Grounding Gate already measured) can still keep
    a round the comparison would throw away: if coverage rose on at least one axis, fell on
    neither, and no new deterministic blocker appeared, the worse binding must not be the
    one the loop keeps (`coverage_improved_over_pairwise`). That is not a second score model.

    The regression veto that can still overrule a preferred comparison reads only the
    deterministic half of the blocker list, by defect identity (`_regression`). The model's
    half is prose it rewrites on every call, so counting it put the noisiest signal in the
    room in charge of the verdict the least noisy one had just given.
    """
    binding = build_object_binding(action, bounded_change, figure_diff)
    blocking_objects = feedback_object_ids(after, severity="blocker")
    comparison = after_pairwise if isinstance(after_pairwise, dict) and str(after_pairwise.get("judgement_kind") or "") == "pairwise" else None
    if before is None:
        # The first candidate of a run is a starting point, not an improvement: there is no
        # earlier measurement to subtract, so the delta is left empty and the record says no
        # comparison happened. Reporting the candidate score itself as a delta read as
        # "improved by N" in every consumer of the record, training data included.
        return {
            "accepted": True,
            "reason": "initial_candidate",
            "score_delta": None,
            "comparison": "none",
            "pairwise": None,
            "after_measured": measured_score(after) is not None,
            "new_blockers": 0,
            "preserve_failures": [],
            "target_metric": target_metric,
            "object_binding": binding,
            "blocking_objects": blocking_objects,
        }
    before_blockers, after_blockers = _blockers(before), _blockers(after)
    introduced, new_blockers = _regression(before, after)
    preserve = list(preserve_failures or [])
    before_score, after_score = measured_score(before), measured_score(after)
    uncomparable = {
        "accepted": False,
        "target_metric": target_metric,
        "score_delta": None,
        "before_score": before_score,
        "after_score": after_score,
        "before_blockers": before_blockers,
        "after_blockers": after_blockers,
        "new_blockers": new_blockers,
        "new_blocking_defects": introduced,
        "preserve_failures": preserve,
        "object_binding": binding,
        "blocking_objects": blocking_objects,
        "evidence": _round_evidence(before, after, introduced, preserve),
    }
    # A round that re-emitted the same panels, nodes and arrows, and whose action moved no node
    # and no asset, repaired nothing, however the two measurements read. The comparison cannot
    # settle this: asked twice about one figure a model returns two answers, so a round that
    # changed nothing was accepted whenever the noise fell the right way, and the experience it
    # banked described a repair no reader could see. A diff that was never taken is not a veto,
    # and an action that regenerated an asset is not caught by it, because the object keys the
    # figure diff compares do not carry the icon the asset produced.
    if binding["figure_changed"] is False and not binding["touched_node_ids"] and not binding["touched_asset_ids"]:
        return {**uncomparable, "reason": "figure_unchanged", "comparison": "figure_unchanged"}
    if comparison is not None:
        verdict = pairwise_verdict(comparison)
        if not verdict:
            return {**uncomparable, "reason": "pairwise_unavailable", "comparison": "pairwise_unavailable", "pairwise": comparison}
        preferred = verdict == "after"
        coverage_gain = _coverage_improved(before_grounding, after_grounding)
        # The mechanism does not veto. A round the comparison preferred is accepted, and what
        # the gate found is written beside it as evidence: a defect the comparison judged worth
        # the trade is work for the next round, not a reason to throw this one away. The three
        # factual refusals still refuse -- nothing measured, nothing moved, a mechanism that
        # repaired itself -- because those are questions about the measurement, not the figure.
        # One ruler. The seat that read both figures decides the round; what coverage did is
        # recorded beside the decision and never spends it, in either direction. Two rulers is
        # what let a round the seat refused be kept because a number rose, and it is what made
        # "the model said the new figure is better" mean one thing here and another in the pair
        # comparison -- the same judgement, read two ways, deciding two different rounds.
        accepted = bool(preferred)
        reason = "pairwise_preferred" if accepted else "pairwise_not_preferred"
        delta = round(float(_metric_score(after, target_metric)) - float(_metric_score(before, target_metric)), 4) if before_score is not None and after_score is not None else None
        return {
            "accepted": accepted,
            "reason": reason,
            "comparison": "pairwise",
            "pairwise": comparison,
            "pairwise_verdict": verdict,
            "coverage_improved": coverage_gain,
            "coverage_disagreed": bool(coverage_gain) != bool(preferred),
            "target_metric": target_metric,
            "score_delta": delta,
            "before_score": before_score,
            "after_score": after_score,
            "before_blockers": before_blockers,
            "after_blockers": after_blockers,
            "new_blockers": new_blockers,
            "new_blocking_defects": introduced,
            "preserve_failures": preserve,
            "object_binding": binding,
            "blocking_objects": blocking_objects,
            "evidence": _round_evidence(before, after, introduced, preserve),
        }
    if not baseline_ready or before_score is None:
        return {**uncomparable, "reason": "baseline_unavailable", "comparison": "baseline_unavailable"}
    if after_score is None:
        return {**uncomparable, "reason": "score_measurement_failed", "comparison": "measurement_failed"}
    delta = float(_metric_score(after, target_metric)) - float(_metric_score(before, target_metric))
    accepted = delta > tolerance
    return {
        "accepted": accepted,
        "reason": "target_improved" if accepted else "no_significant_improvement",
        "target_metric": target_metric,
        "score_delta": round(delta, 4),
        "comparison": "compared",
        "before_score": before_score,
        "after_score": after_score,
        "before_target": round(float(_metric_score(before, target_metric)), 4),
        "after_target": round(float(_metric_score(after, target_metric)), 4),
        "before_blockers": before_blockers,
        "after_blockers": after_blockers,
        "new_blockers": new_blockers,
        "new_blocking_defects": introduced,
        "preserve_failures": preserve,
        "object_binding": binding,
        "blocking_objects": blocking_objects,
        "evidence": _round_evidence(before, after, introduced, preserve),
    }


def _decisive_pairwise(decision: dict[str, Any]) -> bool:
    """Whether an accept rested on a paired judgement the acceptance role called clear."""

    if str(decision.get("comparison") or "") != "pairwise":
        return False
    if str(decision.get("pairwise_verdict") or "") != "after":
        return False
    record = decision.get("pairwise") if isinstance(decision.get("pairwise"), dict) else {}
    block = record.get("pairwise") if isinstance(record.get("pairwise"), dict) else {}
    margin = str(record.get("margin") or block.get("margin") or "").strip().casefold()
    if margin != "clear":
        return False
    binding = decision.get("object_binding") if isinstance(decision.get("object_binding"), dict) else {}
    return binding.get("figure_changed") is not False


def measurement_noise_claim(
    reflection: dict[str, Any] | None, decision: dict[str, Any] | None
) -> tuple[bool, str]:
    """Read whether a reflection is calling this round's own measurement noise.

    It is read, and it never spends the round. ``measurement_noise_risk`` used to withdraw an
    accept outright and the older ``randomness_risk`` did so too unless a clear paired judgement
    had decided it -- and the fallback reflection set both on every round that did not improve,
    so the mechanism was withdrawing rounds on the model's behalf. A doubt the seat itself
    raised is one more piece of evidence: it is recorded on the round, and a later round settles
    it. The second element is the note recorded with the doubt.
    """

    source = reflection if isinstance(reflection, dict) else {}
    target = decision if isinstance(decision, dict) else {}
    if bool(source.get("measurement_noise_risk")):
        return False, "a doubt about the verdict is recorded; the comparison still decides the round"
    if bool(source.get("randomness_risk")):
        if _decisive_pairwise(target):
            return False, "the round was decided by a paired judgement the acceptance role called clear"
        return False, "a doubt about the drawing is recorded; the comparison still decides the round"
    return False, ""


def apply_reflection_policy(
    decision: dict[str, Any],
    action: dict[str, Any] | None,
    reflection: dict[str, Any] | None,
    *,
    injected_experience: dict[str, Any] | None = None,
    condition: dict[str, Any] | None = None,
    implementation: dict[str, Any] | None = None,
    pairwise: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Let a reflection's stated next policy change what this round does.

    A reflection is a claim by the model that produced the judgement, so it may only
    tighten: a policy can take the guidance of this round away, but nothing here can create an
    improvement that was not measured, and nothing here takes away one that was. The four
    signals are:

    - ``measurement_noise_risk`` / ``randomness_risk``: a doubt about this round's own
      measurement. It is recorded on the round as ``noise_doubt`` and never spends it: the
      comparison is the instrument that judged the two figures, and the doubt is evidence a
      later round settles. Withdrawing an accept on the model's behalf is what the fallback
      reflection's flags used to do, and it is what the mechanism no longer does.
    - ``next_policy: disable``: the experience this round ran under is disabled, so it is
      out of the retrieval status set immediately.
    - ``next_policy: narrow_scope``: a condition is appended to that experience, so it is
      not retrieved for a state like this one again.
    - ``next_policy: revise``: the reflection's replacement action becomes the next
      round's proposal, but only when it names an action the loop may run, with the parameters
      that action needs. The loop still routes it through Memory, so a revision is not a way
      around injection.
    - ``next_policy: revise_rule``: the reflection states a mechanism-level rule -- one of the
      two mechanism actions with its parameters, no object ids and no coordinates. The rule is
      recorded on the policy, kept for the library that improves later advice, and handed back as
      ``revised_rule_action`` so the next round may propose it once. This round's own bounded move
      is untouched, so stating a rule can never widen what the loop may run.

    - ``next_policy: follow_up``: the round named a reason why the rule it ran did not land, that
      reason was weighed against the compile's own report, and what it wrote is one more attempt --
      the same rule said more precisely, or a repair aimed at the object the comparison faulted.
      Only the cheap layers may answer it, and the budget belongs to the rule rather than to the
      wording, so a reflection that rephrases itself every round still runs out of attempts.

    The returned ``experience_effects`` are descriptions; the caller owns the store and
    applies them, which keeps this module free of persistence.
    """

    source = reflection if isinstance(reflection, dict) else {}
    updated = dict(decision if isinstance(decision, dict) else {})
    injected_id = str((injected_experience or {}).get("experience_id") or "")
    events: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    applied: list[str] = []
    randomness_flagged = bool(source.get("randomness_risk")) or bool(source.get("measurement_noise_risk"))
    _withdraws, noise_note = measurement_noise_claim(source, updated)
    if randomness_flagged and noise_note:
        # Recorded, never spent. The round stays whatever the comparison said, and the doubt
        # travels with it so a later round can settle it.
        updated["noise_doubt"] = True
        events.append(
            {
                "event": "measurement_noise_doubt_recorded",
                "action": str((action or {}).get("type") or ""),
                "reason": noise_note,
                "accepted": bool(updated.get("accepted")),
                "accepted_reason": str(updated.get("reason") or ""),
            }
        )
    policy = str(source.get("next_policy") or "keep").strip() or "keep"
    if policy not in REFLECTION_POLICIES:
        policy = "keep"
    # The reason a losing round gives is read against the records the round already holds,
    # before anything acts on it: the one answer that buys another attempt is the one worth
    # checking, and a claim the evidence does not support is recorded rather than spent.
    attribution, attribution_rejection = verify_attribution(source, implementation, pairwise)
    stated_attribution = str(source.get("attribution") or "").strip()
    if stated_attribution and not attribution:
        events.append(
            {
                "event": "attribution_unsupported",
                "attribution": stated_attribution,
                "reason": attribution_rejection,
            }
        )
    revised: dict[str, Any] = {}
    revised_rule: dict[str, Any] = {}
    revised_rule_action: dict[str, Any] = {}
    follow_up_action: dict[str, Any] = {}
    if policy == "disable" and injected_id:
        effects.append({"kind": "disable", "experience_id": injected_id, "reason": "reflection_next_policy_disable"})
        applied.append("disable")
        events.append({"event": "reflection_policy_applied", "policy": "disable", "experience_id": injected_id})
    elif policy == "narrow_scope" and injected_id:
        excluded = dict(condition or {})
        excluded["reason"] = "reflection_next_policy_narrow_scope"
        effects.append({"kind": "exclude_condition", "experience_id": injected_id, "condition": excluded, "reason": "reflection_next_policy_narrow_scope"})
        applied.append("narrow_scope")
        events.append({"event": "reflection_policy_applied", "policy": "narrow_scope", "experience_id": injected_id, "condition": excluded})
    elif policy == "revise":
        candidate = source.get("revised_action") if isinstance(source.get("revised_action"), dict) else {}
        candidate_type, raw_name = aliased_action_name(candidate)
        # A mechanism-level rule carries parameters where a repair carries object ids, so the
        # parameters are the whole prescription. The reader the executor and the compiler share
        # decides whether they arrived; a bare name is refused here for the same reason a name
        # the loop does not know is, and the two refusals are told apart in the event.
        mechanism_rule = (
            normalize_mechanism_rule({**candidate, "type": candidate_type})
            if candidate_type in MECHANISM_ACTIONS
            else {}
        )
        if candidate_type in MECHANISM_ACTIONS and not mechanism_rule:
            events.append(
                {
                    "event": "revised_action_rejected",
                    "policy": "revise",
                    "requested_action": candidate_type,
                    "reason": "the revised action names a mechanism-level rule but carries none of its parameters",
                }
            )
        elif candidate_type:
            revised = {key: value for key, value in candidate.items() if key != "experience_id"}
            revised["type"] = candidate_type
            scope = _compact_name(revised.get("scope_type") or revised.get("scope"))
            if scope:
                revised["scope_type"] = "slot" if scope in {"slot", "slots"} else scope
            if mechanism_rule:
                revised.update(mechanism_rule)
            applied.append("revise")
            events.append({"event": "reflection_policy_applied", "policy": "revise", "action": candidate_type, **({"aliased_from": raw_name} if _compact_name(raw_name) != candidate_type else {})})
        else:
            # An absent replacement and an unsupported one are both refused, but they are
            # different events: only one of them means the model named an action the loop
            # may not run.
            events.append(
                {
                    "event": "revised_action_rejected",
                    "policy": "revise",
                    "requested_action": str(candidate.get("type") or candidate.get("action") or ""),
                    "reason": "the reflection named no revised action" if not candidate else "the revised action is not one the loop may run",
                }
            )
    elif policy == "revise_rule":
        # Only the mechanism vocabulary is accepted here, and only with parameters attached:
        # a rule that names nothing the compiler reads is refused rather than recorded as a
        # lesson. Nothing about this round's move changes, which is what keeps the policy on
        # the "tighten only" side of the line.
        candidate_rule = source.get("rule") if isinstance(source.get("rule"), dict) else {}
        revised_rule = normalize_mechanism_rule(candidate_rule)
        if revised_rule:
            applied.append("revise_rule")
            events.append({"event": "reflection_policy_applied", "policy": "revise_rule", "rule": revised_rule})
        else:
            events.append(
                {
                    "event": "revised_rule_rejected",
                    "policy": "revise_rule",
                    "requested_rule": candidate_rule,
                    "reason": "the reflection named no mechanism-level rule the compiler can run",
                }
            )
    elif policy == "follow_up":
        # A rule that lost is not thrown away on one reading. The round said which of the two
        # checkable reasons it thinks applies, that claim was just weighed against the compile's
        # report, and what it wrote next is one more attempt: the same rule said more precisely, or
        # a repair aimed at the object the comparison faulted. Only the cheap layers answer here,
        # because the whole reason for asking again is that it costs seconds and not minutes.
        refined = normalize_mechanism_rule(
            source.get("refined_rule") if isinstance(source.get("refined_rule"), dict) else {}
        )
        if attribution not in FOLLOW_UP_ATTRIBUTIONS:
            events.append(
                {
                    "event": "follow_up_rejected",
                    "policy": "follow_up",
                    "attribution": stated_attribution,
                    "reason": "the round's attribution does not earn another attempt",
                    "detail": attribution_rejection,
                }
            )
        elif refined:
            # The rule was carried out badly, so the answer is the same rule said better. It is
            # kept as a stated rule as well, because a rule a later paper could be drawn by is
            # worth recording whether or not this round went right.
            revised_rule = refined
            revised_rule_action = {**refined, "source": "reflection_refined_rule"}
            follow_up_action = dict(revised_rule_action)
            applied.append("follow_up")
            events.append(
                {
                    "event": "follow_up_scheduled",
                    "policy": "follow_up",
                    "attribution": attribution,
                    "kind": "refined_rule",
                    "rule": refined,
                }
            )
        else:
            follow_up = source.get("follow_up") if isinstance(source.get("follow_up"), dict) else {}
            candidate_type, raw_name = aliased_action_name(follow_up)
            klass = action_class(candidate_type)
            if not candidate_type:
                events.append(
                    {
                        "event": "follow_up_rejected",
                        "policy": "follow_up",
                        "attribution": attribution,
                        "reason": "the round named no follow-up repair",
                    }
                )
            elif klass not in (FIGURE_PROGRAM_CLASS, COMPILE_RULE_CLASS):
                events.append(
                    {
                        "event": "follow_up_rejected",
                        "policy": "follow_up",
                        "attribution": attribution,
                        "requested_action": candidate_type,
                        "reason": (
                            "a repair that re-runs a planning stage costs minutes to hours, so it is not an answer to a rule that may only have been carried out badly"
                            if klass
                            else "the named follow-up is not one the loop may run"
                        ),
                    }
                )
            else:
                proposed = {key: value for key, value in follow_up.items() if key != "experience_id"}
                proposed["type"] = candidate_type
                proposed["source"] = "reflection_follow_up"
                mechanism = normalize_mechanism_rule(proposed) if klass == COMPILE_RULE_CLASS else {}
                if klass == COMPILE_RULE_CLASS and not mechanism:
                    events.append(
                        {
                            "event": "follow_up_rejected",
                            "policy": "follow_up",
                            "attribution": attribution,
                            "requested_action": candidate_type,
                            "reason": "the follow-up names a compile-time rule but carries none of its parameters",
                        }
                    )
                else:
                    if mechanism:
                        proposed.update(mechanism)
                    follow_up_action = proposed
                    applied.append("follow_up")
                    events.append(
                        {
                            "event": "follow_up_scheduled",
                            "policy": "follow_up",
                            "attribution": attribution,
                            "kind": klass,
                            "action": candidate_type,
                            **({"aliased_from": raw_name} if _compact_name(raw_name) != candidate_type else {}),
                        }
                    )
    if not revised_rule and policy != "follow_up":
        # A stated rule and the round's next policy are two different statements, and the
        # prompt asks for both. Requiring the model to also change ``next_policy`` dropped
        # every rule a reflection wrote while leaving ``next_policy`` at ``keep`` -- which is
        # why the suggestion library never filled. A rule names no figure and changes nothing
        # about this round's move, so promoting a stated rule stays on the tighten-only side.
        stated = source.get("rule") if isinstance(source.get("rule"), dict) else {}
        promoted = normalize_mechanism_rule(stated)
        if promoted:
            if policy == "disable":
                events.append({"event": "revised_rule_withheld", "policy": "disable", "rule": promoted, "reason": "the reflection disabled this round's guidance, so its rule is not kept"})
            else:
                revised_rule = promoted
                applied.append("revise_rule")
                if policy != "revise":
                    policy = "revise_rule"
                events.append({"event": "reflection_policy_applied", "policy": "revise_rule", "because": "a rule was stated and the policy field left it at " + str(source.get("next_policy") or "keep"), "rule": promoted})
    if revised_rule:
        # A rule the reflection stated is also a move the loop can make: the next round proposes
        # it once, through the ordinary judgement, so a rule that keeps helping is measured
        # rather than only counted, and one that does not help costs one round and stops.
        revised_rule_action = {**revised_rule, "source": "reflection_rule"}
    policy_record = {
        "next_policy": policy,
        "applied": applied,
        "randomness_flagged": randomness_flagged,
        # The doubt is recorded on the round; the round is not withdrawn on it.
        "measurement_noise_doubt": bool(updated.get("noise_doubt")),
        "measurement_noise_note": noise_note,
        "revised_action": revised or None,
        "revised_rule": revised_rule or None,
        "revised_rule_action": revised_rule_action or None,
        "follow_up_action": follow_up_action or None,
        "attribution": attribution,
        "attribution_rejection": attribution_rejection,
        "injected_experience_id": injected_id,
        "condition": dict(condition or {}),
    }
    updated["reflection_policy"] = policy_record
    return {"decision": updated, "policy": policy_record, "events": events, "experience_effects": effects, "revised_action": revised, "revised_rule": revised_rule, "revised_rule_action": revised_rule_action, "follow_up_action": follow_up_action, "attribution": attribution, "attribution_rejection": attribution_rejection}


def validate_reflection_rule(
    reflection: dict[str, Any] | None, *, figure: dict[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a reflection's rule where it is written, and refuse one that cannot travel.

    ``normalize_mechanism_rule`` refuses a rule the compiler cannot run;
    ``case_object_evidence`` refuses one that names the figure it came from, by object id, by
    coordinate, or by one of that figure's own labels. Either refusal empties the rule and
    hands the reason back, so the round records why the rule was dropped instead of the
    library looking empty for a cause nobody can read off the run. A reflection that stated no
    rule keeps its empty rule and reports no refusal.
    """

    source = dict(reflection) if isinstance(reflection, dict) else {}
    stated = source.get("rule") if isinstance(source.get("rule"), dict) else {}
    if not stated:
        source["rule"] = {}
        return source, {}
    evidence = case_object_evidence(stated, figure=figure)
    if evidence:
        source["rule"] = {}
        return source, {"reason": "rule_binds_to_case_objects", "rule": stated, "evidence": evidence}
    normalized = normalize_mechanism_rule(stated)
    if not normalized:
        source["rule"] = {}
        return source, {"reason": "not_a_mechanism_rule", "rule": stated}
    # The prose is the one part a model writes and the only prose the library keeps, so it is
    # carried onto the normalized rule rather than dropped with the parameters the compiler
    # does not read. It was checked against the figure above, before anything was kept.
    prose = {key: stated[key] for key in ("guidance", "advice") if str(stated.get(key) or "").strip()}
    source["rule"] = {**normalized, **prose}
    return source, {}


def default_reflection(decision: dict[str, Any], action: dict[str, Any], *, retry_index: int) -> dict[str, Any]:
    improved = bool(decision.get("accepted"))
    return normalize_reflection({
        "outcome": "improved" if improved else "regressed",
        "failure_type": "none" if improved else "new_blocker" if decision.get("new_blockers") else "overfit_or_bad_scope",
        "why_changed": "The bounded action improved the frozen score without adding blockers." if improved else "The proposed experience did not produce a significant isolated improvement; it may be too broad, domain-specific, or overfit to the source case.",
        "scope_assessment": "appropriate" if improved else "too_broad",
        "domain_mismatch": False,
        # The mechanism does not call its own reading noise on the model's behalf. These two
        # flags used to be set on every round that did not improve, and they were the noise
        # claim that withdrew those rounds; a round nobody wrote a reflection for says so
        # instead, and a doubt has to come from the seat to exist at all.
        "randomness_risk": False,
        "measurement_noise_risk": False,
        "reflection_source": "default",
        "next_policy": "keep" if improved else "narrow_scope" if retry_index < 2 else "disable",
        "revised_action": {**action, "retry_index": retry_index + 1},
    })


def experience_id(source_case: str, action: dict[str, Any], round_index: int) -> str:
    raw = f"{source_case}|{round_index}|{json.dumps(action, sort_keys=True, ensure_ascii=False)}"
    return "direct-exp-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
