"""Build portable experience records from accepted direct-layout repairs."""

from __future__ import annotations

import hashlib
from typing import Any

from ...analysis.graph.defect_patterns import defect_patterns
from ...analysis.graph.retrieval import count_bucket as retrieval_count_bucket
from ...analysis.graph.subgraphs import extract_subgraphs
from ...contracts.evolution import DERIVED_SCORE_SOURCE, EXPERIENCE_ACTIONS, EXPERIENCE_SCHEMA_VERSION, introduced_defects
from ...contracts.memory_parameters import icon_is_transferable, stamp_executable_parameters
from ...contracts.repair_actions import prescription_matches_heuristic_default
from .controller import ACTION_BUDGET_KEYS
from .rule_binding import render_rule_statement, split_rule_binding

EXPERIENCE_SCOPES = {"global", "slot", "interface", "connector"}
SCOPE_ALIASES = {"compile": "global", "semantic": "global"}
DEFAULT_MATCH_THRESHOLD = 0.6
DEFAULT_PROVENANCE_SOURCE = "direct_evolution_process"

#: The actions the mechanism's own policy proposes. A round that ran one of these was not the
#: advisory Judge's repair, so it is not a lesson the library may claim: the policy proposes
#: the same move with or without Memory, and filing it as a discovery is how a self-repair
#: became an "experience" that could never transfer to another paper.
#:
#: ``color_feedback`` is the theme planner reacting to the colour it just measured -- the same
#: always-on path as the decomposability policy, one stage later. It was missing here, so every
#: colour round was filed as an experience; none of them could carry an applicability (no
#: proposal wrote one), and each one stopped the validation ladder at ``applicability_unspecified``.
MECHANISM_POLICY_SOURCES = frozenset({"decomposition_policy", "color_feedback"})

# Which kind of Memory a bounded repair produces, and the stage a Router must inject it at.
ACTION_MEMORY_PROFILE = {
    "reanalyze_semantics": ("scientific", "global_planning"),
    "replan_structure": ("layout", "global_planning"),
    "resegment": ("segmentation", "segmentation"),
    "relayout_global": ("layout", "global_planning"),
    "relayout_slots": ("layout", "ppt_compilation"),
    "resize_nodes": ("density", "ppt_compilation"),
    "reroute": ("connector", "ppt_compilation"),
    "repair_interfaces": ("connector", "ppt_compilation"),
    "regen_assets": ("icon", "icon_generation"),
    # The theme planner owns colour, so a theme round is recorded as theme Memory that has to
    # be injected at theme_planning; both names were already legal contract values.
    "replan_theme": ("theme", "theme_planning"),
    # The mechanism-level rules are read where the figure is drawn, so both are injected at
    # ppt_compilation. They name a rule, not this figure's objects, which is what lets the row
    # travel to the next paper at all.
    "router_constraint": ("connector", "ppt_compilation"),
    "layout_constraint": ("layout", "ppt_compilation"),
}


#: The lifecycle fields a library row keeps beside its rule. Every one of these is a verdict,
#: a count, a score or an identity -- none names an object of the figure the row was measured
#: on. Everything that does name one goes into the binding, which stays in the run directory,
#: so the library holds a claim that can travel and the evidence stays where it belongs.
LIBRARY_ROW_KEYS = (
    "schema_version",
    "domain",
    "experience_id",
    "trigger",
    "preserve",
    "forbidden_changes",
    "judge_fingerprint",
    "source_case",
    "source_round",
    "accepted",
    "repair_gain",
    "comparison",
    "pairwise_verdict",
    "provenance",
    "outcome",
    "measurement_instrument",
    "validated_cases",
    "failed_cases",
    "side_effects",
    "strategy_id",
)


def experience_library_record(experience: dict[str, Any], *, binding_ref: str = "") -> dict[str, Any]:
    """Save one row as the rule that travels, with a pointer to the binding that stays.

    The builder may still call an accepted measurement ``current_case_validated``. The library
    keeps that row as ``candidate`` until a paired validation walks the ladder for its
    applicability. A single paper's retries do not become ``overfit_or_failed``.

    A row whose object ids stayed on it could never be retrieved for another paper -- the next
    figure has no ``rel_12`` -- so the ids, the boxes and the graph scope are split off into the
    binding, which the caller writes to the run directory and points at with ``binding_ref``.
    """

    status = str(experience.get("status") or "candidate")
    if status in {"current_case_validated", "overfit_or_failed"}:
        status = "candidate"
    rule, _binding = split_rule_binding(experience)
    row = {key: experience[key] for key in LIBRARY_ROW_KEYS if key in experience}
    row.update(rule)
    row["status"] = status
    if binding_ref:
        row["binding_ref"] = str(binding_ref)
    hypothesis = str(row.get("hypothesis") or "")
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    row["strategy"] = {"instruction": hypothesis, "action": action}
    return row


def experience_binding_record(experience: dict[str, Any]) -> dict[str, Any]:
    """Return the half of a row that belongs to the figure it was measured on.

    This is not a second copy of the row: it is what the row would have carried if it had
    stayed a description of one drawing. It stays in the run directory as the evidence of what
    the measurement was about.
    """

    _rule, binding = split_rule_binding(experience)
    binding["experience_id"] = str(experience.get("experience_id") or "")
    return binding


def written_applicability(action: dict[str, Any] | None, domain_profile: dict[str, Any] | None = None) -> dict[str, str]:
    """Read the applicability the proposal already wrote onto the action.

    The advisory answer writes common or domain when it names the repair. A creator
    rule may already have copied one agreed label onto that same action. Nothing
    here looks at motifs or at how much two subgraphs overlap. A domain label also
    needs the paper's own domain; a paper with no domain cannot be stored as a
    domain experience.
    """

    raw = str((action or {}).get("applicability") or "").strip().casefold()
    profile = domain_profile if isinstance(domain_profile, dict) else {}
    paper_domain = str(profile.get("domain") or "").strip()
    missing = bool(profile.get("domain_missing")) or paper_domain.casefold() in {"", "unknown", "missing"}
    if raw not in {"common", "domain"}:
        return {"applicability": "", "applicability_domain": "", "applicability_reason": "no proposal wrote an applicability"}
    if raw == "domain" and (missing or not paper_domain):
        return {"applicability": "", "applicability_domain": "", "applicability_reason": "a domain experience needs a named paper domain"}
    if raw == "domain":
        return {"applicability": "domain", "applicability_domain": paper_domain, "applicability_reason": ""}
    return {"applicability": "common", "applicability_domain": "", "applicability_reason": ""}


#: The stages whose repairs are about how a figure is drawn rather than about what the paper says.
#: A rule acting at one of them answers the same defect on any paper, so a proposal that wrote no
#: applicability is still readable as one that travels.
COMPOSITION_STAGES = ("ppt_compilation", "repair")


def resolved_applicability(
    action: dict[str, Any] | None, *, apply_at: str = "", domain_profile: dict[str, Any] | None = None
) -> dict[str, str]:
    """Where a row applies, from what the proposal wrote or from what its stage implies.

    One reader for both the library and the validation harness, because they used to disagree:
    the library stored whatever the proposal happened to write, and the ladder refused a row that
    wrote nothing with ``applicability_unspecified``. Three rows were retired that way, and none of
    them was a verdict about the row -- "no measurement was possible" had been recorded as "this
    lesson does not work". A move that acts on the composition is about drawing, and the defect it
    answers looks the same on the next paper, so it is *common*; a move whose effect depends on
    what this paper says is *domain*, and needs the paper's own domain to be named at all.
    """

    written = written_applicability(action, domain_profile)
    if written["applicability"]:
        return written
    if str(written.get("applicability_reason") or "") != "no proposal wrote an applicability":
        # The proposal did say where it applies; what is missing is the paper's own domain, which
        # is a fact about this run rather than about the lesson. A move the seat called domain is
        # not turned into one that travels by a run that could not name its paper's field.
        return written
    stage = str(apply_at or "").strip()
    if not stage:
        profile = ACTION_MEMORY_PROFILE.get(str((action or {}).get("type") or ""))
        stage = str(profile[1]) if profile else ""
    if stage in COMPOSITION_STAGES:
        return {
            "applicability": "common",
            "applicability_domain": "",
            "applicability_reason": "derived from the stage this move acts at: a repair on the composition answers the same defect on another paper",
        }
    # Anything else stays as the proposal wrote it. A move that reads the paper's own content is
    # not one this can decide for the round: "which papers may use this" is a claim about the
    # lesson, and the only honest source of it up here is the answer that proposed the move.
    return written


def normalize_experience_scope(scope_type: Any) -> str:
    """Map an action scope_type onto the narrower scope vocabulary the contract accepts.

    `select_bounded_action` may set scope_type from a feedback stage (`compile`,
    `semantic`), which `validate_experience` rejects outright.
    """
    value = str(scope_type or "global").strip().casefold()
    value = SCOPE_ALIASES.get(value, value)
    return value if value in EXPERIENCE_SCOPES else "global"


def stage_for_action(action_type: Any) -> str:
    """Return the stage whose repair an action's Memory has to be injected at."""

    return ACTION_MEMORY_PROFILE.get(str(action_type or "").strip(), ("repair", "repair"))[1]


# The bucket vocabulary lives in the analysis layer, where the record is read; re-exporting it
# here keeps the writer and the reader using one definition instead of two that drifted apart.
count_bucket = retrieval_count_bucket


def exclusion_context(layout: dict[str, Any] | None, domain_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Describe the state a repair failed under, so the row it came from can be excluded here.

    The keys are exactly the ones retrieval can compare (topology, the two size buckets and,
    when the run resolved one, the domain), so an appended condition matches the states it
    should and leaves every other state alone.
    """

    source = layout if isinstance(layout, dict) else {}
    domain = str((domain_profile or {}).get("domain") or "").strip()
    context = {
        "topology": str(source.get("topology") or "unknown"),
        "node_count_bucket": count_bucket(len(source.get("nodes") or [])),
        "connector_count_bucket": count_bucket(len(source.get("connectors") or [])),
    }
    patterns = defect_patterns(source)
    if patterns:
        context["defect_patterns"] = patterns
    if domain and domain != "unknown":
        context["domain"] = domain
    return context


def _touched_object_ids(bounded_change: dict[str, Any] | None) -> list[str]:
    """Report the ids the executor touched, with no fallback to the ids it was asked for.

    The fallback is what let a request that matched nothing read as "changed four objects".
    The requested ids are kept, but in their own field.
    """
    change = bounded_change or {}
    touched = [str(item) for item in (change.get("changed_node_ids") or []) if str(item)]
    touched += [str(item) for item in (change.get("changed_asset_ids") or []) if str(item)]
    return sorted(dict.fromkeys(touched))


def _requested_object_ids(action: dict[str, Any]) -> list[str]:
    """Report every object an action asked to reach, across the typed target fields."""
    requested: list[str] = []
    for key in ("object_ids", "semantic_node_ids", "edge_ids", "slot_ids"):
        requested += [str(item) for item in (action.get(key) or []) if str(item)]
    return sorted(dict.fromkeys(requested))


def _action_diff(bounded_change: dict[str, Any] | None) -> dict[str, Any]:
    """Return the diff of the action itself, which is what this repair measured."""
    change = bounded_change or {}
    diff = change.get("object_diff")
    return dict(diff) if isinstance(diff, dict) else {}


def _transferable_guidance(action: dict[str, Any] | None, reflection: dict[str, Any] | None) -> str:
    """The one part of a rule a model writes: how to apply the move, and when not to.

    The advisory Judge writes it beside the defect it found, and the reflection may restate
    it. The condition is computed and the move comes from a closed vocabulary, so this is the
    only prose in the row -- which is why it is the only prose the library keeps.
    """

    source = action if isinstance(action, dict) else {}
    stated = str(source.get("transferable_advice") or "").strip()
    if stated:
        return stated
    reflection_source = reflection if isinstance(reflection, dict) else {}
    rule = reflection_source.get("rule")
    if isinstance(rule, dict):
        stated = str(rule.get("guidance") or rule.get("advice") or "").strip()
        if stated:
            return stated
    # The sentence a later stage prompt should have carried is the same sentence a later paper is
    # drawn by: both say which requirement was left unwritten and which way to write it. A
    # reflection that states a mechanism rule and no guidance has said it in the patch instead, and
    # the row used to be filed with an empty prose half for that reason -- which the library gate
    # then refused as having nothing that travels, so a round the loop had just measured improved
    # neither entered the library nor reached another paper. The patch's own wording is read here
    # for the same reason the guidance is: the model wrote it, and it is the only part that travels.
    patch = reflection_source.get("prompt_patch")
    if isinstance(patch, dict):
        return str(patch.get("directive") or "").strip()
    return ""


def _reported_reflection(reflection: dict[str, Any] | None) -> dict[str, Any]:
    """Keep the reflection's own words, labelled as a claim rather than a measurement."""
    source = reflection if isinstance(reflection, dict) else {}
    return {
        "verified": False,
        "note": "The reflection's own statements. Only the conditions the Builder could check against a measurement entered this record's conditions.",
        "outcome": str(source.get("outcome") or ""),
        "failure_type": str(source.get("failure_type") or ""),
        "why_changed": str(source.get("why_changed") or ""),
        "scope_assessment": str(source.get("scope_assessment") or ""),
        "domain_mismatch": bool(source.get("domain_mismatch", False)),
        "randomness_risk": bool(source.get("randomness_risk", False)),
        "measurement_noise_risk": bool(source.get("measurement_noise_risk", False)),
        "next_policy": str(source.get("next_policy") or ""),
    }


def _measured_footprint(touched_ids: list[str], layout: dict[str, Any]) -> dict[str, Any]:
    """Report where a repair actually landed, so a scope claim can be checked against it."""
    nodes = {str(item.get("id") or ""): item for item in layout.get("nodes", []) if isinstance(item, dict)}
    connectors = {str(item.get("id") or ""): item for item in layout.get("connectors", []) if isinstance(item, dict)}
    panels = sorted({str(nodes[item].get("panel_id") or "") for item in touched_ids if item in nodes and str(nodes[item].get("panel_id") or "")})
    return {
        "touched_node_ids": [item for item in touched_ids if item in nodes],
        "touched_connector_ids": [item for item in touched_ids if item in connectors],
        "panel_ids": panels,
    }


def _verified_scope(scope: str, reflection: dict[str, Any] | None, touched_ids: list[str], layout: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Narrow the scope only when the reflection's claim survives the measured footprint.

    A reflection may call its own repair too broad. That claim enters the record's scope
    only when the executor's own footprint supports it: the repair either lived inside one
    panel, or only ever touched connectors. Anything wider keeps the scope the action
    declared and records why the claim was refused.
    """
    claim = str((reflection or {}).get("scope_assessment") or "")
    if claim != "too_broad":
        return scope, {"narrowed": False, "claim": claim, "reason": "the reflection did not claim a narrower scope"}
    footprint = _measured_footprint(touched_ids, layout)
    if len(footprint["panel_ids"]) == 1 and footprint["touched_node_ids"]:
        return "slot", {"narrowed": True, "claim": claim, "panel_id": footprint["panel_ids"][0], "evidence": "every touched node sits in one panel"}
    if footprint["touched_connector_ids"] and not footprint["touched_node_ids"]:
        return "connector", {"narrowed": True, "claim": claim, "evidence": "the repair only ever touched connectors"}
    return scope, {"narrowed": False, "claim": claim, "panel_ids": footprint["panel_ids"], "reason": "the measured footprint is wider than the claimed scope"}


def _motifs_for_changes(visual_graph: dict[str, Any] | None, changed_ids: list[str]) -> list[str]:
    """Read the motifs of the subgraphs the repair touched, falling back to the whole graph."""
    if not isinstance(visual_graph, dict) or not visual_graph:
        return []
    try:
        subgraphs, _ = extract_subgraphs(visual_graph, include_unknown=True)
    except Exception:
        return []
    targets = set(changed_ids)
    touched = [item for item in subgraphs if str((item.get("properties") or {}).get("center_node_id") or "") in targets]
    source = touched or subgraphs
    return sorted({str(item.get("motif") or "unknown") for item in source})


def _problem_metrics(previous: dict[str, Any]) -> dict[str, float]:
    """Capture the continuous state the repair was applied to, for later matching."""
    dimensions = previous.get("dimension_scores") if isinstance(previous.get("dimension_scores"), dict) else {}
    return {str(key): round(float(value), 4) for key, value in dimensions.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _introduced_blockers(previous: dict[str, Any] | None, current: dict[str, Any] | None) -> tuple[list[str], int]:
    """Which defects this round introduced over the round it replaced, and how many.

    The round decision asks the same question out of the same two answers, so the rule that
    answers it lives in one place (``introduced_defects``) instead of one per caller: two rulers
    for one question is how a round came to be accepted by the coherence check and refused by
    the library. This is a reading, not a verdict -- a defect this round introduced is recorded
    on the row and spent by the validation ladder, so it no longer needs to drop the row.
    """

    reading = introduced_defects(previous, current)
    return list(reading["defects"]), int(reading["count"])


def _with_applied_budget(action: dict[str, Any] | None, bounded_change: dict[str, Any] | None) -> dict[str, Any]:
    """Copy the magnitudes the executor used onto the action a later paper will replay.

    The request the Judge issued names the move and not the step size. The step size is
    computed from ``retry_index`` at execution time, and the paper that injects this row
    starts that counter at zero. Without the magnitudes on the action, injection falls
    back to the shipped defaults and draws the same figure the default policy already draws.
    """

    recorded = dict(action or {})
    change = bounded_change if isinstance(bounded_change, dict) else {}
    sources = [
        change.get("applied_budget") if isinstance(change.get("applied_budget"), dict) else {},
        change.get("action") if isinstance(change.get("action"), dict) else {},
    ]
    for key in ACTION_BUDGET_KEYS:
        if _number(recorded.get(key)) is not None:
            continue
        for source in sources:
            value = _number(source.get(key))
            if value is not None:
                recorded[key] = value
                break
    return recorded


def _state_measurement(state: dict[str, Any] | None) -> tuple[float | None, str]:
    """Read the one measurement a round's state carries, and name the instrument that took it.

    Two instruments can measure a state the loop kept. The scoring role owns the first number,
    and an acceptance seat that answers only the two-state question leaves none -- which is
    exactly the product's choice seat, and the reason every round used to arrive here with no
    measurement at all. The deterministic gate runs on every state before any model is asked, so
    its score is the second and it is always there. Whichever reported is the measurement, and
    the row says which one: a gate score and a model's score are not the same evidence, and a
    later reader must not have to guess which one it is holding.
    """

    body = state if isinstance(state, dict) else {}
    # A defect-table number is not a measurement of the figure, so it is read past: the
    # row falls to the deterministic gate, which measured this same figure on its own.
    if str(body.get("score_source") or "") != DERIVED_SCORE_SOURCE and body.get("score_measured") is not False:
        value = body.get("overall_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), "acceptance_score"
    value = body.get("gate_score")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), "deterministic_gate"
    return None, ""


def _instrument_label(before: str, after: str) -> str:
    """Name the instrument a gain was taken with, and say so when the two sides disagree.

    A gain is only a gain when one instrument measured both ends of it. Two different
    instruments still leave a difference, and it is recorded, but the label has to say that the
    number is not a like-for-like reading, because that is what a later reader would assume.
    """

    if not before or not after:
        return after or before or "unmeasured"
    if before == after:
        return after
    return f"{before}_to_{after}"


#: What one measured row concludes about the move it ran. Only ``worse`` becomes a refutation.
#: ``unmeasured`` keeps whatever the model wrote and states no result of its own, because a round
#: the Gate rolled back has shown nothing about the move when the reading it rested on was
#: withdrawn, or when the reason the figure got worse was that the rule never reached it.
EXPERIENCE_VERDICTS = ("improved", "worse", "unmeasured")


def experience_verdict(reflection: dict[str, Any] | None, *, accepted: bool) -> str:
    """Read what the round itself concluded about the move it just ran.

    The Gate decides whether the figure is kept; the reflection decides what the round taught. The
    two used to be conflated: a rolled-back round wrote "do not reuse this move" whatever the
    reason for the rollback, which turned "the measurement was untrustworthy" and "the compiler
    never carried the rule out" into evidence that the move itself had failed. Neither of those
    has shown that, and a loss that was never measured is not a loss.
    """

    if accepted:
        return "improved"
    source = reflection if isinstance(reflection, dict) else {}
    if bool(source.get("measurement_noise_risk")):
        # The seat doubts the very reading this rollback rests on. A doubtful reading has shown
        # nothing about the move, so the row states no result of its own instead of inventing a loss.
        return "unmeasured"
    if str(source.get("attribution") or "").strip() == "not_carried_out":
        # The round says the rule is right and the figure never obeyed it. That is a reason to run
        # the same lesson again, not a reason to write the move off.
        return "unmeasured"
    if str(source.get("outcome") or "").strip() in {"improved", "inconclusive"}:
        # The round says the figure moved forward, or could not tell which way it moved, and
        # something other than this move rolled the round back. Neither is evidence against the move.
        return "unmeasured"
    # The round says the figure got worse -- or the caller handed the builder no reading at all, so
    # the Gate's rollback is the only one there is. Only then is the move written up as disproved.
    return "worse"


def build_direct_experience(*, round_index: int, previous: dict[str, Any], current: dict[str, Any], layout: dict[str, Any], action: dict[str, Any] | None = None, reflection: dict[str, Any] | None = None, source_case: str = "", **extra: Any) -> dict[str, Any] | None:
    return build_direct_experience_result(round_index=round_index, previous=previous, current=current, layout=layout, action=action, reflection=reflection, source_case=source_case, **extra)["experience"]


def build_direct_experience_result(
    *,
    round_index: int,
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    layout: dict[str, Any],
    action: dict[str, Any] | None = None,
    reflection: dict[str, Any] | None = None,
    source_case: str = "",
    bounded_change: dict[str, Any] | None = None,
    compiled_state_diff: dict[str, Any] | None = None,
    figure_diff: dict[str, Any] | None = None,
    visual_graph: dict[str, Any] | None = None,
    domain_profile: dict[str, Any] | None = None,
    judge_fingerprint: dict[str, Any] | None = None,
    provenance_source: str = DEFAULT_PROVENANCE_SOURCE,
    decomposability_violations: list[dict[str, Any]] | None = None,
    color_metrics: dict[str, Any] | None = None,
    color_theme: dict[str, Any] | None = None,
    icon_mode: str = "",
    round_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one candidate Memory, reporting why a repair could not become one.

    A repair that measured no improvement is no longer dropped: it becomes a ``refuted``
    experience that carries the negative gain, the measurement it came from and the
    condition it must not be reused under. A repair that still leaves blockers is still
    refused, because its effect cannot be attributed to the action.

    A repair with no earlier accepted state is refused outright: a gain is a difference, and
    a missing measurement is not a zero to subtract from. That distinction is the whole
    reason a first draft may not become Memory.

    A round that won by drawing the figure again is not a refusal either, even though there is
    nothing to keep: its move is not one the catalogue can hand another round. It is left out of
    the library the same way a self-repair is, and the record says which of the two happened. The
    two refusals above come first, because a round that measured nothing did not produce a reading
    for any move to be judged by.

    ``accepted`` follows the same instrument that decided the round. Pairwise preference is
    that instrument when ``round_decision`` carries it; ``repair_gain`` stays the absolute
    score delta and is never re-used to overturn the round.
    """

    policy_source = str((action or {}).get("source") or "").strip()
    if policy_source in MECHANISM_POLICY_SOURCES:
        # The mechanism proposed this move on its own policy, so the round is not evidence about
        # the advisory Judge's advice. The run records it as a self-repair; the library is left
        # alone, and so is the prompt-patch store, because a rule the pipeline already applies
        # on its own is not a finding.
        return {
            "ok": False,
            "reason": "mechanism_self_repair",
            "experience": None,
            "mechanism_source": policy_source,
            "note": "The round ran the mechanism's own policy, not the advisory Judge's repair; it is recorded as a self-repair and never as an experience.",
        }

    earlier = previous if isinstance(previous, dict) else {}
    measured_after, after_instrument = _state_measurement(current)
    if measured_after is None:
        # A row is a measurement of two states. When every instrument that could have read the
        # kept figure reported nothing, the arithmetic below would compare against a placeholder
        # zero and write a gain nobody observed, so the round is refused instead.
        return {"ok": False, "reason": "no_after_measurement", "experience": None, "note": "Memory needs the round to have been scored.", "before_score": None, "after_score": None}
    after = float(measured_after)
    measured_before, before_instrument = _state_measurement(earlier)
    if measured_before is None:
        # The arithmetic below would compare this candidate against a placeholder zero and
        # report its whole score as a gain, which is exactly how a first draft started reading
        # as "improved by 0.75" in every consumer of the record.
        return {"ok": False, "reason": "no_previous_accepted_round", "experience": None, "note": "Memory needs an earlier accepted round to compare against.", "before_score": None, "after_score": after}
    before = float(measured_before)
    measurement_instrument = _instrument_label(before_instrument, after_instrument)
    declared_action = str((action or {}).get("type") or (action or {}).get("action") or "").strip()
    if declared_action and declared_action not in EXPERIENCE_ACTIONS:
        # The round won by drawing the figure again rather than by a repair the catalogue holds, so
        # there is no move to hand another round and no rule to keep. What it is not is a refusal:
        # a row carrying that move could only ever be refused at the store, so writing it as one put
        # a "rejected" notice on every accepted redraw and buried the refusals that do mean
        # something. The two refusals above stay ahead of it deliberately -- a round nothing
        # measured, or one with no earlier accepted state, produced no reading at all, and the
        # reason belongs to that rather than to the move the round happened to carry.
        return {
            "ok": False,
            "reason": "not_a_catalogue_repair",
            "experience": None,
            "action": declared_action,
            "note": "The round won by planning the figure again rather than by a repair in the catalogue, so there is no rule to keep.",
            "before_score": before,
            "after_score": after,
        }

    decision = round_decision if isinstance(round_decision, dict) else {}
    comparison = str(decision.get("comparison") or "")
    pairwise_verdict = str(decision.get("pairwise_verdict") or "")
    if comparison:
        accepted = bool(decision.get("accepted"))
    else:
        accepted = after > before
    # A defect this round introduced no longer drops the row. The round measured a gain, and the
    # measurement is what the row is for; refusing it threw the whole reading away over one
    # defect, and the library stayed empty while the figures kept improving. The defect is
    # written on the row below, carried by the ladder, and spent there: the row may not reach
    # another paper until a later validation shows the defect does not reproduce.
    introduced, new_blockers = _introduced_blockers(earlier, current)

    selected_action = stamp_executable_parameters(
        _with_applied_budget(action or {"type": "relayout_global"}, bounded_change),
        bounded_change=bounded_change,
        color_theme=color_theme,
        layout=layout,
        icon_mode=icon_mode,
    )
    if str(selected_action.get("type") or "") == "regen_assets" and str(icon_mode or selected_action.get("icon_mode") or "").strip() and not icon_is_transferable(selected_action, icon_mode=icon_mode or str(selected_action.get("icon_mode") or "")):
        return {
            "ok": False,
            "reason": "icon_mode_not_transferable",
            "experience": None,
            "note": "regen_assets is only Memory when Image2 actually redrew the slot; a placeholder cannot migrate.",
        }
    if prescription_matches_heuristic_default(selected_action):
        return {
            "ok": False,
            "reason": "indistinguishable_from_default_policy",
            # Not a finding, but not nothing either: the run has just confirmed that the shipped
            # default already does this, which is what keeps a pair of identical arms from being
            # filed as evidence about a repair.
            "mark": "common_knowledge",
            "experience": None,
            "note": "This move is the shipped first-retry budget; injecting it cannot differ from an arm that was not given Memory.",
        }
    action_type = str(selected_action.get("type") or "relayout_global")
    memory_type, apply_at = ACTION_MEMORY_PROFILE.get(action_type, ("repair", "repair"))
    topology = str(layout.get("topology") or "unknown")
    domain = str((domain_profile or {}).get("domain") or "unknown")
    context = {
        "topology": topology,
        "node_count_bucket": count_bucket(len(layout.get("nodes", []))),
        "connector_count_bucket": count_bucket(len(layout.get("connectors", []))),
    }
    # The shape of the illness, not just its size: the two buckets alone cannot tell a
    # four-way fan-out from a tidy pipeline, so a row could not recognise a different paper
    # that has the same defect. Computed from the structure, never from a model, and left out
    # entirely when the figure has no defect of a named shape.
    patterns = defect_patterns(layout)
    if patterns:
        context["defect_patterns"] = patterns

    touched_ids = _touched_object_ids(bounded_change)
    requested_ids = _requested_object_ids(selected_action)
    action_diff = _action_diff(bounded_change)
    required_motifs = _motifs_for_changes(visual_graph, touched_ids or requested_ids)
    changed_fields = [str(item) for item in (action_diff.get("changed_fields") or [])]
    compiled_changed_fields = [str(item) for item in ((compiled_state_diff or {}).get("changed_fields") or [])]
    problem_metrics = _problem_metrics(earlier)
    trigger = {
        "domain": domain,
        "topology": topology,
        "motifs": required_motifs,
        "object_state": {**context, "changed_object_count": len(touched_ids), "changed_fields": changed_fields, "requested_object_count": len(requested_ids)},
        "problem_metrics": problem_metrics,
        "match_threshold": DEFAULT_MATCH_THRESHOLD,
    }

    raw = f"{source_case}|{round_index}|{context}|{selected_action}|{before}|{after}"
    measured_gain = round(after - before, 4)
    # What the library keeps is a rule, so the row's own sentence is rendered from the rule's
    # three parts instead of borrowed from the reflection. ``why_changed`` explains whether this
    # round improved; it says nothing about how such a figure is drawn, and it used to be the
    # sentence a later paper was handed. The model's words live in ``guidance``.
    verdict = experience_verdict(reflection, accepted=accepted)
    guidance = _transferable_guidance(selected_action, reflection)
    # Where the sentence came from. A patch is rendered from the model's own words when it wrote
    # one and left the patch blank, and the mechanical fallback below is neither the model's nor a
    # requirement, so the reader has to be able to tell them apart.
    guidance_source = "model" if guidance else ""
    if verdict == "worse" and not guidance:
        # The round did show this move made the figure worse, and the reflection wrote no sentence
        # a later paper could use. The mechanical reading is then the honest fallback -- but it is
        # only ever a fallback, and it never replaces something the model did write.
        detail = f"pairwise_verdict={pairwise_verdict or 'before'}; " if comparison == "pairwise" else ""
        guidance = f"Do not reuse this move under this condition: {detail}repair_gain={measured_gain:+}."
        guidance_source = "measured"
    hypothesis = render_rule_statement(condition=context, action=selected_action, guidance=guidance)
    graph_scope = {
        "graph_ids": list(selected_action.get("graph_ids") or []),
        "subgraph_ids": list(selected_action.get("subgraph_ids") or []),
        "scientific_node_ids": list(selected_action.get("semantic_node_ids") or []),
        "edge_ids": list(selected_action.get("edge_ids") or []),
        "slot_ids": list(selected_action.get("slot_ids") or []),
        "object_ids": touched_ids or requested_ids,
    }
    if not any(graph_scope.values()):
        graph_scope["object_ids"] = [str(item.get("id")) for item in layout.get("nodes", []) if isinstance(item, dict) and item.get("id")]

    scope, scope_evidence = _verified_scope(normalize_experience_scope(selected_action.get("scope_type")), reflection, touched_ids, layout)
    applicability_record = resolved_applicability(selected_action, apply_at=apply_at, domain_profile=domain_profile)
    if verdict != "worse":
        # Nothing here has shown the move is wrong, so the row states no result of its own and
        # forbids nothing. What it does keep is the model's own words and the measurement it came
        # from, so a later reader can tell "untried" apart from "measured and rejected".
        excluded_conditions = []
    elif comparison == "pairwise":
        excluded_conditions = [{**context, "reason": "pairwise_not_preferred", "pairwise_verdict": pairwise_verdict or "before", "repair_gain": measured_gain}]
    else:
        excluded_conditions = [{**context, "reason": "repair_gain_not_positive", "repair_gain": measured_gain}]
    if bool((reflection or {}).get("domain_mismatch")) and domain and domain != "unknown":
        # The reflection claimed the repair does not travel to this domain; that is the one
        # claim a run can act on immediately, so it becomes an excluded condition.
        excluded_conditions.append({"dimension": "domain", "value": domain, "reason": "reflection_domain_mismatch"})
    if new_blockers:
        # The defect the round introduced is filed as a condition this row must not be reused
        # under, so a later figure that carries it is not handed this repair again. The
        # underlying measurement is the same one the round decision wrote, never a second read.
        excluded_conditions.append({**context, "reason": "carried_new_defect", "defects": list(introduced), "count": int(new_blockers)})
    observed_effect = {
        "overall_score_delta": measured_gain,
        "before_score": round(before, 4),
        "after_score": round(after, 4),
        "dimension_deltas": _dimension_deltas(previous, current),
        "changed_object_ids": touched_ids,
        "requested_object_ids": requested_ids,
        "changed_object_count": len(touched_ids),
        "changed_fields": changed_fields,
        "action_changed": bool((bounded_change or {}).get("applied")),
        "figure_changed": (figure_diff or {}).get("changed") if isinstance(figure_diff, dict) else None,
        "figure_changed_object_ids": [str(item) for item in ((figure_diff or {}).get("changed_object_ids") or []) if str(item)],
        "compiled_changed_fields": compiled_changed_fields,
        # The breaks this round measured, kept so a refusal can be enforced against the same
        # metrics the round was judged on instead of against a sentence nobody can check.
        "decomposability_violations": [
            {
                "metric": str(item.get("metric") or ""),
                "kind": str(item.get("kind") or ""),
                "value": item.get("value"),
                "threshold": item.get("threshold"),
            }
            for item in (decomposability_violations or [])
            if isinstance(item, dict) and str(item.get("metric") or "")
        ],
        # The colour metrics the round measured, as evidence only: they are never a gain and
        # never a gate. A round that changed no colour simply records the same pair twice.
        "color_metrics": {
            "before": dict((color_metrics or {}).get("before") or {}),
            "after": dict((color_metrics or {}).get("after") or {}),
        },
    }
    condition_evidence = {
        "changed_object_ids": touched_ids,
        "requested_object_ids": requested_ids,
        "motifs_from": "touched_subgraphs" if touched_ids else "whole_graph" if required_motifs else "none",
        "problem_metrics_from": "judge_dimension_scores" if problem_metrics else "none",
        "measured_by": str((bounded_change or {}).get("contract") or "judge_score_delta"),
        "scope_evidence": scope_evidence,
    }
    experience = {
        "schema_version": EXPERIENCE_SCHEMA_VERSION,
        "domain": "direct_editable",
        "scope": scope,
        "applicability": applicability_record["applicability"],
        "applicability_domain": applicability_record["applicability_domain"],
        "applicability_reason": applicability_record["applicability_reason"],
        "memory_type": memory_type,
        "apply_at": apply_at,
        "problem_type": str(selected_action.get("problem") or "").strip(),
        "trigger": trigger,
        "required_motifs": required_motifs,
        "preserve": [str(item) for item in (current.get("preserve") or []) if str(item)],
        "forbidden_changes": [str(item) for item in (selected_action.get("forbidden_changes") or []) if str(item)],
        "judge_fingerprint": dict(judge_fingerprint or {}),
        "reported_reflection": _reported_reflection(reflection),
        "condition_evidence": condition_evidence,
        "experience_id": "direct-exp-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16],
        "condition": context,
        "context": context,
        "graph_scope": graph_scope,
        "hypothesis": hypothesis,
        "guidance": guidance,
        "guidance_source": guidance_source,
        "action": selected_action,
        "expected_effect": {"overall_score": "increase", "new_blockers": 0},
        "observed_effect": observed_effect,
        "source_case": source_case,
        "source_round": round_index,
        "accepted": accepted,
        "carried_new_defect": bool(new_blockers),
        "new_blocking_defects": list(introduced),
        "new_blocking_defect_count": int(new_blockers),
        "repair_gain": measured_gain,
        "comparison": comparison,
        "pairwise_verdict": pairwise_verdict,
        "measurement_instrument": measurement_instrument,
        "object_diff": action_diff,
        "compiled_object_diff": dict(compiled_state_diff or {}),
        "excluded_conditions": excluded_conditions,
        "provenance": {
            "source": str(provenance_source or DEFAULT_PROVENANCE_SOURCE),
            "run": source_case,
            "round": round_index,
            "measured": True,
            "instrument": measurement_instrument,
            "before_score": round(before, 4),
            "after_score": round(after, 4),
        },
        "outcome": {"accepted": accepted, "before_score": before, "after_score": after, "score_delta": measured_gain, "new_blockers": list(introduced), "comparison": comparison, "pairwise_verdict": pairwise_verdict},
        "validated_cases": [source_case] if accepted and source_case else [],
        # A row may only name a case as failed when the round showed the move failed on it. A
        # withdrawn reading and a rule the compile never reached both leave the case unjudged.
        "failed_cases": [source_case] if verdict == "worse" and source_case else [],
        "side_effects": [],
        # The rung follows the round's own verdict rather than the Gate's: a row the Gate rolled
        # back on a reading that was withdrawn, or on a rule that never reached the figure, is
        # recorded as unmeasured. That rung is kept out of retrieval and out of the rule
        # aggregation, so it neither guides nor misleads.
        "status": "current_case_validated" if accepted else ("refuted" if verdict == "worse" else "measurement_noise"),
        "verdict": verdict,
        "strategy_id": "direct-strategy-" + hashlib.sha1(str(context).encode("utf-8")).hexdigest()[:12],
        "strategy": {"instruction": hypothesis, "action": selected_action},
    }
    return {"ok": accepted, "reason": "built" if accepted else ("refuted" if verdict == "worse" else "unmeasured"), "verdict": verdict, "experience": experience, "before_score": before, "after_score": after}


def _dimension_deltas(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    """Record per-dimension movement so a repair's effect stays visible below the score quantum."""
    before = previous.get("dimension_scores") if isinstance(previous.get("dimension_scores"), dict) else {}
    after = current.get("dimension_scores") if isinstance(current.get("dimension_scores"), dict) else {}
    deltas: dict[str, float] = {}
    for key in sorted(set(before) | set(after)):
        start, end = before.get(key), after.get(key)
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and not isinstance(start, bool) and not isinstance(end, bool):
            deltas[str(key)] = round(float(end) - float(start), 4)
    return deltas
