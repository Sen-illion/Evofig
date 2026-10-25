"""Direct paper-to-PPTX generation and self-evolution workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import shutil
import os
import json
import time

from ..analysis.paper.method_extractor import extract_method_figure_content
from ..analysis.graph import build_scientific_graph, extract_subgraphs
from ..composition.pptx import compile_ppt
from ..composition.pptx_render import render_deck_preview
from ..composition import CURATED_LAYOUT_RULES, apply_color_theme_to_program, arrange_node_contents, guard_overlay_geometry, keep_panels_below_title_band, apply_connector_route_selections, apply_memory_routing_constraints, apply_direct_layout_repairs, apply_layout_constraints, build_editable_graph, fit_text_program_globally, optimize_direct_typography, resolve_direct_connectors, resolve_joint_geometry, style_and_route_arrows
from ..evaluation.geometry_quality import evaluate_density_quality, evaluate_geometry_quality
from ..evaluation.text_readability import evaluate_text_readability
from ..evaluation.reference_adoption import evaluate_reference_adoption
from ..evaluation.delivery_status import classify_delivery_status
from ..evaluation.color_quality import evaluate_color_quality
from ..evaluation.color_experience import evaluate_color_experience_usage
from ..evaluation.connector_visibility import evaluate_connector_visibility
from ..evaluation.rendered_density import measure_rendered_density
from ..evaluation.direct_connector_review import review_connector_routes
from ..evaluation.direct_color_review import disabled_color_repair_status, review_color_draft
from ..planning.panel_style import build_panel_style_plan
from ..planning.panel_target import check_design_against_panel_target, panel_target_from_specification
from ..evaluation.direct_visual_review import review_direct_draft
from ..evaluation.direct_judges import DirectJudgeInput, DirectPairJudgeInput, ReferenceGapInput, build_direct_judges
from ..evaluation.graph_alignment import align_graph_layers
from ..evaluation.grounding_gate import evaluate_grounding_gate
from ..evaluation.decomposability import build_decomposability_report
from ..evaluation.content_coverage import REPORT_FILE as CONTENT_COVERAGE_FILE, measure_content_coverage
from ..evaluation.prompt_patch_check import check_prohibited_patches
from ..contracts.structure import (
    DECOMPOSABILITY_MAX_CROSS_PANEL_CONNECTORS,
    DECOMPOSABILITY_MAX_NODE_SLOT_FANOUT,
    DECOMPOSABILITY_MIN_PANEL_AREA,
    DECOMPOSABILITY_MIN_PANEL_OCCUPANCY,
)
from ..contracts.prompt_patch import PROMPT_ASSEMBLY_ORDER, PROMPT_PATCH_CONTRACT, PROMPT_PATCH_VERSION, normalize_prompt_patch
from ..contracts.suggestion_tiers import suggestion_tiers_record
from ..contracts.repair_actions import attach_heuristic_budget
from ..analysis.graph.defect_patterns import defect_patterns
from ..analysis.reference import (
    analyze_reference_image,
    reference_image_path,
    select_gap_reference,
    select_positive_framework_reference,
)
from ..evolution.direct import ExperienceStore, HumanReviewStore, build_direct_experience_result, build_patches_for_experience
from ..evolution.direct import resolve_domain_profile
from ..evolution.direct import resolve_retrieval_statuses
from ..evolution.direct import FOLLOW_UP_ATTEMPTS, FIGURE_PROGRAM_CLASS, action_execution_fingerprint, action_is_executable, append_event, apply_reflection_policy, compare_candidate, content_gap_signal, create_state_snapshot, default_reflection, derived_score, diff_figure_programs, diff_state_snapshots, exclusion_context, measured_score, mechanism_rule_fingerprint, next_follow_up_turn, next_rule_turn, refuting_rule_ids, restore_state_snapshot, select_next_round_proposal, validate_reflection_rule, stage_for_action, action_class, action_class_policy, measurement_request, plan_partial_rollback
from ..evolution.direct import round_proposal_order
from ..evolution.direct.experience import MECHANISM_POLICY_SOURCES, experience_binding_record, experience_library_record
from ..evolution.direct.creator_rules import rules_for_paper
from ..evolution.direct.snapshots import graph_layer_digest, layer_change
from ..evolution.direct import inject_experience, seed_injection
from ..evolution.direct import CURRENT_CASE_STATUS, OWN_CASE_ASSEMBLY_STATUSES, ClaimLedger, PromptPatchStore, SuggestionRuleStore, action_reuses_assets, assemble_prompt_patches, evolution_metrics, measured_metrics, prompt_revision_record, select_own_case_candidates
from ..evolution.direct.file_lock import append_jsonl
from ..evolution.direct import CreatorRuleLedger, build_memory_application_trace, read_creator_rule_manifest
from ..contracts import CREATOR_RULE_CURATED_SOURCE, CREATOR_RULE_MANIFEST_CONTRACT, CREATOR_RULE_MANIFEST_VERSION, CREATOR_RULE_REFUTATION_LIMIT, PROMPT_PATCH_REFUTATION_LIMIT, load_curated_standard, merge_creator_rule_manifests, normalize_creator_rule_manifest
from ..evolution.direct.creator_rules import CREATOR_RULE_REFUTING_REASONS
from ..contracts.fault_attribution import stage_requirements
from ..contracts.evolution import DERIVED_SCORE_SOURCE, EXPERIENCE_ACTIONS, MECHANISM_ACTIONS, normalize_pairwise_result, pairwise_verdict
from ..contracts.memory_parameters import density_targets_from, layout_rules_from, replayable_theme, router_rules_from
from ..planning.prompt_assembly import effective_thresholds, patch_parameters, render_patch_block
from ..generation.asset_metrics import trim_program_icon_assets
from ..generation.icon_assets import generate_icon_assets
from ..composition.semantic_figure import build_semantic_figure_program
from ..analysis.paper.preparation import prepare_paper_figure_contract
from ..contracts.semantic import FrozenContractUnusable, adopt_upstream_ground_truth, load_upstream_contract, merge_upstream_content_plan
from ..contracts.design_plan import DESIGN_PLAN_SEED_FILE
from ..contracts.method_figure import FIGURE_CONTENT_PLAN_FILE
from ..analysis.reference.exemplars import resolve_project_exemplars
from ..planning.paper_figure_structure import (
    create_layout_candidate,
    plan_paper_figure_structure,
    structure_from_design_plan,
)
from ..planning.vlm_structure_planner import plan_structure_vlm
from ..planning.scientific_hierarchy import plan_scientific_hierarchy
from ..planning.density_advisory import build_density_advisory
from ..planning.decomposition_policy import plan_decomposition_action
from ..planning.figure_text_planner import plan_figure_text_vlm
from ..planning.relation_explanations import plan_relation_explanations_vlm
from ..planning.ppt_composition_planner import plan_ppt_composition_vlm
from ..planning.feedback_repair_planner import plan_feedback_repairs_vlm
from ..planning.color_theme_planner import plan_color_theme_vlm
from ..planning.visual_graph import build_visual_graph, resolve_design_required_edge_ids
from ..utils import ensure_dir, read_json, write_json
from ..contracts.text_ownership import canonicalize_node_text_program
from .progress import build_progress_reporter
from .direct_repair_execution import execute_direct_repair, figure_move_evidence, mechanism_rule_evidence, record_unexecuted_repair, write_mechanism_rule_report
from .judge_binding import resolve_judge_binding
from .stage_replay import load_stage_replay
from .round_moves import (
    MECHANISM_DEFAULT_SOURCE,
    guard_reason,
    is_mechanism_move,
    move_conflict_reason,
    move_source,
    move_unexecutable_reason,
    plan_round_moves,
    round_moves_record,
)
from .run_cost import model_calls_from_progress, move_cost_record, round_cost_record, stage_seconds_between, stage_seconds_from_progress, summarize_round_costs, write_round_cost, write_run_cost_report
from .stage_trace import ACTION_TRACE_STAGE, STAGE_GATE_FILE, attribution_verdict, build_stage_trace, compact_stage_trace, ledger_attribution, stage_gate_report


def _prestaged_icon_root(source: str | Path | None) -> Path | None:
    """The icons a wave pre-staged into the directory the first attempt writes.

    A reuse wave hands a paper the PNGs a mapping table already certified, placing them
    where that paper's first attempt would have put them, so no slot is drawn twice.  That
    directory is not a replayable attempt: it carries no reading and no contract, so the
    replay declines it and the icons have to be found on their own.  Only a directory that
    never got past paper preparation is read this way - an attempt that did real work is
    replayed, not raided, because its own reports say what was paid for.
    """

    if not source:
        return None
    root = Path(source)
    if not root.is_dir():
        return None
    if (root / "input_manifest.json").is_file() or (root / "document_model.json").is_file():
        return None
    for candidate in sorted(root.glob("round_*/candidate_*")):
        assets = candidate / "assets"
        if assets.is_dir():
            return assets
    return None


def _inherited_icon_root(
    stage_replay: dict[str, Any] | None,
    prestaged_icons: Path | None,
    reuse_replayed_icons: str = "on",
):
    """The Icons an attempt may keep from outside itself, subject to ``reuse_replayed_icons``.

    Two mechanisms drop PNGs where a fresh attempt would have drawn them: a replay of an
    earlier attempt (``stage_replay["icon_root"]``) and a reuse wave that pre-staged certified
    assets. Only the replay is governed here. An experiment about inference spend has to draw
    its Icons again -- the Icons are most of what a paper costs, so a rung that quietly
    inherited them would report the baseline's bill under its own name. The replay's own report
    already says the Icons were re-planned; this is what makes the run agree with it.
    """

    replayed = (stage_replay or {}).get("icon_root")
    if str(reuse_replayed_icons or "on").strip().casefold() in ("off", "false", "0", "no"):
        replayed = None
    return replayed or prestaged_icons
from .upstream_retry import describe_retry_feedback, run_upstream_retry, upstream_retry_kind


def _prioritize_repair_intents(review: dict[str, Any]) -> dict[str, Any]:
    """Make mixed visual repair intent precedence explicit and reproducible."""
    priority = {"replan_structure": 0, "resize_node": 1, "reserve_connector_lane": 1, "move_connector_label": 1, "rebalance_panel": 2, "translate_group": 2, "distribute_horizontal": 2, "distribute_vertical": 2, "align_objects": 2}
    review["repair_intents"] = sorted(list(review.get("repair_intents") or []), key=lambda item: priority.get(str(item.get("action")), 99))
    review["repair_policy"] = "replan_structure takes precedence; otherwise execute bounded repairs in contract order"
    return review


def _memory_context(layout: dict[str, Any], previous_candidate: Path | None, domain_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Describe the state Memory ranks experiences against.

    The motifs come from the last accepted candidate's Visual Graph, because they are what
    decides whether a repair learned on a comparable figure is relevant to this one.
    The domain is named too, because a condition appended after a failed round is often
    domain-specific and could never be matched if the query stayed silent about it.
    """

    motifs: list[str] = []
    if previous_candidate:
        visual_graph = previous_candidate / "visual_graph.json"
        if visual_graph.is_file():
            try:
                subgraphs, _ = extract_subgraphs(read_json(visual_graph), include_unknown=True)
                motifs = sorted({str(item.get("motif") or "unknown") for item in subgraphs})
            except Exception:
                motifs = []
    return {
        "domain": str((domain_profile or {}).get("domain") or ""),
        "domain_missing": bool((domain_profile or {}).get("domain_missing")),
        "topology": layout.get("topology"),
        "nodes": layout.get("nodes") or [],
        "connectors": layout.get("connectors") or [],
        "motifs": motifs,
    }


def _trained_judge_flag(advisory: dict[str, Any] | None, acceptance: dict[str, Any] | None) -> bool:
    """Say whether a trained model took part in judging this run.

    The two seats answer separately and either of them may be the heuristic gate, so the flag
    is true as soon as one recorded answer came from a trained model. A report that always
    answered false let a run judged by the trained checkpoint describe itself as judged by
    rules, which is the one thing this field exists to tell apart.
    """

    return any(bool((record or {}).get("trained")) for record in (advisory, acceptance))


def _next_action_with_memory(retrieved: list[dict[str, Any]], proposal: dict[str, Any], layout: dict[str, Any], icons: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Decide the next round's action, preferring an experience that fits its stage.

    Returns the action to run, the injection that chose it and the routing report. Nothing
    routed means the proposal stands, so a library without a fitting experience changes
    nothing about how the loop repairs the figure.
    """

    injection = inject_experience(retrieved, proposal, known_ids=_known_object_ids(layout, icons))
    return injection["action"] or proposal, dict(injection["injection"]), injection["report"]


def _known_object_ids(layout: dict[str, Any], icons: dict[str, Any]) -> set[str]:
    """Collect the object ids the current state still holds, so an injection cannot invent one."""

    known = {str(item.get("id")) for item in layout.get("nodes", []) if isinstance(item, dict) and item.get("id")}
    known |= {str(item.get("asset_id")) for item in (icons or {}).get("icons", []) if isinstance(item, dict) and item.get("asset_id")}
    return {item for item in known if item}


def _round_human_preferences(directory: str | Path | None, given: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """The person's own instructions as they stand at the top of one round.

    Read again for every round rather than once for the iteration. A person answers a card about the
    figure in front of them, and the sentence they write belongs to the round that follows it; read
    once per iteration, that sentence sits out the rest of the iteration and three more rounds are
    spent under an instruction the person has already replaced. A caller that named no folder keeps
    the list it handed in, so a caller that predates this channel behaves exactly as it did.
    """

    if not str(directory or "").strip():
        return [dict(item) for item in (given or [])]
    try:
        store = HumanReviewStore(Path(str(directory)))
        rows = store.entries() if store.entries_path.exists() else []
    except Exception:  # noqa: BLE001 - a channel nobody can read must not stop the round
        return [dict(item) for item in (given or [])]
    if not rows:
        return [dict(item) for item in (given or [])]
    return [dict(row) for row in rows if isinstance(row, dict)]


def person_router_rules(preferences: list[dict[str, Any]] | None, stage: str) -> dict[str, Any]:
    """The drawing rules a person's own sentences state, for the stage they apply at.

    A sentence a person writes is prose, and prose cannot be handed to the router that actually
    draws the connectors. A preference may therefore carry the machine form of its own sentence
    under ``condition.router`` -- the same keys a measured rule states -- and it is read here so a
    person's rule binds the geometry rather than only being visible to the stage that plans it.
    """

    rules: dict[str, Any] = {}
    for item in preferences or []:
        row = item if isinstance(item, dict) else {}
        applies_at = str(row.get("apply_at") or "").strip()
        if applies_at and applies_at != str(stage):
            continue
        condition = row.get("condition") if isinstance(row.get("condition"), dict) else {}
        rules.update(router_rules_from(condition.get("router")))
        rules.update(router_rules_from(row.get("router")))
        rules.update(router_rules_from(condition.get("constraints")))
    return rules


def person_rule_patches(preferences: list[dict[str, Any]] | None, stage: str) -> list[dict[str, Any]]:
    """The person's own sentences, in the shape a stage prompt already renders.

    A preference is filed with the stage it applies at, exactly like a prompt patch, so it can ride
    the block a stage already reads: a stage that draws connectors is then told, in the person's own
    words, how they want connectors drawn. Before this the sentence reached the advisory seat and
    the reflection and nothing that decides geometry -- which is how a figure came back carrying the
    detour the person had just objected to, with the round's own account of itself saying the rule
    had been handed to it.

    The entry keeps ``source: person`` and the stage it applies at, so an injection reads as a
    person's instruction rather than as a patch that earned a rung.
    """

    # One sentence, one entry: a person who files the same instruction twice -- the second time with
    # the machine form of it attached -- has said one thing, and a prompt that carries it twice
    # reads as two rules. The later row wins and the earlier one's conditions merge into it, so the
    # machine form still travels with the sentence it belongs to.
    kept: dict[str, dict[str, Any]] = {}
    for item in preferences or []:
        row = item if isinstance(item, dict) else {}
        directive = str(row.get("directive") or "").strip()
        if not directive:
            continue
        applies_at = str(row.get("apply_at") or "").strip()
        if applies_at and applies_at != str(stage):
            continue
        key = " ".join(directive.casefold().split())
        previous = kept.get(key)
        condition = dict(row.get("condition") or {}) if isinstance(row.get("condition"), dict) else {}
        if previous is not None:
            condition = {**dict(previous.get("condition") or {}), **condition}
        kept[key] = {**row, "condition": condition}
    rows: list[dict[str, Any]] = []
    for row in kept.values():
        directive = str(row.get("directive") or "").strip()
        patch = normalize_prompt_patch(
            {
                "patch_id": "person-rule-" + str(row.get("review_id") or row.get("entry_id") or "")[:20],
                "apply_at": str(stage),
                "class": "person_rule",
                "action_type": str(row.get("action_type") or ""),
                "condition": dict(row.get("condition") or {}) if isinstance(row.get("condition"), dict) else {},
                "directive": directive,
                "status": str(row.get("status") or "candidate"),
                "written_by": str(row.get("reviewer") or "human"),
            }
        )
        patch["source"] = "person"
        rows.append(patch)
    return rows


def _decomposition_proposal(policy: dict[str, Any] | None) -> dict[str, Any]:
    """Turn a measured decomposability break into a bounded action proposal.

    The policy decides *whether* to act; the proposal is the same shape an Online Judge
    feedback produces, so it is checked against the actions the loop may run, executed by
    ``apply_bounded_action`` and accepted or rolled back by the ordinary comparison.
    """

    source = policy if isinstance(policy, dict) else {}
    if not source.get("required") or not source.get("action"):
        return {}
    # The policy is code, so it may name the catalog nudge the executor requires. A VLM
    # answer that omitted factor/dx/dy stays incomplete; this path does not guess for it.
    return attach_heuristic_budget({
        "type": str(source["action"]),
        "scope_type": str(source.get("scope_type") or "global"),
        # The targets are typed because the action decides what can be targeted: a
        # node-level reflow is handed nodes, and a routing repair is handed connectors.
        # The measured objects stay in the policy's own record.
        "semantic_node_ids": [str(item) for item in source.get("target_node_ids") or []],
        "edge_ids": [str(item) for item in source.get("target_connector_ids") or []],
        "problem": str(source.get("reason") or ""),
        "expected_effect": "clear the measured decomposability break without changing scientific content",
        "source": "decomposition_policy",
    })


def _changed_object_ids(before: Any, after: Any, *, collection: str, id_key: str) -> list[str]:
    """Name the objects a re-run added, removed or rewrote, new and missing ones included."""

    def index(value: Any) -> dict[str, str]:
        items = (value or {}).get(collection) if isinstance(value, dict) else None
        return {
            str(item.get(id_key) or ""): json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
            for item in (items or [])
            if isinstance(item, dict) and str(item.get(id_key) or "")
        }

    left, right = index(before), index(after)
    return [key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]


def _opening_repair(
    action: dict[str, Any] | None,
    stage_replay: dict[str, Any] | None,
    stage_replay_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """The change an inherited iteration opens with, or nothing when it has to plan again.

    The review of the figure this run inherits named what is wrong with it, so the run may open
    by executing that change rather than by planning the same stages a second time. Three things
    make that the wrong move, and each is checked here rather than assumed:

    * the state the change acts on has to be the inherited one -- a design or an icon set this run
      planned itself is a different figure, and a change asked of the old one is not a change to
      this one;
    * the action has to be a bounded repair: a stage re-run plans the same stages anyway, which is
      exactly what this opening exists to avoid;
    * the action has to carry what its executor reads, which is the check every later round runs
      its own moves through.
    """

    source = action if isinstance(action, dict) else {}
    if not source or not str(source.get("type") or ""):
        return {}
    if str(source.get("type")) == "initial" or upstream_retry_kind(source):
        return {}
    if not isinstance(stage_replay, dict) or not stage_replay:
        return {}
    replanned = {str(item) for item in (stage_replay_report or {}).get("replanned_stages") or []}
    if {"design_seed", "icons"} & replanned:
        return {}
    if not action_is_executable(source):
        return {}
    opening = dict(source)
    opening.setdefault("scope_type", "global")
    # The tag keeps the record readable: this move came from the review of the figure the
    # iteration inherited, not from the round's own advisory answer, and the library is told
    # the difference when it looks at who proposed a move.
    opening["source"] = "iteration_review"
    return opening


def _round_runs_a_move(round_index: int, action: dict[str, Any] | None) -> bool:
    """Whether a round executes ``action`` rather than compiling the figure this run planned.

    Round 0 is the run's own drawing, so its ``initial`` marker is not a repair: it compiles the
    stages this run planned. An iteration that inherited a finished figure opens on the change
    the review of that figure asked for instead, and that change is a real move, executed and
    measured like any later round's.
    """

    if round_index > 0:
        return True
    return str((action or {}).get("type") or "") not in {"", "initial"}


def _merge_move_changes(move_records: list[dict[str, Any]]) -> dict[str, Any]:
    """The round's execution record when it ran more than one move.

    The compiled figure carries every move's edit, so the record the candidate loop reads has
    to name every changed asset -- otherwise only the last move's icons are redrawn and the
    first move's edit is compiled against stale art. Each move keeps its own execution record
    beside this one, so "which move changed what" survives the merge.
    """

    records = [item for item in move_records if isinstance(item, dict)]
    if not records:
        return {}
    first = dict(records[0].get("execution") or {})
    if len(records) == 1:
        return first
    asset_ids: list[str] = []
    for record in records:
        for asset_id in (record.get("execution") or {}).get("changed_asset_ids") or []:
            if str(asset_id) not in asset_ids:
                asset_ids.append(str(asset_id))
    return {
        **first,
        "changed_asset_ids": asset_ids,
        "moves": [str(item.get("type") or "") for item in records],
        "move_count": len(records),
    }


def _upstream_retry_execution(
    initial_state: dict[str, Any],
    proposal_state: dict[str, Any],
    action: dict[str, Any],
    *,
    round_index: int,
    retry_index: int,
    kind: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Measure what the escalation changed, the same way a bounded action is measured.

    The re-run is a real action on the state, so its report carries the same evidence a
    bounded action carries: the diff that says whether anything moved at all.
    """

    before = create_state_snapshot(initial_state, label="before_upstream_retry", round_index=round_index)
    after = create_state_snapshot(proposal_state, label="after_upstream_retry", round_index=round_index)
    diff = diff_state_snapshots(before, after)
    changed_nodes = _changed_object_ids(initial_state.get("layout"), proposal_state.get("layout"), collection="nodes", id_key="id")
    changed_assets = _changed_object_ids(initial_state.get("icons"), proposal_state.get("icons"), collection="icons", id_key="asset_id")
    report = {
        "contract": "rfs.direct_action_execution",
        "version": 1,
        "action": action,
        "action_type": str(action.get("type") or ""),
        "applied": bool(diff.get("changed")),
        "geometric_applied": bool(changed_nodes or changed_assets),
        "program_applied": True,
        "changed_node_ids": changed_nodes,
        "changed_asset_ids": changed_assets,
        "program_changes": [{"object_type": "upstream_retry", "after": str(kind)}],
        "object_diff": diff,
        "preserve_checks": {
            "scientific_contract_unchanged": before["state"].get("method") == after["state"].get("method"),
            "input_state_unmodified": True,
        },
        "retry_index": retry_index,
        "upstream_retry": str(kind),
    }
    return report, before, after


def _report_upstream_retry(event_path: Path, round_index: int, report: dict[str, Any]) -> None:
    """Write the escalation's three events, so a refusal and a failure read apart."""

    kind = str(report.get("kind") or "")
    if not kind:
        return
    if not report.get("attempted"):
        append_event(event_path, "upstream_retry_declined", round=round_index, kind=kind, reason=str(report.get("declined_reason") or ""), provider_error="")
        return
    append_event(event_path, "upstream_retry_started", round=round_index, kind=kind, retry_index=int(report.get("retry_index") or 0))
    if report.get("applied"):
        append_event(event_path, "upstream_retry_completed", round=round_index, kind=kind, changed=bool(report.get("changed")))
    else:
        append_event(event_path, "upstream_retry_declined", round=round_index, kind=kind, reason=str(report.get("declined_reason") or ""), provider_error=str(report.get("provider_error") or ""))


def _retry_summary(report: dict[str, Any]) -> dict[str, Any]:
    """The escalation's own record, without the contracts it produced (those are on disk)."""

    skip = {"method", "design", "structure", "method_report", "structure_report"}
    return {key: value for key, value in (report or {}).items() if key not in skip}


#: The deterministic colour metrics the colour review already measures. They are recorded as
#: evidence beside a theme round; they never decide whether a round is accepted.
COLOR_METRIC_KEYS = (
    "minimum_text_contrast",
    "distinct_structural_hues",
    "maximum_distinct_hues",
    "connector_mismatch_count",
)


def _color_metrics(color_quality: dict[str, Any] | None) -> dict[str, Any]:
    """Read the four measured colour metrics out of a colour quality report."""

    metrics = (color_quality or {}).get("metrics") or {}
    return {key: metrics.get(key) for key in COLOR_METRIC_KEYS}


def _evolution_should_stop(
    *,
    evolution_enabled: bool,
    fixed_rounds: bool,
    accepted: bool,
    judgement: dict[str, Any] | None,
    pending_seed_action: dict[str, Any] | None,
) -> bool:
    """Whether the round loop is finished once this round is written.

    A round that reached a passing figure ends the loop. A rolled-back round cannot, even when
    the candidate it discarded passed: the loop kept the earlier state, so that answer belongs
    to a figure nobody serves. Reading it anyway stopped a run on a figure the acceptance role
    had scored 0.0 with two blocking issues, and it swallowed the escalation the round had
    already queued for the next one.
    """

    if not evolution_enabled:
        return True
    if fixed_rounds or pending_seed_action:
        return False
    return bool(accepted) and str((judgement or {}).get("status") or "") == "pass"


def _action_with_applicability(action: dict[str, Any], rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Copy one agreed applicability from the rules that guided this round.

    Several rules that disagree leave the action unlabeled. The paper domain
    is filled later, when the experience is written.
    """

    recorded = dict(action or {})
    if str(recorded.get("applicability") or "").strip():
        return recorded
    values = sorted({
        str(rule.get("applicability") or "").strip().casefold()
        for rule in rules
        if str(rule.get("applicability") or "").strip().casefold() in {"common", "domain"}
    })
    if len(values) == 1:
        recorded["applicability"] = values[0]
    return recorded


def _apply_choice_scores(before: dict[str, Any] | None, after: dict[str, Any], pairwise: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Write the defect-table scores from a choice comparison onto both states.

    The numbers are derived from the defect the comparison named, so they are labelled
    ``derived_from_defect`` and marked unmeasured: they are kept so the record shows
    which defect was named, and every decision reads the deterministic gate's own
    reading instead. A state that carries a measurement of its own is left alone.
    """

    scores = pairwise.get("choice_scores") if isinstance(pairwise, dict) else None
    if not isinstance(scores, dict):
        return before, after

    def merge(state: dict[str, Any] | None, side: str) -> dict[str, Any] | None:
        payload = scores.get(side)
        if not isinstance(payload, dict) or payload.get("overall_score") is None:
            return state
        base = dict(state or {})
        if base.get("score_measured") is True:
            return base
        base["overall_score"] = payload["overall_score"]
        base["dimension_scores"] = payload.get("dimension_scores") or base.get("dimension_scores") or {}
        base["score_source"] = DERIVED_SCORE_SOURCE
        base["score_measured"] = False
        return base

    merged_before = merge(before, "before")
    merged_after = merge(after, "after")
    return merged_before, merged_after if isinstance(merged_after, dict) else after


def _record_experience_build_refusal(
    *, event_path: Path, round_dir: Path, round_index: int, build: dict[str, Any], action: dict[str, Any]
) -> None:
    """Write down why a round did not become an experience, naming a self-repair as one.

    A round the mechanism repaired on its own policy is a real change and stays in the run,
    but it is not a lesson: the policy proposes the same move with or without Memory, so
    filing it would put a shipped default into the library as if it were a discovery.

    A round that won by drawing the figure again is the same kind of thing from the other side:
    its move is not one the catalogue can hand another round, so there is nothing to file -- and
    there is nothing to refuse either.
    """

    reason = str(build.get("reason") or "")
    payload = {key: value for key, value in build.items() if key != "experience"}
    if reason == "not_a_catalogue_repair":
        # The round moved the figure forward and there is simply nothing to file: the move it won
        # with is not one the catalogue can hand to another round. That is not a refusal -- writing
        # it as one put a "rejected" notice on every accepted redraw and buried the refusals that
        # do mean something.
        write_json(round_dir / "experience_not_a_repair.json", {"stage": "build", "round": round_index, **payload})
        append_event(event_path, "experience_not_a_repair", round=round_index, action=str(build.get("action") or ""))
        return
    if reason == "mechanism_self_repair":
        write_json(round_dir / "mechanism_self_repair.json", {"stage": "build", "round": round_index, "action": str(action.get("type") or ""), **payload})
        append_event(event_path, "mechanism_self_repair_not_memorized", round=round_index, action=str(action.get("type") or ""), mechanism_source=str(build.get("mechanism_source") or ""))
        return
    write_json(round_dir / "experience_rejected.json", {"stage": "build", "round": round_index, **payload})
    append_event(event_path, "experience_rejected", round=round_index, stage="build", reason=reason)


def _defer_rule_write_off(
    *,
    pending_repair_rows: dict[str, dict[str, Any]],
    round_dir: Path,
    round_index: int,
    rule_fingerprint: str,
    row: dict[str, Any],
    attempts_spent: int,
    decision_reason: str,
) -> dict[str, Any]:
    """Keep a losing rule's refutation row instead of storing it, while the rule can be retried.

    A rule that lost was measured once, and one reading is evidence about the attempt rather than
    about the rule: the round that produced it also said why it thinks the rule did not land, and
    a claim worth another attempt is worth testing before the library is told the rule is wrong. So
    the row is built from that round -- the round that holds the measurement -- and kept aside. It
    is stored when the rule runs out of attempts, and dropped if the iteration moves on first, which
    is neither a pass nor a refutation and is recorded as neither.
    """

    pending_repair_rows[str(rule_fingerprint)] = row
    write_json(round_dir / "experience_deferred_for_repair.json", {
        "contract": "rfs.experience_deferred_for_repair",
        "version": 1,
        "round": int(round_index),
        "reason": "the rule is not written off while it still has a follow-up to run",
        "rule_fingerprint": str(rule_fingerprint),
        "attempts_spent": int(attempts_spent),
        "attempts_allowed": int(FOLLOW_UP_ATTEMPTS),
        "decision_reason": str(decision_reason),
        "experience": row,
    })
    return {"ok": False, "deferred": True, "reason": "deferred_for_rule_repair"}


def _measured_experience(
    *,
    round_index: int,
    candidate: dict[str, Any],
    previous: dict[str, Any],
    current: dict[str, Any],
    layout: dict[str, Any],
    action: dict[str, Any],
    reflection: dict[str, Any],
    paper: str | Path,
    domain_profile: dict[str, Any],
    frozen_judge: Any,
    judge_mode: str,
    judge_model: str | None,
    frozen_judge_model: str | None,
    judge_checkpoint: str = "",
    color_metrics: dict[str, Any] | None = None,
    color_theme: dict[str, Any] | None = None,
    icon_mode: str = "",
    round_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Memory row one measured round produced, whether it passed or not.

    Both branches read the same measurement, so a repair that did not survive the Gate
    becomes evidence instead of being dropped: the builder turns a non-positive gain into a
    ``refuted`` row that names the condition it must not be reused under. When the round
    was decided pairwise, that verdict is the ``accepted`` bit; the score delta stays on
    the row as trend evidence.
    """

    candidate_dir = Path(candidate["dir"])
    bounded_change = dict(candidate.get("bounded_change") or {})
    review = candidate.get("connector_review") if isinstance(candidate.get("connector_review"), dict) else {}
    if review.get("selections") and "routing_constraints" not in bounded_change:
        bounded_change["routing_constraints"] = {"selections": list(review.get("selections") or [])}
    return build_direct_experience_result(
        round_index=round_index,
        previous=previous,
        current=current,
        layout=layout,
        action=action,
        reflection=reflection,
        source_case=str(Path(paper).resolve()),
        bounded_change=bounded_change,
        compiled_state_diff=read_json(candidate_dir / "state_diff.json") if (candidate_dir / "state_diff.json").exists() else {},
        figure_diff=candidate.get("figure_diff") or {},
        visual_graph=read_json(candidate_dir / "visual_graph.json") if (candidate_dir / "visual_graph.json").exists() else {},
        domain_profile=domain_profile,
        judge_fingerprint={
            "role": str(getattr(frozen_judge, "role", "unknown")),
            "mode": str(current.get("mode") or judge_mode),
            "trained": bool(current.get("trained")),
            "model": str(frozen_judge_model or judge_model or getattr(frozen_judge, "model", "") or ""),
            "checkpoint": judge_checkpoint,
        },
        decomposability_violations=list((candidate.get("decomposability_report") or {}).get("violations") or []),
        color_metrics=color_metrics,
        color_theme=color_theme,
        icon_mode=icon_mode,
        round_decision=round_decision,
    )


def _default_policy_fingerprint(decomposition_action: dict[str, Any] | None) -> str:
    """The move the mechanism's own policy would have made this round, as a fingerprint.

    This is what the library's "common knowledge" gate compares a new row against: a row whose
    move is the one the run would have made with no Memory at all is a fact about the shipped
    default, not a finding, and no ON arm can differ from an OFF arm by replaying it. The
    decomposability policy is the mechanism's always-on repair path, so it is the whole of the
    shipped default; a colour re-plan is only ever scheduled after a measured review asked for
    one, which is a demand the run would not have made on its own, so it is not a default.
    """

    return action_execution_fingerprint(_decomposition_proposal(decomposition_action))


def _library_row_with_binding(
    experience: dict[str, Any], round_dir: Path, *, default_action_fingerprint: str = ""
) -> dict[str, Any]:
    """Prepare the row the library keeps, and park its binding in the run directory.

    A row earns its name by travelling, so only the rule enters the library: the condition, the
    mechanism move and its parameters. The ids and the boxes it was measured on belong to this
    figure, and they stay here in the run as the evidence of what the measurement was about.
    """

    binding = experience_binding_record(experience)
    binding_path = round_dir / "experience_binding.json"
    write_json(binding_path, binding)
    row = experience_library_record(experience, binding_ref=str(binding_path))
    if default_action_fingerprint:
        row["default_action_fingerprint"] = str(default_action_fingerprint)
    return row


def _prompt_patch_context(patch_store: Any, *, stage: str) -> dict[str, Any]:
    """What that stage's prompt already asks for, so the seat writes the sentence it is missing.

    The seat was asked to say which requirement a stage failed to write, and it wrote the same
    sentence every time: it could not see what that stage already asked for, so the only requirement
    it had in view was the one in front of it. What it is shown here is that stage's own requirement
    list, the sentences already standing at that stage, and the sentences the library holds -- which
    turns "say what is missing" into a question it can answer rather than one it can only answer
    twice. A round that names no stage adds no key, so every other reflection prompt is unchanged.
    """

    name = str(stage or "").strip()
    if not name:
        return {}
    rows = [row for row in ((patch_store.records_with_status() if patch_store is not None else []) or []) if isinstance(row, dict)]
    here = [row for row in rows if str(row.get("apply_at") or "") == name]
    library = sorted({str(row.get("directive") or "") for row in rows if str(row.get("directive") or "")})
    return {
        "stage": name,
        "requirements_this_stage_owns": [str(item) for item in (stage_requirements().get(name) or ())],
        "already_asked_for_here": [str(row.get("directive") or "") for row in here if str(row.get("directive") or "")],
        "already_in_library": library[:12],
        "library_sentences": len(library),
    }


def _record_prompt_patch(
    patch_store: Any,
    *,
    round_dir: Path,
    round_index: int,
    experience: dict[str, Any],
    event_path: Path,
    authored: dict[str, Any] | None = None,
) -> str:
    """Render one measured row into its patch, or record why it became none.

    The sentence the patch carries is the reflection's, and the standing behind it is the row's
    own measurement, so the two arrive together: a row that cannot render a patch is recorded
    with its reason instead of being silently dropped, and a patch the measurement does not
    to the rung its own paper may inject from and says so in the record, and because the
    record also retires the stage prompts this run had already assembled, a later round of
    the same iteration draws it back in as soon as it plans that stage again. A patch applies
    at one stage, so the round it lands in is the next round that runs that stage.
    """

    experience_id = str(experience.get("experience_id") or "")
    rendering = build_patches_for_experience(experience, authored=authored)
    patch = rendering.get("patch")
    if not isinstance(patch, dict):
        write_json(round_dir / "prompt_patch_rejected.json", {"round": round_index, "experience_id": experience_id, **{key: value for key, value in rendering.items() if key != "patch"}})
        append_event(event_path, "prompt_patch_rejected", round=round_index, experience_id=experience_id, reason=str(rendering.get("reason") or ""))
        append_event(event_path, "prompt_patch_missing", round=round_index, experience_id=experience_id, reason=str(rendering.get("reason") or ""), note=str(rendering.get("note") or ""))
        return {}
    # A sentence the library already asks for is not a second finding. It is folded into the row
    # that stands -- one more round behind the same words -- rather than filed again under a new id,
    # which is how eighteen patches came to hold five different things.
    merge = patch_store.merge_sentence(patch) if patch_store is not None else {"ok": True, "merged": False}
    if merge.get("merged"):
        write_json(
            round_dir / "prompt_patch_candidate.json",
            {
                "patch": merge["patch"],
                "round": round_index,
                "experience_id": experience_id,
                "authored_by": str(rendering.get("authored_by") or "template"),
                "merged_into": str(merge.get("patch_id") or ""),
                "support": int(merge.get("support") or 0),
            },
        )
        append_event(
            event_path,
            "prompt_patch_merged",
            round=round_index,
            experience_id=experience_id,
            patch_id=str(merge.get("patch_id") or ""),
            directive=str(patch.get("directive") or ""),
            support=int(merge.get("support") or 0),
            note="the library already asks for this sentence at this stage; the round joins it",
        )
        permit = rendering.get("permit") if isinstance(rendering.get("permit"), dict) else {}
        already = ""
        if patch_store is not None:
            already = str((patch_store.get(str(merge.get("patch_id") or "")) or {}).get("status") or "")
        if permit.get("granted") and already not in OWN_CASE_ASSEMBLY_STATUSES:
            # The sentence was in the library on somebody else's word and this round is the first
            # measurement behind it, so the licence is granted here exactly as it would have been.
            status = str(permit.get("status") or CURRENT_CASE_STATUS)
            patch_store.transition(str(merge.get("patch_id") or ""), status, reason=str(permit.get("reason") or ""), round=round_index)
            append_event(event_path, "prompt_patch_permitted", round=round_index, experience_id=experience_id, patch_id=str(merge.get("patch_id") or ""), status=status, reason=str(permit.get("reason") or ""))
        return merge["patch"]
    stored = patch_store.append_result(patch) if patch_store is not None else {"ok": False, "reason": "the prompt patch library is disabled for this run"}
    if not stored.get("ok"):
        write_json(round_dir / "prompt_patch_rejected.json", {"round": round_index, "experience_id": experience_id, "reason": str(stored.get("reason") or ""), "errors": list(stored.get("errors") or []), "patch": patch})
        append_event(event_path, "prompt_patch_rejected", round=round_index, experience_id=experience_id, patch_id=str(patch.get("patch_id") or ""), reason=str(stored.get("reason") or ""))
        append_event(event_path, "prompt_patch_missing", round=round_index, experience_id=experience_id, patch_id=str(patch.get("patch_id") or ""), reason=str(stored.get("reason") or ""))
        return {}
    patch_id = str(patch.get("patch_id") or "")
    permit = rendering.get("permit") if isinstance(rendering.get("permit"), dict) else {}
    write_json(
        round_dir / "prompt_patch_candidate.json",
        {
            "patch": patch,
            "round": round_index,
            "experience_id": experience_id,
            "authored_by": str(rendering.get("authored_by") or "template"),
            "authored_discarded": str(rendering.get("authored_discarded") or ""),
            "derived_from": str(rendering.get("derived_from") or ""),
            "work_permit": permit,
        },
    )
    append_event(
        event_path,
        "prompt_patch_recorded",
        round=round_index,
        experience_id=experience_id,
        patch_id=patch_id,
        patch_class=str(patch.get("class") or ""),
        apply_at=str(patch.get("apply_at") or ""),
        patch_condition=dict(patch.get("condition") or {}),
        authored_by=str(rendering.get("authored_by") or "template"),
        directive=str(patch.get("directive") or ""),
    )
    if str(rendering.get("derived_from") or "") == "rule":
        # The seat stated a rule and left the patch blank. The patch is that rule's sentence, and
        # the record says so rather than letting it read as a sentence the template wrote.
        append_event(
            event_path,
            "prompt_patch_derived",
            round=round_index,
            experience_id=experience_id,
            patch_id=patch_id,
            apply_at=str(patch.get("apply_at") or ""),
            directive=str(patch.get("directive") or ""),
            reason="the seat wrote a rule and no patch, so the patch is the rule's own sentence",
        )
    if permit.get("granted"):
        status = str(permit.get("status") or CURRENT_CASE_STATUS)
        patch_store.transition(patch_id, status, reason=str(permit.get("reason") or ""), round=round_index)
        append_event(event_path, "prompt_patch_permitted", round=round_index, experience_id=experience_id, patch_id=patch_id, status=status, reason=str(permit.get("reason") or ""))
    else:
        append_event(event_path, "prompt_patch_missing", round=round_index, experience_id=experience_id, patch_id=patch_id, reason=str(permit.get("reason") or ""), note="the patch is stored as a candidate and is injected nowhere until a measurement stands behind it")
    return patch


#: The refusal reasons that are a measurement of the rules the round ran under: the comparison
#: preferred the other figure, or the improvement did not clear the acceptance dead band. A round
#: that could not be measured at all says nothing about the rules that guided it.
RULE_LOSS_REASONS = frozenset(str(item) for item in CREATOR_RULE_REFUTING_REASONS)
#: The matching accept reasons: the round was decided by a comparison or a measurement, so it is a
#: measurement of the rules it ran under and clears their streak.
RULE_WIN_REASONS = frozenset({"pairwise_preferred", "coverage_improved_over_pairwise", "target_improved"})


def _charge_injected_patches(
    *,
    patch_store: Any,
    event_path: Path,
    round_index: int,
    injected: Any,
    reason: str,
) -> list[str]:
    """Charge the patches a round drew its prompts under with that round's own outcome.

    A patch is a sentence the stage prompts carry, so a round that ran under it is a measurement
    of it: the round that did not come out better counts against the patch, and a patch that has
    lost ``PROMPT_PATCH_REFUTATION_LIMIT`` rounds in a row stops being injected. A round that won
    clears the streak -- the count is consecutive, not a lifetime tally -- and a round that could
    not be measured at all is charged to nothing, because "not measured" is not "measured worse".

    The verdicts are written as the patch library's own lifecycle events, so the patch's record
    holds the whole history rather than a number one run kept to itself.
    """

    if patch_store is None:
        return []
    patch_ids = sorted({str(item) for item in (injected or ()) if str(item)})
    if not patch_ids:
        return []
    if str(reason or "") in RULE_WIN_REASONS:
        for patch_id in patch_ids:
            patch_store.append_event(patch_id, "upheld", round=int(round_index))
            append_event(event_path, "prompt_patch_upheld", round=round_index, patch_id=patch_id)
        return patch_ids
    if str(reason or "") not in RULE_LOSS_REASONS:
        return []
    charged: list[str] = []
    for patch_id in patch_ids:
        patch_store.append_event(patch_id, "refuted", round=int(round_index), reason="the round drawn under this patch did not come out better")
        streak = patch_store.refutation_streak(patch_id)
        disabled = int(streak) >= PROMPT_PATCH_REFUTATION_LIMIT
        if disabled:
            patch_store.disable(patch_id, reason="prompt_patch_refutations_reached")
        charged.append(patch_id)
        append_event(
            event_path,
            "prompt_patch_refuted",
            round=round_index,
            patch_id=patch_id,
            refutations=int(streak),
            disabled=bool(disabled),
            reason=str(reason or ""),
        )
    return charged


def _suggestion_local_fix(
    *,
    judgement: dict[str, Any] | None,
    action: dict[str, Any] | None,
    authored: dict[str, Any] | None,
    executed: bool,
    reason: str = "",
) -> dict[str, Any]:
    """The half of a round's answer that is about this figure: what was changed, and why there.

    The stage comes from the seat's own attribution when it named one, and otherwise from the
    stage the move it ran applies at, so the local fix always names a place a prompt could be
    changed. The requirement and the evidence are the seat's words and the numbers it read them
    off; the action is the move the round actually ran.
    """

    source = judgement if isinstance(judgement, dict) else {}
    written = authored if isinstance(authored, dict) else {}
    moved = action if isinstance(action, dict) else {}
    stage = str(source.get("fault_stage") or written.get("apply_at") or stage_for_action(str(moved.get("type") or "")) or "")
    return {
        "apply_at": stage,
        "requirement": str(source.get("fault_requirement") or ""),
        "directive": str(written.get("directive") or ""),
        "action": dict(moved),
        "evidence": [dict(item) for item in (source.get("fault_evidence") or []) if isinstance(item, dict)],
        "executed": bool(executed),
        "reason": str(reason or ""),
    }


def _record_suggestion_tiers(
    *,
    round_dir: Path,
    round_index: int,
    judgement: dict[str, Any] | None,
    action: dict[str, Any] | None,
    authored: dict[str, Any] | None,
    patch: dict[str, Any] | None,
    executed: bool,
    patch_reason: str = "",
) -> dict[str, Any]:
    """Land this round's advice in both tiers: the fix it ran, and the rule it may not run yet.

    A round that rendered no patch still writes a portable tier, carrying the sentence the seat
    wrote and the reason it did not travel, so "nothing travelled" and "the travelling half was
    never written" are different records rather than one silence.
    """

    written = authored if isinstance(authored, dict) else {}
    rendered = patch if isinstance(patch, dict) else {}
    portable = {
        "patch_id": str(rendered.get("patch_id") or ""),
        "apply_at": str(rendered.get("apply_at") or written.get("apply_at") or ""),
        "directive": str(rendered.get("directive") or written.get("directive") or ""),
        "class": str(rendered.get("class") or ""),
        "status": str(rendered.get("status") or ""),
        "condition": dict(rendered.get("condition") or written.get("condition") or {}),
        "reason": "" if rendered else str(patch_reason or "the round rendered no patch"),
    }
    record = suggestion_tiers_record(
        round_index,
        local_fix=_suggestion_local_fix(judgement=judgement, action=action, authored=authored, executed=executed, reason=patch_reason),
        portable_rule=portable,
        authored_by=str((patch or {}).get("written_by") or ("model" if str(written.get("directive") or "").strip() else "template")),
        transferable_half=bool(str(written.get("directive") or "").strip() or str(rendered.get("directive") or "").strip()),
    )
    write_json(round_dir / "suggestion_tiers.json", record)
    return record


def _acceptance_input(
    *,
    method: dict[str, Any],
    program: dict[str, Any],
    icons: dict[str, Any],
    preview: Path,
    domain_profile: dict[str, Any],
    grounding_gate: dict[str, Any] | None,
    allowed_scope: dict[str, Any],
    layer_plan: dict[str, Any] | None = None,
) -> DirectJudgeInput:
    """Build the acceptance question's input for one compiled candidate.

    It carries no prior feedback and no experience: the acceptance question is "what is this
    figure worth", and a reviewer that is shown what someone wanted changed is no longer an
    independent measurement of the result.
    
    It does carry the layer this round is on. The acceptance seat is the seat that also says where
    the round goes next, and it cannot say that without the pipeline's own record of where the work
    stands. A caller that hands over no plan adds no key, so a question asked without one renders
    exactly as it did before.
    """

    return DirectJudgeInput(
        method=method,
        layout={"panels": program.get("panels", []), "nodes": program.get("semantic_nodes", []), "connectors": program.get("arrows", [])},
        icons=icons,
        program=program,
        preview=preview,
        domain_profile=domain_profile,
        allowed_scope=allowed_scope,
        grounding_gate=grounding_gate,
        layer_plan=dict(layer_plan) if isinstance(layer_plan, dict) and layer_plan else None,
    )


def _preview_path(directory: Path | None) -> Path | None:
    """Return the render one candidate directory holds, or None when there is nothing to show.

    The advisory role is asked about a pair, so it needs the render of the state the loop has
    already accepted. A directory that never rendered one is not an error: the question simply
    falls back to the single-figure form it had before.
    """

    if not directory:
        return None
    for name in ("final_draft_preview.png", "draft_preview.png", "preview.png"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _before_preview(best_dir: Path | None, baseline_from: Any) -> str:
    """The render of the figure a round started from, for the pair a person is shown.

    A round that has not accepted anything yet has an empty ``best_dir``, and for its first round
    the only "before" that exists is the version the run inherited. That version is the baseline the
    round is measured against, so its render is what belongs beside the candidate: a person asked to
    choose between two figures and shown one is not being asked a question.
    """

    shown = _preview_path(best_dir)
    if shown is None and baseline_from:
        shown = _preview_path(Path(str(baseline_from)))
    return str(shown or "")


def _accepted_preview(directory: Path | None, baseline: Any) -> Any:
    """Return the render of the state the loop has already accepted, or ``None``.

    The accepted state is the last state this iteration accepted; while nothing here has
    been accepted the run is still standing on the figure it was handed, and that figure is
    what the round is measured against. Reading the accepted directory alone left the second
    image empty until a round was accepted -- and since most rounds roll back, the advisory
    seat was asked about one figure in nearly every call. A lone figure gives it nothing to
    compare, so it answered "no defect" however the figure looked.

    A directory that exists but rendered nothing does not fall back: a round here was
    accepted, so the inherited figure is no longer the state the loop has already accepted.
    """

    render = _preview_path(directory)
    if render is not None:
        return render
    if directory:
        return None
    return getattr(baseline, "preview", None) if baseline is not None else None


def _ask_reference_gap(
    judge: Any,
    *,
    gap_reference: dict[str, Any] | None,
    selected: dict[str, Any],
    round_index: int,
    event_path: str | Path,
) -> dict[str, Any]:
    """Ask the advisory seat how this round's figure differs from a good one, if there is one.

    The question is skipped, rather than asked with nothing to compare against, when the library
    holds no usable framework figure. A seat asked to name a gap against no reference answers
    about nothing, and a round that filed that answer as a finding would read as a channel that
    works while measuring nothing. The answer is filed beside the round and carried into the
    reflection; it is never folded into a score.
    """

    reference = (gap_reference or {}).get("selected") if isinstance(gap_reference, dict) else {}
    reference = reference if isinstance(reference, dict) else {}
    if not reference:
        return {}
    ask = getattr(judge, "reference_gap", None)
    if not callable(ask):
        # A seat that does not implement the question is recorded rather than passed over in
        # silence: "this seat was never asked" and "this seat looked and saw no gap" are different
        # facts about a round, and a round that cannot tell them apart cannot judge its own advice.
        append_event(event_path, "reference_gap_skipped", round=round_index, reason="the seat does not answer the reference-gap question")
        return {}
    ours = Path(str(selected.get("dir") or "")) / "final_draft_preview.png"
    reference_path = Path(str(reference.get("source_path") or ""))
    if not ours.is_file() or not reference_path.is_file():
        return {}
    record = ask(
        ReferenceGapInput(
            reference_path=reference_path,
            ours_path=ours,
            reference_id=str(reference.get("id") or ""),
            reference_note=json.dumps(list(reference.get("why_good") or []), ensure_ascii=False)[:900],
            ours_note=f"our own output, version {ours.parent.name} of the paper being evolved",
        )
    )
    record["reference"] = reference
    record["ours_path"] = str(ours)
    append_event(
        event_path,
        "reference_gap_recorded",
        round=round_index,
        reference_id=str(reference.get("id") or ""),
        gap_class=str(record.get("gap_class") or ""),
        readable=bool(record.get("readable")),
        conforms=bool(record.get("conforms")),
        requested_actions=[str((item or {}).get("action") or "") for item in record.get("requested_actions") or []],
    )
    return record


def _measure_acceptance(judge: Any, judge_input: DirectJudgeInput) -> tuple[dict[str, Any], str]:
    """Ask the acceptance role one candidate is worth, and survive not being answered.

    A measurement that could not be taken comes back as an unmeasured answer carrying its
    reason, because "the measurement failed" and "the figure got worse" must not become the
    same record: the round is refused with ``score_measurement_failed`` rather than read
    against a fabricated zero.
    """

    try:
        return judge.evaluate(judge_input), ""
    except Exception as exc:  # noqa: BLE001 - a failed measurement is a recorded outcome, not a crash
        error = f"{type(exc).__name__}: {exc}"
        return (
            {
                "contract": "rfs.judge_result",
                "judge_role": str(getattr(judge, "role", "unknown")),
                "overall_score": None,
                "score_measured": False,
                "status": "unmeasured",
                "dimension_scores": {},
                "stage_feedback": [],
                "blocking_issues": [],
                "blocking_issue_count": 0,
                "confidence": 0.0,
                "measurement_error": error,
            },
            error,
        )


#: How many candidates one round may draw. The old code capped this at 2 by writing the number
#: into the loop; the ceiling is named now so a config can ask for more without an edit.
CANDIDATE_ROUND_LIMIT = 4


def _pairwise_capable(judge: Any) -> bool:
    """Whether a judge can answer the two-state question at all.

    The offline judge has no model behind it, so a run in that mode keeps the measured-score
    comparison and its record says so. A judge that does have the capability is never allowed
    to fall back on a failed comparison: a comparison nobody could take refuses the round.
    """

    return callable(getattr(judge, "evaluate_pairwise", None))


def _candidate_rank_key(candidate: dict[str, Any]) -> tuple[bool, float]:
    """The deterministic order candidates fall back to when no comparison can be taken."""

    return (
        measured_score(candidate.get("acceptance")) is not None,
        float(measured_score(candidate.get("acceptance")) or 0.0) * 0.7
        + float(candidate.get("visual_score") or 0.0) * 0.3,
    )


def select_round_candidate(
    candidates: list[dict[str, Any]],
    *,
    compare: "Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None] | None" = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick the round's figure and record how it was picked.

    A single candidate is the round. With more than one, the candidates settle it against each
    other first -- by the acceptance role when it can answer a two-state question, and by the
    measured-score order the loop always used when it cannot -- and only then does the winner go
    up against the state the loop already accepted.
    """

    if not candidates:
        raise ValueError("a round must produce at least one candidate")
    ranked = sorted(candidates, key=_candidate_rank_key, reverse=True)

    def _candidate_records() -> list[dict[str, Any]]:
        """The per-candidate scores as they stand now, after any comparison scored them."""

        return [
            {
                "candidate_index": int(item.get("candidate_index", item.get("variant", 0))),
                "dir": str(item.get("dir") or ""),
                "acceptance_score": measured_score(item.get("acceptance")),
                "derived_score": derived_score(item.get("acceptance")),
                # Read beside ``derived_score`` and never as a score: the number is the defect
                # table's row for the defect the seat named, written onto whichever figure was
                # blamed, so two rounds that blame different defects get different numbers for
                # the same figure. It is here so the record shows which defect was named.
                "derived_from": "the named defect, not a measurement",
                "visual_score": float(item.get("visual_score") or 0.0),
            }
            for item in candidates
        ]

    report: dict[str, Any] = {
        "contract": "rfs.round_selection",
        "version": 1,
        "candidate_count": len(candidates),
        "candidates": _candidate_records(),
        "comparisons": [],
    }
    if len(candidates) == 1:
        report["method"] = "single_candidate"
        winner = candidates[0]
    elif compare is None:
        report["method"] = "measured_score_rank"
        winner = ranked[0]
    else:
        report["method"] = "pairwise_tournament"
        winner = candidates[0]
        for challenger in candidates[1:]:
            comparison = compare(winner, challenger)
            verdict = pairwise_verdict(comparison)
            # The comparison is the only instrument that scores these two figures, so its
            # defect-table scores are written back onto both candidates. Dropping them left
            # every candidate of a compared round carrying a null score.
            winner["acceptance"], challenger["acceptance"] = _apply_choice_scores(
                winner.get("acceptance"), challenger.get("acceptance"), comparison
            )
            report["comparisons"].append(
                {
                    "incumbent": int(winner.get("candidate_index", winner.get("variant", 0))),
                    "challenger": int(challenger.get("candidate_index", challenger.get("variant", 0))),
                    "verdict": verdict,
                    "margin": ((comparison or {}).get("pairwise") or {}).get("margin") if isinstance(comparison, dict) else None,
                    # What the seat said, in its own terms: which figure it called wrong and which
                    # defect it named. A reader gets the reason rather than a looked-up number.
                    "why": str((comparison or {}).get("verdict_reason") or ""),
                }
            )
            # A comparison nobody could read leaves the incumbent in place: an unreadable answer
            # must not decide a round any more than an unmeasurable score may.
            if verdict == "after":
                winner = challenger
        # The comparison's winner is the tournament's answer; the deterministic order is
        # taken again only to report where that winner landed on it. The defect-table
        # scores the comparison wrote do not move that order, because they are derived
        # from the defect named rather than measured from the figure.
        ranked = sorted(candidates, key=_candidate_rank_key, reverse=True)
        report["candidates"] = _candidate_records()
    report["winner"] = int(winner.get("candidate_index", winner.get("variant", 0)))
    report["winner_dir"] = str(winner.get("dir") or "")
    report["winner_rank_by_score"] = ranked.index(winner)
    return winner, report


def _measure_pairwise(judge: Any, pair_input: DirectPairJudgeInput) -> tuple[dict[str, Any], str]:
    """Ask the acceptance role which of two states a reader is better served by.

    A comparison that could not be taken comes back as an answer carrying its reason, rather
    than as a verdict borrowed from subtracting two rounded numbers.
    """

    try:
        return judge.evaluate_pairwise(pair_input), ""
    except Exception as exc:  # noqa: BLE001 - a failed comparison is a recorded outcome, not a crash
        error = f"{type(exc).__name__}: {exc}"
        return (
            {
                **normalize_pairwise_result(
                    {},
                    role=str(getattr(judge, "role", "unknown")),
                    candidate_a=pair_input.candidate_a,
                    candidate_b=pair_input.candidate_b,
                ),
                "judge_role": str(getattr(judge, "role", "unknown")),
                "pairwise_unavailable": error,
            },
            error,
        )


def _with_creator_rules(brief: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Fold the measured rule list into the brief the planners already receive.

    The rules join under the keys ``exemplar_style_brief.stage4_manifest`` already uses, so
    the structure and composition planners are guided without the planning layer learning a
    second channel, and a run with no rules keeps exactly the brief it had.
    """

    source = brief if isinstance(brief, dict) else {}
    if not isinstance(manifest, dict) or not manifest.get("rules"):
        return source
    existing = source.get("stage4_manifest") if isinstance(source.get("stage4_manifest"), dict) else {}
    merged = dict(existing)
    for key, value in (manifest.get("stage4_manifest") or {}).items():
        if key == "creator_rules":
            merged[key] = value
        elif isinstance(value, list) and value:
            merged[key] = [*merged.get(key, []), *[item for item in value if item not in merged.get(key, [])]]
        elif isinstance(value, dict) and value:
            merged[key] = {**(merged.get(key) if isinstance(merged.get(key), dict) else {}), **value}
        elif key not in merged:
            merged[key] = value
    policy = source.get("stage4_execution_policy") if isinstance(source.get("stage4_execution_policy"), dict) else {}
    return {
        **source,
        "stage4_manifest": merged,
        "stage4_execution_policy": {**policy, **(manifest.get("stage4_execution_policy") or {})},
        "creator_rules": {
            "contract": manifest.get("contract"),
            "version": manifest.get("version"),
            "min_support": manifest.get("min_support"),
            "buckets": manifest.get("buckets") or {},
            "rules": manifest.get("rules") or [],
            "withheld": manifest.get("withheld") or [],
        },
    }


def _text_items_with_planned_roles(composition: dict[str, Any], text_plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Keep the role the text plan gave an item that the composition handed back.

    The composition plan answers with the same text ids as the text plan but
    describes them as boxes and styles only. Treating those boxes as the whole
    text program dropped every role: a relation label stopped being a label, so
    the arrow drew a second copy of it, and both the label and the panel title
    fell back to the unroled size range and came out below the readable
    minimum, which cost the paper a whole redo.
    """

    items = [item for item in (composition.get("text_items") or []) if isinstance(item, dict)]
    if not items:
        return [item for item in ((text_plan or {}).get("items") or []) if isinstance(item, dict)]
    planned: dict[str, dict[str, Any]] = {}
    for item in (text_plan or {}).get("items") or []:
        if not isinstance(item, dict):
            continue
        for key in (str(item.get("id") or ""), str(item.get("semantic_source_id") or "")):
            if key:
                planned.setdefault(key, item)
    merged: list[dict[str, Any]] = []
    for item in items:
        if not str(item.get("text_role") or item.get("role") or ""):
            source = planned.get(str(item.get("id") or "")) or planned.get(str(item.get("semantic_source_id") or ""))
            role = str((source or {}).get("text_role") or (source or {}).get("role") or "")
            if role:
                item = {**item, "text_role": role, "role": role}
                for key in ("semantic_source_id", "target_id", "panel_id", "font_size_role", "min_readable_size_pt", "max_lines", "priority"):
                    if source.get(key) is not None and item.get(key) is None:
                        item[key] = source[key]
        merged.append(item)
    return merged


#: What a connector means, as opposed to where it goes. The composition pass is asked for
#: placement and routing, and the model that writes it hands back connectors carrying an id, a
#: source and a target as often as not: on the own-set row 84 of 273 composed connectors arrived
#: with no type at all, and the arrows lost their relation labels with it. A connector stripped
#: of these cannot be bound back to the scientific edge it came from, which fails the Grounding
#: Gate, and the repair loop then has no change it is allowed to accept -- every round of the
#: 2026-10-08 continuous probe was refused as a new blocker for exactly this. So the structure
#: plan stays the authority on what an edge is, and the composition plan is read for where it
#: goes.
CONNECTOR_SEMANTIC_FIELDS = (
    "type",
    "relation_type",
    "label",
    "label_evidence_ids",
    "evidence_ids",
    "paper_relation_id",
    "explanation_source",
)


def _restore_connector_semantics(planned: list[Any], composed: list[Any]) -> list[dict[str, Any]]:
    """Give every composed connector back the meaning the structure plan stated for it.

    A connector is matched by id first and by its source/target pair second, because the
    composition pass may renumber the edges it was handed while keeping their endpoints. Only
    blank fields are filled: a composition plan that did state a relation keeps its own word, so
    this repairs a dropped field rather than overruling the planner that stated it. A connector
    naming no endpoint is dropped here, which is what the merge it replaces did as well.
    """

    by_id: dict[str, dict[str, Any]] = {}
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for item in planned or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or "")
        if key:
            by_id.setdefault(key, item)
        pair = (
            str(item.get("source") or item.get("source_id") or ""),
            str(item.get("target") or item.get("target_id") or ""),
        )
        if pair[0] and pair[1]:
            by_pair.setdefault(pair, item)
    restored: list[dict[str, Any]] = []
    for index, connector in enumerate(composed or []):
        if not isinstance(connector, dict):
            continue
        source = connector.get("source") or connector.get("source_id")
        target = connector.get("target") or connector.get("target_id")
        if not (source and target):
            continue
        merged = {
            **connector,
            "id": str(connector.get("id") or f"composition_edge_{index + 1:02d}"),
            "source": source,
            "target": target,
        }
        planned_item = by_id.get(str(merged["id"])) or by_pair.get((str(source), str(target))) or {}
        for field in CONNECTOR_SEMANTIC_FIELDS:
            if not merged.get(field) and planned_item.get(field):
                merged[field] = planned_item[field]
        restored.append(merged)
    return restored


def _compile_direct(
    specification: dict[str, Any],
    layout: dict[str, Any],
    assets: dict[str, Any],
    out: Path,
    title: str | None,
    *,
    text_plan: dict[str, Any] | None = None,
    composition: dict[str, Any] | None = None,
    reference_template: dict[str, Any] | None = None,
    panel_style_plan: dict[str, Any] | None = None,
    memory_action: dict[str, Any] | None = None,
    person_rules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    composition = composition if isinstance(composition, dict) else {}
    overrides = {
        str(item.get("id")): item
        for item in composition.get("nodes", [])
        if isinstance(item, dict) and item.get("id")
    }
    if overrides:
        # The overlay is a styling pass; a stale box of its own must not be able
        # to drop a card on top of its neighbour after the structure stage has
        # already resolved the geometry.
        layout, overlay_guard = guard_overlay_geometry(layout, node_overrides=overrides)
    else:
        overlay_guard = {"status": "pass", "rejected_count": 0, "rejected": []}
    if composition.get("connectors"):
        normalized_connectors = _restore_connector_semantics(layout.get("connectors") or [], composition["connectors"])
        if normalized_connectors:
            layout = {**layout, "connectors": normalized_connectors}
    node_map = {str(node.get("id")): node for node in layout.get("nodes", [])}
    connectors = []
    for edge in layout.get("connectors", []):
        source, target = node_map.get(str(edge.get("source"))), node_map.get(
            str(edge.get("target"))
        )
        # Preserve an already selected route.  If no route exists, leave it
        # empty so the canonical connector router can choose an edge-to-edge
        # path with obstacle avoidance; never synthesize center-to-center
        # geometry here because that overwrites visual-review repairs.
        path = edge.get("path_percent") if isinstance(edge.get("path_percent"), list) else []
        connectors.append({**edge, "path_percent": path})
    semantic_plan = {
        "applied": True,
        "topology": layout.get("topology"),
        "nodes": [{**node, "field": "modules"} for node in layout.get("nodes", [])],
        "connectors": connectors,
    }
    program = build_semantic_figure_program(
        specification,
        aspect_ratio=layout.get("canvas", {}).get("aspect_ratio", "16:9"),
        title=title,
        show_title=bool(title),
        semantic_plan=semantic_plan,
    )
    layout_nodes = {str(item.get("id") or ""): item for item in layout.get("nodes", []) if isinstance(item, dict)}
    for node in program.get("semantic_nodes", []):
        source = layout_nodes.get(str(node.get("id") or ""), {})
        for field in ("semantic_source_id", "evidence_ids", "must_show", "should_render", "parent_id", "detail_level", "kind", "render_policy", "object_type", "visual_role", "container_child_ids"):
            if field in source:
                node[field] = source[field]
    asset_paths = {
        str(item.get("asset_id")): str(item.get("path"))
        for item in assets.get("assets", [])
    }
    # Bind generated icon assets back to semantic nodes even when the planner
    # omitted asset_id. This keeps generation and PPT composition contracts in
    # sync without changing the planner's layout semantics.
    for node in layout.get("nodes", []):
        if not isinstance(node, dict) or node.get("asset_id"):
            continue
        candidate_id = f"icon_{node.get('id')}"
        if candidate_id in asset_paths:
            node["asset_id"] = candidate_id
    slots = []
    for node in layout.get("nodes", []):
        if str(node.get("render_policy") or "").casefold() == "panel_only" or not node.get("asset_id"):
            continue
        bbox = node["bbox_percent"]
        asset_id = str(node.get("asset_id") or f"icon_{node['id']}")
        slots.append(
            {
                "id": f"slot_{node['id']}",
                "asset_id": asset_id,
                "paper_concept": node.get("label") or node["id"],
                "bbox_percent": {
                    "x": bbox["x"] + bbox["w"] * 0.16,
                    "y": bbox["y"] + bbox["h"] * 0.08,
                    "w": bbox["w"] * 0.68,
                    "h": bbox["h"] * 0.58,
                },
                "composition_type": "symbol_cutout",
                "show_slot_caption": False,
                "z_index": 35,
            }
        )
    program["slots"] = slots
    program["assets"] = [
        {"id": key, "path": value, "source": "direct_icon"}
        for key, value in asset_paths.items()
    ]
    if composition.get("panels"):
        # Composition planning is an overlay: VLM may return only geometry
        # fields. Merge by stable panel ID so structural titles/roles are not
        # dropped when the overlay is sparse.
        planned_panels = {str(item.get("id")): item for item in composition["panels"] if isinstance(item, dict) and item.get("id")}
        # The composition VLM is an overlay planner and is allowed to omit
        # semantic fields such as panel titles.  Normalize those fields at
        # this boundary so the deterministic compiler never dereferences a
        # missing title (and so sparse overlays remain backwards compatible).
        structural_panels = {str(item.get("id")): item for item in program.get("panels", []) if isinstance(item, dict) and item.get("id")}
        for panel_id, item in list(planned_panels.items()):
            if not str(item.get("title") or "").strip():
                fallback = structural_panels.get(panel_id, {})
                item = {**item, "title": str(fallback.get("title") or fallback.get("label") or panel_id.replace("_", " ").title())}
                planned_panels[panel_id] = item
        guarded_panels, panel_guard = guard_overlay_geometry(
            {"panels": program.get("panels", []), "nodes": program.get("semantic_nodes", [])},
            panel_overrides=planned_panels,
        )
        program["panels"] = guarded_panels["panels"]
        overlay_guard = {
            **overlay_guard,
            "status": "corrected" if (overlay_guard.get("rejected") or panel_guard.get("rejected")) else "pass",
            "rejected_count": int(overlay_guard.get("rejected_count") or 0) + int(panel_guard.get("rejected_count") or 0),
            "rejected": [*(overlay_guard.get("rejected") or []), *(panel_guard.get("rejected") or [])],
        }
        known_panel_ids = {str(panel.get("id")) for panel in program["panels"]}
        program["panels"].extend(item for panel_id, item in planned_panels.items() if panel_id not in known_panel_ids)
    # All compiler-facing panels must have a human-readable title, including
    # panels introduced by a VLM overlay or by future planner variants.
    for panel in program.get("panels", []):
        if isinstance(panel, dict) and not str(panel.get("title") or "").strip():
            panel["title"] = str(panel.get("label") or panel.get("id") or "Panel").replace("_", " ").title()
    # The title owns the top of the canvas. A panel box may arrive from the
    # structure stage or from the composition overlay, so the band is reserved
    # once, here, on the final list rather than only on one of those paths.
    program["panels"], title_band_report = keep_panels_below_title_band(
        program.get("panels", []), program.get("title_block")
    )
    placements = {
        str(item.get("asset_id") or item.get("node_id")): item
        for item in composition.get("icon_placements", [])
        if isinstance(item, dict)
    }
    for slot in program["slots"]:
        placement = placements.get(str(slot.get("asset_id"))) or placements.get(
            str(slot.get("id", "")).removeprefix("slot_")
        )
        if placement and isinstance(placement.get("bbox_percent"), dict):
            slot["bbox_percent"] = placement["bbox_percent"]
    text_items = _text_items_with_planned_roles(composition, text_plan)
    node_label_items = {
        str(item.get("semantic_source_id")): item
        for item in text_items
        if isinstance(item, dict)
        and str(item.get("text_role") or item.get("role") or "").casefold() in {"node_label", "primary_node", "secondary_node"}
    }
    for node in program.get("semantic_nodes", []):
        planned = node_label_items.get(str(node.get("semantic_source_id") or node.get("semantic_node_id") or node.get("id")))
        if planned:
            node["label"] = planned.get("short_text") or planned.get("text") or node.get("label")
            node["font_size_pt"] = planned.get("font_size_pt") or (14 if int(planned.get("priority") or 2) <= 1 else 11)
            node["font_size_role"] = planned.get("font_size_role") or ("primary_node" if int(planned.get("priority") or 2) <= 1 else "secondary_node")
            node["bold"] = str(planned.get("font_weight") or "bold").lower() == "bold"
    program["text_program"] = {
        "summary": "VLM-planned native editable text.",
        "items": [
            {
                **item,
                "visible": True,
                "layer_ownership": "editable_text_layer",
                "editable_in": "pptx",
            }
            for item in text_items
            if isinstance(item, dict)
            and not (
                str(item.get("text_role") or item.get("role") or "").casefold() == "panel_title"
                and "".join(ch for ch in str(item.get("text") or "").casefold() if ch.isalnum())
                in {
                    "".join(ch for ch in str(node.get("label") or "").casefold() if ch.isalnum())
                    for node in program.get("semantic_nodes", [])
                    if str(node.get("render_policy") or "").casefold() not in {"panel_only", "hidden"}
                }
            )
            and isinstance(item.get("bbox_percent"), dict)
        ],
    }
    program = apply_color_theme_to_program(program, panel_style_plan)
    program, joint_geometry_report = resolve_joint_geometry(program)
    write_json(out / "joint_geometry_report.json", joint_geometry_report)
    # Icon rasters carry a transparent frame around the mark. Trim it first so
    # the slot geometry below maps onto the glyph that actually reaches the
    # slide instead of onto invisible padding.
    program, asset_trim_report = trim_program_icon_assets(program)
    write_json(out / "asset_trim_report.json", asset_trim_report)
    program, node_layout_report = arrange_node_contents(program)
    write_json(out / "node_internal_layout_report.json", node_layout_report)
    # A mechanism-level layout rule travels on the Memory row that learned it and is applied to
    # whatever geometry was just laid out, so the rule is about the drawing, not the figure it
    # was measured on. It runs before the text and connector passes so they see final boxes.
    layout_rules = layout_rules_from((memory_action or {}).get("layout_constraints")) or layout_rules_from(memory_action)
    # A rule the library learned wins outright, so an ON/OFF pair still differs by exactly that
    # rule. When the library names none the hand-written standard's own rule is applied instead:
    # without it an empty shelf meant no layout rule ran at all, and every panel kept the sparse
    # geometry the model drew. The report says which of the two was used.
    layout_rule_source = "memory" if layout_rules else "curated_standard"
    program, layout_constraint_report = apply_layout_constraints(program, layout_rules or dict(CURATED_LAYOUT_RULES))
    layout_constraint_report["source"] = layout_rule_source
    write_json(out / "layout_constraint_report.json", layout_constraint_report)
    # Node-internal layout supplies the final label bbox. Canonicalize only
    # after that pass so every visible label has exactly one text owner.
    program, text_ownership_report = canonicalize_node_text_program(program)
    write_json(out / "text_ownership_report.json", text_ownership_report)
    program, connector_report = resolve_direct_connectors(program)
    # Geometry and relation labels are resolved first; the canonical arrow
    # resolver then makes the single final path/style decision.  This prevents
    # compatibility defaults from masking semantic/reference-aware styling.
    memory = memory_action if isinstance(memory_action, dict) else {}
    router_rules = router_rules_from(memory.get("routing_constraints")) or router_rules_from(memory)
    person_router = person_router_rules(person_rules, "ppt_compilation")
    if person_router:
        # A person's rule settles the drawing after the library has said its part: the library's
        # rule was measured on another figure, the sentence was said about this one.
        router_rules = {**router_rules, **person_router}
        person_router_report = {"contract": "rfs.person_router_rules", "version": 1, "rules": dict(person_router)}
        write_json(out / "person_router_rules.json", person_router_report)
    program = style_and_route_arrows(program, out, mode="reference", repair_routing=True, constraints=router_rules)
    routing = memory.get("routing_constraints") if isinstance(memory.get("routing_constraints"), dict) else {}
    if routing.get("selections"):
        program, memory_route_report = apply_memory_routing_constraints(program, routing)
        write_json(out / "memory_routing_report.json", memory_route_report)
    density_targets = memory.get("density_targets") if isinstance(memory.get("density_targets"), dict) else {}
    if density_targets:
        program["density_targets"] = density_targets
    visibility_report = evaluate_connector_visibility(program)
    program = apply_color_theme_to_program(program, panel_style_plan)
    if replayable_theme(memory.get("color_theme")):
        themed_plan = dict(panel_style_plan or {})
        themed_plan["color_theme"] = replayable_theme(memory.get("color_theme"))
        program = apply_color_theme_to_program(program, themed_plan)
    program, typography_report = optimize_direct_typography(program)
    program, text_layout_report = fit_text_program_globally(program)
    write_json(out / "connector_geometry_report.json", connector_report)
    write_json(out / "connector_visibility_report.json", visibility_report)
    write_json(out / "typography_plan.json", typography_report)
    write_json(out / "global_text_layout_report.json", text_layout_report)
    readability_report = evaluate_text_readability(program)
    write_json(out / "text_readability_report.json", readability_report)
    geometry_report = evaluate_geometry_quality(program)
    write_json(out / "geometry_quality_report.json", geometry_report)
    density_report = evaluate_density_quality(program)
    write_json(out / "density_quality_report.json", density_report)
    reference_report = evaluate_reference_adoption(reference_template or {}, composition)
    write_json(out / "reference_adoption_report.json", reference_report)
    write_json(out / "overlay_geometry_guard.json", overlay_guard)
    write_json(out / "title_band_report.json", title_band_report)
    write_json(out / "figure_program.json", program)
    pptx = compile_ppt(program, out)
    return {
        "pptx": str(pptx),
        "figure_program": str(out / "figure_program.json"),
        "node_count": len(program.get("semantic_nodes", [])),
        "connector_count": len(program.get("arrows", [])),
    }


def _report_model_call(progress: Any, call: str, round_index: int, variant: int, metadata: dict[str, Any] | None) -> None:
    """Put one model call's own timing on the progress stream.

    A stage clock says how long a stage held; it cannot say how much of that was a model. The
    carrier event is written with a terminal status but is never paired with a ``started``, so
    it adds this call to the round's ledger without closing the stage the call ran inside.
    """

    provider = metadata if isinstance(metadata, dict) else {}
    if not provider:
        return
    progress({"stage": "vlm_call", "status": "completed", "call": call, "round": round_index, "variant": variant, "provider": dict(provider)})


def _preview_from_deck(program_path: Path, out_dir: Path, preview_path: Path) -> Path:
    """Draw the preview from the compiled deck this attempt will deliver.

    The visual reviewer only ever sees a PNG, so a preview drawn by a second
    renderer silently changes what gets reviewed. Rendering the compiled deck
    keeps the review pointed at the delivered figure; when no Office engine can
    do it, the program renderer draws instead and ``preview_render_report.json``
    records which drawing was used.
    """

    program = read_json(program_path)
    preview, info = render_deck_preview(program, out_dir / "editable_composition.pptx", out_dir, preview_path)
    report_path = out_dir / "preview_render_report.json"
    report = read_json(report_path) if report_path.exists() else {
        "summary": "How each preview of this attempt was drawn.",
        "renders": [],
    }
    report.setdefault("renders", []).append({**info, "program": str(program_path)})
    write_json(report_path, report)
    return preview


def _upstream_method_reuse(prepared: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The middle layer's method contract, when this run adopted one it can use.

    The middle layer read the same paper, so the plan it wrote is worth folding into
    the plan this run extracts for itself. It is handed back rather than used in place
    of that extraction: what the middleware wrote is aimed at the specification it
    built for itself and at no panel target, and this run's own plan is the one aimed
    at the figure the paper printed. A missing or self-flagged contract leaves nothing
    to fold and this run plans alone, which is what it did before upstream contracts
    existed.
    """
    content = prepared.get("frozen_method")
    if not isinstance(content, dict) or not content:
        return {}, {}
    stored = prepared.get("frozen_method_report") if isinstance(prepared.get("frozen_method_report"), dict) else {}
    if stored.get("ok") is False:
        return {}, {}
    return content, {
        "summary": "Method contract read from the middle layer's frozen contract.",
        "ok": True,
        "mode": "frozen_upstream",
        "fallback_used": False,
        "source": str((prepared.get("frozen_contract") or {}).get("source") or ""),
        "node_count": sum(
            len(content.get(field) or [])
            for field in ("inputs", "modules", "outputs", "innovations")
        ),
        "relation_count": len(content.get("relations") or []),
        "stored_report": stored,
    }


def _build_scientific_layers(
    method: dict[str, Any],
    *,
    paper_id: str,
    evidence_map: dict[str, Any] | None,
    root: Path,
    provenance: str = "",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the paper's scientific graph and subgraphs, and write both out.

    The graph is the paper's own topology, and it is also the substrate the
    Grounding Gate, the bindings and the alignment reports are measured against.
    Whether it is built before planning or after the design is the whole
    difference between the full configuration and the row that answers "does the
    explicit structure earn its keep", so both call sites go through here and
    write to the same place.

    ``provenance`` is the stamp the graph report carries when the graph was not
    built before planning. It is written here rather than patched in by the
    caller, because the report reaches disk on the way out.
    """

    graph, ground_truth, graph_report = build_scientific_graph(
        method,
        paper_id=paper_id,
        evidence_map=evidence_map,
    )
    if provenance:
        graph_report["provenance"] = provenance
    subgraphs, subgraphs_report = extract_subgraphs(graph)
    write_json(root / "scientific_graph.json", graph)
    write_json(root / "scientific_ground_truth.json", ground_truth)
    write_json(root / "scientific_graph_report.json", graph_report)
    # The subgraphs are written in both switch positions on purpose: the choice
    # only decides whether planning reads them, and a row that did not read them
    # still has to carry the same artifact set as the row that did, for
    # comparison and for retrieval.
    write_json(root / "scientific_subgraphs.json", subgraphs)
    write_json(root / "scientific_subgraphs_report.json", subgraphs_report)
    return graph, ground_truth, graph_report


def _resolve_paper_id(
    known: str,
    prepared: dict[str, Any],
    paper_review: dict[str, Any],
    paper: str | Path,
) -> str:
    """Name the paper this run is about, preferring what the caller already knows.

    The identifier the caller was started with is read first, because it can be the only place the
    name exists: a paper the middle layer indexed under a path carries a title but no identifier,
    and a title names no paper on disk. Reading the input path instead names the paper after the
    folder it sits in -- for these runs that folder is called ``paper``, which is the same name for
    every paper and therefore no name at all -- and that name is what made the reference-gap probe
    read the library folder ``paperimage`` as this paper's own output and keep every exemplar out
    of the pool. The path stays as the last resort for a caller that passes nothing at all and
    hands the run a file, where the stem is at least this paper's own name.

    A run handed the middle layer's contract reads the contract's own provenance next: its review
    carries no ``paper_id``, so reading only the identity block would detach every artifact this
    run writes from the paper the middle layer indexed.
    """

    paper_identity = paper_review.get("paper_identity") or {}
    contract_provenance = paper_review.get("graph_provenance") or {}
    return str(
        str(known or "").strip()
        or prepared.get("paper_id")
        or paper_identity.get("paper_id")
        or contract_provenance.get("paper_id")
        or paper_identity.get("doi")
        or Path(paper).stem
    )


def run_paper_evolve_editable(
    paper: str | Path,
    out: str | Path,
    *,
    paper_id: str = "",
    reference: str | Path | None = None,
    read_mode: str = "fast",
    planner_mode: str = "heuristic",
    aspect_ratio: str = "16:9",
    max_rounds: int = 3,
    candidates_per_round: int = 2,
    icon_mode: str = "image2",
    judge_mode: str = "heuristic",
    judge_model: str | None = None,
    experience_enabled: bool = True,
    experience_library: str | Path | None = None,
    retrieval_statuses: Any = None,
    evolution_enabled: bool = True,
    method_mode: str | None = None,
    reference_mode: str = "heuristic",
    icon_candidates: int = 1,
    structure_mode: str = "vlm",
    text_plan_mode: str = "vlm",
    ppt_plan_mode: str = "vlm",
    structure_model: str | None = None,
    text_plan_model: str | None = None,
    ppt_plan_model: str | None = None,
    color_theme_mode: str = "vlm",
    color_theme_model: str | None = None,
    vlm_timeout: int = 180,
    vlm_retries: int = 1,
    reference_library: str | Path | None = None,
    reference_top_k: int = 5,
    exemplar_guidance: bool = True,
    curated_standard: str | Path | None = None,
    visual_review_mode: str = "auto",
    visual_review_model: str | None = None,
    domain_profile: str = "auto",
    online_judge_model: str | None = None,
    frozen_judge_model: str | None = None,
    judge_registry: str | Path | None = None,
    judge_model_id: str | None = None,
    max_retries_per_experience: int = 3,
    evolution_scope: str = "joint",
    seed_experience: dict[str, Any] | None = None,
    seed_prompt_patch: dict[str, Any] | None = None,
    fixed_rounds: bool = False,
    run_seed: int = 1729,
    replay_upstream_from: str | Path | None = None,
    fork_before: str = "",
    reuse_replayed_icons: str = "on",
    reuse_upstream_from: str | Path | None = None,
    upstream_from: str | Path | None = None,
    baseline_from: str | Path | None = None,
    first_round_action: dict[str, Any] | None = None,
    bulk_figures: bool = False,
    upstream_retry_enabled: bool = True,
    frozen_reground_min_ratio: float | None = None,
    acceptance_checkpoint: str = "",
    acceptance_base_model: str = "",
    acceptance_generate: Any = None,
    #: How the acceptance seat is asked about the round's own layer: ``off`` keeps the two-field
    #: question, ``record`` files its answer without acting on it, ``authority`` lets that answer
    #: move the work face. Read from the continuous block, passed through to the seat.
    acceptance_direction: str = "off",
    #: The person sitting at this round's judgement points, when one is. Called with the two figures,
    #: their readings and the model's own answer; what it returns decides, and a round nobody sits at
    #: never calls it, so a run with no person at the seat is unchanged.
    human_gate: Any = None,
    online_judge_checkpoint: str = "",
    online_judge_base_model: str = "",
    online_ask_json: Any = None,
    judge_holdout_paths: tuple[str, ...] | list[str] = (),
    graph_context: bool = True,
    middle_layer: str = "on",
    #: What the round re-asks between its two moves: the operator's stop file, the disk
    #: floor in either of the two units the loop states it in, and the wall-clock deadline
    #: of this call's budget. Absent keys mean "that guard is not armed", never "assume it
    #: is fine".
    move_guard: dict[str, Any] | None = None,
    figure_target: str = "overview",
    #: What a person asked for, in the shape a suggestion rule and a prompt patch share. It is
    #: handed to the reflection as a standing instruction about how this figure should be drawn,
    #: so a person's own reading reaches the round instead of staying a note beside the run.
    human_preferences: list[dict[str, Any]] | None = None,
    #: Where those instructions are filed. Named rather than handed over as a list, because the
    #: list is read again at the top of every round: a rule filed while the iteration is running
    #: belongs to the round that follows it, not to the iteration after it. The list above stays
    #: the fallback for a folder nobody can read, so a run without the channel carries what it did.
    human_preferences_dir: str | Path | None = None,
    #: The layer this round is allowed to work on, the order the layers are built in, and the
    #: question the reflection has to answer about it. Whether a layer is finished with is the
    #: model's judgement and only the model's, and the reflection is where it is given -- so the
    #: question has to reach the reflection, or a round works on whatever looks worst and a layer
    #: that is already right gets planned again. A round handed no plan adds no key, so every other
    #: reflection prompt stays byte-identical.
    layer_plan: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Create a direct editable baseline with explicit Judge-ready artifacts.

    ``reuse_upstream_from`` is the same idea as ``replay_upstream_from`` for a
    paper that is being tried again: the reading, the contracts and the Icons
    are already on disk and already paid for, so a retry keeps them and re-does
    the layout the verdict complained about.  A source that no longer matches
    the input is declined here instead of ending the run, which is what the
    strict ``replay_upstream_from`` is for.

    ``fork_before`` names the first stage a paired run plans for itself while the run it is
    being compared against supplies everything before it. It is what makes a controlled pair
    controlled: both arms start from the same frozen prefix and differ only from the fork
    stage on, instead of each planning its own colours and names and being judged apart.

    ``upstream_from`` points at the middle layer's directory for this paper.
    When that directory carries a frozen figure specification and the review
    behind it, this run adopts them as the authority for what the figure has to
    show and plans nothing about the paper itself; when it cannot be read, the
    reason is recorded and the contract is planned in this run as before.

    ``baseline_from`` points at a finished run for this paper whose kept figure is the
    state this run's first round is measured against. The continuous runner hands it the
    previous iteration, so a first round is a real comparison instead of an unmeasured
    starting point; a run that was not handed one records ``baseline_unavailable`` rather
    than inventing a score for it.

    ``first_round_action`` is the change the first round of an iteration that inherited a
    finished figure runs. The review of that figure already named what is wrong with it, so
    the run opens by executing that change against the state it inherited instead of
    planning the same stages again -- an inherited version whose stages are planned a second
    time has been paid for twice, and the review it was just handed goes unread. The opening
    is only honoured when the state it acts on is the inherited one; a run that planned its
    own design or icons keeps its ``initial`` first round, because a change asked of the old
    figure is not a change to this one.

    ``judge_registry`` and ``judge_model_id`` name the registered checkpoint this run is
    measured by. They are resolved before anything is read or written, because a checkpoint
    that is not allowed to serve should cost nothing, and the identity they resolve to is
    what every Memory row of this run carries in place of the served model name.

    ``middle_layer`` is the switch the supplementary ablation turns off. ``on`` is the
    full configuration: the paper is turned into an explicit scientific graph, its
    subgraphs and a scientific hierarchy plan, and planning reads them. ``off`` never
    builds any of the three while planning, so the figure is planned from the method
    contract alone; the graph is rebuilt once the design exists, purely as the
    measurement the Grounding Gate and the alignment reports are computed against, so
    the two rows differ in exactly one thing.

    ``read_mode`` is how much of the paper is read before anything is drawn. ``fast`` is
    the product's own read: one planning call over a shortened evidence budget, with no
    reference image handed to the planner. ``full`` pays for a second opinion first -- a
    review pass of its own -- then hands the planner the reference images and a wider
    evidence budget, and merges the deterministic review plan into the model's. It is a
    different ruler: the review it pays for is a call the fast read never makes, so the
    two must not be mixed inside one row.
    """
    middle_layer = "off" if str(middle_layer).strip().casefold() in {"off", "false", "0", "no"} else "on"
    read_mode = "full" if str(read_mode).strip().casefold() == "full" else "fast"
    evidence_structure = middle_layer == "on"
    if not evidence_structure:
        # The middle layer is the paper's explicit structure: the scientific graph,
        # its subgraphs and the scientific hierarchy plan. This row removes it, so
        # planning is given the method contract alone and has no graph to read. The
        # graph is built again after the design (see below) purely as the
        # measurement the Grounding Gate and the alignment reports are computed
        # against, so the repair loop is the same one the full configuration runs.
        graph_context = False
    judge_binding = resolve_judge_binding(judge_mode=judge_mode, registry=judge_registry, model_id=judge_model_id)
    judge_checkpoint = str(judge_binding.get("identity") or "")
    root = ensure_dir(out).resolve()
    # The run's own event log is named here because the patch assembler reads it: which patch a
    # stage injected is recorded when the stage's prompt is built, and that happens before the
    # round loop below is set up.
    event_path = root / "evolution_events.jsonl"
    run_started = time.perf_counter()
    if visual_review_mode == "auto":
        visual_review_mode = "vlm" if ppt_plan_mode == "vlm" else "heuristic"
    # Prompt patches are a second view of the same measured Memory: they live beside the
    # experience library, and each stage resolves its own once, keeping the audit trail of
    # which patch its prompt carried and which patch lost the precedence decision.
    patch_library = Path(experience_library) if experience_library else root.parent / "paper_evolve_experience_library"
    # Which rungs this run is willing to draw Memory from. A named bar has to be resolved before
    # the first round, so a name the library does not know refuses the run instead of quietly
    # reading the default one and reporting a bar it never applied.
    retrieval_bar = resolve_retrieval_statuses(retrieval_statuses)
    store = ExperienceStore(patch_library) if experience_enabled else None
    patch_store = PromptPatchStore(patch_library) if experience_enabled else None
    # The advice rules a reflection states (P9) are kept beside the library that measures them.
    # Only rules past the support bar ever reach a prompt, and only the advisory one reads them.
    suggestion_store = SuggestionRuleStore(patch_library) if experience_enabled else None
    # The claims a reflection writes are settled by later rounds and by other papers, so the
    # ledger sits beside the library rather than inside one run's directory.
    claim_ledger = ClaimLedger(patch_library) if experience_enabled else None
    seeded_prompt_patch = normalize_prompt_patch(seed_prompt_patch) if isinstance(seed_prompt_patch, dict) and seed_prompt_patch else None
    prompt_patch_stages: dict[str, dict[str, Any]] = {}
    # The patch ids this run's own rounds earned the right to inject. Read by ``stage_patches``
    # below, which is why it is declared before the assembler rather than beside the round loop.
    own_case_patch_ids: set[str] = set()
    # A stage prompt is assembled once and reused, which is what made a patch this run permitted
    # miss its own iteration: the prompt it belonged in had already been assembled. This
    # generation is bumped when a patch is permitted, and a stage resolved under an older
    # generation is stale, so the next time that stage is asked the patch is assembled in.
    prompt_patch_generation = 0
    # The patches the prompts this round's candidate was drawn under, and every stage prompt the
    # run assembled with the round it was assembled in. The first is what the round is measured
    # against; the second is the reading that says which prompt version a round ran on.
    round_injection_ids: set[str] = set()
    prompt_patch_timeline: list[dict[str, Any]] = []
    prompt_timeline_round = {"round": 0}
    # The rung a patch earns from its own paper travels with the paper rather than only with the
    # run that measured it: every row records the paper it was measured on, so a later iteration
    # of that same paper may draw the patch back in. Another paper resolves to another path, so
    # the match is what lets one paper reuse its own measurement and stops it reaching a paper
    # that measured nothing.
    own_case_run_key = str(Path(paper).resolve())
    # The Creator side is guided by the aggregated rule list, which is read once here so the
    # structure and composition prompts are planned under the same guidance all run long.
    # Rows arrive with their effective status, so a quarantine or a refutation recorded as a
    # lifecycle event keeps a row out of the rules exactly as it keeps it out of retrieval.
    creator_ledger = CreatorRuleLedger(patch_library) if experience_enabled else None
    creator_rules_manifest = (
        read_creator_rule_manifest(
            experiences=store.records_with_status(),
            patches=patch_store.records_with_status(),
            ledger=creator_ledger,
            experience_library=patch_library,
            domain=str(domain_profile or ""),
        )
        if store is not None and patch_store is not None
        else normalize_creator_rule_manifest({})
    )
    # The hand-written standard rides beside the measured rules under its own source, so the
    # planners are steered by an explicit idea of "good" without any of it being filed as a
    # finding. An empty value turns it off and leaves the measured manifest exactly as it was.
    # The refutation ledger is applied to the union: a standard is built from the authored list
    # rather than aggregated from measured rows, so without this pass the ledger could write one
    # off and the standard would go on being injected under it.
    curated_manifest = (
        load_curated_standard(curated_standard)
        if not (
            isinstance(curated_standard, str)
            and curated_standard.strip().casefold() in {"", "off", "none", "disabled"}
        )
        else normalize_creator_rule_manifest({})
    )
    creator_rules_manifest = merge_creator_rule_manifests(
        creator_rules_manifest,
        curated_manifest,
        refutations=creator_ledger.counters() if creator_ledger is not None else None,
    )

    # The person's own instructions as of the round now running. ``stage_patches`` reads this, so a
    # sentence a person writes after looking at a figure reaches the stages that draw, in the same
    # iteration. It is a holder rather than an argument because the stages are assembled deep inside
    # the round while the list is re-read at the top of it.
    person_rules: dict[str, Any] = {
        "current": [dict(item) for item in (human_preferences or []) if isinstance(item, dict)]
    }

    def reading_feedback(note: str = "") -> str:
        """The instructions the reading and the content plan have earned, and this round's note.

        These two steps state their instructions in prose rather than in a prompt block, so a patch
        applying at either of them rides in the feedback slot the reading prompt already has.
        Without this the two stages could store a patch and never inject it: the contract accepts
        both stage names, and no assembly step asked for either.
        """

        block = render_patch_block(
            [item for stage in ("paper_reading", "content_planning") for item in stage_patches(stage)]
        ).strip()
        parts = [part for part in (block, str(note or "").strip()) if part]
        return "\n".join(parts)

    def note_stage_prompt(stage: str, *, revision: str, patch_ids: Any, cached: bool) -> None:
        """Write down which prompt version one round ran a stage under.

        Every resolution is written, including the ones served from the cache, so a round that
        reused the previous round's prompt reads as that prompt rather than as no prompt at all.
        Two identical readings for the same round and stage are collapsed, so asking a stage
        twice under one prompt does not read as two different versions.
        """

        reading = {
            "round": int(prompt_timeline_round.get("round") or 0),
            "stage": str(stage),
            "revision": str(revision or ""),
            "generation": int(prompt_patch_generation),
            "cache_hit": bool(cached),
            "patch_ids": [str(item) for item in (patch_ids or []) if str(item)],
        }
        for seen in prompt_patch_timeline:
            if (
                int(seen.get("round") or 0) == reading["round"]
                and str(seen.get("stage") or "") == reading["stage"]
                and int(seen.get("generation") or 0) == reading["generation"]
                and [str(item) for item in (seen.get("patch_ids") or [])] == reading["patch_ids"]
            ):
                return
        prompt_patch_timeline.append(reading)

    def stage_patches(stage: str) -> list[dict[str, Any]]:
        """Return the patches one stage injects, resolving and auditing them once.

        Injection is the step that used to leave no trace at all: a stage either received a
        patch or did not, and the run recorded neither. Both answers are written down here --
        which patch a stage took, and why it took none -- because "no patch was injected" and
        "the injection path is broken" read identically without them.
        """

        cached = prompt_patch_stages.get(stage)
        if cached is not None and int(cached.get("generation") or 0) == prompt_patch_generation:
            note_stage_prompt(
                stage,
                revision=str(cached.get("revision") or ""),
                patch_ids=[str(item.get("patch_id") or "") for item in cached.get("patches") or []],
                cached=True,
            )
            return cached["patches"]
        if seeded_prompt_patch is not None:
            selection = assemble_prompt_patches(
                [seeded_prompt_patch] if str(seeded_prompt_patch["apply_at"]) == stage else [],
                apply_at=stage,
                statuses={str(seeded_prompt_patch["status"])},
            )
        elif patch_store is not None:
            # The rung a patch earned from this paper's own case is read here and nowhere else:
            # its measurement is this run's own round, so the row it stands for is offered back
            # to this run and to no other paper. As the run may be one of several writing the
            # same library, the row itself is not enough to tell whose measurement it was -- the
            # ids this run permitted are the ones that travel.
            candidates = select_own_case_candidates(
                patch_store.records_with_status(), paper_key=own_case_run_key, permitted_ids=own_case_patch_ids
            )
            selection = assemble_prompt_patches(candidates, apply_at=stage, statuses=OWN_CASE_ASSEMBLY_STATUSES)
        else:
            selection = assemble_prompt_patches([], apply_at=stage)
        person_rows = person_rule_patches(person_rules["current"], stage)
        if person_rows:
            # A person's sentence outranks a patch that says the same thing: the patch was learned
            # from a figure, the sentence was said while looking at this one. Dropping the patch
            # rather than the person is also what makes the two visible as a disagreement, because
            # the injection names the sentence and the patch stops being named.
            said = {str(item.get("directive") or "").strip().casefold() for item in person_rows}
            selection["patches"] = person_rows + [
                item
                for item in selection["patches"]
                if str(item.get("directive") or "").strip().casefold() not in said
            ]
            append_event(
                event_path,
                "person_rule_injected",
                round=int(prompt_timeline_round.get("round") or 0),
                stage=stage,
                rule_ids=[str(item.get("patch_id") or "") for item in person_rows],
                directives=[str(item.get("directive") or "") for item in person_rows],
                because="a person filed this sentence for this stage, so this stage is drawn under it",
            )
        prompt_patch_stages[stage] = {
            **selection["report"],
            "patches": selection["patches"],
            # Which version of this stage's prompt the run resolved: ``base`` for the shipped
            # wording, and otherwise the exact patch versions riding on it. The record says both,
            # so "the prompt changed" and "the prompt changed to what" are one reading.
            **prompt_revision_record(stage, selection["patches"]),
            # The generation this stage was assembled under, so "this prompt is stale" and "this
            # prompt is current" are both readable from the record.
            "generation": prompt_patch_generation,
        }
        selected = [item for item in selection["patches"] if isinstance(item, dict)]
        patch_ids = [str(item.get("patch_id") or "") for item in selected]
        round_injection_ids.update(item for item in patch_ids if item)
        note_stage_prompt(
            stage,
            revision=str(prompt_patch_stages[stage].get("revision") or ""),
            patch_ids=patch_ids,
            cached=False,
        )
        if selected:
            append_event(
                event_path,
                "prompt_patch_injected",
                round=int(prompt_timeline_round.get("round") or 0),
                stage=stage,
                patch_ids=patch_ids,
                classes=[str(item.get("class") or "") for item in selected],
                statuses=[str(item.get("status") or "") for item in selected],
                directives=[str(item.get("directive") or "") for item in selected],
                # A patch this iteration permitted itself is injected into the very next stage
                # prompt that is assembled, rather than waiting for the next iteration.
                same_iteration_injection=any(item in own_case_patch_ids for item in patch_ids),
            )
        else:
            append_event(
                event_path,
                "prompt_patch_missing",
                round=int(prompt_timeline_round.get("round") or 0),
                stage=stage,
                considered=int(selection["report"].get("considered") or 0),
                reason=str(selection["report"].get("reason") or ""),
            )
        return selection["patches"]

    progress = build_progress_reporter(root)
    progress(
        {
            "stage": "preflight",
            "status": "completed",
            "message": "configuration inspected",
            "env_file_exists": (Path.cwd() / ".env").exists(),
            "api_base_injected": bool(os.getenv("API_BASE", "").strip()),
            "api_key_injected": bool(
                (os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
            ),
            "icon_mode": icon_mode,
        }
    )
    progress(
        {"stage": "paper_preparation", "status": "started", "message": str(Path(paper))}
    )
    # The middle layer read this paper, so a run pointed at one folds that reading
    # into the one it makes for itself instead of paying for a second opinion about
    # the same paper. A source that cannot be read is reported here and the contract
    # is planned in this run, as it was before.
    upstream = load_upstream_contract(upstream_from)
    upstream_used = upstream["status"] == "loaded"
    upstream_report = {
        "summary": "The middle layer's frozen contract, when this run was pointed at one.",
        "status": upstream["status"],
        "used": upstream_used,
        "source": upstream["source"],
        "declined_reason": upstream["declined_reason"],
        "specification_source": "middle_layer_merged" if upstream_used else "planned_by_this_run",
        # Only what the middleware's contract still stands in for whole is a call this
        # run skipped. The review and the plan are not: this run reads the paper and
        # plans the figure itself and folds the middleware's reading into what it read,
        # and the merge reports say what each fold carried.
        "provider_calls_avoided": {},
        "paper_identity": (upstream["paper_review"].get("paper_identity") or {}) if upstream_used else {},
        # Filled once the contract is really adopted: a contract that is loaded here and
        # declined while the paper is prepared must leave no copy behind.
        "scientific_ground_truth": {"adopted": False, "file": "", "sha256": "", "source": ""},
    }
    write_json(root / "upstream_contract_report.json", upstream_report)
    if upstream["status"] == "declined":
        progress({"stage": "upstream_contract", "status": "declined", "message": upstream["declined_reason"]})
    elif upstream_used:
        progress({"stage": "upstream_contract", "status": "reused", "message": upstream["source"]})
    preparation = dict(
        paper=paper,
        out=root,
        planner_mode=planner_mode,
        planner_model=None,
        ocr_engine="off",
        ocr_lang="en_ch",
        preferences_path=None,
        positive_references=[str(reference)] if reference else [],
        negative_references=[],
        aspect_ratio=aspect_ratio,
        language="English",
        domain_profile="auto",
        read_mode=read_mode,
    )
    try:
        prepared = prepare_paper_figure_contract(
            **preparation,
            frozen_contract=upstream if upstream_used else None,
            frozen_reground_min_ratio=frozen_reground_min_ratio,
        )
    except FrozenContractUnusable as error:
        # The quotes do not land on this parse, so the figure would not be the
        # middleware's contract and could not be paired. Stop before any paid
        # planning instead of drawing a contract this run invents.
        upstream_report.update({
            "status": "declined",
            "used": False,
            "declined_reason": str(error),
            "specification_source": "declined",
            "provider_calls_avoided": {"review": 1, "planner": 1, "structure": 1, "icons": 1},
        })
        upstream_used = False
        write_json(root / "upstream_contract_report.json", upstream_report)
        progress({"stage": "upstream_contract", "status": "declined", "message": str(error)})
        raise
    # A run that adopted the middle layer's contract files one verbatim copy of that
    # contract's own ground truth beside its own derivative view. The pairing builder reads
    # the copy: this run's ``scientific_ground_truth.json`` is compiled from its figure
    # content plan and is therefore never byte-identical to the middle layer's file, so
    # comparing the two could only ever refuse an honest run.
    adopted_ground_truth = (
        adopt_upstream_ground_truth(root, upstream.get("ground_truth_path"))
        if upstream_used
        else {"adopted": False, "file": "", "sha256": "", "source": ""}
    )
    upstream_report["scientific_ground_truth"] = adopted_ground_truth
    write_json(root / "upstream_contract_report.json", upstream_report)
    if not prepared.get("ok"):
        reason = prepared.get('errors') or (prepared.get('planning_validation') or {}).get('errors') or [prepared.get('status')]
        raise ValueError(f"Paper preparation failed: {reason}")
    # A plan that names the paper's modules but not the sentence behind each one
    # is thin, not broken: the figure is drawn and the run says which objects
    # went unsourced.
    paper_contract_gaps = [str(item) for item in ((prepared.get("planning_validation") or {}).get("grounding_gaps") or [])]
    progress({"stage": "paper_preparation", "status": "completed", "grounding_gap_count": len(paper_contract_gaps)})
    # Every vision call below has to be handed pixels. The frozen snapshot keeps
    # a paper own framework figure as a PDF, and a provider answers a PDF with
    # "unable to process input image" on every attempt, so the reference is
    # rendered once into this run and a reference that cannot be read is dropped
    # here: the run then degrades and says so instead of failing on the API.
    model_reference: str | Path | None = reference
    if reference:
        reference_image, reference_image_report = reference_image_path(reference, work_dir=root / "inputs")
        write_json(root / "reference_image_report.json", reference_image_report)
        model_reference = reference_image
        if reference_image is None:
            progress({"stage": "reference_image", "status": "failed", "message": str(reference_image_report.get("error") or reference_image_report.get("status") or "reference asset could not be read")})
    reuse_source = replay_upstream_from or reuse_upstream_from
    prestaged_icons = _prestaged_icon_root(reuse_upstream_from)
    replay_declined_reason = ""
    stage_replay = None
    if reuse_source:
        try:
            stage_replay = load_stage_replay(
                reuse_source,
                paper=paper,
                reference=reference,
                freeze_layout=bool(replay_upstream_from),
                fork_before=str(fork_before or ""),
                # A controlled experiment needs both arms to start from one identical state, so a
                # stage it would have to re-plan is refused. The continuous loop, which asked for
                # whatever the last version can lend, gets the per-stage answer instead.
                strict=bool(replay_upstream_from),
            )
        except ValueError as exc:
            if replay_upstream_from:
                raise
            replay_declined_reason = str(exc)
    stage_replay_report = dict((stage_replay or {}).get("report") or {})
    if reuse_source:
        stage_replay_report.setdefault("source", str(reuse_source))
        stage_replay_report.setdefault("frozen_icon_asset_ids", [])
        # Record what was asked for, not only what was answered: a replay that declines leaves
        # no other trace of the fork stage, and "we tried to inherit from here" is the fact an
        # audit of the continuous loop needs.
        stage_replay_report.setdefault("fork_before", str(fork_before or ""))
        stage_replay_report.setdefault("reuse_requested", True)
    if replay_declined_reason:
        stage_replay_report["declined_reason"] = replay_declined_reason
        progress({"stage": "upstream_reuse", "status": "declined", "message": replay_declined_reason})
    elif stage_replay:
        progress({
            "stage": "upstream_reuse",
            "status": "reused",
            "message": str(reuse_source),
            "reused_stages": list(stage_replay_report.get("reused_stages") or []),
            "replanned_stages": list(stage_replay_report.get("replanned_stages") or []),
        })
    write_json(root / "stage_reuse_report.json", stage_replay_report)
    specification = prepared["plan"].get("figure_specification", {})
    # Which figure the design is aimed at. ``overview`` is the product's own reading
    # of the method and is byte for byte today's path. ``paper`` follows the panels
    # the paper printed, which is what a public benchmark scores; the target is read
    # once here and reaches the extraction and the structure planner, and the
    # delivered figure is measured against it at the end.
    figure_target = "paper" if str(figure_target).strip().casefold() == "paper" else "overview"
    panel_target = (
        panel_target_from_specification(specification)
        if figure_target == "paper"
        else {"available": False, "reason": "figure_target_overview", "source": "", "panel_count": 0, "label_count": 0, "panels": []}
    )
    adopted_method, adopted_validation = _upstream_method_reuse(prepared)
    progress(
        {
            "stage": "method_extraction",
            "status": "started",
            "message": method_mode or planner_mode,
        }
    )
    method_merge_report: dict[str, Any] = {}
    if stage_replay:
        method = stage_replay["method"]
        method_validation = stage_replay["method_validation"]
    else:
        method, method_validation = extract_method_figure_content(
            specification,
            paper_review=prepared.get("paper_review"),
            mode=method_mode or planner_mode,
            provenance={"paper": str(Path(paper).resolve()), "planner_mode": planner_mode},
            feedback=reading_feedback(),
            figure_target=figure_target,
            panel_target=panel_target,
        )
        if adopted_method:
            # The middle layer read the same paper at a depth this run's own pass does not
            # reach, but what it wrote is aimed at the specification it built for itself and
            # at no panel target, and the ids it names belong to the inventory it read rather
            # than to the contract this run merged. Taking it whole would hand the figure the
            # middleware's reading of the paper; folding it in keeps the run's aim and adds
            # the objects the run would otherwise have left out.
            method, method_merge_report = merge_upstream_content_plan(
                method,
                adopted_method,
                source=str(adopted_validation.get("source") or ""),
            )
            write_json(root / "upstream_content_plan_merge.json", method_merge_report)
    progress(
        {
            "stage": "method_extraction",
            "status": "completed",
            "validation_ok": method_validation.get("ok"),
            "reused": bool(stage_replay),
            "source": (
                "middle_layer_merged" if method_merge_report
                else "middle_layer" if adopted_method
                else "this_run"
            ),
        }
    )
    write_json(root / FIGURE_CONTENT_PLAN_FILE, method)
    write_json(root / "method_extraction_report.json", method_validation)
    if stage_replay and stage_replay.get("relation_explanation_report"):
        relation_explanation_report = stage_replay["relation_explanation_report"]
    else:
        method, relation_explanation_report = plan_relation_explanations_vlm(
            method, model=structure_model, timeout=vlm_timeout, retries=vlm_retries
        )
    write_json(root / "relation_explanation_report.json", relation_explanation_report)
    write_json(root / FIGURE_CONTENT_PLAN_FILE, method)
    if upstream_used:
        grounding = prepared.get("frozen_contract") if isinstance(prepared.get("frozen_contract"), dict) else {}
        upstream_report.update({
            "method_status": upstream["method_status"],
            "method_source": (
                "middle_layer_merged" if method_merge_report
                else "middle_layer" if adopted_method
                else "planned_by_this_run"
            ),
            "method_evidence_referenced": int((grounding or {}).get("method_referenced_evidence_ids") or 0),
            "method_evidence_unresolved": int((grounding or {}).get("method_unresolved_evidence_count") or 0),
        })
        if (grounding or {}).get("method_dropped_reason"):
            upstream_report["method_declined_reason"] = str(grounding["method_dropped_reason"])
        if adopted_method and not method_merge_report:
            # Only a plan taken whole skips the extraction this run would otherwise have paid
            # for. A merged plan did pay for it; the merge report is what says so.
            upstream_report["provider_calls_avoided"]["method_contract"] = 1
        if method_merge_report:
            upstream_report["method_merge"] = method_merge_report
        if adopted_method and not int(relation_explanation_report.get("eligible_count") or 0):
            # Every relation of the adopted contract already carries its label,
            # so the explanation step had nothing left to ask the provider for.
            upstream_report["provider_calls_avoided"]["relation_explanations"] = 1
        write_json(root / "upstream_contract_report.json", upstream_report)
    paper_review = prepared.get("paper_review") or {}
    # See _resolve_paper_id: the caller's own identifier is read first, so a paper the
    # middle layer indexed under no identifier is still named by the run that started it.
    paper_id = _resolve_paper_id(paper_id, prepared, paper_review, paper)
    if judge_holdout_paths:
        from ..judge_training.holdout_papers import assert_paper_not_held_out

        assert_paper_not_held_out(paper_id, judge_holdout_paths)
    resolved_domain = resolve_domain_profile(domain_profile, prepared.get("paper_review") or {})
    creator_rules_manifest = rules_for_paper(creator_rules_manifest, resolved_domain)
    evidence_map = read_json(root / "evidence_map.json") if (root / "evidence_map.json").exists() else None
    scientific_graph: dict[str, Any] = {}
    scientific_ground_truth: dict[str, Any] = {}
    scientific_graph_report: dict[str, Any] = {}
    if evidence_structure:
        scientific_graph, scientific_ground_truth, scientific_graph_report = _build_scientific_layers(
            method, paper_id=paper_id, evidence_map=evidence_map, root=root
        )
    library = Path(reference_library) if reference_library else (Path.cwd() / "output" / "reference_learning_color_v1" / "reference_library.jsonl")
    # The configured path travels in beside the readable one, so the resolution record can say
    # whether this run had no library configured at all or was pointed at one that holds
    # nothing usable: the acceptance probe reads a different field for each.
    exemplar_selection = (
        resolve_project_exemplars(
            root=root,
            paper_review=prepared.get("paper_review") or {},
            selected_template={},
            reference_library=library if exemplar_guidance and library.exists() else None,
            top_k=reference_top_k,
            configured_library=library if reference_library else None,
        )
        if exemplar_guidance
        else None
    )
    exemplar_brief = exemplar_selection.get("brief") if exemplar_selection else {}
    exemplar_brief = _with_creator_rules(exemplar_brief, creator_rules_manifest)
    write_json(root / "exemplar_selection.json", exemplar_selection or {})
    write_json(root / "exemplar_style_brief.json", exemplar_brief)
    # The one good figure this paper's own output will be read against, chosen once for the whole
    # run rather than per round: the comparison is about how our figure differs from a figure of
    # this kind, and re-picking the reference every round would move the ruler the gap is stated
    # in. The paper's own figures are kept out of the pool, because a paper compared with itself
    # has no gap to name.
    gap_reference = (
        select_gap_reference(
            prepared.get("paper_review") or {},
            library,
            exclude_tokens=[str(paper_id or ""), str(paper_id or "").replace("/", "_")],
        )
        if exemplar_guidance and library.exists()
        else {"ok": False, "reason": "no reference library was configured for this run", "selected": {}}
    )
    write_json(root / "reference_gap_reference.json", gap_reference)
    progress(
        {"stage": "reference_analysis", "status": "started", "message": reference_mode}
    )
    # An earlier iteration only lends its reference analysis when it actually has one: a source
    # whose analysis is incomplete is a recorded gap in its reuse report, and the analysis is
    # made again here rather than refusing the whole source.
    replayed_profile = (stage_replay or {}).get("reference_profile")
    if replayed_profile:
        reference_profile = replayed_profile
    elif reference_mode != "off":
        reference_profile = analyze_reference_image(reference, method=specification, mode=reference_mode, work_dir=root / "inputs")
    else:
        reference_profile = {
            "status": "disabled",
            "influence": {"adopted": [], "rejected": ["reference_analysis_disabled"]},
        }
    reference_profile["exemplar_color_rules"] = exemplar_brief.get("color_rules") or []
    progress(
        {
            "stage": "reference_analysis",
            "status": "completed",
            "analysis_status": reference_profile.get("status"),
        }
    )
    write_json(
        root / "reference_style_profile.json",
        {
            "status": reference_profile.get("status"),
            **(reference_profile.get("style") or {}),
        },
    )
    write_json(
        root / "reference_layout_profile.json",
        {
            "status": reference_profile.get("status"),
            **(reference_profile.get("layout") or {}),
        },
    )
    write_json(
        root / "reference_influence_report.json",
        reference_profile.get("influence") or {},
    )
    write_json(root / "reference_layout_template.json", reference_profile.get("reference_layout_template") or {})
    write_json(root / "reference_layout_template_validation.json", reference_profile.get("reference_layout_template_validation") or {})
    scientific_hierarchy: dict[str, Any] = {}
    hierarchy_report: dict[str, Any] = {}
    if evidence_structure:
        progress({"stage": "scientific_hierarchy_planning", "status": "started"})
        scientific_hierarchy, hierarchy_report = (
            (stage_replay.get("scientific_hierarchy") or {}, stage_replay.get("hierarchy_validation") or {})
            if stage_replay
            else plan_scientific_hierarchy(
                method,
                exemplar_style_brief=exemplar_brief,
                model=structure_model,
                timeout=vlm_timeout,
                retries=vlm_retries,
            )
            if structure_mode == "vlm"
            else ({}, {"mode": "heuristic", "fallback_used": False})
        )
        write_json(root / "scientific_hierarchy_plan.json", scientific_hierarchy)
        write_json(root / "scientific_hierarchy_validation.json", hierarchy_report)
        progress({"stage": "scientific_hierarchy_planning", "status": "completed", "section_count": len(scientific_hierarchy.get("sections", [])), "submodule_count": len(scientific_hierarchy.get("submodules", [])), "fallback_used": hierarchy_report.get("fallback_used"), "repaired": hierarchy_report.get("repaired"), "repair_rounds": hierarchy_report.get("repair_rounds"), "continued_with_invalid_plan": hierarchy_report.get("continued_with_invalid_plan")})
    else:
        # The hierarchy is the paper's candidate inventory for macro regions and
        # nested nodes. A row that never built one plans without it and writes
        # nothing: an absent artifact is the truth about this row, and a stub one
        # would read as a hierarchy that was planned and came back empty.
        progress({"stage": "scientific_hierarchy_planning", "status": "skipped", "reason": "middle_layer_off"})
    progress({"stage": "structure_planning", "status": "started"})
    design, design_report = (
        plan_structure_vlm(
            method,
            reference_profile,
            model_reference,
            model=structure_model,
            timeout=vlm_timeout,
            retries=vlm_retries,
            exemplar_style_brief=exemplar_brief,
            wireframe_out=root / "structure_wireframe.png",
            scientific_hierarchy=scientific_hierarchy,
            aspect_ratio=aspect_ratio,
            initial_plan=(stage_replay or {}).get("design_seed"),
            frozen_layout=bool((stage_replay or {}).get("design_seed")),
            bulk_figures=bulk_figures,
            prompt_patches=stage_patches("global_planning"),
            graph_context=graph_context,
            evidence_structure=evidence_structure,
            figure_target=figure_target,
            panel_target=panel_target,
        )
        if structure_mode == "vlm"
        else ({}, {"mode": "heuristic", "fallback_used": False})
    )
    if not evidence_structure:
        # Built now, once the figure exists and can no longer be influenced by it:
        # this row's planning never saw the paper's explicit structure. It lands in
        # the same place and is measured the same way, so the Grounding Gate and
        # the alignment reports are exactly the ones the full configuration
        # computes, and the repair loop is not a second difference between the rows.
        scientific_graph, scientific_ground_truth, scientific_graph_report = _build_scientific_layers(
            method,
            paper_id=paper_id,
            evidence_map=evidence_map,
            root=root,
            provenance="post_hoc_measurement",
        )
    # The design stage is what decides which paper relations must be visible, so
    # its chosen connectors become the required-edge list the Grounding Gate audits.
    design_required_edges = (
        resolve_design_required_edge_ids(design, scientific_graph)
        if isinstance(design, dict) and isinstance(design.get("connectors"), list)
        else {"required_edge_ids": [], "matched_connector_ids": [], "unmatched_connector_ids": [], "match_sources": {}}
    )
    design_required_edge_ids = list(design_required_edges.get("required_edge_ids") or [])
    if design_required_edge_ids:
        scientific_graph["required_edge_ids"] = sorted(
            {*scientific_graph.get("required_edge_ids", []), *design_required_edge_ids}
        )
        scientific_ground_truth["required_edge_ids"] = list(scientific_graph["required_edge_ids"])
        scientific_graph_report["required_edge_ids_source"] = "design_plan"
        scientific_graph_report["design_plan_required_edge_ids"] = design_required_edge_ids
        scientific_graph_report["design_plan_matched_connector_ids"] = design_required_edges.get("matched_connector_ids") or []
        scientific_graph_report["design_plan_unmatched_connector_ids"] = design_required_edges.get("unmatched_connector_ids") or []
        write_json(root / "scientific_graph.json", scientific_graph)
        write_json(root / "scientific_ground_truth.json", scientific_ground_truth)
        write_json(root / "scientific_graph_report.json", scientific_graph_report)
    adaptation_report = design_report.get("panel_adaptation") or {}
    if adaptation_report:
        write_json(root / "panel_adaptation_report.json", adaptation_report)
    structure = (
        structure_from_design_plan(
            design,
            aspect_ratio=aspect_ratio,
            reference=str(reference) if reference else None,
            reference_profile=reference_profile,
            exemplar_brief=exemplar_brief,
        )
        if design and design.get("nodes")
        else plan_paper_figure_structure(
            specification,
            aspect_ratio=aspect_ratio,
            reference=str(reference) if reference else None,
            reference_profile=reference_profile,
            exemplar_brief=exemplar_brief,
        )
    )
    write_json(root / "design_plan.json", design or structure["figure_layout_plan"])
    write_json(root / "design_plan_validation.json", design_report)
    write_json(root / "design_plan_raw.json", design_report.get("raw_response") or {})
    # Written only when there is a response behind it: a row that planned no design carries no
    # seed, and an empty file would read as "this run built its figure from nothing" rather than
    # as a stage the source never planned.
    seed_response = design_report.get("seed_response")
    if seed_response:
        write_json(root / DESIGN_PLAN_SEED_FILE, seed_response)
    write_json(root / "canvas_safe_area_report.json", design_report.get("canvas_safe_area") or {})
    # Recompute density after content-driven panel adaptation; the VLM report
    # describes pre-adaptation geometry and can otherwise hide oversized,
    # low-information panels that were just compacted.
    stage4_manifest = (exemplar_brief or {}).get("stage4_manifest") if isinstance(exemplar_brief, dict) else None
    seed_density = density_targets_from((seed_experience or {}).get("action"))
    density_rules = dict(stage4_manifest or {})
    if seed_density:
        density_rules["density_targets"] = {**(density_rules.get("density_targets") or {}), **seed_density}
    structure_density = build_density_advisory(design or structure["figure_layout_plan"], density_rules)
    if adaptation_report:
        structure_density["panel_adaptation"] = adaptation_report
    write_json(root / "structure_density_report.json", structure_density)
    write_json(root / "structure_replan_attempts.json", design_report.get("replan_attempts") or [])
    progress(
        {
            "stage": "structure_planning",
            "status": "completed",
            "node_count": len(structure["figure_layout_plan"].get("nodes", [])),
            "icon_count": len(structure["icon_generation_plan"].get("icons", [])),
            "plan_repair_used": bool(design_report.get("plan_repair_used")),
            "continued_with_invalid_plan": bool(design_report.get("continued_with_invalid_plan")),
            "bulk_figures": bool(bulk_figures),
            "plan_repair_skipped": bool(design_report.get("plan_repair_skipped")),
            "density_replan_skipped": bool(design_report.get("density_replan_skipped")),
        }
    )
    for name in (
        "global_structure_plan",
        "segmentation_plan",
        "figure_layout_plan",
        "icon_generation_plan",
    ):
        write_json(root / f"{name}.json", structure[name])
    write_json(root / "panel_style_plan.json", structure.get("panel_style_plan") or {})
    text_plan, text_report = (
        (stage_replay["figure_text_plan"], stage_replay.get("figure_text_plan_validation") or {})
        if stage_replay and stage_replay.get("figure_text_plan")
        else plan_figure_text_vlm(
            method,
            design or structure["figure_layout_plan"],
            model=text_plan_model,
            timeout=vlm_timeout,
            retries=vlm_retries,
            exemplar_style_brief=exemplar_brief,
        )
        if text_plan_mode == "vlm"
        else ({"items": []}, {"mode": "heuristic", "fallback_used": False})
    )
    write_json(root / "figure_text_plan.json", text_plan)
    write_json(root / "figure_text_plan_validation.json", text_report)
    write_json(root / "figure_text_plan_raw.json", text_report.get("raw_response") or {})
    progress({"stage": "color_theme_planning", "status": "started", "message": color_theme_mode})
    seed_theme = replayable_theme(((seed_experience or {}).get("action") or {}).get("color_theme"))
    if seed_theme:
        color_theme = seed_theme
        color_theme["provenance"] = {"source": "memory", "fallback_used": False, "reference_style_used": False, "exemplar_rules_used": bool(exemplar_brief)}
        color_theme_report = {"ok": True, "mode": "memory", "fallback_used": False}
    elif stage_replay and stage_replay.get("color_theme"):
        color_theme = dict(stage_replay["color_theme"])
        color_theme_report = dict(stage_replay.get("color_theme_validation") or {})
        color_theme["provenance"] = {"source": "replay", "fallback_used": False, "reference_style_used": True, "exemplar_rules_used": bool(exemplar_brief), "replayed_from": str(stage_replay["root"])}
    else:
        color_theme, color_theme_report = plan_color_theme_vlm(
            method,
            design or structure["figure_layout_plan"],
            text_plan,
            reference_profile,
            exemplar_brief,
            mode=color_theme_mode,
            model=color_theme_model,
            timeout=vlm_timeout,
            retries=vlm_retries,
            prompt_patches=stage_patches("theme_planning"),
        )
        if color_theme_mode != "vlm":
            color_theme["provenance"] = {"source": "safe_fallback", "fallback_used": True}
        else:
            color_theme["provenance"] = {"source": "vlm" if not color_theme_report.get("fallback_used") else "safe_fallback", "fallback_used": bool(color_theme_report.get("fallback_used")), "reference_style_used": True, "exemplar_rules_used": bool(exemplar_brief)}
    if color_theme_mode == "vlm" and not color_theme_report.get("ok"):
        # The plan was already handed back to the model three times. A missing
        # role is not a reason to lose the paper, so the deterministic palette
        # takes over and the report says the model answer was rejected.
        rejected_reason = str(color_theme_report.get("failure_reason") or "; ".join(str(item) for item in (color_theme_report.get("errors") or [])) or "invalid color theme")
        fallback_theme, fallback_report = plan_color_theme_vlm(
            method,
            design or structure["figure_layout_plan"],
            text_plan,
            reference_profile,
            exemplar_brief,
            mode="fallback",
            model=color_theme_model,
            timeout=vlm_timeout,
            retries=vlm_retries,
        )
        fallback_theme["provenance"] = {"source": "safe_fallback", "fallback_used": True, "reference_style_used": True, "exemplar_rules_used": bool(exemplar_brief), "rejected_vlm_plan": rejected_reason}
        color_theme = fallback_theme
        color_theme_report = {**fallback_report, "mode": "vlm_then_safe_fallback", "fallback_used": True, "failure_reason": rejected_reason, "rejected_theme": color_theme_report.get("theme") or {}, "attempts": color_theme_report.get("attempts") or []}
        write_json(root / "color_theme_validation.json", color_theme_report)
        write_json(root / "color_theme_raw.json", color_theme_report.get("raw_response") or {})
    if design and design.get("nodes"):
        structure = structure_from_design_plan(design, aspect_ratio=aspect_ratio, reference=str(reference) if reference else None, reference_profile=reference_profile, exemplar_brief=exemplar_brief, color_theme=color_theme)
        for name in ("global_structure_plan", "segmentation_plan", "figure_layout_plan", "icon_generation_plan"):
            write_json(root / f"{name}.json", structure[name])
        write_json(root / "panel_style_plan.json", structure.get("panel_style_plan") or {})
    write_json(root / "color_theme.json", color_theme)
    style_plan = dict(structure.get("panel_style_plan") or {})
    style_plan["color_theme"] = color_theme
    structure["panel_style_plan"] = style_plan
    write_json(root / "panel_style_plan.json", style_plan)
    write_json(root / "color_theme_validation.json", color_theme_report)
    write_json(root / "color_theme_raw.json", color_theme_report.get("raw_response") or {})
    write_json(
        root / "color_experience_usage_report.json",
        evaluate_color_experience_usage(
            color_theme,
            supplied_recipe_ids=color_theme_report.get("supplied_recipe_ids") or [],
        ),
    )
    if color_theme_mode == "vlm":
        progress(
            {
                "stage": "color_theme_planning",
                "status": "completed" if color_theme_report.get("ok") else "failed",
                "fallback_used": bool(color_theme_report.get("fallback_used")),
                "message": str(color_theme_report.get("failure_reason") or "")[:200],
            }
        )
    if not color_theme_report.get("ok"):
        raise RuntimeError("colour theme planning failed and the deterministic palette did not validate either: " + str(color_theme_report.get("failure_reason") or "invalid color theme"))
    title = (prepared.get("paper_review") or {}).get("paper_identity", {}).get("title")
    rounds = []
    best_score = -1.0
    # Two answers, two variables. ``best_acceptance`` is the measurement of the last accepted
    # state, which is what the next round must be compared against; ``best_judgement`` is the
    # advisory answer that guided it, which is what the next action reads. Subtracting one from
    # the other is exactly the mistake the split exists to prevent.
    best_acceptance = None
    best_judgement = None
    # A run has no baseline until a round was accepted and measured; until then every round is
    # refused as ``baseline_unavailable`` rather than read against a placeholder.
    baseline_ready = False
    # The accepted state's own question input, kept so the next round can show it beside the
    # candidate it produced instead of describing it from a stored number.
    best_acceptance_input: DirectJudgeInput | None = None
    best_grounding: dict[str, Any] | None = None
    best_layer_digests: dict[str, str] | None = None
    best_compiled = None
    best_dir: Path | None = None
    best_visual_review = None
    best_visual_review_validation = None
    best_color_theme = color_theme
    best_color_theme_report = color_theme_report
    best_panel_style_plan = structure.get("panel_style_plan") or {}
    # Every accepted round is snapshotted so the run can deliver the highest-scoring one:
    # the loop itself keeps chaining from the last accepted state, but the figure handed
    # back is chosen by score, because a later round can be accepted and still read lower.
    accepted_snapshots: list[dict[str, Any]] = []
    # The colour problem the last accepted round measured, kept so the next round's theme
    # re-plan and the advisory role can see it. It is context for a question, never a target.
    pending_color_feedback: dict[str, Any] = {}
    baseline_color_metrics: dict[str, Any] = {}
    current_layout = structure["figure_layout_plan"]
    current_icon_plan = structure["icon_generation_plan"]
    icons_inherited = (stage_replay_report.get("stage_outcomes") or {}).get("icons", {}).get("status") == "reused"
    if stage_replay and icons_inherited:
        frozen_ids = set(stage_replay_report.get("frozen_icon_asset_ids") or [])
        required_ids = {
            str(item.get("asset_id"))
            for item in current_icon_plan.get("icons", [])
            if isinstance(item, dict) and item.get("asset_id")
        }
        missing_frozen_ids = sorted(required_ids - frozen_ids)
        stage_replay_report.update({
            "required_icon_asset_ids": sorted(required_ids),
            "missing_frozen_icon_asset_ids": missing_frozen_ids,
            "icon_inventory_ok": not missing_frozen_ids,
        })
        write_json(root / "stage_reuse_report.json", stage_replay_report)
        if missing_frozen_ids:
            if replay_upstream_from:
                raise RuntimeError("stage replay is missing frozen Icon assets: " + ", ".join(missing_frozen_ids))
            # A retry plans the boxes again, so an Icon the earlier attempt
            # cannot supply is simply drawn rather than ending the attempt.
            progress(
                {
                    "stage": "icon_generation",
                    "status": "partial_reuse",
                    "message": f"{len(missing_frozen_ids)} of {len(required_ids)} Icons are not in the earlier attempt",
                }
            )
    current_action: dict[str, Any] = {"type": "initial", "scope_type": "global"}
    # An iteration that inherited a finished figure opens on the change the review of that
    # figure asked for, rather than on the same planning the earlier iteration already paid
    # for. The reviewed figure is the state this round acts on, so the change is a real
    # repair and not a redraw; a run that planned its own design or icons, or one that was
    # handed no review at all, keeps the ``initial`` first round it always had.
    opening_action = _opening_repair(first_round_action, stage_replay, stage_replay_report)
    if opening_action:
        current_action = opening_action
    # The moves the next round should carry, and the move a guard deferred out of the round
    # that just ran. A deferred move is pinned as the next round's first move, so a suggestion
    # that was already decided on is run rather than lost.
    pending_round_moves: list[dict[str, Any]] = []
    pending_deferred_move: dict[str, Any] = {}
    pending_seed_action = dict((seed_experience or {}).get("action") or {})
    if pending_seed_action:
        pending_seed_action["experience_id"] = str((seed_experience or {}).get("experience_id") or "")
    action_retry_index = 0
    skipped_action_types: set[str] = set()
    # A rule a reflection stated is tried once per iteration: the identity of a rule is its
    # normalized parameters, so two spellings of one rule are one turn. Cleared wherever the
    # iteration ends, which is the same place the skipped families and the retry count reset.
    tried_rule_fingerprints: set[str] = set()
    # The attempts a losing rule has already spent, keyed by the rule rather than by the wording of
    # its repair: a reflection that rephrases itself every round must still run out of attempts. The
    # rows those rounds would have written off wait beside them, because one reading is not a
    # refutation while the rule still has an attempt to run.
    tried_follow_ups: dict[str, int] = {}
    pending_repair_rows: dict[str, dict[str, Any]] = {}
    write_json(
        root / "structure_candidates.json",
        {"candidates": structure.get("structure_candidates", [])},
    )
    # A bound checkpoint supplies the served model name when the caller did not give one:
    # the registry entry is what says which Judge this run meant, and repeating its id on
    # the command line only creates a second place for the two to disagree.
    bound_model = str(judge_binding.get("model_id") or "")
    online_judge, frozen_judge = build_direct_judges(
        mode="bootstrap" if judge_mode == "vlm" else judge_mode,
        online_model=online_judge_model or judge_model or bound_model or None,
        frozen_model=frozen_judge_model or judge_model or bound_model or None,
        timeout=vlm_timeout,
        retries=vlm_retries,
        acceptance_checkpoint=acceptance_checkpoint,
        acceptance_base_model=acceptance_base_model,
        acceptance_generate=acceptance_generate,
        advisory_checkpoint=online_judge_checkpoint,
        advisory_base_model=online_judge_base_model,
        advisory_ask=online_ask_json,
        acceptance_direction=acceptance_direction,
    )
    injected_experience: dict[str, Any] = {}
    retrieved: list[dict[str, Any]] = []
    retrieval_report: dict[str, Any] = {}
    routing_report: dict[str, Any] = {}
    retrieval_reports: list[dict[str, Any]] = []
    routing_reports: list[dict[str, Any]] = []
    decomposability_history: list[dict[str, Any]] = []
    decomposition_action: dict[str, Any] = {}
    reflection_policy: dict[str, Any] = {}
    # Every stage resolves its patches once, up front, so the manifest describes the whole
    # run and the prohibition check reads the same set of patches the prompts carried.
    for _patch_stage in ("global_planning", "segmentation", "theme_planning", "icon_generation", "ppt_compilation", "repair"):
        stage_patches(_patch_stage)
    prohibited_patches = [
        patch
        for _stage in sorted(prompt_patch_stages)
        for patch in prompt_patch_stages[_stage]["patches"]
        if str(patch.get("class")) == "prohibited"
    ]
    # A rule guides a stage for the whole run, so the round record can name the ones it ran
    # under, and a rollback can be attributed to the guidance rather than to nothing.
    applied_rules_by_stage = {str(stage): list(ids) for stage, ids in (creator_rules_manifest.get("buckets") or {}).items()}
    applied_rule_ids = sorted({str(rule_id) for ids in applied_rules_by_stage.values() for rule_id in ids})
    rule_sources = {
        str(rule.get("rule_id") or ""): {
            "class": str(rule.get("class") or ""),
            "apply_at": str(rule.get("apply_at") or ""),
            "experiences": [str(item) for item in rule.get("source_experience_ids") or []],
            "patches": [str(item) for item in rule.get("source_patch_ids") or []],
        }
        for rule in creator_rules_manifest.get("rules") or []
    }
    segmentation_parameters = patch_parameters(stage_patches("segmentation"))
    decomposability_limits = effective_thresholds(
        {
            "min_panel_occupancy": DECOMPOSABILITY_MIN_PANEL_OCCUPANCY,
            "max_cross_panel_connectors": DECOMPOSABILITY_MAX_CROSS_PANEL_CONNECTORS,
            "max_node_slot_fanout": DECOMPOSABILITY_MAX_NODE_SLOT_FANOUT,
            "min_panel_area": DECOMPOSABILITY_MIN_PANEL_AREA,
        },
        segmentation_parameters,
    )
    round_limit = max(1, min(12, int(max_rounds))) if evolution_enabled else 1
    # What each layer of a repair costs, and which of the round's voices asked for the expensive
    # one. The seat is given the price list and decides; the executor refuses nothing for the layer
    # a repair works at, so this is accounting rather than a gate, and the only thing that stops a
    # move is the budget and the clock the round is already guarded by between its moves.
    coach_fault_stage = ""
    stalled_rounds = 0
    measured_escalation = False
    content_gap = ""

    def _layer_costs_for_prompt() -> dict[str, Any]:
        """What each layer of a repair costs, plus the reading the loop took of this figure.

        The seat used to be handed the layers it was not allowed to ask for, so that it would not
        spend its answer on a repair the executor would refuse. Nothing refuses a repair for the
        layer it works at any more: whether a stage re-run is worth its minutes is the seat's call,
        which is why it is given the price of each layer instead of a list of what is forbidden.
        The loop's own reading, when one fired, travels beside the prices -- a seat told only "here
        is a defect" cannot tell a figure nobody checked from one that measured as unreadable in
        pieces. A round carries no key when the table has nothing to say, so a caller that never
        asked for the prices renders exactly as it did before.
        """

        policy = action_class_policy(
            coach_fault_stage=coach_fault_stage,
            stalled_rounds=stalled_rounds,
            measured_escalation=measured_escalation,
            content_gap=content_gap,
        )
        costs = {str(name): str(price) for name, price in (policy.get("costs") or {}).items()}
        if not costs:
            return {}
        payload: dict[str, Any] = {"costs": costs}
        if policy.get("measured"):
            payload["measured"] = {
                "signal": str(policy.get("signal") or ""),
                "reason": str(policy.get("reason") or ""),
            }
        return payload
    round_costs: list[dict[str, Any]] = []
    # The state this run's first round is measured against. An iteration handed the previous
    # iteration's kept figure opens with a real comparison instead of an unmeasured starting
    # point; one that was not handed any says so rather than reading a score into nothing.
    baseline_input: DirectJudgeInput | None = None
    baseline_measurement: dict[str, Any] | None = None
    if baseline_from:
        # Imported here rather than at module level: the validation workflow reads this module.
        from .direct_experience_validation import kept_figure_judge_input
        baseline_input = kept_figure_judge_input(
            baseline_from,
            domain=str(resolved_domain.get("domain") or ""),
            allowed_scope={"scope": "frozen_acceptance", "round": -1},
        )
        if baseline_input is None:
            append_event(event_path, "baseline_unavailable", round=0, reason="the baseline run left no kept figure to show", source=str(baseline_from))
        elif _pairwise_capable(frozen_judge):
            # The baseline figure is measured with the same instrument the round is, so the
            # regression veto reads a real list of defects the earlier figure already had
            # instead of counting every defect of the new one as newly introduced.
            baseline_measurement, baseline_error = _measure_acceptance(frozen_judge, baseline_input)
            append_event(event_path, "baseline_measured", round=0, role=getattr(frozen_judge, "role", "unknown"), judgement=baseline_measurement, error=baseline_error or None)
    def _round_action(
        state: dict[str, Any],
        action: dict[str, Any],
        *,
        retry_index: int,
        move_index: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Run one move of a round and return the state it produced.

        A round may run two moves: the mechanism's policy first, and the coach's suggestion
        on top of what it left behind. Both go through this one path, so the second move is
        executed, recorded and reported exactly like the first rather than through a second
        implementation that could drift from it. "move_index" only decides which file the
        move's own upstream-retry record lands in; the state handed in is the state acted on,
        so the second move sees the first move's figure.
        """

        nonlocal method, design, structure
        nonlocal scientific_graph, scientific_ground_truth, scientific_graph_report
        nonlocal round_color_theme, round_color_theme_report, round_panel_style_plan
        retry_report_name = "upstream_retry.json" if move_index == 0 else f"upstream_retry_move_{move_index:02d}.json"
        upstream_retry_report: dict[str, Any] = {}
        if _round_runs_a_move(round_index, action):
            # §6.2: a global topology break, a scientific error, or a figure no local nudge
            # can separate is answered upstream -- the method extraction and the framework
            # structure planning are re-run under the measured break -- instead of falling
            # through to the bounded node move the three escalation actions used to share.
            # A stage that cannot be re-run is recorded with its reason; it never quietly
            # becomes a bounded action.
            retry_kind = upstream_retry_kind(action)
            if retry_kind:
                # A stage re-run is the expensive answer, and the seat is the one that decides
                # whether it is worth its minutes: the prices travelled to it, and the layer table
                # refuses nothing. What is recorded here is the account -- which layer the move
                # works at, what that layer spends, and which of the round's voices had asked for
                # it -- so a round that spent an hour re-planning reads back as one something
                # asked an hour for. The hard line against spending is the budget and the clock,
                # which the round is already guarded by between its moves.
                class_policy = action_class_policy(
                    coach_fault_stage=coach_fault_stage,
                    stalled_rounds=stalled_rounds,
                    measured_escalation=measured_escalation,
                )
                measured_move = measurement_request(action)
                move_class = action_class(action.get("type"))
                append_event(
                    event_path,
                    "action_class_allowed",
                    round=round_index,
                    action=str(action.get("type") or ""),
                    action_class=move_class,
                    stage_rerun_kind=retry_kind,
                    cost=str((class_policy.get("costs") or {}).get(move_class) or ""),
                    reason=str(class_policy.get("reason") or ""),
                    escalated=bool(class_policy.get("escalated")) or measured_move,
                    stalled_rounds=int(class_policy.get("stalled_rounds") or 0),
                    measured=measured_move or bool(class_policy.get("measured")),
                    signal=str(class_policy.get("signal") or "") or ("decomposition_policy" if measured_move else ""),
                )
            if retry_kind:
                # Each stage is re-run by the mode that owns it: the structure stage by
                # ``structure_mode``, the theme stage by ``color_theme_mode`` (a theme re-plan
                # is a provider call, so a non-vlm colour mode declines it with its reason),
                # and the semantic stage by the mode that reads the paper.
                retry_mode = (
                    structure_mode
                    if retry_kind == "structure"
                    else color_theme_mode
                    if retry_kind == "theme"
                    else (method_mode or planner_mode)
                )
                retry_design_seed = design if isinstance(design, dict) and design else None
                upstream_retry_report = run_upstream_retry(
                    action,
                    enabled=upstream_retry_enabled,
                    planner_mode=retry_mode,
                    feedback=describe_retry_feedback(decomposition_action),
                    retry_index=retry_index,
                    method=method,
                    design=design,
                    theme=round_color_theme,
                    color_feedback=pending_color_feedback,
                    structure_planner=lambda retry_method, note: plan_structure_vlm(
                        retry_method,
                        reference_profile,
                        model_reference,
                        model=structure_model,
                        timeout=vlm_timeout,
                        retries=vlm_retries,
                        exemplar_style_brief=exemplar_brief,
                        wireframe_out=round_dir / "structure_wireframe.png",
                        scientific_hierarchy=scientific_hierarchy,
                        aspect_ratio=aspect_ratio,
                        initial_plan=retry_design_seed,
                        bulk_figures=bulk_figures,
                        prompt_patches=stage_patches("global_planning"),
                        replan_feedback=note,
                        graph_context=graph_context,
                        evidence_structure=evidence_structure,
                        figure_target=figure_target,
                        panel_target=panel_target,
                    ),
                    structure_builder=lambda retried: structure_from_design_plan(
                        retried,
                        aspect_ratio=aspect_ratio,
                        reference=str(reference) if reference else None,
                        reference_profile=reference_profile,
                        exemplar_brief=exemplar_brief,
                    ),
                    method_extractor=lambda note: extract_method_figure_content(
                        specification,
                        paper_review=prepared.get("paper_review"),
                        mode=method_mode or planner_mode,
                        provenance={"paper": str(Path(paper).resolve()), "planner_mode": planner_mode, "source": "upstream_retry"},
                        feedback=reading_feedback(note),
                        figure_target=figure_target,
                        panel_target=panel_target,
                    ),
                    # The theme planner owns colour. It is handed the palette this round
                    # compiles under and the colour problem the last accepted round measured,
                    # and it returns a whole new theme; nothing here writes a colour itself.
                    theme_planner=lambda feedback: plan_color_theme_vlm(
                        method,
                        retry_design_seed or design,
                        text_plan,
                        reference_profile,
                        exemplar_brief,
                        mode=color_theme_mode,
                        model=color_theme_model,
                        timeout=vlm_timeout,
                        retries=vlm_retries,
                        previous_theme=round_color_theme,
                        color_feedback=feedback,
                        prompt_patches=stage_patches("theme_planning"),
                    ),
                    theme_applier=lambda new_theme: structure_from_design_plan(
                        retry_design_seed or design,
                        aspect_ratio=aspect_ratio,
                        reference=str(reference) if reference else None,
                        reference_profile=reference_profile,
                        exemplar_brief=exemplar_brief,
                        color_theme=new_theme,
                    ),
                )
                write_json(round_dir / retry_report_name, _retry_summary(upstream_retry_report))
                _report_upstream_retry(event_path, round_index, upstream_retry_report)
            if upstream_retry_report.get("applied"):
                # The re-run's own contracts replace this round's proposal: the round compiles,
                # measures and decides on the figure the upstream stage just planned.
                retried_structure = upstream_retry_report.get("structure") or {}
                method = upstream_retry_report.get("method") or method
                if str(upstream_retry_report.get("kind") or "") == "semantics":
                    scientific_graph, scientific_ground_truth, scientific_graph_report = _build_scientific_layers(
                        method,
                        paper_id=paper_id,
                        evidence_map=evidence_map,
                        root=root,
                        # The method was planned again, so the graph is rebuilt from it; it
                        # stays a measurement this row never planned against.
                        provenance="" if evidence_structure else "post_hoc_measurement",
                    )
                design = upstream_retry_report.get("design") or design
                structure = retried_structure or structure
                # A re-planned palette takes effect this round and is inherited only if the
                # round survives; the panel styles the new theme produced travel with it.
                if retried_theme := (upstream_retry_report.get("theme") or {}):
                    round_color_theme = retried_theme
                    round_color_theme_report = upstream_retry_report.get("theme_report") or {}
                    round_panel_style_plan = retried_structure.get("panel_style_plan") or round_panel_style_plan
                proposal_state = {
                    "method": method,
                    "layout": retried_structure.get("figure_layout_plan") or {},
                    "icons": retried_structure.get("icon_generation_plan") or {},
                }
                bounded_change, round_before_snapshot, action_after_snapshot = _upstream_retry_execution(
                    # The palette travels in the compared state, because a theme re-plan is a
                    # real change to the figure the round produced: without it the report would
                    # say nothing happened while the panels and arrows were re-coloured.
                    {**state, "theme": best_color_theme},
                    {**proposal_state, "theme": round_color_theme},
                    action,
                    round_index=round_index,
                    retry_index=retry_index,
                    kind=str(upstream_retry_report.get("kind") or ""),
                )
            elif retry_kind:
                proposal_state, bounded_change, round_before_snapshot, action_after_snapshot = record_unexecuted_repair(
                    state,
                    action,
                    round_index=round_index,
                    reason=str(upstream_retry_report.get("declined_reason") or ""),
                )
            else:
                proposal_state, bounded_change, round_before_snapshot, action_after_snapshot = execute_direct_repair(
                    state,
                    action,
                    round_index=round_index,
                    retry_index=retry_index,
                    domain_profile=resolved_domain,
                )
        else:
            proposal_state = state
            round_before_snapshot = create_state_snapshot(state, label="before_initial_candidate", round_index=round_index)
            action_after_snapshot = create_state_snapshot(state, label="after_initial_candidate", round_index=round_index)
            bounded_change = {
                "contract": "rfs.direct_action_execution",
                "version": 1,
                "action": action,
                "action_type": "initial",
                "applied": False,
                "geometric_applied": False,
                "program_applied": False,
                "changed_node_ids": [],
                "changed_asset_ids": [],
                "program_changes": [],
                "object_diff": diff_state_snapshots(round_before_snapshot, action_after_snapshot),
                "preserve_checks": {"scientific_contract_unchanged": True, "input_state_unmodified": True},
                "retry_index": 0,
            }

        return proposal_state, bounded_change, round_before_snapshot, action_after_snapshot, upstream_retry_report

    for round_index in range(round_limit):
        round_dir = ensure_dir(root / f"round_{round_index:02d}")
        round_started = time.perf_counter()
        round_stage_clock = stage_seconds_from_progress(root)
        round_call_clock = len(model_calls_from_progress(root))
        # The person's own instructions, read again here rather than once for the iteration: a rule
        # filed while this iteration is running belongs to the round after it. The round's planning
        # and its reflection are both handed this list, so the rule reaches the change the round
        # makes and not only the lesson it writes about the change.
        round_preferences = _round_human_preferences(human_preferences_dir, human_preferences)
        person_rules["current"] = list(round_preferences)
        if [str(item.get("entry_id") or "") for item in round_preferences] != [
            str(item.get("entry_id") or "") for item in (human_preferences or [])
        ]:
            append_event(
                event_path,
                "human_preferences_refreshed",
                round=round_index,
                entry_ids=[str(item.get("entry_id") or "") for item in round_preferences],
                because="a person's instruction list was read again at the top of this round",
            )
        current_action = _action_with_applicability(current_action, list(creator_rules_manifest.get("rules") or []))
        append_event(event_path, "round_started", round=round_index, action=current_action, retry_index=action_retry_index)
        prompt_timeline_round["round"] = int(round_index)
        round_injection_ids = set()
        # A move the mechanism's own policy proposed is recorded as a move in its own right.
        # The loop used to leave only the note written when the round was refused, so a round
        # that ran a mechanism move and was then superseded left no trace of who moved the
        # figure or why -- and the same action standing next to the advisory suggestion looked
        # like it had come from the Judge.
        mechanism_source = str(current_action.get("source") or "")
        if mechanism_source in MECHANISM_POLICY_SOURCES:
            append_event(
                event_path,
                "mechanism_move_recorded",
                round=round_index,
                action=str(current_action.get("type") or ""),
                source=mechanism_source,
                reason=str(current_action.get("problem") or ""),
                scope_type=str(current_action.get("scope_type") or ""),
            )
        progress(
            {
                "stage": "round",
                "status": "started",
                "message": f"round {round_index + 1}/{round_limit}",
                "round": round_index,
            }
        )
        retrieval = (
            store.retrieve_with_report(
                limit=5,
                domain="direct_editable",
                context=_memory_context(current_layout, best_dir, resolved_domain),
                statuses=retrieval_bar,
            )
            if store
            else {"rows": [], "report": {}}
        )
        retrieved = list(retrieval["rows"])
        retrieval_report = retrieval["report"]
        retrieval_reports.append({"round": round_index, **retrieval_report})
        append_event(event_path, "experience_retrieved", round=round_index, candidates=int(retrieval_report.get("candidates") or 0), mode=str(retrieval_report.get("mode") or "memory_disabled"), selected=[row.get("evidence", {}).get("experience_id") for row in retrieved])
        candidates = []
        # The theme this round compiles under starts at the accepted one, not at the first
        # draft's: a theme re-plan is accepted like any other change, and a rolled-back round
        # leaves the accepted palette where it was.
        round_color_theme = best_color_theme
        round_color_theme_report = best_color_theme_report
        round_panel_style_plan = best_panel_style_plan
        initial_state = {"method": method, "layout": current_layout, "icons": current_icon_plan}
        # The moves this round runs: the mechanism's own policy first, the coach's suggestion
        # second. The two used to compete for a single slot, so whichever the order put first
        # ate the whole round -- a round with a decomposition or colour move never ran the
        # repair the coach asked for, and the pairing "this suggestion -> how it turned out"
        # was never collected. They are a list now, and the figure the round ships holds both:
        # the second move acts on the state the first one left behind.
        round_moves = [
            dict(item)
            for item in (pending_round_moves or [])
            if isinstance(item, dict) and str(item.get("type") or "")
        ]
        if not round_moves and str(current_action.get("type") or ""):
            round_moves = [dict(current_action)]
        move_guard_values = move_guard if isinstance(move_guard, dict) else {}
        move_records: list[dict[str, Any]] = []
        round_upstream_retries: list[dict[str, Any]] = []
        deferred_move: dict[str, Any] = {}
        move_state = initial_state
        for move_index, move_action in enumerate(round_moves):
            if move_index:
                # The stop the operator left, the disk the run must not fill, and the round's
                # own budget were only read between rounds. A round about to spend a second
                # move asks again, and the answer pins the move it was about to run as the
                # next round's first move -- so a move already decided on is deferred rather
                # than lost, and the mechanism's own move, running first, cannot crowd it out.
                defer_reason = guard_reason(
                    stop_file=move_guard_values.get("stop_file"),
                    disk_path=move_guard_values.get("disk_path"),
                    disk_floor_gb=float(move_guard_values.get("disk_floor_gb") or 0.0),
                    disk_floor_ratio=float(move_guard_values.get("disk_floor_ratio") or 0.0),
                    budget_deadline=float(move_guard_values.get("budget_deadline") or 0.0),
                )
                defer_event = "move_deferred"
                if not defer_reason:
                    # Two refusals that are not about the operator or the disk: the same move
                    # twice, or a move the plan no longer has a target for. Each is named, so
                    # "the round ran one move" is a readable fact rather than a silence.
                    conflict = move_conflict_reason(round_moves[0], move_action)
                    missing = "" if conflict else move_unexecutable_reason(move_action, move_state)
                    defer_reason = conflict or missing
                    defer_event = "move_conflict" if conflict else "move_unexecutable" if missing else "move_deferred"
                if defer_reason:
                    deferred_move = dict(move_action)
                    append_event(
                        event_path,
                        defer_event,
                        round=round_index,
                        move=move_index,
                        action=str(move_action.get("type") or ""),
                        source=move_source(move_action, mechanism_sources=MECHANISM_POLICY_SOURCES),
                        reason=defer_reason,
                    )
                    break
            move_action = _action_with_applicability(move_action, list(creator_rules_manifest.get("rules") or []))
            move_started = time.perf_counter()
            (
                proposal_state,
                bounded_change,
                move_before_snapshot,
                action_after_snapshot,
                move_retry_report,
            ) = _round_action(
                move_state, move_action, retry_index=action_retry_index, move_index=move_index
            )
            move_records.append(
                {
                    "index": move_index,
                    "action": move_action,
                    "type": str(move_action.get("type") or ""),
                    "source": move_source(move_action, mechanism_sources=MECHANISM_POLICY_SOURCES),
                    "is_mechanism": is_mechanism_move(move_action, mechanism_sources=MECHANISM_POLICY_SOURCES),
                    "state_before": move_before_snapshot,
                    "state_after": action_after_snapshot,
                    "execution": bounded_change,
                    "upstream_retry_applied": bool((move_retry_report or {}).get("applied")),
                    "elapsed_seconds": round(time.perf_counter() - move_started, 3),
                }
            )
            round_upstream_retries.append(move_retry_report)
            move_state = proposal_state
        if move_records:
            round_before_snapshot = move_records[0]["state_before"]
            action_after_snapshot = move_records[-1]["state_after"]
            bounded_change = _merge_move_changes(move_records)
        else:
            # Nothing was runnable: the round says it ran no move rather than reporting a
            # move it did not run.
            round_before_snapshot = create_state_snapshot(initial_state, label="before_round", round_index=round_index)
            action_after_snapshot = create_state_snapshot(initial_state, label="after_round", round_index=round_index)
            proposal_state = initial_state
            bounded_change = {}
        if deferred_move:
            # The move the guard stopped is the next round's first move, so a suggestion that
            # was already decided on is run rather than lost.
            pending_deferred_move = dict(deferred_move)
        model_move = move_records[-1]["action"] if len(move_records) > 1 else {}
        # The move the round's Memory row is about. A round that ran the mechanism's always-on
        # default and then the coach's suggestion is credited to the suggestion: the default is
        # what the run would have done with no Memory at all, and filing it as a finding is what
        # the library's own gate refuses anyway. A re-plan the round's measured review demanded
        # is not a default, so a round it led keeps its own credit.
        credited_move = model_move if str(current_action.get("source") or "") == MECHANISM_DEFAULT_SOURCE else {}
        round_retry_report = next(
            (item for item in round_upstream_retries if isinstance(item, dict) and item.get("applied")), {}
        )
        proposal_layout = proposal_state["layout"]
        proposal_icons = proposal_state["icons"]
        write_json(round_dir / "state_before.json", round_before_snapshot)
        write_json(round_dir / "state_after_action.json", action_after_snapshot)
        write_json(round_dir / "action_execution.json", bounded_change)
        write_json(
            round_dir / "round_moves.json",
            round_moves_record(round_index, move_records, source="planned" if pending_round_moves else "single"),
        )
        structure_candidates = structure.get("structure_candidates") or [
            {"plan": proposal_layout}
        ]
        # The figure a reader would see, from the last accepted candidate. Round 0 has none,
        # so its candidate records the diff as unavailable rather than as unchanged.
        previous_program = (
            read_json(best_dir / "figure_program.json")
            if best_dir and (best_dir / "figure_program.json").is_file()
            else {}
        )
        # A move at the figure-program or compile-rule layer edits the drawing, not the picture:
        # it re-lays the same paid icons around a new arrangement, so it must not re-buy them.
        # Only a stage re-run, or a move that names assets of its own, may ask for art again.
        icon_target_ids = [str(item) for item in (bounded_change.get("changed_asset_ids") or []) if str(item)]
        if action_reuses_assets(current_action.get("type")):
            icon_target_ids = []
        for variant in range(max(1, min(CANDIDATE_ROUND_LIMIT, int(candidates_per_round)))):
            layout = (
                structure_candidates[variant % len(structure_candidates)]["plan"]
                if round_index == 0
                else proposal_layout if variant == 0 else create_layout_candidate(proposal_layout, round_index + variant + 1)
            )
            candidate_dir = ensure_dir(
                root / f"round_{round_index:02d}" / f"candidate_{variant:02d}"
            )
            candidate_color_theme = round_color_theme
            candidate_color_theme_report = round_color_theme_report
            candidate_panel_style_plan = round_panel_style_plan
            asset_report = generate_icon_assets(
                proposal_icons,
                candidate_dir,
                mode=icon_mode,
                candidates=icon_candidates,
                progress=progress,
                reuse_assets_from=(
                    (best_dir / "assets")
                    if best_dir
                    else _inherited_icon_root(stage_replay, prestaged_icons, reuse_replayed_icons)
                ),
                target_asset_ids=icon_target_ids or None,
                prompt_patches=stage_patches("icon_generation"),
            )
            composition, composition_report = (
                plan_ppt_composition_vlm(
                    design or layout,
                    text_plan,
                    asset_report,
                    model_reference,
                    reference_template=reference_profile.get("reference_layout_template") or {},
                    exemplar_style_brief=exemplar_brief,
                    model=ppt_plan_model,
                    timeout=vlm_timeout,
                    retries=vlm_retries,
                    color_theme=candidate_color_theme,
                    prompt_patches=stage_patches("ppt_compilation"),
                )
                if ppt_plan_mode == "vlm"
                else ({}, {"mode": "heuristic", "fallback_used": False})
            )
            write_json(candidate_dir / "ppt_composition_plan.json", composition)
            write_json(
                candidate_dir / "ppt_composition_validation.json", composition_report
            )
            write_json(candidate_dir / "ppt_composition_plan_raw.json", composition_report.get("raw_response") or {})
            _report_model_call(progress, "ppt_composition_plan", round_index, variant, composition_report.get("provider"))
            progress(
                {
                    "stage": "ppt_compilation",
                    "status": "started",
                    "round": round_index,
                    "variant": variant,
                }
            )
            compiled = _compile_direct(
                specification,
                layout,
                asset_report,
                candidate_dir,
                title,
                text_plan=text_plan,
                composition=composition,
                reference_template=reference_profile.get("reference_layout_template") or {},
                panel_style_plan=candidate_panel_style_plan,
                memory_action=current_action,
                person_rules=list(person_rules["current"]),
            )
            draft_program_path = candidate_dir / "figure_program.json"
            progress({"stage": "draft_render", "status": "started", "round": round_index, "variant": variant})
            draft_preview = _preview_from_deck(draft_program_path, candidate_dir, candidate_dir / "draft_preview.png")
            progress({"stage": "draft_render", "status": "completed", "round": round_index, "variant": variant})
            progress({"stage": "visual_review", "status": "started", "round": round_index, "variant": variant})
            visual_review_call: dict[str, Any] = {}
            visual_review, visual_review_validation = review_direct_draft(
                read_json(draft_program_path), draft_preview,
                reference_path=model_reference,
                mode=visual_review_mode, model=visual_review_model,
                timeout=vlm_timeout, retries=vlm_retries,
                call_metadata=visual_review_call,
            )
            _report_model_call(progress, "visual_review", round_index, variant, visual_review_call)
            typography_plan = read_json(candidate_dir / "typography_plan.json") if (candidate_dir / "typography_plan.json").exists() else {}
            if typography_plan.get("repair_requests") and visual_review_validation.get("ok", False):
                typography_intents = []
                for request in typography_plan["repair_requests"]:
                    if request.get("action") == "resize_node" and request.get("node_id"):
                        typography_intents.append({"action": "resize_node", "panel_id": next((str(node.get("panel_id") or "") for node in layout.get("nodes", []) if str(node.get("id")) == str(request.get("node_id"))), ""), "object_ids": [request.get("node_id")], "target_bbox_percent": request.get("minimum_bbox_percent"), "reason": request.get("reason")})
                    elif request.get("action") == "reserve_connector_lane":
                        typography_intents.append(request)
                visual_review.setdefault("repair_intents", []).extend(typography_intents)
                visual_review["status"] = "needs_repair"
                progress({"stage": "typography_repair", "status": "started", "round": round_index, "variant": variant})
            if not visual_review_validation.get("ok", False):
                visual_review["status"] = "blocked"
                visual_review["repair_intents"] = []
                visual_review["requires_replan"] = False
            # Explicit priority: structural replan > bounded geometry > typography.
            visual_review = _prioritize_repair_intents(visual_review)
            write_json(candidate_dir / "visual_review_00.json", visual_review)
            write_json(candidate_dir / "visual_review_00_validation.json", visual_review_validation)
            if visual_review.get("requires_replan") or visual_review.get("repair_intents") or visual_review.get("feedback"):
                progress({"stage": "visual_replan", "status": "started", "round": round_index, "variant": variant, "scope": "bounded_feedback_repair"})
                replan_call: dict[str, Any] = {}
                repair_plan, replan_report = plan_feedback_repairs_vlm(read_json(draft_program_path), visual_review, model=visual_review_model, timeout=min(vlm_timeout, 120), retries=vlm_retries, call_metadata=replan_call)
                _report_model_call(progress, "visual_replan", round_index, variant, replan_call)
                write_json(candidate_dir / "visual_replan_validation.json", replan_report)
                write_json(candidate_dir / "feedback_repair_plan.json", repair_plan)
                if replan_report.get("ok") and repair_plan.get("repair_intents"):
                    repaired, repair_report = apply_direct_layout_repairs(read_json(draft_program_path), repair_plan)
                    write_json(candidate_dir / "visual_repair_report_00.json", repair_report)
                    if repair_report.get("applied_count", 0):
                        layout = {**layout, "panels": repaired.get("panels", layout.get("panels", [])), "nodes": repaired.get("semantic_nodes", layout.get("nodes", [])), "connectors": repaired.get("arrows", layout.get("connectors", []))}
                        repaired_nodes = {str(item.get("id")): item for item in layout.get("nodes", []) if isinstance(item, dict)}
                        composition = {**composition, "nodes": [{**item, **repaired_nodes.get(str(item.get("id")), {})} for item in composition.get("nodes", []) if isinstance(item, dict)]}
                        program_for_repair = read_json(draft_program_path)
                        program_for_repair["semantic_nodes"] = repaired.get("semantic_nodes", program_for_repair.get("semantic_nodes", []))
                        program_for_repair["panels"] = repaired.get("panels", program_for_repair.get("panels", []))
                        program_for_repair["arrows"] = repaired.get("arrows", program_for_repair.get("arrows", []))
                        changed_ids = set(repair_report.get("changed_node_ids", []))
                        requested_connector_ids = set(repair_report.get("requested_connector_ids", []))
                        for arrow in program_for_repair.get("arrows", []):
                            if str(arrow.get("id")) in requested_connector_ids or str(arrow.get("source") or arrow.get("source_id")) in changed_ids or str(arrow.get("target") or arrow.get("target_id")) in changed_ids:
                                arrow["path_percent"] = []
                        program_for_repair, _ = resolve_joint_geometry(program_for_repair)
                        program_for_repair, _ = arrange_node_contents(program_for_repair)
                        program_for_repair, _ = canonicalize_node_text_program(program_for_repair)
                        program_for_repair, _ = resolve_direct_connectors(program_for_repair)
                        write_json(draft_program_path, program_for_repair)
                        compiled = _compile_direct(specification, layout, asset_report, candidate_dir, title, text_plan=text_plan, composition=composition, reference_template=reference_profile.get("reference_layout_template") or {}, panel_style_plan=candidate_panel_style_plan, memory_action=current_action, person_rules=list(person_rules["current"]))
                        final_preview = _preview_from_deck(draft_program_path, candidate_dir, candidate_dir / "final_draft_preview.png")
                        if not bulk_figures:
                            recheck_call: dict[str, Any] = {}
                            visual_review, visual_review_validation = review_direct_draft(read_json(draft_program_path), final_preview, reference_path=model_reference, mode=visual_review_mode, model=visual_review_model, timeout=vlm_timeout, retries=vlm_retries, call_metadata=recheck_call)
                            _report_model_call(progress, "visual_review_recheck", round_index, variant, recheck_call)
                            write_json(candidate_dir / "visual_review_01.json", visual_review)
                            write_json(candidate_dir / "visual_review_01_validation.json", visual_review_validation)
                progress({"stage": "visual_replan", "status": "completed" if replan_report.get("ok") else "failed", "round": round_index, "variant": variant})
            elif visual_review.get("repair_intents"):
                progress({"stage": "visual_repair", "status": "started", "round": round_index, "variant": variant})
                repair_input = read_json(draft_program_path)
                if not repair_input.get("panels"):
                    repair_input = {"panels": layout.get("panels", []), "semantic_nodes": layout.get("nodes", [])}
                repaired, repair_report = apply_direct_layout_repairs(repair_input, visual_review)
                write_json(candidate_dir / "visual_repair_report_00.json", repair_report)
                if repair_report.get("applied_count", 0):
                    layout = {**layout, "panels": repaired.get("panels", layout.get("panels", [])), "nodes": repaired.get("semantic_nodes", layout.get("nodes", [])), "connectors": repaired.get("arrows", layout.get("connectors", []))}
                    repaired_nodes = {str(item.get("id")): item for item in layout.get("nodes", []) if isinstance(item, dict)}
                    repaired_composition = {**composition, "nodes": [
                        {**item, **repaired_nodes.get(str(item.get("id")), {})}
                        for item in composition.get("nodes", []) if isinstance(item, dict)
                    ]}
                    composition = repaired_composition
                    repaired_program = read_json(draft_program_path)
                    repaired_program["semantic_nodes"] = repaired.get("semantic_nodes", repaired_program.get("semantic_nodes", []))
                    repaired_program["panels"] = repaired.get("panels", repaired_program.get("panels", []))
                    repaired_program["arrows"] = repaired.get("arrows", repaired_program.get("arrows", []))
                    changed_ids = set(repair_report.get("changed_node_ids", []))
                    requested_connector_ids = set(repair_report.get("requested_connector_ids", []))
                    for arrow in repaired_program.get("arrows", []):
                        if str(arrow.get("id")) in requested_connector_ids or str(arrow.get("source") or arrow.get("source_id")) in changed_ids or str(arrow.get("target") or arrow.get("target_id")) in changed_ids:
                            arrow["path_percent"] = []
                    repaired_program, _ = resolve_joint_geometry(repaired_program)
                    repaired_program, _ = arrange_node_contents(repaired_program)
                    repaired_program, _ = canonicalize_node_text_program(repaired_program)
                    repaired_program, _ = resolve_direct_connectors(repaired_program)
                    write_json(draft_program_path, repaired_program)
                    compiled = _compile_direct(
                        specification, layout, asset_report, candidate_dir, title,
                        text_plan=text_plan, composition=repaired_composition,
                        reference_template=reference_profile.get("reference_layout_template") or {},
                        panel_style_plan=candidate_panel_style_plan,
                        memory_action=current_action,
                        person_rules=list(person_rules["current"]),
                    )
                    final_preview = _preview_from_deck(draft_program_path, candidate_dir, candidate_dir / "final_draft_preview.png")
                    if not bulk_figures:
                        # A bulk corpus run keeps the review that found the
                        # defects and skips the paid re-check of the repair
                        # within the same run.
                        recheck_call = {}
                        visual_review_01, visual_review_01_validation = review_direct_draft(
                            read_json(draft_program_path), final_preview,
                            reference_path=model_reference,
                            mode=visual_review_mode, model=visual_review_model, timeout=vlm_timeout, retries=vlm_retries,
                            call_metadata=recheck_call,
                        )
                        _report_model_call(progress, "visual_review_recheck", round_index, variant, recheck_call)
                        if not visual_review_01_validation.get("ok", False):
                            visual_review_01["status"] = "blocked"
                        visual_review = visual_review_01
                        visual_review_validation = visual_review_01_validation
                        write_json(candidate_dir / "visual_review_01.json", visual_review_01)
                        write_json(candidate_dir / "visual_review_01_validation.json", visual_review_01_validation)
                progress({"stage": "typography_repair", "status": "completed", "round": round_index, "variant": variant})
                progress({"stage": "visual_repair", "status": "completed", "round": round_index, "variant": variant})
            progress({"stage": "visual_review", "status": "completed" if visual_review_validation.get("ok") else "failed", "round": round_index, "variant": variant, "recheck_skipped": bool(bulk_figures)})
            connector_review_call: dict[str, Any] = {}
            connector_review, connector_review_validation = review_connector_routes(
                read_json(draft_program_path),
                candidate_dir / ("final_draft_preview.png" if (candidate_dir / "final_draft_preview.png").exists() else "draft_preview.png"),
                reference_path=model_reference, mode=visual_review_mode,
                model=visual_review_model, timeout=vlm_timeout, retries=vlm_retries,
                call_metadata=connector_review_call,
            )
            _report_model_call(progress, "connector_review", round_index, variant, connector_review_call)
            write_json(candidate_dir / "connector_review.json", connector_review)
            write_json(candidate_dir / "connector_review_validation.json", connector_review_validation)
            if connector_review.get("selections") and connector_review_validation.get("ok"):
                connector_program, connector_apply_report = apply_connector_route_selections(read_json(draft_program_path), connector_review["selections"])
                write_json(candidate_dir / "connector_route_selection_report.json", connector_apply_report)
                write_json(draft_program_path, connector_program)
                compile_ppt(connector_program, candidate_dir)
                write_json(candidate_dir / "connector_visibility_report.json", evaluate_connector_visibility(connector_program))
                _preview_from_deck(draft_program_path, candidate_dir, candidate_dir / "final_draft_preview.png")
            # Color review is intentionally last: geometry and typography must be settled first.
            color_quality = evaluate_color_quality(read_json(draft_program_path))
            write_json(candidate_dir / "color_quality_report.json", color_quality)
            color_review_call: dict[str, Any] = {}
            color_review, color_review_validation = review_color_draft(
                read_json(draft_program_path),
                candidate_dir / ("final_draft_preview.png" if (candidate_dir / "final_draft_preview.png").exists() else "draft_preview.png"),
                color_quality,
                reference_path=model_reference,
                mode=visual_review_mode,
                model=visual_review_model,
                timeout=vlm_timeout,
                retries=vlm_retries,
                call_metadata=color_review_call,
            )
            _report_model_call(progress, "color_review", round_index, variant, color_review_call)
            write_json(candidate_dir / "color_review.json", color_review)
            write_json(candidate_dir / "color_review_validation.json", color_review_validation)
            # Color review is report-only. The initial VLM theme is authoritative
            # for this run; automatic color re-planning was removed because it
            # could overwrite a coherent theme with a less aesthetic fallback.
            write_json(
                candidate_dir / "color_repair_status.json",
                disabled_color_repair_status(color_review),
            )
            progress({"stage": "color_repair", "status": "disabled", "repair_requested": bool(color_review.get("repair_required")), "round": round_index, "variant": variant})
            progress(
                {
                    "stage": "ppt_compilation",
                    "status": "completed",
                    "round": round_index,
                    "variant": variant,
                }
            )
            progress(
                {"stage": "online_judge", "status": "started", "message": judge_mode}
            )
            candidate_program = read_json(candidate_dir / "figure_program.json")
            figure_diff = diff_figure_programs(previous_program, candidate_program)
            write_json(candidate_dir / "figure_object_diff.json", figure_diff)
            # The round's own trace, built from what it already wrote: which stages it replanned,
            # which stage its action applies at, and the numbers the compiled figure moved. It is
            # what the advisory seat reads back before it is allowed to say where a defect entered.
            stage_trace = build_stage_trace(
                round_index=round_index,
                before_program=previous_program,
                after_program=candidate_program,
                reuse_report=stage_replay_report if isinstance(stage_replay_report, dict) else {},
                action=current_action,
            )
            write_json(candidate_dir / "stage_trace.json", stage_trace)
            layout_nodes_by_id = {
                str(item.get("id") or ""): item
                for item in layout.get("nodes", [])
                if isinstance(item, dict)
            }
            graph_nodes = []
            for item in candidate_program.get("semantic_nodes", []):
                source = layout_nodes_by_id.get(str(item.get("id") or ""), {})
                graph_nodes.append({**source, **item, "semantic_source_id": source.get("semantic_source_id") or item.get("semantic_source_id") or item.get("id")})
            visual_graph, visual_graph_report = build_visual_graph(
                {
                    "canvas": layout.get("canvas") or {},
                    "topology": layout.get("topology"),
                    "panels": candidate_program.get("panels") or layout.get("panels") or [],
                    "nodes": graph_nodes,
                    "connectors": candidate_program.get("arrows") or [],
                },
                scientific_graph=scientific_graph,
            )
            editable_graph, graph_bindings, editable_graph_report = build_editable_graph(
                candidate_program,
                visual_graph=visual_graph,
                scientific_graph=scientific_graph,
            )
            graph_alignment_report = align_graph_layers(
                scientific_graph, visual_graph, editable_graph, graph_bindings
            )
            grounding_gate = evaluate_grounding_gate(graph_alignment_report)
            write_json(candidate_dir / "visual_graph.json", visual_graph)
            write_json(candidate_dir / "visual_graph_report.json", visual_graph_report)
            write_json(candidate_dir / "editable_graph.json", editable_graph)
            write_json(candidate_dir / "editable_graph_report.json", editable_graph_report)
            write_json(candidate_dir / "graph_bindings.json", graph_bindings)
            write_json(candidate_dir / "graph_alignment_report.json", graph_alignment_report)
            write_json(candidate_dir / "grounding_gate.json", grounding_gate)
            candidate_decomposability = build_decomposability_report(
                editable_graph,
                visual_graph,
                candidate_program,
                panels=candidate_program.get("panels") or layout.get("panels") or [],
                round_index=round_index,
                history=decomposability_history,
                thresholds=decomposability_limits,
            )
            write_json(candidate_dir / "decomposability_report.json", candidate_decomposability)
            # A prohibition is only enforcement if a machine checks it: the same round that
            # measured the metrics decides whether this candidate broke one, and a broken
            # prohibition leaves through the existing accept/rollback path as a blocker.
            prohibition_check = check_prohibited_patches(candidate_decomposability, prohibited_patches)
            write_json(candidate_dir / "prompt_patch_check.json", prohibition_check)
            if prohibition_check["violations"]:
                append_event(
                    event_path,
                    "prompt_patch_prohibition_blocked",
                    round=round_index,
                    variant=variant,
                    patch_ids=[str(item.get("patch_id") or "") for item in prohibition_check["violations"]],
                    blockers=list(prohibition_check["blockers"]),
                )
            candidate_after_snapshot = create_state_snapshot(
                {
                    "method": method,
                    "layout": layout,
                    "icons": proposal_icons,
                    "program": candidate_program,
                    "graphs": {
                        "scientific_graph_id": scientific_graph.get("graph_id"),
                        "visual_graph_id": visual_graph.get("graph_id"),
                        "editable_graph_id": editable_graph.get("graph_id"),
                    },
                    "grounding_gate": grounding_gate,
                },
                label=f"candidate_{variant:02d}_after_compile",
                round_index=round_index,
            )
            write_json(candidate_dir / "state_before.json", round_before_snapshot)
            write_json(candidate_dir / "state_after.json", candidate_after_snapshot)
            write_json(candidate_dir / "state_diff.json", diff_state_snapshots(round_before_snapshot, candidate_after_snapshot))
            write_json(candidate_dir / "action_execution.json", bounded_change)
            candidate_layout = {"panels": candidate_program.get("panels", []), "nodes": candidate_program.get("semantic_nodes", []), "connectors": candidate_program.get("arrows", [])}
            # Advice the library has earned, offered to the advisory role for this figure: a rule
            # is offered only when its own defect pattern is one this figure has, so the ask is
            # phrased in the terms of the illness actually on screen.
            advice_selection = (
                suggestion_store.select(apply_at="ppt_compilation", defect_pattern=defect_patterns(candidate_layout))
                if suggestion_store is not None
                else {"rules": [], "rejected": [], "considered": 0}
            )
            advice_for_prompt = list(advice_selection.get("rules") or [])
            if advice_for_prompt:
                append_event(event_path, "suggestion_rules_offered", round=round_index, variant=variant, rule_ids=[str(item.get("rule_id") or "") for item in advice_for_prompt])
            judgement = online_judge.evaluate(DirectJudgeInput(
                method=method,
                layout=candidate_layout,
                icons=proposal_icons,
                program=candidate_program,
                preview=candidate_dir / "final_draft_preview.png" if (candidate_dir / "final_draft_preview.png").exists() else candidate_dir / "draft_preview.png",
                domain_profile=resolved_domain,
                prior_feedback=list((best_judgement or {}).get("stage_feedback", [])),
                allowed_scope={"scope": "global_or_candidate", "round": round_index},
                experiences=retrieved,
                grounding_gate=grounding_gate,
                prompt_patches=stage_patches("repair"),
                color_report=pending_color_feedback,
                suggestion_rules=advice_for_prompt,
                human_preferences=list(round_preferences),
                comparison_preview=_accepted_preview(best_dir, baseline_input),
                stage_trace=compact_stage_trace(stage_trace),
                layer_costs=_layer_costs_for_prompt(),
            ))
            append_event(event_path, "judge_feedback_recorded", round=round_index, role=getattr(online_judge, "role", "unknown"), judgement=judgement)
            # The evidence blocks the same call handed the seat, so a claim about one of them is
            # read against what the seat could actually see rather than against the trace alone.
            # The figure itself is deliberately left out: its own field names would let any word
            # the drawing happens to use pass for a measurement.
            shown_evidence = {
                "grounding_gate": grounding_gate,
                "color_report": pending_color_feedback,
                "suggestion_rules": advice_for_prompt,
                "prompt_patches": stage_patches("repair"),
                "experiences": retrieved,
            }
            attribution = attribution_verdict(judgement, stage_trace, shown=shown_evidence)
            if not attribution["ok"]:
                # An attribution that names a stage the round never ran, or a field the round
                # never put in front of the seat, is not evidence: it is dropped and recorded
                # rather than kept beside the judgement as though it had been checked.
                append_event(
                    event_path,
                    "attribution_rejected",
                    round=round_index,
                    variant=variant,
                    reason=attribution["reason"],
                    attribution=attribution["attribution"],
                )
                judgement = {key: value for key, value in judgement.items() if key not in ("fault_stage", "fault_requirement", "fault_evidence")}
            elif attribution["attribution"].get("fault_evidence_dropped"):
                # Part of the claim was kept and part of it named something the round never
                # showed: the kept part stands, and the part that was thrown away is written
                # down where the round ledger can be read against it.
                append_event(
                    event_path,
                    "attribution_partly_dropped",
                    round=round_index,
                    variant=variant,
                    kept=attribution["attribution"].get("fault_evidence"),
                    dropped=attribution["attribution"].get("fault_evidence_dropped"),
                )
            progress(
                {
                    "stage": "online_judge",
                    "status": "completed",
                    "judge_status": judgement.get("status"),
                    "score": judgement.get("overall_score"),
                }
            )
            # The second question is asked here, per candidate, so the round decides on a
            # measurement of the same instrument it will later be compared against. It never
            # goes through the advisory gate short-circuit: a gate verdict is not a model score,
            # and the two must never be subtracted from one another.
            candidate_acceptance_input = _acceptance_input(
                method=method,
                program=candidate_program,
                icons=proposal_icons,
                preview=candidate_dir / "final_draft_preview.png" if (candidate_dir / "final_draft_preview.png").exists() else candidate_dir / "draft_preview.png",
                domain_profile=resolved_domain,
                grounding_gate=grounding_gate,
                allowed_scope={"scope": "frozen_acceptance", "round": round_index},
                layer_plan=layer_plan,
            )
            acceptance, acceptance_error = _measure_acceptance(frozen_judge, candidate_acceptance_input)
            append_event(event_path, "acceptance_measured", round=round_index, variant=variant, role=getattr(frozen_judge, "role", "unknown"), judgement=acceptance, error=acceptance_error or None)
            candidates.append(
                {
                    "variant": variant,
                    "candidate_index": variant,
                    "layout": layout,
                    "asset_report": asset_report,
                    "compiled": compiled,
                    "judgement": judgement,
                    # The gate's verdict travels with the candidate the round decided on, so the
                    # ledger can say what the coach claimed even when the claim was refused.
                    "attribution": attribution,
                    "acceptance": acceptance,
                    "acceptance_error": acceptance_error,
                    # The compiled state's own question input travels with the candidate, because
                    # one of these inputs becomes the "before" side of the next round's comparison.
                    "acceptance_input": candidate_acceptance_input,
                    "visual_review": visual_review,
                    "visual_review_validation": visual_review_validation,
                    "color_quality": color_quality,
                    "color_review": color_review,
                    "color_theme": candidate_color_theme,
                    "color_theme_report": candidate_color_theme_report,
                    "panel_style_plan": candidate_panel_style_plan,
                    "visual_score": sum(float(visual_review.get("scores", {}).get(key) or 0.0) for key in ("balance", "readability", "density", "flow_clarity")) / 4.0 if any(visual_review.get("scores", {}).values()) else (1.0 if visual_review.get("status") == "pass" else 0.0),
                    "dir": str(candidate_dir),
                    "composition": composition,
                    "icon_plan": proposal_icons,
                    "bounded_change": bounded_change,
                    "grounding_gate": grounding_gate,
                    "decomposability_report": candidate_decomposability,
                    "prompt_patch_check": prohibition_check,
                    "prompt_patch_blockers": list(prohibition_check["blockers"]),
                    "figure_diff": figure_diff,
                    "stage_trace": stage_trace,
                    "state_after": candidate_after_snapshot,
                }
            )
        # Only a measured candidate can win: one nobody could score is not the best of them, and
        # the deterministic visual check only settles ties between answers the same role gave.
        def _compare_candidates(incumbent: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any] | None:
            """Ask the acceptance role which of two candidates a reader is better served by."""

            judgement, error = _measure_pairwise(
                frozen_judge,
                DirectPairJudgeInput(before=incumbent["acceptance_input"], after=challenger["acceptance_input"]),
            )
            append_event(
                event_path,
                "candidate_compared",
                round=round_index,
                incumbent=int(incumbent.get("candidate_index", incumbent.get("variant", 0))),
                challenger=int(challenger.get("candidate_index", challenger.get("variant", 0))),
                judgement=judgement,
                error=error or None,
            )
            return judgement

        selected, round_selection = select_round_candidate(
            candidates,
            compare=_compare_candidates if len(candidates) > 1 and _pairwise_capable(frozen_judge) else None,
        )
        # The selected candidate's trace is the round's trace: it is the one the decision was
        # made about, and the one the dataset builder reads into the trajectory record.
        write_json(round_dir / "stage_trace.json", selected.get("stage_trace") or {})
        write_json(round_dir / "round_selection.json", round_selection)
        append_event(
            event_path,
            "round_candidate_selected",
            round=round_index,
            method=round_selection.get("method"),
            winner=round_selection.get("winner"),
            candidate_count=round_selection.get("candidate_count"),
        )
        # The chosen candidate's own acceptance answer is this round's "after". It is not asked
        # again: a second call would measure the figure and the sampling noise together, and the
        # result would be a different number for the same drawing.
        # The colour problem the chosen candidate measured, kept as the next round's input to
        # the theme re-plan and to the advisory role. The four metrics are recorded as evidence:
        # they never decide whether the round is accepted.
        round_color_metrics = {
            "before": dict(baseline_color_metrics),
            "after": _color_metrics(selected.get("color_quality")),
        }
        selected_color_review = selected.get("color_review") or {}
        if selected_color_review.get("repair_required"):
            color_feedback = {
                "contract": "rfs.color_feedback",
                "version": 1,
                "round": round_index,
                "candidate_index": int(selected.get("candidate_index", selected["variant"])),
                "repair_required": True,
                "feedback": [dict(item) for item in (selected_color_review.get("feedback") or []) if isinstance(item, dict)],
                "metrics": dict(round_color_metrics),
            }
            write_json(round_dir / "color_feedback.json", color_feedback)
            pending_color_feedback = color_feedback
        else:
            # The palette the round produced drew no complaint, so there is nothing left for
            # the next round to re-plan against. Stale feedback would ask about a fixed problem.
            pending_color_feedback = {}
        frozen = selected["acceptance"]
        # The round is decided by showing the accepted state and the chosen candidate together
        # and asking the acceptance role which of the two a reader is better served by. Two
        # absolute scores subtracted from one another were measuring the model's rounding: of 74
        # numbers the served model produced, every one but a single 0.65 sat on a 0.1 grid. Round
        # 0 has no earlier state to show, and a judge with no model behind it (the offline mode)
        # cannot answer the question at all, so both keep the measured-score path -- and the
        # record says which of the two decided the round.
        pairwise: dict[str, Any] | None = None
        pairwise_error = ""
        # Compared whenever there is a state to compare against: a round the run has already
        # accepted, or the baseline figure the run was handed. A first round used to have
        # neither and was accepted unmeasured, which is what left its experience without an
        # "after" reading; with a baseline it is measured like every other round.
        pairwise_before = best_acceptance_input if best_acceptance_input is not None else baseline_input
        if _pairwise_capable(frozen_judge) and pairwise_before is not None:
            pairwise, pairwise_error = _measure_pairwise(
                frozen_judge,
                DirectPairJudgeInput(before=pairwise_before, after=selected["acceptance_input"]),
            )
            append_event(event_path, "acceptance_compared", round=round_index, role=getattr(frozen_judge, "role", "unknown"), judgement=pairwise, error=pairwise_error or None)
            # The before state is the accepted measurement when there is one, and the baseline
            # figure's own measurement otherwise. The comparison's choice scores are written onto
            # it, so the baseline carries both the defects it already had and a score.
            before_state = best_acceptance if best_acceptance is not None else baseline_measurement
            before_state, frozen = _apply_choice_scores(before_state, frozen, pairwise)
            best_acceptance = before_state
            selected["acceptance"] = frozen
            # What the acceptance seat said about the round's own layer, filed beside the round.
            # It decides nothing here: the seat's two-field answer is what decided the round, and
            # this half is a record the loop reads back to ask whether the seat can be trusted
            # with the direction at all. Nothing is written when the seat was not asked.
            if str(pairwise.get("direction_authority") or "off") in {"record", "authority"}:
                write_json(
                    round_dir / "acceptance_direction.json",
                    {
                        "contract": "rfs.acceptance_direction",
                        "version": 1,
                        "round": int(round_index),
                        "authority": str(pairwise.get("direction_authority") or ""),
                        "asked_about": str((layer_plan or {}).get("working_on") or ""),
                        "answer": dict(pairwise.get("direction") or {}),
                        "problem": str(pairwise.get("direction_problem") or ""),
                    },
                )
        if len(move_records) > 1 and _pairwise_capable(frozen_judge) and pairwise_before is not None:
            # Each move is measured on its own as well as in the round's total. The second move
            # ran on top of the first, so the round's own number cannot say which of the two
            # earned it; the figure the first move alone produced is compiled and shown to the
            # same role, and that comparison is what the first move is billed for. A compile
            # that fails is recorded as an unmeasured move rather than silently dropped.
            try:
                move_dir = ensure_dir(round_dir / "move_00")
                move_asset_report = generate_icon_assets(
                    move_records[0]["state_after"]["state"]["icons"],
                    move_dir,
                    mode=icon_mode,
                    candidates=icon_candidates,
                    progress=progress,
                    reuse_assets_from=(best_dir / "assets") if best_dir else prestaged_icons,
                    target_asset_ids=None,
                    prompt_patches=stage_patches("icon_generation"),
                )
                _compile_direct(
                    specification,
                    move_records[0]["state_after"]["state"]["layout"],
                    move_asset_report,
                    move_dir,
                    title,
                    text_plan=text_plan,
                    composition={},
                    reference_template=reference_profile.get("reference_layout_template") or {},
                    panel_style_plan=round_panel_style_plan,
                    memory_action=move_records[0]["action"],
                    person_rules=list(person_rules["current"]),
                )
                move_program_path = move_dir / "figure_program.json"
                move_preview = _preview_from_deck(move_program_path, move_dir, move_dir / "draft_preview.png")
                move_input = _acceptance_input(
                    method=method,
                    program=read_json(move_program_path),
                    icons=move_records[0]["state_after"]["state"]["icons"],
                    preview=move_preview,
                    domain_profile=resolved_domain,
                    grounding_gate=selected.get("grounding_gate"),
                    allowed_scope={"scope": "frozen_acceptance", "round": round_index},
                    layer_plan=layer_plan,
                )
                move_acceptance, move_error = _measure_acceptance(frozen_judge, move_input)
                move_pairwise, move_pairwise_error = _measure_pairwise(
                    frozen_judge, DirectPairJudgeInput(before=pairwise_before, after=move_input)
                )
                # The first move's score is read off the comparison, exactly as the round's own
                # is: a seat that answers the choice question returns no absolute number, and
                # reading one off its single-image answer would score the move with an instrument
                # the round does not use.
                _, move_scored = _apply_choice_scores(None, move_acceptance, move_pairwise)
                move_records[0]["measurement"] = {
                    "acceptance": move_scored,
                    "acceptance_error": move_error,
                    "pairwise": move_pairwise,
                    "pairwise_error": move_pairwise_error,
                    "score": measured_score(move_scored),
                    "dir": str(move_dir),
                }
                append_event(
                    event_path,
                    "move_measured",
                    round=round_index,
                    move=0,
                    action=str(move_records[0]["type"]),
                    score=measured_score(move_scored),
                    error=move_error or move_pairwise_error or None,
                )
            except Exception as exc:
                move_records[0]["measurement"] = {"score": None, "error": str(exc)}
                append_event(event_path, "move_unmeasured", round=round_index, move=0, error=str(exc))
        elif round_index == 0 and pairwise_before is None:
            # Nothing was supplied to compare against, so this round opens the run unmeasured.
            append_event(event_path, "baseline_unavailable", round=round_index, reason="no earlier accepted state and no baseline run supplied")
        if move_records:
            move_records[-1]["measurement"] = {
                "acceptance": frozen,
                "pairwise": pairwise,
                "score": measured_score(frozen),
                "dir": str(selected["dir"]),
            }
            # Written a second time now that each move carries its own measurement: the first
            # write lands before the round has decided which candidate it keeps.
            write_json(
                round_dir / "round_moves.json",
                round_moves_record(round_index, move_records, source="planned" if pending_round_moves else "single"),
            )
        score = measured_score(frozen)
        # A round that ran the mechanism's move and then the coach's suggestion is decided
        # about the figure both produced, but it is the suggestion that becomes an experience,
        # so the record names that move rather than whichever ran first.
        decision = compare_candidate(
            best_acceptance,
            frozen,
            action=(credited_move or current_action),
            bounded_change=selected.get("bounded_change"),
            figure_diff=selected.get("figure_diff"),
            baseline_ready=baseline_ready,
            after_pairwise=pairwise,
            before_grounding=best_grounding,
            after_grounding=selected.get("grounding_gate"),
        )
        prohibition_blockers = list(selected.get("prompt_patch_blockers") or [])
        if prohibition_blockers:
            # The Judge never saw this: the compiled artifact broke a prohibition the
            # library had already measured, so the round rolls back through the Gate it
            # always used rather than a second, patch-specific path.
            decision = {**decision, "accepted": False, "reason": "prompt_patch_prohibition_violated", "prohibited_prompt_patches": prohibition_blockers}
        # What the compile did with this round's rule, read before the reflection rather than after
        # it. A reflection asked why a rule failed cannot answer without seeing whether the rule was
        # carried out, and the round already holds that answer: the compile wrote it down when it
        # drew the candidate. Every other round leaves this empty, so no other prompt moves.
        # Read the good figure against this round's candidate before the reflection, because the
        # reflection is where it is used: a coach asked why the figure is worse than it should be
        # cannot say so without having been shown what a good figure of this kind looks like.
        reference_gap = _ask_reference_gap(
            online_judge,
            gap_reference=gap_reference,
            selected=selected,
            round_index=round_index,
            event_path=event_path,
        )
        if reference_gap:
            write_json(round_dir / "reference_gap_report.json", reference_gap)
        implementation_evidence: dict[str, Any] = {}
        if str(current_action.get("type") or "") in MECHANISM_ACTIONS:
            implementation_evidence = mechanism_rule_evidence(
                round_index=round_index,
                action=current_action,
                candidate_dir=Path(selected["dir"]),
                figure_diff=selected.get("figure_diff"),
            )
        elif action_class(current_action.get("type")) == FIGURE_PROGRAM_CLASS:
            # A move on the figure leaves no compile report of its own, so it is given the diff
            # between the program it was handed and the program it produced. Without it, the two
            # attributions that earn another attempt are refused as uncheckable for every round
            # that did not run a rule, and a suggestion that may only have overshot is written off
            # on a single reading.
            implementation_evidence = figure_move_evidence(
                round_index=round_index,
                action=current_action,
                figure_diff=selected.get("figure_diff"),
            )
        # The advisory answer's own attribution rides along: a reflection cannot say which stage's
        # requirement went unwritten if it was never told which one the round's own review named,
        # and a patch is exactly that sentence said again for a later prompt. A round whose answer
        # named no stage adds no key, so every other reflection prompt stays byte-identical.
        advisory_judgement = selected.get("judgement") if isinstance(selected.get("judgement"), dict) else {}
        fault_attribution: dict[str, Any] = {}
        if str(advisory_judgement.get("fault_stage") or "").strip() or str(advisory_judgement.get("fault_requirement") or "").strip():
            fault_attribution = {
                "stage": str(advisory_judgement.get("fault_stage") or "").strip(),
                "requirement": str(advisory_judgement.get("fault_requirement") or "").strip(),
                "evidence": [dict(item) for item in (advisory_judgement.get("fault_evidence") or []) if isinstance(item, dict)],
            }
        patch_context = _prompt_patch_context(
            patch_store,
            stage=str(fault_attribution.get("stage") or stage_for_action(str((current_action or {}).get("type") or "")) or ""),
        )
        outcome = {
            "before": best_acceptance or {},
            "after": frozen,
            "action": current_action,
            "bounded_change": selected.get("bounded_change") or {},
            "new_blockers": decision.get("new_blockers"),
            "preserve_failures": decision.get("preserve_failures") or [],
            "decision": decision,
            # The comparison is part of what the reflection may explain: a round that lost the
            # comparison and a round the model itself flagged as noise are different stories.
            "pairwise": decision.get("pairwise"),
            # The advisory answer rode along: a reflection about why the figure changed can only
            # be written by someone who was told what was asked for and what was measured.
            "advisory_feedback": list((selected.get("judgement") or {}).get("stage_feedback") or []),
            # A mechanism round is given the compile's report and a move on the figure is given
            # the figure's own diff; only a round that neither ran a rule nor moved the figure
            # carries no key, so those prompts stay byte-identical to the ones they were measured
            # under.
            **({"implementation": implementation_evidence} if implementation_evidence else {}),
            # The gap against a good figure rides along for the same reason the implementation
            # does: it is a fact about this round that the reflection can only explain if it was
            # told. A round with no reference adds no key, so every other prompt is byte-identical.
            **({"reference_gap": reference_gap} if reference_gap else {}),
            **({"fault_attribution": fault_attribution} if fault_attribution else {}),
            # The stage's own requirements and the sentences already written for it travel with the
            # round for the same reason the attribution does: a seat cannot write the sentence that
            # is missing when it was never shown what is already there.
            **({"prompt_patch_context": patch_context} if patch_context else {}),
            # A person's own instruction, when there is one is, rides along the way the fault
            # attribution does: the reflection can only follow a sentence it was shown. A round
            # with no such instruction adds no key, so every other reflection prompt is unchanged.
            **({"human_preferences": [dict(item) for item in round_preferences]} if round_preferences else {}),
            # The direction this round is on travels with the round for the same reason a person's
            # instruction does: the reflection can only answer about the layer it was told it is
            # working on.
            **({"layer_plan": dict(layer_plan)} if layer_plan else {}),
        }
        try:
            # The reflection is asked for every round, round 0 included, and the run's own drawing
            # is asked about like any other: a round nobody reflects on cannot state the rule it
            # just demonstrated, and the round that wins is often the one that drew the figure.
            # A seat that cannot answer falls back to the default reflection below, so asking is
            # never the reason a run stops.
            reflection = online_judge.reflect(outcome)
        except Exception as exc:
            reflection = {**default_reflection(decision, current_action, retry_index=action_retry_index), "provider_error": str(exc)}
            # A round nobody reflected on says so, in its own event: the run must not read as
            # if the seat had answered when the loop wrote the words itself.
            append_event(event_path, "reflection_defaulted", round=round_index, reason=str(exc))
        # The rule a reflection writes is read where it is written: a rule the compiler cannot
        # run, or one that names this figure's own boxes, is refused here and recorded, so the
        # library cannot look empty for a reason nobody can read off the run.
        reflection, reflection_rule_rejection = validate_reflection_rule(reflection, figure=current_layout)
        if reflection_rule_rejection:
            append_event(event_path, "reflection_rule_rejected", round=round_index, **reflection_rule_rejection)
        if reflection.get("rule_missing"):
            # The seat was asked a second time for the rule and still stated none. The round is
            # recorded as having taught nothing rather than quietly reported as if it had, so a
            # library that stays empty can be told apart from a round that had nothing to say.
            append_event(event_path, "reflection_rule_missing", round=round_index, next_policy=str(reflection.get("next_policy") or ""), outcome=str(reflection.get("outcome") or ""))
        # The reflection is a claim, so acting on it can only tighten this round. It may
        # withdraw an accept it calls noise, and it may take back the guidance the round ran
        # under; that is how "the model said why it went wrong" reaches the next round
        # instead of staying a note nothing reads.
        reflection_policy = apply_reflection_policy(
            decision,
            current_action,
            reflection,
            injected_experience=injected_experience,
            condition=exclusion_context(current_layout, resolved_domain),
            implementation=implementation_evidence,
            pairwise=decision.get("pairwise"),
        )
        decision = reflection_policy["decision"]
        if not _round_runs_a_move(round_index, current_action):
            # The round ran no move, so it has no repair to revise: the reflection now answers for
            # the drawing this run planned. What it teaches about how figures are drawn is kept --
            # that is the whole point of asking -- but its action is dropped, because carrying it
            # into the next round would have that round redraw what this one has just drawn.
            dropped_action = dict(reflection_policy.get("revised_action") or {})
            if dropped_action:
                append_event(event_path, "reflection_action_dropped", round=round_index, reason="the round ran no move, so it has no repair to revise", action=str(dropped_action.get("type") or ""))
            reflection_policy = {**reflection_policy, "revised_action": {}}
        # The rule this round is asking to repair, read while the round is still the current one:
        # the next round's proposal block only sees ``current_action`` after this one has moved on.
        follow_up_rule_fingerprint = (
            mechanism_rule_fingerprint(current_action) if reflection_policy.get("follow_up_action") else ""
        )
        for event in reflection_policy["events"]:
            payload = dict(event)
            append_event(event_path, str(payload.pop("event")), round=round_index, **payload)
        stated_rule = reflection_policy.get("revised_rule") or {}
        if suggestion_store is not None and stated_rule:
            # The reflection stated how to advise the next paper. It is kept as one witness: the
            # library offers a rule only after two rounds reach it, so acting on a single round
            # is impossible by construction rather than by discipline.
            recorded_rule = suggestion_store.record_statement(
                rule=stated_rule,
                apply_at=str(stated_rule.get("apply_at") or "ppt_compilation"),
                defect_pattern=defect_patterns(current_layout),
                source_run=str(paper),
                round_index=round_index,
            )
            append_event(
                event_path,
                "suggestion_rule_recorded",
                round=round_index,
                rule_id=str((recorded_rule.get("rule") or {}).get("rule_id") or ""),
                support=int(recorded_rule.get("support") or 0),
                usable=bool(recorded_rule.get("usable")),
                recorded=bool(recorded_rule.get("ok")),
                reason=str(recorded_rule.get("reason") or ""),
            )
        if store is not None:
            for effect in reflection_policy["experience_effects"]:
                if str(effect["kind"]) == "disable":
                    store.transition(str(effect["experience_id"]), "disabled", reason=str(effect["reason"]), round=round_index)
                else:
                    store.append_event(str(effect["experience_id"]), "experience_condition_excluded", condition=effect.get("condition") or {}, reason=str(effect["reason"]), round=round_index)
        # A verdict the round itself doubts -- the reflection said this comparison could come out
        # the other way -- is put to the seat once more, with the loop's own readings of the two
        # figures in the question, before the round is decided on it. Both answers are kept: a
        # repeat stands as the answer and the round says it was asked twice, and a different answer
        # is the answer the round is decided on. The doubt is recorded either way, because "we
        # asked twice and got the same thing" is a fact about the instrument, not about the figure.
        if pairwise and pairwise_before is not None and _pairwise_capable(frozen_judge) and decision.get("noise_doubt"):
            first_verdict = pairwise_verdict(pairwise)
            second_pairwise, second_error = _measure_pairwise(
                frozen_judge,
                DirectPairJudgeInput(
                    before=pairwise_before,
                    after=selected["acceptance_input"],
                    candidate_a=str(decision.get("candidate_a") or ""),
                    candidate_b=str(decision.get("candidate_b") or ""),
                ),
            )
            second_verdict = pairwise_verdict(second_pairwise)
            flipped = bool(second_verdict) and second_verdict != first_verdict
            decision = {
                **decision,
                "seat_reasked": (
                    f"changed: {first_verdict} then {second_verdict}"
                    if flipped
                    else f"repeated: {first_verdict or 'no verdict'} then {second_verdict or 'no verdict'}"
                    + (f" ({second_error})" if second_error else "")
                ),
            }
            if flipped:
                pairwise = second_pairwise
                before_state, frozen = _apply_choice_scores(before_state, frozen, pairwise)
                best_acceptance = before_state
                selected["acceptance"] = frozen
                decision = {
                    **decision,
                    "pairwise": pairwise,
                    "pairwise_verdict": second_verdict,
                    "accepted": second_verdict == "after" or bool(decision.get("coverage_improved")),
                    "reason": "pairwise_preferred" if second_verdict == "after" else "pairwise_not_preferred",
                }
            append_event(
                event_path,
                "seat_reasked",
                round=round_index,
                why="the round doubted its own comparison",
                first=first_verdict or "unavailable",
                second=second_verdict or "unavailable",
                flipped=flipped,
            )
        # A person at this seat decides this round. The model's own answer is kept beside theirs --
        # both are written down, in the round's folder and on the question's own card -- because the
        # distance between the two is the signal this channel exists to produce. A question nobody
        # answers leaves the round exactly as the model decided it.
        human_note: dict[str, Any] = {}
        # Only where there are two figures to choose between. A round with nothing before it has no
        # "A" to prefer, and asking a person to keep the earlier figure when there is no earlier
        # figure is a question whose only answer throws the run away.
        if human_gate is not None and pairwise_before is not None:
            # The layer this round worked on. It is the one the person is asked about after the
            # figure: whether it is finished with yet, and -- if it is -- which layer the fault
            # has moved down to. Asked in that order because a person who is still unhappy with
            # this layer has no business naming a lower one, which is the whole of the direction.
            # The layer the round is on is the one the run was told to work on. The move's own
            # action name is only a fallback: an action maps to a stage through the memory profile,
            # so a round working on the composition can read back as "repair" and the person would
            # be asked whether a layer nobody touched is finished with.
            round_stage = str((layer_plan or {}).get("working_on") or "") or stage_for_action(
                str((credited_move or current_action or {}).get("type") or "")
            )
            answer = human_gate(
                {
                    "kind": "figure_choice",
                    "paper": str(paper_id or Path(str(paper)).stem),
                    "round": int(round_index),
                    "stage": round_stage,
                    "question": (
                        "Which figure is the better one to keep: A (before this round) or B (this round's)? "
                        f"Then: is the layer this round worked on ({round_stage or 'unknown'}) finished with -- "
                        "answer `settled: yes` if it is, or leave it and say what is still wrong with it. "
                        "Only if it is finished with, name the layer the fault has moved to in `next_layer` "
                        "(any of paper_reading, content_planning, global_planning, segmentation, "
                        "theme_planning, icon_generation, ppt_compilation, repair), or leave it empty."
                    ),
                    "figures": {
                        "A": _before_preview(best_dir, baseline_from),
                        "B": _preview_path(Path(str(selected["dir"]))),
                    },
                    "readings": {"before": measured_score(best_acceptance), "after": measured_score(frozen)},
                    "model_answer": {
                        "choice": "B" if decision.get("accepted") else "A",
                        "accepted": bool(decision.get("accepted")),
                        "reason": str(decision.get("reason") or ""),
                    },
                }
            )
            human_note = dict(answer or {})
            if bool(human_note.get("answered")):
                decided_by_person = str(human_note.get("choice") or "").strip().upper()
                if decided_by_person in {"A", "B"}:
                    person_accepts = decided_by_person == "B"
                    model_accepted = bool(decision.get("accepted"))
                    decision = {
                        **decision,
                        "model_accepted": model_accepted,
                        "accepted": person_accepts,
                        "reason": "human_preferred" if person_accepts else "human_rejected",
                        "decided_by": "human",
                        "human_choice": decided_by_person,
                        "human_note": str(human_note.get("note") or ""),
                    }
                    append_event(
                        event_path,
                        "human_verdict_applied",
                        round=round_index,
                        choice=decided_by_person,
                        model_choice="B" if model_accepted else "A",
                        flipped=person_accepts != model_accepted,
                        card=str(human_note.get("card") or ""),
                    )
            # The layer half of the same answer, in the words the loop reads back: a layer the
            # person called finished with, and the layer they say the fault moved to. Both are
            # recorded here rather than acted on -- moving the work face is the loop's own act,
            # and it reads this file next to the iteration the way it reads the model's own answer.
            fields = human_note.get("fields") if isinstance(human_note.get("fields"), dict) else {}
            settled_yes = str(fields.get("settled") or fields.get("layer_settled") or "").strip().casefold() in {"yes", "y", "true", "settled", "done", "1"}
            next_layer = str(fields.get("next_layer") or fields.get("direction") or "").strip()
            write_json(
                round_dir / "human_verdict.json",
                {
                    "contract": "rfs.human_verdict",
                    "version": 1,
                    "round": int(round_index),
                    "kind": "figure_choice",
                    "stage": str(round_stage or ""),
                    "settled_layer": str(round_stage or "") if (bool(human_note.get("answered")) and settled_yes and round_stage) else "",
                    "settled_because": str(human_note.get("note") or "") if settled_yes else "",
                    "direction_layer": next_layer,
                    "answer": human_note,
                },
            )
        accepted = bool(decision["accepted"])
        # The coach's own words are where the next round starts from, and a round that moved
        # nothing is what unlocks the expensive layer after two in a row.
        round_fault_stage = str((selected.get("judgement") or {}).get("fault_stage") or "").strip()
        if round_fault_stage:
            coach_fault_stage = round_fault_stage
        stalled_rounds = 0 if accepted else stalled_rounds + 1
        # The round's total, and each move's share of it. A round that ran two moves has to
        # answer "what did the coach's suggestion cost" without subtracting one action's number
        # from another's, so the parts are written beside the total rather than inferred.
        intermediate_measured = bool(move_records and move_records[0].get("measurement"))
        round_generations = sum(1 for item in move_records if item.get("upstream_retry_applied"))
        move_costs = []
        for item in move_records:
            is_last = item is move_records[-1]
            move_costs.append(
                move_cost_record(
                    index=int(item.get("index") or 0),
                    action=item.get("action"),
                    action_class=action_class(str(item.get("type") or "")),
                    source=str(item.get("source") or ""),
                    figure_generations=1 if item.get("upstream_retry_applied") else 0,
                    compile_calls=len(candidates) if is_last else (1 if item.get("measurement") else 0),
                    render_calls=len(candidates) if is_last else (1 if item.get("measurement") else 0),
                    judge_calls=(
                        (2 * len(candidates) + (1 if pairwise else 0) + (1 if round_index else 0))
                        if is_last
                        else (2 if item.get("measurement") else 0)
                    ),
                    stage_reruns=1 if item.get("upstream_retry_applied") else 0,
                    measured=bool(item.get("measurement")) or is_last,
                )
            )
        round_cost = write_round_cost(
            round_dir,
            round_index=round_index,
            action=current_action,
            action_class=action_class(current_action.get("type")),
            candidate_count=len(candidates),
            figure_generations=round_generations,
            compile_calls=len(candidates) + (1 if intermediate_measured else 0),
            render_calls=len(candidates) + (1 if intermediate_measured else 0),
            judge_calls=2 * len(candidates) + (1 if pairwise else 0) + (1 if round_index else 0) + (2 if intermediate_measured else 0),
            stage_reruns=round_generations,
            stage_seconds=stage_seconds_between(round_stage_clock, stage_seconds_from_progress(root)),
            elapsed_seconds=time.perf_counter() - round_started,
            moves=move_costs,
            model_calls=model_calls_from_progress(root)[round_call_clock:],
        )
        round_costs.append(round_cost)
        # What the round paid for and what it kept: the reuse counts of every paid icon, and how
        # many figure generations the round ran. Auxiliary numbers, not a gate -- they answer
        # "did a cheap move re-buy art it already had" without deciding anything.
        icon_reuse: dict[str, int] = {}
        icon_paid = 0
        for candidate in candidates:
            report = candidate.get("asset_report") if isinstance(candidate.get("asset_report"), dict) else {}
            for note_key, count in (report.get("reuse_summary") or {}).items():
                icon_reuse[str(note_key)] = icon_reuse.get(str(note_key), 0) + int(count or 0)
            icon_paid += int(report.get("api_requests_attempted") or 0)
        write_json(
            round_dir / "reuse_report.json",
            {
                "round": round_index,
                "action": str((current_action or {}).get("type") or ""),
                "action_class": action_class(current_action.get("type")),
                "assets_must_reuse": action_reuses_assets(current_action.get("type")),
                "icon_targets": sorted(icon_target_ids),
                "icon_reuse": icon_reuse,
                "icon_generations": icon_paid,
                "figure_generations": round_generations,
            },
        )
        if str(current_action.get("type") or "") in MECHANISM_ACTIONS:
            # A mechanism rule moves nothing where it runs, so its own execution report cannot
            # say whether the round was drawn by it. This reads the compile's own record of what
            # it applied together with the figure diff the round already computed, writes it into
            # the round, and puts it on the event stream: "this figure was drawn by the rule" is
            # otherwise unreadable from the run.
            mechanism_report = write_mechanism_rule_report(
                round_dir,
                round_index=round_index,
                action=current_action,
                candidate_dir=Path(selected["dir"]),
                figure_diff=selected.get("figure_diff"),
                round_cost=round_cost,
            )
            append_event(
                event_path,
                "mechanism_rule_applied" if mechanism_report["took_effect"] else "mechanism_rule_no_effect",
                round=round_index,
                rule_type=mechanism_report["rule_type"],
                rule=mechanism_report["rule"],
                source=mechanism_report["source"],
                compile_applied=len(mechanism_report["compile_applied"]),
                compile_unmet=len(mechanism_report["compile_unmet"]),
                figure_changed=mechanism_report["figure_changed"],
                changed_object_ids=mechanism_report["changed_object_ids"],
                generation_calls=mechanism_report["generation_calls"],
                stage_reruns=mechanism_report["stage_reruns"],
            )
        candidate_path = Path(selected["dir"])
        current_digests = {
            "scientific": graph_layer_digest(scientific_graph),
            "visual": graph_layer_digest(read_json(candidate_path / "visual_graph.json") if (candidate_path / "visual_graph.json").is_file() else None),
            "editable": graph_layer_digest(read_json(candidate_path / "editable_graph.json") if (candidate_path / "editable_graph.json").is_file() else None),
        }
        kept_score = measured_score(best_acceptance) if isinstance(best_acceptance, dict) else None
        layer_record = layer_change(
            best_layer_digests,
            current_digests,
            before_score=kept_score,
            after_score=score,
            accepted=accepted,
        )
        write_json(round_dir / "layer_change.json", layer_record)
        write_json(root / f"round_{round_index:02d}" / "repair_decision.json", decision)
        write_json(root / f"round_{round_index:02d}" / "reflection.json", reflection)
        # What this round measured, and what the reflection now claims. The same reading that
        # settles earlier rounds' claims is written beside them, so "the sentence was checked"
        # and "the sentence was believed" cannot be confused later.
        # The numbers a claim or an attribution may name: the gate's own defect counts and
        # dimension scores, plus the stage trace's before/after readings under their plain
        # names, which is how the seat spells the field it read. All of them are readings, so a
        # claim named against one is settled by a later reading and never by a model's prose.
        trace_metrics = {}
        if isinstance(stage_trace, dict):
            after_readings = (stage_trace.get("metrics") or {}).get("after")
            if isinstance(after_readings, dict):
                trace_metrics = {str(key): value for key, value in after_readings.items()}
        round_metrics = measured_metrics(selected.get("acceptance"), extra=trace_metrics)
        claim_sources = str(paper_id or "") or str(Path(paper).resolve())
        settled_claims = claim_ledger.settle_open(round_metrics, round_index=round_index, source_run=claim_sources) if claim_ledger is not None else []
        registered_claims = claim_ledger.register(reflection.get("claims"), round_index=round_index, paper=claim_sources) if claim_ledger is not None else []
        for outcome in settled_claims:
            append_event(
                event_path,
                "claim_settled",
                round=round_index,
                claim_id=str(outcome.get("claim_id") or ""),
                verdict="confirmed" if str(outcome.get("event")) == "claim_confirmed" else "refuted",
                metric=str(outcome.get("metric") or ""),
                observed=outcome.get("observed"),
                claim=str(outcome.get("claim") or ""),
            )
        if registered_claims:
            append_event(event_path, "claims_registered", round=round_index, claim_ids=[str(item.get("claim_id") or "") for item in registered_claims])
        write_json(
            round_dir / "claims.json",
            {
                "round": round_index,
                "metrics": round_metrics,
                "registered": [str(item.get("claim_id") or "") for item in (reflection.get("claims") or [])],
                "registered_now": [str(item.get("claim_id") or "") for item in registered_claims],
                "settled": settled_claims,
            },
        )
        with (root / "judge_experiences.jsonl").open("a", encoding="utf-8") as handle:
            # One row per round, and it describes the candidate the round decided on: the loop's
            # last candidate is not the chosen one whenever there were several, and a row that
            # describes a candidate nobody selected cannot be read back as this round's record.
            # The guidance the round ran under is an observation, not a target: the row records
            # it so a later training export can see which rules were in force.
            # The row carries the comparison as well as the two answers: the round was decided by
            # the comparison, and a row that only carries the scores cannot be read back as the
            # evidence the decision was made on.
            handle.write(json.dumps({"round": round_index, "candidate_index": int(selected.get("candidate_index", selected["variant"])), "role": getattr(online_judge, "role", "unknown"), "acceptance_role": getattr(frozen_judge, "role", "unknown"), "action": current_action, "judgement": selected["judgement"], "acceptance": frozen, "acceptance_pairwise": pairwise, "layer_change": layer_record, "frozen": frozen, "reflection": reflection, "applied_rules": sorted(applied_rule_ids), "applied_prompt_patches": sorted({patch_id for record in prompt_patch_stages.values() for patch_id in [str(item.get("patch_id") or "") for item in record["patches"]]})}, ensure_ascii=False) + "\n")
        decomposition_action = plan_decomposition_action(selected.get("decomposability_report"), evolution_scope=evolution_scope)
        write_json(round_dir / "decomposability_action.json", decomposition_action)
        violation_metrics = [str(item.get("metric") or "") for item in decomposition_action.get("violations") or []]
        if decomposition_action.get("escalation"):
            append_event(event_path, "decomposability_escalated", round=round_index, action=decomposition_action.get("action"), scope_type=decomposition_action.get("scope_type"), reason=decomposition_action.get("reason"), violations=violation_metrics)
        elif decomposition_action.get("violations"):
            append_event(event_path, "decomposability_blocked", round=round_index, action=decomposition_action.get("action"), scope_type=decomposition_action.get("scope_type"), reason=decomposition_action.get("reason"), violations=violation_metrics)
        # The loop's own reading of the figure travels into the next round the way the coach's
        # attribution does. A round that measured a break the coach did not name must not have to
        # wait for the coach to name it before the layer that answers it may run.
        measured_escalation = bool(decomposition_action.get("escalation"))
        # The figure measured against the paper rather than against its own contract. The report is
        # written beside the round because a judge trained on thin figures has to be able to read
        # that one was thin, and the reading travels into the next round the way the attribution
        # does. It gates nothing: the round it measures has already been accepted or rolled back.
        content_coverage = measure_content_coverage(
            paper_review=prepared.get("paper_review") or {},
            visual_graph=read_json(candidate_path / "visual_graph.json") if (candidate_path / "visual_graph.json").is_file() else {},
            reference_layout_template=reference_profile.get("reference_layout_template") or {},
        )
        write_json(round_dir / CONTENT_COVERAGE_FILE, content_coverage)
        append_event(
            event_path,
            "content_coverage_measured",
            round=round_index,
            status=str(content_coverage.get("status") or ""),
            declared_modules=int((content_coverage.get("paper_budget") or {}).get("declared", {}).get("modules") or 0),
            delivered_nodes=int((content_coverage.get("delivered") or {}).get("nodes") or 0),
            advisories=[str(item.get("code") or "") for item in content_coverage.get("advisories") or [] if isinstance(item, dict)],
        )
        content_gap = content_gap_signal(content_coverage)
        write_json(
            round_dir / STAGE_GATE_FILE,
            stage_gate_report(round_index=round_index, trace=stage_trace, coverage=content_coverage),
        )
        decomposability_history.append(selected.get("decomposability_report") or {})
        rounds.append(
            {
                "round": round_index,
                "candidate_count": len(candidates),
                "selected_variant": selected["variant"],
                "candidate_index": int(selected.get("candidate_index", selected["variant"])),
                # The advisory answer is what the loop acted on; the acceptance answer is the
                # measurement it decided with. They are recorded apart because they answer
                # different questions and only one of them is a number.
                "online_judgement": selected["judgement"],
                "acceptance_judgement": frozen,
                "acceptance_pairwise": pairwise,
                "frozen_judgement": frozen,
                "accepted": accepted,
                "repair_action": current_action,
                **ledger_attribution(selected.get("attribution")),
                "injected_experience": dict(injected_experience) if injected_experience.get("experience_id") else None,
                "repair_decision": decision,
                "reflection": reflection,
                "reflection_policy": dict(reflection_policy.get("policy") or {}),
                "bounded_change": selected.get("bounded_change"),
                "figure_object_diff": selected.get("figure_diff"),
                "layer_change": layer_record,
                "decomposability_report": selected.get("decomposability_report"),
                "decomposition_action": decomposition_action,
                "upstream_retry": _retry_summary(round_retry_report),
                "applied_prompt_patches": {
                    stage: [str(item.get("patch_id") or "") for item in record["patches"]]
                    for stage, record in sorted(prompt_patch_stages.items())
                },
                "prompt_revisions": {
                    stage: {
                        "base_revision": str(record.get("base_revision") or ""),
                        "revision": str(record.get("revision") or ""),
                        "trial": bool(record.get("trial")),
                    }
                    for stage, record in sorted(prompt_patch_stages.items())
                },
                "applied_rules": dict(applied_rules_by_stage),
                "prompt_patch_check": selected.get("prompt_patch_check"),
                # What this round measured, so the metrics of batch H are computed from the
                # readings themselves rather than from a summary of them.
                "round_metrics": round_metrics,
            }
        )
        if accepted:
            previous_acceptance = best_acceptance
            best_score, best_acceptance, best_judgement, best_compiled, current_layout = (
                score,
                frozen,
                selected["judgement"],
                selected["compiled"],
                selected["layout"],
            )
            best_acceptance_input = selected["acceptance_input"]
            best_grounding = selected.get("grounding_gate")
            best_layer_digests = dict(current_digests)
            # A score the deterministic gate produced is not a measurement by the scoring role, so
            # an accepted state carrying one is measured again straight away. Until that
            # measurement exists the next round has no baseline and says so, instead of
            # subtracting one instrument from another (which is what made a gate 0.75 look like a
            # regression against a model 0.40).
            baseline_source = "measured" if measured_score(frozen) is not None else "unmeasured"
            baseline_ready = measured_score(frozen) is not None
            if frozen.get("gate_only"):
                baseline_gate_score = frozen.get("overall_score")
                remeasured, remeasure_error = _measure_acceptance(
                    frozen_judge,
                    _acceptance_input(
                        method=method,
                        program=read_json(Path(selected["dir"]) / "figure_program.json"),
                        icons=selected["icon_plan"],
                        preview=Path(selected["dir"]) / "final_draft_preview.png" if (Path(selected["dir"]) / "final_draft_preview.png").exists() else Path(selected["dir"]) / "draft_preview.png",
                        domain_profile=resolved_domain,
                        grounding_gate=selected.get("grounding_gate"),
                        allowed_scope={"scope": "frozen_acceptance", "round": round_index},
                        layer_plan=layer_plan,
                    ),
                )
                if measured_score(remeasured) is None:
                    baseline_source = "gate_only_unmeasured"
                    baseline_ready = False
                    append_event(event_path, "baseline_remeasurement_failed", round=round_index, error=remeasure_error or "the answer carried no score")
                else:
                    baseline_source = "remeasured_after_gate_only"
                    baseline_ready = True
                    best_acceptance = remeasured
                    best_score = measured_score(remeasured)
                    append_event(event_path, "baseline_remeasured", round=round_index, gate_score=baseline_gate_score, score=best_score)
            append_event(event_path, "baseline_recorded", round=round_index, source=baseline_source, score=best_score, ready=baseline_ready)
            rounds[-1].update({"baseline_source": baseline_source, "baseline_ready": baseline_ready, "baseline_score": best_score})
            current_icon_plan = selected["icon_plan"]
            best_visual_review = selected.get("visual_review")
            best_visual_review_validation = selected.get("visual_review_validation")
            best_color_theme = selected.get("color_theme") or round_color_theme
            best_color_theme_report = selected.get("color_theme_report") or round_color_theme_report
            best_panel_style_plan = selected.get("panel_style_plan") or round_panel_style_plan
            # The accepted state's colour metrics become the "before" the next round's colour
            # evidence is reported against.
            baseline_color_metrics = dict(round_color_metrics["after"])
            best_dir = Path(selected["dir"])
            accepted_snapshots.append(
                {
                    "round": int(round_index) + 1,
                    "score": score,
                    "measured": measured_score(frozen),
                    "dir": best_dir,
                    "acceptance": frozen,
                    "judgement": selected["judgement"],
                    "compiled": selected["compiled"],
                    "layout": current_layout,
                    "acceptance_input": selected["acceptance_input"],
                    "grounding": selected.get("grounding_gate"),
                    "layer_digests": dict(current_digests),
                    "icon_plan": current_icon_plan,
                    "visual_review": best_visual_review,
                    "visual_review_validation": best_visual_review_validation,
                    "color_theme": best_color_theme,
                    "color_theme_report": best_color_theme_report,
                    "panel_style_plan": best_panel_style_plan,
                }
            )
            recorded_patch: dict[str, Any] = {}
            if store:
                # The Builder owns the rule that a row needs an earlier accepted state to be a
                # measurement, so a first accepted round is refused there rather than here.
                build = _measured_experience(
                    round_index=round_index,
                    candidate=selected,
                    previous=previous_acceptance,
                    current=frozen,
                    layout=current_layout,
                    action=(credited_move or current_action),
                    reflection=reflection,
                    paper=paper,
                    domain_profile=resolved_domain,
                    frozen_judge=frozen_judge,
                    judge_mode=judge_mode,
                    judge_model=judge_model,
                    frozen_judge_model=frozen_judge_model,
                    judge_checkpoint=judge_checkpoint,
                    color_metrics=round_color_metrics,
                    color_theme=best_color_theme or round_color_theme,
                    icon_mode=icon_mode,
                    round_decision=decision,
                )
                experience = build["experience"]
                if not experience:
                    _record_experience_build_refusal(event_path=event_path, round_dir=round_dir, round_index=round_index, build=build, action=(credited_move or current_action))
                elif not experience.get("accepted"):
                    write_json(round_dir / "experience_rejected.json", {"stage": "build", "round": round_index, "reason": "round_accepted_memory_not_credited", "decision_reason": str(decision.get("reason") or ""), "repair_gain": experience.get("repair_gain"), "comparison": experience.get("comparison"), "note": "The round kept the figure, but the Memory row did not agree it was an accepted measurement; the lifecycle is not overwritten."})
                    append_event(event_path, "experience_rejected", round=round_index, stage="build", reason="round_accepted_memory_not_credited", decision_reason=str(decision.get("reason") or ""))
                else:
                    library_row = _library_row_with_binding(experience, round_dir, default_action_fingerprint=_default_policy_fingerprint(decomposition_action))
                    stored = store.append_result(library_row, figure=current_layout)
                    if not stored["ok"]:
                        write_json(round_dir / "experience_rejected.json", {"stage": "store", "round": round_index, **stored, "experience": library_row})
                        append_event(event_path, "experience_rejected", round=round_index, stage="store", reason=stored["reason"], errors=stored.get("errors"))
                    else:
                        # The run keeps the whole measurement; the library keeps only the rule.
                        # A row folded into an older one keeps its measurement in its own round
                        # directory, and the event names the row the library now holds, so a
                        # reader following the id lands on a row that exists.
                        merged_into = str(stored.get("merged_into") or "")
                        recorded_id = str(stored.get("experience_id") or experience["experience_id"])
                        write_json(
                            round_dir / "experience_candidate.json",
                            {**experience, "merged_into": merged_into} if merged_into else experience,
                        )
                        append_event(event_path, "experience_recorded", round=round_index, experience_id=recorded_id, merged_into=merged_into or None)
                        recorded_patch = _record_prompt_patch(patch_store, round_dir=round_dir, round_index=round_index, experience=library_row, event_path=event_path, authored=reflection.get("prompt_patch"))
                        recorded_patch_id = str((recorded_patch or {}).get("patch_id") or "")
                        if recorded_patch_id:
                            own_case_patch_ids.add(recorded_patch_id)
                            rounds[-1].update({"prompt_patch_recorded": recorded_patch_id})
                            # The stage prompts this run already assembled are stale now, so the
                            # next time a stage is asked its prompt is assembled again and the
                            # patch rides in this same iteration.
                            prompt_patch_generation += 1
            # The round's advice lands in both tiers whether or not the library took a row: the
            # half that ran here is the round's own record, and the half that travels is what the
            # seat wrote for a later prompt. A round whose row was refused still answered the
            # question, and the answer is what a reader compares against what the round did.
            _record_suggestion_tiers(
                round_dir=round_dir,
                round_index=round_index,
                judgement=selected.get("judgement"),
                action=(credited_move or current_action),
                authored=reflection.get("prompt_patch"),
                patch=recorded_patch,
                executed=True,
            )
            append_event(event_path, "candidate_accepted", round=round_index, score=score, action=current_action)
            skipped_action_types.clear()
            if creator_ledger is not None:
                # A guided round that survived the Gate confirms its rules, which is what
                # makes the refutation count a streak rather than a lifetime tally.
                for rule_id in applied_rule_ids:
                    creator_ledger.record_confirmation(rule_id, round_index=round_index)
            # The same reading for the patches the prompts carried: a round that came out better
            # clears their streak, so what the limit counts is losses in a row.
            _charge_injected_patches(
                patch_store=patch_store,
                event_path=event_path,
                round_index=round_index,
                injected=round_injection_ids,
                reason=str(decision.get("reason") or ""),
            )
            write_json(round_dir / "accepted_state.json", selected["state_after"])
            action_retry_index = 0
            tried_rule_fingerprints.clear()
            tried_follow_ups.clear()
            # The iteration moved forward while these rules were under repair, so they are neither
            # disproved nor kept. Saying that out loud is cheaper than leaving a row that claims one.
            for _pending_fingerprint in list(pending_repair_rows):
                append_event(event_path, "rule_repair_dropped", round=round_index, rule_fingerprint=_pending_fingerprint, reason="the iteration moved forward before the rule's follow-ups were spent")
            pending_repair_rows.clear()
        else:
            # A rolled-back round leaves the baseline where it was, and says so rather than
            # leaving the record silent about what the next round will be compared against.
            rounds[-1].update({"baseline_source": "unchanged", "baseline_ready": baseline_ready})
            append_event(event_path, "candidate_rolled_back", round=round_index, score=score, action=current_action, reason=decision.get("reason"), reflection=reflection)
            # The patches this round's prompts carried are measured by this round. A round that
            # could not be measured charges nothing, so the count is losses and not noise.
            _charge_injected_patches(
                patch_store=patch_store,
                event_path=event_path,
                round_index=round_index,
                injected=round_injection_ids,
                reason=str(decision.get("reason") or ""),
            )
            # A rule whose attempts are all spent is written off here, with the row the round that
            # lost built and kept aside. Until then the row waits: a rule with an attempt still to
            # run has not been disproved, and the figure having rolled back does not change that.
            round_own_rule_fingerprint = mechanism_rule_fingerprint(current_action)
            rolling_rule_fingerprint = round_own_rule_fingerprint or follow_up_rule_fingerprint
            if (
                rolling_rule_fingerprint
                and int(tried_follow_ups.get(rolling_rule_fingerprint, 0)) >= FOLLOW_UP_ATTEMPTS
                and rolling_rule_fingerprint in pending_repair_rows
            ):
                written_off_row = pending_repair_rows.pop(rolling_rule_fingerprint)
                written_off = store.append_result(written_off_row, figure=current_layout) if store is not None else {"ok": False, "reason": "no_experience_store"}
                if written_off.get("ok"):
                    write_json(round_dir / "experience_written_off.json", written_off_row)
                    append_event(event_path, "rule_written_off", round=round_index, experience_id=str(written_off_row.get("experience_id") or ""), rule_fingerprint=rolling_rule_fingerprint, attempts=FOLLOW_UP_ATTEMPTS)
                else:
                    write_json(round_dir / "experience_rejected.json", {"stage": "write_off", "round": round_index, **written_off, "experience": written_off_row})
                    append_event(event_path, "experience_rejected", round=round_index, stage="write_off", reason=written_off.get("reason"))
            failed_type = str(current_action.get("type") or "")
            if failed_type in EXPERIENCE_ACTIONS:
                skipped_action_types = {failed_type}
            # The measurement that decided the round decides Memory too, and it decides it the
            # same way: a repair the Gate refused is still evidence about a condition, so it is
            # written down as a ``refuted`` row rather than dropped. Two measurements cannot be
            # recorded, though, because a row may not say the opposite of the round: a round with
            # no earlier accepted state has nothing to compare against, and a rolled-back round
            # whose score delta came out positive under a score comparison (the difference sat in
            # the acceptance dead band, the reflection called it noise, or it broke a prohibition)
            # can be neither validated nor recorded as refuted. Pairwise refusals may still write
            # a refuted row with a positive score delta, because the score was never the verdict.
            # Both score-path disagreements are reported as rejections rather than silently skipped.
            # ``action_retry_index`` is deliberately untouched: it counts the guidance this round
            # ran under, and a new row is not that guidance.
            recorded_patch = {}
            if store and str((credited_move or current_action).get("type") or "") in EXPERIENCE_ACTIONS:
                refused = _measured_experience(
                    round_index=round_index,
                    candidate=selected,
                    previous=best_acceptance,
                    current=frozen,
                    layout=current_layout,
                    action=(credited_move or current_action),
                    reflection=reflection,
                    paper=paper,
                    domain_profile=resolved_domain,
                    frozen_judge=frozen_judge,
                    judge_mode=judge_mode,
                    judge_model=judge_model,
                    frozen_judge_model=frozen_judge_model,
                    judge_checkpoint=judge_checkpoint,
                    color_metrics=round_color_metrics,
                    color_theme=best_color_theme or round_color_theme,
                    icon_mode=icon_mode,
                    round_decision=decision,
                )
                refuted = refused["experience"]
                if not refuted:
                    _record_experience_build_refusal(event_path=event_path, round_dir=round_dir, round_index=round_index, build=refused, action=current_action)
                elif bool(refuted.get("accepted")):
                    write_json(round_dir / "experience_rejected.json", {"stage": "build", "round": round_index, "reason": "measured_gain_positive_but_rolled_back", "decision_reason": str(decision.get("reason") or ""), "repair_gain": refuted.get("repair_gain"), "note": "A rolled-back round cannot validate an experience, and a refuted row cannot record a positive gain."})
                    append_event(event_path, "experience_rejected", round=round_index, stage="build", reason="measured_gain_positive_but_rolled_back", decision_reason=str(decision.get("reason") or ""))
                else:
                    library_row = _library_row_with_binding(refuted, round_dir, default_action_fingerprint=_default_policy_fingerprint(decomposition_action))
                    if round_own_rule_fingerprint and int(tried_follow_ups.get(round_own_rule_fingerprint, 0)) < FOLLOW_UP_ATTEMPTS:
                        # This round ran the rule itself and the rule has attempts left, so the row
                        # that says the rule is disproved waits. The figure still rolls back: that is
                        # the Gate, and nothing here loosens it.
                        stored = _defer_rule_write_off(
                            pending_repair_rows=pending_repair_rows,
                            round_dir=round_dir,
                            round_index=round_index,
                            rule_fingerprint=round_own_rule_fingerprint,
                            row=library_row,
                            attempts_spent=int(tried_follow_ups.get(round_own_rule_fingerprint, 0)),
                            decision_reason=str(decision.get("reason") or ""),
                        )
                        append_event(event_path, "experience_deferred_for_repair", round=round_index, rule_fingerprint=round_own_rule_fingerprint, attempts_spent=int(tried_follow_ups.get(round_own_rule_fingerprint, 0)), attempts_allowed=FOLLOW_UP_ATTEMPTS, decision_reason=str(decision.get("reason") or ""))
                    else:
                        stored = store.append_result(library_row, figure=current_layout)
                    if stored.get("deferred"):
                        pass
                    elif not stored["ok"]:
                        write_json(round_dir / "experience_rejected.json", {"stage": "store", "round": round_index, **stored, "experience": library_row})
                        append_event(event_path, "experience_rejected", round=round_index, stage="store", reason=stored["reason"], errors=stored.get("errors"))
                    else:
                        # The run keeps the whole measurement; the library keeps only the rule.
                        write_json(round_dir / "experience_refuted.json", refuted)
                        append_event(event_path, "experience_refuted", round=round_index, experience_id=refuted["experience_id"], gain=refuted.get("repair_gain"), excluded_conditions=refuted.get("excluded_conditions"))
                        recorded_patch = _record_prompt_patch(patch_store, round_dir=round_dir, round_index=round_index, experience=library_row, event_path=event_path, authored=reflection.get("prompt_patch"))
                        recorded_patch_id = str((recorded_patch or {}).get("patch_id") or "")
                        if recorded_patch_id:
                            own_case_patch_ids.add(recorded_patch_id)
                            rounds[-1].update({"prompt_patch_recorded": recorded_patch_id})
                            # The stage prompts this run already assembled are stale now, so the
                            # next time a stage is asked its prompt is assembled again and the
                            # patch rides in this same iteration.
                            prompt_patch_generation += 1
            # A refused round is evidence for a prohibition rather than for a repair, but it is
            # still advice in two tiers: what it would have done to this figure, and the rule the
            # seat says a later prompt should carry. Both are recorded here, next to the verdict.
            _record_suggestion_tiers(
                round_dir=round_dir,
                round_index=round_index,
                judgement=selected.get("judgement"),
                action=(credited_move or current_action),
                authored=reflection.get("prompt_patch"),
                patch=recorded_patch,
                executed=False,
                patch_reason=str(decision.get("reason") or ""),
            )
            # A rule that guided this round and did not survive the Gate loses standing: the
            # round counts against it, the rows behind it lose their validated status, and once
            # the streak reaches the limit the rule stops being injected at all. The loss is only
            # charged where it can be placed on one rule: the reason has to say the guided figure
            # itself came out worse, and the rule has to guide a stage this round worked at. Every
            # injected rule used to be charged for every rollback alike, so ten rules that shared a
            # run shared one number that described the run and none of them.
            if applied_rule_ids and creator_ledger is not None:
                action_type = str(current_action.get("type") or "")
                round_stages = {
                    str(stage)
                    for stage in (ACTION_TRACE_STAGE.get(action_type, ""), stage_for_action(action_type))
                    if str(stage or "").strip()
                }
                chargeable = {
                    rule_id: str((rule_sources.get(rule_id) or {}).get("apply_at") or "")
                    for rule_id in applied_rule_ids
                    if str((rule_sources.get(rule_id) or {}).get("class") or "") != "prohibited"
                }
                charged, spared = refuting_rule_ids(
                    rule_stages=chargeable,
                    reason=str(decision.get("reason") or ""),
                    round_stages=round_stages,
                )
                spared.extend(
                    {"rule_id": rule_id, "reason": "a prohibition is the boundary, so a rollback cannot refute it"}
                    for rule_id in applied_rule_ids
                    if rule_id not in chargeable
                )
                append_event(
                    event_path,
                    "creator_rule_refutation_scoped",
                    round=round_index,
                    action=action_type,
                    reason=str(decision.get("reason") or ""),
                    stage=sorted(round_stages),
                    refuted=list(charged),
                    spared=spared,
                )
                for rule_id in charged:
                    guidance = rule_sources.get(rule_id) or {}
                    source_ids = [*guidance.get("experiences", []), *guidance.get("patches", [])]
                    disabled = creator_ledger.refutation_count(rule_id) + 1 >= CREATOR_RULE_REFUTATION_LIMIT
                    record = creator_ledger.record_refutation(rule_id, round_index=round_index, reason=str(decision.get("reason") or ""), source_ids=source_ids, disabled=disabled)
                    if store is not None:
                        for experience_id in guidance.get("experiences") or []:
                            store.transition(str(experience_id), "refuted", reason="creator_rule_refuted", round=round_index)
                    if disabled and patch_store is not None:
                        for patch_id in guidance.get("patches") or []:
                            patch_store.disable(str(patch_id), reason="creator_rule_refutations_reached")
                    append_event(event_path, "creator_rule_refuted", round=round_index, rule_id=rule_id, refutations=int(record["refutations"]), disabled=bool(disabled), reason=str(decision.get("reason") or ""))
            restored_state, rollback_report = restore_state_snapshot(round_before_snapshot)
            # A refused round used to give everything back, so a move that measured better in one
            # place and worse in another lost both. The round is settled object by object here,
            # widest step first: an object that came out better is kept, an object that moved
            # without coming out worse is kept too, and only the objects that came out worse -- and
            # the objects the comparison blamed -- are restored from the snapshot. A step that
            # would leave the figure with more faults than it started with drops to the next one,
            # and the last step is a whole rollback, so a settlement never ships a figure worse
            # than the snapshot. When nothing is kept, the merge has to be the snapshot exactly.
            # The comparison's own answer about where the figure is worse is what steers it: the
            # defect it named is read as the objects to give back.
            blame_categories: list[str] = []
            if str(decision.get("pairwise_verdict") or "") == "before":
                pairwise_record = decision.get("pairwise") if isinstance(decision.get("pairwise"), dict) else {}
                named_defect = str(pairwise_record.get("defect") or (pairwise_record.get("pairwise") or {}).get("defect") or "").strip()
                if named_defect:
                    blame_categories = [named_defect]
            party_settlement = plan_partial_rollback(
                (round_before_snapshot or {}).get("state"),
                (selected.get("state_after") or {}).get("state"),
                round_index=round_index,
                action=(credited_move or current_action),
                blame_categories=blame_categories,
            )
            if party_settlement.get("applied") and isinstance(party_settlement.get("merged_state"), dict):
                restored_state = party_settlement["merged_state"]
                append_event(
                    event_path,
                    "partial_rollback_applied",
                    round=round_index,
                    tier=party_settlement.get("tier"),
                    blame=list(party_settlement.get("blame") or []),
                    kept=len(party_settlement.get("kept") or []),
                    added=sum(1 for item in party_settlement.get("additions") or [] if item.get("kept")),
                    reverted=len(party_settlement.get("reverted") or []),
                    before_faults=party_settlement.get("before_faults"),
                    merged_faults=party_settlement.get("merged_faults"),
                )
            else:
                append_event(
                    event_path,
                    "partial_rollback_skipped",
                    round=round_index,
                    tier=party_settlement.get("tier"),
                    merge_unfaithful=bool(party_settlement.get("merge_unfaithful")),
                    reason=str(party_settlement.get("reason") or ""),
                    overlap_ratio=party_settlement.get("overlap_ratio"),
                )
            write_json(
                round_dir / "partial_rollback.json",
                {key: value for key, value in party_settlement.items() if key != "merged_state"},
            )
            for orphan in party_settlement.get("orphaned") or []:
                append_event(
                    event_path,
                    "object_orphaned",
                    round=round_index,
                    object_id=str(orphan.get("object_id") or ""),
                    kind=str(orphan.get("kind") or ""),
                    panel_id=str(orphan.get("panel_id") or ""),
                    settled=str(orphan.get("settled") or ""),
                )
            for added in party_settlement.get("additions") or []:
                append_event(
                    event_path,
                    "object_added",
                    round=round_index,
                    object_id=str(added.get("object_id") or ""),
                    kind=str(added.get("kind") or ""),
                    kept=bool(added.get("kept")),
                )
            method = restored_state["method"]
            current_layout = restored_state["layout"]
            current_icon_plan = restored_state["icons"]
            write_json(round_dir / "rollback_report.json", rollback_report)
            write_json(round_dir / "rolled_back_state.json", create_state_snapshot(restored_state, label="rolled_back", round_index=round_index))
            action_retry_index += 1
            if action_retry_index >= max(1, int(max_retries_per_experience)):
                if store and current_action.get("experience_id"):
                    append_event(event_path, "experience_failed", round=round_index, experience_id=str(current_action["experience_id"]), reason=str(decision.get("reason") or ""), status="candidate")
                break
        progress(
            {
                "stage": "round",
                "status": "completed",
                "round": round_index,
                "accepted": accepted,
                "score": score,
            }
        )
        if _evolution_should_stop(
            evolution_enabled=evolution_enabled,
            fixed_rounds=bool(fixed_rounds),
            accepted=bool(accepted),
            judgement=frozen,
            pending_seed_action=pending_seed_action,
        ):
            break
        if pending_seed_action:
            current_action = pending_seed_action
            seed_id = str(current_action.get("experience_id") or "")
            seeded_row = store.get(seed_id) if store is not None and seed_id else None
            injected_experience = seed_injection(current_action, seed_id, status=str((seeded_row or {}).get("status") or ""))
            pending_round_moves = [dict(current_action)]
            pending_deferred_move = {}
        else:
            # A reflection that asked to revise this round's action has already been checked
            # against the actions the loop may run, so its revision is the next proposal. Soft
            # occupancy used to win every later round because the catalog nudge made it
            # executable; hard breaks still go first, then the rule the same reflection stated,
            # then a measured color repair, then the Judge, then soft decomposability as a
            # fallback. A rule gets one turn per iteration, counted here rather than in the
            # selector, because only this loop knows which iteration it is in.
            rule_action, rule_fingerprint = next_rule_turn(
                reflection_policy.get("revised_rule_action"), tried=tried_rule_fingerprints
            )
            follow_up_action, follow_up_key = next_follow_up_turn(
                reflection_policy.get("follow_up_action"),
                rule_fingerprint=follow_up_rule_fingerprint,
                tried=tried_follow_ups,
            )
            ordered_proposals = round_proposal_order(
                follow_up_action=follow_up_action,
                revised_action=dict(reflection_policy.get("revised_action") or {}),
                decomposition_action=decomposition_action,
                decomposition_proposal=_decomposition_proposal(decomposition_action),
                rule_action=rule_action,
                judgement=best_judgement,
                pending_color_feedback=pending_color_feedback,
                skipped_action_types=skipped_action_types,
            )
            proposal = dict(ordered_proposals[0]) if ordered_proposals else {}
            current_action, injected_experience, routing_report = _next_action_with_memory(retrieved, proposal, current_layout, current_icon_plan)
            if pending_deferred_move and not injected_experience.get("experience_id"):
                # A move a guard deferred out of the previous round runs first, and the
                # mechanism's move keeps the second slot rather than taking the first: the
                # deferral exists precisely so this move is not starved a second time.
                mechanism_move = next(
                    (item for item in ordered_proposals if is_mechanism_move(item, mechanism_sources=MECHANISM_POLICY_SOURCES)),
                    {},
                )
                pending_round_moves = [dict(pending_deferred_move)]
                if mechanism_move and action_execution_fingerprint(mechanism_move) != action_execution_fingerprint(pending_deferred_move):
                    pending_round_moves.append(dict(mechanism_move))
                current_action = dict(pending_deferred_move)
            elif injected_experience.get("experience_id"):
                # A retrieved experience is what this round is about; a second move would be
                # measured together with it and the pairing would name the wrong action.
                pending_round_moves = [dict(current_action)]
            else:
                pending_round_moves = plan_round_moves(
                    ordered_proposals, current_action, mechanism_sources=MECHANISM_POLICY_SOURCES
                )
            pending_deferred_move = {}
            if follow_up_key and action_execution_fingerprint(current_action) == follow_up_key:
                # The attempt is spent where the round is known to be running it rather than where
                # it was offered: a repair Memory displaced was never tried, so it is not spent.
                tried_follow_ups[follow_up_rule_fingerprint] = (
                    int(tried_follow_ups.get(follow_up_rule_fingerprint, 0)) + 1
                )
            if rule_fingerprint and mechanism_rule_fingerprint(current_action) == rule_fingerprint:
                # The rule has had its turn. Memory may still have routed something else, so the
                # turn is only spent when the rule is what this round is about to run.
                tried_rule_fingerprints.add(rule_fingerprint)
            routing_reports.append({"round": round_index, **routing_report})
            if not action_is_executable(current_action):
                append_event(
                    event_path,
                    "repair_refused",
                    round=round_index + 1,
                    reason="no_executable_repair",
                    proposal=proposal,
                    injection=dict(injected_experience),
                )
                break
        pending_seed_action = {}
        if injected_experience.get("experience_id"):
            append_event(event_path, "experience_injected", round=round_index + 1, **injected_experience)
        if evolution_scope == "block" and current_action.get("scope_type") == "global":
            slot_ids = [str(item.get("asset_id")) for item in current_icon_plan.get("icons", [])[:1] if item.get("asset_id")]
            current_action = {"type": "regen_assets", "scope_type": "slot", "slot_ids": slot_ids, "problem": "Block-only evolution requested; improve the weakest available local visual asset."}
        elif evolution_scope == "global" and current_action.get("scope_type") in {"slot", "interface", "connector"}:
            current_action = {"type": "relayout_global", "scope_type": "global", "problem": "Global-only evolution requested; improve overall PPT structure without changing scientific content."}
    # The accepted round with the highest measured score is the run's deliverable. Ties go to
    # the later round, and a run whose accepted rounds carry no measurement keeps the last one.
    delivered_round = int(accepted_snapshots[-1]["round"]) if accepted_snapshots else 0
    chosen_snapshot = accepted_snapshots[-1] if accepted_snapshots else None
    scored_snapshots = [item for item in accepted_snapshots if isinstance(item.get("score"), (int, float))]
    if scored_snapshots:
        chosen_snapshot = max(scored_snapshots, key=lambda item: (float(item["score"]), int(item["round"])))
    if chosen_snapshot is not None and accepted_snapshots and chosen_snapshot is not accepted_snapshots[-1]:
        best_score = chosen_snapshot["score"]
        best_acceptance = chosen_snapshot["acceptance"]
        best_judgement = chosen_snapshot["judgement"]
        best_compiled = chosen_snapshot["compiled"]
        current_layout = chosen_snapshot["layout"]
        best_acceptance_input = chosen_snapshot["acceptance_input"]
        best_grounding = chosen_snapshot["grounding"]
        best_layer_digests = chosen_snapshot["layer_digests"]
        best_dir = chosen_snapshot["dir"]
        current_icon_plan = chosen_snapshot["icon_plan"]
        best_visual_review = chosen_snapshot["visual_review"]
        best_visual_review_validation = chosen_snapshot["visual_review_validation"]
        best_color_theme = chosen_snapshot["color_theme"]
        best_color_theme_report = chosen_snapshot["color_theme_report"]
        best_panel_style_plan = chosen_snapshot["panel_style_plan"]
        append_event(
            event_path,
            "best_round_delivered",
            round=delivered_round,
            score=best_score,
            reason="highest measured score among the accepted rounds",
        )
    if chosen_snapshot is not None:
        delivered_round = int(chosen_snapshot["round"])
    # Set before the delivery branch so the report can always read it: a run that
    # compiled nothing still has to say which figure it was aimed at.
    figure_target_report = None
    if best_compiled and best_dir:
        for name in (
            "editable_composition.pptx",
            "figure_program.json",
            "asset_generation_report.json",
            "asset_quality_report.json",
            "ppt_composition_plan.json",
            "ppt_composition_plan_replanned.json",
            "visual_replan_validation.json",
            "ppt_composition_validation.json",
            "draft_preview.png",
            "final_draft_preview.png",
            "preview_render_report.json",
            "preview.png",
            "visual_review_00.json",
            "visual_review_00_validation.json",
            "visual_review_01.json",
            "visual_review_01_validation.json",
            "visual_repair_report_00.json",
            "connector_visibility_report.json",
            "connector_review.json",
            "connector_review_validation.json",
            "connector_route_selection_report.json",
            "typography_plan.json",
            "geometry_quality_report.json",
            "density_quality_report.json",
            "text_readability_report.json",
            "color_quality_report.json",
            "color_review.json",
            "color_review_validation.json",
            "color_repair_status.json",
            "visual_graph.json",
            "visual_graph_report.json",
            "editable_graph.json",
            "editable_graph_report.json",
            "graph_bindings.json",
            "graph_alignment_report.json",
            "grounding_gate.json",
        ):
            source = best_dir / name
            if source.exists():
                shutil.copyfile(source, root / name)
        if (best_dir / "assets").exists():
            shutil.copytree(best_dir / "assets", root / "assets", dirs_exist_ok=True)
        write_json(root / "color_theme.json", best_color_theme)
        write_json(root / "color_theme_validation.json", best_color_theme_report)
        best_compiled = {
            **best_compiled,
            "pptx": str(root / "editable_composition.pptx"),
            "figure_program": str(root / "figure_program.json"),
        }
        preview = _preview_from_deck(root / "figure_program.json", root, root / "preview.png")
        best_compiled["preview"] = str(preview)
        final_program = read_json(root / "figure_program.json")
        write_json(
            root / "color_experience_usage_report.json",
            evaluate_color_experience_usage(
                best_color_theme,
                final_program,
                supplied_recipe_ids=best_color_theme_report.get("supplied_recipe_ids") or [],
            ),
        )
        final_layout = {"panels": final_program.get("panels", []), "nodes": final_program.get("semantic_nodes", []), "connectors": final_program.get("arrows", [])}
        write_json(root / "figure_layout_plan_final.json", final_layout)
        # The delivered figure measured against the figure the paper printed. It is a
        # report and never a gate: a benchmark run is aimed at the paper's panels, and
        # the coverage it reached is a result rather than a pass or a fail.
        figure_target_report = {
            "figure_target": figure_target,
            "panel_target": {
                key: panel_target.get(key)
                for key in ("available", "source", "reason", "panel_count", "label_count")
            },
            "coverage": check_design_against_panel_target(final_layout, panel_target),
        }
        write_json(root / "figure_target_report.json", figure_target_report)
        final_composition = read_json(best_dir / "ppt_composition_plan_replanned.json") if (best_dir / "ppt_composition_plan_replanned.json").exists() else read_json(best_dir / "ppt_composition_plan.json")
        write_json(root / "ppt_composition_plan_final.json", final_composition)
        write_json(
            root / "rendered_density_report.json",
            measure_rendered_density(
                root / "final_draft_preview.png"
                if (root / "final_draft_preview.png").exists()
                else root / "draft_preview.png"
            ),
        )
        final_color_report = "color_quality_report.json"
        quality_report_names = ("geometry_quality_report.json", "density_quality_report.json", "text_readability_report.json", "rendered_density_report.json", final_color_report)
        quality_reports = {name: read_json(root / name) for name in quality_report_names if (root / name).exists()}
        delivery = classify_delivery_status(
            compiled=best_compiled,
            root=root,
            method_validation=method_validation,
            structure_validation=structure.get("validation"),
            asset_report=read_json(root / "asset_generation_report.json") if (root / "asset_generation_report.json").exists() else None,
            quality_reports=quality_reports,
            visual_review=best_visual_review,
        )
        final_grounding_gate = read_json(root / "grounding_gate.json")
        grounding_passed = final_grounding_gate.get("status") == "pass"
        final_quality_gate = {
            "status": "pass" if delivery["quality_passed"] and grounding_passed else "blocked_grounding" if not grounding_passed else delivery["quality_status"],
            "reports": quality_reports,
            "visual_review_status": (best_visual_review or {}).get("status"),
            "grounding_gate_status": final_grounding_gate.get("status"),
            "grounding_passed": grounding_passed,
            "provenance": "vlm_replan" if (best_dir / "ppt_composition_plan_replanned.json").exists() else "initial_or_local_repair",
            **delivery,
        }
        final_quality_gate["ok"] = bool(final_quality_gate.get("ok") and grounding_passed)
        final_quality_gate["quality_passed"] = bool(final_quality_gate.get("quality_passed") and grounding_passed)
        write_json(root / "final_quality_gate.json", final_quality_gate)
    else:
        final_quality_gate = classify_delivery_status(
            compiled=None,
            root=root,
            method_validation=method_validation,
            structure_validation=structure.get("validation"),
        )
        final_quality_gate = {"status": "failed_generation", "reports": {}, "visual_review_status": None, "provenance": "no_deliverable_candidate", **final_quality_gate}
        write_json(root / "final_quality_gate.json", final_quality_gate)
    # ``judgement.json`` keeps its place as the run's headline answer, and the headline answer is
    # the measurement: a run that reported the advisory answer here would publish a record with no
    # score in it, which every consumer of this file reads as "measured nothing".
    write_json(root / "judgement.json", best_acceptance or {})
    write_json(root / "structure_judgement.json", best_acceptance or {})
    injected_experience_ids = sorted({str((item.get("injected_experience") or {}).get("experience_id") or "") for item in rounds if isinstance(item, dict)} - {""})
    decomposability_rounds = [item.get("decomposability_report") or {} for item in rounds if isinstance(item, dict)]
    decomposability_final = decomposability_rounds[-1] if decomposability_rounds else {}
    decomposability_escalations = [item.get("decomposition_action") or {} for item in rounds if isinstance(item, dict)]
    decomposability_escalations = [item for item in decomposability_escalations if item.get("escalation")]
    prompt_patch_manifest = {
        "contract": PROMPT_PATCH_CONTRACT,
        "version": PROMPT_PATCH_VERSION,
        "order": list(PROMPT_ASSEMBLY_ORDER),
        "library": str(patch_store.root) if patch_store else None,
        "seed_patch_id": str((seeded_prompt_patch or {}).get("patch_id") or "") or None,
        "segmentation_parameters": dict(segmentation_parameters),
        "decomposability_thresholds": dict(decomposability_limits),
        "stages": {
            stage: {key: value for key, value in record.items() if key != "patches"}
            for stage, record in sorted(prompt_patch_stages.items())
        },
        "applied": {
            stage: [str(item.get("patch_id") or "") for item in record["patches"]]
            for stage, record in sorted(prompt_patch_stages.items())
        },
        # Which version of each stage's prompt this run resolved, so the report answers "was this
        # the shipped prompt or a candidate" without reading the patch library. A candidate
        # (``trial``) is a patch that is only measured on this paper so far.
        "revisions": {
            stage: {
                "base_revision": str(record.get("base_revision") or ""),
                "revision": str(record.get("revision") or ""),
                "trial": bool(record.get("trial")),
            }
            for stage, record in sorted(prompt_patch_stages.items())
        },
    }
    write_json(
        root / "prompt_patch_timeline.json",
        {
            "contract": PROMPT_PATCH_CONTRACT,
            "version": PROMPT_PATCH_VERSION,
            "timeline": prompt_patch_timeline,
        },
    )
    creator_rule_refutations = creator_ledger.counters() if creator_ledger is not None else {}
    creator_rules_report = {
        "contract": CREATOR_RULE_MANIFEST_CONTRACT,
        "version": CREATOR_RULE_MANIFEST_VERSION,
        # The authored standard and the earned rules are reported apart, because one is an idea
        # of "good" and the other is a measurement, and only the second may be called evidence.
        "curated": {
            "source": CREATOR_RULE_CURATED_SOURCE,
            "rule_ids": [str(rule.get("rule_id") or "") for rule in curated_manifest.get("rules") or []],
            "count": len(curated_manifest.get("rules") or []),
        },
        "rules": creator_rules_manifest.get("rules") or [],
        "refuted_rules": creator_rules_manifest.get("refuted_rules") or [],
        "buckets": creator_rules_manifest.get("buckets") or {},
        "withheld": creator_rules_manifest.get("withheld") or [],
        "min_support": creator_rules_manifest.get("min_support"),
        "refutation_limit": CREATOR_RULE_REFUTATION_LIMIT,
        "refutations": dict(creator_rule_refutations),
        "injected": dict(applied_rules_by_stage),
    }
    write_json(root / "creator_rules.json", creator_rules_report)
    claim_report = claim_ledger.report() if claim_ledger is not None else {"contract": "rfs.claim_ledger", "version": "", "claims": [], "counts": {}}
    write_json(root / "claim_ledger.json", claim_report)
    # The five numbers that tell self-evolution from iterative repair. They are computed from
    # the recordings above, written beside the run, and appended to one ledger next to the
    # library so a trend across iterations can be read without opening every run.
    metrics_report = evolution_metrics(
        rounds=rounds,
        library_rows=(store.records_with_status() if store is not None else []),
        claim_report=claim_report,
        mechanism_sources=MECHANISM_POLICY_SOURCES,
    )
    write_json(root / "evolution_metrics.json", metrics_report)
    if experience_enabled:
        try:
            append_jsonl(
                patch_library.parent / "evolution_metrics.jsonl",
                {
                    "paper": str(paper_id or ""),
                    "run": str(root),
                    "headline": dict(metrics_report.get("headline") or {}),
                    "counts": {
                        "location": dict(metrics_report.get("location") or {}),
                        "advice": dict(metrics_report.get("advice") or {}),
                        "portability": dict(metrics_report.get("portability") or {}),
                        "claims": dict(metrics_report.get("claims") or {}),
                        "recurrence": dict(metrics_report.get("recurrence") or {}),
                    },
                },
            )
        except Exception:  # noqa: BLE001 - the trend line must never stop a run that finished
            pass
    # The trace answers the audit question the other reports leave open: for each round and
    # each stage, which Memory, which patch and which rule were used, and which lost.
    memory_application_trace = build_memory_application_trace(
        rounds=rounds,
        retrieval_reports=retrieval_reports,
        routing_reports=routing_reports,
        stage_patches=prompt_patch_stages,
        creator_rules=creator_rules_report,
        experience_library=str(store.root) if store else "",
        patch_library=str(patch_store.root) if patch_store else "",
    )
    write_json(root / "memory_application_trace.json", memory_application_trace)
    write_json(root / "evolution_rounds.json", {"rounds": rounds})
    write_json(root / "evolution_state.json", {"status": "completed", "best_score": best_score, "delivered_round": delivered_round, "delivered_by": "highest measured score among the accepted rounds", "accepted_rounds": [{"round": item["round"], "score": item["score"], "measured": item["measured"], "dir": str(item["dir"])} for item in accepted_snapshots], "rounds_completed": len(rounds), "current_action": current_action, "domain_profile": resolved_domain, "best_candidate_dir": str(best_dir) if best_dir else None, "experience_retrieval": retrieval_report, "experience_routing": routing_report, "injected_experience_ids": injected_experience_ids, "decomposability_report": decomposability_final, "decomposition_action": decomposition_action, "prompt_patch_manifest": prompt_patch_manifest, "creator_rules": creator_rules_report, "memory_application_trace": memory_application_trace})
    report = {
        "summary": "Direct paper-to-PPTX Judge evolution run.",
        "ok": bool(final_quality_gate.get("ok")),
        "generation_status": final_quality_gate["generation_status"],
        "quality_status": final_quality_gate["quality_status"],
        "delivery_status": final_quality_gate["delivery_status"],
        "quality_passed": final_quality_gate["quality_passed"],
        "repair_recommended": final_quality_gate["repair_recommended"],
        "quality_issues": final_quality_gate["quality_issues"],
        "blocking_generation_errors": final_quality_gate["blocking_generation_errors"],
        "out_dir": str(root),
        "reference": str(Path(reference).resolve()) if reference else None,
        "stage_replay": stage_replay_report or None,
        "upstream_contract": upstream_report,
        "bulk_figures": bool(bulk_figures),
        "paper_contract_gaps": paper_contract_gaps,
        "paper_contract_gap_count": len(paper_contract_gaps),
        "evolution_enabled": evolution_enabled,
        "middle_layer": middle_layer,
        "scientific_hierarchy": "planned" if evidence_structure else "skipped",
        # A row that never built the middle layer has no graph to read, so the
        # switch is reported as inapplicable rather than as an "off" that would
        # read like the weaker graph-read ablation this row is not.
        "graph_context": ("on" if graph_context else "off") if evidence_structure else "not_applicable",
        "scientific_graph_provenance": "planned" if evidence_structure else "post_hoc_measurement",
        "figure_target": figure_target,
        "panel_target": {
            key: panel_target.get(key)
            for key in ("available", "source", "reason", "panel_count", "label_count")
        },
        "panel_target_coverage": (figure_target_report.get("coverage") if figure_target_report else None),
        "evolution_status": (
            "disabled_single_run" if not evolution_enabled else "completed"
        ),
        "rounds_completed": len(rounds),
        "max_rounds": max(1, int(max_rounds)),
        "max_retries_per_experience": max(1, int(max_retries_per_experience)),
        "evolution_scope": evolution_scope,
        "candidates_per_round": max(1, int(candidates_per_round)),
        "judge_role": (best_judgement or {}).get("judge_role", "bootstrap"),
        "acceptance_judge_role": (best_acceptance or {}).get("judge_role", "bootstrap"),
        "trained_judge": _trained_judge_flag(best_judgement, best_acceptance),
        "icon_mode": icon_mode,
        "icon_candidates": icon_candidates,
        "judge_mode": judge_mode,
        "judge_binding": judge_binding,
        "online_judge_role": getattr(online_judge, "role", "unknown"),
        "frozen_judge_role": getattr(frozen_judge, "role", "unknown"),
        "domain_profile": resolved_domain,
        "experience_enabled": experience_enabled,
        "retrieval_statuses": sorted(retrieval_bar) if retrieval_bar else [],
        "experience_library": str(store.root) if store else None,
        "suggestion_rule_library": str(suggestion_store.root) if suggestion_store else None,
        "seed_experience_id": str((seed_experience or {}).get("experience_id") or "") or None,
        "run_seed": int(run_seed),
        "retrieved_strategy_count": len(retrieved),
        "retrieved_strategy_ids": [row.get("evidence", {}).get("experience_id") for row in retrieved],
        "injected_experience_ids": injected_experience_ids,
        "experience_retrieval": retrieval_report,
        "experience_routing": routing_report,
        "decomposability_report": decomposability_final,
        "decomposition_action": decomposition_action,
        "decomposability_escalation_count": len(decomposability_escalations),
        "prompt_patch_manifest": prompt_patch_manifest,
        "prompt_patch_timeline": prompt_patch_timeline,
        "creator_rules": creator_rules_report,
        "claim_ledger": claim_report,
        "evolution_metrics": metrics_report,
        "evolution_metrics_headline": dict(metrics_report.get("headline") or {}),
        "memory_application_trace": memory_application_trace,
        "method_validation": method_validation,
        "structure_validation": structure["validation"],
        "reference_analysis_status": reference_profile.get("status"),
        "visual_review_mode": visual_review_mode,
        "visual_review_status": (best_visual_review or {}).get("status") if best_compiled else None,
        "visual_review_validation": best_visual_review_validation if best_compiled else None,
        # ``judgement`` is the measurement of the accepted state; the advisory answer that drove
        # the loop is reported beside it under its own name, never merged into it.
        "judgement": best_acceptance,
        "acceptance_judgement": best_acceptance,
        "advisory_judgement": best_judgement,
        "rounds": rounds,
        "note": "evolution disabled for this run" if not evolution_enabled else None,
        **(best_compiled or {}),
    }
    report.update(
        {
            "structure_mode": structure_mode,
            "structure_report": design_report,
            "text_plan_mode": text_plan_mode,
            "text_plan_report": text_report,
        "ppt_plan_mode": ppt_plan_mode,
        "color_theme_mode": color_theme_mode,
        "color_theme_source": str((best_color_theme.get("provenance") or {}).get("source") or "unknown"),
        "color_theme_fallback_used": bool(best_color_theme_report.get("fallback_used")),
        "color_theme": best_color_theme,
            "exemplar_guidance": exemplar_guidance,
            "reference_library": str(library) if exemplar_guidance and library.exists() else None,
            "exemplar_selection": exemplar_selection,
        }
    )
    report["cost"] = write_run_cost_report(root, elapsed_seconds=time.perf_counter() - run_started)
    report["round_cost"] = summarize_round_costs(round_costs)
    write_json(root / "paper_evolve_editable_report.json", report)
    progress(
        {
            "stage": "workflow",
            "status": "completed",
            "message": str(root / "editable_composition.pptx"),
            "ok": report["ok"],
        }
    )
    return report
