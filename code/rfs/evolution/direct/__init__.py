"""Experience helpers for direct paper-to-PPTX evolution."""

from .action_policy import ACTION_CLASSES, CHEAP_LAYERS, CLASS_ORDER, COMPILE_RULE_CLASS, DECOMPOSITION_POLICY_SOURCE, FIGURE_BELOW_PAPER_BUDGET, FIGURE_PROGRAM_CLASS, LAYER_COSTS, PAPER_BUDGET_THIN, RESTART_FROM_ARCHIVE, STALLED_ROUNDS_BEFORE_ESCALATION, STAGE_RERUN_CLASS, action_class, action_class_policy, action_reuses_assets, content_gap_signal, measurement_request
from .experience import build_direct_experience, build_direct_experience_result, count_bucket, exclusion_context, normalize_experience_scope, resolved_applicability, stage_for_action
from .domain import resolve_domain_profile
from .controller import ACTION_BUDGET_KEYS, FOLLOW_UP_ATTEMPTS, action_is_executable, aliased_action_name, apply_bounded_action, apply_reflection_policy, append_event, color_feedback_proposal, compare_candidate, default_reflection, derived_score, measured_score, mechanism_rule_fingerprint, next_follow_up_turn, next_rule_turn, select_bounded_action, select_next_round_proposal, round_proposal_order, action_execution_fingerprint, validate_reflection_rule, verify_attribution
from .validation import ValidationCase, paired_configuration_hash, validate_experience_cases
from .store import ExperienceStore, resolve_retrieval_statuses
from .snapshots import create_state_snapshot, diff_figure_programs, diff_state_snapshots, restore_state_snapshot, skeleton_state_hash, state_hash, verify_state_snapshot
from .routing import inject_experience, route_by_stage, route_experiences, seed_injection
from .prompt_patch import (
    CURRENT_CASE_STATUS,
    OWN_CASE_ASSEMBLY_STATUSES,
    PROMPT_PATCH_REFUTATION_LIMIT,
    PromptPatchStore,
    assemble_prompt_patches,
    build_patches_for_experience,
    build_prompt_patch,
    patch_set_revision,
    patch_sentence_key,
    patch_work_permit,
    merge_patch_sentences,
    prompt_revision_record,
    select_own_case_candidates,
)
from .claims import CLAIM_LEDGER_FILE, CLAIM_SETTLEMENT_LIMIT, ClaimLedger, measured_metrics
from .partial_rollback import (
    OBJECT_OVERLAP_FLOOR,
    PARTIAL_ROLLBACK_CONTRACT,
    object_readings,
    plan_partial_rollback,
    validate_partial_rollback,
)
from .metrics import ROUND_METRICS_CONTRACT, ROUND_METRICS_VERSION, METRICS_CONTRACT, METRICS_VERSION, evolution_metrics, layer_walk, reflection_yield, round_metrics
from .suggestion_rules import SuggestionRuleStore
from .creator_rules import CreatorRuleLedger, build_creator_rule_manifest, read_creator_rule_manifest, refuting_rule_ids
from .human_preference import (
    DEFAULT_REVIEW_KIND,
    HUMAN_PREFERENCE_CONTRACT,
    HUMAN_PREFERENCE_ENTRY_CONTRACT,
    HUMAN_PREFERENCE_ENTRY_VERSION,
    HUMAN_PREFERENCE_VERSION,
    HUMAN_REVIEW_DIR,
    PREFERENCE_STATUSES,
    REVIEW_KINDS,
    HumanReviewStore,
    collect_review_items,
    export_choice_items,
    human_review_stats,
    preference_entry,
    review_kind,
    unconsumed_verdicts,
    validate_human_review,
    validate_preference_entry,
    write_review_bundle,
)
from .trace import build_memory_application_trace

__all__ = ["CLAIM_LEDGER_FILE", "CLAIM_SETTLEMENT_LIMIT", "ClaimLedger", "OBJECT_OVERLAP_FLOOR", "PARTIAL_ROLLBACK_CONTRACT", "object_readings", "plan_partial_rollback", "validate_partial_rollback", "METRICS_CONTRACT", "METRICS_VERSION", "evolution_metrics", "measured_metrics", "CreatorRuleLedger", "ExperienceStore", "PromptPatchStore", "SuggestionRuleStore", "patch_set_revision", "prompt_revision_record", "ValidationCase", "ACTION_BUDGET_KEYS", "ACTION_CLASSES", "CHEAP_LAYERS", "CLASS_ORDER", "action_reuses_assets", "DECOMPOSITION_POLICY_SOURCE", "FIGURE_BELOW_PAPER_BUDGET", "FIGURE_PROGRAM_CLASS", "LAYER_COSTS", "FOLLOW_UP_ATTEMPTS", "PAPER_BUDGET_THIN", "RESTART_FROM_ARCHIVE", "STALLED_ROUNDS_BEFORE_ESCALATION", "action_class", "action_class_policy", "action_execution_fingerprint", "content_gap_signal", "measurement_request", "COMPILE_RULE_CLASS", "FIGURE_PROGRAM_CLASS", "STAGE_RERUN_CLASS", "action_is_executable", "aliased_action_name", "apply_bounded_action", "apply_reflection_policy", "append_event", "assemble_prompt_patches", "build_creator_rule_manifest", "build_direct_experience", "build_direct_experience_result", "build_memory_application_trace", "build_patches_for_experience", "build_prompt_patch", "patch_work_permit", "patch_sentence_key", "merge_patch_sentences", "PROMPT_PATCH_REFUTATION_LIMIT", "select_own_case_candidates", "CURRENT_CASE_STATUS", "OWN_CASE_ASSEMBLY_STATUSES", "color_feedback_proposal", "compare_candidate", "count_bucket", "create_state_snapshot", "default_reflection", "diff_figure_programs", "diff_state_snapshots", "exclusion_context", "inject_experience", "derived_score", "measured_score", "mechanism_rule_fingerprint", "next_follow_up_turn", "next_rule_turn", "normalize_experience_scope", "paired_configuration_hash", "read_creator_rule_manifest", "refuting_rule_ids", "resolve_domain_profile", "resolve_retrieval_statuses", "resolved_applicability", "restore_state_snapshot", "route_by_stage", "route_experiences", "seed_injection", "select_bounded_action", "select_next_round_proposal", "round_proposal_order", "skeleton_state_hash", "stage_for_action", "state_hash", "validate_experience_cases", "validate_reflection_rule", "verify_attribution", "verify_state_snapshot", "HUMAN_PREFERENCE_CONTRACT", "HUMAN_PREFERENCE_VERSION", "HumanReviewStore", "collect_review_items", "write_review_bundle", "validate_human_review", "export_choice_items", "human_review_stats", "DEFAULT_REVIEW_KIND", "HUMAN_PREFERENCE_ENTRY_CONTRACT", "HUMAN_PREFERENCE_ENTRY_VERSION", "HUMAN_REVIEW_DIR", "PREFERENCE_STATUSES", "REVIEW_KINDS", "preference_entry", "review_kind", "unconsumed_verdicts", "validate_preference_entry", "ROUND_METRICS_CONTRACT", "ROUND_METRICS_VERSION", "round_metrics", "layer_walk", "reflection_yield"]
