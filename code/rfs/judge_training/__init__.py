"""Dataset preparation utilities for the multi-task figure Judge."""

from .dataset_builder import build_dataset, records_from_run, validate_dataset
from .dataset_audit import audit_dataset
from .schemas import TASK_TYPES, normalize_example, validate_example
from .judge_schema import (
    AUXILIARY_DIAGNOSIS_TYPES,
    DIAGNOSIS_TYPES,
    GATE_DIAGNOSIS_TYPES,
    JUDGE_CONTRACT_VERSION,
    JUDGEMENT_TASK_TYPES,
    PROCESS_TASK_TYPES,
    QUALITY_DIMENSIONS,
    SEVERITY_LEVELS,
    defect_label,
    entity_ids,
    grounding_field,
    has_blocking_diagnosis,
    human_verdict_quality_scores,
    is_legacy_dataset_version,
    normalize_judgement,
    pairwise_choice,
    scientific_score,
    validate_judgement,
)
from .evaluator import build_suite_manifest, compare_shadow_predictions, evaluate_predictions, load_suite_manifest
from .preference_sources import (
    SourceUnavailable,
    defect_distribution,
    open_corruption_source,
    open_generation_source,
    open_middleware_source,
    open_source,
    write_source_manifest,
)
from .baseline import run_baseline
from .training_export import export_training_data
from .dpo_slices import pair_family, parse_sizes, scale_quotas, slice_mixed_dpo
from .preference_pairs import (
    PAIR_CONTRACT,
    PAIR_SCHEMA_VERSION,
    REJECTED_FAMILIES,
    apply_human_review,
    build_preference_pairs,
    build_review_queue,
    export_preference_pairs,
    round_outcome,
)
from .training import (
    StepOutcome,
    TrainingConfig,
    TrainingUnavailable,
    build_backend,
    dpo_loss,
    preference_logprob_term,
    load_training_records,
    train_dpo,
    version_lock,
)
from .bakeoff import run_bakeoff
from .registry import register_checkpoint, resolve_checkpoint, validate_manifest
from .real_vs_generated import (
    REAL_PAIR_CONTRACT,
    REAL_PAIR_VERSION,
    build_real_vs_generated_examples,
    gate_verdict,
    shared_contract,
)
from .promotion import (
    ADVERSARIAL_CASES,
    GATE_CLASSES,
    adversarial_gates,
    build_promotion_report,
    per_class_gates,
    promotion_decision,
    shadow_gate,
)
from .choice_gate import (
    CHOICE_ADVERSARIAL_CASES,
    CHOICE_GATE_SCHEMA_VERSION,
    CHOICE_GATE_THRESHOLDS,
    build_choice_gate_report,
    pair_table as choice_pair_table,
    registry_manifest as choice_registry_manifest,
)
from .framework_manifest import build_framework_manifest
from .evidence_builder import build_raw_manifest_evidence
from .evidence_schema import normalize_structured_evidence, repair_unique_evidence_pages, validate_raw_evidence, validate_structured_evidence
from .d0_selection import select_d0
from .structured_evidence import generate_d0_structured_evidence
from .evidence_reviewer import review_evidence_directory
from .framework_dataset import build_framework_judge_dataset, validate_framework_judge_dataset
from .corruption_generator import build_corruption_manifest, plan_corruptions
from .corruption_runner import run_corruptions
from .corruption_verifier import verify_corruptions
from .corruption_dataset import build_corrupted_examples, build_stratified_audit_queue, build_local_corrupted_examples
from .corruption_wave import build_corruption_source, corruption_contract, run_local_corruption_wave
from .local_corruptions import generate_region_candidates, generate_approved_corruptions, apply_local_corruption
from .local_region_reviewer import review_local_regions, screen_regions_geometrically
from .freeze_dataset import freeze_d1_dataset, validate_frozen_dataset
from .literature_freeze import build_strict_training_view
from .strict_candidate_freeze import freeze_strict_candidates
from .pilot_selection import (
    build_pilot_payloads,
    candidate_pool,
    demote_rejected,
    gate_labels,
    make_structure_gate_runner,
    select_pilot,
    select_pilot_verified,
    validate_pilot_version,
    write_pilot_version,
)
from .perceptual_hash import (
    audit_perceptual_leakage,
    build_perceptual_hash_report,
    write_perceptual_hash_report,
)
from .split_ledger import build_split_payloads, parse_ratios, validate_split_version, write_split_version
from .literature_collection import collect_literature_papers
from .arxiv_matching import match_arxiv_candidates

__all__ = ["build_split_payloads", "parse_ratios", "validate_split_version", "write_split_version", "build_perceptual_hash_report", "write_perceptual_hash_report", "audit_perceptual_leakage", "build_pilot_payloads", "candidate_pool", "select_pilot", "select_pilot_verified",
    "demote_rejected", "gate_labels", "make_structure_gate_runner", "validate_pilot_version",
    "screen_regions_geometrically", "build_corruption_source", "corruption_contract", "run_local_corruption_wave",
    "write_pilot_version", "SourceUnavailable", "open_middleware_source", "open_generation_source", "open_corruption_source", "defect_distribution", "JUDGE_CONTRACT_VERSION", "QUALITY_DIMENSIONS", "SEVERITY_LEVELS", "PROCESS_TASK_TYPES", "JUDGEMENT_TASK_TYPES", "DIAGNOSIS_TYPES", "GATE_DIAGNOSIS_TYPES", "AUXILIARY_DIAGNOSIS_TYPES", "normalize_judgement", "validate_judgement", "scientific_score", "has_blocking_diagnosis", "pairwise_choice", "grounding_field", "defect_label", "entity_ids", "human_verdict_quality_scores", "is_legacy_dataset_version", "TASK_TYPES", "build_dataset", "records_from_run", "validate_dataset", "audit_dataset", "normalize_example", "validate_example", "evaluate_predictions", "compare_shadow_predictions", "load_suite_manifest", "build_suite_manifest", "register_checkpoint", "resolve_checkpoint", "validate_manifest", "build_framework_manifest", "build_raw_manifest_evidence", "normalize_structured_evidence", "repair_unique_evidence_pages", "validate_raw_evidence", "validate_structured_evidence", "select_d0", "generate_d0_structured_evidence", "review_evidence_directory", "build_framework_judge_dataset", "validate_framework_judge_dataset", "build_corruption_manifest", "plan_corruptions", "run_corruptions", "verify_corruptions", "build_corrupted_examples", "build_stratified_audit_queue", "build_local_corrupted_examples", "generate_region_candidates", "generate_approved_corruptions", "apply_local_corruption", "review_local_regions", "freeze_d1_dataset", "validate_frozen_dataset", "build_strict_training_view", "freeze_strict_candidates", "collect_literature_papers", "match_arxiv_candidates", "run_baseline", "export_training_data", "run_bakeoff",
    "PAIR_CONTRACT", "PAIR_SCHEMA_VERSION", "REJECTED_FAMILIES", "build_preference_pairs",
    "export_preference_pairs", "build_review_queue", "apply_human_review", "round_outcome",
    "pair_family", "parse_sizes", "scale_quotas", "slice_mixed_dpo",
    "StepOutcome", "TrainingConfig", "TrainingUnavailable", "build_backend", "dpo_loss", "preference_logprob_term",
    "load_training_records", "train_dpo", "version_lock",
    "REAL_PAIR_CONTRACT", "REAL_PAIR_VERSION", "build_real_vs_generated_examples",
    "shared_contract", "gate_verdict",
    "ADVERSARIAL_CASES", "GATE_CLASSES", "per_class_gates", "adversarial_gates",
    "shadow_gate", "promotion_decision", "build_promotion_report",
    "CHOICE_ADVERSARIAL_CASES", "CHOICE_GATE_SCHEMA_VERSION", "CHOICE_GATE_THRESHOLDS",
    "build_choice_gate_report", "choice_pair_table", "choice_registry_manifest",
    "write_source_manifest", "open_source"]
