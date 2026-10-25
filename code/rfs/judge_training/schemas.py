"""Stable schemas for Judge training and Shadow evaluation records.

Task vocabulary and judgement validation come from
:mod:`rfs.judge_training.judge_schema`; this module only adds the row-level envelope
(ids, split, assets, provenance) and the process-supervision label checks.
"""

from __future__ import annotations

from typing import Any

from .judge_schema import JUDGEMENT_TASK_TYPES, OBSERVED_FIELDS, PROCESS_TASK_TYPES, validate_judgement


TASK_TYPES = set(PROCESS_TASK_TYPES) | set(JUDGEMENT_TASK_TYPES)
FRAMEWORK_TASK_TYPES = {"error_audit", "method_grounding", "pairwise_preference", "scientific_quality", "style_preference"}


def normalize_example(raw: dict[str, Any]) -> dict[str, Any]:
    labels = raw.get("labels") if isinstance(raw.get("labels"), dict) else {}
    provenance = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
    task_type = str(raw.get("task_type") or "").strip()
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported Judge task_type: {task_type}")
    return {
        "example_id": str(raw.get("example_id") or ""),
        "dataset_version": str(raw.get("dataset_version") or "judge-dataset-v002"),
        "ground_truth_id": str(raw.get("ground_truth_id") or ""),
        "task_type": task_type,
        "paper": raw.get("paper") if isinstance(raw.get("paper"), dict) else {},
        "ground_truth": raw.get("ground_truth") if isinstance(raw.get("ground_truth"), dict) else {},
        "split": str(raw.get("split") or ""),
        "candidate_a": raw.get("candidate_a"),
        "candidate_b": raw.get("candidate_b"),
        "preview": raw.get("preview"),
        "object_program": raw.get("object_program"),
        "process_context": raw.get("process_context") if isinstance(raw.get("process_context"), dict) else {},
        "observed": raw.get("observed") if isinstance(raw.get("observed"), dict) else {},
        "pair_group": str(raw.get("pair_group") or ""),
        "repeat_group": str(raw.get("repeat_group") or ""),
        "labels": labels,
        "provenance": provenance,
        "judge_metadata": {
            "judge_role": str((raw.get("judge_metadata") or {}).get("judge_role") or "bootstrap"),
            "trained": bool((raw.get("judge_metadata") or {}).get("trained", False)),
            "human_audited": bool((raw.get("judge_metadata") or {}).get("human_audited", False)),
        },
    }


def validate_example(example: dict[str, Any]) -> list[str]:
    errors = []
    for field in ("example_id", "ground_truth_id", "task_type"):
        if not str(example.get(field) or "").strip():
            errors.append(f"missing {field}")
    if example.get("task_type") not in TASK_TYPES:
        errors.append("invalid task_type")
    labels = example.get("labels") if isinstance(example.get("labels"), dict) else {}
    if example.get("task_type") == "repair":
        leaked = sorted(field for field in OBSERVED_FIELDS if field in labels)
        if leaked:
            errors.append("repair labels are the model target only; observed fields leaked: " + ", ".join(leaked))
        if not labels:
            errors.append("repair labels require the judgement the model must produce")
    if example.get("task_type") in JUDGEMENT_TASK_TYPES:
        errors.extend(f"labels: {error}" for error in validate_judgement(labels, task_type=str(example.get("task_type"))))
    if example.get("task_type") == "trajectory":
        for field in ("paper_evidence", "scientific_ground_truth", "subgraphs", "state_before", "diagnosis", "action", "state_after", "action_execution", "object_diff", "repair_gain", "grounding_gate", "decision", "reflection", "audit_status"):
            if field not in labels:
                errors.append(f"trajectory labels require {field}")
        if labels.get("decision") not in {"accept", "rollback"}:
            errors.append("trajectory decision must be accept or rollback")
        subgraphs = labels.get("subgraphs") if isinstance(labels.get("subgraphs"), dict) else {}
        if any(layer not in subgraphs for layer in ("scientific", "visual", "editable")):
            errors.append("trajectory requires scientific, visual, and editable subgraphs")
        if labels.get("audit_status") not in {"machine_verified", "human_verified"}:
            errors.append("trajectory audit_status must be verified")
    if not example.get("candidate_a") and not example.get("preview"):
        errors.append("candidate_a or preview is required")
    if example.get("task_type") in FRAMEWORK_TASK_TYPES:
        if example.get("split") not in {"train", "validation", "test"}:
            errors.append("framework Judge example requires train/validation/test split")
        if not example.get("paper"):
            errors.append("framework Judge example requires paper evidence")
        if not example.get("ground_truth"):
            errors.append("framework Judge example requires ground_truth")
    return errors
