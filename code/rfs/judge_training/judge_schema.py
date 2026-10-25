"""Canonical Judge judgement contract shared by prompts, datasets, and evaluation.

The judgement contract is the single definition of what a Judge says about one figure
pair or one figure under audit. Prompt building, dataset validation, and metric
extraction all read this module, so the three cannot drift apart.

Two layers are kept deliberately apart:

* the judgement contract (model output, stored under ``labels`` and under the exported
  ``prediction``): ``preferred_candidate``, ``quality_scores``, ``diagnosis`` and
  ``confidence``, plus the task-specific ``grounding`` block;
* record metadata (``quality_label``, ``adversarial_type``, ``pair_group``,
  ``repeat_group``, ``feedback_effective`` and similar bookkeeping), which stays a
  row-level concern and passes through normalization untouched.

The module also owns the deterministic label rules that turn an injected defect or a
binary human verdict into a contract-shaped judgement. Insisting on one table keeps
controlled corruptions, local corruptions, and future preference pairs from inventing
their own scores. Those numbers are training targets, not measurements.

Datasets frozen before this contract (``judge-v001``, ``judge-d1-v001``,
``judge-dataset-v001``) are historical: they are not migrated and must not be mixed
into new training or evaluation runs.
"""

from __future__ import annotations

from typing import Any

JUDGE_CONTRACT_VERSION = "judge-judgement-v1"

QUALITY_DIMENSIONS = (
    "scientific_fidelity",
    "logical_flow",
    "layout_quality",
    "readability",
    "domain_fit",
)
SEVERITY_LEVELS = ("low", "medium", "high")

PAIRWISE_TASK_TYPES = ("image_pairwise", "pairwise_preference", "style_preference")
GROUNDING_TASK_TYPES = ("method_grounding", "error_audit")
PROCESS_TASK_TYPES = ("outcome", "repair", "trajectory")

#: The consequences of a round. They are recorded after the Judge has spoken, so they
#: are supervision and audit material and never part of a Judge output target.
OBSERVED_FIELDS = ("action", "action_is_repair", "action_execution", "object_diff", "repair_gain", "decision", "reflection", "audit_status")

JUDGEMENT_TASK_TYPES = (
    "error_audit",
    "image_aesthetic",
    "image_pairwise",
    "image_scientific",
    "method_grounding",
    "pairwise_preference",
    "ppt_editability",
    "ppt_scientific_preservation",
    "repair_effectiveness",
    "scientific_quality",
    "segmentation_quality",
    "style_preference",
)
TASK_TYPES = PROCESS_TASK_TYPES + JUDGEMENT_TASK_TYPES

GATE_DIAGNOSIS_TYPES = ("module_order_error", "wrong_arrow_target", "overlap")
AUXILIARY_DIAGNOSIS_TYPES = (
    "assumed_worse",
    "beautiful_but_wrong",
    "correct_but_ugly",
    "invented_module",
    "missing_core_entity",
    "missing_core_stage",
    "terminology_corruption",
    "training_inference_confusion",
)
DIAGNOSIS_TYPES = tuple(sorted(GATE_DIAGNOSIS_TYPES + AUXILIARY_DIAGNOSIS_TYPES))
# Two verdicts certify a worse figure without naming a scientific defect: a candidate
# whose science passed but whose deterministic visual reports flagged it, and a candidate
# nothing could fault that still ranks below the paper's own published figure. Neither
# may be counted as a proven defect.
NON_BLOCKING_DIAGNOSIS_TYPES = ("assumed_worse", "correct_but_ugly")

GROUNDING_LIST_FIELDS = (
    "grounded_entities",
    "ungrounded_entities",
    "missing_relations",
    "incorrect_relations",
    "terminology_errors",
)
GROUNDING_MAPPING_FIELDS = ("relation_directions",)

LEGACY_DATASET_VERSIONS = ("judge-v001", "judge-d1-v001", "judge-dataset-v001")

_CANDIDATE_CHOICES = ("A", "B", "TIE")


def _scores(scientific_fidelity: float, logical_flow: float, layout_quality: float, readability: float, domain_fit: float) -> dict[str, float]:
    return {
        "scientific_fidelity": scientific_fidelity,
        "logical_flow": logical_flow,
        "layout_quality": layout_quality,
        "readability": readability,
        "domain_fit": domain_fit,
    }


DEFECT_LABELS: dict[str, dict[str, Any]] = {
    "module_order_error": {"diagnosis_type": "module_order_error", "severity": "high", "quality_scores": _scores(0.0, 0.1, 0.8, 0.9, 0.3)},
    "wrong_arrow_target": {"diagnosis_type": "wrong_arrow_target", "severity": "high", "quality_scores": _scores(0.0, 0.1, 0.9, 0.9, 0.3)},
    "relation_reversal": {"diagnosis_type": "wrong_arrow_target", "severity": "high", "quality_scores": _scores(0.0, 0.1, 0.9, 0.9, 0.3)},
    "missing_relation": {"diagnosis_type": "wrong_arrow_target", "severity": "high", "quality_scores": _scores(0.0, 0.2, 0.9, 0.9, 0.3)},
    "overlap": {"diagnosis_type": "overlap", "severity": "medium", "quality_scores": _scores(0.6, 0.6, 0.1, 0.1, 0.7)},
    "missing_core_entity": {"diagnosis_type": "missing_core_entity", "severity": "high", "quality_scores": _scores(0.0, 0.3, 0.8, 0.9, 0.2)},
    "missing_core_stage": {"diagnosis_type": "missing_core_stage", "severity": "high", "quality_scores": _scores(0.0, 0.1, 0.6, 0.9, 0.2)},
    "invented_module": {"diagnosis_type": "invented_module", "severity": "high", "quality_scores": _scores(0.0, 0.4, 0.9, 0.9, 0.3)},
    "terminology_corruption": {"diagnosis_type": "terminology_corruption", "severity": "high", "quality_scores": _scores(0.0, 0.6, 0.9, 0.9, 0.0)},
    "training_inference_confusion": {"diagnosis_type": "training_inference_confusion", "severity": "high", "quality_scores": _scores(0.0, 0.2, 0.8, 0.9, 0.2)},
    "beautiful_but_wrong": {"diagnosis_type": "beautiful_but_wrong", "severity": "high", "quality_scores": _scores(0.0, 0.4, 1.0, 1.0, 0.4)},
    # Nothing deterministic could fault this candidate, so the ranking is the one the
    # published figure holds by default: a mild, unassertive penalty on every dimension
    # instead of a claimed defect. Reconciled by hand later, which is why it is the only
    # label whose sample records human_audited false.
    "assumed_worse": {"diagnosis_type": "assumed_worse", "severity": "low", "quality_scores": _scores(0.6, 0.7, 0.6, 0.7, 0.6)},
    "correct_but_ugly": {"diagnosis_type": "correct_but_ugly", "severity": "low", "quality_scores": _scores(1.0, 1.0, 0.2, 0.1, 1.0)},
}

# A binary human verdict only certifies scientific fidelity. Published figures stay
# legible, so a rejected figure keeps its visual dimensions and loses the scientific
# ones; an accepted figure is accepted on every dimension.
HUMAN_VERDICT_QUALITY_SCORES: dict[str, dict[str, float]] = {
    "好": _scores(1.0, 1.0, 1.0, 1.0, 1.0),
    "坏": _scores(0.0, 0.0, 1.0, 1.0, 0.0),
}


def defect_label(corruption_type: str) -> dict[str, Any]:
    """Return the deterministic judgement target of one injected defect."""

    key = _as_text(corruption_type)
    if key not in DEFECT_LABELS:
        raise KeyError(f"unsupported defect type: {corruption_type!r}")
    return DEFECT_LABELS[key]


def human_verdict_quality_scores(quality_label: str) -> dict[str, float]:
    """Expand a binary human verdict into the five contract dimensions."""

    key = _as_text(quality_label)[:1]
    if key not in HUMAN_VERDICT_QUALITY_SCORES:
        raise KeyError(f"unsupported quality label: {quality_label!r}")
    return dict(HUMAN_VERDICT_QUALITY_SCORES[key])


def entity_ids(values: Any) -> list[str]:
    """Return stable ids from entity references that may be ids or entity objects."""

    if not isinstance(values, (list, tuple, set)):
        return _as_text_list(values)
    result: list[str] = []
    for item in values:
        if isinstance(item, dict):
            text = _as_text(item.get("entity_id") or item.get("id") or item.get("name"))
        else:
            text = _as_text(item)
        if text and text not in result:
            result.append(text)
    return result


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [_as_text(item) for item in value if _as_text(item)]
    if value in (None, ""):
        return []
    text = _as_text(value)
    return [text] if text else []


def _normalize_diagnosis(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    normalized: dict[str, Any] = {
        "type": _as_text(entry.get("type")).lower(),
        "target": _as_text(entry.get("target")),
        "severity": _as_text(entry.get("severity")).lower(),
        "evidence": _as_text(entry.get("evidence")),
    }
    figure = _as_text(entry.get("figure")).upper()
    if figure:
        normalized["figure"] = figure
    return normalized


def _normalize_grounding(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    normalized: dict[str, Any] = {field: _as_text_list(source.get(field)) for field in GROUNDING_LIST_FIELDS}
    for field in GROUNDING_MAPPING_FIELDS:
        mapping = source.get(field) if isinstance(source.get(field), dict) else {}
        normalized[field] = {_as_text(key): _as_text(item).lower() for key, item in mapping.items() if _as_text(key)}
    return normalized


def judgement_schema(task_type: str) -> dict[str, Any]:
    """Return the required output fields of one judgement task, for prompting."""

    schema: dict[str, Any] = {}
    if task_type in PAIRWISE_TASK_TYPES:
        schema["preferred_candidate"] = "A|B|tie"
    schema["quality_scores"] = {dimension: "number 0..1" for dimension in QUALITY_DIMENSIONS}
    schema["diagnosis"] = [
        {
            "figure": "A|B (optional)",
            "type": "|".join(DIAGNOSIS_TYPES),
            "target": "node, edge, slot, object id or explicit global scope",
            "severity": "|".join(SEVERITY_LEVELS),
            "evidence": "short quote, measurement or object id",
        }
    ]
    schema["confidence"] = "number 0..1"
    if task_type in GROUNDING_TASK_TYPES:
        grounding: dict[str, Any] = {field: "array of strings" for field in GROUNDING_LIST_FIELDS}
        grounding.update({field: "object mapping id to direction" for field in GROUNDING_MAPPING_FIELDS})
        schema["grounding"] = grounding
    return schema


def normalize_judgement(value: Any) -> dict[str, Any]:
    """Return the canonical form of one judgement, passing record metadata through."""

    if not isinstance(value, dict):
        return {}
    result = dict(value)
    choice = _as_text(result.get("preferred_candidate")).upper()
    if choice:
        result["preferred_candidate"] = choice
    scores = result.get("quality_scores")
    if isinstance(scores, dict):
        result["quality_scores"] = {
            dimension: float(_as_float(scores.get(dimension)))
            for dimension in QUALITY_DIMENSIONS
            if _as_float(scores.get(dimension)) is not None
        }
    if isinstance(result.get("diagnosis"), list):
        result["diagnosis"] = [item for item in (_normalize_diagnosis(entry) for entry in result["diagnosis"]) if item]
    confidence = _as_float(result.get("confidence"))
    if confidence is not None:
        result["confidence"] = confidence
    if isinstance(result.get("grounding"), dict):
        result["grounding"] = _normalize_grounding(result["grounding"])
    return result


def validate_judgement(value: Any, *, task_type: str | None = None) -> list[str]:
    """Return contract violations of one judgement; empty means the contract holds."""

    if not isinstance(value, dict):
        return ["judgement must be an object"]
    errors: list[str] = []
    scores = value.get("quality_scores")
    if not isinstance(scores, dict):
        errors.append("quality_scores is required")
    else:
        for dimension in QUALITY_DIMENSIONS:
            score = _as_float(scores.get(dimension))
            if score is None or not 0.0 <= score <= 1.0:
                errors.append(f"quality_scores.{dimension} must be a number 0..1")
    confidence = _as_float(value.get("confidence"))
    if confidence is None or not 0.0 <= confidence <= 1.0:
        errors.append("confidence must be a number 0..1")
    diagnosis = value.get("diagnosis")
    if not isinstance(diagnosis, list):
        errors.append("diagnosis must be an array")
    else:
        for entry in diagnosis:
            if not isinstance(entry, dict):
                errors.append("diagnosis entries must be objects")
                continue
            normalized = _normalize_diagnosis(entry) or {}
            if normalized.get("type") not in DIAGNOSIS_TYPES:
                errors.append(f"diagnosis type is unsupported: {entry.get('type')!r}")
            if not normalized.get("target"):
                errors.append("diagnosis entry requires a target")
            if normalized.get("severity") not in SEVERITY_LEVELS:
                errors.append(f"diagnosis severity is unsupported: {entry.get('severity')!r}")
            if not normalized.get("evidence"):
                errors.append("diagnosis entry requires evidence")
            figure = normalized.get("figure")
            if figure and figure not in {"A", "B"}:
                errors.append(f"diagnosis figure must be A or B: {entry.get('figure')!r}")
    if task_type in PAIRWISE_TASK_TYPES and _as_text(value.get("preferred_candidate")).upper() not in _CANDIDATE_CHOICES:
        errors.append("pairwise judgements require preferred_candidate A, B or tie")
    if task_type in GROUNDING_TASK_TYPES:
        grounding = value.get("grounding")
        if not isinstance(grounding, dict):
            errors.append("grounding is required for grounding tasks")
        else:
            for field in GROUNDING_LIST_FIELDS:
                if field in grounding and not isinstance(grounding.get(field), list):
                    errors.append(f"grounding.{field} must be an array")
            for field in GROUNDING_MAPPING_FIELDS:
                if field in grounding and not isinstance(grounding.get(field), dict):
                    errors.append(f"grounding.{field} must be an object")
    return errors


def scientific_score(value: Any) -> float | None:
    """Return the scientific fidelity score of one judgement, if it carries one."""

    judgement = normalize_judgement(value)
    scores = judgement.get("quality_scores") if isinstance(judgement.get("quality_scores"), dict) else {}
    return _as_float(scores.get("scientific_fidelity"))


def diagnosis_types(value: Any) -> list[str]:
    judgement = normalize_judgement(value)
    diagnosis = judgement.get("diagnosis") if isinstance(judgement.get("diagnosis"), list) else []
    return [str(entry.get("type") or "") for entry in diagnosis if isinstance(entry, dict)]


def has_blocking_diagnosis(value: Any) -> bool:
    """Return True when a judgement reports at least one non-style problem."""

    return any(item and item not in NON_BLOCKING_DIAGNOSIS_TYPES for item in diagnosis_types(value))


def pairwise_choice(value: Any) -> str:
    return str(normalize_judgement(value).get("preferred_candidate") or "")


def grounding(value: Any) -> dict[str, Any]:
    judgement = normalize_judgement(value)
    return judgement.get("grounding") if isinstance(judgement.get("grounding"), dict) else _normalize_grounding({})


def grounding_field(value: Any, field: str) -> Any:
    """Return one grounding field, falling back to an empty container of its shape."""

    fields = grounding(value)
    if field in fields:
        return fields[field]
    return {} if field in GROUNDING_MAPPING_FIELDS else []


def is_legacy_dataset_version(version: Any) -> bool:
    return _as_text(version) in LEGACY_DATASET_VERSIONS
