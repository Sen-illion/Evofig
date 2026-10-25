"""Build chosen and rejected preference pairs from the deterministic Judge sources.

A pair is two judgements of one prompt: the one the deterministic sources certify (a
controlled defect label, a Grounding Gate conclusion, or one of the three gating rules)
and one a weaker Judge could plausibly give. Rejected answers come in two families - a
wrong pick between A and B while the defect was still detected, or a missed or
misdiagnosed defect. Process rounds add a third family: an accepted repair whose gain was
measured, paired with the bounded action a controlled negative round really executed and
really got rolled back. That family never invents its rejected side; without a measured
rollback there is no pair.

Nothing is admitted on faith. A pair is written only when its defect is traceable, the
two answers really differ, the deterministic answer has an unambiguous conclusion, and
the rejected answer provably does not name the injected defect. ``human_audited`` stays
false until a human has looked at the pair, which is why the builder writes the review
queue and a contact sheet of the candidate images next to the pairs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .judge_schema import (
    DEFECT_LABELS,
    DIAGNOSIS_TYPES,
    GATE_DIAGNOSIS_TYPES,
    NON_BLOCKING_DIAGNOSIS_TYPES,
    PAIRWISE_TASK_TYPES,
    PROCESS_TASK_TYPES,
    defect_label,
    diagnosis_types,
    normalize_judgement,
    pairwise_choice,
    validate_judgement,
)
from .prompts import build_prompt
from ..contracts.evolution import normalize_judge_result, validate_judge_result


PAIR_CONTRACT = "rfs.judge_preference_pair"
PAIR_SCHEMA_VERSION = "judge-preference-pair-v1"
REJECTED_FAMILIES = ("wrong_pick", "missed_defect", "misdiagnosed")
PROCESS_REJECTED_FAMILY = "harmful_repair"
WRONG_PICK = "wrong_pick"
MISSED_DEFECT = "missed_defect"
MISDIAGNOSED = "misdiagnosed"
#: What a judge reports when nothing it can name is wrong. ``DEFECT_LABELS`` fixes one
#: score set per diagnosis, so a missed-defect rejection carries this label (and its
#: scores) instead of keeping the certified defect's scores or clearing the diagnosis.
NO_DEFECT_LABEL = "assumed_worse"
#: A missed-defect rejection used to set ``diagnosis`` to ``[]``, which made the rejected
#: JSON much shorter than the certified one. Mean-token DPO then preferred the short
#: answer before any training. The rejection must stay within this fraction of the
#: certified judgement's serialized length.
MISSED_DEFECT_MIN_LENGTH_RATIO = 0.7
#: Labels that rank a figure below the paper's own without naming a defect. A
#: misdiagnosis has to name a defect class, so these are never the class it picks.
NON_DEFECT_DIAGNOSIS_TYPES = frozenset(NON_BLOCKING_DIAGNOSIS_TYPES)
FLIP = {"A": "B", "B": "A"}
SKIPPED_SAMPLE = 50


def _jsonl_line(row: dict[str, Any]) -> str:
    """Encode one record as one physical line.

    U+2028 and U+2029 are legal inside a JSON string and are also line breaks for
    ``str.splitlines``. Escaping them keeps a JSONL reader from splitting a record.
    """

    return (
        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_jsonl_line(row) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{str(key).strip()}={str(item).strip()}" for key, item in value.items() if str(key).strip())
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip() if value is not None else ""


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown").strip().casefold() or "unknown"
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _family_length_stats(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize how much longer or shorter each family's rejected JSON is."""

    buckets: dict[str, list[int]] = {}
    for item in pairs:
        family = str((item.get("rejected") or {}).get("family") or item.get("preference_basis", {}).get("rejected_family") or "unknown")
        chosen = (item.get("chosen") or {}).get("judgement")
        rejected = (item.get("rejected") or {}).get("judgement")
        if not isinstance(chosen, dict) or not isinstance(rejected, dict):
            continue
        buckets.setdefault(family, []).append(_judgement_chars(rejected) - _judgement_chars(chosen))
    report: dict[str, Any] = {}
    for family, deltas in sorted(buckets.items()):
        ordered = sorted(deltas)
        report[family] = {
            "pairs": len(deltas),
            "mean_rejected_minus_chosen_chars": round(sum(deltas) / len(deltas), 3),
            "median_rejected_minus_chosen_chars": ordered[len(ordered) // 2],
        }
    return report


def known_defects(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the injected defect of one example, or an empty list when it has none."""

    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    kind = str(provenance.get("corruption_type") or "").strip()
    if not kind:
        return []
    try:
        defect = defect_label(kind)
    except KeyError:
        return []
    return [{
        "corruption_type": kind,
        "diagnosis_type": str(defect["diagnosis_type"]),
        "severity": str(defect["severity"]),
        "target": _text(provenance.get("target")),
        "source": str(provenance.get("source") or ""),
        "gating": str(defect["diagnosis_type"]) in GATE_DIAGNOSIS_TYPES,
    }]


def chosen_source(row: dict[str, Any]) -> str:
    """Name where the certified judgement comes from."""

    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    source = str(provenance.get("source") or "")
    if "corruption" in source:
        return "controlled_error_label"
    if "real_vs_generated" in source:
        return "grounding_gate"
    if "rag_subgraph" in source:
        return "subgraph_structure_rule"
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    if isinstance(labels.get("grounding"), dict):
        return "grounding_gate"
    return "deterministic_rule" if diagnosis_types(labels) else ""


def _flip_choice(value: dict[str, Any]) -> dict[str, Any]:
    choice = pairwise_choice(value)
    return {**value, "preferred_candidate": FLIP[choice]} if choice in FLIP else dict(value)


def _mirror_figures(value: dict[str, Any]) -> dict[str, Any]:
    diagnosis = []
    for entry in value.get("diagnosis") or []:
        entry = dict(entry)
        if entry.get("figure") in FLIP:
            entry["figure"] = FLIP[entry["figure"]]
        diagnosis.append(entry)
    return {**value, "diagnosis": diagnosis}


def mirror_judgement(value: dict[str, Any]) -> dict[str, Any]:
    """Mirror a judgement onto the swapped A/B orientation of the same pair."""

    return _mirror_figures(_flip_choice(value))


def _other_diagnosis_type(defects: Sequence[dict[str, Any]]) -> str:
    """Return a defect class the certified judgement did not name.

    The replacement has to be a defect class. ``assumed_worse`` and ``correct_but_ugly``
    certify a ranking without naming a defect, so a misdiagnosis that picked one would
    teach the same thing a missed defect does. Excluding them also keeps the pick stable:
    ``assumed_worse`` sorts first, so adding it to the vocabulary would otherwise have
    silently rewritten which wrong class every misdiagnosis names.
    """

    injected = {str(item["diagnosis_type"]) for item in defects}
    for candidate in sorted(DIAGNOSIS_TYPES):
        if candidate not in injected and candidate not in NON_DEFECT_DIAGNOSIS_TYPES:
            return candidate
    return ""


def canonical_scores(diagnosis_type: str) -> dict[str, Any]:
    """Return the five scores ``DEFECT_LABELS`` fixes for one diagnosis type.

    The table is keyed by corruption, and several corruptions report one diagnosis type
    (``relation_reversal`` and ``missing_relation`` both report ``wrong_arrow_target``). A
    diagnosis type is also its own key whenever it has one, so that key is preferred and
    the corruptions sharing the type are only a fallback.
    """

    key = _text(diagnosis_type)
    if not key:
        return {}
    label = DEFECT_LABELS.get(key)
    if label is None:
        label = next((item for item in DEFECT_LABELS.values() if str(item.get("diagnosis_type")) == key), None)
    return dict(label["quality_scores"]) if label else {}


def _injected_defect_names(defects: Sequence[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("diagnosis_type") or "") for item in defects if str(item.get("diagnosis_type") or "")})


def _judgement_chars(value: dict[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _missed_defect_diagnosis(chosen: dict[str, Any], defects: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return the diagnosis a missed-defect rejection names.

    It has to occupy the same fields as the certified diagnosis: an empty list used to
    shrink the JSON by hundreds of characters, which is a length cue rather than a
    judgement. The type is the non-blocking ``assumed_worse`` label, and the evidence
    restates the certified finding as something this answer ignored.
    """

    diagnosis = [item for item in (chosen.get("diagnosis") or []) if isinstance(item, dict)]
    first = diagnosis[0] if diagnosis else {}
    target = str(first.get("target") or "") or (_text(defects[0].get("target")) if defects else "") or "global_scope"
    named = ", ".join(_injected_defect_names(defects)) or "the injected defect"
    certified = str(first.get("evidence") or "").strip() or target
    entry: dict[str, Any] = {
        "type": NO_DEFECT_LABEL,
        "target": target,
        "severity": str(defect_label(NO_DEFECT_LABEL).get("severity") or "low"),
        "evidence": (
            f"no injected defect was found; the figure is claimed to have none of [{named}]. "
            f"certified evidence that was ignored: {certified}"
        ),
    }
    figure = first.get("figure")
    if figure in {"A", "B"}:
        entry["figure"] = figure
    return entry


def rejected_judgement(chosen: dict[str, Any], defects: Sequence[dict[str, Any]], *, family: str) -> dict[str, Any]:
    """Derive one rejected answer from the certified judgement.

    The five scores are a function of the diagnosis, so a rejection that moved only
    ``diagnosis`` used to keep the certified defect's scores: every pair carried two
    identical score sets, nothing ever contrasted them, and the trained Judge received no
    signal for them at all. Each family now moves the scores with the diagnosis it names,
    and a family that leaves the diagnosis alone leaves the scores alone too.
    """

    if family == WRONG_PICK:
        return _flip_choice(chosen)
    if family == MISSED_DEFECT:
        scores = canonical_scores(NO_DEFECT_LABEL) or chosen.get("quality_scores")
        return {**chosen, "diagnosis": [_missed_defect_diagnosis(chosen, defects)], "quality_scores": scores}
    diagnosis = list(chosen.get("diagnosis") or [])
    target = str(diagnosis[0].get("target") or "") if diagnosis else ""
    target = target or (_text(defects[0].get("target")) if defects else "") or "global_scope"
    entry = {"type": _other_diagnosis_type(defects), "target": target, "severity": "medium", "evidence": "the visible figure was read against the wrong defect class"}
    scores = canonical_scores(str(entry["type"])) or chosen.get("quality_scores")
    return {**chosen, "diagnosis": [entry], "quality_scores": scores}


def rejection_reason(chosen: dict[str, Any], rejected: dict[str, Any], defects: Sequence[dict[str, Any]], *, family: str, task_type: str = "") -> str:
    """Return why a pair must not be admitted; an empty string means it may be."""

    if not defects:
        return "the example carries no traceable injected defect"
    if json.dumps(chosen, sort_keys=True) == json.dumps(rejected, sort_keys=True):
        return "the two judgements are the same"
    injected = {str(item["diagnosis_type"]) for item in defects}
    named = set(diagnosis_types(rejected))
    if family == WRONG_PICK:
        if task_type not in PAIRWISE_TASK_TYPES:
            return "a wrong pick is only defined for a pairwise task"
        if pairwise_choice(chosen) not in FLIP:
            return "the certified judgement has no unambiguous pick"
        if pairwise_choice(rejected) == pairwise_choice(chosen):
            return "the rejected answer picks the same candidate"
    if family == MISSED_DEFECT:
        if not named:
            return "the rejected answer leaves the diagnosis empty"
        if named & injected:
            return "the rejected answer still names the injected defect"
        if _judgement_chars(rejected) < MISSED_DEFECT_MIN_LENGTH_RATIO * max(1, _judgement_chars(chosen)):
            return "the rejected answer is much shorter than the certified judgement"
    if family == MISDIAGNOSED:
        if not named:
            return "the rejected answer names no defect at all"
        if named & injected:
            return "the rejected answer names the injected defect"
    return ""


#: The scope a bounded action binds to, so a rejected repair names the objects it touches.
PROCESS_ACTION_SCOPE = {
    "reanalyze_semantics": "semantic",
    "replan_structure": "global",
    "resegment": "global",
    "relayout_global": "global",
    "relayout_slots": "slot",
    "resize_nodes": "slot",
    "reroute": "connector",
    "repair_interfaces": "interface",
    "regen_assets": "slot",
    "replan_theme": "global",
}


def _observed(row: dict[str, Any]) -> dict[str, Any]:
    observed = row.get("observed")
    return observed if isinstance(observed, dict) else {}


def _measured_gain(observed: dict[str, Any]) -> float:
    """Read the measured score delta of a round, never an estimated one."""

    try:
        return float(observed.get("repair_gain") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def round_outcome(observed: dict[str, Any]) -> str:
    """Classify what one bounded repair round achieved: ``improvement``, ``rollback`` or ``other``.

    The comparison the loop itself decided the round on comes first, because that is the
    evidence the decision was taken on: a round whose candidate the acceptance role preferred
    and which was accepted is an improvement even when the two absolute scores would have said
    otherwise, and a rollback the comparison called worse is a rollback even when the after
    score happened to be higher. A figure diff that says nothing moved cancels the round; a diff
    that was never taken does not. A round carrying no comparison at all falls back to the
    recorded score delta, which is all the oldest runs have.
    """

    if not bool(observed.get("action_is_repair")):
        return "other"
    judge = observed.get("judge") if isinstance(observed.get("judge"), dict) else {}
    gate_only = bool(judge.get("gate_only", False))
    decision = str(observed.get("decision") or "")
    comparison = str(observed.get("comparison") or "")
    if comparison in ("pairwise", "compared"):
        # A comparison was taken, so the score delta is not consulted: a round the figure diff
        # says did not move achieved nothing the comparison could have preferred, however the
        # numbers read. A diff that was never taken is not a veto -- older runs and synthesised
        # rounds carry no ``figure_object_diff``, and their comparison still stands.
        if observed.get("figure_changed") is False:
            return "other"
        if comparison == "pairwise":
            verdict = str(observed.get("pairwise_verdict") or "")
            better, worse = verdict == "after", verdict == "before"
        else:
            gain = _measured_gain(observed)
            better, worse = gain > 0.0, gain < 0.0
        if decision == "accept" and better and not gate_only:
            return "improvement"
        if decision == "rollback" and worse:
            return "rollback"
        return "other"
    gain = _measured_gain(observed)
    if decision == "accept" and gain > 0.0 and not gate_only:
        return "improvement"
    if decision == "rollback" and gain < 0.0:
        return "rollback"
    return "other"


def _is_measured_improvement(observed: dict[str, Any]) -> bool:
    """Report whether a bounded repair really improved the candidate."""

    return round_outcome(observed) == "improvement"


def _is_measured_rollback(observed: dict[str, Any]) -> bool:
    """Report whether the executed action really made the candidate worse."""

    return round_outcome(observed) == "rollback"


def _injected_defect(row: dict[str, Any]) -> dict[str, Any]:
    injected = _observed(row).get("injected_defect")
    return injected if isinstance(injected, dict) else {}


def _prompt_key(row: dict[str, Any]) -> str:
    """Identify the candidate state a process judgement is about."""

    observed = _observed(row)
    context = row.get("process_context") if isinstance(row.get("process_context"), dict) else {}
    payload = json.dumps({"run": row.get("ground_truth_id"), "state": observed.get("state_before") or context.get("state_before")}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def rejected_repair_judgement(chosen: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Turn one measured rollback into the judgement that would have requested it.

    The rejected side is the action the negative round really executed, so a pair can only
    exist when a rollback was measured. Its wording comes from the injected defect record
    rather than from an invented complaint.
    """

    observed = _observed(row)
    injected = _injected_defect(row)
    action = injected.get("action") if isinstance(injected.get("action"), dict) else observed.get("action")
    action = dict(action) if isinstance(action, dict) else {}
    action_type = str(action.get("type") or action.get("action") or "")
    problem = str(injected.get("request") or "").strip() or f"the {action_type or 'repair'} request was bound to the wrong object"
    return {
        **chosen,
        "status": "fault",
        "stage_feedback": [{
            "scope_type": PROCESS_ACTION_SCOPE.get(action_type, "global"),
            "problem": problem,
            "evidence": [str(item) for item in (injected.get("target"), action_type) if str(item or "").strip()][:2],
            "semantic_node_ids": list(action.get("semantic_node_ids") or []),
            "slot_ids": list(action.get("slot_ids") or []),
            "object_ids": list(action.get("object_ids") or []),
            "edge_ids": list(action.get("edge_ids") or []),
            "expected_effect": str(action.get("expected_effect") or ""),
            "forbidden_changes": list(action.get("forbidden_changes") or []),
            "requested_actions": [{"action": action_type, **{key: value for key, value in action.items() if key != "type"}}],
        }],
    }


def process_rejection_reason(chosen: dict[str, Any], rejected: dict[str, Any]) -> str:
    """Return why a process pair must not be admitted; an empty string means it may be."""

    chosen_errors = validate_judge_result(chosen)["errors"]
    if chosen_errors:
        return "the chosen side is not a judgement the controller can act on: " + "; ".join(chosen_errors)
    if json.dumps(chosen, sort_keys=True) == json.dumps(rejected, sort_keys=True):
        return "the recorded repair already names the harmful action"
    rejected_errors = validate_judge_result(rejected)["errors"]
    if rejected_errors:
        return "the rejected judgement breaks the controller contract: " + "; ".join(rejected_errors)
    return ""


def _pair_record(row: dict[str, Any], chosen: dict[str, Any], rejected: dict[str, Any], *, family: str, defects: Sequence[dict[str, Any]], source: str, swapped: bool) -> dict[str, Any]:
    candidate_a = str(row.get("candidate_a") or "")
    candidate_b = str(row.get("candidate_b") or "")
    if swapped:
        candidate_a, candidate_b = candidate_b, candidate_a
        chosen, rejected = mirror_judgement(chosen), mirror_judgement(rejected)
    ground_truth_id = str(row.get("ground_truth_id") or "")
    task_type = str(row.get("task_type") or "")
    check_name = {"wrong_pick": "rejected_pick_differs_from_the_certified_pick", "missed_defect": "rejected_answer_names_no_injected_defect", "misdiagnosed": "rejected_answer_names_a_different_defect"}[family]
    return {
        "contract": PAIR_CONTRACT,
        "schema_version": PAIR_SCHEMA_VERSION,
        "example_id": f"{row.get('example_id')}__{family}{'__swap' if swapped else ''}",
        "dataset_version": row.get("dataset_version"),
        "ground_truth_id": ground_truth_id,
        "paper_id": str(row.get("paper_id") or ""),
        "domain": str(row.get("domain") or ""),
        "split": str(row.get("split") or ""),
        "task_type": task_type,
        "pair_group": str(row.get("pair_group") or f"{ground_truth_id}|{task_type}"),
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "images": [item for item in (candidate_a, candidate_b) if item],
        "preview": str(row.get("preview") or ""),
        "prompt": build_prompt({**row, "candidate_a": candidate_a, "candidate_b": candidate_b}),
        "chosen": {"source": source, "judgement": chosen},
        "rejected": {"family": family, "judgement": rejected},
        "labels": row.get("labels"),
        "preference_basis": {
            "known_defects": list(defects),
            "chosen_source": source,
            "rejected_family": family,
            "deterministic_check": {"name": check_name, "passed": True, "injected_defects": sorted({str(item["diagnosis_type"]) for item in defects})},
            "human_audited": False,
        },
    }


def _process_pair_record(chosen_row: dict[str, Any], negative_row: dict[str, Any], chosen: dict[str, Any], rejected: dict[str, Any]) -> dict[str, Any]:
    """Write one measured-improvement/measured-rollback pair of a process round."""

    observed = _observed(negative_row)
    candidate = str(chosen_row.get("candidate_a") or chosen_row.get("preview") or "")
    ground_truth_id = str(chosen_row.get("ground_truth_id") or "")
    return {
        "contract": PAIR_CONTRACT,
        "schema_version": PAIR_SCHEMA_VERSION,
        "example_id": f"{chosen_row.get('example_id')}__{PROCESS_REJECTED_FAMILY}__{negative_row.get('example_id')}",
        "dataset_version": chosen_row.get("dataset_version"),
        "ground_truth_id": ground_truth_id,
        "paper_id": str(chosen_row.get("paper_id") or ""),
        "domain": str(chosen_row.get("domain") or ""),
        "split": str(chosen_row.get("split") or ""),
        "task_type": str(chosen_row.get("task_type") or ""),
        "pair_group": str(chosen_row.get("pair_group") or f"{ground_truth_id}|{chosen_row.get('task_type')}"),
        "candidate_a": candidate,
        "candidate_b": "",
        "images": [candidate] if candidate else [],
        "prompt": build_prompt({**chosen_row, "candidate_a": candidate, "candidate_b": ""}),
        "chosen": {"source": "measured_repair", "judgement": chosen},
        "rejected": {"family": PROCESS_REJECTED_FAMILY, "judgement": rejected},
        "labels": chosen_row.get("labels"),
        "preference_basis": {
            "known_defects": [],
            "chosen_source": "measured_repair",
            "rejected_family": PROCESS_REJECTED_FAMILY,
            "deterministic_check": {
                "name": "rejected_action_was_rolled_back",
                "passed": True,
                "negative_example_id": negative_row.get("example_id"),
                "negative_source": str((negative_row.get("provenance") or {}).get("source") or ""),
                "measured_repair_gain": _measured_gain(observed),
                "harmful_action": observed.get("action") or {},
                "injected_defect": _injected_defect(negative_row),
            },
            "human_audited": False,
        },
    }


def _process_pairs(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair measured improvements with measured rollbacks of the same candidate state."""

    improving: dict[str, list[dict[str, Any]]] = {}
    negatives: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("task_type") or "") != "repair":
            continue
        observed = _observed(row)
        if _is_measured_improvement(observed):
            improving.setdefault(_prompt_key(row), []).append(row)
        elif _is_measured_rollback(observed):
            negatives.append(row)
    pairs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in negatives:
        example_id = str(row.get("example_id") or "")
        candidates = improving.get(_prompt_key(row)) or []
        if not candidates:
            skipped.append({"example_id": example_id, "family": PROCESS_REJECTED_FAMILY, "reason": "no measured improvement was recorded for the same candidate state"})
            continue
        chosen_row = candidates[0]
        chosen = normalize_judge_result(chosen_row.get("labels"))
        rejected = rejected_repair_judgement(chosen, row)
        reason = process_rejection_reason(chosen, rejected)
        if reason:
            skipped.append({"example_id": example_id, "family": PROCESS_REJECTED_FAMILY, "reason": reason})
            continue
        pairs.append(_process_pair_record(chosen_row, row, chosen, rejected))
    return pairs, skipped


def build_preference_pairs(examples_path: str | Path, out_path: str | Path, *, families: Sequence[str] = REJECTED_FAMILIES, include_swapped: bool = False, limit: int = 0) -> dict[str, Any]:
    """Build every admissible pair of one Judge example dataset."""

    source = Path(examples_path)
    rows = _jsonl(source)
    if limit:
        rows = rows[: int(limit)]
    pairs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    process_rows: list[dict[str, Any]] = []
    for row in rows:
        example_id = str(row.get("example_id") or "")
        task_type = str(row.get("task_type") or "")
        if task_type in PROCESS_TASK_TYPES:
            process_rows.append(row)
            continue
        chosen = normalize_judgement(row.get("labels") if isinstance(row.get("labels"), dict) else {})
        errors = validate_judgement(chosen, task_type=task_type)
        if errors:
            skipped.append({"example_id": example_id, "reason": "the certified judgement breaks the contract: " + "; ".join(errors)})
            continue
        defects = known_defects(row)
        source_name = chosen_source(row)
        oriented = [False, True] if include_swapped and row.get("candidate_b") else [False]
        for swapped in oriented:
            for family in families:
                if family == WRONG_PICK and (task_type not in PAIRWISE_TASK_TYPES or not row.get("candidate_b")):
                    continue
                if family == MISSED_DEFECT and not defects:
                    continue
                rejected = rejected_judgement(chosen, defects, family=family)
                reason = rejection_reason(chosen, rejected, defects, family=family, task_type=task_type)
                if reason:
                    skipped.append({"example_id": example_id, "family": family, "reason": reason})
                    continue
                pairs.append(_pair_record(row, chosen, rejected, family=family, defects=defects, source=source_name, swapped=swapped))
    process_pairs, process_skipped = _process_pairs(process_rows)
    pairs.extend(process_pairs)
    skipped.extend(process_skipped)
    pairs.sort(key=lambda item: str(item["example_id"]))
    target = Path(out_path)
    _write_jsonl(target, pairs)
    report = {
        "contract": PAIR_CONTRACT,
        "schema_version": PAIR_SCHEMA_VERSION,
        "source": str(source),
        "source_sha256": _sha256(source) if source.is_file() else "",
        "examples": len(rows),
        "pairs": len(pairs),
        "by_family": _distribution(item["rejected"]["family"] for item in pairs),
        "length_by_family": _family_length_stats(pairs),
        "by_split": _distribution(item.get("split") for item in pairs),
        "by_domain": _distribution(item.get("domain") for item in pairs),
        "gating_pairs": sum(1 for item in pairs if any(entry.get("gating") for entry in item["preference_basis"]["known_defects"])),
        "swapped_pairs": sum(1 for item in pairs if str(item["example_id"]).endswith("__swap")),
        "human_audited": 0,
        "skipped": len(skipped),
        "skipped_sample": skipped[:SKIPPED_SAMPLE],
        "out": str(target),
        "ok": bool(pairs),
    }
    target.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _completion(judgement: dict[str, Any]) -> str:
    return json.dumps(judgement, ensure_ascii=False, separators=(",", ":"))


def export_preference_pairs(pairs_path: str | Path, out: str | Path, *, require_human_audit: bool = False) -> dict[str, Any]:
    """Export pairs to the multi-image DPO JSONL the trainer consumes."""

    source = Path(pairs_path)
    rows = _jsonl(source)
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "dpo.jsonl"
    exported = 0
    missing: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            basis = row.get("preference_basis") if isinstance(row.get("preference_basis"), dict) else {}
            if require_human_audit and not basis.get("human_audited"):
                dropped.append({"example_id": row.get("example_id"), "reason": "not human audited yet"})
                continue
            images = [str(item) for item in (row.get("images") or [])]
            absent = [item for item in images if not Path(item).exists()]
            if absent or not images:
                missing.append({"example_id": row.get("example_id"), "images": absent or "none recorded"})
                continue
            content = [{"type": "text", "text": str(row.get("prompt") or "")}, *({"type": "image", "image": item} for item in images)]
            handle.write(json.dumps({
                "contract": PAIR_CONTRACT,
                "example_id": str(row.get("example_id") or ""),
                "pair_group": str(row.get("pair_group") or ""),
                "dataset_version": row.get("dataset_version"),
                "ground_truth_id": row.get("ground_truth_id"),
                "paper_id": row.get("paper_id"),
                "domain": row.get("domain"),
                "split": row.get("split"),
                "task_type": row.get("task_type"),
                "images": images,
                "messages": [{"role": "user", "content": content}],
                "chosen": _completion((row.get("chosen") or {}).get("judgement") or {}),
                "rejected": _completion((row.get("rejected") or {}).get("judgement") or {}),
                "preference_basis": basis,
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
            exported += 1
    manifest = {
        "schema_version": "judge-dpo-export-v1",
        "contract": PAIR_CONTRACT,
        "format": "dpo",
        "source": str(source),
        "source_sha256": _sha256(source) if source.is_file() else "",
        "records": exported,
        "source_records": len(rows),
        "human_audited_records": sum(1 for row in rows if (row.get("preference_basis") or {}).get("human_audited")),
        "require_human_audit": bool(require_human_audit),
        "by_family": _distribution((row.get("rejected") or {}).get("family") for row in rows),
        "by_split": _distribution(row.get("split") for row in rows),
        "missing_images": missing,
        "dropped": dropped[:SKIPPED_SAMPLE],
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "ok": not missing and bool(exported),
    }
    (dest / "dpo_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest

def _gist(judgement: dict[str, Any]) -> str:
    """Return a one line summary of one judgement for the human reviewer."""

    types = ", ".join(diagnosis_types(judgement)) or "no defect named"
    return f"{pairwise_choice(judgement) or '-'} | {types}"


def _contact_sheet(rows: list[dict[str, Any]], output: Path, *, thumbnail: int = 280) -> dict[str, Any]:
    """Render one contact sheet of the images a pair is actually judged on.

    A pair may carry the image the question is about -- the reference figure of a
    real-vs-generated pair, the query subgraph of a RAG pair. When it does, that image is
    drawn beside the candidates, because "which of these two is the same structure"
    cannot be reviewed by someone who is not shown the structure being asked about. A
    single-image task -- "what is wrong with this figure", "rate this figure" -- carries one
    candidate and still belongs on a sheet, so a sheet is as wide as its widest row.
    """

    from PIL import Image, ImageDraw, ImageFont

    def font(size: int):
        for candidate in (Path(r"<PATH>"), Path(r"<PATH>")):
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size)
        return ImageFont.load_default()

    tiles = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        images = [Path(str(item)) for item in (row.get("images") or [])]
        absent = [str(item) for item in images if not item.exists()]
        if not images or absent:
            skipped.append({"example_id": row.get("example_id"), "images": absent or "none recorded"})
            continue
        query = Path(str(row.get("preview"))) if row.get("preview") else None
        if query is not None and (query in images or not query.exists()):
            # A query that is already a candidate, or that the pair only names as text,
            # must not be drawn twice and must not cost the pair its place on the sheet.
            query = None
        tiles.append((row, query, images))
    if not tiles:
        return {"sheet": "", "pairs": 0, "skipped": skipped}
    with_query = any(query is not None for _, query, _ in tiles)
    columns = max(len(images) for _, _, images in tiles) + (1 if with_query else 0)
    caption_height = 36
    caption_line = 16
    width = thumbnail * columns
    height = thumbnail + caption_height
    sheet = Image.new("RGB", (width, caption_height + height * len(tiles)), "white")
    draw = ImageDraw.Draw(sheet)

    def fit(text: str, size: int) -> tuple[str, Any]:
        """Keep one caption line inside its row: a header wider than the sheet names nothing."""

        handle = font(size)
        if draw.textlength(text, font=handle) <= width - 8:
            return text, handle
        for shorter in range(len(text) - 1, 8, -1):
            clipped = text[:shorter] + "..."
            if draw.textlength(clipped, font=handle) <= width - 8:
                return clipped, handle
        return text[:8] + "...", handle

    for index, (row, query, images) in enumerate(tiles):
        top = caption_height + index * height
        caption = [f"query: {query.name}" if query else "query: not rendered"] if with_query else []
        caption.extend(f"{label}: {image_path.name}" for label, image_path in zip(("A", "B", "C", "D"), images))
        head, head_font = fit(str(row.get("example_id")), 13)
        labels, label_font = fit(" | ".join(caption), 12)
        draw.text((4, top - caption_height + 3), head, fill=(20, 20, 20), font=head_font)
        draw.text((4, top - caption_height + 3 + caption_line), labels, fill=(20, 20, 20), font=label_font)
        for column, image_path in enumerate(([query] if with_query else []) + images):
            if image_path is None:
                continue
            with Image.open(image_path) as opened:
                tile = opened.convert("RGB")
            tile.thumbnail((thumbnail, thumbnail), Image.LANCZOS)
            sheet.paste(tile, (column * thumbnail + (thumbnail - tile.width) // 2, top + (thumbnail - tile.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return {"sheet": str(output), "pairs": len(tiles), "skipped": skipped}


def build_review_queue(pairs_path: str | Path, out_dir: str | Path, *, pairs_per_sheet: int = 8, thumbnail: int = 280, limit: int = 0) -> dict[str, Any]:
    """Write the human review queue and the contact sheets that go with it."""

    rows = _jsonl(Path(pairs_path))
    if limit:
        rows = rows[: int(limit)]
    dest = Path(out_dir)
    queue = []
    for row in rows:
        basis = row.get("preference_basis") if isinstance(row.get("preference_basis"), dict) else {}
        queue.append({
            "pair_id": str(row.get("example_id") or ""),
            "pair_group": str(row.get("pair_group") or ""),
            "split": str(row.get("split") or ""),
            "domain": str(row.get("domain") or ""),
            "task_type": str(row.get("task_type") or ""),
            "family": str((row.get("rejected") or {}).get("family") or ""),
            "known_defects": basis.get("known_defects") or [],
            "images": list(row.get("images") or []),
            "preview": str(row.get("preview") or ""),
            "certified": _gist((row.get("chosen") or {}).get("judgement") or {}),
            "rejected": _gist((row.get("rejected") or {}).get("judgement") or {}),
            "contact_sheet": "",
            "status": "pending",
            "human_audited": bool(basis.get("human_audited")),
        })
    sheets = []
    skipped: list[dict[str, Any]] = []
    per_sheet = max(1, int(pairs_per_sheet))
    for index in range(0, len(rows), per_sheet):
        chunk = rows[index:index + per_sheet]
        sheet = _contact_sheet(chunk, dest / "contact_sheets" / f"sheet_{index // per_sheet:03d}.png", thumbnail=thumbnail)
        if sheet["sheet"]:
            sheets.append(sheet["sheet"])
            for item in queue[index:index + per_sheet]:
                item["contact_sheet"] = sheet["sheet"]
        skipped.extend(sheet["skipped"])
    _write_jsonl(dest / "review_queue.jsonl", queue)
    lines = ["# Judge preference pair review queue", "", f"Pairs: {len(queue)}", ""]
    for sheet in sheets:
        lines.append(f"- {sheet}")
    lines.extend(["", "| pair | split | family | certified | rejected | query | images |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for item in queue:
        lines.append(f"| {item['pair_id']} | {item['split']} | {item['family']} | {item['certified']} | {item['rejected']} | {item['preview']} | {' ; '.join(item['images'])} |")
    (dest / "contact_sheet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "contract": PAIR_CONTRACT,
        "source": str(pairs_path),
        "pairs": len(queue),
        "pending": sum(1 for item in queue if item["status"] == "pending"),
        "already_audited": sum(1 for item in queue if item["human_audited"]),
        "contact_sheets": sheets,
        "pairs_without_images": skipped,
        "queue": str(dest / "review_queue.jsonl"),
        "contact_sheet": str(dest / "contact_sheet.md"),
        "ok": bool(queue),
    }
    (dest / "review_queue.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def apply_human_review(pairs_path: str | Path, decisions_path: str | Path, out_pairs_path: str | Path, *, reviewer: str = "Kira1", reviewed_at: str = "") -> dict[str, Any]:
    """Mark reviewed pairs as human audited and drop the pairs a human rejected.

    ``human_audited`` is only ever set here, from a recorded human decision; the
    builder itself can never set it.
    """

    rows = _jsonl(Path(pairs_path))
    decisions = {}
    for row in _jsonl(Path(decisions_path)):
        key = str(row.get("pair_id") or row.get("example_id") or "")
        if key:
            decisions[key] = row
    kept: list[dict[str, Any]] = []
    approved: list[str] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row.get("example_id") or "")
        decision = decisions.get(pair_id)
        if not decision:
            kept.append(row)
            continue
        verdict = str(decision.get("decision") or "").strip().casefold()
        if verdict in {"approve", "approved", "keep"}:
            approved.append(pair_id)
            kept.append({**row, "preference_basis": {**(row.get("preference_basis") or {}), "human_audited": True, "human_review": {"reviewer": str(decision.get("reviewer") or reviewer), "reviewed_at": str(decision.get("reviewed_at") or reviewed_at), "decision": verdict, "notes": str(decision.get("notes") or "")}}})
            continue
        dropped.append({"example_id": pair_id, "decision": verdict or "unspecified", "notes": str(decision.get("notes") or "")})
    target = Path(out_pairs_path)
    _write_jsonl(target, kept)
    report = {
        "contract": PAIR_CONTRACT,
        "source": str(pairs_path),
        "decisions": len(decisions),
        "approved": len(approved),
        "dropped": dropped,
        "pairs": len(kept),
        "human_audited": sum(1 for row in kept if (row.get("preference_basis") or {}).get("human_audited")),
        "out": str(target),
        "ok": bool(kept),
    }
    target.with_suffix(".review.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
