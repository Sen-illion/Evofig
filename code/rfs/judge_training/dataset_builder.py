"""Build Judge process-supervision data from direct evolution runs only.

One round teaches three things, and the sample envelope keeps them apart:

- ``labels`` is what the model must produce for that task. Only the ``repair`` task is a
  Judge-output target: the loop asks that question twice, so a round yields one advisory
  example (the defects and the bounded repair that was asked for) and one acceptance example
  (the score and the blockers that were reported). Each label is exactly what that role's
  served prompt asked for, and neither carries the other role's fields; ``outcome`` and
  ``trajectory`` are result supervision, whose target is the recorded consequence of the round.
- ``observed`` is what happened after the Judge spoke: the execution, the object diff,
  the score delta, the decision, the reflection. It is audit and supervision material
  and is never part of the repair target, because a Judge cannot know it when it speaks.
- ``process_context`` is what was knowable before the Judge spoke.

``paper`` is what the paper states about its own method: the middleware's method inventory
with the figure's own selection keys stripped (see :mod:`.method_inventory`), so a Judge is
never scored against the taste of a model that was not trained for it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .method_inventory import method_inventory_view, read_method_inventory
from .schemas import normalize_example, validate_example
from .split_ledger import EXPLORATION_SPLIT, resolve_split
from ..analysis.graph import extract_subgraphs
from ..contracts.evolution import ACCEPTANCE_ROLE_FAMILY, ADVISORY_ROLE_FAMILY, judge_role_family, normalize_judge_result
from ..evolution.direct import verify_state_snapshot
from ..evolution.direct.actions import SUPPORTED_ACTIONS
from .preference_pairs import round_outcome
from .prompts import judgement_target

PROCESS_TASKS = ("outcome", "repair", "trajectory")
ROLE_EXPORT_FAMILIES = (ADVISORY_ROLE_FAMILY, ACCEPTANCE_ROLE_FAMILY)
ROLE_EXPORT_FILES = {ADVISORY_ROLE_FAMILY: "advisory", ACCEPTANCE_ROLE_FAMILY: "acceptance"}
EXCLUDED_SPLIT_FILES = ("hidden_test", EXPLORATION_SPLIT)
TRAINABLE_SPLITS = ("train", "validation")
DEFAULT_PROVENANCE_SOURCE = "direct_evolution_process"


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _lacks_trainable_contrast(observed: dict[str, Any]) -> bool:
    """Whether this round is ON==OFF or otherwise has no before/after contrast to learn from.

    A repair that left the figure unchanged cannot teach either role: the advisory answer
    has no consequence, and the acceptance pair is two pictures of the same state.
    """

    if observed.get("figure_changed") is False:
        return True
    diff = observed.get("object_diff") if isinstance(observed.get("object_diff"), dict) else {}
    if diff.get("changed") is False:
        return True
    before = observed.get("state_before") if isinstance(observed.get("state_before"), dict) else {}
    after = observed.get("state_after") if isinstance(observed.get("state_after"), dict) else {}
    return bool(before and after and before == after)


def _role_family(record: dict[str, Any]) -> str:
    metadata = record.get("judge_metadata") if isinstance(record.get("judge_metadata"), dict) else {}
    return judge_role_family(str(metadata.get("judge_role") or record.get("judge_role") or ""))


def _run_id(run: Path) -> str:
    """Name the ground truth a run drew, stably across directories.

    A run reports the paper it processed; when the report does not name one, the archived
    input manifest does. The run directory only stands in as a last resort, so one paper
    recorded in two directories (a real run and a round synthesised from it) still
    resolves to a single ground truth instead of two.
    """

    report = _read(run / "paper_evolve_editable_report.json")
    manifest = _read(run / "input_manifest.json")
    paper = str(
        report.get("paper")
        or report.get("source_case")
        or manifest.get("paper_original")
        or paper_identity(run).get("paper_id")
        or run.resolve()
    )
    return "direct-run-" + hashlib.sha1(paper.encode("utf-8")).hexdigest()[:12]


def paper_identity(run: Path) -> dict[str, Any]:
    """Return the identity a run directory records for the paper it drew."""
    review = _read(run / "paper_review.json")
    identity = review.get("paper_identity") if isinstance(review.get("paper_identity"), dict) else {}
    graph = _read(run / "scientific_graph.json")
    aliases = [
        f"{key}:{value}"
        for key, value in identity.items()
        if key != "title" and isinstance(value, (str, int, float)) and str(value).strip()
    ]
    return {
        "paper_id": str(identity.get("paper_id") or graph.get("paper_id") or ""),
        "title": identity.get("title"),
        "identity_aliases": aliases,
    }


def _preview(run: Path, round_index: int, selected_variant: Any) -> str:
    variant = int(selected_variant or 0)
    candidate = run / f"round_{round_index:02d}" / f"candidate_{variant:02d}"
    for name in ("final_draft_preview.png", "draft_preview.png"):
        path = candidate / name
        if path.is_file():
            return str(path)
    return str(candidate / "draft_preview.png")


def _diagnosis(round_record: dict[str, Any]) -> dict[str, Any]:
    """Return what the two roles said about this round, each read from its own answer.

    The score and the blocker tally are the acceptance answer's, because those are the
    quantities the loop compared; the defect list stays the advisory answer's, because that is
    the question it was asked.
    """

    online = round_record.get("online_judgement") if isinstance(round_record.get("online_judgement"), dict) else {}
    acceptance = _acceptance_answer(round_record)
    return {
        "status": online.get("status"),
        "overall_score": acceptance.get("overall_score"),
        "stage_feedback": list(online.get("stage_feedback") or []),
        "blocking_issue_count": int(acceptance.get("blocking_issue_count") or 0),
    }


def _acceptance_answer(round_record: dict[str, Any]) -> dict[str, Any]:
    """Return the raw acceptance answer of one round, under the key it is recorded with.

    ``frozen_judgement`` is the name older rounds used for the same answer; reading it keeps a
    run recorded before the split exportable instead of silently dropping its score.
    """

    for key in ("acceptance_judgement", "frozen_judgement"):
        value = round_record.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _round_action(raw_action: Any) -> tuple[dict[str, Any], bool]:
    """Return the round action and whether it is a bounded repair action.

    A first round reports ``initial``: that is the initial generation, not a repair
    the Judge chose, so it must never be counted as a controlled repair action.
    """
    action = dict(raw_action) if isinstance(raw_action, dict) else {}
    action_type = str(action.get("type") or action.get("action") or "")
    return action, action_type in SUPPORTED_ACTIONS


def _round_number(record: dict[str, Any], default: int = -1) -> int:
    try:
        value = record.get("round")
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _process_example(run: Path, round_record: dict[str, Any], task_type: str, labels: dict[str, Any], *, paper: dict[str, Any], ground_truth: dict[str, Any], observed: dict[str, Any] | None = None, process_context: dict[str, Any] | None = None, split: str = "", split_resolution: dict[str, Any] | None = None, source: str = DEFAULT_PROVENANCE_SOURCE, judge_role: str = "process_supervision", variant: str = "") -> dict[str, Any]:
    round_index = int(round_record.get("round") or 0)
    example = {
        "example_id": f"{run.name}-{task_type}{f'-{variant}' if variant else ''}-{round_index:04d}",
        "dataset_version": "judge-dataset-v004",
        "ground_truth_id": _run_id(run),
        "task_type": task_type,
        "split": split,
        "preview": _preview(run, round_index, round_record.get("selected_variant")),
        "object_program": str(run / f"round_{round_index:02d}" / f"candidate_{int(round_record.get('selected_variant') or 0):02d}" / "figure_program.json"),
        "labels": labels,
        "observed": observed or {},
        "process_context": process_context or {},
        "paper": paper,
        "ground_truth": ground_truth,
        "provenance": {"run_dir": str(run), "round": round_index, "source": source, "split_resolution": dict(split_resolution or {})},
        # The role is the one whose prompt this label answers: it decides which output schema the
        # export renders, so a row can never be trained against the other role's question.
        "judge_metadata": {"judge_role": judge_role, "trained": False, "human_audited": False},
    }
    return normalize_example(example)


def _declared_source(root: Path) -> str:
    """Return the provenance source a run directory declares for its rounds."""

    return str(_read(root / "paper_evolve_editable_report.json").get("provenance_source") or DEFAULT_PROVENANCE_SOURCE)


def records_from_run(run: str | Path, *, split_index: dict[str, str] | None = None, source: str = "") -> list[dict[str, Any]]:
    """Create outcome, repair, and trajectory records for every direct round that forked.

    A repair round whose figure did not change (ON==OFF, or no recorded object diff) is
    dropped: it cannot teach the advisory role a consequence or give the acceptance role two
    different pictures. ``source`` names where the round came from. Negative rounds
    synthesised offline pass ``controlled_negative_round`` so a sample can never be mistaken
    for a real run observation; otherwise the run directory's own declaration, or the
    real-run default, is used.
    """

    root = Path(run).resolve()
    resolved_source = source or _declared_source(root)
    rounds = _read(root / "evolution_rounds.json").get("rounds") or []
    if not (root / "evolution_events.jsonl").is_file() or not isinstance(rounds, list):
        return []
    events = _jsonl(root / "evolution_events.jsonl")
    memory_rows = _jsonl(root / "judge_experiences.jsonl")
    records: list[dict[str, Any]] = []
    identity = paper_identity(root)
    if split_index is None:
        split, split_resolution = "", {"source": "no_ledger"}
    else:
        resolved, split_resolution = resolve_split(
            split_index,
            paper_id=identity["paper_id"],
            title=identity["title"],
            identity_aliases=identity["identity_aliases"],
        )
        split = resolved or EXPLORATION_SPLIT
    evidence_map = _read(root / "evidence_map.json")
    inventory = read_method_inventory(root)
    scientific_graph = _read(root / "scientific_graph.json")
    scientific_ground_truth = _read(root / "scientific_ground_truth.json")
    scientific_subgraphs = extract_subgraphs(scientific_graph, include_unknown=True)[0] if scientific_graph else []
    paper_evidence = {
        "source_path": evidence_map.get("source_path"),
        "paper_id": scientific_graph.get("paper_id"),
        "evidence_spans": list(evidence_map.get("evidence") or []),
        "method_inventory": method_inventory_view(inventory),
    }
    for raw in rounds:
        if not isinstance(raw, dict):
            continue
        round_index = _round_number(raw, 0)
        snapshot_root = root / f"round_{round_index:02d}"
        selected_variant = int(raw.get("selected_variant") or 0)
        candidate_root = snapshot_root / f"candidate_{selected_variant:02d}"
        before_snapshot = _read(snapshot_root / "state_before.json")
        after_snapshot = _read(candidate_root / "state_after.json")
        state_before = before_snapshot.get("state") if isinstance(before_snapshot.get("state"), dict) else {}
        state_after = after_snapshot.get("state") if isinstance(after_snapshot.get("state"), dict) else {}
        diagnosis = _diagnosis(raw)
        action, action_is_repair = _round_action(raw.get("repair_action"))
        decision_record = _read(snapshot_root / "repair_decision.json") or (raw.get("repair_decision") if isinstance(raw.get("repair_decision"), dict) else {})
        accepted = bool(raw.get("accepted"))
        decision = "accept" if accepted else "rollback"
        repair_gain = float(decision_record.get("score_delta") or 0.0)
        memory = next((item for item in memory_rows if _round_number(item) == round_index), {})
        visual_graph = _read(candidate_root / "visual_graph.json")
        editable_graph = _read(candidate_root / "editable_graph.json")
        subgraphs = {
            "scientific": scientific_subgraphs,
            "visual": extract_subgraphs(visual_graph, include_unknown=True)[0] if visual_graph else [],
            "editable": extract_subgraphs(editable_graph, include_unknown=True)[0] if editable_graph else [],
        }
        action_execution = _read(candidate_root / "action_execution.json") or _read(snapshot_root / "action_execution.json")
        object_diff = _read(candidate_root / "state_diff.json") or action_execution.get("object_diff") or {}
        figure_diff = _read(candidate_root / "figure_object_diff.json") or _read(snapshot_root / "figure_object_diff.json")
        figure_changed = figure_diff.get("changed") if isinstance(figure_diff.get("changed"), bool) else None
        grounding_gate = _read(candidate_root / "grounding_gate.json")
        reflection = _read(snapshot_root / "reflection.json")
        # The round's stage trace rides into the trajectory record: the process-supervision half of
        # the training plan needs the intermediate layers, and a trace not written down here can
        # only be recovered by re-running the round that produced it.
        stage_trace = _read(candidate_root / "stage_trace.json") or _read(snapshot_root / "stage_trace.json")
        audit_checks = {
            "before_snapshot_verified": verify_state_snapshot(before_snapshot),
            "after_snapshot_verified": verify_state_snapshot(after_snapshot),
            "scientific_ground_truth_present": bool(scientific_ground_truth),
            "three_graph_layers_present": bool(scientific_graph and visual_graph and editable_graph),
            "grounding_gate_recorded": bool(grounding_gate),
            "decision_recorded": bool(decision_record),
        }
        audit_status = "machine_verified" if all(audit_checks.values()) else "incomplete"
        # Each answer is normalized under the role that produced it, so a label can never carry
        # what that role's served prompt did not ask for.
        online_raw = raw.get("online_judgement") if isinstance(raw.get("online_judgement"), dict) else {}
        acceptance_raw = _acceptance_answer(raw)
        judgement = normalize_judge_result(online_raw, role=str(online_raw.get("judge_role") or ADVISORY_ROLE_FAMILY))
        acceptance = normalize_judge_result(acceptance_raw, role=str(acceptance_raw.get("judge_role") or ACCEPTANCE_ROLE_FAMILY)) if acceptance_raw else {}
        observed = {
            "action": action,
            "action_is_repair": action_is_repair,
            "action_execution": action_execution,
            "object_diff": object_diff,
            "repair_gain": repair_gain,
            "decision": decision,
            "decision_reason": str(decision_record.get("reason") or ""),
            "injected_defect": decision_record.get("injected_defect") if isinstance(decision_record.get("injected_defect"), dict) else {},
            "reflection": reflection,
            # The comparison the round was decided on, and whether the figure really moved:
            # a round can be accepted with a lower absolute score (or rolled back with a higher
            # one), and only these two fields can tell those apart from a plain improvement.
            "comparison": str(decision_record.get("comparison") or ""),
            "pairwise_verdict": str(decision_record.get("pairwise_verdict") or ""),
            "figure_changed": figure_changed,
            "audit_status": audit_status,
            "audit_checks": audit_checks,
            "state_before": state_before,
            "state_after": state_after,
            "grounding_gate": grounding_gate,
            "subgraphs": subgraphs,
            "bounded_change": raw.get("bounded_change") or {},
            "memory": memory,
            "events": [event for event in events if _round_number(event) == round_index],
            "judge": {"gate_only": bool(judgement.get("gate_only")), "judge_role": judgement.get("judge_role"), "prompt_version": judgement.get("prompt_version")},
        }
        # Derived only from what was recorded, never from anything a role said: it is audit and
        # training material, so it is stamped on the observation and deliberately kept out of
        # ``labels``, which may only carry what the served prompt asked the model for.
        observed["feedback_effective"] = round_outcome(observed)
        if action_is_repair and _lacks_trainable_contrast(observed):
            continue
        trajectory = {
            "paper_evidence": paper_evidence,
            "scientific_ground_truth": scientific_ground_truth,
            "subgraphs": subgraphs,
            "state_before": state_before,
            "diagnosis": diagnosis,
            "action": action,
            "action_is_repair": action_is_repair,
            "state_after": state_after,
            "stage_trace": stage_trace,
            "action_execution": action_execution,
            "object_diff": object_diff,
            "repair_gain": repair_gain,
            "grounding_gate": grounding_gate,
            "decision": decision,
            "reflection": reflection,
            "audit_status": audit_status,
            "audit_checks": audit_checks,
            "bounded_change": raw.get("bounded_change") or {},
            "memory": memory,
            "events": [event for event in events if _round_number(event) == round_index],
        }
        context = {"subgraphs": subgraphs, "state_before": state_before}
        common = {
            "paper": paper_evidence,
            "ground_truth": scientific_ground_truth,
            "split": split,
            "split_resolution": split_resolution,
            "source": resolved_source,
        }
        records.append(_process_example(root, raw, "outcome", {"state_after": state_after, "repair_gain": repair_gain, "action_is_repair": action_is_repair, "grounding_gate": grounding_gate, "decision": decision, "audit_status": audit_status}, process_context={**context, "diagnosis": diagnosis, "action": action, "action_execution": action_execution}, **common))
        records.append(_process_example(root, raw, "repair", judgement_target(judgement, role=str(judgement.get("judge_role") or ADVISORY_ROLE_FAMILY)), observed=observed, process_context={**context, "grounding_gate": grounding_gate, "allowed_scope": raw.get("bounded_change") or {}, "prior_feedback": []}, judge_role=str(judgement.get("judge_role") or "process_supervision"), **common))
        if acceptance:
            # The same round, asked the other question. Its observation block stays empty because
            # what the round did next belongs to the advisory example: this one teaches only what
            # the figure was worth when it was scored.
            records.append(_process_example(root, raw, "repair", judgement_target(acceptance, role=str(acceptance.get("judge_role") or ACCEPTANCE_ROLE_FAMILY)), process_context={**context, "grounding_gate": grounding_gate, "allowed_scope": raw.get("bounded_change") or {}, "prior_feedback": []}, judge_role=str(acceptance.get("judge_role") or "process_supervision"), variant="acceptance", **common))
        records.append(_process_example(root, raw, "trajectory", trajectory, process_context=context, **common))
    return records


def run_dirs(source: Path) -> list[Path]:
    """Return every direct-evolution run directory under one root."""

    if (source / "evolution_events.jsonl").is_file() and (source / "evolution_rounds.json").is_file():
        return [source]
    return sorted({path.parent for path in source.rglob("evolution_events.jsonl") if (path.parent / "evolution_rounds.json").is_file()})


def build_dataset(runs_root: str | Path, out: str | Path, *, dataset_version: str = "judge-dataset-v004", split_index: dict[str, str] | None = None) -> dict[str, Any]:
    """Build combined, task-specific, and role-split JSONL files from direct trajectories.

    Repair examples are also written to ``advisory.jsonl`` and ``acceptance.jsonl`` by
    ``judge_role_family``. Rounds with no ON/OFF contrast never reach this writer.
    With a split index, every sample inherits the frozen split of the paper it came
    from and the training view keeps only ``train`` and ``validation`` rows;
    ``hidden_test`` and ``exploration`` rows are written next to it but never into
    the training files. Without an index the earlier behaviour is kept unchanged.
    """
    source = Path(runs_root).resolve()
    requested = Path(out).resolve()
    if requested.suffix.casefold() == ".jsonl":
        target = requested
        task_root = target.parent
        task_prefix = target.stem + "_"
    else:
        task_root = requested
        target = task_root / "judge_examples.jsonl"
        task_prefix = ""
    task_root.mkdir(parents=True, exist_ok=True)
    runs = run_dirs(source)
    records = [record for run in runs for record in records_from_run(run, split_index=split_index)]
    valid: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    excluded: dict[str, int] = {name: 0 for name in EXCLUDED_SPLIT_FILES}
    handles = {task: (task_root / f"{task_prefix}{task}.jsonl").open("w", encoding="utf-8") for task in PROCESS_TASKS}
    role_handles = {family: (task_root / f"{task_prefix}{ROLE_EXPORT_FILES[family]}.jsonl").open("w", encoding="utf-8") for family in ROLE_EXPORT_FAMILIES}
    excluded_handles: dict[str, Any] = {}
    if split_index is not None:
        excluded_handles = {name: (task_root / f"{task_prefix}{name}.jsonl").open("w", encoding="utf-8") for name in EXCLUDED_SPLIT_FILES}
    try:
        with target.open("w", encoding="utf-8") as combined:
            for record in records:
                record["dataset_version"] = dataset_version
                record_errors = validate_example(record)
                if record_errors:
                    errors.append({"example_id": record.get("example_id"), "errors": record_errors})
                    continue
                line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                accepted.append(record)
                split = str(record.get("split") or "")
                if split_index is not None and split not in TRAINABLE_SPLITS:
                    name = split if split in excluded_handles else EXPLORATION_SPLIT
                    excluded[name] += 1
                    excluded_handles[name].write(line)
                    continue
                combined.write(line)
                handles[record["task_type"]].write(line)
                if str(record.get("task_type") or "") == "repair":
                    family = _role_family(record)
                    if family in role_handles:
                        role_handles[family].write(line)
                valid.append(record)
    finally:
        for handle in handles.values():
            handle.close()
        for handle in role_handles.values():
            handle.close()
        for handle in excluded_handles.values():
            handle.close()
    task_counts = {task: sum(item["task_type"] == task for item in valid) for task in PROCESS_TASKS}
    role_counts = {ROLE_EXPORT_FILES[family]: sum(str(item.get("task_type") or "") == "repair" and _role_family(item) == family for item in valid) for family in ROLE_EXPORT_FAMILIES}
    split_counts: dict[str, int] = {}
    for item in accepted:
        name = str(item.get("split") or "unassigned")
        split_counts[name] = split_counts.get(name, 0) + 1
    manifest = {
        "summary": "Direct evolution process-supervision dataset built.",
        "dataset_version": dataset_version,
        "source_root": str(source),
        "out": str(target),
        "records": len(valid),
        "runs": len(runs),
        "errors": errors,
        "task_counts": task_counts,
        "role_counts": role_counts,
        "role_files": [f"{task_prefix}{ROLE_EXPORT_FILES[family]}.jsonl" for family in ROLE_EXPORT_FAMILIES],
        "forked_trajectories": role_counts["advisory"] > 0 and role_counts["acceptance"] > 0,
        "judge_status": "process_supervision",
        "split_filtering": "split_ledger" if split_index is not None else "off",
        "split_counts": dict(sorted(split_counts.items())),
        "trainable_splits": list(TRAINABLE_SPLITS),
        "excluded_counts": excluded,
        "excluded_files": [f"{task_prefix}{name}.jsonl" for name in EXCLUDED_SPLIT_FILES] if split_index is not None else [],
    }
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate_dataset(path: str | Path) -> dict[str, Any]:
    """Check one built dataset file and report the label shapes it actually carries.

    ``ok`` only tracks schema errors. The extra counts and ``warnings`` describe what
    the file is able to teach: a file with no rollback decision and no controlled repair
    action cannot supervise repair behaviour, which is reported without failing the build.
    """
    source = Path(path)
    if source.is_dir():
        source = source / "judge_examples.jsonl"
    records = _jsonl(source)
    errors = []
    for index, record in enumerate(records, start=1):
        item_errors = validate_example(record)
        if item_errors:
            errors.append({"line": index, "errors": item_errors})
    trajectory_records = [record for record in records if str(record.get("task_type") or "") == "trajectory"]
    counted = trajectory_records or records
    decisions = {"accept": 0, "rollback": 0}
    repair_actions = 0
    for record in counted:
        labels = record.get("labels") if isinstance(record.get("labels"), dict) else {}
        decision = str(labels.get("decision") or "")
        if decision in decisions:
            decisions[decision] += 1
        if labels.get("action_is_repair") is True:
            repair_actions += 1
    warnings = []
    if counted and not decisions["rollback"]:
        warnings.append("no rollback negative: every recorded decision is accept")
    if counted and not repair_actions:
        warnings.append("no controlled repair action: rounds only report the initial action")
    return {"summary": "Direct process dataset validation.", "path": str(source.resolve()), "records": len(records), "errors": errors, "warnings": warnings, "decisions": decisions, "repair_actions": repair_actions, "ok": not errors}
