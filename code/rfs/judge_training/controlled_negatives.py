"""Synthesise the measured rollback rounds the direct pipeline never produced.

Thirteen real rounds accepted thirteen times, so process supervision held no example of
"that request made the figure worse" and could not teach when to roll back. This module
builds the missing side offline from an existing run, without touching the round loop.

One controlled round is produced like this:

1. read the source round's own ``state_before`` and the candidate program it compiled;
2. pick a scripted request: a legal bounded action carrying a parameter the shipped
   controller never sets (a shift or a resize far past the budget), plus the structural
   damage the action would compile when it is honoured;
3. execute it for real with the pipeline's own executors -- ``apply_bounded_action`` and
   ``apply_program_action`` -- and rebuild the candidate through the composition
   primitives the loop uses (``arrange_node_contents``, ``resolve_direct_connectors``);
4. rebuild the visual and editable graphs, the layer alignment and the Grounding Gate with
   the same functions the loop calls;
5. decide with ``compare_candidate`` and ``diff_state_snapshots`` against the undamaged
   candidate projected through that same path.

No gate verdict, score, or decision is written by hand: every number in the round is
measured, and a script is admitted only when it really made the candidate worse. When no
target of a script regresses, the round is reported as unusable instead of fabricated.

Two families of script ship here. The geometry family only moves cards, so the real gate
keeps answering ``pass`` and the deterministic Judge measures the loss. The structural
family compiles a request the bounded vocabulary has no executor for -- adding a label the
scientific graph forbids -- and the real Grounding Gate answers ``blocked`` for it, with no
invented evidence. Dropping a stage moves the Judge score without moving the gate, because
the frozen scientific graphs declare no required node.

A request that changes nothing measurable is deliberately not shipped as a script:
retargeting a connector leaves every frozen graph's verdict untouched, since no scientific
edge in them is declared required, and admitting it would mean writing the verdict by hand.

The written round keeps the source candidate's preview: geometry and graphs are rebuilt
from the damaged plan, but nothing here re-renders a picture. ``controlled_negative.json``
records that, together with the source run, the script, the target and both measurements.
"""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .dataset_builder import paper_identity, run_dirs
from .method_inventory import INVENTORY_FILES, PLAN_FILES
from .split_ledger import EXPLORATION_SPLIT, build_split_index, load_split_ledger, resolve_split
from ..composition.direct_connectors import resolve_direct_connectors
from ..composition.editable_graph import build_editable_graph
from ..composition.node_layout import arrange_node_contents
from ..contracts.evolution import normalize_judge_result, validate_judge_result
from ..evaluation.direct_judges import DirectJudgeInput, HeuristicDirectJudge
from ..evaluation.geometry_quality import evaluate_density_quality, evaluate_geometry_quality
from ..evaluation.graph_alignment import align_graph_layers
from ..evaluation.grounding_gate import evaluate_grounding_gate
from ..evolution.direct.actions import apply_program_action
from ..evolution.direct.controller import apply_bounded_action, compare_candidate, default_reflection
from ..evolution.direct.snapshots import create_state_snapshot, diff_state_snapshots
from ..planning.visual_graph import build_visual_graph
from ..utils import read_json, write_json

CONTROLLED_NEGATIVE_SOURCE = "controlled_negative_round"
CONTROLLED_NEGATIVE_REASON = "controlled_negative"
REPORT_NAME = "controlled_negative.json"
TRAINABLE_SPLITS = ("train", "validation")
SHARED_ROOT_FILES = (
    "paper_evolve_editable_report.json",
    "input_manifest.json",
    "paper_review.json",
    "scientific_graph.json",
    "scientific_ground_truth.json",
    "evidence_map.json",
    # A Judge sample reads the paper's own method list, so the round carries both
    # halves of the middleware contract too; a source run that stored a half under
    # neither of its names simply contributes nothing here.
    *INVENTORY_FILES,
    *PLAN_FILES,
)


@dataclass(frozen=True)
class NegativeScript:
    """One scripted out-of-budget request and why the pipeline should refuse it."""

    name: str
    action_type: str
    scope_type: str
    request: str
    note: str
    parameters: dict[str, Any]
    targets: str
    program_damage: str = ""


NEGATIVE_SCRIPTS: tuple[NegativeScript, ...] = (
    NegativeScript(
        name="shift_lead_node_into_neighbour",
        action_type="relayout_slots",
        scope_type="slot",
        request="the leading card should be shifted across its panel to close the gap",
        note="shifting a card past its slot budget slides it onto its neighbour",
        parameters={"factor": 1.0, "dx": 0.2},
        targets="panel_lead",
    ),
    NegativeScript(
        name="grow_panel_past_budget",
        action_type="resize_nodes",
        scope_type="slot",
        request="every card of the panel should be enlarged to fill the canvas",
        note="growing a whole panel past its budget overlaps the cards it holds",
        parameters={"factor": 3.4},
        targets="panel_all",
    ),
    NegativeScript(
        name="shrink_panel_below_budget",
        action_type="resize_nodes",
        scope_type="slot",
        request="every card of the panel should be shrunk to free space",
        note="shrinking a whole panel far below its budget leaves the figure too sparse",
        parameters={"factor": 0.2},
        targets="panel_all",
    ),
    NegativeScript(
        name="insert_forbidden_label",
        # Adding a labelled component is the structure planner's job, and it is the mirror of
        # ``drop_a_stage`` below. This script used to be labelled ``recompile``, which no
        # longer names a repair the controller will run.
        action_type="replan_structure",
        scope_type="panel",
        request="one more labelled component should be added to the figure",
        note="the compiled label names a mechanism the paper never supports, which the Grounding Gate forbids",
        parameters={},
        targets="forbidden_label",
        program_damage="forbidden_label",
    ),
    NegativeScript(
        name="drop_a_stage",
        action_type="replan_structure",
        scope_type="module",
        request="the last stage should be replanned away to free space",
        note="dropping a stage removes evidence the paper's own flow needs",
        parameters={"factor": 1.0, "dx": 0.0, "dy": 0.0},
        targets="node_each",
        program_damage="drop_node",
    ),
)
SCRIPT_LIMIT = 3
PROGRAM_DAMAGE_KINDS = ("forbidden_label", "drop_node")


def negative_script(name: str) -> NegativeScript:
    """Return one script by name."""

    for script in NEGATIVE_SCRIPTS:
        if script.name == name:
            return script
    raise KeyError(name)


def _rounds(root: Path) -> list[dict[str, Any]]:
    rounds = read_json(root / "evolution_rounds.json").get("rounds")
    return [item for item in rounds if isinstance(item, dict)] if isinstance(rounds, list) else []


def _candidate_dir(root: Path, record: dict[str, Any]) -> Path:
    return root / f"round_{int(record.get('round') or 0):02d}" / f"candidate_{int(record.get('selected_variant') or 0):02d}"


def source_rounds(run: str | Path) -> list[int]:
    """Return the round indices of a run that compiled a candidate to damage."""

    root = Path(run).resolve()
    return [int(record.get("round") or 0) for record in _rounds(root) if ( _candidate_dir(root, record) / "figure_program.json").is_file()]


def _panel_nodes(program: dict[str, Any]) -> list[list[dict[str, Any]]]:
    panels: dict[str, list[dict[str, Any]]] = {}
    for node in program.get("semantic_nodes") or []:
        if isinstance(node, dict) and isinstance(node.get("bbox_percent"), dict):
            panels.setdefault(str(node.get("panel_id") or ""), []).append(node)
    ordered = [sorted(nodes, key=lambda item: float(item["bbox_percent"]["x"])) for nodes in panels.values() if len(nodes) >= 2]
    ordered.sort(key=len, reverse=True)
    return ordered


def script_targets(script: NegativeScript, program: dict[str, Any], scientific_graph: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """Return every target a script may try, as the action arguments and the damage target.

    The action arguments are merged into the recorded action. The damage target says what
    the scripted request would damage: a node id, or a label the scientific graph forbids.
    """

    panels = _panel_nodes(program)
    if script.targets == "panel_lead":
        return [({"object_ids": [nodes[0]["id"]], "semantic_node_ids": [nodes[0]["id"]]}, str(nodes[0]["id"])) for nodes in panels[:SCRIPT_LIMIT]]
    if script.targets == "panel_all":
        ids = [node["id"] for node in nodes]
        return [({"object_ids": ids, "semantic_node_ids": ids}, "") for nodes in panels[:SCRIPT_LIMIT]]
    if script.targets == "node_each":
        return [({"object_ids": [node.get("id")], "semantic_node_ids": [node.get("id")]}, str(node.get("id") or "")) for node in program.get("semantic_nodes") or [] if isinstance(node, dict)]
    if script.targets == "forbidden_label":
        labels = _forbidden_labels(scientific_graph)
        return [({}, label) for label in labels] or [({}, "")]
    return [({}, "")]


def _forbidden_labels(scientific_graph: dict[str, Any]) -> list[str]:
    """Return the labels the paper's own claim explicitly rules out."""

    return [str(value).strip() for value in scientific_graph.get("forbidden_labels") or [] if str(value).strip()]


def damage_program(program: dict[str, Any], scientific_graph: dict[str, Any], script: NegativeScript, target: str) -> tuple[dict[str, Any] | None, str]:
    """Compile the damage one scripted request would do, or report why it cannot be done.

    The bounded executors move geometry only, so a structural request is compiled here and
    then measured by the pipeline's own graph builders, gate and Judge. Returning ``None``
    means the source run offers nothing to damage, and the request is refused rather than
    rearranged into something that happens to look worse.
    """

    if not script.program_damage:
        return deepcopy(program), ""
    damaged = deepcopy(program)
    if script.program_damage == "forbidden_label":
        if not target:
            return None, "the scientific graph declares no forbidden label to inject"
        panels = [str(item.get("id") or "") for item in damaged.get("panels") or [] if isinstance(item, dict)]
        labels = damaged.setdefault("labels", [])
        if not isinstance(labels, list):
            return None, "the candidate program has no label list to add to"
        labels.append({"id": f"label_neg_{len(labels) + 1:02d}", "label": target, "role": "module_label", "panel_id": panels[0] if panels else ""})
        return damaged, f"a label reading {target!r} was compiled into the figure"
    if script.program_damage == "drop_node":
        nodes = [item for item in damaged.get("semantic_nodes") or [] if isinstance(item, dict)]
        if target not in {str(item.get("id") or "") for item in nodes}:
            return None, f"the candidate program has no node {target!r} to drop"
        damaged["semantic_nodes"] = [item for item in nodes if str(item.get("id") or "") != target]
        damaged["arrows"] = [
            item
            for item in damaged.get("arrows") or []
            if isinstance(item, dict) and target not in {str(item.get("source") or ""), str(item.get("target") or ""), str(item.get("source_id") or ""), str(item.get("target_id") or "")}
        ]
        return damaged, f"the stage {target!r} and its connectors were dropped"
    raise KeyError(script.program_damage)


def project_candidate(state: dict[str, Any], program: dict[str, Any], scientific_graph: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one candidate the way the loop does after an action has been executed."""

    layout = state.get("layout") if isinstance(state.get("layout"), dict) else {}
    layout_nodes = {str(item.get("id") or ""): item for item in layout.get("nodes") or [] if isinstance(item, dict)}
    rebuilt = deepcopy(program)
    for node in rebuilt.get("semantic_nodes") or []:
        if not isinstance(node, dict):
            continue
        source = layout_nodes.get(str(node.get("id") or ""))
        if source and isinstance(source.get("bbox_percent"), dict):
            node["bbox_percent"] = dict(source["bbox_percent"])
    rebuilt, _ = arrange_node_contents(rebuilt)
    rebuilt, connector_report = resolve_direct_connectors(rebuilt)
    graph_nodes = [
        {**layout_nodes.get(str(item.get("id") or ""), {}), **item, "semantic_source_id": layout_nodes.get(str(item.get("id") or ""), {}).get("semantic_source_id") or item.get("semantic_source_id") or item.get("id")}
        for item in rebuilt.get("semantic_nodes") or []
        if isinstance(item, dict)
    ]
    visual_graph, visual_report = build_visual_graph(
        {
            "canvas": layout.get("canvas") or {},
            "topology": layout.get("topology"),
            "panels": rebuilt.get("panels") or layout.get("panels") or [],
            "nodes": graph_nodes,
            "connectors": rebuilt.get("arrows") or [],
        },
        scientific_graph=scientific_graph,
    )
    editable_graph, bindings, editable_report = build_editable_graph(rebuilt, visual_graph=visual_graph, scientific_graph=scientific_graph)
    alignment_report = align_graph_layers(scientific_graph, visual_graph, editable_graph, bindings)
    grounding_gate = evaluate_grounding_gate(alignment_report)
    judge_input = DirectJudgeInput(
        method=state.get("method") or {},
        layout=layout,
        icons=state.get("icons") or {},
        program=rebuilt,
        grounding_gate=grounding_gate,
    )
    # Two questions, two answers: the advisory result says what is wrong with this state,
    # the acceptance result measures how good it is. Only the second one can be compared
    # with another measurement, so the decision below reads that one.
    judgement = normalize_judge_result(HeuristicDirectJudge().evaluate(judge_input))
    acceptance = normalize_judge_result(HeuristicDirectJudge(frozen=True).evaluate(judge_input))
    return {
        "program": rebuilt,
        "visual_graph": visual_graph,
        "editable_graph": editable_graph,
        "bindings": bindings,
        "alignment_report": alignment_report,
        "grounding_gate": grounding_gate,
        "judgement": judgement,
        "acceptance": acceptance,
        "geometry_report": evaluate_geometry_quality(rebuilt),
        "density_report": evaluate_density_quality(rebuilt),
        "connector_report": connector_report,
        "visual_report": visual_report,
        "editable_report": editable_report,
    }


def _regressed(before: dict[str, Any], after: dict[str, Any], decision: dict[str, Any]) -> bool:
    """Report whether executing the request really left the candidate worse."""

    if after["grounding_gate"]["status"] != "pass" and before["grounding_gate"]["status"] == "pass":
        return True
    if decision.get("accepted"):
        return False
    return bool(decision.get("new_blockers")) or float(decision.get("score_delta") or 0.0) < 0.0


def measure_negative(state: dict[str, Any], program: dict[str, Any], scientific_graph: dict[str, Any], action: dict[str, Any], *, round_index: int, damaged_program: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute one action for real and measure the outcome against the intact candidate.

    ``damaged_program`` carries the structural damage the request would compile. Geometry
    always travels through the shipped executors; the structure is compiled by the caller
    because no bounded executor can express it.
    """

    before = project_candidate(state, program, scientific_graph)
    damaged_layout, damaged_icons, executor_report = apply_bounded_action(state.get("layout") or {}, state.get("icons") or {}, action)
    damaged_state = {**state, "layout": damaged_layout, "icons": damaged_icons}
    damaged_state, program_report = apply_program_action(damaged_state, action)
    after = project_candidate(damaged_state, damaged_program if damaged_program is not None else program, scientific_graph)
    decision = compare_candidate(before["acceptance"], after["acceptance"])
    before_snapshot = create_state_snapshot({"method": state.get("method") or {}, "layout": state.get("layout") or {}, "icons": state.get("icons") or {}}, label="before_controlled_action", round_index=round_index)
    after_snapshot = create_state_snapshot(
        {
            "method": damaged_state.get("method") or {},
            "layout": damaged_state.get("layout") or {},
            "icons": damaged_state.get("icons") or {},
            "program": after["program"],
            "grounding_gate": after["grounding_gate"],
        },
        label="after_controlled_action",
        round_index=round_index,
    )
    return {
        "before": before,
        "after": after,
        "decision": decision,
        "before_snapshot": before_snapshot,
        "after_snapshot": after_snapshot,
        "object_diff": diff_state_snapshots(before_snapshot, after_snapshot),
        "bounded_change": {
            "contract": "rfs.direct_action_execution",
            "version": 1,
            "action": action,
            "action_type": str(action.get("type") or ""),
            "applied": bool(executor_report.get("changed_node_ids") or executor_report.get("changed_asset_ids") or program_report.get("applied")),
            "changed_node_ids": executor_report.get("changed_node_ids") or [],
            "changed_asset_ids": executor_report.get("changed_asset_ids") or [],
            "program_changes": program_report.get("program_changes") or [],
            "object_diff": {},
            "preserve_checks": {"scientific_contract_unchanged": True, "input_state_unmodified": True},
            "retry_index": 0,
        },
        "regressed": False,
    }


def _write_run(root: Path, source_root: Path, record: dict[str, Any], measurement: dict[str, Any], script: NegativeScript, action: dict[str, Any], *, damage_note: str, preview_source: Path) -> None:
    """Write a run-shaped directory holding only the controlled round."""

    round_index = int(record.get("round") or 0)
    round_dir = root / f"round_{round_index:02d}"
    candidate_dir = round_dir / "candidate_00"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for name in SHARED_ROOT_FILES:
        source = source_root / name
        if not source.is_file():
            continue
        if name == "paper_evolve_editable_report.json":
            report = read_json(source)
            report["provenance_source"] = CONTROLLED_NEGATIVE_SOURCE
            report["source_run"] = str(source_root)
            report["source_round"] = round_index
            write_json(root / name, report)
            continue
        shutil.copyfile(source, root / name)
    if preview_source.is_file():
        shutil.copyfile(preview_source, candidate_dir / preview_source.name)
    after = measurement["after"]
    decision = {**measurement["decision"], "reason": CONTROLLED_NEGATIVE_REASON, "controlled_reason": measurement["decision"].get("reason"), "injected_defect": {
        "kind": script.name,
        "action": action,
        "target": str(action.get("object_ids") or action.get("semantic_node_ids") or ""),
        "request": script.request,
        "why_harmful": script.note,
        "damage": damage_note,
        "note": "The pipeline compiled a repair that the bounded controller never issues.",
    }}
    write_json(round_dir / "state_before.json", measurement["before_snapshot"])
    write_json(round_dir / "state_after_action.json", measurement["after_snapshot"])
    write_json(round_dir / "action_execution.json", {**measurement["bounded_change"], "object_diff": measurement["object_diff"]})
    write_json(round_dir / "repair_decision.json", decision)
    write_json(round_dir / "reflection.json", default_reflection(decision, action, retry_index=0))
    write_json(candidate_dir / "state_after.json", measurement["after_snapshot"])
    write_json(candidate_dir / "state_diff.json", measurement["object_diff"])
    write_json(candidate_dir / "action_execution.json", {**measurement["bounded_change"], "object_diff": measurement["object_diff"]})
    write_json(candidate_dir / "figure_program.json", after["program"])
    write_json(candidate_dir / "visual_graph.json", after["visual_graph"])
    write_json(candidate_dir / "visual_graph_report.json", after["visual_report"])
    write_json(candidate_dir / "editable_graph.json", after["editable_graph"])
    write_json(candidate_dir / "editable_graph_report.json", after["editable_report"])
    write_json(candidate_dir / "graph_bindings.json", after["bindings"])
    write_json(candidate_dir / "graph_alignment_report.json", after["alignment_report"])
    write_json(candidate_dir / "grounding_gate.json", after["grounding_gate"])
    write_json(candidate_dir / "geometry_quality_report.json", after["geometry_report"])
    write_json(candidate_dir / "density_quality_report.json", after["density_report"])
    frozen = normalize_judge_result(
        HeuristicDirectJudge(frozen=True).evaluate(
            DirectJudgeInput(
                method=measurement["after_snapshot"]["state"].get("method") or {},
                layout=measurement["after_snapshot"]["state"].get("layout") or {},
                icons=measurement["after_snapshot"]["state"].get("icons") or {},
                program=after["program"],
                grounding_gate=after["grounding_gate"],
            )
        )
    )
    write_json(root / "evolution_rounds.json", {"rounds": [{
        "round": round_index,
        "candidate_count": 1,
        "selected_variant": 0,
        "online_judgement": after["judgement"],
        # The measurement the decision was taken with, recorded under its own name: the advisory
        # answer says what is wrong with the damaged candidate, this one says what it is worth.
        "acceptance_judgement": frozen,
        "frozen_judgement": frozen,
        "accepted": False,
        "repair_action": action,
        "repair_decision": decision,
        "bounded_change": measurement["bounded_change"],
        "reflection": read_json(round_dir / "reflection.json"),
    }]})
    events = [
        {"seq": 1, "event": "round_started", "round": round_index, "action": action, "retry_index": 0},
        {"seq": 2, "event": "judge_feedback_recorded", "round": round_index, "role": after["judgement"].get("judge_role"), "judgement": after["judgement"]},
        {"seq": 3, "event": "candidate_rolled_back", "round": round_index, "score": measurement["decision"].get("score_delta"), "action": action, "reason": CONTROLLED_NEGATIVE_REASON},
    ]
    (root / "evolution_events.jsonl").write_text("".join(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n" for event in events), encoding="utf-8")


def build_negative_round(run: str | Path, out: str | Path, *, round_index: int, script: NegativeScript) -> dict[str, Any]:
    """Try one script against one source round; write the round only when it regressed."""

    source_root = Path(run).resolve()
    record = next((item for item in _rounds(source_root) if int(item.get("round") or 0) == round_index), None)
    if record is None:
        return {"ok": False, "reason": "the run records no such round", "round": round_index, "script": script.name}
    candidate_dir = _candidate_dir(source_root, record)
    program = read_json(candidate_dir / "figure_program.json")
    scientific_graph = read_json(source_root / "scientific_graph.json")
    state = read_json(source_root / f"round_{round_index:02d}" / "state_before.json").get("state")
    if not isinstance(program, dict) or not program.get("semantic_nodes") or not isinstance(state, dict) or not isinstance(scientific_graph, dict):
        return {"ok": False, "reason": "the source round has no compiled candidate or no scientific graph", "round": round_index, "script": script.name}
    tried: list[dict[str, Any]] = []
    for target, damage_target in script_targets(script, program, scientific_graph):
        damaged_program, damage_note = damage_program(program, scientific_graph, script, damage_target)
        if damaged_program is None:
            tried.append({"target": damage_target, "reason": damage_note, "regressed": False})
            continue
        action = {"type": script.action_type, "scope_type": script.scope_type, "problem": script.request, **script.parameters, **target}
        measurement = measure_negative(state, program, scientific_graph, action, round_index=round_index, damaged_program=damaged_program)
        measurement["regressed"] = _regressed(measurement["before"], measurement["after"], measurement["decision"])
        measurement["damage_note"] = damage_note
        tried.append({"action": action, "damage": damage_note, "score_delta": measurement["decision"].get("score_delta"), "reason": measurement["decision"].get("reason"), "regressed": measurement["regressed"]})
        if not measurement["regressed"]:
            continue
        root = Path(out).resolve() / f"{source_root.name}-neg-{script.name}"
        _write_run(root, source_root, record, measurement, script, action, damage_note=damage_note, preview_source=candidate_dir / "draft_preview.png")
        report = {
            "contract": "rfs.controlled_negative",
            "version": 1,
            "source": CONTROLLED_NEGATIVE_SOURCE,
            "source_run": str(source_root),
            "source_round": round_index,
            "run": str(root),
            "script": script.name,
            "action": action,
            "injected_defect": {"kind": script.name, "request": script.request, "why_harmful": script.note, "target": action.get("object_ids") or action.get("semantic_node_ids") or damage_target, "damage": damage_note},
            "before_score": measurement["before"]["acceptance"].get("overall_score"),
            "after_score": measurement["after"]["acceptance"].get("overall_score"),
            "before_gate": measurement["before"]["grounding_gate"]["status"],
            "after_gate": measurement["after"]["grounding_gate"]["status"],
            "decision": measurement["decision"],
            "geometry": {"before": measurement["before"]["geometry_report"]["status"], "after": measurement["after"]["geometry_report"]["status"]},
            "density": {"before": measurement["before"]["density_report"]["status"], "after": measurement["after"]["density_report"]["status"]},
            "preview_source": str(candidate_dir / "draft_preview.png"),
            "render_note": "geometry, graphs and the gate are rebuilt from the damaged plan; no new picture is rendered, so the round reuses the source preview.",
            "tried": tried,
            "ok": True,
        }
        write_json(root / REPORT_NAME, report)
        return report
    measured = [item for item in tried if item.get("action")]
    reason = "no target of the script measurably made the candidate worse" if measured else str(tried[0].get("reason") or "") if tried else "the script offered no target for this run"
    return {"ok": False, "reason": reason, "round": round_index, "script": script.name, "tried": tried}


def build_controlled_negatives(runs: str | Path, out: str | Path, *, round_index: int | None = None, scripts: Sequence[str] | None = None, split: str | Path | None = None, limit: int = 0) -> dict[str, Any]:
    """Build every admissible controlled round under one runs root.

    A source paper that the frozen split ledger places in ``hidden_test`` or
    ``exploration`` is skipped: synthesised negatives may only ever teach the training and
    validation views.
    """

    chosen = [negative_script(name) for name in (scripts or [script.name for script in NEGATIVE_SCRIPTS])]
    split_index = build_split_index(load_split_ledger(split)) if split else None
    rounds: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for run in run_dirs(Path(runs).resolve()):
        identity = paper_identity(run)
        resolved, split_resolution = resolve_split(split_index, paper_id=identity["paper_id"], title=identity["title"], identity_aliases=identity["identity_aliases"]) if split_index is not None else ("", {"source": "no_ledger"})
        if split_index is not None and resolved not in TRAINABLE_SPLITS:
            skipped.append({"run": str(run), "reason": f"split {resolved or EXPLORATION_SPLIT} is not trainable", "split_resolution": split_resolution})
            continue
        indices = [round_index] if round_index is not None else source_rounds(run)
        for index in indices:
            if limit and len(rounds) >= int(limit):
                break
            for script in chosen:
                result = build_negative_round(run, out, round_index=int(index), script=script)
                if result.get("ok"):
                    rounds.append({**result, "split": resolved, "split_resolution": split_resolution})
                    break
                failures.append(result)
    manifest = {
        "contract": "rfs.controlled_negatives",
        "version": 1,
        "runs_root": str(Path(runs).resolve()),
        "out": str(Path(out).resolve()),
        "scripts": [script.name for script in chosen],
        "rounds": len(rounds),
        "skipped": skipped,
        "failures": failures,
        "source": CONTROLLED_NEGATIVE_SOURCE,
        "results": [{key: value for key, value in item.items() if key in {"run", "source_run", "source_round", "script", "split", "before_score", "after_score", "action"}} for item in rounds],
    }
    Path(out).resolve().mkdir(parents=True, exist_ok=True)
    write_json(Path(out).resolve() / "controlled_negatives.json", manifest)
    return manifest
