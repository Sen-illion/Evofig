"""Tests for the controlled negative round production line.

Thirteen real rounds accepted thirteen times, so the process dataset held no example of a
repair that made a figure worse. These tests pin down that the synthesised round is shaped
like a real one, that its verdicts come from the pipeline's own gate, Judge and controller,
and that a script which cannot be measured is refused instead of written.
"""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import create_state_snapshot, diff_state_snapshots
from rfs.evolution.direct.actions import SUPPORTED_ACTIONS
from rfs.judge_training import build_dataset, records_from_run, validate_dataset
from rfs.judge_training.controlled_negatives import (
    CONTROLLED_NEGATIVE_SOURCE,
    NEGATIVE_SCRIPTS,
    build_controlled_negatives,
    build_negative_round,
    negative_script,
    project_candidate,
    source_rounds,
)
from rfs.judge_training.preference_pairs import build_preference_pairs
from rfs.judge_training.method_inventory import SELECTION_KEYS
from rfs.utils import write_json

FORBIDDEN = "unsupported external knowledge base"


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _node(node_id: str, x: float) -> dict:
    return {
        "id": node_id,
        "label": node_id.replace("_", " ").title(),
        "panel_id": "module_row",
        "bbox_percent": {"x": x, "y": 0.2, "w": 0.2, "h": 0.2},
        "shape_kind": "rounded_rect",
        "editable_in": "pptx",
        "object_type": "icon_with_label",
        "z_index": 30,
    }


def _program() -> dict:
    return {
        "canvas": {"width_percent": 1.0, "height_percent": 1.0},
        "panels": [],
        "semantic_nodes": [_node("module_01", 0.06), _node("module_02", 0.36), _node("module_03", 0.66)],
        "arrows": [
            {"id": "edge_01", "source": "module_01", "target": "module_02", "source_id": "module_01", "target_id": "module_02", "type": "data_flow", "semantic_role": "module_flow", "path_percent": [[0.26, 0.3], [0.36, 0.3]]},
            {"id": "edge_02", "source": "module_02", "target": "module_03", "source_id": "module_02", "target_id": "module_03", "type": "data_flow", "semantic_role": "module_flow", "path_percent": [[0.56, 0.3], [0.66, 0.3]]},
        ],
        "labels": [],
        "cards": [],
        "slots": [],
        "groups": [],
        "control_shapes": [],
        "text_program": {"items": []},
    }


def _scientific_graph(*, forbidden: tuple[str, ...] = (FORBIDDEN,)) -> dict:
    return {
        "contract": "rfs.scientific_graph",
        "version": 1,
        "graph_id": "sg",
        "paper_id": "p",
        "kind": "scientific",
        "nodes": [
            {"id": "sn1", "kind": "module", "label": "A", "role": "module", "required": False, "properties": {"source_contract_id": "module_01"}},
            {"id": "sn2", "kind": "module", "label": "B", "role": "module", "required": False, "properties": {"source_contract_id": "module_02"}},
            {"id": "sn3", "kind": "output", "label": "C", "role": "output", "required": False, "properties": {"source_contract_id": "module_03"}},
        ],
        "edges": [
            {"id": "se1", "source": "sn1", "target": "sn2", "relation_type": "data_flow", "properties": {"source_node_contract_id": "module_01", "target_node_contract_id": "module_02"}},
            {"id": "se2", "source": "sn2", "target": "sn3", "relation_type": "data_flow", "properties": {"source_node_contract_id": "module_02", "target_node_contract_id": "module_03"}},
        ],
        "required_node_ids": [],
        "required_edge_ids": [],
        "forbidden_labels": [item for item in forbidden],
    }


def _state() -> dict:
    return {
        "method": {"modules": [{"id": "module_01"}]},
        "layout": {
            "canvas": {"width_percent": 1.0, "height_percent": 1.0},
            "nodes": [
                {"id": "module_01", "bbox_percent": {"x": 0.06, "y": 0.2, "w": 0.2, "h": 0.2}},
                {"id": "module_02", "bbox_percent": {"x": 0.36, "y": 0.2, "w": 0.2, "h": 0.2}},
                {"id": "module_03", "bbox_percent": {"x": 0.66, "y": 0.2, "w": 0.2, "h": 0.2}},
            ],
        },
        "icons": {},
    }


def _write_round(root: Path, index: int, *, action: dict, accepted: bool, gain: float) -> None:
    """Write the shared files of one compiled round, the way the loop records it."""

    program, scientific, state = _program(), _scientific_graph(), _state()
    projected = project_candidate(state, program, scientific)
    round_dir = root / f"round_{index:02d}"
    candidate = round_dir / "candidate_00"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "draft_preview.png").write_bytes(b"preview")
    before = create_state_snapshot(state, label=f"before_{index}", round_index=index)
    after = create_state_snapshot({**state, "program": program}, label=f"after_{index}", round_index=index)
    diff = diff_state_snapshots(before, after)
    write_json(round_dir / "state_before.json", before)
    write_json(round_dir / "state_after_action.json", after)
    write_json(round_dir / "action_execution.json", {"action_type": str(action.get("type") or ""), "object_diff": diff, "preserve_checks": {"scientific_contract_unchanged": True}})
    write_json(round_dir / "repair_decision.json", {"accepted": accepted, "reason": "target_improved_without_regression" if accepted else "no_significant_improvement", "score_delta": gain, "new_blockers": 0})
    write_json(round_dir / "reflection.json", {"outcome": "improved" if accepted else "regressed", "why_changed": "bounded repair"})
    write_json(candidate / "state_before.json", before)
    write_json(candidate / "state_after.json", after)
    write_json(candidate / "state_diff.json", diff)
    write_json(candidate / "action_execution.json", {"action_type": str(action.get("type") or ""), "object_diff": diff})
    write_json(candidate / "figure_program.json", projected["program"])
    write_json(candidate / "visual_graph.json", projected["visual_graph"])
    write_json(candidate / "visual_graph_report.json", projected["visual_report"])
    write_json(candidate / "editable_graph.json", projected["editable_graph"])
    write_json(candidate / "editable_graph_report.json", projected["editable_report"])
    write_json(candidate / "graph_bindings.json", projected["bindings"])
    write_json(candidate / "graph_alignment_report.json", projected["alignment_report"])
    write_json(candidate / "grounding_gate.json", projected["grounding_gate"])


def _round_record(index: int, action: dict, *, score: float, gain: float, accepted: bool = True) -> dict:
    """Record one round the way the loop records it, with its own online judgement."""

    return {
        "round": index,
        "selected_variant": 0,
        "accepted": accepted,
        "repair_action": action,
        "online_judgement": {
            "overall_score": score,
            "status": "fault",
            "confidence": 0.6,
            "stage_feedback": [{"scope_type": "slot", "problem": "the panel is cramped", "evidence": ["module_01"], "slot_ids": ["module_01"], "requested_actions": [{"action": "resize_nodes"}]}],
        },
        "repair_decision": {"accepted": accepted, "score_delta": gain, "new_blockers": 0},
    }


def _make_run(root: Path, *, rounds: int = 2, forbidden: tuple[str, ...] = (FORBIDDEN,)) -> Path:
    """Create a run holding one initial round and, optionally, one accepted repair."""

    _write(root / "paper_evolve_editable_report.json", {"paper": "paper.pdf", "title": "T"})
    _write(root / "input_manifest.json", {"paper_original": "paper.pdf"})
    _write(root / "paper_review.json", {"paper_identity": {"paper_id": "p", "title": "T"}})
    _write(root / "scientific_graph.json", _scientific_graph(forbidden=forbidden))
    _write(root / "scientific_ground_truth.json", {"graph_id": "sg", "required_node_ids": ["sn1"]})
    _write(root / "method_figure_content.json", {"modules": [{"id": "module_01"}]})
    _write(root / "figure_specification.json", {
        "figure_goal": "the paper's own goal",
        "modules": [{"id": "module_01", "label": "encoder", "role": "module", "must_show": True}],
    })
    _write(root / "evidence_map.json", {"source_path": "paper.pdf", "evidence": [{"id": "E1", "text": "evidence"}]})
    _write_round(root, 0, action={"type": "initial", "scope_type": "global"}, accepted=True, gain=0.5)
    records = [_round_record(0, {"type": "initial", "scope_type": "global"}, score=0.5, gain=0.5)]
    events = [{"seq": 1, "event": "round_started", "round": 0}]
    if rounds > 1:
        repair = {"type": "resize_nodes", "scope_type": "slot", "slot_ids": ["module_01"]}
        _write_round(root, 1, action=repair, accepted=True, gain=0.25)
        records.append(_round_record(1, repair, score=0.75, gain=0.25))
        events.append({"seq": 2, "event": "candidate_accepted", "round": 1})
    _write(root / "evolution_rounds.json", {"rounds": records})
    (root / "evolution_events.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in events), encoding="utf-8")
    (root / "judge_experiences.jsonl").write_text("", encoding="utf-8")
    return root


class NegativeScriptTests(unittest.TestCase):
    def test_every_script_uses_the_shipped_action_vocabulary(self):
        for script in NEGATIVE_SCRIPTS:
            self.assertIn(script.action_type, SUPPORTED_ACTIONS, script.name)

    def test_project_candidate_reproduces_a_clean_baseline(self):
        projected = project_candidate(_state(), _program(), _scientific_graph())
        self.assertEqual("pass", projected["grounding_gate"]["status"])
        self.assertTrue(projected["alignment_report"]["ok"])

    def test_a_script_that_cannot_be_compiled_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run", forbidden=())
            out = Path(temp) / "out"
            result = build_negative_round(root, out, round_index=0, script=negative_script("insert_forbidden_label"))
            self.assertFalse(result["ok"])
            self.assertIn("no forbidden label", result["reason"])
            self.assertEqual([], list(out.rglob("evolution_rounds.json")) if out.exists() else [])


class ControlledNegativeRoundTests(unittest.TestCase):
    def test_the_gate_block_is_the_real_gate_verdict(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            result = build_negative_round(root, Path(temp) / "out", round_index=0, script=negative_script("insert_forbidden_label"))
            self.assertTrue(result["ok"], result)
            self.assertEqual("pass", result["before_gate"])
            self.assertEqual("blocked", result["after_gate"])
            gate = json.loads((Path(result["run"]) / "round_00" / "candidate_00" / "grounding_gate.json").read_text(encoding="utf-8"))
            self.assertEqual("blocked", gate["status"])
            self.assertFalse(gate["checks"]["forbidden_labels_absent"])
            self.assertIn(f"forbidden scientific label appears in editable output: {FORBIDDEN}", gate["blocking_issues"])

    def test_the_round_is_shaped_like_a_real_one_and_measures_a_rollback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            result = build_negative_round(root, Path(temp) / "out", round_index=0, script=negative_script("drop_a_stage"))
            self.assertTrue(result["ok"], result)
            run = Path(result["run"])
            for name in ("evolution_rounds.json", "evolution_events.jsonl", "paper_evolve_editable_report.json", "scientific_graph.json"):
                self.assertTrue((run / name).is_file(), name)
            for name in ("state_before.json", "state_after_action.json", "action_execution.json", "repair_decision.json", "reflection.json"):
                self.assertTrue((run / "round_00" / name).is_file(), name)
            for name in ("figure_program.json", "visual_graph.json", "editable_graph.json", "graph_bindings.json", "graph_alignment_report.json", "grounding_gate.json", "state_after.json"):
                self.assertTrue((run / "round_00" / "candidate_00" / name).is_file(), name)
            decision = json.loads((run / "round_00" / "repair_decision.json").read_text(encoding="utf-8"))
            self.assertFalse(decision["accepted"])
            self.assertEqual("controlled_negative", decision["reason"])
            self.assertLess(float(decision["score_delta"]), 0.0)
            self.assertEqual("drop_a_stage", decision["injected_defect"]["kind"])

    def test_the_recorded_judgement_belongs_to_the_damaged_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            result = build_negative_round(root, Path(temp) / "out", round_index=0, script=negative_script("insert_forbidden_label"))
            record = json.loads((Path(result["run"]) / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"][0]
            self.assertEqual("blocked", record["online_judgement"]["status"])
            self.assertTrue(record["online_judgement"]["gate_only"])
            # The advisory answer is not asked for a score and does not report one; the acceptance
            # answer is the measurement, and both describe the damaged candidate this round wrote.
            self.assertIsNone(record["online_judgement"]["overall_score"])
            self.assertFalse(record["online_judgement"]["score_measured"])
            self.assertEqual(0.0, record["acceptance_judgement"]["overall_score"])
            self.assertEqual(record["acceptance_judgement"]["overall_score"], record["frozen_judgement"]["overall_score"])

    def test_the_source_run_is_never_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            before = (root / "evolution_rounds.json").read_text(encoding="utf-8")
            build_controlled_negatives(root, Path(temp) / "out")
            self.assertEqual(before, (root / "evolution_rounds.json").read_text(encoding="utf-8"))
            self.assertEqual([0, 1], source_rounds(root))


class ControlledNegativeDatasetTests(unittest.TestCase):
    def _synthesise(self, temp: str, *, round_index: int = 0, forbidden: tuple[str, ...] = (FORBIDDEN,)) -> tuple[Path, Path]:
        runs = Path(temp) / "runs"
        _make_run(runs / "source", forbidden=forbidden)
        manifest = build_controlled_negatives(runs / "source", runs / "synthesised", round_index=round_index, scripts=["insert_forbidden_label"])
        self.assertEqual(1, manifest["rounds"], manifest["failures"])
        return runs, Path(temp) / "dataset"

    def test_the_synthesised_records_clear_both_dataset_warnings(self):
        with tempfile.TemporaryDirectory() as temp:
            runs, out = self._synthesise(temp)
            build_dataset(runs, out)
            report = validate_dataset(out)
            self.assertEqual([], report["errors"])
            self.assertEqual([], report["warnings"])
            self.assertEqual(1, report["decisions"]["rollback"])
            self.assertGreaterEqual(report["repair_actions"], 1)

    def test_the_synthesised_records_declare_their_own_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            runs, _ = self._synthesise(temp)
            synthesised = [path.parent for path in runs.rglob("evolution_events.jsonl") if "synthesised" in str(path)]
            self.assertEqual(1, len(synthesised))
            records = records_from_run(synthesised[0])
            self.assertEqual({CONTROLLED_NEGATIVE_SOURCE}, {item["provenance"]["source"] for item in records})
            rollback = next(item for item in records if item["task_type"] == "trajectory")
            self.assertEqual("rollback", rollback["labels"]["decision"])
            self.assertTrue(rollback["labels"]["action_is_repair"])
            self.assertLess(float(rollback["labels"]["repair_gain"]), 0.0)
            self.assertEqual("blocked", rollback["labels"]["grounding_gate"]["status"])

    def test_a_synthesised_round_carries_the_papers_inventory(self):
        # A Judge sample of a synthesised round reads the same material as one of a real
        # round: the paper's own method list, never the figure's selection keys.
        with tempfile.TemporaryDirectory() as temp:
            runs, _ = self._synthesise(temp)
            synthesised = [path.parent for path in runs.rglob("evolution_events.jsonl") if "synthesised" in str(path)]
            rollback = next(item for item in records_from_run(synthesised[0]) if item["task_type"] == "trajectory")

            self.assertEqual("the paper's own goal", rollback["paper"]["method_inventory"]["figure_goal"])
            self.assertEqual(["encoder"], [item["label"] for item in rollback["paper"]["method_inventory"]["modules"]])
            for key in SELECTION_KEYS:
                self.assertNotIn(f'"{key}"', json.dumps(rollback, ensure_ascii=False))

    def test_a_hidden_test_paper_is_never_used(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            ledger = Path(temp) / "split_ledger.jsonl"
            ledger.write_text(json.dumps({"paper_id": "p", "split": "hidden_test"}) + "\n", encoding="utf-8")
            manifest = build_controlled_negatives(root, Path(temp) / "out", split=ledger)
            self.assertEqual(0, manifest["rounds"])
            self.assertEqual(1, len(manifest["skipped"]))

    def test_a_train_paper_is_synthesised_and_keeps_its_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            ledger = Path(temp) / "split_ledger.jsonl"
            ledger.write_text(json.dumps({"paper_id": "p", "split": "train"}) + "\n", encoding="utf-8")
            manifest = build_controlled_negatives(root, Path(temp) / "out", round_index=0, split=ledger, scripts=["insert_forbidden_label"])
            self.assertEqual(1, manifest["rounds"], manifest["failures"])
            self.assertEqual("train", manifest["results"][0]["split"])

    def test_the_synthesised_round_pairs_with_the_real_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            runs, dataset = self._synthesise(temp, round_index=1)
            build_dataset(runs, dataset)
            pairs = build_preference_pairs(dataset / "judge_examples.jsonl", Path(temp) / "pairs")
            self.assertTrue(pairs["ok"], pairs)
            self.assertEqual(1, pairs["pairs"])
            self.assertEqual(["harmful_repair"], list(pairs["by_family"]))
            self.assertEqual([], pairs["skipped_sample"])


if __name__ == "__main__":
    unittest.main()
