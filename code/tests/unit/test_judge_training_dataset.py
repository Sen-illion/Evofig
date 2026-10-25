"""Tests for direct evolution process-supervision datasets."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import create_state_snapshot, diff_state_snapshots
from rfs.judge_training import build_dataset, export_training_data, records_from_run, validate_dataset
from rfs.judge_training.method_inventory import SELECTION_KEYS
from rfs.judge_training.prompts import ACCEPTANCE_OUTPUT_SCHEMA, ADVISORY_OUTPUT_SCHEMA
from rfs.judge_training.schemas import normalize_example, validate_example


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_run(
    root: Path,
    *,
    repair_action: dict | None = None,
    accepted: bool = True,
    gain: float | None = None,
    comparison: str = "",
    pairwise_verdict: str = "",
    figure_changed: bool | None = None,
) -> Path:
    """Create the smallest run directory that satisfies the process schema."""
    action = repair_action if isinstance(repair_action, dict) else {"type": "reroute", "edge_ids": ["e1"]}
    if gain is None:
        gain = 0.2 if accepted else -0.1
    candidate = root / "round_00" / "candidate_00"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "draft_preview.png").write_bytes(b"preview")
    _write(candidate / "figure_program.json", {})
    before = create_state_snapshot({"method": {"modules": [{"id": "n1"}]}, "layout": {"nodes": [{"id": "n1"}]}, "icons": {}}, label="before", round_index=0)
    after = create_state_snapshot({"method": {"modules": [{"id": "n1"}]}, "layout": {"nodes": [{"id": "n1"}]}, "icons": {}, "program": {}}, label="after", round_index=0)
    _write(root / "round_00" / "state_before.json", before)
    _write(candidate / "state_after.json", after)
    state_diff = diff_state_snapshots(before, after)
    _write(candidate / "state_diff.json", state_diff)
    _write(candidate / "action_execution.json", {"action_type": str(action.get("type") or ""), "object_diff": state_diff, "preserve_checks": {"scientific_contract_unchanged": True}})
    _write(candidate / "grounding_gate.json", {"status": "pass", "ok": True})
    scientific = {"graph_id": "sg", "paper_id": "p", "kind": "scientific", "nodes": [{"id": "sn1"}, {"id": "sn2"}], "edges": [{"id": "se1", "source": "sn1", "target": "sn2", "relation_type": "data_flow"}]}
    visual = {"graph_id": "vg", "paper_id": "p", "kind": "visual", "nodes": [{"id": "vn1"}, {"id": "vn2"}], "edges": [{"id": "ve1", "source": "vn1", "target": "vn2", "relation_type": "data_flow"}]}
    editable = {"graph_id": "eg", "paper_id": "p", "kind": "editable", "nodes": [{"id": "en1"}, {"id": "en2"}], "edges": [{"id": "ee1", "source": "en1", "target": "en2", "relation_type": "data_flow"}]}
    _write(root / "scientific_graph.json", scientific)
    _write(root / "scientific_ground_truth.json", {"graph_id": "sg", "required_node_ids": ["sn1"]})
    _write(root / "method_figure_content.json", {"modules": [{"id": "n1"}]})
    _write(root / "evidence_map.json", {"source_path": "paper.pdf", "evidence": [{"id": "E1", "text": "evidence"}]})
    _write(candidate / "visual_graph.json", visual)
    _write(candidate / "editable_graph.json", editable)
    _write(root / "round_00" / "repair_decision.json", {"accepted": accepted, "score_delta": gain, "comparison": comparison, "pairwise_verdict": pairwise_verdict})
    if figure_changed is not None:
        _write(candidate / "figure_object_diff.json", {"available": True, "changed": figure_changed, "changed_object_ids": ["n1"] if figure_changed else []})
    _write(root / "round_00" / "reflection.json", {"why_changed": "bounded repair helped"})
    (root / "evolution_events.jsonl").write_text(json.dumps({"seq": 1, "event": "candidate_accepted" if accepted else "candidate_rolled_back", "round": 0}) + "\n", encoding="utf-8")
    (root / "judge_experiences.jsonl").write_text(json.dumps({"round": 0, "action": action}) + "\n", encoding="utf-8")
    _write(root / "evolution_rounds.json", {"rounds": [{"round": 0, "selected_variant": 0, "candidate_index": 0, "accepted": accepted, "online_judgement": {"judge_role": "online_bootstrap", "overall_score": 0.7, "dimension_scores": {"geometry_cleanliness": 0.5}, "status": "fault", "confidence": 0.6, "stage_feedback": [{"scope_type": "connector", "problem": "edge e1 crosses a panel border", "evidence": ["e1"], "edge_ids": ["e1"], "requested_actions": [{"action": "reroute"}], "stage": "interface", "severity": "major"}]}, "acceptance_judgement": {"judge_role": "frozen_bootstrap", "overall_score": 0.8, "dimension_scores": {"geometry_cleanliness": 0.6}, "status": "pass", "confidence": 0.7, "blocking_issues": []}, "frozen_judgement": {"judge_role": "frozen_bootstrap", "overall_score": 0.8}, "repair_action": action, "repair_decision": {"accepted": accepted, "score_delta": gain}}]})
    return root


class JudgeTrainingDatasetTests(unittest.TestCase):
    def test_schema_rejects_unknown_task(self):
        with self.assertRaises(ValueError):
            normalize_example({"task_type": "unknown"})

    def test_trajectory_requires_full_transition(self):
        example = normalize_example({"task_type": "trajectory", "example_id": "x", "ground_truth_id": "g", "preview": "p.png", "labels": {}})
        self.assertIn("trajectory labels require state_before", validate_example(example))

    def test_builds_three_process_sample_types(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            _make_run(root)
            out = Path(temp) / "dataset"
            manifest = build_dataset(root, out)
            # One round asks two questions, so it yields two repair examples: the advisory answer
            # and the acceptance answer.
            self.assertEqual(manifest["records"], 4)
            self.assertEqual(manifest["task_counts"], {"outcome": 1, "repair": 2, "trajectory": 1})
            self.assertTrue(validate_dataset(out)["ok"])
            self.assertTrue(all((out / f"{task}.jsonl").is_file() for task in ("outcome", "repair", "trajectory")))
            self.assertEqual(manifest["role_counts"], {"advisory": 1, "acceptance": 1})
            self.assertTrue(manifest["forked_trajectories"])
            advisory = [json.loads(line) for line in (out / "advisory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            acceptance = [json.loads(line) for line in (out / "acceptance.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(["online_bootstrap"], [item["judge_metadata"]["judge_role"] for item in advisory])
            self.assertEqual(["frozen_bootstrap"], [item["judge_metadata"]["judge_role"] for item in acceptance])
            trajectory = next(item for item in records_from_run(root) if item["task_type"] == "trajectory")
            self.assertEqual(trajectory["labels"]["decision"], "accept")
            self.assertEqual(trajectory["labels"]["action"]["edge_ids"], ["e1"])
            self.assertEqual("machine_verified", trajectory["labels"]["audit_status"])
            self.assertEqual({"scientific", "visual", "editable"}, set(trajectory["labels"]["subgraphs"]))
            self.assertTrue(trajectory["labels"]["object_diff"]["changed"])
            self.assertTrue(trajectory["labels"]["action_is_repair"])
            export = export_training_data(out, Path(temp) / "training")
            self.assertEqual(4, export["records"])
            self.assertEqual({"advisory": 1, "acceptance": 1}, export["role_counts"])
            self.assertTrue((Path(temp) / "training" / "process.jsonl").is_file())
            self.assertTrue((Path(temp) / "training" / "advisory.jsonl").is_file())
            self.assertTrue((Path(temp) / "training" / "acceptance.jsonl").is_file())

    def test_the_paper_block_is_the_inventory_and_not_the_figures_selection(self):
        # The run's own plan says which concepts it wanted drawn. What a Judge reads is
        # the inventory stored beside it, with the selection keys stripped.
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            _write(root / "figure_specification.json", {
                "figure_goal": "the paper's own goal",
                "modules": [{
                    "id": "n1", "label": "encoder", "role": "module", "detail_level": "module",
                    "evidence_ids": ["E1"], "visualization_value": 1.0, "must_show": True,
                    "should_render": True, "render_priority": 1,
                }],
            })
            record = next(item for item in records_from_run(root) if item["task_type"] == "trajectory")

            self.assertEqual("the paper's own goal", record["paper"]["method_inventory"]["figure_goal"])
            self.assertEqual(["encoder"], [item["label"] for item in record["paper"]["method_inventory"]["modules"]])
            self.assertEqual(record["paper"], record["labels"]["paper_evidence"])
            for key in SELECTION_KEYS:
                self.assertNotIn(f'"{key}"', json.dumps(record, ensure_ascii=False))

    def test_the_sample_version_names_the_inventory_reporting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")

            manifest = build_dataset(root, Path(temp) / "dataset")

            self.assertEqual("judge-dataset-v004", manifest["dataset_version"])
            record = next(item for item in records_from_run(root) if item["task_type"] == "outcome")
            self.assertEqual("judge-dataset-v004", record["dataset_version"])

    def test_the_repair_label_is_exactly_the_judgement_the_prompt_asks_for(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            repair = next(item for item in records_from_run(root) if item["task_type"] == "repair")
            self.assertEqual(set(ADVISORY_OUTPUT_SCHEMA), set(repair["labels"]))
            self.assertEqual("fault", repair["labels"]["status"])
            self.assertEqual({"connector"}, {item["scope_type"] for item in repair["labels"]["stage_feedback"]})
            self.assertEqual([{"action": "reroute", "edge_ids": ["e1"]}], repair["labels"]["stage_feedback"][0]["requested_actions"])

    def test_each_role_gets_its_own_example_and_its_own_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            repairs = [item for item in records_from_run(root) if item["task_type"] == "repair"]

            self.assertEqual(2, len(repairs))
            advisory, acceptance = repairs
            self.assertEqual("online_bootstrap", advisory["judge_metadata"]["judge_role"])
            self.assertEqual("frozen_bootstrap", acceptance["judge_metadata"]["judge_role"])
            self.assertEqual(set(ADVISORY_OUTPUT_SCHEMA), set(advisory["labels"]))
            self.assertNotIn("overall_score", advisory["labels"])
            self.assertEqual(set(ACCEPTANCE_OUTPUT_SCHEMA), set(acceptance["labels"]))
            self.assertNotIn("stage_feedback", acceptance["labels"])
            self.assertEqual(0.8, acceptance["labels"]["overall_score"])
            self.assertEqual([], validate_example(acceptance), validate_example(acceptance))
            self.assertTrue(acceptance["example_id"].endswith("acceptance-0000"))
            # The observation block belongs to the advisory example: the acceptance question is
            # about what the figure is worth, not about what the round did next.
            self.assertEqual({}, acceptance["observed"])

    def test_the_observed_block_carries_the_consequences_the_judge_never_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            repair = next(item for item in records_from_run(root) if item["task_type"] == "repair")
            observed = repair["observed"]
            self.assertEqual("accept", observed["decision"])
            self.assertEqual(0.2, observed["repair_gain"])
            self.assertTrue(observed["action_is_repair"])
            self.assertIn("program", observed["state_after"] or {})
            self.assertTrue(validate_example(repair) == [], validate_example(repair))

    def test_a_consequence_can_never_be_smuggled_into_a_repair_label(self):
        example = normalize_example({"task_type": "repair", "example_id": "x", "ground_truth_id": "g", "preview": "p.png", "labels": {"overall_score": 0.5, "decision": "accept"}})
        self.assertIn("repair labels are the model target only; observed fields leaked: decision", validate_example(example))

    def test_provenance_can_name_a_synthesised_round_family(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            records = records_from_run(root, source="controlled_negative_round")
            self.assertEqual({"controlled_negative_round"}, {item["provenance"]["source"] for item in records})
            self.assertEqual({"direct_evolution_process"}, {item["provenance"]["source"] for item in records_from_run(root)})

    def test_a_run_directory_can_declare_its_own_provenance_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            _write(root / "paper_evolve_editable_report.json", {"paper": "paper.pdf", "provenance_source": "controlled_negative_round"})
            self.assertEqual({"controlled_negative_round"}, {item["provenance"]["source"] for item in records_from_run(root)})

    def test_a_first_round_initial_action_is_not_a_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run", repair_action={"type": "initial"})
            trajectory = next(item for item in records_from_run(root) if item["task_type"] == "trajectory")
            self.assertFalse(trajectory["labels"]["action_is_repair"])
            self.assertEqual({"type": "initial"}, trajectory["labels"]["action"])

    def test_an_uncontrolled_action_is_not_a_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run", repair_action={"type": "rewrite_everything"})
            trajectory = next(item for item in records_from_run(root) if item["task_type"] == "trajectory")
            self.assertFalse(trajectory["labels"]["action_is_repair"])

    def test_dataset_validation_warns_when_every_round_only_accepts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            out = Path(temp) / "dataset"
            build_dataset(root, out)
            report = validate_dataset(out)
            self.assertTrue(report["ok"])
            self.assertEqual({"accept": 1, "rollback": 0}, report["decisions"])
            self.assertEqual(1, report["repair_actions"])
            self.assertEqual(["no rollback negative: every recorded decision is accept"], report["warnings"])

    def test_dataset_validation_warns_when_rounds_carry_no_repair_action(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run", repair_action={"type": "initial"}, accepted=False)
            out = Path(temp) / "dataset"
            build_dataset(root, out)
            report = validate_dataset(out)
            self.assertTrue(report["ok"])
            self.assertEqual({"accept": 0, "rollback": 1}, report["decisions"])
            self.assertEqual(0, report["repair_actions"])
            self.assertEqual(["no controlled repair action: rounds only report the initial action"], report["warnings"])


    def test_the_observation_carries_the_comparison_the_round_was_decided_on(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run", comparison="pairwise", pairwise_verdict="after", figure_changed=True)
            repair = next(item for item in records_from_run(root) if item["task_type"] == "repair")

            self.assertEqual("pairwise", repair["observed"]["comparison"])
            self.assertEqual("after", repair["observed"]["pairwise_verdict"])
            self.assertTrue(repair["observed"]["figure_changed"])

    def test_an_accepted_round_with_a_lower_score_is_still_an_improvement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(
                Path(temp) / "run",
                accepted=True,
                gain=-0.1,
                comparison="pairwise",
                pairwise_verdict="after",
                figure_changed=True,
            )
            repair = next(item for item in records_from_run(root) if item["task_type"] == "repair")

            # The score went the wrong way; the comparison the loop decided on did not.
            self.assertEqual(-0.1, repair["observed"]["repair_gain"])
            self.assertEqual("improvement", repair["observed"]["feedback_effective"])

    def test_a_rolled_back_round_with_a_higher_score_is_still_a_rollback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(
                Path(temp) / "run",
                accepted=False,
                comparison="pairwise",
                pairwise_verdict="before",
                figure_changed=True,
            )
            repair = next(item for item in records_from_run(root) if item["task_type"] == "repair")

            self.assertEqual("rollback", repair["observed"]["feedback_effective"])

    def test_the_effective_feedback_stays_out_of_the_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run", comparison="pairwise", pairwise_verdict="after", figure_changed=True)
            repair = next(item for item in records_from_run(root) if item["task_type"] == "repair")

            self.assertNotIn("feedback_effective", repair["labels"])
            self.assertNotIn("feedback_effective", json.dumps(repair["labels"], ensure_ascii=False))

    def test_a_round_with_no_figure_contrast_is_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run", figure_changed=False)
            self.assertEqual([], records_from_run(root))
            manifest = build_dataset(root, Path(temp) / "dataset")
            self.assertEqual(0, manifest["records"])
            self.assertFalse(manifest["forked_trajectories"])


if __name__ == "__main__":
    unittest.main()
