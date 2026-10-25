"""Split inheritance for direct evolution process-supervision samples."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import create_state_snapshot, diff_state_snapshots
from rfs.judge_training import build_dataset, records_from_run
from rfs.judge_training.split_ledger import (
    EXPLORATION_SPLIT,
    build_split_index,
    resolve_split,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_run(root: Path, *, paper_id: str, title: str) -> Path:
    """Create the smallest run directory that satisfies the process schema."""
    run = root
    candidate = run / "round_00" / "candidate_00"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "draft_preview.png").write_bytes(b"preview")
    _write(candidate / "figure_program.json", {})
    before = create_state_snapshot({"method": {"modules": [{"id": "n1"}]}, "layout": {"nodes": [{"id": "n1"}]}, "icons": {}}, label="before", round_index=0)
    after = create_state_snapshot({"method": {"modules": [{"id": "n1"}]}, "layout": {"nodes": [{"id": "n1"}]}, "icons": {}, "program": {}}, label="after", round_index=0)
    _write(run / "round_00" / "state_before.json", before)
    _write(candidate / "state_after.json", after)
    state_diff = diff_state_snapshots(before, after)
    _write(candidate / "state_diff.json", state_diff)
    _write(candidate / "action_execution.json", {"action_type": "reroute", "object_diff": state_diff, "preserve_checks": {"scientific_contract_unchanged": True}})
    _write(candidate / "grounding_gate.json", {"status": "pass", "ok": True})
    _write(run / "scientific_graph.json", {"graph_id": "sg", "paper_id": paper_id, "kind": "scientific", "nodes": [{"id": "sn1"}, {"id": "sn2"}], "edges": [{"id": "se1", "source": "sn1", "target": "sn2", "relation_type": "data_flow"}]})
    _write(run / "scientific_ground_truth.json", {"graph_id": "sg", "required_node_ids": ["sn1"]})
    _write(run / "method_figure_content.json", {"modules": [{"id": "n1"}]})
    _write(run / "evidence_map.json", {"source_path": "paper.pdf", "evidence": [{"id": "E1", "text": "evidence"}]})
    _write(run / "paper_review.json", {"paper_identity": {"title": title, "paper_id": paper_id}})
    _write(candidate / "visual_graph.json", {"graph_id": "vg", "paper_id": paper_id, "kind": "visual", "nodes": [{"id": "vn1"}, {"id": "vn2"}], "edges": [{"id": "ve1", "source": "vn1", "target": "vn2", "relation_type": "data_flow"}]})
    _write(candidate / "editable_graph.json", {"graph_id": "eg", "paper_id": paper_id, "kind": "editable", "nodes": [{"id": "en1"}, {"id": "en2"}], "edges": [{"id": "ee1", "source": "en1", "target": "en2", "relation_type": "data_flow"}]})
    _write(run / "round_00" / "repair_decision.json", {"accepted": True, "score_delta": 0.2})
    _write(run / "round_00" / "reflection.json", {"why_changed": "bounded repair helped"})
    (run / "evolution_events.jsonl").write_text(json.dumps({"seq": 1, "event": "candidate_accepted", "round": 0}) + "\n", encoding="utf-8")
    (run / "judge_experiences.jsonl").write_text(json.dumps({"round": 0, "action": {"type": "reroute"}}) + "\n", encoding="utf-8")
    _write(run / "evolution_rounds.json", {"rounds": [{"round": 0, "selected_variant": 0, "candidate_index": 0, "accepted": True, "online_judgement": {"judge_role": "online_bootstrap", "overall_score": 0.7, "stage_feedback": [], "requested_actions": []}, "acceptance_judgement": {"judge_role": "frozen_bootstrap", "overall_score": 0.8, "status": "pass", "blocking_issues": []}, "repair_action": {"type": "reroute", "edge_ids": ["e1"]}, "repair_decision": {"accepted": True, "score_delta": 0.2}}]})
    return run


class SplitAssignmentTests(unittest.TestCase):
    def test_split_index_resolves_every_identity_namespace(self):
        index = build_split_index([
            {"paper_id": "10.1000/example", "title": "An Example Paper", "split": "train"},
            {"identity_keys": {"doi": ["10.2000/held-out"], "title": ["held out paper"]}, "split": "hidden_test"},
        ])
        self.assertEqual(index["paper_id:10.1000/example"], "train")
        self.assertEqual(index["title:an example paper"], "train")
        self.assertEqual(index["doi:10.2000/held-out"], "hidden_test")

    def test_resolve_split_prefers_identifiers_over_titles(self):
        index = build_split_index([{"paper_id": "p-1", "title": "shared title", "split": "validation"}])
        split, resolution = resolve_split(index, paper_id="p-1", title="shared title")
        self.assertEqual("validation", split)
        self.assertEqual("split_ledger", resolution["source"])
        self.assertEqual("paper_id:p-1", resolution["matched_key"])
        split, resolution = resolve_split(index, paper_id="unknown", title="shared title")
        self.assertEqual("validation", split)
        self.assertTrue(resolution["matched_key"].startswith("title:"))

    def test_resolve_split_reports_unknown_papers(self):
        split, resolution = resolve_split(build_split_index([{"paper_id": "p-1", "split": "train"}]), paper_id="p-2", title="other")
        self.assertIsNone(split)
        self.assertEqual("unresolved", resolution["source"])
        self.assertTrue(resolution["candidates"])

    def test_records_inherit_the_source_paper_split(self):
        with tempfile.TemporaryDirectory() as temp:
            run = _make_run(Path(temp) / "run", paper_id="p-1", title="An Example Paper")
            index = build_split_index([{"paper_id": "p-1", "title": "An Example Paper", "split": "train"}])
            records = records_from_run(run, split_index=index)
            self.assertEqual(4, len(records))
            self.assertEqual({"train"}, {item["split"] for item in records})
            self.assertEqual({"split_ledger"}, {item["provenance"]["split_resolution"]["source"] for item in records})

    def test_unknown_paper_becomes_an_exploration_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            run = _make_run(Path(temp) / "run", paper_id="p-9", title="Collected Later")
            records = records_from_run(run, split_index=build_split_index([{"paper_id": "p-1", "split": "train"}]))
            self.assertEqual({EXPLORATION_SPLIT}, {item["split"] for item in records})
            self.assertEqual({"unresolved"}, {item["provenance"]["split_resolution"]["source"] for item in records})

    def test_records_keep_an_empty_split_without_a_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            run = _make_run(Path(temp) / "run", paper_id="p-1", title="An Example Paper")
            self.assertEqual({""}, {item["split"] for item in records_from_run(run)})

    def test_build_dataset_keeps_only_trainable_splits_in_the_training_view(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = _make_run(root / "run", paper_id="p-1", title="An Example Paper")
            index = build_split_index([{"paper_id": "p-1", "title": "An Example Paper", "split": "validation"}])
            manifest = build_dataset(run, root / "dataset", split_index=index)
            self.assertEqual(4, manifest["records"])
            self.assertEqual({"validation": 4}, manifest["split_counts"])
            self.assertEqual("split_ledger", manifest["split_filtering"])
            self.assertEqual({"hidden_test": 0, EXPLORATION_SPLIT: 0}, manifest["excluded_counts"])
            self.assertTrue((root / "dataset" / "outcome.jsonl").is_file())

    def test_build_dataset_writes_hidden_test_and_exploration_apart(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = _make_run(root / "run", paper_id="p-9", title="Collected Later")
            index = build_split_index([{"paper_id": "p-1", "split": "train"}])
            manifest = build_dataset(run, root / "dataset", split_index=index)
            self.assertEqual(0, manifest["records"])
            self.assertEqual({"hidden_test": 0, EXPLORATION_SPLIT: 4}, manifest["excluded_counts"])
            self.assertEqual(4, len((root / "dataset" / f"{EXPLORATION_SPLIT}.jsonl").read_text(encoding="utf-8").splitlines()))
            self.assertEqual("", (root / "dataset" / "judge_examples.jsonl").read_text(encoding="utf-8"))

    def test_build_dataset_separates_hidden_test_papers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = _make_run(root / "run", paper_id="p-1", title="An Example Paper")
            index = build_split_index([{"paper_id": "p-1", "title": "An Example Paper", "split": "hidden_test"}])
            manifest = build_dataset(run, root / "dataset", split_index=index)
            self.assertEqual(0, manifest["records"])
            self.assertEqual({"hidden_test": 4, EXPLORATION_SPLIT: 0}, manifest["excluded_counts"])
            hidden = (root / "dataset" / "hidden_test.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(4, len(hidden))
            self.assertEqual({"hidden_test"}, {json.loads(line)["split"] for line in hidden})
            self.assertEqual("", (root / "dataset" / "trajectory.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
