"""Tests for building, reviewing and exporting Judge preference pairs."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from rfs.judge_training.judge_schema import defect_label
from rfs.judge_training.preference_pairs import (
    MISSED_DEFECT_MIN_LENGTH_RATIO,
    apply_human_review,
    build_preference_pairs,
    build_review_queue,
    export_preference_pairs,
    known_defects,
    rejected_judgement,
    rejection_reason,
)


SCORES = {"scientific_fidelity": 0.0, "logical_flow": 0.1, "layout_quality": 0.8, "readability": 0.9, "domain_fit": 0.3}


def _figure(root, name, colour):
    path = root / name
    image = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 20, 60, 60), fill=colour)
    image.save(path)
    return str(path)


def _example(root, *, example_id, task_type, kind, diagnosis_type, pick=None, candidate_b=True, provenance=True, preview=False):
    source = _figure(root, f"{example_id}-a.png", "blue")
    corrupted = _figure(root, f"{example_id}-b.png", "red") if candidate_b else None
    query = _figure(root, f"{example_id}-query.png", "green") if preview else None
    labels = {"quality_label": "坏", "quality_scores": dict(SCORES), "diagnosis": [{"type": diagnosis_type, "target": "encoder->decoder", "severity": "high", "evidence": "the arrow points the wrong way"}], "confidence": 1.0}
    if pick:
        labels["preferred_candidate"] = pick
        labels["diagnosis"][0]["figure"] = "B"
    row = {
        "example_id": example_id,
        "dataset_version": "judge-dataset-v003",
        "ground_truth_id": "arxiv-2401.00001",
        "paper_id": "2401.00001",
        "domain": "cv",
        "split": "train",
        "task_type": task_type,
        "candidate_a": source,
        "candidate_b": corrupted,
        "labels": labels,
        "pair_group": f"arxiv-2401.00001|{task_type}",
        "paper": {"title": "Chain model", "abstract": "Abstract", "evidence_spans": [{"evidence_id": "e1", "quote": "encoder feeds decoder"}]},
        "ground_truth": {"method_summary": "encoder to decoder", "required_entities": ["encoder", "decoder"], "required_relations": ["encoder->decoder"]},
        "provenance": {"source": "deterministic_local_corruption", "corruption_type": kind, "target": "encoder->decoder"},
    }
    if not provenance:
        row.pop("provenance")
    if query:
        row["preview"] = query
    return row


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding="utf-8")


def _read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sheet_sizes(paths):
    """Read the contact sheets' sizes without leaving the files open."""

    sizes = []
    for path in paths:
        with Image.open(path) as opened:
            sizes.append(opened.size)
    return sizes


def _process_row(root, *, example_id, round_index, state_before, decision, gain, action, source="direct_evolution_process", injected=None):
    """One process ``repair`` row: the judgement target plus the measured consequence."""
    return {
        "example_id": example_id,
        "dataset_version": "judge-dataset-v003",
        "ground_truth_id": "direct-run-8f2c11aa",
        "task_type": "repair",
        "split": "train",
        "preview": _figure(root, f"{example_id}.png", "blue"),
        "labels": {
            "overall_score": 0.7,
            "dimension_scores": {"geometry_cleanliness": 0.5},
            "status": "fault",
            "confidence": 0.6,
            "preserve": [],
            "blocking_issues": [],
            "stage_feedback": [{
                "scope_type": "connector",
                "problem": "edge e1 crosses a panel border",
                "evidence": ["e1"],
                "semantic_node_ids": [],
                "slot_ids": [],
                "object_ids": [],
                "edge_ids": ["e1"],
                "expected_effect": "route e1 inside its own panel",
                "forbidden_changes": [],
                "requested_actions": [{"action": "reroute"}],
            }],
        },
        "observed": {
            "action": action,
            "action_is_repair": True,
            "action_execution": {"applied": True},
            "object_diff": {"changed": True},
            "repair_gain": gain,
            "decision": decision,
            "reflection": {},
            "audit_status": "machine_verified",
            "state_before": state_before,
            "injected_defect": injected or {},
            "judge": {"gate_only": False},
        },
        "process_context": {"state_before": state_before},
        "provenance": {"source": source, "round": round_index},
    }


class BuildPreferencePairsTests(unittest.TestCase):
    def test_builds_the_three_families_from_one_pairwise_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "examples.jsonl"
            _write(source, [_example(root, example_id="p1", task_type="pairwise_preference", kind="relation_reversal", diagnosis_type="wrong_arrow_target", pick="A")])
            report = build_preference_pairs(source, root / "pairs.jsonl")
            pairs = _read(root / "pairs.jsonl")

        self.assertTrue(report["ok"])
        self.assertEqual(sorted(report["by_family"]), ["misdiagnosed", "missed_defect", "wrong_pick"])
        self.assertGreaterEqual(report["length_by_family"]["missed_defect"]["median_rejected_minus_chosen_chars"], 0)
        self.assertEqual(report["gating_pairs"], 3)
        for pair in pairs:
            basis = pair["preference_basis"]
            self.assertEqual(pair["pair_group"], "arxiv-2401.00001|pairwise_preference")
            self.assertEqual(basis["chosen_source"], "controlled_error_label")
            self.assertTrue(basis["deterministic_check"]["passed"])
            self.assertFalse(basis["human_audited"])
            self.assertEqual(basis["known_defects"][0]["diagnosis_type"], "wrong_arrow_target")
            self.assertEqual(pair["chosen"]["source"], "controlled_error_label")

    def test_the_rejected_answer_never_names_the_injected_defect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "examples.jsonl"
            _write(source, [_example(root, example_id="p1", task_type="pairwise_preference", kind="relation_reversal", diagnosis_type="wrong_arrow_target", pick="A")])
            build_preference_pairs(source, root / "pairs.jsonl")
            pairs = _read(root / "pairs.jsonl")

        for pair in pairs:
            injected = {item["diagnosis_type"] for item in pair["preference_basis"]["known_defects"]}
            named = {item["type"] for item in pair["rejected"]["judgement"]["diagnosis"]}
            if pair["rejected"]["family"] == "wrong_pick":
                self.assertNotEqual(pair["rejected"]["judgement"]["preferred_candidate"], pair["chosen"]["judgement"]["preferred_candidate"])
            else:
                self.assertEqual(named & injected, set())

    def test_swapped_pairs_mirror_the_orientation_and_the_pick(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "examples.jsonl"
            _write(source, [_example(root, example_id="p1", task_type="pairwise_preference", kind="overlap", diagnosis_type="overlap", pick="A")])
            report = build_preference_pairs(source, root / "pairs.jsonl", include_swapped=True)
            pairs = {pair["example_id"]: pair for pair in _read(root / "pairs.jsonl")}

        self.assertEqual(report["swapped_pairs"], 3)
        straight = pairs["p1__wrong_pick"]
        swapped = pairs["p1__wrong_pick__swap"]
        self.assertEqual(straight["chosen"]["judgement"]["preferred_candidate"], "A")
        self.assertEqual(swapped["chosen"]["judgement"]["preferred_candidate"], "B")
        self.assertEqual(straight["candidate_a"], swapped["candidate_b"])
        self.assertEqual(straight["pair_group"], swapped["pair_group"])

    def test_an_example_without_a_traceable_defect_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "examples.jsonl"
            _write(source, [_example(root, example_id="p1", task_type="pairwise_preference", kind="", diagnosis_type="overlap", pick="A", provenance=False)])
            report = build_preference_pairs(source, root / "pairs.jsonl")
            pairs = _read(root / "pairs.jsonl")

        self.assertEqual(pairs, [])
        self.assertFalse(report["ok"])
        self.assertIn("no traceable injected defect", report["skipped_sample"][0]["reason"])

    def test_a_process_row_without_a_measured_consequence_produces_no_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "examples.jsonl"
            _write(source, [_example(root, example_id="p1", task_type="outcome", kind="overlap", diagnosis_type="overlap", candidate_b=False)])
            report = build_preference_pairs(source, root / "pairs.jsonl")

        self.assertEqual(report["pairs"], 0)
        self.assertEqual(report["skipped"], 0)


class RejectedScoreTests(unittest.TestCase):
    """A rejected answer's scores have to match the diagnosis it names.

    ``DEFECT_LABELS`` fixes one score set per defect, so a rejection that moved only the
    diagnosis kept the certified defect's scores. Nothing in the corpus then contrasted the
    scores, and a trained Judge would receive no signal for them at all.
    """

    def _families(self, root, *, injected="wrong_arrow_target"):
        source = root / "examples.jsonl"
        _write(source, [_example(root, example_id="p1", task_type="pairwise_preference", kind="relation_reversal", diagnosis_type=injected, pick="A")])
        build_preference_pairs(source, root / "pairs.jsonl")
        return {pair["preference_basis"]["rejected_family"]: pair for pair in _read(root / "pairs.jsonl")}

    def test_a_misdiagnosis_carries_the_scores_of_the_class_it_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            families = self._families(Path(tmp))
        chosen = families["misdiagnosed"]["chosen"]["judgement"]
        rejected = families["misdiagnosed"]["rejected"]["judgement"]
        named = rejected["diagnosis"][0]["type"]
        self.assertNotEqual(named, "wrong_arrow_target")
        self.assertEqual(rejected["quality_scores"], defect_label(named)["quality_scores"])
        self.assertNotEqual(rejected["quality_scores"], chosen["quality_scores"])

    def test_a_missed_defect_carries_the_scores_of_finding_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            families = self._families(Path(tmp))
        chosen = families["missed_defect"]["chosen"]["judgement"]
        rejected = families["missed_defect"]["rejected"]["judgement"]
        self.assertEqual(len(rejected["diagnosis"]), 1)
        self.assertEqual(rejected["diagnosis"][0]["type"], "assumed_worse")
        self.assertEqual(rejected["diagnosis"][0]["target"], chosen["diagnosis"][0]["target"])
        self.assertIn("no injected defect was found", rejected["diagnosis"][0]["evidence"])
        self.assertIn(chosen["diagnosis"][0]["evidence"], rejected["diagnosis"][0]["evidence"])
        self.assertEqual(rejected["quality_scores"], defect_label("assumed_worse")["quality_scores"])
        self.assertNotEqual(rejected["quality_scores"], chosen["quality_scores"])
        chosen_chars = len(json.dumps(chosen, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        rejected_chars = len(json.dumps(rejected, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        self.assertGreaterEqual(rejected_chars, MISSED_DEFECT_MIN_LENGTH_RATIO * chosen_chars)

    def test_a_wrong_pick_leaves_the_scores_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            families = self._families(Path(tmp))
        chosen = families["wrong_pick"]["chosen"]["judgement"]
        rejected = families["wrong_pick"]["rejected"]["judgement"]
        self.assertEqual(rejected["quality_scores"], chosen["quality_scores"])
        self.assertNotEqual(rejected["preferred_candidate"], chosen["preferred_candidate"])

    def test_a_misdiagnosis_never_names_a_label_that_certifies_no_defect(self):
        for injected in ("wrong_arrow_target", "beautiful_but_wrong", "assumed_worse"):
            with self.subTest(injected=injected), tempfile.TemporaryDirectory() as tmp:
                families = self._families(Path(tmp), injected=injected)
            named = families["misdiagnosed"]["rejected"]["judgement"]["diagnosis"][0]["type"]
            self.assertNotIn(named, ("assumed_worse", "correct_but_ugly"))


class RejectionRuleTests(unittest.TestCase):
    def setUp(self):
        self.defects = [{"corruption_type": "overlap", "diagnosis_type": "overlap", "severity": "medium", "target": "encoder", "source": "deterministic_local_corruption", "gating": True}]
        self.chosen = {"preferred_candidate": "A", "quality_scores": dict(SCORES), "diagnosis": [{"type": "overlap", "target": "encoder", "severity": "medium", "evidence": "modules overlap"}], "confidence": 1.0}

    def test_identical_answers_are_refused(self):
        self.assertEqual(rejection_reason(self.chosen, dict(self.chosen), self.defects, family="missed_defect", task_type="error_audit"), "the two judgements are the same")

    def test_a_rejected_answer_that_still_names_the_defect_is_refused(self):
        named = {"preferred_candidate": "A", "quality_scores": dict(SCORES), "diagnosis": [{"type": "overlap", "target": "encoder", "severity": "low", "evidence": "faint overlap"}], "confidence": 0.6}
        self.assertIn("still names the injected defect", rejection_reason(self.chosen, named, self.defects, family="missed_defect", task_type="error_audit"))

    def test_an_empty_missed_defect_diagnosis_is_refused(self):
        empty = {**self.chosen, "diagnosis": []}
        self.assertIn("leaves the diagnosis empty", rejection_reason(self.chosen, empty, self.defects, family="missed_defect", task_type="error_audit"))

    def test_a_short_missed_defect_rejection_is_refused(self):
        long_chosen = {
            **self.chosen,
            "diagnosis": [{
                "type": "overlap",
                "target": "encoder",
                "severity": "medium",
                "evidence": "modules overlap across the panel and cover the caption of the neighbouring block; " * 8,
            }],
        }
        short = {
            **long_chosen,
            "diagnosis": [{"type": "assumed_worse", "target": "encoder", "severity": "low", "evidence": "none"}],
        }
        self.assertIn("much shorter", rejection_reason(long_chosen, short, self.defects, family="missed_defect", task_type="error_audit"))

    def test_the_built_missed_defect_rejection_is_admitted(self):
        rejected = rejected_judgement(self.chosen, self.defects, family="missed_defect")
        self.assertEqual(rejection_reason(self.chosen, rejected, self.defects, family="missed_defect", task_type="error_audit"), "")

    def test_an_unclear_pick_is_refused(self):
        self.assertIn("no unambiguous pick", rejection_reason(self.chosen | {"preferred_candidate": "TIE"}, self.chosen | {"preferred_candidate": "B"}, self.defects, family="wrong_pick", task_type="pairwise_preference"))

    def test_known_defects_are_read_from_the_provenance(self):
        self.assertEqual(known_defects({"provenance": {"corruption_type": "module_order_error"}})[0]["diagnosis_type"], "module_order_error")
        self.assertEqual(known_defects({"provenance": {"corruption_type": "not_a_defect"}}), [])
        self.assertEqual(known_defects({}), [])


class ProcessPairTests(unittest.TestCase):
    """A measured rollback is the only admissible rejected side of a process pair."""

    state = {"method": {"modules": [{"id": "n1"}]}, "layout": {"nodes": [{"id": "n1"}], "connectors": [{"id": "e1"}]}}

    def _rows(self, root, *, negative_gain=-0.35, negative_source="controlled_negative_round", injected=True):
        improved = _process_row(root, example_id="run-a-repair-0001", round_index=1, state_before=self.state, decision="accept", gain=0.2, action={"type": "reroute", "edge_ids": ["e1"]})
        rolled_back = _process_row(
            root,
            example_id="neg-a-repair-0002",
            round_index=2,
            state_before=self.state,
            decision="rollback",
            gain=negative_gain,
            action={"type": "reroute", "edge_ids": ["e2"]},
            source=negative_source,
            injected={"kind": "wrong_required_edge_target", "target": "e2", "request": "the required edge should be re-routed away from its source", "action": {"type": "reroute", "edge_ids": ["e2"]}} if injected else None,
        )
        return [improved, rolled_back]

    def test_a_measured_rollback_becomes_the_rejected_side_of_a_measured_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "examples.jsonl"
            _write(source, self._rows(root))
            report = build_preference_pairs(source, root / "pairs.jsonl")
            pairs = _read(root / "pairs.jsonl")
            export = export_preference_pairs(root / "pairs.jsonl", root / "dpo")
            dpo = _read(root / "dpo" / "dpo.jsonl")

        self.assertTrue(report["ok"])
        self.assertEqual(1, report["pairs"])
        self.assertEqual({"harmful_repair": 1}, report["by_family"])
        pair = pairs[0]
        self.assertEqual("measured_repair", pair["chosen"]["source"])
        self.assertEqual("harmful_repair", pair["rejected"]["family"])
        self.assertEqual(["e2"], pair["rejected"]["judgement"]["stage_feedback"][0]["edge_ids"])
        self.assertEqual("connector", pair["rejected"]["judgement"]["stage_feedback"][0]["scope_type"])
        self.assertEqual([{"action": "reroute", "edge_ids": ["e2"]}], pair["rejected"]["judgement"]["stage_feedback"][0]["requested_actions"])
        self.assertNotEqual(pair["chosen"]["judgement"], pair["rejected"]["judgement"])
        check = pair["preference_basis"]["deterministic_check"]
        self.assertEqual("rejected_action_was_rolled_back", check["name"])
        self.assertEqual(-0.35, check["measured_repair_gain"])
        self.assertEqual("controlled_negative_round", check["negative_source"])
        self.assertEqual(1, len(pair["images"]))
        self.assertEqual(1, export["records"])
        self.assertEqual(["text", "image"], [item["type"] for item in dpo[0]["messages"][0]["content"]])

    def test_a_rollback_survives_the_pair_without_a_recorded_defect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "examples.jsonl"
            _write(source, self._rows(root, injected=False))
            build_preference_pairs(source, root / "pairs.jsonl")
            pair = _read(root / "pairs.jsonl")[0]

        self.assertEqual(["e2"], pair["rejected"]["judgement"]["stage_feedback"][0]["edge_ids"])
        self.assertIn("bound to the wrong object", pair["rejected"]["judgement"]["stage_feedback"][0]["problem"])

    def test_a_rollback_without_a_measured_improvement_is_reported_not_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "examples.jsonl"
            rows = self._rows(root)
            rows[0]["observed"]["repair_gain"] = 0.0
            _write(source, rows)
            report = build_preference_pairs(source, root / "pairs.jsonl")
            pairs = _read(root / "pairs.jsonl")

        self.assertEqual([], pairs)
        self.assertFalse(report["ok"])
        self.assertEqual(1, report["skipped"])
        self.assertIn("no measured improvement", report["skipped_sample"][0]["reason"])

    def test_a_run_that_never_rolled_back_teaches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "examples.jsonl"
            rows = self._rows(root)
            rows[1]["observed"]["decision"] = "accept"
            rows[1]["observed"]["repair_gain"] = 0.1
            _write(source, rows)
            report = build_preference_pairs(source, root / "pairs.jsonl")

        self.assertEqual(0, report["pairs"])
        self.assertEqual(0, report["skipped"])


class ExportAndReviewTests(unittest.TestCase):
    def _pairs(self, root):
        source = root / "examples.jsonl"
        _write(source, [_example(root, example_id="p1", task_type="pairwise_preference", kind="relation_reversal", diagnosis_type="wrong_arrow_target", pick="A")])
        build_preference_pairs(source, root / "pairs.jsonl")
        return root / "pairs.jsonl"

    def _query_pairs(self, root):
        source = root / "examples.jsonl"
        _write(source, [
            _example(root, example_id="q1", task_type="pairwise_preference", kind="relation_reversal", diagnosis_type="wrong_arrow_target", pick="A", preview=True),
            _example(root, example_id="q2", task_type="pairwise_preference", kind="relation_reversal", diagnosis_type="wrong_arrow_target", pick="A"),
        ])
        build_preference_pairs(source, root / "pairs.jsonl")
        return root / "pairs.jsonl"

    def test_a_pair_keeps_the_image_the_question_is_about(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = self._query_pairs(root)
            rows = _read(pairs)
            report = build_review_queue(pairs, root / "review", pairs_per_sheet=8)
            queue = _read(root / "review" / "review_queue.jsonl")
            sheets = _sheet_sizes(report["contact_sheets"])
            pair_with_query = next(row for row in rows if row.get("preview"))
            preview_exists = Path(pair_with_query["preview"]).is_file()
            previews = {item["pair_id"]: item["preview"] for item in queue}

        self.assertTrue(preview_exists)
        for pair_id, preview in previews.items():
            self.assertEqual(preview.endswith("q1-query.png"), pair_id.startswith("q1__"))
        # One sheet for both pairs, three columns wide: the query, A, then B.
        self.assertEqual(len(sheets), 1)
        self.assertEqual(sheets[0][0], 280 * 3)

    def test_a_single_image_task_still_lands_on_a_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "examples.jsonl"
            row = _example(root, example_id="s1", task_type="error_audit", kind="relation_reversal", diagnosis_type="wrong_arrow_target", candidate_b=False, preview=True)
            # error_audit is a grounding task, so its certified judgement names the grounded entities.
            row["labels"]["grounding"] = {"grounded_entities": ["encoder", "decoder"]}
            _write(source, [row])
            build_preference_pairs(source, root / "pairs.jsonl")
            report = build_review_queue(root / "pairs.jsonl", root / "review", pairs_per_sheet=8)
            sheets = _sheet_sizes(report["contact_sheets"])
            queue = _read(root / "review" / "review_queue.jsonl")

        # Both families of a single-image example are reviewable: one column wide, no skips.
        self.assertEqual(len(queue), 2)
        self.assertEqual(report["pairs_without_images"], [])
        self.assertEqual(len(sheets), 1)
        # The untouched figure the corruption was made from sits beside the corrupted one.
        self.assertEqual(sheets[0][0] // 280, 2)
        for item in queue:
            self.assertEqual(len(item["images"]), 1)
            self.assertTrue(item["contact_sheet"].endswith("sheet_000.png"))

    def test_a_pair_without_a_query_image_still_lands_on_a_two_column_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = self._pairs(root)
            report = build_review_queue(pairs, root / "review", pairs_per_sheet=8)
            sheets = _sheet_sizes(report["contact_sheets"])
            markdown = (root / "review" / "contact_sheet.md").read_text(encoding="utf-8")

        self.assertEqual(len(sheets), 1)
        self.assertEqual(sheets[0][0] // 280, 2)
        self.assertIn("| query | images |", markdown)

    def test_dpo_export_writes_prompt_chosen_and_rejected_with_a_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = self._pairs(root)
            manifest = export_preference_pairs(pairs, root / "dpo")
            records = _read(root / "dpo" / "dpo.jsonl")

        self.assertTrue(manifest["ok"])
        self.assertEqual(manifest["records"], len(records))
        self.assertEqual(len(manifest["sha256"]), 64)
        record = records[0]
        self.assertEqual([item["type"] for item in record["messages"][0]["content"]], ["text", "image", "image"])
        self.assertNotEqual(record["chosen"], record["rejected"])
        self.assertEqual(len(record["images"]), 2)
        self.assertEqual(record["preference_basis"]["deterministic_check"]["passed"], True)

    def test_require_human_audit_drops_unaudited_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = self._pairs(root)
            manifest = export_preference_pairs(pairs, root / "dpo", require_human_audit=True)
            records = _read(root / "dpo" / "dpo.jsonl")

        self.assertEqual(records, [])
        self.assertEqual(manifest["records"], 0)
        self.assertEqual(len(manifest["dropped"]), 3)

    def test_review_queue_builds_contact_sheets_and_human_review_marks_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = self._pairs(root)
            report = build_review_queue(pairs, root / "review")
            queue = _read(root / "review" / "review_queue.jsonl")
            rows = _read(pairs)
            decisions = root / "decisions.jsonl"
            _write(decisions, [
                {"pair_id": rows[0]["example_id"], "decision": "approved", "reviewer": "<AUTHOR>", "notes": "ok"},
                {"pair_id": rows[1]["example_id"], "decision": "rejected", "reviewer": "<AUTHOR>", "notes": "bad pair"},
            ])
            reviewed = apply_human_review(pairs, decisions, root / "pairs_reviewed.jsonl")
            kept = _read(root / "pairs_reviewed.jsonl")
            sheets = list(report["contact_sheets"])
            sheet_exists = bool(sheets) and Path(sheets[0]).exists()
            contact_markdown = (root / "review" / "contact_sheet.md").read_text(encoding="utf-8")

        self.assertTrue(report["ok"])
        self.assertEqual(report["pending"], 3)
        self.assertEqual(len(sheets), 1)
        self.assertTrue(sheet_exists)
        self.assertIn("p1__wrong_pick", contact_markdown)
        self.assertIn("p1-a.png", contact_markdown)
        self.assertEqual(len(queue), 3)
        self.assertEqual(reviewed["approved"], 1)
        self.assertEqual(reviewed["human_audited"], 1)
        self.assertEqual(len(kept), 2)
        audited = [row for row in kept if row["preference_basis"]["human_audited"]]
        self.assertEqual(audited[0]["preference_basis"]["human_review"]["reviewer"], "<AUTHOR>")


if __name__ == "__main__":
    unittest.main()
