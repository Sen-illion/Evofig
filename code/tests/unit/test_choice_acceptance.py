"""Acceptance short answers, holdout exclusion, and the three-layer change record."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evaluation.direct_judges import ChoiceAcceptanceJudge, DirectJudgeInput, DirectPairJudgeInput, build_direct_judges
from rfs.evolution.direct.snapshots import layer_change
from rfs.judge_training.holdout_papers import assert_paper_not_held_out, load_blocked_paper_ids


def _pair(generate, before_preview: Path, after_preview: Path) -> dict:
    judge = ChoiceAcceptanceJudge(generate=generate)
    before = DirectJudgeInput(method={"title": "paper"}, layout={"nodes": [{"id": "n", "label": "encoder"}]}, icons={}, program={}, preview=before_preview)
    after = DirectJudgeInput(method={"title": "paper"}, layout={"nodes": [{"id": "n", "label": "encoder"}]}, icons={}, program={}, preview=after_preview)
    return judge.evaluate_pairwise(DirectPairJudgeInput(before=before, after=after))


class ChoiceAcceptanceTests(unittest.TestCase):
    def test_a_wrong_before_figure_scores_the_defect_and_keeps_the_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before = root / "before.png"
            after = root / "after.png"
            before.write_bytes(b"a")
            after.write_bytes(b"b")
            seen: list[dict] = []

            def generate(record):
                seen.append(record)
                return json.dumps({"wrong_figure": "A", "defect": "overlap"})

            result = _pair(generate, before, after)
            advisory, frozen = build_direct_judges(mode="heuristic", acceptance_generate=generate)
            single = frozen.evaluate(DirectJudgeInput(method={}, layout={}, icons={}, program={}))

        self.assertEqual(result["preferred_candidate"], "B")
        self.assertEqual(result["wrong_figure"], "A")
        self.assertEqual(result["defect"], "overlap")
        self.assertIsNone(result["pairwise_unavailable"])
        self.assertEqual(0.6, result["choice_scores"]["before"]["overall_score"])
        self.assertEqual(1.0, result["choice_scores"]["after"]["overall_score"])
        self.assertEqual(0.6, result["choice_scores"]["before"]["dimension_scores"]["scientific_fidelity"])
        self.assertEqual([], result["stage_feedback"])
        self.assertEqual([], result["repair"])
        self.assertIn("wrong_figure", seen[0]["messages"][0]["content"][0]["text"])
        self.assertEqual("frozen_choice", frozen.role)
        self.assertEqual("online_heuristic", advisory.role)
        self.assertEqual([], single["stage_feedback"])
        self.assertIsNone(single["overall_score"])
        self.assertFalse(single["score_measured"])
        # The seat answers no score of its own -- and the deterministic gate it runs first keeps
        # its own reading on the answer. Without it, a round whose pair question could not be
        # asked carried no measurement at all and could never become Memory.
        self.assertEqual(0.0, single["gate_score"])
        self.assertIn("deterministic_gate", single["gate_dimension_scores"])

    def test_the_choice_seat_carries_the_gate_counts_and_not_only_its_names(self):
        """The comparison weighs counts, so the seat has to forward them.

        A seat that forwarded only the names left the other side of the comparison with nothing
        to read, and "no count was reported" then read as "no defect was introduced" -- the same
        reading that let a round which plainly introduced a defect come out clean.
        """

        _, frozen = build_direct_judges(mode="heuristic")
        broken = DirectJudgeInput(
            method={"modules": [{"id": "n"}], "relations": []},
            layout={"panels": [], "nodes": [], "connectors": []},
            icons={"icons": [{"asset_id": "icon_n", "semantic_node_ids": ["n"], "prompt": "n"}]},
            program={"panels": [], "semantic_nodes": [], "slots": [], "text_program": {"items": []}, "arrows": []},
        )

        result = frozen.evaluate(broken)

        self.assertTrue(result["deterministic_blocker_counts"])
        self.assertEqual(sorted(result["deterministic_blockers"]), sorted(result["deterministic_blocker_counts"]))

    def test_a_wrong_after_figure_prefers_the_figure_already_kept(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before = root / "before.png"
            after = root / "after.png"
            before.write_bytes(b"a")
            after.write_bytes(b"b")
            result = _pair(lambda record: json.dumps({"wrong_figure": "B", "defect": "no_defect"}), before, after)
        self.assertEqual(result["preferred_candidate"], "A")
        self.assertEqual(1.0, result["choice_scores"]["before"]["overall_score"])
        self.assertEqual(0.6, result["choice_scores"]["after"]["overall_score"])

    def test_the_choice_scores_are_derived_and_never_measured(self):
        """The numbers are looked up in the defect table, so they are labelled and unmeasured."""

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before = root / "before.png"
            after = root / "after.png"
            before.write_bytes(b"a")
            after.write_bytes(b"b")
            result = _pair(lambda record: json.dumps({"wrong_figure": "A", "defect": "overlap"}), before, after)
        for side in ("before", "after"):
            self.assertEqual("derived_from_defect", result["choice_scores"][side]["score_source"])
            self.assertFalse(result["choice_scores"][side].get("score_measured", False))

    def test_an_illegal_answer_is_an_unavailable_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before = root / "before.png"
            after = root / "after.png"
            before.write_bytes(b"a")
            after.write_bytes(b"b")
            result = _pair(lambda record: "the left one looks worse", before, after)
        self.assertEqual(result["preferred_candidate"], "")
        self.assertIn("two-field", result["pairwise_unavailable"])
        self.assertEqual({}, result["choice_scores"])


class HoldoutPaperTests(unittest.TestCase):
    def test_a_missing_list_refuses_the_run(self):
        with self.assertRaises(FileNotFoundError):
            load_blocked_paper_ids([])
        with self.assertRaises(FileNotFoundError):
            load_blocked_paper_ids([Path("missing-holdout.jsonl")])

    def test_a_listed_paper_is_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "holdout.jsonl"
            path.write_text(json.dumps({"paper_id": "paper-a"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                assert_paper_not_held_out("paper-a", [path])
            blocked = assert_paper_not_held_out("paper-b", [path])
        self.assertEqual({"paper-a"}, blocked)


class LayerChangeTests(unittest.TestCase):
    def test_the_first_kept_figure_has_no_earlier_layer_to_compare(self):
        current = {"scientific": "s", "visual": "v", "editable": "e"}
        record = layer_change(None, current, before_score=None, after_score=0.4, accepted=True)
        self.assertIsNone(record["scientific_changed"])
        self.assertIsNone(record["visual_changed"])
        self.assertIsNone(record["editable_changed"])
        moved = layer_change(current, {"scientific": "s2", "visual": "v", "editable": "e"}, before_score=0.4, after_score=0.7, accepted=False)
        self.assertTrue(moved["scientific_changed"])
        self.assertFalse(moved["visual_changed"])
        self.assertFalse(moved["editable_changed"])
        self.assertEqual(0.3, moved["score_delta"])
        self.assertFalse(moved["accepted"])


if __name__ == "__main__":
    unittest.main()
