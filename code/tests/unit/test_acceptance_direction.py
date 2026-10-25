"""The acceptance seat: the whole-first rule, the two readings, and the direction question."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evaluation.direct_judges import (
    ChoiceAcceptanceJudge,
    DirectJudgeInput,
    DirectPairJudgeInput,
    choice_pair_judgement,
    pair_readings,
)
from rfs.judge_training.choice_items import choice_prompt, parse_choice, parse_choice_answer
from rfs.workflows.continuous_evolution import _human_seat_spec, acceptance_directions

MENU = ["overlap", "no_defect", "wrong_arrow_target"]


def _old_prompt(title, evidence, menu):
    options = ", ".join(menu)
    return (
        "Look at figure A and figure B. Choose which figure is wrong and which defect it has. "
        "wrong_figure must be A or B. "
        f"defect must be one of: {options}. "
        'Reply with only {"wrong_figure":"A or B","defect":"one listed option"}. '
        f"Paper title: {title}\n"
        f"Paper evidence: {evidence}"
    )


class ChoicePromptTests(unittest.TestCase):
    def test_a_question_with_no_rule_or_readings_is_the_question_it_always_was(self):
        self.assertEqual(_old_prompt("T", "E", MENU), choice_prompt("T", "E", MENU))


class ReadChoiceTests(unittest.TestCase):
    def test_the_pair_is_read_out_of_an_answer_that_also_took_the_direction_question(self):
        text = json.dumps({"wrong_figure": "B", "defect": "overlap", "layer": "segmentation", "settled": False, "still_wrong": "text overlap", "because": "density"})
        self.assertEqual(("B", "overlap"), parse_choice(text, MENU))

    def test_prose_after_the_object_is_still_no_answer(self):
        text = json.dumps({"wrong_figure": "B", "defect": "overlap"}) + " because B overlaps"
        self.assertIsNone(parse_choice(text, MENU))

    def test_an_answer_cut_off_before_it_closed_still_keeps_its_two_fields(self):
        text = '{"wrong_figure":"B","defect":"overlap","layer":"segmentation","settled":tr'
        self.assertEqual(("B", "overlap"), parse_choice(text, MENU))

    def test_the_direction_question_is_read_in_the_order_it_is_asked(self):
        settled = parse_choice_answer(json.dumps({"wrong_figure": "A", "defect": "no_defect", "settled": True, "next_layer": "icon_generation", "because": "clean"}), MENU, ("icon_generation", "segmentation"))
        self.assertTrue(settled["direction"]["settled"])
        self.assertEqual("icon_generation", settled["direction"]["next_layer"])
        self.assertEqual("read", settled["direction_problem"])

    def test_a_silent_direction_half_leaves_the_pair_answer_alone(self):
        silent = parse_choice_answer(json.dumps({"wrong_figure": "B", "defect": "overlap"}), MENU, ("segmentation",))
        self.assertEqual("B", silent["wrong_figure"])
        self.assertEqual({}, silent["direction"])
        self.assertTrue(silent["direction_problem"])

    def test_the_pair_verdict_is_unchanged_by_the_direction_half(self):
        plain = choice_pair_judgement(json.dumps({"wrong_figure": "A", "defect": "overlap"}))
        with_direction = choice_pair_judgement(json.dumps({"wrong_figure": "A", "defect": "overlap", "settled": False, "still_wrong": "overlap"}))
        self.assertEqual("B", plain["preferred_candidate"])
        self.assertEqual(plain["preferred_candidate"], with_direction["preferred_candidate"])
        self.assertEqual({}, plain["direction"])
        self.assertEqual("overlap", with_direction["direction"]["still_wrong"])


class PairReadingsTests(unittest.TestCase):
    def test_each_side_is_measured_from_its_own_program(self):
        before = DirectJudgeInput(method={}, layout={}, icons={}, program={"panels": [], "semantic_nodes": [], "arrows": []})
        after = DirectJudgeInput(method={}, layout={}, icons={}, program={"panels": [], "semantic_nodes": [], "arrows": []})
        text = pair_readings(before, after)
        self.assertIn("A: crossings=", text)
        self.assertIn("B: crossings=", text)

    def test_a_side_with_no_program_adds_nothing(self):
        self.assertEqual("", pair_readings(DirectJudgeInput(method={}, layout={}, icons={}, program={}), None))


class AcceptanceSeatTests(unittest.TestCase):
    def _pair(self, generate, direction="record", layer_plan=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before = root / "before.png"
            after = root / "after.png"
            before.write_bytes(b"a")
            after.write_bytes(b"b")
            seen: list[dict] = []

            def asked(record):
                seen.append(record)
                return json.dumps({"wrong_figure": "B", "defect": "overlap", "settled": False, "still_wrong": "overlap", "because": "readings"})

            judge = ChoiceAcceptanceJudge(generate=asked if generate is None else generate, direction=direction)
            inputs = dict(method={"title": "p"}, layout={"nodes": [{"id": "n", "label": "encoder"}]}, icons={}, program={"panels": [{"id": "p1", "x": 0, "y": 0, "w": 40, "h": 40}], "semantic_nodes": [], "arrows": []})
            result = judge.evaluate_pairwise(
                DirectPairJudgeInput(
                    before=DirectJudgeInput(preview=before, **inputs),
                    after=DirectJudgeInput(preview=after, layer_plan=layer_plan, **inputs),
                )
            )
            return result, seen

    def test_the_authority_the_seat_was_given_travels_with_its_answer(self):
        result, _ = self._pair(None, direction="record")
        self.assertEqual("record", result["direction_authority"])
        self.assertEqual("overlap", result["direction"]["still_wrong"])
        self.assertEqual("B", result["wrong_figure"])

    def test_the_direction_question_is_only_put_when_the_seat_is_asked_for_it(self):
        _, asked_off = self._pair(None, direction="off")
        _, asked_on = self._pair(None, direction="authority", layer_plan={"working_on": "segmentation", "layer_order": ["content_planning", "segmentation"]})
        self.assertNotIn("settled", asked_off[0]["messages"][0]["content"][0]["text"])
        text = asked_on[0]["messages"][0]["content"][0]["text"]
        self.assertIn("Judge the whole figure before you judge anything in it.", text)
        self.assertIn("It is working on the layer segmentation", text)
        self.assertIn("crossings=", text)

    def test_a_seat_that_did_not_take_the_question_still_answers_the_comparison(self):
        def only_pair(record):
            return json.dumps({"wrong_figure": "A", "defect": "no_defect"})

        result, _ = self._pair(only_pair, direction="record")
        self.assertEqual("B", result["preferred_candidate"])
        self.assertEqual({}, result["direction"])
        self.assertIsNone(result["pairwise_unavailable"])


class DirectionRecordTests(unittest.TestCase):
    def test_the_human_seat_block_is_off_unless_it_is_written(self):
        spec = _human_seat_spec(None)
        self.assertFalse(spec["enabled"])
        self.assertEqual([], spec["seats"])
        self.assertEqual("record", spec["direction_authority"])

    def test_a_written_direction_authority_is_read_and_a_typo_falls_back(self):
        self.assertEqual("authority", _human_seat_spec({"enabled": True, "direction_authority": "authority"})["direction_authority"])
        self.assertEqual("record", _human_seat_spec({"direction_authority": "whatever"})["direction_authority"])

    def test_the_ledger_reads_every_round_the_seat_was_asked_about(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "round_00").mkdir()
            (root / "round_01").mkdir()
            (root / "round_00" / "acceptance_direction.json").write_text(
                json.dumps({"round": 0, "authority": "record", "asked_about": "segmentation", "answer": {"settled": False}}), encoding="utf-8"
            )
            rows = acceptance_directions(root)
        self.assertEqual(1, len(rows))
        self.assertEqual("segmentation", rows[0]["asked_about"])
        self.assertEqual({"settled": False}, rows[0]["answer"])


if __name__ == "__main__":
    unittest.main()