"""A content-loss verdict that both figures' own modules contradict is asked once more."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evaluation.direct_judges import (
    CONTENT_LOSS_DEFECTS,
    ChoiceAcceptanceJudge,
    DirectJudgeInput,
    DirectPairJudgeInput,
    content_loss_contradicted,
    drawn_content,
)

PROGRAM = {
    "panels": [{"id": "p", "x": 0, "y": 0, "w": 40, "h": 40}],
    "semantic_nodes": [{"id": "n1", "label": "encoder"}, {"id": "n2", "label": "decoder"}],
    "arrows": [{"source": "n1", "target": "n2"}],
}


def _seat(answers, program=None):
    """A seated acceptance judge whose generator hands back one prepared answer per call."""

    seen: list[dict] = []
    body = program if program is not None else PROGRAM

    def generate(record):
        seen.append(record)
        return answers[min(len(seen) - 1, len(answers) - 1)]

    judge = ChoiceAcceptanceJudge(generate=generate)
    return judge, seen, body


def _ask(judge, program, second_program=None):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "a.png").write_bytes(b"a")
        (root / "b.png").write_bytes(b"b")
        return judge.evaluate_pairwise(
            DirectPairJudgeInput(
                before=DirectJudgeInput(method={}, layout={}, icons={}, program=program, preview=root / "a.png"),
                after=DirectJudgeInput(method={}, layout={}, icons={}, program=second_program or program, preview=root / "b.png"),
            )
        )


class DrawnContentTests(unittest.TestCase):
    def test_what_a_figure_draws_is_read_by_label_not_by_id(self):
        body = drawn_content(PROGRAM)
        self.assertEqual(["decoder", "encoder"], body["labels"])
        self.assertEqual([("n1", "n2")], body["connections"])

    def test_a_module_claim_is_contradicted_only_when_both_figures_agree(self):
        self.assertEqual("", content_loss_contradicted(DirectJudgeInput(method={}, layout={}, icons={}, program=PROGRAM), DirectJudgeInput(method={}, layout={}, icons={}, program=PROGRAM), "overlap"))
        self.assertIn("same 2 modules", content_loss_contradicted(DirectJudgeInput(method={}, layout={}, icons={}, program=PROGRAM), DirectJudgeInput(method={}, layout={}, icons={}, program=PROGRAM), "missing_core_entity"))
        thinner = {**PROGRAM, "semantic_nodes": [{"id": "n1", "label": "encoder"}]}
        self.assertEqual("", content_loss_contradicted(DirectJudgeInput(method={}, layout={}, icons={}, program=PROGRAM), DirectJudgeInput(method={}, layout={}, icons={}, program=thinner), "missing_core_entity"))

    def test_the_classes_the_check_watches_are_the_ones_about_content(self):
        self.assertIn("missing_core_entity", CONTENT_LOSS_DEFECTS)
        self.assertIn("invented_module", CONTENT_LOSS_DEFECTS)
        self.assertNotIn("overlap", CONTENT_LOSS_DEFECTS)


class ContentLossReaskTests(unittest.TestCase):
    def test_the_same_answer_twice_stands_and_says_it_was_asked_twice(self):
        judge, seen, program = _seat([json.dumps({"wrong_figure": "B", "defect": "missing_core_entity"})] * 2)
        result = _ask(judge, program)
        self.assertEqual("missing_core_entity", result["defect"])
        self.assertTrue(str(result["seat_reasked"]).startswith("repeated"))
        self.assertEqual(2, len(seen))
        self.assertIn("same 2 modules", seen[1]["messages"][0]["content"][0]["text"])

    def test_a_different_answer_is_the_answer(self):
        judge, seen, program = _seat(
            [
                json.dumps({"wrong_figure": "B", "defect": "missing_core_entity"}),
                json.dumps({"wrong_figure": "A", "defect": "overlap"}),
            ]
        )
        result = _ask(judge, program)
        self.assertEqual("overlap", result["defect"])
        self.assertEqual("A", result["wrong_figure"])
        self.assertEqual("B", result["preferred_candidate"])
        self.assertTrue(str(result["seat_reasked"]).startswith("changed"))
        self.assertEqual(2, len(seen))

    def test_a_verdict_about_lines_is_never_asked_again(self):
        judge, seen, program = _seat([json.dumps({"wrong_figure": "B", "defect": "overlap"})] * 2)
        result = _ask(judge, program)
        self.assertEqual("overlap", result["defect"])
        self.assertEqual("", result["seat_reasked"])
        self.assertEqual(1, len(seen))

    def test_the_sentence_and_the_not_a_score_note_travel_with_the_answer(self):
        judge, _, program = _seat([json.dumps({"wrong_figure": "B", "defect": "overlap"})])
        result = _ask(judge, program)
        self.assertIn("no score was taken", result["verdict_reason"])
        self.assertIn("overlap", result["verdict_reason"])
        self.assertEqual("derived_from_defect", result["choice_scores"]["after"]["score_source"])


if __name__ == "__main__":
    unittest.main()