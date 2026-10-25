"""The one reading of a priority word, shared by the contract, the planner and the layout.

The table used to exist three times -- in the method contract, the structure planner and the
card layout -- and the three did not agree. The contract's copy had no table at all: it read
whatever the model wrote with ``float()``, so ``importance: "critical"`` raised, and the
boundary review of the finished figure died with it four iterations in a row.
"""

import unittest

from rfs.contracts.method_figure import normalize_method_figure_content, validate_method_figure_content
from rfs.contracts.priority import DEFAULT_PRIORITY, PRIORITY_WORDS, priority_number


class PriorityNumberTests(unittest.TestCase):
    def test_every_word_the_table_knows_reads_as_its_number(self):
        for word, expected in PRIORITY_WORDS.items():
            for spelling in (word, word.upper(), f"  {word}  "):
                self.assertEqual(priority_number(spelling), expected, spelling)

    def test_a_number_is_taken_as_itself_and_kept_on_the_scale(self):
        self.assertEqual(priority_number(0.7), 0.7)
        self.assertEqual(priority_number("0.7"), 0.7)
        self.assertEqual(priority_number(-1), 0.0)
        self.assertEqual(priority_number(4), 1.0)

    def test_a_word_the_table_does_not_know_is_the_default_and_not_an_error(self):
        for value in ("very high", "", None, [], {"word": "critical"}, object()):
            self.assertEqual(priority_number(value), DEFAULT_PRIORITY)

    def test_the_caller_may_name_its_own_default(self):
        self.assertEqual(priority_number(None, default=0.2), 0.2)


class MethodFigureWordPriorityTests(unittest.TestCase):
    def test_a_priority_written_as_a_word_no_longer_ends_the_review(self):
        content = normalize_method_figure_content({
            "inputs": [{"id": "papers", "label": "Papers", "importance": "critical", "evidence_ids": ["E1"]}],
            "modules": [{"id": "reader", "label": "Reader", "visual_priority": "high", "evidence_ids": ["E1"]}],
            "outputs": [{"id": "figure", "label": "Figure", "importance": "low", "evidence_ids": ["E1"]}],
        })
        self.assertEqual(content["inputs"][0]["scientific_importance"], PRIORITY_WORDS["critical"])
        self.assertEqual(content["modules"][0]["visualization_value"], PRIORITY_WORDS["high"])
        self.assertEqual(content["outputs"][0]["scientific_importance"], PRIORITY_WORDS["low"])
        report = validate_method_figure_content(content)
        self.assertTrue(report["ok"], report["errors"])


if __name__ == "__main__":
    unittest.main()
