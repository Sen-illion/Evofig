"""Tests for the one ruler that says what a round introduced."""

import unittest

from rfs.contracts.evolution import introduced_defects


class IntroducedDefectBasisTests(unittest.TestCase):
    """Three ways to read the same question, and which one a pair of answers supports."""

    def test_both_sides_carrying_counts_are_read_as_counts(self):
        reading = introduced_defects(
            {"deterministic_blocker_counts": {"interface:box_pass": 3}},
            {"deterministic_blocker_counts": {"interface:box_pass": 2}},
        )
        self.assertEqual(reading["basis"], "counts")
        # Three occurrences down to two is a defect that got smaller, not one that appeared.
        self.assertEqual(reading["defects"], [])
        self.assertEqual(reading["count"], 0)

    def test_a_kind_whose_count_rose_is_the_one_introduced(self):
        reading = introduced_defects(
            {"deterministic_blocker_counts": {"interface:box_pass": 1}},
            {"deterministic_blocker_counts": {"interface:box_pass": 1, "interface:boundary_violation": 4}},
        )
        self.assertEqual(reading["basis"], "counts")
        self.assertEqual(reading["defects"], ["interface:boundary_violation"])

    def test_one_side_carrying_counts_falls_back_to_the_names(self):
        reading = introduced_defects(
            {"deterministic_blockers": ["interface:box_pass"]},
            {
                "deterministic_blockers": ["interface:box_pass", "interface:reroute"],
                "deterministic_blocker_counts": {"interface:box_pass": 1, "interface:reroute": 2},
            },
        )
        self.assertEqual(reading["basis"], "names")
        self.assertEqual(reading["defects"], ["interface:reroute"])

    def test_an_empty_count_map_is_not_a_reading_of_zero(self):
        # The seat that dropped the counts reported an empty map. Reading that as "found nothing"
        # is how a round that introduced a defect came out as one that introduced none.
        reading = introduced_defects(
            {"deterministic_blockers": ["interface:box_pass"], "deterministic_blocker_counts": {}},
            {
                "deterministic_blockers": ["interface:box_pass", "interface:boundary_violation"],
                "deterministic_blocker_counts": {},
            },
        )
        self.assertEqual(reading["basis"], "names")
        self.assertEqual(reading["defects"], ["interface:boundary_violation"])

    def test_neither_side_separating_its_defects_compares_the_totals(self):
        reading = introduced_defects({"blocking_issue_count": 2}, {"blocking_issue_count": 5})
        self.assertEqual(reading["basis"], "totals")
        self.assertEqual(reading["count"], 3)
        self.assertEqual(reading["defects"], [])

    def test_a_falling_total_is_not_an_introduction(self):
        reading = introduced_defects({"blocking_issue_count": 5}, {"blocking_issue_count": 2})
        self.assertEqual(reading["count"], 0)
        self.assertEqual(reading["basis"], "totals")


if __name__ == "__main__":
    unittest.main()
