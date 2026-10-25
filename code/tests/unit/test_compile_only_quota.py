"""A rule-class row is asked to travel across more papers than a two-arm row.

The compile-only path measures a rule by compiling one figure twice, which costs minutes instead
of hours, so the default exam is widened for exactly those rows. A suite that says how many cases
it wants still decides, and a row measured by two real runs keeps the narrow default.
"""

from __future__ import annotations

import unittest

from rfs.analysis.graph import COMPILE_ONLY_CASE_QUOTA, DEFAULT_CASE_QUOTA
from rfs.evolution.direct.validation import PAIRED_RUN_PATH, COMPILE_ONLY_PATH
from rfs.workflows.direct_experience_validation import _compile_only_reason


class CompileOnlyQuotaTests(unittest.TestCase):
    def test_the_widened_default_holds_between_six_and_ten_cases(self):
        total = sum(COMPILE_ONLY_CASE_QUOTA.values())
        self.assertGreaterEqual(total, 6)
        self.assertLessEqual(total, 10)
        # Every group the narrow default asks for is asked for at least as hard.
        for group, count in DEFAULT_CASE_QUOTA.items():
            self.assertGreaterEqual(COMPILE_ONLY_CASE_QUOTA[group], count, group)

    def test_the_groups_a_common_row_must_face_stay_funded(self):
        self.assertGreater(COMPILE_ONLY_CASE_QUOTA["cross_domain_same_structure"], 0)
        self.assertGreater(COMPILE_ONLY_CASE_QUOTA["boundary"], 0)

    def test_a_mechanism_action_earns_the_wide_quota_and_a_repair_does_not(self):
        rule = {"apply_at": "ppt_compilation", "action": {"type": "layout_constraint", "params": {"align_grid": 0.02}}}
        self.assertEqual("", _compile_only_reason(rule))
        self.assertNotEqual(COMPILE_ONLY_PATH, PAIRED_RUN_PATH)
        repair = {"apply_at": "segmentation", "action": {"type": "resegment", "params": {}}}
        self.assertNotEqual("", _compile_only_reason(repair))


if __name__ == "__main__":
    unittest.main()
