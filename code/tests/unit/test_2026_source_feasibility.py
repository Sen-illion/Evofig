"""Unit tests for the bounded 2026 source feasibility pilot."""

from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "extra-dataset" / "scripts" / "validate_2026_source_feasibility.py"
SPEC = importlib.util.spec_from_file_location("validate_2026_source_feasibility", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
match_title, norm, paper_type = MODULE.match_title, MODULE.norm, MODULE.paper_type


class FeasibilityHelpersTests(unittest.TestCase):
    def test_norm_removes_markup_and_punctuation(self):
        self.assertEqual(norm("A <i>Framework</i>: Test!"), "a framework test")

    def test_exact_title_match_wins(self):
        match, score, method = match_title("A Framework", [{"arxiv_id": "2601.1", "title_norm": "a framework"}])
        self.assertEqual(match["arxiv_id"], "2601.1")
        self.assertEqual(score, 1.0)
        self.assertEqual(method, "exact_title")

    def test_fuzzy_match_requires_high_similarity(self):
        match, score, method = match_title("A Framework for Tests", [{"arxiv_id": "2601.1", "title_norm": "a framework for test"}])
        self.assertIsNotNone(match)
        self.assertGreaterEqual(score, 0.94)
        self.assertEqual(method, "fuzzy_title")

    def test_acl_long_identifier_is_explicit_evidence(self):
        status, reason = paper_type({"paper_id": "2026.acl-long.12"})
        self.assertEqual(status, "known_long_or_full")
        self.assertEqual(reason, "explicit_metadata_or_acl_id")

    def test_findings_is_not_long_full(self):
        status, _ = paper_type({"paper_id": "2026.findings-acl.12"})
        self.assertEqual(status, "unknown")


if __name__ == "__main__":
    unittest.main()
