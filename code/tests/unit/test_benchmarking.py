"""Tests for direct-pipeline benchmark discovery and validation."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evaluation.benchmarking import DIRECT_SUITE, list_benchmark_cases, validate_benchmark_case


class BenchmarkingTests(unittest.TestCase):
    def test_valid_direct_case(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp)
            (case / "inputs").mkdir()
            (case / "inputs" / "paper.pdf").write_bytes(b"%PDF-test")
            (case / "case.json").write_text(json.dumps({"case_id": "c1", "suite": DIRECT_SUITE, "thresholds": {}, "paper": "inputs/paper.pdf"}), encoding="utf-8")
            (case / "expected_contract.json").write_text(json.dumps({"required_nodes": [], "required_edges": []}), encoding="utf-8")
            self.assertTrue(validate_benchmark_case(case)["ok"])

    def test_lists_only_direct_suite(self):
        with tempfile.TemporaryDirectory() as temp:
            result = list_benchmark_cases(temp)
            self.assertEqual(result["cases"], [])


if __name__ == "__main__":
    unittest.main()
