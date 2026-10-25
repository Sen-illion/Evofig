"""Tests for direct pipeline output validation."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evaluation.outputs import LEGACY_ARTIFACT_NAMES, REQUIRED_JSON, validate_output


class DirectOutputValidationTests(unittest.TestCase):
    def test_accepts_complete_direct_run_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in REQUIRED_JSON:
                (root / name).write_text("{}", encoding="utf-8")
            (root / "grounding_gate.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
            (root / "editable_composition.pptx").write_bytes(b"pptx")
            (root / "evolution_events.jsonl").write_text(json.dumps({"event": "round_started"}) + "\n", encoding="utf-8")
            self.assertTrue(validate_output(root)["ok"])

    def test_rejects_missing_trajectory_and_delivery(self):
        with tempfile.TemporaryDirectory() as temp:
            result = validate_output(temp)
            self.assertFalse(result["ok"])
            self.assertTrue(any("editable_composition.pptx" in item for item in result["errors"]))

    def test_accepts_a_run_that_stored_the_plan_under_its_earlier_name(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in REQUIRED_JSON:
                (root / LEGACY_ARTIFACT_NAMES.get(name, name)).write_text("{}", encoding="utf-8")
            (root / "grounding_gate.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
            (root / "editable_composition.pptx").write_bytes(b"pptx")
            (root / "evolution_events.jsonl").write_text(json.dumps({"event": "round_started"}) + "\n", encoding="utf-8")

            self.assertFalse((root / "figure_content_plan.json").is_file())
            self.assertTrue(validate_output(root)["ok"], validate_output(root)["errors"])


if __name__ == "__main__":
    unittest.main()
