"""Regression tests for the bundled framework-output validator."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "research-figure-studio"
    / "scripts"
    / "validate_framework_outputs.py"
)


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("bundled_output_validator", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load validator: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BundledOutputValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = _load_validator_module()

    def test_accepts_current_positive_negative_reference_contract(self) -> None:
        manifest = {
            "paper_archived": "inputs/paper.pdf",
            "positive_references": ["inputs/positive_references/reference.png"],
            "negative_references": [],
        }
        self.assertEqual(
            self.validator.archived_reference_paths(manifest),
            ["inputs/positive_references/reference.png"],
        )

    def test_keeps_legacy_reference_contract_compatible(self) -> None:
        manifest = {
            "paper_archived": "inputs/paper.pdf",
            "reference_archived": "inputs/reference.png",
        }
        self.assertEqual(
            self.validator.archived_reference_paths(manifest),
            ["inputs/reference.png"],
        )

    def test_accepts_canonical_direct_pipeline_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in self.validator.DIRECT_REQUIRED_JSON:
                payload = {"summary": name}
                if name == "grounding_gate.json":
                    payload["status"] = "pass"
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            (root / "evolution_events.jsonl").write_text(
                json.dumps({"event": "round_started"}) + "\n",
                encoding="utf-8",
            )
            with zipfile.ZipFile(root / self.validator.MAIN_PPTX_NAME, "w") as archive:
                archive.writestr("[Content_Types].xml", "<Types />")

            self.validator.validate_direct_pipeline_output(root)

    def test_accepts_a_run_that_stored_the_plan_under_its_earlier_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in self.validator.DIRECT_REQUIRED_JSON:
                stored = self.validator.EARLIER_ARTIFACT_NAMES.get(name, name)
                payload = {"summary": name}
                if name == "grounding_gate.json":
                    payload["status"] = "pass"
                (root / stored).write_text(json.dumps(payload), encoding="utf-8")
            (root / "evolution_events.jsonl").write_text(
                json.dumps({"event": "round_started"}) + "\n",
                encoding="utf-8",
            )
            with zipfile.ZipFile(root / self.validator.MAIN_PPTX_NAME, "w") as archive:
                archive.writestr("[Content_Types].xml", "<Types />")

            self.assertFalse((root / "figure_content_plan.json").is_file())
            self.validator.validate_direct_pipeline_output(root)


    def test_arrow_quality_accepts_the_new_crossing_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "arrow_quality_report.json"
            path.write_text(json.dumps({"crossing_count": 2, "average_aesthetic_score": 91.0}), encoding="utf-8")

            self.validator.validate_arrow_quality_report(path)

    def test_arrow_quality_still_accepts_the_archived_crossing_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "arrow_quality_report.json"
            path.write_text(json.dumps({"total_crossing_count": 2, "average_aesthetic_score": 91.0}), encoding="utf-8")

            self.validator.validate_arrow_quality_report(path)

    def test_arrow_quality_needs_one_of_the_crossing_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "arrow_quality_report.json"
            path.write_text(json.dumps({"average_aesthetic_score": 91.0}), encoding="utf-8")

            with self.assertRaises(SystemExit):
                self.validator.validate_arrow_quality_report(path)


if __name__ == "__main__":
    unittest.main()
