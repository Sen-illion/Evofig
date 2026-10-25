"""The reference-library resolution record names why a run had no visual goal."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.analysis.reference.exemplars import resolve_project_exemplars


def _row(identifier, label, macro_pattern="left-to-right workflow with labelled modules and connector pathways"):
    return json.dumps(
        {
            "id": identifier,
            "quality_label": label,
            "layout": {"macro_pattern": macro_pattern},
            "structure": {"grouping_style": "bands"},
            "text": {"label_style": "short"},
            "connectors": {"grammar": ["orthogonal"]},
        },
        ensure_ascii=False,
    )


class ReferenceLibraryResolutionTests(unittest.TestCase):
    def _resolve(self, root, *, configured, readable, top_k=5):
        return resolve_project_exemplars(
            root=root,
            paper_review={},
            selected_template={},
            reference_library=readable,
            top_k=top_k,
            configured_library=configured,
        )

    def _record(self, root):
        return json.loads((root / "reference_library_resolution.json").read_text(encoding="utf-8"))

    def test_no_library_configured_is_named_as_such(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self._resolve(root, configured=None, readable=None)
            record = self._record(root)
        self.assertIsNone(result)
        self.assertEqual(record["status"], "not_configured")

    def test_a_configured_path_that_does_not_exist_is_named_as_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            missing = root / "library" / "reference_library.jsonl"
            result = self._resolve(root, configured=missing, readable=None)
            record = self._record(root)
        self.assertIsNone(result)
        self.assertEqual(record["status"], "configured_missing")
        self.assertEqual(record["source"], str(missing))

    def test_a_library_with_no_positive_exemplar_is_named_empty_and_does_not_crash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "reference_library.jsonl"
            library.write_text(_row("neg-1", "negative") + "\n", encoding="utf-8")
            result = self._resolve(root, configured=library, readable=library)
            record = self._record(root)
        self.assertIsNone(result)
        self.assertEqual(record["status"], "configured_empty")

    def test_a_selected_library_records_the_counts_it_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "reference_library.jsonl"
            library.write_text(
                _row("pos-1", "positive") + "\n" + _row("neg-1", "negative") + "\n",
                encoding="utf-8",
            )
            result = self._resolve(root, configured=library, readable=library)
            record = self._record(root)
        self.assertIsNotNone(result)
        self.assertEqual(record["status"], "selected")
        self.assertEqual(record["positive_count"], 1)
        self.assertEqual(record["framework_positive_count"], 1)
        self.assertEqual(record["excluded_by_kind"], {"data_panel": [], "unknown": []})
        self.assertEqual(record["negative_count"], 1)
        self.assertEqual(record["record_count"], 2)

    def test_a_library_of_data_figures_is_named_rather_than_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "reference_library.jsonl"
            library.write_text(_row("pos-1", "positive", "Balanced 2-by-3 small-multiple grid") + "\n", encoding="utf-8")
            result = self._resolve(root, configured=library, readable=library)
            record = self._record(root)
        self.assertIsNone(result)
        self.assertEqual(record["status"], "configured_no_framework_positive")
        self.assertEqual(record["excluded_by_kind"], {"data_panel": ["pos-1"], "unknown": []})


if __name__ == "__main__":
    unittest.main()
