"""A reference is offered as a framework exemplar only when it says it is one."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.analysis.reference.exemplars import (
    NoFrameworkExemplarError,
    resolve_project_exemplars,
    select_reference_exemplars,
)
from rfs.analysis.reference.figure_kind import (
    classify_figure_kind,
    is_framework_figure,
    partition_by_kind,
)


def _framework_record(identifier, label="positive", **extra):
    record = {
        "id": identifier,
        "quality_label": label,
        "layout": {"macro_pattern": "left-to-right workflow with labelled modules and connector pathways"},
        "structure": {"grouping_style": "rounded containers held in aligned columns"},
        "density": {"level": "medium"},
        "text": {"label_style": "short bold module titles"},
        "connectors": {"grammar": ["orthogonal arrows"]},
        "why_good": ["The process stages read left to right."],
    }
    record.update(extra)
    return record


def _data_record(identifier, label="positive", **extra):
    record = {
        "id": identifier,
        "quality_label": label,
        "layout": {"macro_pattern": "Balanced 2-by-3 small-multiple grid"},
        "structure": {"grouping_style": "Lettered small multiples with column comparison"},
        "density": {"level": "high", "whitespace_policy": "generous internal plotting margins"},
        "text": {"label_style": "compact axis labels"},
        "connectors": {"grammar": ["thin arrows"]},
        "why_good": ["Consistent axes and scales support visual comparison."],
    }
    record.update(extra)
    return record


def _write_library(path, records):
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


class FigureKindTests(unittest.TestCase):
    def test_a_record_that_names_measured_content_is_a_data_figure(self):
        verdict = classify_figure_kind(_data_record("data-1"))
        self.assertEqual(verdict["kind"], "data_panel")
        self.assertIn("small-multiple", verdict["data_evidence"])
        self.assertFalse(is_framework_figure(_data_record("data-1")))

    def test_a_record_that_names_a_workflow_is_a_framework_figure(self):
        verdict = classify_figure_kind(_framework_record("fw-1"))
        self.assertEqual(verdict["kind"], "framework")
        self.assertTrue(verdict["framework_evidence"])

    def test_a_record_with_no_evidence_either_way_is_unknown_not_guessed(self):
        verdict = classify_figure_kind({"id": "x", "layout": {"macro_pattern": "three columns"}})
        self.assertEqual(verdict["kind"], "unknown")
        self.assertEqual(verdict["source"], "analysis_text")

    def test_two_measured_content_words_outrank_the_workflow_words_beside_them(self):
        mixed = _framework_record(
            "mixed",
            layout={"macro_pattern": "workflow with a small-multiple and a box plot evidence band"},
        )
        verdict = classify_figure_kind(mixed)
        self.assertEqual(verdict["kind"], "data_panel")
        self.assertEqual(verdict["data_evidence"], ["small-multiple", "box plot"])

    def test_one_measured_inset_beside_a_workflow_stays_a_framework_figure(self):
        # A framework figure that carries a single measured inset -- one micrograph, one
        # contour map -- describes that inset with the same words a data figure uses, so a
        # lone hit no longer drops a record that also names its workflow.
        verdict = classify_figure_kind(
            _framework_record("inset", layout={"macro_pattern": "workflow with one micrograph inset"})
        )
        self.assertEqual(verdict["kind"], "framework")
        self.assertEqual(verdict["data_evidence"], ["micrograph"])

    def test_a_lone_measured_word_without_a_workflow_word_is_still_a_data_figure(self):
        verdict = classify_figure_kind(
            {"id": "lone", "layout": {"macro_pattern": "three columns built around a contour map"}}
        )
        self.assertEqual(verdict["kind"], "data_panel")
        self.assertEqual(verdict["data_evidence"], ["contour"])

    def test_an_explicit_kind_on_the_record_wins_over_the_text(self):
        verdict = classify_figure_kind(_data_record("declared", figure_kind="framework"))
        self.assertEqual(verdict["kind"], "framework")
        self.assertEqual(verdict["source"], "record")

    def test_partition_keeps_the_order_it_arrived_in(self):
        buckets = partition_by_kind([_framework_record("a"), _data_record("b"), {"id": "c"}])
        self.assertEqual([record["id"] for record in buckets["framework"]], ["a"])
        self.assertEqual([record["id"] for record in buckets["data_panel"]], ["b"])
        self.assertEqual([record["id"] for record in buckets["unknown"]], ["c"])


class ReferenceSelectionFilterTests(unittest.TestCase):
    def test_a_data_figure_is_not_offered_even_when_it_could_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "reference_library.jsonl"
            _write_library(library, [_framework_record("fw-1"), _data_record("data-1")])
            result = select_reference_exemplars(paper_review={}, library=library, out=root / "out", top_k=5)
            selection = json.loads((root / "out" / "exemplar_selection.json").read_text(encoding="utf-8"))
        self.assertEqual([item["id"] for item in result["selected_positive"]], ["fw-1"])
        self.assertEqual(selection["framework_positive_count"], 1)
        self.assertEqual(selection["positive_count"], 2)
        self.assertEqual(selection["excluded_by_kind"]["data_panel"], ["data-1"])
        self.assertEqual(selection["excluded_detail"][0]["evidence"][0], "small-multiple")

    def test_a_library_of_data_figures_only_refuses_instead_of_offering_one(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "reference_library.jsonl"
            _write_library(library, [_data_record("data-1"), _data_record("data-2")])
            with self.assertRaises(NoFrameworkExemplarError) as caught:
                select_reference_exemplars(paper_review={}, library=library, out=root / "out", top_k=5)
        self.assertEqual(caught.exception.excluded["data_panel"], ["data-1", "data-2"])

    def test_the_resolution_record_names_the_kind_of_what_was_left_out(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "reference_library.jsonl"
            _write_library(library, [_data_record("data-1")])
            result = resolve_project_exemplars(
                root=root,
                paper_review={},
                selected_template={},
                reference_library=library,
                top_k=5,
                configured_library=library,
            )
            record = json.loads((root / "reference_library_resolution.json").read_text(encoding="utf-8"))
        self.assertIsNone(result)
        self.assertEqual(record["status"], "configured_no_framework_positive")
        self.assertEqual(record["excluded_by_kind"], {"data_panel": ["data-1"], "unknown": []})

    def test_a_framework_library_still_reports_how_many_it_left_out(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "reference_library.jsonl"
            _write_library(library, [_framework_record("fw-1"), _data_record("data-1"), _framework_record("fw-2")])
            result = resolve_project_exemplars(
                root=root,
                paper_review={},
                selected_template={},
                reference_library=library,
                top_k=5,
                configured_library=library,
            )
            record = json.loads((root / "reference_library_resolution.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(result)
        self.assertEqual(record["status"], "selected")
        self.assertEqual(record["positive_count"], 3)
        self.assertEqual(record["framework_positive_count"], 2)
        self.assertEqual(record["excluded_by_kind"], {"data_panel": ["data-1"], "unknown": []})


if __name__ == "__main__":
    unittest.main()
