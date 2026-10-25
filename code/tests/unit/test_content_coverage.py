"""Tests for paper-versus-figure content coverage."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evaluation.content_coverage import (
    REPORT_CONTRACT,
    measure_content_coverage,
)
from rfs.evaluation.run_supervision import judge_attempt, load_attempt_reports


def _review(*, modules: int, relations: int) -> dict:
    """Return a review whose entries are all marked as required."""
    def entry(index: int) -> dict:
        return {
            "id": f"item_{index}",
            "statement": f"item {index}",
            "visible_label": f"item {index}",
            "status": "required",
            "must_appear_in_figure": True,
        }
    return {
        "inputs": [entry(0)],
        "modules": [entry(100 + index) for index in range(modules)],
        "outputs": [entry(200)],
        "relations": [entry(300 + index) for index in range(relations)],
    }


def _graph(*, nodes: int, edges: int, bound: int) -> dict:
    return {
        "nodes": [{"id": f"vpanel.{i}", "kind": "panel"} for i in range(1)]
        + [{"id": f"vnode.{i}", "kind": "module"} for i in range(nodes)],
        "edges": [
            {"id": f"vedge.{i}", "scientific_edge_ids": ["sedge.x"] if i < bound else []}
            for i in range(edges)
        ],
    }


class ContentCoverageTests(unittest.TestCase):
    def test_a_thin_paper_reading_is_named_even_when_the_figure_matches_it(self):
        # The figure drew everything the review asked for, so the alignment report
        # would call this perfect. The coverage report has to say why it is thin.
        report = measure_content_coverage(
            paper_review=_review(modules=3, relations=4),
            visual_graph=_graph(nodes=5, edges=4, bound=4),
        )

        self.assertEqual(report["contract"], REPORT_CONTRACT)
        self.assertEqual(report["status"], "warning")
        self.assertEqual(report["ratios"]["required_node_ratio"], 1.0)
        self.assertEqual(report["ratios"]["required_edge_ratio"], 1.0)
        self.assertIn("paper_budget_thin", [item["code"] for item in report["advisories"]])

    def test_a_paper_that_read_out_enough_content_passes(self):
        report = measure_content_coverage(
            paper_review=_review(modules=8, relations=12),
            visual_graph=_graph(nodes=10, edges=12, bound=12),
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["advisories"], [])
        self.assertEqual(report["paper_budget"]["required"]["modules"], 8)

    def test_a_figure_that_draws_less_than_the_paper_declared_is_reported(self):
        report = measure_content_coverage(
            paper_review=_review(modules=8, relations=12),
            visual_graph=_graph(nodes=4, edges=5, bound=5),
        )

        codes = [item["code"] for item in report["advisories"]]
        self.assertIn("figure_below_paper_budget", codes)
        self.assertLess(report["ratios"]["declared_node_ratio"], 1.0)

    def test_arrows_that_no_relation_claims_are_reported(self):
        report = measure_content_coverage(
            paper_review=_review(modules=8, relations=12),
            visual_graph=_graph(nodes=12, edges=11, bound=1),
        )

        codes = [item["code"] for item in report["advisories"]]
        self.assertIn("arrows_unbound", codes)
        self.assertEqual(report["ratios"]["bound_edge_share"], round(1 / 11, 4))

    def test_an_attempt_without_a_review_or_a_graph_is_not_measured(self):
        report = measure_content_coverage()

        self.assertEqual(report["status"], "not_measured")
        self.assertEqual(report["advisories"], [])
        self.assertIsNone(report["ratios"]["declared_node_ratio"])

    def test_the_declared_budget_is_the_bar_and_not_the_required_subset(self):
        # The review marked 2 of its 8 modules as having to appear, and the figure drew 5 nodes
        # against the 4 that subset asks for -- so the narrower reading passes it. The paper
        # declared 10 nodes, and a figure carrying half of what the paper has is exactly what
        # this report exists to name. Counting only the must-appear entries left the denominator
        # small enough that no round of a whole run ever raised this reading.
        review = _review(modules=8, relations=6)
        for index, entry in enumerate(review["modules"]):
            if index >= 2:
                entry["must_appear_in_figure"] = False
                entry["status"] = "optional"
        report = measure_content_coverage(
            paper_review=review,
            visual_graph=_graph(nodes=5, edges=6, bound=6),
        )

        self.assertGreater(report["ratios"]["required_node_ratio"], 1.0)
        self.assertLess(report["ratios"]["declared_node_ratio"], 1.0)
        self.assertIn("figure_below_paper_budget", [item["code"] for item in report["advisories"]])
        # The report states both numbers, so "which reading fired" is a reading, not a guess.
        detail = [item["detail"] for item in report["advisories"] if item["code"] == "figure_below_paper_budget"][0]
        self.assertIn("5 nodes", detail)
        self.assertIn("10", detail)

    def test_optional_review_entries_do_not_inflate_the_budget(self):
        review = _review(modules=8, relations=12)
        review["modules"].append({"id": "optional", "status": "optional"})
        report = measure_content_coverage(
            paper_review=review,
            visual_graph=_graph(nodes=10, edges=12, bound=12),
        )

        self.assertEqual(report["paper_budget"]["declared"]["modules"], 9)
        self.assertEqual(report["paper_budget"]["required"]["modules"], 8)


class SupervisionVerdictTests(unittest.TestCase):
    def test_the_verdict_records_coverage_without_adding_a_blocking_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "paper_evolve_editable_report.json").write_text(
                json.dumps({"generation_status": "completed", "delivery_status": "available"}),
                encoding="utf-8",
            )
            for name, payload in (
                ("paper_review.json", _review(modules=3, relations=4)),
                ("visual_graph.json", _graph(nodes=5, edges=4, bound=4)),
            ):
                (root / name).write_text(json.dumps(payload), encoding="utf-8")

            verdict = judge_attempt(load_attempt_reports(root), paper_id="10.0/test")

        self.assertEqual(verdict["content_coverage"]["status"], "warning")
        self.assertEqual(verdict["metrics"]["paper_budget_modules"], 3)
        self.assertEqual(verdict["status"], "clean")
        self.assertEqual(verdict["reasons"], [])
        self.assertIn("paper_budget_thin", [item["code"] for item in verdict["advisories"]])


if __name__ == "__main__":
    unittest.main()