"""Tests for the continuous quality signal behind the quantized gate score."""

import unittest

from rfs.evaluation.interactive_judgement import deterministic_dimension_scores
from rfs.evolution.direct.controller import compare_candidate


class DeterministicDimensionScoreTests(unittest.TestCase):
    def test_clean_geometry_and_full_density_score_one(self):
        scores = deterministic_dimension_scores(
            {key: 0 for key in ("icon_text_overlap_count", "node_overlap_count", "hierarchy_containment_count", "connector_label_overlap_count", "text_pass_count", "box_pass_count", "node_boundary_violation_count")},
            {"canvas_occupancy": 0.5, "panel_count": 2, "sparse_panel_count": 0, "editable_text_area": 0.2, "targets": {"minimum_occupancy": 0.45}},
        )
        self.assertEqual(scores["geometry_cleanliness"], 1.0)
        self.assertEqual(scores["density_fitness"], 1.0)
        self.assertEqual(scores["panel_occupancy"], 1.0)

    def test_violations_degrade_smoothly_rather_than_as_a_single_step(self):
        one = deterministic_dimension_scores({"box_pass_count": 1}, {})
        three = deterministic_dimension_scores({"box_pass_count": 3}, {})
        self.assertEqual(one["geometry_cleanliness"], 0.5)
        self.assertEqual(three["geometry_cleanliness"], 0.25)
        self.assertLess(three["geometry_cleanliness"], one["geometry_cleanliness"])

    def test_children_nested_in_their_own_container_are_not_a_violation(self):
        # 10.1371/journal.pcbi.1014000 compiled one real violation and two
        # children sitting inside their own container; counting the nesting made
        # its cleanliness 0.25 instead of 0.5, so nesting a child properly would
        # have read as a regression to a repair that compares this dimension.
        nested = deterministic_dimension_scores({"node_boundary_violation_count": 1, "hierarchy_containment_count": 2}, {})
        clean = deterministic_dimension_scores({"node_boundary_violation_count": 1}, {})
        self.assertEqual(nested["geometry_cleanliness"], 0.5)
        self.assertEqual(nested["geometry_cleanliness"], clean["geometry_cleanliness"])

    def test_partial_occupancy_is_reported_as_a_fraction_of_the_target(self):
        scores = deterministic_dimension_scores({}, {"canvas_occupancy": 0.18, "targets": {"minimum_occupancy": 0.45}})
        self.assertEqual(scores["density_fitness"], 0.4)

    def test_sparse_panels_reduce_panel_occupancy_proportionally(self):
        scores = deterministic_dimension_scores({}, {"panel_count": 4, "sparse_panel_count": 1})
        self.assertEqual(scores["panel_occupancy"], 0.75)


class TargetMetricComparisonTests(unittest.TestCase):
    def test_default_comparison_still_reads_the_overall_score(self):
        decision = compare_candidate({"overall_score": 0.75}, {"overall_score": 0.75, "dimension_scores": {"density_fitness": 0.9}})
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["target_metric"], "overall_score")
        self.assertEqual(decision["reason"], "no_significant_improvement")

    def test_a_continuous_target_metric_sees_improvement_below_the_score_quantum(self):
        decision = compare_candidate(
            {"overall_score": 0.75, "dimension_scores": {"density_fitness": 0.40}},
            {"overall_score": 0.75, "dimension_scores": {"density_fitness": 0.46}},
            target_metric="density_fitness",
        )
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["score_delta"], 0.06)
        self.assertEqual(decision["before_target"], 0.4)
        self.assertEqual(decision["after_target"], 0.46)

    def test_a_missing_target_metric_falls_back_to_the_overall_score(self):
        decision = compare_candidate({"overall_score": 0.5}, {"overall_score": 0.9}, target_metric="not_measured")
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["score_delta"], 0.4)


if __name__ == "__main__":
    unittest.main()
