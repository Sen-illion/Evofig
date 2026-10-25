"""The five numbers that tell self-evolution from iterative repair."""

from __future__ import annotations

import unittest

from rfs.evolution.direct.metrics import (
    advice_improvement,
    claim_settlement_rate,
    evolution_metrics,
    location_accuracy,
    metric_direction,
    portable_pass_rate,
    recurrence,
    round_metrics,
)


def _round(**overrides):
    record = {"fault_stage": "ppt_compilation", "round_metrics": {}, "accepted": True, "repair_action": {}, "repair_decision": {}}
    record.update(overrides)
    return record


class LocationAccuracyTests(unittest.TestCase):
    def test_a_claim_the_next_reading_bears_out_counts_as_borne_out(self):
        previous = _round(fault_evidence=[{"field": "stage_trace.metrics.crossing_count", "value": 9}], round_metrics={"crossing_count": 9})
        following = _round(fault_stage="", round_metrics={"crossing_count": 2})

        result = location_accuracy([previous, following])

        self.assertEqual((result["checked"], result["borne_out"]), (1, 1))
        self.assertEqual(result["rate"], 1.0)

    def test_a_claim_about_a_number_the_next_round_did_not_measure_is_not_counted(self):
        previous = _round(fault_evidence=[{"field": "crossing_count"}], round_metrics={"crossing_count": 9})
        following = _round(fault_stage="", round_metrics={"node_occupancy": 0.9})

        result = location_accuracy([previous, following])

        self.assertEqual(result["checked"], 0)
        self.assertIsNone(result["rate"])

    def test_a_metric_with_no_known_direction_is_not_scored(self):
        previous = _round(fault_evidence=[{"field": "node_count"}], round_metrics={"node_count": 12})
        following = _round(fault_stage="", round_metrics={"node_count": 4})

        self.assertEqual(location_accuracy([previous, following])["checked"], 0)
        self.assertEqual(metric_direction("node_count"), "")

    def test_the_arrow_reports_obstacle_total_is_a_down_is_better_reading(self):
        self.assertEqual(metric_direction("route_obstacle_overlap_count"), "down")

    def test_a_claim_that_worsened_is_checked_and_not_borne_out(self):
        previous = _round(fault_evidence=[{"field": "crossing_count"}], round_metrics={"crossing_count": 2})
        following = _round(fault_stage="", round_metrics={"crossing_count": 5})

        result = location_accuracy([previous, following])

        self.assertEqual((result["checked"], result["borne_out"]), (1, 0))


class AdviceImprovementTests(unittest.TestCase):
    def test_a_model_move_that_measured_better_counts(self):
        rounds = [_round(repair_action={"type": "reroute", "source": ""}, repair_decision={"reason": "pairwise_preferred"})]

        self.assertEqual(advice_improvement(rounds)["rate"], 1.0)

    def test_the_mechanisms_own_default_is_not_the_models_advice(self):
        rounds = [
            _round(repair_action={"type": "resegment", "source": "decomposition_policy"}, repair_decision={"reason": "pairwise_preferred"}),
            _round(repair_action={"type": "reroute", "source": ""}, repair_decision={"reason": "pairwise_not_preferred"}),
        ]

        result = advice_improvement(rounds, mechanism_sources={"decomposition_policy"})

        self.assertEqual((result["offered"], result["improved"]), (1, 0))

    def test_accepted_without_an_improvement_reason_is_not_an_improvement(self):
        rounds = [_round(repair_action={"type": "reroute"}, repair_decision={"reason": "initial_candidate"})]

        self.assertEqual(advice_improvement(rounds)["improved"], 0)


class PortabilityTests(unittest.TestCase):
    def test_only_decided_rows_are_in_the_denominator(self):
        rows = [
            {"status": "promoted"},
            {"status": "cross_domain_validated"},
            {"status": "overfit_or_failed"},
            {"status": "candidate"},
            {"status": "measurement_noise"},
        ]

        result = portable_pass_rate(rows)

        self.assertEqual((result["decided"], result["portable"], result["failed"], result["cross_domain"]), (3, 2, 1, 2))
        self.assertEqual(result["rate"], round(2 / 3, 4))


class ClaimSettlementTests(unittest.TestCase):
    def test_the_rate_is_read_from_the_ledger_counts(self):
        report = {"counts": {"registered": 4, "settled": 3, "confirmed": 1, "refuted": 2, "open": 1}}

        result = claim_settlement_rate(report)

        self.assertEqual(result["rate"], 0.75)
        self.assertEqual(result["open"], 1)

    def test_a_ledger_with_nothing_registered_reports_no_rate(self):
        self.assertIsNone(claim_settlement_rate({"counts": {}})["rate"])


class RecurrenceTests(unittest.TestCase):
    def test_a_defect_that_comes_back_smaller_counts(self):
        rounds = [_round(round_metrics={"crossing_count": 5}), _round(round_metrics={"crossing_count": 2})]

        result = recurrence(rounds)

        self.assertEqual((result["reappearances"], result["held_or_better"]), (1, 1))

    def test_a_defect_that_comes_back_bigger_does_not(self):
        rounds = [_round(round_metrics={"crossing_count": 2}), _round(round_metrics={"crossing_count": 5})]

        self.assertEqual(recurrence(rounds)["held_or_better"], 0)


class EvolutionMetricsTests(unittest.TestCase):
    def test_the_headline_names_all_five_and_leaves_an_empty_denominator_null(self):
        report = evolution_metrics(rounds=[], library_rows=[], claim_report={"counts": {}})

        self.assertEqual(
            sorted(report["headline"]),
            ["advice_improvement_rate", "claim_settlement_rate", "location_accuracy", "portable_pass_rate", "recurrence_rate"],
        )
        self.assertTrue(all(value is None for value in report["headline"].values()))


class CrossRoundCounterTests(unittest.TestCase):
    """The trend counts the verdict the seat gave, in the seat's own words.

    The seat answers "which of the two did you prefer", so its word is the version it chose
    (``current_better`` / ``history_better``) and not a direction the counter has to infer. A
    counter that only knew ``better``/``worse`` read every comparison the loop made as one nobody
    had decided, which is the one reading the trend exists to avoid.
    """

    def _counts(self, verdict: str) -> dict:
        return round_metrics(paper="paper_one", iteration=3, cross_round=[{"verdict": verdict}])["cross_round"]

    def test_a_round_that_beat_the_papers_best_counts_as_a_win(self):
        self.assertEqual(
            {"comparisons": 1, "wins": 1, "losses": 0, "ties": 0, "undecided": 0},
            self._counts("current_better"),
        )

    def test_a_round_that_came_out_worse_than_the_best_counts_as_a_loss(self):
        counts = self._counts("history_better")
        self.assertEqual(1, counts["losses"])
        self.assertEqual(0, counts["wins"])

    def test_a_tie_is_a_tie(self):
        self.assertEqual(1, self._counts("tie")["ties"])

    def test_a_verdict_nobody_can_place_is_undecided_rather_than_a_tie(self):
        self.assertEqual(
            {"comparisons": 1, "wins": 0, "losses": 0, "ties": 0, "undecided": 1},
            self._counts("not_compared"),
        )

    def test_a_round_that_compared_nothing_is_not_counted_as_a_comparison(self):
        self.assertEqual(0, round_metrics(paper="paper_one", iteration=1, cross_round=[])["cross_round"]["comparisons"])


if __name__ == "__main__":
    unittest.main()
