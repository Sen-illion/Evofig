"""The layer table prices a repair, and a round writes down what it actually spent."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import action_class, action_class_policy, action_reuses_assets, content_gap_signal, measurement_request
from rfs.evolution.direct.action_policy import ACTION_CLASSES, STAGE_RERUN_CLASS
from rfs.workflows.run_cost import (
    ROUND_COST_FILE,
    round_cost_record,
    stage_seconds_between,
    stage_seconds_from_progress,
    summarize_round_costs,
    write_round_cost,
)
from rfs.workflows.upstream_retry import UPSTREAM_RETRY_KINDS


class ActionClassTests(unittest.TestCase):
    def test_every_bounded_action_is_placed_on_a_layer(self):
        for name in ("reroute", "resize_nodes", "relayout_slots", "repair_interfaces", "relayout_global"):
            self.assertEqual("figure_program", action_class(name))
        for name in ("router_constraint", "layout_constraint"):
            self.assertEqual("compile_rule", action_class(name))
        for name in ("resegment", "replan_structure", "reanalyze_semantics", "replan_theme"):
            self.assertEqual("stage_rerun", action_class(name))

    def test_the_expensive_layer_is_exactly_what_re_runs_a_stage(self):
        # The gate here and the re-run in the workflow must not disagree about which actions
        # are expensive, so they read the same set.
        self.assertEqual(set(UPSTREAM_RETRY_KINDS), set(ACTION_CLASSES[STAGE_RERUN_CLASS]))

    def test_a_name_the_loop_does_not_run_has_no_layer(self):
        self.assertEqual("", action_class("not_an_action"))
        self.assertEqual("", action_class(None))

    def test_a_move_at_a_cheap_layer_must_leave_the_paid_icons_alone(self):
        for name in ("reroute", "repair_interfaces", "relayout_slots", "resize_nodes", "relayout_global", "router_constraint", "layout_constraint"):
            self.assertTrue(action_reuses_assets(name), name)

    def test_a_stage_rerun_may_ask_for_art_again(self):
        for name in ("resegment", "replan_structure", "reanalyze_semantics", "replan_theme"):
            self.assertFalse(action_reuses_assets(name), name)

    def test_a_name_the_loop_does_not_run_is_not_forced_to_reuse(self):
        # The guard is a shield for the moves the loop understands, not a guess about the rest.
        self.assertFalse(action_reuses_assets("regen_assets"))
        self.assertFalse(action_reuses_assets("not_an_action"))


class ActionClassPolicyTests(unittest.TestCase):
    """The layer table prices a repair and records who asked for it; it refuses nothing.

    It used to be a gate. A fault the coach placed at the compiled figure could only be answered
    by the cheap layers, and the expensive one had to be earned -- by an attribution that pointed
    upstream, by two rounds in a row that moved nothing, or by one of the loop's own readings.
    Which layer a repair works at is a fact about the repair, and how much to spend is the seat's
    decision: the prices travel to the seat, and the round is bounded by its budget and its clock.
    """

    def test_every_layer_is_open_however_the_coach_placed_the_fault(self):
        for stage in ("ppt_compilation", "global_planning", "paper_reading", ""):
            with self.subTest(stage=stage):
                policy = action_class_policy(coach_fault_stage=stage)
                self.assertEqual(["figure_program", "compile_rule", "stage_rerun"], policy["allowed"])
                self.assertFalse(policy["escalated"])
                self.assertEqual("", policy["signal"])

    def test_the_table_states_what_each_layer_spends(self):
        costs = action_class_policy()["costs"]
        self.assertEqual(set(costs), {"figure_program", "compile_rule", "stage_rerun"})
        self.assertIn("no model call", costs["figure_program"])
        self.assertIn("no model call", costs["compile_rule"])
        self.assertIn("minutes to hours", costs["stage_rerun"])

    def test_one_stalled_round_is_not_yet_a_reading(self):
        policy = action_class_policy(coach_fault_stage="ppt_compilation", stalled_rounds=1)
        self.assertFalse(policy["escalated"])
        self.assertEqual("", policy["signal"])
        self.assertIn("the coach placed the fault at ppt_compilation", policy["reason"])

    def test_two_stalled_rounds_are_recorded_as_a_voice_for_the_expensive_layer(self):
        policy = action_class_policy(coach_fault_stage="ppt_compilation", stalled_rounds=2)
        self.assertTrue(policy["escalated"])
        self.assertEqual("stalled_rounds", policy["signal"])
        # It is the loop's count of its own rounds rather than a measurement of the figure.
        self.assertFalse(policy["measured"])
        self.assertIn("2 rounds in a row moved nothing", policy["reason"])
        # The attribution stays in the account, and it still closes nothing.
        self.assertIn("ppt_compilation", policy["reason"])
        self.assertEqual(["figure_program", "compile_rule", "stage_rerun"], policy["allowed"])

    def test_the_loops_own_measurement_travels_as_a_measured_voice(self):
        policy = action_class_policy(coach_fault_stage="ppt_compilation", measured_escalation=True)
        self.assertTrue(policy["escalated"])
        self.assertTrue(policy["measured"])
        self.assertEqual("decomposition_policy", policy["signal"])
        self.assertIn("cannot be read in pieces", policy["reason"])

    def test_a_figure_below_the_papers_budget_is_recorded_as_measured_evidence(self):
        policy = action_class_policy(coach_fault_stage="ppt_compilation", content_gap="figure_below_paper_budget")
        self.assertTrue(policy["escalated"])
        self.assertTrue(policy["measured"])
        self.assertEqual("figure_below_paper_budget", policy["signal"])
        self.assertIn("drew less than the paper", policy["reason"])

    def test_a_thin_paper_reads_as_a_fact_about_the_paper(self):
        policy = action_class_policy(content_gap="paper_budget_thin")
        self.assertTrue(policy["measured"])
        self.assertEqual("paper_budget_thin", policy["signal"])
        self.assertIn("read again", policy["reason"])

    def test_the_coach_attribution_is_a_voice_without_a_measurement(self):
        upstream = action_class_policy(coach_fault_stage="global_planning")
        self.assertIn("the coach placed the fault at global_planning", upstream["reason"])
        self.assertEqual("", upstream["signal"])
        self.assertFalse(upstream["measured"])
        self.assertFalse(upstream["escalated"])

    def test_the_coverage_report_is_read_for_the_signal_it_carries(self):
        self.assertEqual("figure_below_paper_budget", content_gap_signal({"advisories": [{"code": "figure_below_paper_budget"}]}))
        self.assertEqual("paper_budget_thin", content_gap_signal({"advisories": [{"code": "paper_budget_thin"}]}))
        # A figure below the paper's budget is the reading a re-plan can act on, so it is the one
        # reported when both are present: the thin source is a fact about the paper, not the figure.
        self.assertEqual(
            "figure_below_paper_budget",
            content_gap_signal({"advisories": [{"code": "paper_budget_thin"}, {"code": "figure_below_paper_budget"}]}),
        )
        # A reading that is about neither -- arrows the paper never claimed -- unlocks nothing.
        self.assertEqual("", content_gap_signal({"advisories": [{"code": "arrows_unbound"}]}))
        # A report that was never measured is an absent reading, not a clean bill of health.
        self.assertEqual("", content_gap_signal({"status": "not_measured"}))
        self.assertEqual("", content_gap_signal(None))

    def test_a_move_the_loops_own_measurement_asked_for(self):
        self.assertTrue(measurement_request({"type": "resegment", "source": "decomposition_policy"}))
        self.assertFalse(measurement_request({"type": "resegment"}))
        self.assertFalse(measurement_request("resegment"))
        self.assertFalse(measurement_request(None))


class RoundCostTests(unittest.TestCase):
    def test_a_round_that_only_edited_the_program_spent_no_generation_call(self):
        record = round_cost_record(
            round_index=3,
            action={"type": "reroute"},
            action_class="figure_program",
            candidate_count=2,
            compile_calls=2,
            render_calls=2,
            judge_calls=5,
        )
        self.assertEqual("rfs.round_cost", record["contract"])
        self.assertEqual(0, record["figure_generations"])
        self.assertEqual(0, record["stage_reruns"])
        self.assertEqual("0 generation call(s), 2 compile(s), 5 judge answer(s), 0 stage re-run(s)", record["summary"])

    def test_a_stage_re_run_is_what_a_generation_call_is(self):
        record = round_cost_record(
            round_index=1,
            action={"type": "replan_structure"},
            action_class="stage_rerun",
            candidate_count=1,
            figure_generations=1,
            stage_reruns=1,
            compile_calls=1,
        )
        self.assertEqual(1, record["figure_generations"])
        self.assertIn("1 generation call(s)", record["summary"])

    def test_the_round_writes_its_ledger_beside_its_own_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            round_dir = Path(temp) / "round_00"
            round_dir.mkdir()
            record = write_round_cost(round_dir, round_index=0, candidate_count=1, compile_calls=1)
            written = json.loads((round_dir / ROUND_COST_FILE).read_text(encoding="utf-8"))
            self.assertEqual(record, written)

    def test_the_run_summary_adds_the_rounds_up(self):
        summary = summarize_round_costs(
            [
                round_cost_record(round_index=0, candidate_count=1, compile_calls=1, judge_calls=2, elapsed_seconds=10.0),
                round_cost_record(
                    round_index=1,
                    candidate_count=2,
                    compile_calls=2,
                    judge_calls=5,
                    stage_reruns=1,
                    figure_generations=1,
                    elapsed_seconds=30.5,
                ),
            ]
        )
        self.assertEqual(2, summary["rounds"])
        self.assertEqual(3, summary["totals"]["compile_calls"])
        self.assertEqual(7, summary["totals"]["judge_calls"])
        self.assertEqual(1, summary["totals"]["stage_reruns"])
        self.assertEqual(40.5, summary["elapsed_seconds"])
        self.assertIn("2 round(s)", summary["summary"])

    def test_the_stage_clock_is_read_back_from_the_runs_own_progress_stream(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [
                {"stage": "paper_preparation", "status": "started", "elapsed_seconds": 0.0},
                {"stage": "paper_preparation", "status": "completed", "elapsed_seconds": 4.0},
                {"stage": "icons", "status": "started", "elapsed_seconds": 4.0},
                {"stage": "icons", "status": "completed", "elapsed_seconds": 6.5},
                {"stage": "round", "status": "started", "elapsed_seconds": 6.5},
            ]
            (root / "progress.jsonl").write_text("\n".join(json.dumps(item) for item in events), encoding="utf-8")
            seconds = stage_seconds_from_progress(root)
            self.assertEqual({"paper_preparation": 4.0, "icons": 2.5}, seconds)
            # A stage still open when the reading was taken contributes nothing yet.
            self.assertNotIn("round", seconds)
            # And a stage that did not advance during the round is not reported as having cost it.
            self.assertEqual(
                {"icons": 1.5},
                stage_seconds_between({"icons": 1.0, "paper_preparation": 4.0}, {"icons": 2.5, "paper_preparation": 4.0}),
            )


if __name__ == "__main__":
    unittest.main()
