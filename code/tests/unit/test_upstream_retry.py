"""Tests for going back to the method extraction and the structure planning on an escalation."""

import unittest

from rfs.planning.decomposition_policy import plan_decomposition_action
from rfs.workflows.paper_evolve_editable import _retry_summary, _upstream_retry_execution
from rfs.workflows.upstream_retry import (
    UPSTREAM_RETRY_CONTRACT,
    describe_retry_feedback,
    run_upstream_retry,
    upstream_retry_kind,
)


def _method(node_id: str = "n1") -> dict:
    return {"figure_goal": "show it", "modules": [{"id": node_id, "label": node_id}]}


def _design(node_id: str = "n1") -> dict:
    return {"nodes": [{"id": node_id, "label": node_id}], "panels": [{"id": "p1"}], "connectors": []}


def _planner(method, feedback):
    return _design("n2"), {"mode": "vlm", "token_count": 10}


def _builder(design):
    return {
        "figure_layout_plan": {"nodes": [{"id": "n2", "label": "n2"}], "panels": [{"id": "p1"}], "connectors": []},
        "icon_generation_plan": {"icons": [{"asset_id": "icon_n2"}]},
        "structure_candidates": [],
    }


class UpstreamRetryKindTests(unittest.TestCase):
    def test_only_the_escalations_ask_for_a_re_run(self):
        self.assertEqual(upstream_retry_kind({"type": "resegment"}), "structure")
        self.assertEqual(upstream_retry_kind({"type": "replan_structure"}), "structure")
        self.assertEqual(upstream_retry_kind({"type": "reanalyze_semantics"}), "semantics")
        self.assertEqual(upstream_retry_kind({"type": "replan_theme"}), "theme")
        for other in ("relayout_global", "relayout_slots", "reroute", "regen_assets", "initial", ""):
            self.assertEqual(upstream_retry_kind({"type": other}), "", other)

    def test_a_block_scope_escalation_is_downgraded_so_nothing_is_re_run(self):
        policy = plan_decomposition_action(
            {
                "status": "hard_violations",
                "violations": [{"metric": "orphan_nodes", "kind": "hard", "object_ids": ["n1"], "target_node_ids": ["n1"]}],
                "repeat": {"soft_streak": 0},
            },
            evolution_scope="block",
        )
        self.assertFalse(policy["escalation"])
        self.assertEqual(upstream_retry_kind({"type": policy["action"] or "none"}), "")


class DescribeRetryFeedbackTests(unittest.TestCase):
    def test_the_note_names_the_metrics_and_the_typed_targets(self):
        policy = {
            "reason": "a hard decomposability break cannot be repaired locally",
            "violations": [
                {"metric": "orphan_nodes", "target_node_ids": ["n3"], "target_connector_ids": []},
                {"metric": "slot_escape", "target_node_ids": ["n9"], "target_connector_ids": ["e4"]},
            ],
        }
        note = describe_retry_feedback(policy)
        self.assertIn("orphan_nodes", note)
        self.assertIn("slot_escape", note)
        self.assertIn("n3", note)
        self.assertIn("n9", note)
        self.assertIn("e4", note)
        self.assertIn("never add an object the paper does not state", note)

    def test_an_empty_policy_still_asks_for_the_same_kind_of_re_plan(self):
        note = describe_retry_feedback(None)
        self.assertTrue(note.strip())
        self.assertIn("decomposability break", note)

    def test_the_note_says_which_way_the_measured_objects_have_to_move(self):
        # Naming the objects and stopping is a restatement of the prompt the first attempt ran, so
        # the second attempt had no reason to come out different. The direction is what makes it a
        # different question, and it is read off the metric the measurement named.
        note = describe_retry_feedback(
            {
                "violations": [
                    {"metric": "cross_panel_connectors", "target_connector_ids": ["e1", "e2"]},
                    {"metric": "slot_boundary_overflows", "target_node_ids": ["n7"]},
                ]
            }
        )
        self.assertIn("e1", note)
        self.assertIn("n7", note)
        self.assertIn("side that faces its target", note)
        self.assertIn("wholly inside the panel it belongs to", note)

    def test_the_note_says_what_the_re_run_will_be_measured_against(self):
        note = describe_retry_feedback({"violations": [{"metric": "orphan_nodes", "target_node_ids": ["n3"]}]})
        self.assertIn("measured against the same numbers: orphan_nodes", note)

    def test_a_break_the_direction_table_does_not_know_still_gets_a_direction(self):
        note = describe_retry_feedback({"violations": [{"metric": "some_new_break", "target_node_ids": ["n1"]}]})
        self.assertIn("separable from their neighbours", note)


class RunUpstreamRetryTests(unittest.TestCase):
    def test_a_structure_escalation_re_plans_and_reports_a_changed_state(self):
        report = run_upstream_retry(
            {"type": "resegment"},
            planner_mode="vlm",
            feedback="separate the objects",
            method=_method(),
            design=_design(),
            structure_planner=_planner,
            structure_builder=_builder,
        )
        self.assertEqual(report["contract"], UPSTREAM_RETRY_CONTRACT)
        self.assertEqual(report["kind"], "structure")
        self.assertTrue(report["attempted"])
        self.assertTrue(report["applied"])
        self.assertTrue(report["changed"])
        self.assertNotEqual(report["before_hash"], report["after_hash"])
        self.assertTrue(report["structure"]["figure_layout_plan"]["nodes"])

    def test_a_semantic_escalation_re_extracts_the_method_before_it_plans(self):
        seen: list[str] = []

        def extractor(note):
            seen.append("extract")
            return {"figure_goal": "again", "modules": [{"id": "n5"}]}, {"mode": "vlm"}

        def planner(method, note):
            seen.append(f"plan:{method['figure_goal']}")
            return _design("n5"), {}

        report = run_upstream_retry(
            {"type": "reanalyze_semantics"},
            planner_mode="vlm",
            feedback="re-decide ownership",
            method=_method(),
            design=_design(),
            structure_planner=planner,
            structure_builder=_builder,
            method_extractor=extractor,
        )
        self.assertEqual(seen, ["extract", "plan:again"])
        self.assertTrue(report["applied"])
        self.assertEqual(report["method"]["figure_goal"], "again")

    def test_a_non_escalation_action_is_declined_with_its_reason(self):
        report = run_upstream_retry({"type": "relayout_global"}, planner_mode="vlm")
        self.assertFalse(report["attempted"])
        self.assertFalse(report["applied"])
        self.assertIn("does not ask", report["declined_reason"])

    def test_a_heuristic_planner_declines_and_names_the_mode(self):
        report = run_upstream_retry({"type": "resegment"}, planner_mode="heuristic", structure_planner=_planner, structure_builder=_builder)
        self.assertFalse(report["attempted"])
        self.assertEqual(report["declined_reason"], "planner_mode_heuristic")

    def test_the_switch_declines_without_losing_the_kind(self):
        report = run_upstream_retry({"type": "resegment"}, enabled=False, planner_mode="vlm", structure_planner=_planner, structure_builder=_builder)
        self.assertEqual(report["kind"], "structure")
        self.assertFalse(report["attempted"])
        self.assertIn("switched off", report["declined_reason"])

    def test_a_failing_provider_is_recorded_rather_than_raised(self):
        def boom(*_args, **_kwargs):
            raise RuntimeError("429 rate limited")

        report = run_upstream_retry(
            {"type": "replan_structure"},
            planner_mode="vlm",
            method=_method(),
            design=_design(),
            structure_planner=boom,
            structure_builder=_builder,
        )
        self.assertTrue(report["attempted"])
        self.assertFalse(report["applied"])
        self.assertEqual(report["declined_reason"], "provider_error")
        self.assertIn("429 rate limited", report["provider_error"])

    def test_a_re_plan_with_no_layout_is_declined_after_the_call(self):
        report = run_upstream_retry(
            {"type": "resegment"},
            planner_mode="vlm",
            method=_method(),
            design=_design(),
            structure_planner=_planner,
            structure_builder=lambda _design: {"figure_layout_plan": {}, "icon_generation_plan": {}},
        )
        self.assertTrue(report["attempted"])
        self.assertFalse(report["applied"])
        self.assertEqual(report["declined_reason"], "the re-plan produced no layout")

    def test_a_re_plan_that_changes_nothing_is_marked_unchanged(self):
        same = _design()
        report = run_upstream_retry(
            {"type": "resegment"},
            planner_mode="vlm",
            method=_method(),
            design=same,
            structure_planner=lambda method, note: (dict(same), {}),
            structure_builder=_builder,
        )
        self.assertTrue(report["applied"])
        self.assertFalse(report["changed"])


class RetryExecutionReportTests(unittest.TestCase):
    def _state(self, node_id: str) -> dict:
        return {"method": _method(), "layout": {"nodes": [{"id": node_id}]}, "icons": {"icons": [{"asset_id": "icon_n1"}]}}

    def test_a_re_run_that_moves_a_node_is_recorded_as_applied(self):
        report, before, after = _upstream_retry_execution(
            self._state("n1"), self._state("n2"), {"type": "resegment"}, round_index=1, retry_index=0, kind="structure"
        )
        self.assertTrue(report["applied"])
        self.assertTrue(report["geometric_applied"])
        self.assertEqual(report["changed_node_ids"], ["n1", "n2"])
        self.assertEqual(report["upstream_retry"], "structure")
        self.assertTrue(before["state_hash"])
        self.assertTrue(after["state_hash"])

    def test_a_re_run_that_moved_nothing_is_recorded_as_idle(self):
        report, _before, _after = _upstream_retry_execution(
            self._state("n1"), self._state("n1"), {"type": "resegment"}, round_index=1, retry_index=0, kind="structure"
        )
        self.assertFalse(report["applied"])
        self.assertFalse(report["geometric_applied"])
        self.assertEqual(report["changed_node_ids"], [])

    def test_the_summary_drops_the_contracts_the_re_run_produced(self):
        report = {"kind": "structure", "applied": True, "changed": True, "method": {"a": 1}, "design": {"b": 2}, "structure": {"c": 3}, "method_report": {"d": 4}, "structure_report": {"e": 5}}
        summary = _retry_summary(report)
        self.assertEqual(summary, {"kind": "structure", "applied": True, "changed": True})


if __name__ == "__main__":
    unittest.main()
