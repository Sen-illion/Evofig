"""A stage trace records what each stage changed and the numbers the figure moved."""

from __future__ import annotations

import unittest

from rfs.contracts.evolution import STAGE_ORDER
from rfs.evaluation.direct_judges import DirectJudgeInput
from rfs.evaluation.judge_prompt import online_judge_row
from rfs.workflows.stage_trace import (
    ACTION_TRACE_STAGE,
    METRIC_KEYS,
    STAGE_GATE_CONTRACT,
    STAGE_GATE_FILE,
    STAGE_GATE_VERSION,
    STAGE_TRACE_CONTRACT,
    STAGE_TRACE_VERSION,
    TRACE_STAGES,
    action_trace_stage,
    attribution_verdict,
    build_stage_trace,
    compact_stage_trace,
    figure_metrics,
    stage_gate_report,
)


def _node(node_id, panel_id="panel_a"):
    return {
        "id": node_id,
        "panel_id": panel_id,
        "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
        "label": node_id,
    }


def _program(nodes, *, arrows=(), panels=("panel_a",)):
    return {
        "panels": [
            {"id": panel, "bbox_percent": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5}}
            for panel in panels
        ],
        "semantic_nodes": [_node(node) for node in nodes],
        "slots": [],
        "arrows": [dict(arrow) for arrow in arrows],
    }


def _crossing_arrows():
    return [
        {"id": "a1", "path_percent": [[0.1, 0.1], [0.9, 0.9]]},
        {"id": "a2", "path_percent": [[0.1, 0.9], [0.9, 0.1]]},
    ]


def _judge_input(**overrides):
    base = {"method": {}, "layout": {}, "icons": {}, "program": {}}
    base.update(overrides)
    return DirectJudgeInput(**base)


class StageTraceTests(unittest.TestCase):
    def test_the_vocabulary_is_the_contract_stage_order(self):
        self.assertEqual(TRACE_STAGES, STAGE_ORDER)
        self.assertEqual(ACTION_TRACE_STAGE["reroute"], "ppt_compilation")
        self.assertEqual(ACTION_TRACE_STAGE["replan_structure"], "global_planning")

    def test_the_vocabulary_names_where_a_content_defect_entered(self):
        from rfs.judge_training.prompts import ADVISORY_OUTPUT_SCHEMA

        # Both entries lead the order, because a defect that entered at either has no answer
        # below it: nothing downstream can put back a module the reading never extracted.
        self.assertEqual(STAGE_ORDER[:2], ("paper_reading", "content_planning"))
        menu = ADVISORY_OUTPUT_SCHEMA["fault_stage"]
        for stage in ("paper_reading", "content_planning"):
            self.assertIn(stage, menu)

    def test_a_round_carries_a_row_for_every_stage_it_can_be_asked_about(self):
        trace = build_stage_trace(round_index=0, before_program=_program(["a"]), after_program=_program(["a"]))
        self.assertEqual([item["stage"] for item in trace["stages"]], list(STAGE_ORDER))
        # A round that touched neither still reports both, so a coach that names the reading is
        # read against the trace instead of refused as a stage that never ran.
        verdict = attribution_verdict(
            {
                "fault_stage": "content_planning",
                "fault_evidence": [{"stage": "content_planning", "field": "node_count", "value": "1"}],
            },
            trace,
            shown={},
        )
        self.assertTrue(verdict["ok"], verdict)

    def test_metrics_are_read_from_the_program_alone(self):
        metrics = figure_metrics(_program(["a", "b", "c"]))
        self.assertEqual(metrics["node_count"], 3)
        self.assertEqual(metrics["panel_count"], 1)
        self.assertEqual(set(metrics), set(METRIC_KEYS))
        crossing = figure_metrics(_program(["a"], arrows=_crossing_arrows()))
        self.assertEqual(crossing["connector_count"], 2)
        self.assertEqual(crossing["crossing_count"], 1)

    def test_a_round_that_changed_nothing_compacts_to_nothing(self):
        trace = build_stage_trace(round_index=0, before_program=_program(["a"]), after_program=_program(["a"]))
        self.assertEqual(trace["contract"], STAGE_TRACE_CONTRACT)
        self.assertEqual(trace["version"], STAGE_TRACE_VERSION)
        self.assertEqual(trace["changed_stages"], [])
        self.assertEqual(compact_stage_trace(trace), {})

    def test_a_replanned_stage_and_its_action_are_named_as_the_stages_that_changed(self):
        reuse = {
            "stage_outcomes": {
                "color_theme": {"status": "replanned", "reason": "palette drift"},
                "icons": {"status": "reused"},
            }
        }
        trace = build_stage_trace(
            round_index=2,
            before_program=_program(["a"]),
            after_program=_program(["a", "b"]),
            reuse_report=reuse,
            action={"type": "reroute", "edge_ids": ["e1"]},
        )
        self.assertIn("theme_planning", trace["changed_stages"])
        self.assertIn("ppt_compilation", trace["changed_stages"])
        self.assertNotIn("icon_generation", trace["changed_stages"])
        self.assertEqual(trace["metrics"]["delta"]["node_count"], 1)
        compact = compact_stage_trace(trace)
        self.assertEqual(compact["changed_stages"], trace["changed_stages"])
        self.assertEqual(compact["metric_delta"]["node_count"], 1.0)
        self.assertEqual(compact["action"], {"type": "reroute", "apply_at": "ppt_compilation"})

    def test_an_action_that_applies_nowhere_is_a_silent_stage(self):
        self.assertEqual(action_trace_stage({"type": "unknown_action"}), "")
        self.assertEqual(action_trace_stage(None), "")

    def test_the_measured_fields_travel_so_the_coach_can_cite_one(self):
        """A field the round did not move is the evidence for a defect it did not fix."""

        trace = build_stage_trace(
            round_index=1,
            before_program=_program(["a"]),
            after_program=_program(["a", "b"]),
            action={"type": "resegment"},
        )
        compact = compact_stage_trace(trace)
        self.assertEqual(set(compact["metrics"]), set(METRIC_KEYS))
        self.assertEqual(compact["metrics"]["node_count"], 2)
        # panel_occupancy did not change, so it is absent from the deltas but still measurable:
        # an attribution that cites it is readable, which is the whole point of carrying it.
        self.assertNotIn("panel_occupancy", compact["metric_delta"])
        verdict = attribution_verdict(
            {
                "fault_stage": "segmentation",
                "fault_evidence": [
                    {"stage": "segmentation", "field": "panel_occupancy", "value": compact["metrics"]["panel_occupancy"], "bound": 0.45}
                ],
            },
            trace,
        )
        self.assertTrue(verdict["ok"], verdict)

    def test_a_field_the_trace_never_measured_is_still_refused(self):
        trace = build_stage_trace(
            round_index=1,
            before_program=_program(["a"]),
            after_program=_program(["a", "b"]),
            action={"type": "resegment"},
        )
        verdict = attribution_verdict(
            {"fault_stage": "segmentation", "fault_evidence": [{"stage": "segmentation", "field": "node_coverage", "value": "0.86"}]},
            trace,
        )
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "fault_evidence_unreadable")


class StageTracePromptTests(unittest.TestCase):
    def test_the_advisory_row_carries_the_trace_and_a_measurement_row_never_does(self):
        trace = {"round": 1, "changed_stages": ["ppt_compilation"], "metric_delta": {"crossing_count": 3}}
        advisory = online_judge_row(_judge_input(stage_trace=trace), role="online_bootstrap")
        self.assertEqual(advisory["process_context"]["stage_trace"], trace)
        acceptance = online_judge_row(_judge_input(stage_trace=trace), role="frozen_bootstrap")
        self.assertNotIn("stage_trace", acceptance["process_context"])

    def test_a_round_that_changed_nothing_adds_no_prompt_key(self):
        row = online_judge_row(_judge_input(stage_trace={}), role="online_bootstrap")
        self.assertNotIn("stage_trace", row["process_context"])

    def test_a_pairwise_half_never_carries_the_trace(self):
        row = online_judge_row(_judge_input(stage_trace={"changed_stages": ["ppt_compilation"]}), role="online_bootstrap", pairwise=True)
        self.assertNotIn("stage_trace", row["process_context"])

    def test_the_layers_that_are_still_open_travel_to_the_advisory_seat_alone(self):
        layers = {
            "allowed": ["figure_program", "compile_rule"],
            "reason": "the coach placed the fault at ppt_compilation, which the cheap layers answer",
            "deferred": {"action": "replan_structure", "reason": "the coach placed the fault at ppt_compilation"},
        }
        advisory = online_judge_row(_judge_input(layer_costs=layers), role="online_bootstrap")
        self.assertEqual(advisory["process_context"]["layer_costs"], layers)
        # A measurement role asking for a repair is not measuring, and a comparison half is one
        # figure: neither is ever told which layers are open.
        acceptance = online_judge_row(_judge_input(layer_costs=layers), role="frozen_bootstrap")
        self.assertNotIn("layer_costs", acceptance["process_context"])
        pairwise = online_judge_row(_judge_input(layer_costs=layers), role="online_bootstrap", pairwise=True)
        self.assertNotIn("layer_costs", pairwise["process_context"])

    def test_a_round_with_every_layer_open_adds_no_prompt_key(self):
        row = online_judge_row(_judge_input(layer_costs={}), role="online_bootstrap")
        self.assertNotIn("layer_costs", row["process_context"])


class StageGateTests(unittest.TestCase):
    """What each stage did is recorded beside the reading that stage has."""

    def _coverage(self, **overrides):
        base = {
            "status": "warning",
            "paper_budget": {"declared": {"nodes": 9, "edges": 5, "modules": 3, "relations": 4}},
            "delivered": {"nodes": 4, "edges": 2, "bound_edges": 1},
            "advisories": [{"code": "figure_below_paper_budget"}],
        }
        base.update(overrides)
        return base

    def test_every_stage_says_whether_it_moved_and_what_it_read(self):
        trace = build_stage_trace(round_index=1, before_program=_program(["a"]), after_program=_program(["a", "b"]))
        gate = stage_gate_report(round_index=1, trace=trace, coverage=self._coverage())
        self.assertEqual(STAGE_GATE_CONTRACT, gate["contract"])
        self.assertEqual(STAGE_GATE_VERSION, gate["version"])
        self.assertEqual([item["stage"] for item in gate["stages"]], list(STAGE_ORDER))
        self.assertEqual([item["stage"] for item in gate["stages"] if item["changed"]], trace["changed_stages"])

    def test_the_reading_is_the_papers_budget_and_what_the_figure_delivered(self):
        trace = build_stage_trace(round_index=1, before_program=_program(["a"]), after_program=_program(["a"]))
        gate = stage_gate_report(round_index=1, trace=trace, coverage=self._coverage())
        by_stage = {item["stage"]: item for item in gate["stages"]}
        self.assertEqual(by_stage["paper_reading"]["reading"]["declared_modules"], 3)
        self.assertEqual(by_stage["paper_reading"]["reading"]["declared_edges"], 5)
        self.assertEqual(by_stage["content_planning"]["reading"]["delivered_nodes"], 4)
        self.assertEqual(by_stage["content_planning"]["reading"]["bound_edges"], 1)
        self.assertEqual(gate["advisories"], ["figure_below_paper_budget"])

    def test_a_stage_with_no_report_of_its_own_reads_nothing(self):
        trace = build_stage_trace(round_index=1, before_program=_program(["a"]), after_program=_program(["a"]))
        gate = stage_gate_report(round_index=1, trace=trace, coverage={})
        by_stage = {item["stage"]: item for item in gate["stages"]}
        self.assertEqual(by_stage["paper_reading"]["reading"], {})
        self.assertEqual(gate["content_coverage_status"], "not_measured")

    def test_the_gate_records_and_gates_nothing(self):
        gate = stage_gate_report(round_index=2, trace={}, coverage={})
        self.assertIn("gates the round", gate["note"])
        # Nothing in the file is a verdict: no ok/no status of its own, and the coverage status it
        # repeats is the report's own word for whether it was measured.
        self.assertNotIn("ok", gate)
        self.assertNotIn("status", gate)
        self.assertEqual(STAGE_GATE_FILE, "stage_gate.json")


if __name__ == "__main__":
    unittest.main()
