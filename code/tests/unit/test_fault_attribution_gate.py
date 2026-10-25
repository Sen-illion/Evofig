"""An attribution that names a stage the trace never ran is dropped, not kept as evidence."""

from __future__ import annotations

import unittest

from rfs.workflows.stage_trace import attribution_verdict, build_stage_trace, ledger_attribution


def _program():
    return {
        "panels": [{"id": "panel_a", "bbox_percent": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5}}],
        "semantic_nodes": [
            {"id": "a", "panel_id": "panel_a", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}
        ],
        "slots": [],
        "arrows": [],
    }


def _trace():
    program = _program()
    return build_stage_trace(
        round_index=1,
        before_program=program,
        after_program=program,
        action={"type": "reroute"},
    )


class FaultAttributionGateTests(unittest.TestCase):
    def test_saying_nothing_is_allowed(self):
        verdict = attribution_verdict({}, _trace())
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["attribution"], {})

    def test_a_stage_outside_the_traces_vocabulary_is_rejected(self):
        verdict = attribution_verdict(
            {"fault_stage": "layout_engine_v2", "fault_evidence": [{"field": "node_count", "value": 3}]},
            _trace(),
        )
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "fault_stage_not_in_trace")

    def test_naming_a_stage_without_evidence_is_rejected(self):
        verdict = attribution_verdict({"fault_stage": "ppt_compilation"}, _trace())
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "fault_evidence_missing")

    def test_evidence_that_names_no_measured_field_is_rejected(self):
        verdict = attribution_verdict(
            {"fault_stage": "ppt_compilation", "fault_evidence": [{"field": "ghost_metric", "value": 3}]},
            _trace(),
        )
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "fault_evidence_unreadable")

    def test_evidence_against_a_measured_field_is_kept(self):
        verdict = attribution_verdict(
            {
                "fault_stage": "ppt_compilation",
                "fault_evidence": [
                    {"stage": "ppt_compilation", "field": "crossing_count", "value": 6, "bound": 2}
                ],
            },
            _trace(),
        )
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["attribution"]["fault_stage"], "ppt_compilation")

    def test_a_measured_zero_is_a_reading_not_a_missing_value(self):
        # ``crossing_count`` reads 0 in exactly the rounds that are fine, and it is the field a
        # coach names when it says the connectors were left alone. Reading the value as ``x or ""``
        # made the zero look absent and threw the whole claim away with it.
        verdict = attribution_verdict(
            {
                "fault_stage": "ppt_compilation",
                "fault_evidence": [
                    {"stage": "ppt_compilation", "field": "crossing_count", "value": 0, "bound": 0},
                    {"stage": "ppt_compilation", "field": "wrapped_label_count", "value": 8, "bound": 0},
                ],
            },
            _trace(),
        )
        self.assertTrue(verdict["ok"])
        self.assertEqual(len(verdict["attribution"]["fault_evidence"]), 2)

    def test_an_evidence_item_with_no_value_at_all_is_still_rejected(self):
        verdict = attribution_verdict(
            {"fault_stage": "ppt_compilation", "fault_evidence": [{"stage": "ppt_compilation", "field": "crossing_count"}]},
            _trace(),
        )
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "fault_evidence_unreadable")

    def test_a_metric_named_by_the_path_the_seat_read_it_at_is_kept(self):
        # The seat is shown the trace inside ``process_context``, and it wrote the path it read --
        # ``process_context.stage_trace.metrics.node_occupancy`` -- while the gate held only the
        # bare key, so a claim about a number the round had just measured was dropped for a
        # spelling.
        verdict = attribution_verdict(
            {
                "fault_stage": "ppt_compilation",
                "fault_evidence": [
                    {
                        "stage": "ppt_compilation",
                        "field": "process_context.stage_trace.metrics.crossing_count",
                        "value": 6,
                        "bound": 2,
                    }
                ],
            },
            _trace(),
        )
        self.assertTrue(verdict["ok"])
        self.assertEqual(len(verdict["attribution"]["fault_evidence"]), 1)

    def test_a_field_of_another_block_the_round_showed_is_kept(self):
        # The colour report and the grounding Gate are shown to the seat in the same call, so a
        # claim about one of their fields is about something the round really measured.
        verdict = attribution_verdict(
            {
                "fault_stage": "ppt_compilation",
                "fault_evidence": [
                    {"stage": "ppt_compilation", "field": "grounding_gate.edge_coverage", "value": 0.75, "bound": 1.0},
                    {"stage": "ppt_compilation", "field": "color_report.feedback[0].reason", "value": "the accent is muddy"},
                ],
            },
            _trace(),
            shown={"grounding_gate": {"edge_coverage": 0.75}, "color_report": {"feedback": [{"reason": "the accent is muddy"}]}},
        )
        self.assertTrue(verdict["ok"])
        self.assertEqual(len(verdict["attribution"]["fault_evidence"]), 2)

    def test_one_unreadable_item_does_not_discard_the_one_beside_it(self):
        # The round the coach was reviewing had located a real defect and cited a field the round
        # never put in front of it in the same breath. Throwing the whole claim away cost the
        # location; the unreadable half is dropped and named instead.
        verdict = attribution_verdict(
            {
                "fault_stage": "ppt_compilation",
                "fault_evidence": [
                    {"stage": "ppt_compilation", "field": "crossing_count", "value": 6, "bound": 2},
                    {"stage": "ppt_compilation", "field": "ghost_metric", "value": 3},
                ],
            },
            _trace(),
        )
        self.assertTrue(verdict["ok"])
        self.assertEqual([item["field"] for item in verdict["attribution"]["fault_evidence"]], ["crossing_count"])
        self.assertEqual([item["field"] for item in verdict["attribution"]["fault_evidence_dropped"]], ["ghost_metric"])
        line = ledger_attribution(verdict)
        self.assertEqual(line["fault_stage"], "ppt_compilation")
        self.assertEqual([item["field"] for item in line["fault_evidence_dropped"]], ["ghost_metric"])

    def test_without_a_trace_any_attribution_is_rejected(self):
        verdict = attribution_verdict(
            {"fault_stage": "ppt_compilation", "fault_evidence": [{"field": "crossing_count", "value": 6}]},
            {},
        )
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "no_stage_trace")


class LedgerAttributionTests(unittest.TestCase):
    """What the round ledger says about the claim, including when it was refused."""

    def test_a_refused_claim_is_recorded_as_refused_not_as_silence(self):
        verdict = attribution_verdict(
            {"fault_stage": "ppt_compilation", "fault_evidence": [{"field": "ghost_metric", "value": 3}]},
            _trace(),
        )
        line = ledger_attribution(verdict)
        self.assertEqual(line["fault_stage"], "")
        self.assertEqual(line["fault_evidence"], [])
        self.assertEqual(line["attribution_rejected"], "fault_evidence_unreadable")

    def test_a_claim_the_trace_could_read_is_kept(self):
        verdict = attribution_verdict(
            {
                "fault_stage": "ppt_compilation",
                "fault_evidence": [{"stage": "ppt_compilation", "field": "crossing_count", "value": 6, "bound": 2}],
            },
            _trace(),
        )
        line = ledger_attribution(verdict)
        self.assertEqual(line["fault_stage"], "ppt_compilation")
        self.assertEqual(line["fault_evidence"][0]["field"], "crossing_count")
        self.assertEqual(line["attribution_rejected"], "")

    def test_naming_nothing_is_not_a_rejection(self):
        line = ledger_attribution(attribution_verdict({}, _trace()))
        self.assertEqual(
            line,
            {"fault_stage": "", "fault_requirement": "", "fault_requirement_status": "", "fault_evidence": [], "attribution_rejected": ""},
        )

    def test_a_claim_that_names_a_stage_and_stops_keeps_its_location(self):
        # The stage and its readable evidence are the expensive half of the claim: they say where
        # the fix goes and what was measured. Refusing the whole claim over a missing sentence
        # would cost the round its location, so the omission is recorded on the claim instead.
        verdict = attribution_verdict(
            {"fault_stage": "ppt_compilation", "fault_evidence": [{"stage": "ppt_compilation", "field": "crossing_count", "value": 6, "bound": 2}]},
            _trace(),
        )
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["attribution"]["fault_stage"], "ppt_compilation")
        self.assertEqual(verdict["attribution"]["fault_requirement"], "")
        self.assertEqual(verdict["attribution"]["fault_requirement_status"], "unstated")
        self.assertEqual(ledger_attribution(verdict)["fault_requirement_status"], "unstated")

    def test_the_requirement_the_stage_failed_to_write_travels_with_the_claim(self):
        verdict = attribution_verdict(
            {
                "fault_stage": "ppt_compilation",
                "fault_requirement": "a connector leaves its node on the side that faces its target and runs down a channel shared with its neighbours instead of across a panel",
                "fault_evidence": [{"stage": "ppt_compilation", "field": "crossing_count", "value": 6, "bound": 2}],
            },
            _trace(),
        )
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["attribution"]["fault_requirement_status"], "on_menu")
        self.assertIn("channel shared with its neighbours", ledger_attribution(verdict)["fault_requirement"])

    def test_a_requirement_the_stage_was_never_offered_is_kept_and_flagged(self):
        # A requirement nobody wrote down is still somewhere to look, so it is kept; it is flagged
        # because the ledger has to be able to tell a menu answer from an invented one when it
        # scores how often the location was right.
        verdict = attribution_verdict(
            {
                "fault_stage": "ppt_compilation",
                "fault_requirement": "use a bigger font for every label",
                "fault_evidence": [{"stage": "ppt_compilation", "field": "crossing_count", "value": 6, "bound": 2}],
            },
            _trace(),
        )
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["attribution"]["fault_requirement_status"], "invented")
        self.assertEqual(verdict["attribution"]["fault_requirement"], "use a bigger font for every label")

    def test_a_missing_verdict_reads_as_no_claim(self):
        self.assertEqual(ledger_attribution(None)["attribution_rejected"], "")



if __name__ == "__main__":
    unittest.main()
