"""The symptom-to-stage table the advisory seat answers from, and the gate checks against."""

from __future__ import annotations

import unittest

from rfs.contracts.evolution import STAGE_ORDER
from rfs.contracts.fault_attribution import (
    ATTRIBUTION_COSTS,
    FAULT_ATTRIBUTION,
    attribution_stages,
    render_fault_attribution_for_prompt,
    requirement_is_on_menu,
    row_for_stage,
    stage_requirements,
    validate_fault_attribution,
)


class FaultAttributionTableTests(unittest.TestCase):
    def test_the_table_reads_row_by_row(self):
        self.assertEqual([], validate_fault_attribution())

    def test_every_row_sends_its_symptom_to_a_stage_the_pipeline_runs(self):
        for row in FAULT_ATTRIBUTION:
            self.assertIn(row["stage"], STAGE_ORDER)
            for item in row["alternatives"]:
                self.assertIn(item["stage"], STAGE_ORDER)

    def test_no_two_rows_claim_the_same_symptom(self):
        symptoms = [" ".join(str(row["symptom"]).casefold().split()) for row in FAULT_ATTRIBUTION]
        self.assertEqual(len(symptoms), len(set(symptoms)))

    def test_tangled_connectors_are_sent_to_the_drawing_and_not_to_the_structure(self):
        # The cheapest stage that can carry the fix is the one that should be named: the drawing
        # can move a line without re-planning anything, so a tangle is a routing defect first.
        row = row_for_stage("ppt_compilation")
        self.assertIn("connectors", row["symptom"])
        self.assertEqual(row["cost"], "cheap")
        self.assertIn("prefer changing the drawing alone", row["note"])
        alternate = next(item for item in row["alternatives"] if item["stage"] == "segmentation")
        self.assertIn("wrong panels", alternate["when"])

    def test_a_bad_palette_is_sent_to_the_palette_stage_and_not_back_through_the_paper(self):
        row = row_for_stage("theme_planning")
        self.assertEqual(row["cost"], "cheap")
        self.assertIn("paper is not re-read", row["note"])

    def test_the_icon_row_is_a_declared_dead_end(self):
        row = row_for_stage("icon_generation")
        self.assertEqual(row["cost"], "deferred")
        self.assertIn("not where this loop's effort goes", row["note"])

    def test_location_stays_inside_the_order_the_pipeline_runs(self):
        for stage in attribution_stages():
            self.assertIn(stage, STAGE_ORDER)
        self.assertIn("ppt_compilation", attribution_stages())
        self.assertIn("paper_reading", attribution_stages())


class RequirementMenuTests(unittest.TestCase):
    def test_a_requirement_written_back_from_the_menu_is_recognised(self):
        menu = stage_requirements()["ppt_compilation"]
        for sentence in menu:
            self.assertTrue(requirement_is_on_menu("ppt_compilation", sentence))

    def test_punctuation_and_case_do_not_change_the_answer(self):
        self.assertTrue(
            requirement_is_on_menu(
                "theme_planning",
                "  At most three base colours, AND the accent colour kept for the state labels. ",
            )
        )

    def test_a_requirement_no_stage_was_offered_is_not_on_the_menu(self):
        self.assertFalse(requirement_is_on_menu("ppt_compilation", "use a bigger font for every label"))
        self.assertFalse(requirement_is_on_menu("", "at most three base colours"))
        self.assertFalse(requirement_is_on_menu("theme_planning", ""))

    def test_every_stage_in_the_table_offers_a_menu(self):
        for stage in attribution_stages():
            self.assertTrue(stage_requirements()[stage], stage)


class AttributionPromptTests(unittest.TestCase):
    def test_the_prompt_block_carries_every_stage_and_its_cost(self):
        block = render_fault_attribution_for_prompt()
        for row in FAULT_ATTRIBUTION:
            self.assertIn(str(row["stage"]), block)
            self.assertIn(str(row["cost"]), block)
            self.assertIn(str(row["requirement"]), block)
        for cost in ATTRIBUTION_COSTS:
            self.assertIn(cost, block)

    def test_the_prompt_block_asks_for_the_three_things_the_gate_reads(self):
        block = render_fault_attribution_for_prompt()
        for field in ("fault_stage", "fault_requirement", "fault_evidence"):
            self.assertIn(field, block)


if __name__ == "__main__":
    unittest.main()
