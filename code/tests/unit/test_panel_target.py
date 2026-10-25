"""The paper's own panels become a target a benchmark run draws to.

The target is read from the contract, sent to the two prompts that decide what the
figure shows, and used at the end to measure how much of the paper's own panel
structure the delivered figure carries. It is an instruction and a measurement,
never a gate: a miss is reported, never refused.
"""

from __future__ import annotations

import unittest

from rfs.planning.panel_target import (
    NO_PANELS_REASON,
    PANEL_TARGET_SOURCE,
    check_design_against_panel_target,
    compact_panel_target,
    panel_target_from_specification,
)


SPECIFICATION = {
    "panel_graphs": [
        {
            "id": "panel_a",
            "panel_label": "a",
            "label": "Overview",
            "nodes": [
                {"instance_id": "n1", "name": "Encoder"},
                {"instance_id": "n2", "name": "Decoder"},
                {"instance_id": "n3", "name": "Encoder"},
            ],
        },
        {
            "id": "panel_b",
            "panel_label": "b",
            "label": "Training",
            "nodes": [{"instance_id": "n4", "name": "Loss"}],
        },
    ]
}


class PanelTargetFromSpecificationTests(unittest.TestCase):
    def test_it_reads_the_papers_panels_and_their_visible_labels(self):
        target = panel_target_from_specification(SPECIFICATION)
        self.assertTrue(target["available"])
        self.assertEqual(target["reason"], "")
        self.assertEqual(target["source"], PANEL_TARGET_SOURCE)
        self.assertEqual(target["panel_count"], 2)
        self.assertEqual(target["label_count"], 3)
        self.assertEqual([panel["labels"] for panel in target["panels"]], [["Encoder", "Decoder"], ["Loss"]])

    def test_a_paper_without_panels_has_no_target(self):
        target = panel_target_from_specification({})
        self.assertFalse(target["available"])
        self.assertEqual(target["reason"], NO_PANELS_REASON)
        self.assertEqual(target["panels"], [])

    def test_panels_without_a_labelled_node_are_not_a_target(self):
        target = panel_target_from_specification({"panel_graphs": [{"id": "p", "nodes": []}, {"id": "q"}]})
        self.assertFalse(target["available"])
        self.assertEqual(target["reason"], NO_PANELS_REASON)

    def test_it_holds_at_most_the_stated_labels_per_panel(self):
        specification = {"panel_graphs": [{"id": "p", "nodes": [{"name": "L%d" % i} for i in range(30)]}]}
        target = panel_target_from_specification(specification, max_items_per_panel=5)
        self.assertEqual(len(target["panels"][0]["labels"]), 5)
        self.assertEqual(target["panels"][0]["node_count"], 30)

    def test_it_is_deterministic(self):
        self.assertEqual(
            panel_target_from_specification(SPECIFICATION),
            panel_target_from_specification(SPECIFICATION),
        )


class CompactPanelTargetTests(unittest.TestCase):
    def test_an_unavailable_target_yields_no_block(self):
        self.assertEqual(compact_panel_target(panel_target_from_specification({})), "")
        self.assertEqual(compact_panel_target(None), "")

    def test_it_caps_the_panels_and_their_labels(self):
        specification = {
            "panel_graphs": [
                {"id": "p%d" % i, "panel_label": chr(ord("a") + i), "nodes": [{"name": "L%d_%d" % (i, j)} for j in range(20)]}
                for i in range(12)
            ]
        }
        target = panel_target_from_specification(specification)
        block = compact_panel_target(target, max_panels=3, max_labels_per_panel=2)
        self.assertIn('"panel":"a"', block)
        self.assertIn('"panel":"c"', block)
        self.assertNotIn('"panel":"d"', block)
        self.assertIn("L0_1", block)
        self.assertNotIn("L0_2", block)

    def test_it_is_deterministic(self):
        target = panel_target_from_specification(SPECIFICATION)
        self.assertEqual(compact_panel_target(target), compact_panel_target(target))


def design(*labels, panels=1):
    return {
        "panels": [{"id": "p%d" % i} for i in range(panels)],
        "nodes": [{"label": label} for label in labels],
        "connectors": [],
    }


class CheckDesignAgainstPanelTargetTests(unittest.TestCase):
    def test_an_unavailable_target_is_reported_as_disabled(self):
        report = check_design_against_panel_target(design("Encoder"), panel_target_from_specification({}))
        self.assertFalse(report["enabled"])
        self.assertEqual(report["reason"], NO_PANELS_REASON)

    def test_a_design_that_carries_every_label_scores_full(self):
        target = panel_target_from_specification(SPECIFICATION)
        report = check_design_against_panel_target(design("Encoder", "Decoder", "Loss", panels=2), target)
        self.assertTrue(report["enabled"])
        self.assertEqual(report["labels_matched_count"], 3)
        self.assertEqual(report["labels_missing_count"], 0)
        self.assertEqual(report["label_coverage"], 1.0)
        self.assertEqual(report["container_coverage"], 1.0)
        self.assertEqual(report["panels_target_count"], 2)
        self.assertEqual(report["panels_produced_count"], 2)

    def test_a_wrapped_label_counts_as_contained_but_not_as_matched(self):
        target = panel_target_from_specification(SPECIFICATION)
        report = check_design_against_panel_target(design("Encoder", "Decoder unit"), target)
        self.assertEqual(report["labels_matched_count"], 1)
        self.assertEqual(report["labels_contained_count"], 2)
        self.assertEqual(report["labels_missing_count"], 1)
        self.assertLess(report["label_coverage"], report["container_coverage"])
        self.assertIn("Loss", report["missing_labels"])

    def test_it_is_deterministic(self):
        target = panel_target_from_specification(SPECIFICATION)
        plan = design("Encoder", "Decoder unit")
        self.assertEqual(
            check_design_against_panel_target(plan, target),
            check_design_against_panel_target(plan, target),
        )


if __name__ == "__main__":
    unittest.main()
