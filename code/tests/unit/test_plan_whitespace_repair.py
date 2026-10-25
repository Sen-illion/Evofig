"""Tests for the whitespace a planner wraps inside its own names.

A model answer that escapes a line break in the middle of a key or an identifier
is valid JSON, so it used to parse, then fail every grounding check against the
paper.  These tests pin the repair that removes that whitespace before the plan
is validated.
"""

from __future__ import annotations

import unittest
from unittest import mock

from rfs.planning import paper_contract
from rfs.planning.paper_contract import (
    apply_evidence_fixes,
    collect_evidence_slots,
    normalize_plan,
    repair_plan_evidence,
    repair_plan_whitespace,
)


class RepairPlanWhitespaceTests(unittest.TestCase):
    def test_keys_and_identifiers_lose_their_wrapped_line_breaks(self) -> None:
        repaired = repair_plan_whitespace(
            {
                "\nfigure_goal": "one story",
                "central\n_claim": "a claim",
                "must_\nshow": [{"id": "I_\npool_A", "visible_label": "Inhibitory pool"}],
            }
        )
        self.assertIn("figure_goal", repaired)
        self.assertIn("central_claim", repaired)
        self.assertEqual(repaired["must_show"][0]["id"], "I_pool_A")

    def test_prose_keeps_a_single_space_and_is_trimmed(self) -> None:
        repaired = repair_plan_whitespace({"summary": "  two\n\nlines\there  "})
        self.assertEqual(repaired["summary"], "two lines here")

    def test_identifier_values_are_cleaned_at_any_depth(self) -> None:
        repaired = repair_plan_whitespace(
            {
                "panel_graphs": [
                    {
                        "id": "panel_a",
                        "relations": [{"source": "x_\nE", "target": "I_pool_A", "evidence_ids": ["E0001\n"]}],
                    }
                ]
            }
        )
        relation = repaired["panel_graphs"][0]["relations"][0]
        self.assertEqual(relation["source"], "x_E")
        self.assertEqual(relation["evidence_ids"], ["E0001"])

    def test_normalize_plan_cleans_every_planner_answer(self) -> None:
        normalized = normalize_plan({"figure_specification": {"must_\nshow": [{"id": "a_\nb"}]}}, {"aspect_ratio": "16:9"})
        self.assertEqual(normalized["figure_specification"]["must_show"][0]["id"], "a_b")
        self.assertEqual(normalized["layout_intent"]["canvas_ratio"], "16:9")

    def test_cleaned_names_are_the_ones_the_repair_can_address(self) -> None:
        spec = repair_plan_whitespace({"modules": [{"id": "I_\npool_A", "visible_label": "Pool", "evidence_ids": []}]})
        self.assertEqual(list(collect_evidence_slots(spec)), ["module:I_pool_A"])


class ApplyEvidenceFixWhitespaceTests(unittest.TestCase):
    def test_a_key_that_differs_only_by_whitespace_still_lands(self) -> None:
        spec = {"modules": [{"id": "I_pool_A", "visible_label": "Pool", "evidence_ids": []}]}
        outcome = apply_evidence_fixes(spec, [{"key": "module:I_ pool_A\n", "evidence_ids": ["E0001"]}], valid_evidence={"E0001"})
        self.assertEqual([item["key"] for item in outcome["applied"]], ["module:I_pool_A"])
        self.assertEqual(spec["modules"][0]["evidence_ids"], ["E0001"])


class RepairAnswerRecordTests(unittest.TestCase):
    def test_the_answer_is_recorded_even_when_nothing_can_be_applied(self) -> None:
        plan = {"figure_specification": {"modules": [{"id": "m1", "visible_label": "Encoder", "evidence_ids": []}]}}
        with (
            mock.patch.object(paper_contract, "vlm_credentials_available", return_value=True),
            mock.patch.object(paper_contract, "call_vlm_json", return_value={"evidence_fixes": []}),
        ):
            _, metadata = repair_plan_evidence(plan, {"evidence": [{"id": "E0001", "text": "one"}]}, ["module m1 has no evidence_ids"])
        self.assertEqual(metadata["fixes_returned"], 0)
        self.assertEqual(metadata["answer_keys"], ["evidence_fixes"])
        self.assertEqual(metadata["raw_response"], {"evidence_fixes": []})

    def test_an_unreadable_answer_is_recorded_with_its_keys(self) -> None:
        plan = {"figure_specification": {"modules": [{"id": "m1", "visible_label": "Encoder", "evidence_ids": []}]}}
        with (
            mock.patch.object(paper_contract, "vlm_credentials_available", return_value=True),
            mock.patch.object(paper_contract, "call_vlm_json", return_value={"fixes": [{"key": "module:m1", "evidence_ids": ["E0001"]}]}),
        ):
            _, metadata = repair_plan_evidence(plan, {"evidence": [{"id": "E0001", "text": "one"}]}, ["module m1 has no evidence_ids"])
        self.assertEqual(metadata["fixes_returned"], 0)
        self.assertEqual(metadata["answer_keys"], ["fixes"])
        self.assertEqual(metadata["applied"], [])


if __name__ == "__main__":
    unittest.main()