"""Where a prompt patch lands, and how a prohibition is enforced against a measurement."""

from __future__ import annotations

import unittest

from rfs.contracts.prompt_patch import PROMPT_ASSEMBLY_ORDER
from rfs.evaluation.decomposability import build_decomposability_report
from rfs.evaluation.prompt_patch_check import check_prohibited_patches
from rfs.evolution.direct.prompt_patch import build_prompt_patch
from rfs.generation.icon_assets import build_icon_prompt
from rfs.judge_training.prompts import build_prompt
from rfs.planning.prompt_assembly import (
    PARAMETER_PATCH_KEYS,
    PATCH_BLOCK_LABEL,
    assemble_prompt,
    effective_thresholds,
    patch_parameters,
    render_patch_block,
)
from rfs.planning.vlm_structure_planner import _build_structure_prompt


MEASURED = {
    "experience_id": "direct-exp-1",
    "apply_at": "global_planning",
    "action": {"type": "replan_structure", "scope_type": "global"},
    "condition": {"topology": "pipeline"},
    "accepted": True,
    "repair_gain": 0.2,
    "validated_cases": ["a.pdf"],
    "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True},
}


def advisory(apply_at: str = "global_planning") -> dict:
    return build_prompt_patch(experience={**MEASURED, "apply_at": apply_at})


class PromptPlacementTests(unittest.TestCase):
    def test_the_patch_level_sits_between_the_prompt_and_the_reference_prior(self):
        prompt = assemble_prompt("BASE_INSTRUCTIONS", [advisory()], reference_prior="VISUAL_PRIOR={}")

        self.assertLess(prompt.index("BASE_INSTRUCTIONS"), prompt.index(PATCH_BLOCK_LABEL))
        self.assertLess(prompt.index(PATCH_BLOCK_LABEL), prompt.index("VISUAL_PRIOR={}"))

    def test_no_patch_leaves_the_prompt_byte_for_byte_unchanged(self):
        self.assertEqual(assemble_prompt("BASE", [], reference_prior=""), "BASE")
        self.assertEqual(render_patch_block([]), "")

    def test_the_structure_prompt_carries_the_patch_below_the_schema_and_above_the_prior(self):
        prompt = _build_structure_prompt(
            {"modules": [{"id": "m1"}], "relations": []},
            None,
            {},
            None,
            aspect_ratio="16:9",
            content_top=0.11,
            content_bottom=0.96,
            prompt_patches=[advisory()],
        )

        self.assertLess(prompt.index("METHOD="), prompt.index(PATCH_BLOCK_LABEL))
        self.assertLess(prompt.index(PATCH_BLOCK_LABEL), prompt.index("VISUAL_PRIOR="))

    def test_the_icon_prompt_carries_the_patch_above_the_reference_rules(self):
        prompt = build_icon_prompt({"asset_id": "icon_1", "subject": "encoder"}, prompt_patches=[advisory("icon_generation")])

        self.assertIn(PATCH_BLOCK_LABEL, prompt)
        self.assertLess(prompt.index(PATCH_BLOCK_LABEL), prompt.index("Do not create a complete illustration"))

    def test_the_judge_prompt_carries_the_patch_above_the_required_schema(self):
        row = {"task_type": "repair", "judge_role": "online", "paper": {}, "ground_truth": {}, "labels": {}, "candidate": {}, "process_context": {}}
        plain = build_prompt(row)
        with_patch = build_prompt({**row, "prompt_patches": [advisory("repair")]})

        self.assertNotIn(PATCH_BLOCK_LABEL, plain)
        self.assertLess(with_patch.index(PATCH_BLOCK_LABEL), with_patch.index("Required output schema"))
        self.assertTrue(with_patch.startswith(plain[: plain.index("Required output schema")].rstrip()))

    def test_the_assembly_order_constant_is_re_exported_unchanged(self):
        self.assertEqual(PROMPT_ASSEMBLY_ORDER[3], "promoted_patches")


class SegmentationParameterTests(unittest.TestCase):
    def test_only_segmentation_parameters_move_and_the_first_precedence_wins(self):
        winner = {"apply_at": "segmentation", "parameters": {"min_panel_occupancy": 0.6}, "patch_id": "prompt-patch-a", "version": 1, "median_repair_gain": 0.3, "support": 3}
        loser = {"apply_at": "segmentation", "parameters": {"min_panel_occupancy": 0.2, "max_cross_panel_connectors": 1}, "patch_id": "prompt-patch-b", "version": 1, "median_repair_gain": 0.1, "support": 1}
        other_stage = {"apply_at": "repair", "parameters": {"min_panel_occupancy": 0.9}, "patch_id": "prompt-patch-c", "version": 1}

        overrides = patch_parameters([loser, winner, other_stage])

        self.assertEqual(overrides, {"min_panel_occupancy": 0.6, "max_cross_panel_connectors": 1})
        self.assertTrue(set(PARAMETER_PATCH_KEYS) >= set(overrides))

    def test_an_override_only_replaces_a_threshold_that_exists(self):
        merged = effective_thresholds({"min_panel_occupancy": 0.45}, {"min_panel_occupancy": 0.6, "unknown": 1})

        self.assertEqual(merged, {"min_panel_occupancy": 0.6})

    def test_the_report_measures_against_the_parameterised_boundary(self):
        program = {"semantic_nodes": [{"id": "n1", "panel_id": "p1"}, {"id": "n2", "panel_id": "p2"}], "arrows": [{"id": "a1", "source": "n1", "target": "n2"}]}
        panels = [
            {"id": "p1", "bbox_percent": {"x": 0.05, "y": 0.1, "w": 0.4, "h": 0.4}},
            {"id": "p2", "bbox_percent": {"x": 0.55, "y": 0.1, "w": 0.4, "h": 0.4}},
        ]
        default = build_decomposability_report(None, None, program, panels=panels)
        tightened = build_decomposability_report(None, None, program, panels=panels, thresholds={"max_cross_panel_connectors": 0})

        self.assertEqual(default["metrics"]["cross_panel_connectors"], 1)
        self.assertNotIn("cross_panel_connectors", [item["metric"] for item in default["violations"]])
        self.assertIn("cross_panel_connectors", [item["metric"] for item in tightened["violations"]])
        self.assertEqual(tightened["thresholds"]["max_cross_panel_connectors"], 0.0)


class ProhibitionCheckTests(unittest.TestCase):
    def _prohibition(self, **condition) -> dict:
        refuted = {**MEASURED, "accepted": False, "repair_gain": -0.1}
        return build_prompt_patch(experience=refuted, condition={"metric": "cross_panel_connectors", "op": "<=", "value": 2, **condition})

    def test_a_clean_report_keeps_the_prohibition_and_passes(self):
        check = check_prohibited_patches({"metrics": {"cross_panel_connectors": 1}}, [self._prohibition()])

        self.assertTrue(check["ok"])
        self.assertEqual(check["violations"], [])
        self.assertEqual([item["patch_id"] for item in check["checked"]], [self._prohibition()["patch_id"]])

    def test_a_broken_prohibition_becomes_a_blocker(self):
        check = check_prohibited_patches({"metrics": {"cross_panel_connectors": 3}}, [self._prohibition()])

        self.assertFalse(check["ok"])
        self.assertEqual(len(check["blockers"]), 1)
        self.assertIn("cross_panel_connectors=3.0", check["blockers"][0])

    def test_an_unmeasurable_prohibition_is_reported_rather_than_passed(self):
        check = check_prohibited_patches({"metrics": {}}, [self._prohibition()])

        self.assertTrue(check["ok"])
        self.assertEqual(check["unchecked"][0]["reason"], "cross_panel_connectors was not measured this round")

    def test_an_unenforceable_prohibition_is_reported_rather_than_passed(self):
        unenforceable = {"patch_id": "prompt-patch-x", "version": 1, "apply_at": "repair", "class": "prohibited", "condition": {"topology": "pipeline"}, "directive": "Never apply resegment when topology=pipeline.", "source_experience_ids": ["direct-exp-1"], "support": 1, "median_repair_gain": -0.1, "status": "promoted", "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "measured": True}}

        check = check_prohibited_patches({"metrics": {"cross_panel_connectors": 9}}, [unenforceable])

        self.assertTrue(check["ok"])
        self.assertEqual(check["unchecked"][0]["reason"], "the prohibition names no measurable metric bound")


if __name__ == "__main__":
    unittest.main()
