"""Advisory Judge actions must be legal, bound to objects, and empty means stop."""

from __future__ import annotations

import unittest

from rfs.contracts.evolution import normalize_judge_result, validate_judge_result
from rfs.evaluation.interactive_judgement import judge_direct_run
from rfs.evolution.direct.controller import select_bounded_action


def _program(*, arrows=None, nodes=None, slots=None, panels=None):
    return {
        "panels": panels or [{"id": "p", "bbox_percent": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}],
        "semantic_nodes": nodes or [
            {"id": "a", "panel_id": "p", "bbox_percent": {"x": 0.05, "y": 0.1, "w": 0.15, "h": 0.15}},
            {"id": "b", "panel_id": "p", "bbox_percent": {"x": 0.75, "y": 0.1, "w": 0.15, "h": 0.15}},
            {"id": "c", "panel_id": "p", "bbox_percent": {"x": 0.35, "y": 0.12, "w": 0.2, "h": 0.15}},
        ],
        "slots": slots or [
            {"id": "slot_a", "asset_id": "icon_a", "panel_id": "p", "parent_id": "a", "bbox_percent": {"x": 0.06, "y": 0.11, "w": 0.05, "h": 0.05}},
            {"id": "slot_b", "asset_id": "icon_b", "panel_id": "p", "parent_id": "b", "bbox_percent": {"x": 0.76, "y": 0.11, "w": 0.05, "h": 0.05}},
        ],
        "arrows": arrows or [],
        "text_program": {"items": [{"id": "t", "text": "encoder", "bbox_percent": {"x": 0.05, "y": 0.9, "w": 0.1, "h": 0.04}}]},
    }


class AdvisoryActionSchemaTests(unittest.TestCase):
    def test_a_named_repair_without_required_ids_is_dropped(self):
        judgement = normalize_judge_result(
            {
                "judge_role": "online_bootstrap",
                "stage_feedback": [{"scope_type": "connector", "problem": "arrows overlap", "requested_actions": [{"action": "reroute"}]}],
            },
            role="online_bootstrap",
        )
        self.assertEqual([], judgement["repair"])
        self.assertIsNone(select_bounded_action(judgement))

    def test_edge_ids_on_the_action_itself_are_kept(self):
        action = select_bounded_action(
            {
                "stage_feedback": [
                    {
                        "scope_type": "connector",
                        "problem": "arrows overlap",
                        "requested_actions": [{"action": "reroute", "edge_ids": ["e1"]}],
                    }
                ]
            }
        )
        self.assertEqual("reroute", action["type"])
        self.assertEqual(["e1"], action["edge_ids"])

    def test_an_empty_advisory_list_is_not_a_nudge(self):
        self.assertIsNone(select_bounded_action({"stage_feedback": [{"scope_type": "interface", "requested_actions": []}]}))

    def test_a_repair_named_outside_a_stage_block_is_read(self):
        # The seat writes the menu as one flat list, so the repairs land at the top level and
        # stage_feedback keeps only the prose. Reading the nested spelling alone ran no move
        # while the answer had already said how to fix the figure.
        judgement = normalize_judge_result(
            {
                "judge_role": "online_trained",
                "status": "fault",
                "stage_feedback": [{"scope_type": "global", "problem": "arrows overlap"}],
                "requested_actions": [{"action": "reroute", "edge_ids": ["rel_a_b"]}],
            },
            role="online_trained",
        )
        chosen = select_bounded_action(judgement)
        self.assertEqual("reroute", chosen["type"])
        self.assertEqual(["rel_a_b"], chosen["edge_ids"])

    def test_a_repair_named_twice_is_read_once(self):
        judgement = normalize_judge_result(
            {
                "judge_role": "online_trained",
                "status": "fault",
                "stage_feedback": [
                    {"scope_type": "connector", "problem": "arrows overlap", "requested_actions": [{"action": "reroute", "edge_ids": ["e1"]}]}
                ],
                "requested_actions": [{"action": "reroute", "edge_ids": ["e1"]}],
            },
            role="online_trained",
        )
        self.assertEqual(1, len(judgement["repair"]))

    def test_a_loosely_named_repair_keeps_the_verdict_it_came_with(self):
        # A repair named outside a stage block is a repair, not a verdict: a seat that said
        # pass keeps saying pass, and the entry is still readable by the validator.
        raw = {
            "judge_role": "online_trained",
            "status": "pass",
            "requested_actions": [{"action": "reroute", "edge_ids": ["e1"]}],
        }
        judgement = normalize_judge_result(raw, role="online_trained")
        self.assertEqual("pass", judgement["status"])
        self.assertEqual(1, len(judgement["stage_feedback"]))
        self.assertTrue(validate_judge_result(raw)["ok"])

    def test_a_mechanism_rule_named_outside_a_stage_block_is_read(self):
        judgement = normalize_judge_result(
            {
                "judge_role": "online_trained",
                "status": "fault",
                "stage_feedback": [{"scope_type": "global", "problem": "too sparse"}],
                "requested_actions": [{"action": "layout_constraint", "params": {"panel_occupancy_min": 0.5}}],
            },
            role="online_trained",
        )
        chosen = select_bounded_action(judgement)
        self.assertEqual("layout_constraint", chosen["type"])
        self.assertEqual({"panel_occupancy_min": 0.5}, chosen["params"])

    def test_an_acceptance_answer_carries_no_repair(self):
        judgement = normalize_judge_result(
            {
                "judge_role": "frozen_choice",
                "overall_score": 0.7,
                "requested_actions": [{"action": "reroute", "edge_ids": ["e1"]}],
            },
            role="frozen_choice",
        )
        self.assertEqual([], judgement["stage_feedback"])
        self.assertEqual([], judgement["repair"])

    def test_crossing_arrows_prescribe_reroute_with_edge_ids(self):
        layout = {
            "panels": [{"id": "p", "bbox_percent": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}],
            "nodes": [
                {"id": "a", "panel_id": "p", "bbox_percent": {"x": 0.05, "y": 0.1, "w": 0.15, "h": 0.15}},
                {"id": "b", "panel_id": "p", "bbox_percent": {"x": 0.75, "y": 0.1, "w": 0.15, "h": 0.15}},
                {"id": "c", "panel_id": "p", "bbox_percent": {"x": 0.35, "y": 0.12, "w": 0.2, "h": 0.15}},
            ],
            "connectors": [],
        }
        result = judge_direct_run(
            {"modules": [{"id": "a"}, {"id": "b"}, {"id": "c"}]},
            layout,
            {"icons": [
                {"asset_id": "icon_a", "semantic_node_ids": ["a"], "prompt": "a"},
                {"asset_id": "icon_b", "semantic_node_ids": ["b"], "prompt": "b"},
            ]},
            program=_program(
                arrows=[
                    {
                        "id": "e1",
                        "source": "a",
                        "target": "b",
                        "routing_algorithm": "direct-final",
                        "path_percent": [[0.1, 0.18], [0.8, 0.18]],
                    }
                ]
            ),
        )
        interface = next(item for item in result["stage_feedback"] if item.get("stage") == "interface")
        self.assertIn("e1", interface.get("edge_ids") or [])
        chosen = select_bounded_action({"stage_feedback": [interface]})
        self.assertEqual("reroute", chosen["type"])
        self.assertIn("e1", chosen["edge_ids"])

    def test_a_single_sparse_panel_prescribes_a_slot_resize(self):
        layout = {
            "panels": [{"id": "p", "bbox_percent": {"x": 0.0, "y": 0.05, "w": 0.95, "h": 0.9}}],
            "nodes": [{"id": "a", "panel_id": "p", "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.08, "h": 0.08}}],
            "connectors": [],
        }
        result = judge_direct_run(
            {"modules": [{"id": "a"}]},
            layout,
            {"icons": [{"asset_id": "icon_a", "semantic_node_ids": ["a"], "prompt": "a"}]},
            program=_program(
                panels=[{"id": "p", "bbox_percent": {"x": 0.0, "y": 0.05, "w": 0.95, "h": 0.9}}],
                nodes=[{"id": "a", "panel_id": "p", "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.08, "h": 0.08}}],
                slots=[{"id": "slot_a", "asset_id": "icon_a", "panel_id": "p", "parent_id": "a", "bbox_percent": {"x": 0.11, "y": 0.21, "w": 0.04, "h": 0.04}}],
                arrows=[],
            ),
        )
        structure = next(item for item in result["stage_feedback"] if "sparse panels" in str(item.get("problem") or ""))
        chosen = select_bounded_action({"stage_feedback": [structure]})
        self.assertIn(chosen["type"], {"resize_nodes", "relayout_slots"})
        self.assertTrue(chosen.get("slot_ids") or chosen.get("object_ids"))


if __name__ == "__main__":
    unittest.main()
