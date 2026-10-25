"""Tests for compiler-readable Memory parameters and their injection."""

from __future__ import annotations

import unittest

from rfs.composition.direct_connectors import apply_memory_routing_constraints
from rfs.composition.typography import optimize_direct_typography
from rfs.contracts.memory_parameters import (
    density_targets_from,
    icon_is_transferable,
    injection_blocked_reason,
    replayable_theme,
    stamp_executable_parameters,
)
from rfs.evolution.direct.experience import build_direct_experience_result
from rfs.evolution.direct.routing import inject_experience
from rfs.planning.density_advisory import build_density_advisory
from rfs.workflows.upstream_retry import run_upstream_retry


PALETTE = {
    "theme_name": "memory-blue",
    "canvas_background": "#F8FAFC",
    "text": {"primary": "#173042", "secondary": "#4A6572", "muted": "#8A9AA8", "inverse": "#FFFFFF"},
    "connectors": {"primary": "#477FA8", "secondary": "#7A9BB0", "feedback": "#C47B5A"},
    "roles": {"stage": {"fill": "#EAF4FA", "border": "#477FA8", "text": "#173042"}},
    "role_mapping": {"encoder": "stage"},
    "panels": {"paper_specific_panel": {"fill": "#FF0000"}},
}


def _row(action, *, apply_at="theme_planning"):
    return {
        "experience_id": "e-theme",
        "status": "promoted",
        "apply_at": apply_at,
        "action": action,
        "forbidden_changes": [],
        "repair_gain": 0.2,
        "validated_cases": ["a.pdf"],
        "outcome": {"score_delta": 0.2},
        "strategy": {"instruction": "replay the palette", "action": action},
    }


class ReplayableThemeTests(unittest.TestCase):
    def test_panel_ids_are_dropped_and_roles_travel(self):
        replayed = replayable_theme(PALETTE)
        self.assertEqual("#F8FAFC", replayed["canvas_background"])
        self.assertEqual("#173042", replayed["text"]["primary"])
        self.assertIn("stage", replayed["roles"])
        self.assertNotIn("panels", replayed)

    def test_an_empty_dict_is_not_a_palette(self):
        self.assertEqual({}, replayable_theme({"theme_name": "blank"}))


class ThemeMemoryTests(unittest.TestCase):
    def test_a_successful_theme_row_carries_the_palette(self):
        result = build_direct_experience_result(
            round_index=1,
            previous={"overall_score": 0.4},
            current={"overall_score": 0.7, "blocking_issue_count": 0},
            layout={"topology": "pipeline", "nodes": [{"id": "n1"}], "connectors": []},
            action={"type": "replan_theme", "scope_type": "global"},
            source_case="paper.pdf",
            color_theme=PALETTE,
        )
        self.assertTrue(result["ok"])
        theme = result["experience"]["action"]["color_theme"]
        self.assertEqual("#477FA8", theme["connectors"]["primary"])
        self.assertNotIn("panels", theme)

    def test_injection_copies_the_palette_and_heuristic_mode_applies_it(self):
        action = stamp_executable_parameters({"type": "replan_theme"}, color_theme=PALETTE)
        injection = inject_experience([_row(action)], {"type": "replan_theme"}, known_ids=())
        self.assertEqual("replan_theme", injection["action"]["type"])
        self.assertEqual("#EAF4FA", injection["action"]["color_theme"]["roles"]["stage"]["fill"])
        applied = []

        def apply_theme(theme):
            applied.append(theme)
            return {"figure_layout_plan": {"nodes": [{"id": "n1"}]}}

        report = run_upstream_retry(
            injection["action"],
            planner_mode="heuristic",
            theme={"canvas_background": "#FFFFFF"},
            design={"nodes": [{"id": "n1"}]},
            theme_applier=apply_theme,
        )
        self.assertTrue(report["applied"])
        self.assertEqual("memory", report["theme_report"]["mode"])
        self.assertEqual(1, len(applied))

    def test_a_theme_row_without_a_palette_is_not_injected(self):
        injection = inject_experience(
            [_row({"type": "replan_theme"})],
            {"type": "replan_theme"},
        )
        self.assertIsNone(injection["action"])
        self.assertIn("palette", injection["injection"]["reason"])


class DensityMemoryTests(unittest.TestCase):
    def test_density_targets_travel_and_the_advisory_reads_them(self):
        action = stamp_executable_parameters(
            {"type": "resize_nodes", "factor": 1.12, "dx": 0.01, "dy": 0.0, "object_ids": ["n1"]},
            layout={"density_targets": {"panel_occupancy_min": 0.62, "minimum_font_size": 13.0, "node_spacing": 0.04}},
        )
        self.assertEqual(0.62, action["density_targets"]["panel_occupancy_min"])
        advisory = build_density_advisory(
            {"panels": [{"id": "p", "bbox_percent": {"x": 0, "y": 0.1, "w": 0.5, "h": 0.8}}], "nodes": [{"id": "n1", "panel_id": "p", "bbox_percent": {"x": 0.05, "y": 0.2, "w": 0.1, "h": 0.1}}]},
            {"action": action},
        )
        self.assertEqual(0.62, advisory["experience_rules_applied"]["panel_occupancy_min"])
        self.assertEqual(13.0, advisory["experience_rules_applied"]["minimum_font_size"])

    def test_typography_raises_the_floor_to_the_stored_size(self):
        program = {
            "canvas": {"width_in": 13.5, "height_in": 7.5},
            "density_targets": {"minimum_font_size": 14.0},
            "semantic_nodes": [{"id": "n1", "label": "Encoder", "font_size_role": "node_label", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.08}}],
            "text_program": {"items": []},
        }
        updated, report = optimize_direct_typography(program)
        self.assertGreaterEqual(updated["semantic_nodes"][0]["min_readable_size_pt"], 14.0)
        self.assertEqual(1, report["node_count"])


class ConnectorMemoryTests(unittest.TestCase):
    def test_stored_paths_replay_onto_matching_arrows(self):
        program = {
            "arrows": [
                {"id": "e1", "source": "a", "target": "b", "relation_type": "data_flow", "path_percent": [[0.1, 0.1], [0.2, 0.2]]},
            ]
        }
        updated, report = apply_memory_routing_constraints(
            program,
            {"selections": [{"arrow_id": "e1", "path_percent": [[0.2, 0.3], [0.6, 0.3]]}]},
        )
        self.assertEqual("applied", report["status"])
        self.assertEqual([[0.2, 0.3], [0.6, 0.3]], updated["arrows"][0]["path_percent"])


class IconMemoryTests(unittest.TestCase):
    def test_placeholder_icon_repairs_are_not_written(self):
        result = build_direct_experience_result(
            round_index=1,
            previous={"overall_score": 0.4},
            current={"overall_score": 0.7, "blocking_issue_count": 0},
            layout={"topology": "pipeline", "nodes": [{"id": "n1"}], "connectors": []},
            action={"type": "regen_assets", "scope_type": "slot", "slot_ids": ["icon_n"]},
            source_case="paper.pdf",
            icon_mode="placeholder",
        )
        self.assertFalse(result["ok"])
        self.assertEqual("icon_mode_not_transferable", result["reason"])

    def test_image2_icon_repairs_stamp_the_slot_constraint(self):
        action = stamp_executable_parameters(
            {"type": "regen_assets", "slot_ids": ["icon_n"], "problem": "draw a cleaner encoder glyph"},
            icon_mode="image2",
        )
        self.assertTrue(icon_is_transferable(action, icon_mode="image2"))
        self.assertEqual(["icon_n"], action["icon_prompt_constraint"]["slot_ids"])
        self.assertEqual("", injection_blocked_reason(action))

    def test_a_placeholder_row_is_not_injected(self):
        injection = inject_experience(
            [_row({"type": "regen_assets", "slot_ids": ["icon_n"], "transferable": False, "icon_mode": "placeholder"}, apply_at="icon_generation")],
            {"type": "regen_assets", "slot_ids": ["icon_n"]},
            known_ids={"icon_n"},
        )
        self.assertIsNone(injection["action"])
        self.assertIn("Image2", injection["injection"]["reason"])


if __name__ == "__main__":
    unittest.main()
