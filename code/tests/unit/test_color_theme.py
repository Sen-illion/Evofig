import unittest
from unittest.mock import patch

from rfs.contracts.color_theme import normalize_color_theme, resolve_role_style, validate_color_theme
from rfs.planning.color_theme_planner import plan_color_theme_vlm
from rfs.planning.panel_style import build_panel_style_plan


THEME = {
    "theme_name": "academic_teal",
    "theme_intent": "restrained",
    "canvas_background": "#F8FAFC",
    "text": {"primary": "#193247", "secondary": "#526676", "muted": "#71808B", "inverse": "#FFFFFF"},
    "connectors": {"primary": "#54758A", "secondary": "#7B929F", "feedback": "#A17B55"},
    "panels": {"p1": {"fill": "#EDF3F7", "border": "#477FA8", "header": "#315C78", "header_text": "#FFFFFF", "body_text": "#193247"}},
    "roles": {"input": {"fill": "#E3EFF7", "border": "#477FA8", "text": "#193247"}, "module": {"fill": "#E8F3F0", "border": "#4D8B7B", "text": "#193247"}, "output": {"fill": "#F5EEE7", "border": "#B07A4D", "text": "#193247"}},
}


class ColorThemeTests(unittest.TestCase):
    def test_valid_theme_drives_panel_style_without_palette_rewrite(self):
        theme = normalize_color_theme(THEME)
        self.assertTrue(validate_color_theme(theme, {"p1"})["ok"])
        style = build_panel_style_plan([{"id": "p1"}], color_theme=theme)
        self.assertEqual(style["panels"][0]["fill_color_hex"], "#EDF3F7")
        self.assertEqual(style["panels"][0]["border_color_hex"], "#477FA8")
        self.assertEqual(style["palette_decision_trace"][0]["reason"], "color_theme_planner")

    def test_pure_red_panel_border_is_invalid(self):
        theme = normalize_color_theme({**THEME, "panels": {"p1": {**THEME["panels"]["p1"], "border": "#FF0000"}}})
        self.assertFalse(validate_color_theme(theme, {"p1"})["ok"])

    @patch("rfs.planning.color_theme_planner.call_vlm_json", return_value=THEME)
    def test_vlm_theme_is_validated_and_returned(self, mocked):
        theme, report = plan_color_theme_vlm({}, {"panels": [{"id": "p1"}]}, {"items": []})
        self.assertTrue(report["ok"])
        self.assertFalse(report["fallback_used"])
        self.assertEqual(theme["panels"]["p1"]["fill"], "#EDF3F7")
        mocked.assert_called_once()

    @patch("rfs.planning.color_theme_planner.call_vlm_json", return_value={"canvas_background": "#F8F9FA", "text": THEME["text"], "connectors": THEME["connectors"], "panel": {"fill": "#FFFFFF", "border": "#CBD5E1", "header": "#F1F5F9", "header_text": "#193247", "body_text": "#334155"}, "semantic_roles": {"input": "#009E73", "module": "#56B4E9", "output": "#E69F00"}, "usage_policy": "restrained", "provenance": "VLM planned"})
    def test_generic_vlm_theme_is_bound_to_concrete_panels(self, mocked):
        design = {"panels": [{"id": "input_panel", "role": "input"}, {"id": "result_panel", "role": "output"}]}
        theme, report = plan_color_theme_vlm({}, design, {"items": []})
        self.assertTrue(report["ok"])
        self.assertFalse(report["fallback_used"])
        self.assertEqual(theme["panels"]["input_panel"]["border"], "#009E73")
        self.assertEqual(theme["panels"]["result_panel"]["border"], "#E69F00")

    @patch("rfs.planning.color_theme_planner.call_vlm_json", return_value={"canvas_background": "#F8F9FA", "text": THEME["text"], "connectors": THEME["connectors"], "panels": {"input": {"fill": "#F8FAFC", "border": "#00785E", "header": "#00785E", "header_text": "#FFFFFF", "body_text": "#193247"}, "module": {"fill": "#F1F5F9", "border": "#26739C", "header": "#26739C", "header_text": "#FFFFFF", "body_text": "#193247"}, "output": {"fill": "#F4FBF6", "border": "#A65D00", "header": "#A65D00", "header_text": "#FFFFFF", "body_text": "#193247"}}, "accents": {"primary": "#26739C"}, "usage_policy": "restrained", "provenance": "VLM planned"})
    def test_role_panel_theme_is_expanded_to_actual_panel_ids(self, mocked):
        design = {"panels": [{"id": "panel_input", "role": "input"}, {"id": "panel_methods", "role": "module"}, {"id": "panel_results", "role": "output"}]}
        theme, report = plan_color_theme_vlm({}, design, {"items": []})
        self.assertTrue(report["ok"])
        self.assertFalse(report["fallback_used"])
        self.assertEqual(theme["panels"]["panel_methods"]["border"], "#26739C")
        self.assertEqual(theme["roles"]["output"]["border"], "#A65D00")

    @patch("rfs.planning.color_theme_planner.call_vlm_json", return_value={"canvas_background": "#FFFFFF", "text": THEME["text"], "connectors": THEME["connectors"], "panel": {"fill": "#F8FAFC", "border": "#CFD8DC", "header": "#ECEFF1", "header_text": "#193247", "body_text": "#526676"}, "semantic_roles": THEME["roles"], "accents": ["#0288D1"], "usage_policy": "restrained", "provenance": "VLM planned"})
    def test_structured_semantic_roles_are_preserved(self, mocked):
        design = {"panels": [{"id": "panel_input", "role": "input"}, {"id": "panel_results", "role": "output"}]}
        theme, report = plan_color_theme_vlm({}, design, {"items": []})
        self.assertTrue(report["ok"])
        self.assertFalse(report["fallback_used"])
        self.assertEqual(theme["roles"]["input"]["border"], "#477FA8")
        self.assertEqual(theme["panels"]["panel_results"]["border"], "#B07A4D")

    @patch(
        "rfs.planning.color_theme_planner.call_vlm_json",
        return_value={
            **THEME,
            "palette_strategy": {
                "dominant_family": "blue",
                "supporting_families": ["amber"],
                "harmony": "analogous",
                "rationale": "Separate generation from evaluation.",
            },
            "role_mapping": {"generator": "processing", "training_objective": "evaluation"},
            "roles": {
                "processing": {"fill": "#E7F5FF", "border": "#4D8BB8", "text": "#193247"},
                "evaluation": {"fill": "#FFF4E6", "border": "#C98232", "text": "#5B3714"},
            },
            "self_check": {"all_panel_ids_covered": True, "all_node_roles_covered": True},
        },
    )
    def test_actual_design_roles_are_mapped_and_preserved(self, mocked):
        design = {
            "panels": [{"id": "p1", "role": "pipeline"}],
            "nodes": [{"id": "g", "role": "generator"}, {"id": "e", "role": "training_objective"}],
        }
        theme, report = plan_color_theme_vlm({}, design, {"items": []})
        self.assertTrue(report["ok"])
        self.assertFalse(report["fallback_used"])
        self.assertEqual(resolve_role_style(theme, "generator")["border"], "#4D8BB8")
        self.assertEqual(resolve_role_style(theme, "training_objective")["border"], "#C98232")
        prompt = mocked.call_args.args[0]
        self.assertIn("Do not assume a fixed input/module/output role list", prompt)
        self.assertIn('"role":"generator"', prompt)

    def test_missing_actual_role_mapping_is_invalid(self):
        report = validate_color_theme(normalize_color_theme(THEME), {"p1"}, {"generator"})
        self.assertFalse(report["ok"])
        self.assertIn("missing node role theme: generator", report["errors"])


if __name__ == "__main__":
    unittest.main()
