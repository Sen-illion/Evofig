import unittest

from rfs.evaluation.color_quality import evaluate_color_quality


class ColorQualityTests(unittest.TestCase):
    def test_feedback_connector_uses_feedback_theme_color(self):
        theme = {"panels": {}, "connectors": {"primary": "#54758A", "feedback": "#A05050"}, "usage_policy": {"max_distinct_hues": 4}}
        program = {"style": {"color_theme": theme, "panel_styles": {}}, "panels": [], "arrows": [{"id": "feedback", "relation_type": "feedback", "stroke_color": "#A05050"}]}
        report = evaluate_color_quality(program)
        self.assertEqual(report["metrics"]["connector_mismatch_count"], 0)

    def test_theme_propagation_and_connector_consistency(self):
        theme = {"panels": {"p": {"fill": "#EDF3F7", "border": "#477FA8", "header": "#315C78", "body_text": "#193247"}}, "connectors": {"primary": "#54758A"}, "usage_policy": {"max_distinct_hues": 4}}
        program = {"style": {"color_theme": theme, "color_theme_source": "vlm", "panel_styles": {"p": {"fill_color": "#EDF3F7", "stroke_color": "#477FA8", "header_color": "#315C78", "text_color": "#193247"}}}, "panels": [], "arrows": [{"id": "a", "stroke_color": "#54758A"}]}
        report = evaluate_color_quality(program)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["metrics"]["connector_mismatch_count"], 0)

    def test_a_pale_panel_fill_is_not_flagged_as_oversaturated(self):
        # HLS reads a near-white tint as strongly saturated (#F0FDFA measures 0.77), so a
        # pale surface must not be reported as a large surface that is too saturated.
        theme = {"panels": {"p": {"fill": "#F0FDFA", "border": "#4D8B7B", "header": "#315C78", "body_text": "#193247"}}, "connectors": {"primary": "#54758A"}, "usage_policy": {"max_distinct_hues": 4}}
        program = {"style": {"color_theme": theme, "color_theme_source": "vlm", "panel_styles": {"p": {"fill_color": "#F0FDFA", "stroke_color": "#4D8B7B", "header_color": "#315C78", "text_color": "#193247"}}}, "panels": [], "arrows": [{"id": "a", "stroke_color": "#54758A"}]}
        report = evaluate_color_quality(program)
        self.assertFalse(any("saturated" in issue["reason"] for issue in report["issues"]))
        self.assertEqual(report["status"], "pass")

    def test_theme_conflict_is_reported(self):
        program = {"style": {"color_theme": {"panels": {"p": {"fill": "#FFFFFF", "border": "#477FA8", "header": "#315C78", "body_text": "#193247"}}, "connectors": {"primary": "#54758A"}}, "panel_styles": {"p": {"fill_color": "#FFFFFF", "stroke_color": "#FF0000", "header_color": "#315C78", "text_color": "#193247"}}}, "panels": [], "arrows": []}
        report = evaluate_color_quality(program)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["issue_count"])
