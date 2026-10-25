"""Tests for authoritative color propagation to editable objects."""

from __future__ import annotations

import unittest

from rfs.composition.color_application import apply_color_theme_to_program


class ColorApplicationTests(unittest.TestCase):
    def test_theme_reaches_panels_nodes_connectors_and_text(self):
        theme = {
            "canvas_background": "#FAFAFA",
            "text": {"primary": "#111111", "secondary": "#333333"},
            "connectors": {"primary": "#123456", "secondary": "#456789", "feedback": "#A05050"},
            "panels": {"p1": {"fill": "#EEF6F4", "border": "#287060", "header": "#287060", "header_text": "#FFFFFF", "body_text": "#111111"}},
            "role_mapping": {"module": "process"},
            "roles": {"process": {"fill": "#E0F2EE", "border": "#287060", "text": "#102B25"}},
            "provenance": {"source": "vlm"},
        }
        plan = {"color_theme": theme, "panels": [{"panel_id": "p1", "fill_color_hex": "#EEF6F4", "border_color_hex": "#287060", "header_color_hex": "#287060", "header_text_color_hex": "#FFFFFF", "text_color_hex": "#111111"}]}
        program = {
            "canvas": {}, "panels": [{"id": "p1"}],
            "semantic_nodes": [{"id": "n1", "semantic_source_id": "source-n1", "panel_id": "p1", "role": "module"}],
            "arrows": [{"id": "a1", "relation_type": "data_flow"}, {"id": "a2", "relation_type": "feedback"}],
            "text_program": {"items": [
                {"id": "t1", "text_role": "panel_title", "target_id": "p1"},
                {"id": "t2", "text_role": "node_label", "target_id": "source-n1"},
                {"id": "t3", "text_role": "connector_label"},
            ]},
        }
        result = apply_color_theme_to_program(program, plan)
        self.assertEqual(result["style"]["panel_styles"]["p1"]["header_color"], "#287060")
        self.assertEqual(result["semantic_nodes"][0]["fill_color"], "#E0F2EE")
        self.assertEqual(result["arrows"][0]["stroke_color"], "#123456")
        self.assertEqual(result["arrows"][1]["stroke_color"], "#A05050")
        self.assertEqual([item["color_hex"] for item in result["text_program"]["items"]], ["#FFFFFF", "#102B25", "#333333"])


if __name__ == "__main__":
    unittest.main()
