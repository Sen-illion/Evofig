"""The paper title keeps its lane: nothing is drawn over it, and its ink fits."""

from __future__ import annotations

import unittest

from rfs.composition.overlay_geometry import keep_panels_below_title_band
from rfs.composition.text_layout import fit_text_program_globally
from rfs.evaluation.geometry_quality import evaluate_geometry_quality


def _title_block(title: str = "A short title") -> dict:
    return {
        "title": title,
        "subtitle": "Editable paper framework",
        "bbox_percent": {"x": .04, "y": .015, "w": .92, "h": .08},
        "title_font_size": 20,
        "subtitle_font_size": 9,
    }


def _panel(panel_id: str, box: dict) -> dict:
    return {"id": panel_id, "title": panel_id.title(), "role": "stage", "bbox_percent": dict(box)}


def _program(title: str) -> dict:
    return {
        "canvas": {"width_in": 13.333, "height_in": 7.5},
        "style": {"font_family": "Arial"},
        "title_block": _title_block(title),
        "panels": [], "cards": [], "slots": [], "semantic_nodes": [],
        "text_program": {"items": []},
    }


class PanelVersusTitleBandTests(unittest.TestCase):
    def test_a_panel_reaching_into_the_title_lane_is_shortened_from_the_top(self) -> None:
        panels = [_panel("stage_a", {"x": .02, "y": .05, "w": .56, "h": .90})]
        kept, report = keep_panels_below_title_band(panels, _title_block())
        box = kept[0]["bbox_percent"]
        self.assertEqual(report["status"], "adjusted")
        self.assertEqual(report["adjustment_count"], 1)
        self.assertEqual(report["adjustments"][0]["panel_id"], "stage_a")
        self.assertGreaterEqual(box["y"], .095)
        self.assertAlmostEqual(box["y"] + box["h"], .95, places=4)
        self.assertEqual(box["x"], .02)
        self.assertEqual(panels[0]["bbox_percent"]["y"], .05)

    def test_a_panel_already_below_the_title_lane_is_untouched(self) -> None:
        panels = [_panel("stage_a", {"x": .02, "y": .11, "w": .56, "h": .85})]
        kept, report = keep_panels_below_title_band(panels, _title_block())
        self.assertEqual(report["status"], "pass")
        self.assertEqual(kept[0]["bbox_percent"], {"x": .02, "y": .11, "w": .56, "h": .85})

    def test_a_canvas_without_a_title_keeps_its_panels(self) -> None:
        panels = [_panel("stage_a", {"x": .02, "y": .01, "w": .56, "h": .98})]
        kept, report = keep_panels_below_title_band(panels, None)
        self.assertEqual(report["status"], "not_applicable")
        self.assertEqual(kept[0]["bbox_percent"]["y"], .01)


class TitleTextFitTests(unittest.TestCase):
    def test_a_long_title_is_fitted_into_the_box_the_compiler_draws(self) -> None:
        fitted, report = fit_text_program_globally(
            _program("AM EM GYM : INTERACTIVE MEMORY BENCHMARKING FOR ASSISTANTS IN LONG - HORIZON CONVERSATIONS")
        )
        entry = next(item for item in report["items"] if item["text_id"] == "title_block")
        self.assertEqual(entry["font_size_role"], "paper_title")
        self.assertLessEqual(entry["line_count"], 2)
        self.assertFalse(entry["overflow"])
        self.assertLessEqual(fitted["title_block"]["title_font_size"], 20)
        self.assertEqual(fitted["title_block"]["measured_line_count"], entry["line_count"])
        self.assertEqual(report["status"], "pass")

    def test_a_short_title_keeps_its_size_on_one_line(self) -> None:
        fitted, report = fit_text_program_globally(_program("Memory Assistants"))
        entry = next(item for item in report["items"] if item["text_id"] == "title_block")
        self.assertEqual(fitted["title_block"]["title_font_size"], 20)
        self.assertEqual(entry["line_count"], 1)
        self.assertFalse(entry["overflow"])

    def test_a_canvas_without_a_title_block_is_reported_without_one(self) -> None:
        program = _program("Memory Assistants")
        program.pop("title_block")
        _, report = fit_text_program_globally(program)
        self.assertFalse(any(item["text_id"] == "title_block" for item in report["items"]))


class TitleOverlapReportTests(unittest.TestCase):
    def _report(self, panel_y: float) -> dict:
        return evaluate_geometry_quality({
            "canvas": {"width_in": 13.333, "height_in": 7.5},
            "title_block": _title_block(),
            "panels": [_panel("stage_a", {"x": .02, "y": panel_y, "w": .56, "h": .5})],
            "semantic_nodes": [], "slots": [], "arrows": [], "text_program": {"items": []},
        })

    def test_a_panel_over_the_title_is_reported_and_blocks(self) -> None:
        report = self._report(.05)
        self.assertEqual(report["title_overlap_count"], 1)
        self.assertEqual(report["title_overlaps"][0]["object_type"], "panel")
        self.assertEqual(report["title_overlaps"][0]["object_id"], "stage_a")
        self.assertEqual(report["status"], "blocked")

    def test_a_clean_canvas_reports_no_title_overlap(self) -> None:
        report = self._report(.11)
        self.assertEqual(report["title_overlap_count"], 0)
        self.assertEqual(report["status"], "pass")


if __name__ == "__main__":
    unittest.main()
