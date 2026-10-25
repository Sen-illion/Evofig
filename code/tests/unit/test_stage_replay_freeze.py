"""Stage replay freezes the color theme and figure text plan when present."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rfs.workflows.stage_replay import load_stage_replay, reuse_forecast

PAPER_BYTES = b"%PDF-1.4 stage replay fixture\n"
PNG_BYTES = b"\x89PNG\r\n\x1a\n"

DESIGN = {
    "canvas": {"aspect_ratio": "16:9", "margin": 0.02},
    "topology": "unknown",
    "panels": [
        {"id": "stage_one", "title": "Phase One", "role": "stage", "bbox_percent": {"x": 0.04, "y": 0.15, "w": 0.3, "h": 0.75}},
        {"id": "stage_two", "title": "Phase Two", "role": "stage", "bbox_percent": {"x": 0.38, "y": 0.15, "w": 0.3, "h": 0.75}},
    ],
    "nodes": [
        {
            "id": "node_alpha",
            "label": "Alpha",
            "semantic_source_id": "obj_alpha",
            "panel_id": "stage_one",
            "bbox_percent": {"x": 0.08, "y": 0.3, "w": 0.12, "h": 0.2},
        },
        {
            "id": "node_beta",
            "label": "Beta",
            "semantic_source_id": "obj_beta",
            "panel_id": "stage_two",
            "bbox_percent": {"x": 0.42, "y": 0.3, "w": 0.12, "h": 0.2},
        },
    ],
    "connectors": [],
}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_source(root: Path, *, with_optional: bool) -> tuple[Path, Path]:
    """Lay out the smallest replay source the loader accepts."""

    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    paper = root / "paper.pdf"
    paper.write_bytes(PAPER_BYTES)
    _write(source / "input_manifest.json", {"positive_references": []})
    _write(source / "document_model.json", {"source_sha256": hashlib.sha256(PAPER_BYTES).hexdigest()})
    _write(source / "method_figure_content.json", {"objects": []})
    _write(source / "method_extraction_report.json", {"ok": True, "fallback_used": False})
    _write(source / "scientific_hierarchy_plan.json", {"sections": []})
    _write(source / "scientific_hierarchy_validation.json", {"fallback_used": False})
    _write(source / "design_plan_raw.json", DESIGN)
    for name in ("reference_style_profile.json", "reference_layout_profile.json"):
        _write(source / name, {"status": "complete"})
    for name in (
        "reference_influence_report.json",
        "reference_layout_template.json",
        "reference_layout_template_validation.json",
        "relation_explanation_report.json",
    ):
        _write(source / name, {})
    icon = source / "round_00" / "candidate_00" / "assets" / "icon_alpha" / "selected.png"
    icon.parent.mkdir(parents=True, exist_ok=True)
    icon.write_bytes(PNG_BYTES)
    if with_optional:
        _write(source / "color_theme.json", {"contract": "rfs.color_theme", "theme_name": "Frozen Palette"})
        _write(source / "color_theme_validation.json", {"ok": True})
        _write(source / "figure_text_plan.json", {"items": [{"node_id": "node_alpha", "label": "Alpha"}]})
        _write(source / "figure_text_plan_validation.json", {"ok": True})
    return source, paper


class StageReplayFreezeTests(unittest.TestCase):
    def test_theme_and_text_plan_are_frozen_when_the_source_has_them(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, paper = _build_source(Path(raw), with_optional=True)
            replay = load_stage_replay(source, paper=paper, reference=None)
        stages = replay["report"]["reused_stages"]
        self.assertIn("color_theme", stages)
        self.assertIn("figure_text_plan", stages)
        self.assertEqual(replay["color_theme"]["theme_name"], "Frozen Palette")
        self.assertEqual(len(replay["figure_text_plan"]["items"]), 1)

    def test_a_source_without_them_still_replays(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, paper = _build_source(Path(raw), with_optional=False)
            replay = load_stage_replay(source, paper=paper, reference=None)
        stages = replay["report"]["reused_stages"]
        self.assertNotIn("color_theme", stages)
        self.assertNotIn("figure_text_plan", stages)
        self.assertNotIn("color_theme", replay)


class StageReplayPartialTests(unittest.TestCase):
    """One unusable stage re-plans that stage; it does not void the whole source."""

    def test_a_source_without_a_reference_still_inherits_the_design(self) -> None:
        # A figure can be produced with no reference at all. Analysing a missing reference is the
        # same empty answer on every run, so that absence is carried over too: the reference, the
        # design seed and the boxes all come from the source and nothing is planned again.
        with tempfile.TemporaryDirectory() as raw:
            source, paper = _build_source(Path(raw), with_optional=True)
            for name in ("reference_style_profile.json", "reference_layout_profile.json"):
                _write(source / name, {"status": "not_found"})
            replay = load_stage_replay(source, paper=paper, reference=None)
        self.assertEqual(replay["reference_profile"]["status"], "not_found")
        self.assertIn("reference_analysis", replay["report"]["reused_stages"])
        self.assertNotIn("reference_analysis", replay["report"]["replanned_stages"])
        self.assertIn("design_seed", replay["report"]["reused_stages"])
        gaps = {item["stage"]: item["reason"] for item in replay["report"]["gaps"]}
        self.assertNotIn("reference_analysis", gaps)

    def test_an_invalid_design_seed_replans_only_the_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, paper = _build_source(Path(raw), with_optional=True)
            _write(source / "design_plan_raw.json", {"topology": "unknown", "panels": [], "nodes": [], "connectors": []})
            replay = load_stage_replay(source, paper=paper, reference=None)
        self.assertIsNone(replay["design_seed"])
        self.assertFalse(replay["report"]["layout_frozen"])
        self.assertIn("design_seed", replay["report"]["replanned_stages"])
        self.assertIn("figure_text_plan", replay["report"]["reused_stages"])
        self.assertIn("icons", replay["report"]["reused_stages"])

    def test_a_missing_reference_template_is_a_note_not_a_decline(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, paper = _build_source(Path(raw), with_optional=True)
            for name in ("reference_influence_report.json", "reference_layout_template.json", "reference_layout_template_validation.json"):
                (source / name).unlink()
            replay = load_stage_replay(source, paper=paper, reference=None)
        self.assertIn("reference_analysis", replay["report"]["reused_stages"])
        notes = replay["report"]["stage_outcomes"]["reference_analysis"].get("notes") or []
        self.assertEqual(len(notes), 3)


class ReuseForecastTests(unittest.TestCase):
    """The loop asks what a source would give before it spends the run on it."""

    def test_the_forecast_names_every_stage_before_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, paper = _build_source(Path(raw), with_optional=False)
            forecast = reuse_forecast(source, paper=paper, fork_before="repair")
        self.assertTrue(forecast["accepted"])
        self.assertEqual(forecast["declined_reason"], "")
        self.assertIn("design_seed", forecast["reused_stages"])
        self.assertIn("color_theme", forecast["replanned_stages"])
        self.assertIn("figure_text_plan", forecast["replanned_stages"])

    def test_the_forecast_declines_a_source_for_another_paper(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, paper = _build_source(Path(raw), with_optional=True)
            other = Path(raw) / "other.pdf"
            other.write_bytes(b"%PDF-1.4 another paper\n")
            forecast = reuse_forecast(source, paper=other, fork_before="repair")
        self.assertFalse(forecast["accepted"])
        self.assertIn("SHA-256", forecast["declined_reason"])
        self.assertEqual(forecast["reused_stages"], [])

    def test_the_forecast_reports_a_source_that_is_not_there(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            paper = Path(raw) / "paper.pdf"
            paper.write_bytes(PAPER_BYTES)
            forecast = reuse_forecast(Path(raw) / "missing", paper=paper, fork_before="repair")
        self.assertFalse(forecast["accepted"])
        self.assertIn("does not exist", forecast["declined_reason"])


class StageReplayForkTests(unittest.TestCase):
    """Where a pair forks decides which planning artifacts both arms share."""

    def _replay(self, root: Path, fork_before: str) -> dict:
        source, paper = _build_source(root, with_optional=True)
        return load_stage_replay(source, paper=paper, reference=None, fork_before=fork_before)

    def test_a_fork_at_repair_freezes_everything_a_full_freeze_freezes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            replay = self._replay(Path(raw), "repair")
        self.assertEqual(replay["report"]["fork_before"], "repair")
        self.assertEqual(sorted(replay["report"]["frozen_stages"]), ["design", "icons", "text_plan", "theme"])
        self.assertIn("design_seed", replay["report"]["reused_stages"])
        self.assertIn("icons", replay["report"]["reused_stages"])
        self.assertEqual(replay["color_theme"]["theme_name"], "Frozen Palette")

    def test_a_fork_at_icon_generation_replans_the_icons_and_keeps_the_rest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            replay = self._replay(Path(raw), "icon_generation")
        self.assertEqual(sorted(replay["report"]["frozen_stages"]), ["design", "text_plan", "theme"])
        self.assertNotIn("icons", replay["report"]["reused_stages"])
        self.assertEqual(replay["color_theme"]["theme_name"], "Frozen Palette")

    def test_a_fork_at_theme_planning_replans_the_theme(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            replay = self._replay(Path(raw), "theme_planning")
        self.assertEqual(sorted(replay["report"]["frozen_stages"]), ["design"])
        self.assertNotIn("color_theme", replay["report"]["reused_stages"])
        self.assertNotIn("figure_text_plan", replay["report"]["reused_stages"])
        self.assertNotIn("icons", replay["report"]["reused_stages"])
        self.assertNotIn("color_theme", replay)

    def test_a_fork_at_segmentation_replans_the_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            replay = self._replay(Path(raw), "segmentation")
        self.assertEqual(replay["report"]["frozen_stages"], [])
        self.assertFalse(replay["report"]["layout_frozen"])
        self.assertIsNone(replay["design_seed"])

    def test_an_unknown_fork_stage_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, paper = _build_source(Path(raw), with_optional=True)
            with self.assertRaises(ValueError):
                load_stage_replay(source, paper=paper, reference=None, fork_before="whenever")


if __name__ == "__main__":
    unittest.main()
