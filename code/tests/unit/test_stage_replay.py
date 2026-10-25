"""Tests for validated upstream replay in controlled layout experiments."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rfs.cli import build_parser
from rfs.utils import write_json
from rfs.workflows.stage_replay import load_stage_replay, reuse_forecast


class StageReplayTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        paper = root / "paper.pdf"
        reference = root / "reference.png"
        paper.write_bytes(b"paper")
        reference.write_bytes(b"reference")
        source = root / "source"
        source.mkdir()
        archived_reference = source / "inputs" / "positive_references" / "reference.png"
        archived_reference.parent.mkdir(parents=True)
        archived_reference.write_bytes(b"reference")
        import hashlib

        write_json(source / "input_manifest.json", {"positive_references": [str(archived_reference)]})
        write_json(source / "document_model.json", {"source_sha256": hashlib.sha256(b"paper").hexdigest()})
        write_json(source / "method_figure_content.json", {"modules": []})
        write_json(source / "method_extraction_report.json", {"ok": True, "fallback_used": False})
        write_json(source / "relation_explanation_report.json", {"ok": True})
        write_json(source / "scientific_hierarchy_plan.json", {"sections": []})
        write_json(source / "scientific_hierarchy_validation.json", {"fallback_used": False})
        design = {
            "panels": [
                {"id": "a", "bbox_percent": {"x": .05, "y": .1, "w": .4, "h": .8}},
                {"id": "b", "bbox_percent": {"x": .55, "y": .1, "w": .4, "h": .8}},
            ],
            "nodes": [{"id": "n", "semantic_source_id": "n", "panel_id": "a", "bbox_percent": {"x": .1, "y": .2, "w": .2, "h": .2}}],
            "connectors": [],
        }
        write_json(source / "design_plan_raw.json", design)
        write_json(source / "reference_style_profile.json", {"status": "complete", "palette": []})
        write_json(source / "reference_layout_profile.json", {"status": "complete", "reading_order": []})
        write_json(source / "reference_influence_report.json", {"adopted": []})
        write_json(source / "reference_layout_template.json", {"zones": []})
        write_json(source / "reference_layout_template_validation.json", {"ok": True})
        icon = source / "round_00" / "candidate_00" / "assets" / "icon_n" / "selected.png"
        icon.parent.mkdir(parents=True)
        icon.write_bytes(b"icon")
        return source, paper, reference

    def test_a_fork_at_the_paper_reading_has_nothing_to_lend_and_says_so(self):
        # The method contract and the content plan are what the reading produces, and every
        # artifact this loader lends was planned from that reading. A caller that names the
        # reading is asking for the paper to be read again, and the honest answer is that this
        # source has nothing to give it -- not a plan whose every stage says "re-planned", which
        # a caller reads as an inheritance that failed for some other reason.
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            for stage in ("paper_reading", "content_planning"):
                with self.subTest(stage=stage):
                    with self.assertRaisesRegex(ValueError, "inherits nothing"):
                        load_stage_replay(source, paper=paper, reference=reference, fork_before=stage)

    def test_the_forecast_answers_a_reading_fork_with_the_reason(self):
        # The loop asks for the forecast before it spends the iteration, so a fork that can inherit
        # nothing has to read as declined there rather than as a decline discovered afterwards.
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            forecast = reuse_forecast(source, paper=paper, reference=reference, fork_before="paper_reading")

            self.assertFalse(forecast["accepted"])
            self.assertIn("inherits nothing", forecast["declined_reason"])

    def test_replay_validates_hashes_and_frozen_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            replay = load_stage_replay(source, paper=paper, reference=reference)
            self.assertEqual(replay["report"]["status"], "validated")
            self.assertEqual(replay["report"]["frozen_icon_asset_ids"], ["icon_n"])

    def test_a_different_reference_only_replans_the_reference(self):
        # The reference is a visual prior rather than scientific content: a source analysed
        # against another reference is still worth inheriting from, minus that one stage.
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            reference.write_bytes(b"changed")
            replay = load_stage_replay(source, paper=paper, reference=reference)
            self.assertIsNone(replay["reference_profile"])
            self.assertIn("reference_analysis", replay["report"]["replanned_stages"])
            self.assertIn("design_seed", replay["report"]["reused_stages"])
            self.assertIn("method", replay["report"]["reused_stages"])

    def test_cli_accepts_replay_source(self):
        args = build_parser().parse_args([
            "paper-evolve-editable", "--paper", "paper.pdf", "--out", "out", "--replay-upstream-from", "prior",
        ])
        self.assertEqual(args.replay_upstream_from, "prior")

    def test_a_source_without_a_design_seed_can_still_be_reused(self):
        # A retry after a verdict about the layout keeps the reading, the
        # contracts and the Icons but plans the boxes again, so the earlier
        # layout is neither needed nor wanted.
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            (source / "design_plan_raw.json").unlink()
            replay = load_stage_replay(source, paper=paper, reference=reference, freeze_layout=False)
            self.assertIsNone(replay["design_seed"])
            self.assertFalse(replay["report"]["layout_frozen"])
            self.assertNotIn("design_seed", replay["report"]["reused_stages"])

    def test_a_retry_plans_the_text_and_the_colour_again(self):
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            write_json(source / "figure_text_plan.json", {"texts": []})
            write_json(source / "color_theme.json", {"palette": []})
            frozen = load_stage_replay(source, paper=paper, reference=reference)
            retried = load_stage_replay(source, paper=paper, reference=reference, freeze_layout=False)
            self.assertIn("figure_text_plan", frozen["report"]["reused_stages"])
            self.assertNotIn("figure_text_plan", retried["report"]["reused_stages"])
            self.assertNotIn("figure_text_plan", retried)

    def test_a_missing_design_seed_only_replans_the_boxes(self):
        # A source whose seed is gone is not a worthless source: the contract and the reference
        # analysis are still good, so only the boxes are planned again.
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            (source / "design_plan_raw.json").unlink()
            replay = load_stage_replay(source, paper=paper, reference=reference)
            self.assertIsNone(replay["design_seed"])
            self.assertNotIn("design_seed", replay["report"]["reused_stages"])
            self.assertIn("reference_analysis", replay["report"]["reused_stages"])
            self.assertIn("method", replay["report"]["reused_stages"])
            self.assertIn("no design seed", replay["report"]["gaps"][0]["reason"])
            self.assertIn("design_plan_raw.json", replay["report"]["gaps"][0]["reason"])

    def test_a_controlled_replay_carries_a_source_that_had_no_reference(self):
        # A source with no reference figure to read wrote "not_found", and analysing a missing
        # reference is the same empty answer on every run. Inheriting that absence therefore keeps
        # both arms of a pair on one starting line, so a strict replay carries it instead of
        # refusing the experiment. Refusing it is what made every paired validation fail on a
        # deployment that has no reference library: no pair could be run, so no candidate could
        # ever be measured.
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            write_json(source / "reference_style_profile.json", {"status": "not_found"})
            write_json(source / "reference_layout_profile.json", {"status": "not_found"})

            replay = load_stage_replay(source, paper=paper, reference=reference, strict=True)

            self.assertIn("reference_analysis", replay["report"]["reused_stages"])
            self.assertEqual("not_found", replay["reference_profile"]["status"])
            self.assertEqual(["reference_missing"], replay["reference_profile"]["influence"]["rejected"])

    def test_a_controlled_replay_refuses_a_reference_analysis_that_did_not_complete(self):
        # A reference that was there and could not be read is a different thing from one that was
        # never there: re-planning it could land the two arms on different analyses.
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            write_json(source / "reference_style_profile.json", {"status": "failed"})
            write_json(source / "reference_layout_profile.json", {"status": "failed"})
            with self.assertRaisesRegex(ValueError, "complete reference analysis"):
                load_stage_replay(source, paper=paper, reference=reference, strict=True)

    def test_a_controlled_replay_refuses_a_stage_it_would_have_to_plan_again(self):
        # A gap inside the frozen prefix is a hole in the experiment, not a gap to record.
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            (source / "relation_explanation_report.json").unlink()
            with self.assertRaisesRegex(ValueError, "relation_explanation_report.json"):
                load_stage_replay(source, paper=paper, reference=reference, strict=True)

    def test_a_controlled_replay_refuses_a_different_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            reference.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "reference SHA-256"):
                load_stage_replay(source, paper=paper, reference=reference, strict=True)

    def test_a_seed_is_replayed_from_the_response_the_source_arm_built_from(self):
        # The first response is not always the one the figure was built from: a plan repair replaces
        # it with the answer the model gave when it was shown its own contract violations. Replaying
        # the rejected first response would start the pair on a figure the source arm never drew.
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            repaired = {
                "panels": [
                    {"id": "a", "bbox_percent": {"x": .05, "y": .1, "w": .4, "h": .8}},
                    {"id": "b", "bbox_percent": {"x": .55, "y": .1, "w": .4, "h": .8}},
                ],
                "nodes": [{"id": "repaired_n", "semantic_source_id": "n", "panel_id": "a",
                           "bbox_percent": {"x": .1, "y": .2, "w": .2, "h": .2}}],
                "connectors": [],
            }
            write_json(source / "design_plan_seed.json", repaired)

            replay = load_stage_replay(source, paper=paper, reference=reference, strict=True)

            self.assertEqual([node["id"] for node in replay["design_seed"]["nodes"]], ["repaired_n"])
            self.assertIn("design_seed", replay["report"]["reused_stages"])

    def test_a_source_with_only_the_first_response_is_still_replayed_from_it(self):
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))

            replay = load_stage_replay(source, paper=paper, reference=reference, strict=True)

            self.assertEqual([node["id"] for node in replay["design_seed"]["nodes"]], ["n"])
            self.assertEqual(replay["report"]["stage_outcomes"]["design_seed"]["seed_source"], "design_plan_raw.json")

    def test_a_response_the_contract_outgrew_is_read_from_the_plan_it_finished(self):
        # An arm that repaired its own plan left a first response describing boxes it never drew,
        # and the contract has tightened since. The plan the arm finished with is the artifact that
        # still says what it drew, so a controlled replay reads its boxes from there instead of
        # refusing every pair on that source.
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            write_json(source / "design_plan_raw.json", {
                "panels": [
                    {"id": "a", "bbox_percent": {"x": .05, "y": .1, "w": .4, "h": .8}},
                    {"id": "b", "bbox_percent": {"x": .55, "y": .1, "w": .4, "h": .8}},
                ],
                "nodes": [{"id": "n", "semantic_source_id": "n", "panel_id": None, "parent_id": "a",
                           "bbox_percent": {"x": .92, "y": .2, "w": .12, "h": .2}}],
                "connectors": [],
            })
            write_json(source / "design_plan.json", {
                "panels": [
                    {"id": "a", "bbox_percent": {"x": .05, "y": .1, "w": .4, "h": .8}},
                    {"id": "b", "bbox_percent": {"x": .55, "y": .1, "w": .4, "h": .8}},
                ],
                "nodes": [{"id": "n", "semantic_source_id": "n", "panel_id": None, "parent_id": "a",
                           "bbox_percent": {"x": .88, "y": .95, "w": .08, "h": .04}}],
                "connectors": [],
            })

            replay = load_stage_replay(source, paper=paper, reference=reference, strict=True)

            self.assertEqual(replay["design_seed"]["nodes"][0]["bbox_percent"], {"x": .88, "y": .95, "w": .08, "h": .04})
            outcome = replay["report"]["stage_outcomes"]["design_seed"]
            self.assertEqual(outcome["seed_source"], "design_plan.json")
            self.assertIn("no longer validates", outcome["reason"])

    def test_a_controlled_replay_refuses_a_source_with_no_seed_at_all(self):
        # Falling back to the finished plan is the last resort, not a licence: a source that kept
        # no seed and no plan has nothing a pair could start from, so the replay still refuses.
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            (source / "design_plan_raw.json").unlink()
            with self.assertRaisesRegex(ValueError, "design_plan_raw.json"):
                load_stage_replay(source, paper=paper, reference=reference, strict=True)

    def test_a_source_with_another_paper_is_still_refused(self):
        # The paper is the one thing a replay cannot compromise on: its contracts describe a
        # different document, so nothing downstream could be trusted.
        with tempfile.TemporaryDirectory() as temp:
            source, paper, reference = self._fixture(Path(temp))
            other = Path(temp) / "other.pdf"
            other.write_bytes(b"another paper")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_stage_replay(source, paper=other, reference=reference)


if __name__ == "__main__":
    unittest.main()
