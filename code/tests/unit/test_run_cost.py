import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rfs.workflows.run_cost import summarize_run_cost


class RunCostTests(unittest.TestCase):
    def test_text_and_image_usage_are_priced_apart(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "review.json").write_text(json.dumps({"provider": {"usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}}}), encoding="utf-8")
            (root / "asset_generation_report.json").write_text(
                json.dumps({"api_requests_attempted": 2, "assets": [{"usage": {"prompt_tokens": 20, "completion_tokens": 1290, "total_tokens": 1310, "image_count": 1}}, {"usage": {"prompt_tokens": 20, "completion_tokens": 1290, "total_tokens": 1310, "image_count": 1}}]}),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=False):
                # The icon model decides the image price; a shell that happens to export one
                # must not change what this arithmetic test is measuring.
                os.environ.pop("RFS_IMAGE_MODEL", None)
                os.environ.pop("IMAGE_MODEL", None)
                report = summarize_run_cost(root, elapsed_seconds=12.5)
        self.assertEqual(report["text"]["prompt_tokens"], 1000)
        self.assertEqual(report["image"]["images_billed"], 2)
        self.assertEqual(report["image_usd"], 0.0672)
        self.assertEqual(report["elapsed_seconds"], 12.5)
        self.assertGreater(report["generation_usd"], report["image_usd"])

    def test_a_round_report_is_not_priced_twice(self):
        """The deliverable's report sits at the run root and again inside its round."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drawn = {"api_requests_attempted": 3, "assets": [{"usage": {"total_tokens": 1310, "image_count": 1}}] * 3}
            reused = {"api_requests_attempted": 0, "assets": [{"status": "reused", "api_requests_attempted": 0}] * 3}
            for round_name, report in (("round_00", drawn), ("round_01", reused)):
                target = root / round_name / "candidate_00"
                target.mkdir(parents=True)
                (target / "asset_generation_report.json").write_text(json.dumps(report), encoding="utf-8")
            (root / "asset_generation_report.json").write_text(json.dumps(drawn), encoding="utf-8")
            report = summarize_run_cost(root)
        self.assertEqual(report["image"]["api_requests_attempted"], 3)
        self.assertEqual(report["image"]["images_billed"], 3)
        self.assertEqual(report["image"]["asset_reports_counted"], 2)
        self.assertEqual(report["image"]["duplicate_asset_reports_skipped"], 1)
        self.assertEqual(report["image_usd"], 0.1008)

    def test_a_lone_run_root_report_still_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "asset_generation_report.json").write_text(
                json.dumps({"api_requests_attempted": 4, "assets": [{"usage": {"image_count": 1}}]}),
                encoding="utf-8",
            )
            report = summarize_run_cost(root)
        self.assertEqual(report["image"]["api_requests_attempted"], 4)
        self.assertEqual(report["image"]["asset_reports_counted"], 1)
        self.assertEqual(report["image"]["duplicate_asset_reports_skipped"], 0)


class JudgePriceTests(unittest.TestCase):
    """The nine-item ruler is billed apart from generation but dated in the same snapshot."""

    def test_a_scoring_call_is_priced_from_the_same_snapshot(self):
        from rfs.workflows.run_cost import PRICE_SNAPSHOT, judge_usd

        self.assertEqual(PRICE_SNAPSHOT["judge_model"], "gemini-3.1-flash-lite")
        self.assertEqual(judge_usd(1_000_000, 0), 0.25)
        self.assertEqual(judge_usd(0, 1_000_000), 1.50)
        self.assertEqual(judge_usd(500, 500), 0.000875)
        self.assertEqual(judge_usd(None, None), 0.0)


class IconPriceTests(unittest.TestCase):
    """Icons are a paper's largest cost line, so the ledger prices the model the row used."""

    def draw(self, root):
        (root / "asset_generation_report.json").write_text(
            json.dumps({"api_requests_attempted": 10, "assets": [{"usage": {"image_count": 10}}]}),
            encoding="utf-8",
        )

    def test_the_cheap_icon_model_is_priced_at_its_own_rate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.draw(root)
            with mock.patch.dict(os.environ, {"RFS_IMAGE_MODEL": "z-image-turbo"}):
                report = summarize_run_cost(root)
        self.assertEqual(report["price_snapshot"]["image_model"], "z-image-turbo")
        self.assertEqual(report["image_usd"], 0.075)

    def test_a_row_with_no_named_only_pays_for_the_paper_it_draws(self):
        """The cheap model has to show up as cheap, not as the model the ledger was written on."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.draw(root)
            with mock.patch.dict(os.environ, {"RFS_IMAGE_MODEL": "z-image-turbo"}):
                cheap = summarize_run_cost(root)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("RFS_IMAGE_MODEL", None)
                os.environ.pop("IMAGE_MODEL", None)
                unnamed = summarize_run_cost(root)
        self.assertLess(cheap["image_usd"], unnamed["image_usd"])
        self.assertEqual(unnamed["price_snapshot"]["image_model"], "gemini-3.1-flash-lite-image")
        self.assertIn("no icon model was named", unnamed["price_snapshot"]["image_price_basis"])

    def test_an_unknown_icon_model_falls_back_and_says_so(self):
        from rfs.workflows.run_cost import image_rates

        rates = image_rates("some-model-nobody-priced")
        self.assertEqual(rates["image_usd_per_image"], 0.0336)
        self.assertIn("no icon model was named", rates["image_price_basis"])


class RoundModelCallLedgerTests(unittest.TestCase):
    """A round's ledger names the model calls it made, not only how many it made."""

    def _progress(self, root, events):
        (root / "progress.jsonl").write_text("\n".join(json.dumps(item) for item in events) + "\n", encoding="utf-8")

    def test_the_round_carries_each_call_it_made(self):
        from rfs.workflows.run_cost import model_calls_from_progress, round_cost_record

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._progress(root, [
                {"seq": 1, "elapsed_seconds": 1.0, "stage": "connector_review", "status": "started", "round": 1},
                {"seq": 2, "elapsed_seconds": 191.0, "stage": "vlm_call", "status": "completed", "call": "connector_review", "round": 1, "provider": {"success": True, "elapsed_seconds": 190.0, "prompt_tokens": 8721, "completion_tokens": 17924, "image_count": 1}},
            ])
            calls = model_calls_from_progress(root)
            record = round_cost_record(round_index=1, model_calls=calls)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["call"], "connector_review")
        self.assertEqual(calls[0]["elapsed_seconds"], 190.0)
        self.assertEqual(calls[0]["completion_tokens"], 17924)
        self.assertEqual(record["model_call_count"], 1)
        self.assertEqual(record["model_seconds"], 190.0)
        self.assertEqual(record["model_calls"][0]["call"], "connector_review")

    def test_a_carrier_event_never_closes_the_stage_it_ran_in(self):
        from rfs.workflows.run_cost import stage_seconds_from_progress

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._progress(root, [
                {"seq": 1, "elapsed_seconds": 10.0, "stage": "ppt_compilation", "status": "started"},
                {"seq": 2, "elapsed_seconds": 200.0, "stage": "vlm_call", "status": "completed", "call": "connector_review", "provider": {"success": True, "elapsed_seconds": 190.0}},
                {"seq": 3, "elapsed_seconds": 205.0, "stage": "ppt_compilation", "status": "completed"},
            ])
            seconds = stage_seconds_from_progress(root)

        self.assertEqual(seconds["ppt_compilation"], 195.0)
        self.assertNotIn("vlm_call", seconds)

    def test_an_empty_provider_block_is_not_a_call(self):
        from rfs.workflows.run_cost import model_calls_from_progress

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._progress(root, [
                {"seq": 1, "elapsed_seconds": 1.0, "stage": "vlm_call", "status": "completed", "call": "color_review", "provider": {}},
                {"seq": 2, "elapsed_seconds": 2.0, "stage": "visual_review", "status": "completed"},
            ])
            self.assertEqual(model_calls_from_progress(root), [])
