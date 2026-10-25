"""Tests for the opt-in Paper2Poster aesthetic judge and its agreement report."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rfs.evaluation.aesthetic_judge import agreement_report, judge_aesthetics, parse_aesthetic_scores


class AestheticParseTests(unittest.TestCase):
    def test_three_integers_are_accepted(self):
        parsed = parse_aesthetic_scores({"element_quality": 4, "layout_balance": 3, "engagement": 2})
        self.assertEqual({"element_quality": 4, "layout_balance": 3, "engagement": 2}, parsed)

    def test_a_float_or_an_out_of_range_score_rejects_the_whole_response(self):
        self.assertIsNone(parse_aesthetic_scores({"element_quality": 4.0, "layout_balance": 3, "engagement": 2}))
        self.assertIsNone(parse_aesthetic_scores({"element_quality": 6, "layout_balance": 3, "engagement": 2}))
        self.assertIsNone(parse_aesthetic_scores({"element_quality": True, "layout_balance": 3, "engagement": 2}))


class AestheticJudgeTests(unittest.TestCase):
    def test_the_judge_stays_offline_unless_asked(self):
        calls = []

        def explode(*_args, **_kwargs):
            calls.append(1)
            raise AssertionError("disabled judge must not call the provider")

        report = judge_aesthetics(Path("absent.png"), spec="off", call=explode)
        self.assertFalse(report["available"])
        self.assertEqual("disabled", report["reason"])
        self.assertIsNone(report["element_quality"])
        self.assertEqual([], calls)
        self.assertEqual(64, len(report["prompt_sha256"]))

    def test_an_invalid_model_response_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            preview = Path(tmp) / "preview.png"
            Image.new("RGB", (8, 8), "white").save(preview)
            report = judge_aesthetics(
                preview,
                spec="api",
                call=lambda *_args, **_kwargs: {"element_quality": 9, "layout_balance": 3, "engagement": 2},
            )
        self.assertFalse(report["available"])
        self.assertIsNone(report["layout_balance"])
        self.assertIn("not three integers", report["reason"])


class AgreementTests(unittest.TestCase):
    def test_a_missing_score_stays_out_of_the_denominator(self):
        automatic = [
            {"paper_id": "a", "system_id": "s", "element_quality": 4, "layout_balance": 3, "engagement": 5},
            {"paper_id": "b", "system_id": "s", "element_quality": 2, "layout_balance": None, "engagement": 4},
            {"paper_id": "c", "system_id": "s", "element_quality": 5, "layout_balance": 5, "engagement": 1},
        ]
        human = [
            {"paper_id": "a", "system_id": "s", "element_quality": 4, "layout_balance": 3, "engagement": 4},
            {"paper_id": "b", "system_id": "s", "element_quality": 2, "layout_balance": 1, "engagement": 4},
            {"paper_id": "c", "system_id": "s", "element_quality": 1, "layout_balance": 5, "engagement": 2},
        ]
        report = agreement_report(automatic, human)
        element = report["scores"]["element_quality"]
        balance = report["scores"]["layout_balance"]
        self.assertEqual(3, element["count"])
        self.assertEqual(2, balance["count"])
        self.assertAlmostEqual(2 / 3, element["exact_agreement"], places=4)
        self.assertEqual(1.0, balance["exact_agreement"])
        self.assertIsNotNone(element["spearman_rho"])

    def test_two_pairs_do_not_invent_a_correlation(self):
        automatic = [
            {"paper_id": "a", "system_id": "s", "element_quality": 4, "layout_balance": 3, "engagement": 5},
            {"paper_id": "b", "system_id": "s", "element_quality": 2, "layout_balance": 2, "engagement": 2},
        ]
        human = [
            {"paper_id": "a", "system_id": "s", "element_quality": 5, "layout_balance": 3, "engagement": 5},
            {"paper_id": "b", "system_id": "s", "element_quality": 1, "layout_balance": 2, "engagement": 1},
        ]
        report = agreement_report(automatic, human)
        self.assertEqual(2, report["scores"]["engagement"]["count"])
        self.assertIsNone(report["scores"]["engagement"]["spearman_rho"])


if __name__ == "__main__":
    unittest.main()
