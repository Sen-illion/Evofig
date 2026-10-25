"""Tests for adopting decks an earlier batch drew but never delivered."""

import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from rfs.utils import read_json, write_json
from rfs.workflows.corpus_subset import safe_paper_directory
from rfs.workflows.deck_adoption import ADOPTION_REPORT, adopt_undelivered_decks

PAPER = "10.1016/j.joule.2026.102577"
STAMP = "2026-09-27T00:00:00+00:00"


def _vetoed_report(directory: Path) -> dict:
    """A run that drew its deck and was then judged undeliverable on its contract."""
    return {
        "summary": "Direct paper-to-PPTX Judge evolution run.",
        "generation_status": "failed",
        "quality_status": "needs_repair",
        "delivery_status": "unavailable",
        "quality_passed": False,
        "blocking_generation_errors": ["method_contract_invalid"],
        "ok": False,
        "method_validation": {"ok": False, "errors": ["method object[1] references unknown output_ids: solver_assignment"]},
        "structure_validation": {"layout": {"ok": True, "errors": []}, "icons": {"ok": True, "errors": []}},
        "visual_review_status": "pass",
        "out_dir": str(directory),
    }


def _write_attempt(root: Path, paper_id: str, *, attempt: str = "attempt_01", deck: bool = True, report: dict | None = None, readable: bool = True) -> Path:
    directory = Path(root) / "runs" / safe_paper_directory(paper_id) / attempt
    directory.mkdir(parents=True, exist_ok=True)
    if deck:
        path = directory / "editable_composition.pptx"
        if readable:
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("[Content_Types].xml", "<Types/>")
        else:
            path.write_bytes(b"not a zip")
    write_json(directory / "paper_evolve_editable_report.json", report if report is not None else _vetoed_report(directory))
    return directory


class DeckAdoptionTests(unittest.TestCase):
    def test_a_drawn_deck_is_adopted_and_the_old_verdict_stays_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "source", Path(tmp) / "out"
            attempt = _write_attempt(source, PAPER)
            report = adopt_undelivered_decks(sources=[source], out_root=out, entries=[{"paper_id": PAPER}], adopted_at=lambda: STAMP)

            self.assertEqual(report["adopted"], 1)
            self.assertEqual(report["skipped"], 0)
            target = out / "runs" / safe_paper_directory(PAPER) / "attempt_01"
            self.assertTrue((target / "editable_composition.pptx").is_file())
            self.assertTrue((out / ADOPTION_REPORT).is_file())

            written = read_json(target / "paper_evolve_editable_report.json")
            self.assertEqual(written["delivery_status"], "available")
            self.assertEqual(written["generation_status"], "completed")
            self.assertEqual(written["quality_status"], "warning")
            self.assertFalse(written["quality_passed"])
            self.assertEqual(written["blocking_generation_errors"], [])
            self.assertEqual(written["contract_degradations"], ["method_contract_invalid"])
            self.assertEqual(written["adopted_from"], str(attempt))
            self.assertEqual(written["adopted_at"], STAMP)
            self.assertEqual(written["adopted_verdict"]["delivery_status"], "unavailable")
            self.assertEqual(written["adopted_verdict"]["blocking_generation_errors"], ["method_contract_invalid"])
            # The report names the directory it now lives in, not the one it came from.
            self.assertEqual(written["out_dir"], str(target))

    def test_the_source_root_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "source", Path(tmp) / "out"
            attempt = _write_attempt(source, PAPER)
            adopt_undelivered_decks(sources=[source], out_root=out, entries=[{"paper_id": PAPER}])
            self.assertEqual(read_json(attempt / "paper_evolve_editable_report.json")["delivery_status"], "unavailable")

    def test_a_deck_that_is_not_a_readable_powerpoint_is_skipped_with_its_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "source", Path(tmp) / "out"
            _write_attempt(source, PAPER, readable=False)
            report = adopt_undelivered_decks(sources=[source], out_root=out, entries=[{"paper_id": PAPER}])
            self.assertEqual(report["adopted"], 0)
            self.assertEqual(report["skipped"], 1)
            self.assertIn("no readable deck", report["rows"][0]["reason"])
            self.assertFalse((out / "runs").exists())

    def test_a_paper_with_no_attempt_anywhere_is_skipped_with_its_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "source", Path(tmp) / "out"
            source.mkdir()
            report = adopt_undelivered_decks(sources=[source], out_root=out, entries=[{"paper_id": PAPER}])
            self.assertEqual(report["rows"][0]["reason"], "no attempt directory in any source root")

    def test_a_paper_already_delivered_in_the_target_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "source", Path(tmp) / "out"
            _write_attempt(source, PAPER)
            delivered = _write_attempt(out, PAPER, report={"delivery_status": "available", "quality_status": "passed"})
            report = adopt_undelivered_decks(sources=[source], out_root=out, entries=[{"paper_id": PAPER}])
            self.assertEqual(report["already_in_target"], 1)
            self.assertEqual(read_json(delivered / "paper_evolve_editable_report.json")["quality_status"], "passed")

    def test_the_newest_readable_attempt_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "source", Path(tmp) / "out"
            older = _write_attempt(source, PAPER, attempt="attempt_01")
            newest = _write_attempt(source, PAPER, attempt="attempt_02")
            # The pick is by modification time, and two directories written in the same
            # instant can share one on a coarse clock, so the two stamps are pinned here
            # rather than left to the wall clock: this test is about which attempt wins,
            # not about how fast the two were written.
            os.utime(older, (1_000_000.0, 1_000_000.0))
            os.utime(newest, (1_000_001.0, 1_000_001.0))
            report = adopt_undelivered_decks(sources=[source], out_root=out, entries=[{"paper_id": PAPER}])
            self.assertEqual(report["rows"][0]["adopted_from"], str(newest))

    def test_adoption_calls_no_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "source", Path(tmp) / "out"
            _write_attempt(source, PAPER)
            with mock.patch("rfs.providers.call_vlm_json", side_effect=AssertionError("a provider was called")), \
                 mock.patch("rfs.generation.icon_assets.generate_icon_assets", side_effect=AssertionError("an icon was drawn")):
                report = adopt_undelivered_decks(sources=[source], out_root=out, entries=[{"paper_id": PAPER}])
            self.assertEqual(report["adopted"], 1)

    def test_the_adoption_report_accounts_for_every_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "source", Path(tmp) / "out"
            _write_attempt(source, "10.1000/adopted")
            _write_attempt(source, "10.1000/broken", readable=False)
            _write_attempt(out, "10.1000/settled", report={"delivery_status": "available"})
            entries = [{"paper_id": "10.1000/adopted"}, {"paper_id": "10.1000/broken"}, {"paper_id": "10.1000/settled"}]
            report = adopt_undelivered_decks(sources=[source], out_root=out, entries=entries)
            self.assertEqual((report["papers"], report["adopted"], report["skipped"], report["already_in_target"]), (3, 1, 1, 1))
            self.assertEqual(report["degraded"], 1)
            self.assertEqual(len(report["rows"]), 3)


if __name__ == "__main__":
    unittest.main()
