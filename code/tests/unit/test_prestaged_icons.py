"""Tests for finding the icons a reuse wave pre-staged for a paper's first attempt."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rfs.utils import write_json
from rfs.workflows.paper_evolve_editable import _inherited_icon_root, _prestaged_icon_root


class PrestagedIconRootTests(unittest.TestCase):
    def test_a_staged_attempt_directory_offers_its_icons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged = root / "runs" / "p" / "attempt_01" / "round_00" / "candidate_00" / "assets"
            (staged / "icon_input_pipeline").mkdir(parents=True)
            (staged / "icon_input_pipeline" / "candidate_00.png").write_bytes(b"png")

            found = _prestaged_icon_root(root / "runs" / "p" / "attempt_01")

            self.assertEqual(found, staged)

    def test_a_real_attempt_is_replayed_not_raided(self):
        """An attempt that read the paper owns reports; its icons belong to the replay."""

        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp) / "attempt_01"
            assets = attempt / "round_00" / "candidate_00" / "assets" / "icon_x"
            assets.mkdir(parents=True)
            (assets / "candidate_00.png").write_bytes(b"png")
            write_json(attempt / "input_manifest.json", {"paper": "p"})

            self.assertIsNone(_prestaged_icon_root(attempt))

    def test_a_directory_without_staged_icons_offers_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp) / "attempt_01"
            (attempt / "round_00" / "candidate_00").mkdir(parents=True)

            self.assertIsNone(_prestaged_icon_root(attempt))

    def test_missing_source_offers_nothing(self):
        self.assertIsNone(_prestaged_icon_root(None))
        self.assertIsNone(_prestaged_icon_root(""))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_prestaged_icon_root(Path(tmp) / "absent"))


class InheritedIconRootTests(unittest.TestCase):
    """What a fresh attempt may keep when a replay already paid for the Icons."""

    def test_the_replay_hands_its_icons_over_by_default(self):
        staged = Path("/staged")
        self.assertEqual(
            _inherited_icon_root({"icon_root": staged}, None),
            staged,
        )
        self.assertEqual(
            _inherited_icon_root({"icon_root": staged}, None, "on"),
            staged,
        )

    def test_a_draw_again_run_ignores_the_replay_and_keeps_the_staged_wave(self):
        """The switch governs the replay, not the reuse wave's certified mapping."""

        staged = Path("/staged")
        self.assertIsNone(_inherited_icon_root({"icon_root": Path("/replayed")}, None, "off"))
        self.assertEqual(_inherited_icon_root({"icon_root": Path("/replayed")}, staged, "off"), staged)

    def test_no_replay_and_no_stage_offers_nothing(self):
        for value in (None, {}, {"icon_root": None}):
            self.assertIsNone(_inherited_icon_root(value, None))
            self.assertIsNone(_inherited_icon_root(value, None, "off"))


if __name__ == "__main__":
    unittest.main()
