"""Unit tests for scripts/rescore_baselines_with_current_ruler.py.

Re-reading the baselines is only worth anything if it re-reads the *same* figure with the *same*
questions and the *same* grounding, and if it does not overwrite the scores its authors already
wrote. These cases pin those three, plus the rule that a paper the system never scored is skipped
instead of scored as a blank.
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "rescore_baselines_with_current_ruler",
        ROOT / "scripts" / "rescore_baselines_with_current_ruler.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rescore = load_module()


class RescoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.baseline = self.root / "efb_main"
        self.system = self.baseline / "alpha" / "own" / "10.1_a"
        self.system.mkdir(parents=True)
        (self.system / "metrics.json").write_text("{}", encoding="utf-8")
        (self.system / "preview.png").write_bytes(b"png")
        (self.system / "questions.json").write_text("{\"questions\": {}}", encoding="utf-8")
        self.dest = self.root / "out" / "alpha" / "own" / "10.1_a"

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_system_that_never_scored_the_paper_is_skipped(self):
        bare = self.baseline / "beta" / "own" / "10.1_a"
        bare.mkdir(parents=True)
        (bare / "preview.png").write_bytes(b"png")
        self.assertEqual("system_never_scored_it", rescore.prepare(bare, self.root / "o", None))

    def test_a_folder_with_no_figure_is_reported(self):
        (self.system / "preview.png").unlink()
        self.assertEqual("no_figure", rescore.prepare(self.system, self.root / "o", None))

    def test_the_figure_and_the_frozen_quiz_are_linked_side_by_side(self):
        self.assertIsNone(rescore.prepare(self.system, self.dest, None))
        self.assertTrue((self.dest / "preview.png").is_file())
        self.assertTrue((self.dest / "questions.json").is_file())

    def test_the_systems_own_method_text_wins_over_the_shared_copy(self):
        (self.system / "method.txt").write_text("own", encoding="utf-8")
        shared = self.baseline / "own" / "text" / "10.1_a"
        shared.mkdir(parents=True)
        (shared / "method.txt").write_text("shared", encoding="utf-8")
        chosen = rescore.method_source_for(self.baseline, self.system, "10.1_a")
        self.assertEqual(chosen.read_text(encoding="utf-8"), "own")

    def test_the_shared_method_text_is_the_fallback(self):
        shared = self.baseline / "own" / "text" / "10.1_a"
        shared.mkdir(parents=True)
        (shared / "method.txt").write_text("shared", encoding="utf-8")
        chosen = rescore.method_source_for(self.baseline, self.system, "10.1_a")
        self.assertEqual(chosen.read_text(encoding="utf-8"), "shared")

    def test_scoring_uses_the_frozen_bank_and_never_writes_its_own(self):
        seen = {}

        def fake(run_dir, **kwargs):
            seen.update(kwargs)
            seen["run_dir"] = run_dir
            return {"ok": True, "metrics": {
                "vlm_nine": {"available": True, "model": "gemini-3.1-flash-lite", "prompt_version": "nine-v002"},
                "paper_quiz": {"available": True},
            }}

        original = rescore.score_own_delivery
        rescore.score_own_delivery = fake
        try:
            row = rescore.one((
                "alpha", "10.1/a", "10.1_a", self.system, self.dest, None, False,
            ))
        finally:
            rescore.score_own_delivery = original

        self.assertFalse(seen["allow_question_generation"])
        self.assertEqual("api", seen["paper_quiz_spec"])
        self.assertEqual("api", seen["nine_spec"])
        self.assertTrue(seen["force"])
        self.assertIsNone(seen["embedder"])
        self.assertTrue(row["nine_available"])
        self.assertEqual("gemini-3.1-flash-lite", row["nine_model"])

    def test_method_text_is_handed_to_the_rubric(self):
        method = self.root / "method.txt"
        method.write_text("the method section", encoding="utf-8")
        seen = {}

        def fake(run_dir, **kwargs):
            seen.update(kwargs)
            return {"ok": True, "metrics": {}}

        original = rescore.score_own_delivery
        rescore.score_own_delivery = fake
        try:
            rescore.one(("alpha", "10.1/a", "10.1_a", self.system, self.dest, method, False))
        finally:
            rescore.score_own_delivery = original
        self.assertEqual("the method section", seen["method_text"])

    def test_an_already_scored_paper_is_left_alone_without_force(self):
        self.dest.mkdir(parents=True)
        (self.dest / "metrics.json").write_text("{}", encoding="utf-8")
        calls = []

        def fake(run_dir, **kwargs):
            calls.append(run_dir)
            return {"ok": True, "metrics": {}}

        original = rescore.score_own_delivery
        rescore.score_own_delivery = fake
        try:
            row = rescore.one(("alpha", "10.1/a", "10.1_a", self.system, self.dest, None, False))
        finally:
            rescore.score_own_delivery = original
        self.assertEqual([], calls)
        self.assertEqual("already_scored", row["reason"])


if __name__ == "__main__":
    unittest.main()