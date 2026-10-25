"""Unit tests for scripts/link_frozen_quiz.py.

PaperQuiz is only a comparison if every system answers the same questions. The scorer will
happily write its own bank when it finds none, which is exactly the failure this step exists to
prevent, so the cases pinned here are: the bank is linked when it exists, a paper without one is
named rather than passed over, and a tree that holds no delivery is not mistaken for a paper.
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "link_frozen_quiz", ROOT / "scripts" / "link_frozen_quiz.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quiz = load_module()


class LinkFrozenQuizTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.staging = self.root / "staging"
        self.bank = self.root / "questions"
        self.staging.mkdir()
        self.bank.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _staged(self, name, *, delivery=True):
        paper = self.staging / name
        paper.mkdir(parents=True, exist_ok=True)
        if delivery:
            (paper / "delivery.pptx").write_bytes(b"pptx")
        return paper

    def _bank(self, name):
        paper = self.bank / name
        paper.mkdir(parents=True, exist_ok=True)
        (paper / "questions.json").write_text(
            json.dumps({"questions": {"Question 1": {"question": "q"}}}), encoding="utf-8"
        )
        return paper

    def _run(self, argv=()):
        out = self.root / "report.json"
        code = quiz.main([
            "--staging", str(self.staging),
            "--questions-root", str(self.bank),
            "--json",
            *argv,
        ])
        report = json.loads((self.staging / "_quiz_link_report.json").read_text(encoding="utf-8"))
        return code, report

    def test_a_staged_paper_gets_the_frozen_bank(self):
        self._staged("10.1_a")
        self._bank("10.1_a")
        code, report = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(report["linked"], 1)
        self.assertEqual(report["absent"], [])
        landed = self.staging / "10.1_a" / "questions.json"
        self.assertTrue(landed.exists())
        self.assertEqual(
            landed.read_text(encoding="utf-8"),
            (self.bank / "10.1_a" / "questions.json").read_text(encoding="utf-8"),
        )

    def test_a_paper_without_a_bank_is_named_not_passed_over(self):
        self._staged("10.1_a")
        self._staged("10.1_b")
        self._bank("10.1_a")
        code, report = self._run()
        self.assertEqual(report["linked"], 1)
        self.assertEqual(report["absent"], ["10.1_b"])
        self.assertFalse((self.staging / "10.1_b" / "questions.json").exists())

    def test_a_tree_holding_no_delivery_is_not_a_paper(self):
        self._staged("10.1_a")
        self._bank("10.1_a")
        (self.staging / "waves").mkdir()
        (self.staging / "waves" / "wave_00.json").write_text("{}", encoding="utf-8")
        code, report = self._run()
        self.assertEqual(report["considered"], 1)
        self.assertEqual(report["linked"], 1)

    def test_strict_fails_when_a_bank_is_missing(self):
        self._staged("10.1_a")
        code, _ = self._run(("--strict",))
        self.assertEqual(code, 1)

    def test_a_stale_quiz_is_replaced_by_the_frozen_one(self):
        paper = self._staged("10.1_a")
        self._bank("10.1_a")
        (paper / "questions.json").write_text("{\"questions\": \"stale\"}", encoding="utf-8")
        self._run()
        body = json.loads((paper / "questions.json").read_text(encoding="utf-8"))
        self.assertIn("Question 1", body["questions"])


if __name__ == "__main__":
    unittest.main()