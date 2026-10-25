"""Unit tests for scripts/pair_with_baselines.py.

The paired table is the only place this project's numbers meet the external baselines', so the
three rules that keep it honest are the three things pinned here: ids are joined by one spelling
rather than by folder name, a paper one side never scored drops that pair alone, and a paper the
baseline list never had is counted rather than quietly averaged in. The ruler check is pinned
too, because a silent model mismatch would make every delta in the table meaningless.
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "pair_with_baselines", ROOT / "scripts" / "pair_with_baselines.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pairing = load_module()


def nine(mean, *, model="gemini-3.1-flash-lite", prompt="nine-v001", available=True):
    body = {"available": available, "mean": mean, "prompt_version": prompt, "model": model}
    for item in pairing.NINE_ITEMS:
        body[item] = mean
    return body


def metrics(paper_id, *, nine_body=None, quiz=None, clip=None, editable=True):
    return {
        "paper_id": paper_id,
        "vlm_nine": nine_body or nine(None, available=False),
        "paper_quiz": quiz or {"available": False, "accuracy": None},
        "paper_figure_clip": clip or {"available": False, "cosine": None},
        "editability": {
            "status": "editable" if editable else "non_ooxml",
            "editable": editable,
            "raster_area_ratio": 0.1,
            "native_object_count": 30,
        },
    }


class PairWithBaselinesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.ours = self.root / "ours"
        self.baseline = self.root / "baseline"

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, base, folder, body):
        target = base / folder
        target.mkdir(parents=True, exist_ok=True)
        (target / "metrics.json").write_text(json.dumps(body), encoding="utf-8")

    def _run(self, systems=("alpha",)):
        out = self.root / "out"
        code = pairing.main([
            "--ours-runs", str(self.ours),
            "--baseline-root", str(self.baseline),
            "--systems", *systems,
            "--out", str(out),
        ])
        self.assertEqual(code, 0)
        return json.loads((out / "pairing.json").read_text(encoding="utf-8"))

    def test_ids_join_across_spellings(self):
        self._write(self.ours, "agent/10.1145_3770855.3818074",
                    metrics("10.1145/3770855.3818074", nine_body=nine(3.0)))
        self._write(self.baseline, "alpha/own/10.1145_3770855.3818074",
                    metrics("10.1145/3770855.3818074", nine_body=nine(4.0)))
        report = self._run()
        block = report["systems"]["alpha"]["nine"]
        self.assertEqual(block["mean"]["n"], 1)
        self.assertAlmostEqual(block["mean"]["mean"], -1.0)

    def test_blank_drops_only_that_pair(self):
        self._write(self.ours, "a", metrics("10.1/a", nine_body=nine(3.0)))
        self._write(self.ours, "b", metrics("10.1/b", nine_body=nine(5.0)))
        self._write(self.baseline, "alpha/own/a", metrics("10.1/a", nine_body=nine(None, available=False)))
        self._write(self.baseline, "alpha/own/b", metrics("10.1/b", nine_body=nine(4.0)))
        report = self._run()
        block = report["systems"]["alpha"]["nine"]
        self.assertEqual(block["mean"]["n"], 1)
        self.assertAlmostEqual(block["mean"]["mean"], 1.0)

    def test_papers_the_baseline_never_had_are_counted(self):
        self._write(self.ours, "a", metrics("10.1/a", nine_body=nine(3.0)))
        self._write(self.ours, "z", metrics("10.1/z", nine_body=nine(3.0)))
        self._write(self.baseline, "alpha/own/a", metrics("10.1/a", nine_body=nine(4.0)))
        report = self._run()
        self.assertEqual(report["overlap"]["ours_scored"], 2)
        self.assertEqual(report["overlap"]["joined"], 1)
        self.assertEqual(report["overlap"]["ours_not_joined"], 1)
        self.assertEqual(report["systems"]["alpha"]["papers_shared_with_ours"], 1)

    def test_ruler_mismatch_is_reported_not_averaged_over(self):
        self._write(self.ours, "a", metrics("10.1/a", nine_body=nine(3.0, model="gemini-3.1-flash-lite")))
        self._write(self.baseline, "alpha/own/a", metrics("10.1/a", nine_body=nine(4.0, model="gemini-3-pro-preview")))
        report = self._run()
        self.assertFalse(report["systems"]["alpha"]["rulers"]["ruler_match"])
        self.assertFalse(report["ruler_match_anywhere"])

    def test_matching_rulers_are_recognised(self):
        self._write(self.ours, "a", metrics("10.1/a", nine_body=nine(3.0)))
        self._write(self.baseline, "alpha/own/a", metrics("10.1/a", nine_body=nine(4.0)))
        report = self._run()
        self.assertTrue(report["systems"]["alpha"]["rulers"]["ruler_match"])

    def test_later_root_wins_for_the_same_paper(self):
        self._write(self.ours, "agent/a", metrics("10.1/a", nine_body=nine(None, available=False)))
        self._write(self.root / "scored", "a", metrics("10.1/a", nine_body=nine(3.0)))
        self._write(self.baseline, "alpha/own/a", metrics("10.1/a", nine_body=nine(4.0)))
        out = self.root / "out"
        pairing.main([
            "--ours-runs", str(self.ours),
            "--ours-runs", str(self.root / "scored"),
            "--baseline-root", str(self.baseline),
            "--systems", "alpha",
            "--out", str(out),
        ])
        report = json.loads((out / "pairing.json").read_text(encoding="utf-8"))
        self.assertEqual(report["systems"]["alpha"]["nine"]["mean"]["n"], 1)

    def test_quiz_accuracy_is_paired_separately_from_nine(self):
        self._write(self.ours, "a", metrics("10.1/a", nine_body=nine(3.0),
                                            quiz={"available": True, "accuracy": 0.6}))
        self._write(self.baseline, "alpha/own/a", metrics("10.1/a", nine_body=nine(4.0),
                                                          quiz={"available": True, "accuracy": 0.9}))
        report = self._run()
        block = report["systems"]["alpha"]["paper_quiz"]
        self.assertEqual(block["accuracy"]["n"], 1)
        self.assertAlmostEqual(block["accuracy"]["mean"], -0.3, places=6)


    def test_nine_item_deltas_come_from_the_item_scores(self):
        mine = nine(3.0)
        mine.update(aesthetic=4, accuracy=2)
        theirs = nine(3.0)
        theirs.update(aesthetic=3, accuracy=5)
        self._write(self.ours, "a", metrics("10.1/a", nine_body=mine))
        self._write(self.baseline, "alpha/own/a", metrics("10.1/a", nine_body=theirs))
        report = self._run()
        items = report["systems"]["alpha"]["nine"]["items"]
        self.assertEqual(items["aesthetic"]["n"], 1)
        self.assertAlmostEqual(items["aesthetic"]["mean"], 1.0)
        self.assertAlmostEqual(items["accuracy"]["mean"], -3.0)

    def test_editability_rate_reads_the_verdict_not_a_label(self):
        self._write(self.ours, "a", metrics("10.1/a", nine_body=nine(3.0), editable=True))
        self._write(self.ours, "b", metrics("10.1/b", nine_body=nine(3.0), editable=False))
        self._write(self.baseline, "alpha/own/a", metrics("10.1/a", nine_body=nine(4.0), editable=False))
        self._write(self.baseline, "alpha/own/b", metrics("10.1/b", nine_body=nine(4.0), editable=False))
        report = self._run()
        block = report["systems"]["alpha"]["editability"]
        self.assertEqual(block["ours"]["scored"], 2)
        self.assertAlmostEqual(block["ours"]["rate"], 0.5)
        self.assertAlmostEqual(block["baseline"]["rate"], 0.0)
        self.assertEqual(block["ours"]["labels"], {"editable": 1, "non_ooxml": 1})


if __name__ == "__main__":
    unittest.main()