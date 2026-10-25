"""The staging step's name resolution.

An arm names a paper's output folder with the driver's own ``safe`` and the scorer asks for the
folder ``safe_paper_id`` would give. The two agree except on ids longer than 120 characters, which
``safe_paper_id`` cuts and ``safe`` does not, and a paper that fails to stage is a hole in the
table that the arm's own delivery count still reports as filled. These cases pin that, and the
case that keeps a prefix from binding the wrong paper.
"""

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO = Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "stage_budget_deliveries", REPO / "scripts" / "stage_budget_deliveries.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FindPaperDirTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.arm = Path(self.tmp.name) / "out" / "topk_5"

    def make(self, domain, name):
        path = self.arm / domain / name
        path.mkdir(parents=True)
        return path

    def test_an_exact_name_is_found(self):
        want = self.make("cv", "ICCV2025_short")
        self.assertEqual(want, self.module.find_paper_dir(self.arm, "ICCV2025_short"))

    def test_a_name_the_scorer_truncated_still_finds_the_paper(self):
        long_name = "ICCV2025_" + "dynfacerestore_balancing_" * 6
        want = self.make("cv", long_name)
        truncated = long_name[:120]
        self.assertNotEqual(long_name, truncated)
        self.assertEqual(want, self.module.find_paper_dir(self.arm, truncated))

    def test_a_prefix_shared_by_two_papers_resolves_to_neither(self):
        self.make("cv", "paper_one_and_more")
        self.make("nlp", "paper_one_and_even_more")
        self.assertIsNone(self.module.find_paper_dir(self.arm, "paper_one"))

    def test_an_exact_name_beats_a_longer_sibling(self):
        want = self.make("cv", "paper_one")
        self.make("nlp", "paper_one_and_more")
        self.assertEqual(want, self.module.find_paper_dir(self.arm, "paper_one"))

    def test_a_paper_the_arm_never_ran_is_missing(self):
        self.assertIsNone(self.module.find_paper_dir(self.arm, "nothing_here"))


if __name__ == "__main__":
    unittest.main()
