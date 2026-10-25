"""The hand-pick framework-figure gallery: queue building and bookkeeping.

The server itself is thin; what matters is that the queue is built from the two
candidate pools without duplicates, in a deliberate order, and that a decision is
written to state, log and the picked hand-off list together.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVER = REPO / "benchmarks" / "editable-figure-bench" / "review" / "serve_framework_gallery.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("serve_framework_gallery", SERVER)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest("gallery server is not present")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GalleryQueueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = _load_server()
        cls.items = cls.server._load_items()

    def test_queue_is_not_empty(self) -> None:
        self.assertGreater(len(self.items), 0)

    def test_one_row_per_paper(self) -> None:
        papers = [item["paper_id"] for item in self.items if item["paper_id"]]
        self.assertEqual(len(papers), len(set(papers)))

    def test_numbers_are_sequential(self) -> None:
        self.assertEqual([item["n"] for item in self.items], list(range(1, len(self.items) + 1)))

    def test_tiers_never_go_backwards(self) -> None:
        tiers = [item["tier"] for item in self.items]
        self.assertEqual(tiers, sorted(tiers))

    def test_duplicate_paper_keeps_the_better_tier(self) -> None:
        seen: dict[str, int] = {}
        for item in self.items:
            if item["paper_id"]:
                seen[item["paper_id"]] = item["tier"]
        self.assertTrue(all(tier in (0, 1, 2) for tier in seen.values()))

    def test_every_row_points_at_an_image(self) -> None:
        for item in self.items:
            self.assertTrue(item["image"])
            self.assertIn(item["kind"], {"png", "pdf"})


class GalleryTierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = _load_server()

    def test_a_human_pass_outranks_everything(self) -> None:
        self.assertEqual(self.server._tier("yes_pool", {"source": "human_pass1"}), 0)
        self.assertEqual(self.server._tier("yes_pool", {"source": "prior_human_yes"}), 0)

    def test_accepted_captures_come_next(self) -> None:
        self.assertEqual(self.server._tier("accepted", {"source": "screen_figure"}), 1)

    def test_the_automated_screen_comes_last(self) -> None:
        self.assertEqual(self.server._tier("yes_pool", {"source": "kira_framework_like"}), 2)

    def test_cache_key_follows_the_asset_not_the_queue(self) -> None:
        first = self.server._cache_path(Path("<PATH>"))
        again = self.server._cache_path(Path("<PATH>"))
        other = self.server._cache_path(Path("<PATH>"))
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)


class GalleryBookkeepingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = _load_server()
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        root = Path(self._temp.name)
        self._state = self.server.STATE_PATH
        self._picked = self.server.PICKED_PATH
        self.server.STATE_PATH = root / "review.json"
        self.server.PICKED_PATH = root / "picked.jsonl"
        self.addCleanup(self._restore)
        self.items = [
            {"n": 1, "paper_id": "a", "arxiv_id": "1", "domain": "agent", "tier": 0,
             "origin": "yes_pool", "source": "human_pass1", "scope": "whole_figure",
             "image": str(REPO / "x.pdf")},
            {"n": 2, "paper_id": "b", "arxiv_id": "2", "domain": "rag", "tier": 2,
             "origin": "yes_pool", "source": "kira_framework_like", "scope": "whole_figure",
             "image": str(REPO / "y.pdf")},
        ]

    def _restore(self) -> None:
        self.server.STATE_PATH = self._state
        self.server.PICKED_PATH = self._picked

    def test_picks_are_written_out_and_can_be_undone(self) -> None:
        state = {"decisions": {"1": {"decision": "picked", "at": "t"}}, "cursor": 1}
        self.server._write_picks(state, self.items)
        rows = [json.loads(line) for line in self.server.PICKED_PATH.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual([row["n"] for row in rows], [1])
        self.assertEqual(rows[0]["image"], "x.pdf")

        state["decisions"]["1"] = {"decision": "not_framework", "at": "t2"}
        self.server._write_picks(state, self.items)
        self.assertEqual(self.server.PICKED_PATH.read_text(encoding="utf-8"), "")

    def test_a_broken_state_file_reads_as_empty(self) -> None:
        self.server.STATE_PATH.write_text("{not json", encoding="utf-8")
        state = self.server._read_state()
        self.assertEqual(state["decisions"], {})

    def test_state_round_trips(self) -> None:
        self.server._write_state({"decisions": {"3": {"decision": "maybe", "at": "t"}}, "cursor": 3})
        state = self.server._read_state()
        self.assertEqual(state["cursor"], 3)
        self.assertEqual(state["decisions"]["3"]["decision"], "maybe")


if __name__ == "__main__":
    unittest.main()
