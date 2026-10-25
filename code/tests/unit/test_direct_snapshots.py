"""Tests for direct evolution snapshots, diffs, and rollback."""

from __future__ import annotations

import unittest

from rfs.evolution.direct import create_state_snapshot, diff_figure_programs, diff_state_snapshots, restore_state_snapshot, verify_state_snapshot


def _program(*, node_x: float = 0.1) -> dict:
    return {
        "panels": [{"id": "panel_1", "bbox_percent": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}],
        "semantic_nodes": [{"id": "node_1", "bbox_percent": {"x": node_x, "y": 0.1, "w": 0.2, "h": 0.2}}],
        "arrows": [{"id": "edge_1", "source": "node_1", "target": "node_1"}],
    }


class DirectSnapshotTests(unittest.TestCase):
    def test_verified_snapshot_round_trips_and_diffs(self) -> None:
        before = create_state_snapshot({"layout": {"revision": 1}}, label="before", round_index=1)
        after = create_state_snapshot({"layout": {"revision": 2}}, label="after", round_index=1)
        self.assertTrue(verify_state_snapshot(before))
        diff = diff_state_snapshots(before, after)
        self.assertEqual(["layout"], diff["changed_fields"])
        restored, report = restore_state_snapshot(before)
        self.assertEqual({"layout": {"revision": 1}}, restored)
        self.assertEqual(before["state_hash"], report["restored_hash"])

    def test_tampered_snapshot_cannot_be_restored(self) -> None:
        snapshot = create_state_snapshot({"value": 1}, label="before", round_index=1)
        snapshot["state"]["value"] = 2
        self.assertFalse(verify_state_snapshot(snapshot))
        with self.assertRaises(ValueError):
            restore_state_snapshot(snapshot)


class FigureProgramDiffTests(unittest.TestCase):
    """The figure-level diff is what stops a round that moved nothing reading as a repair."""

    def test_re_emitting_the_same_objects_is_not_a_change(self) -> None:
        diff = diff_figure_programs(_program(), _program())
        self.assertTrue(diff["available"])
        self.assertFalse(diff["changed"])
        self.assertEqual([], diff["changed_object_ids"])

    def test_a_moved_object_is_reported_by_id(self) -> None:
        diff = diff_figure_programs(_program(), _program(node_x=0.6))
        self.assertTrue(diff["changed"])
        self.assertEqual(["node_1"], diff["changed_object_ids"])

    def test_an_added_or_removed_object_counts_as_a_change(self) -> None:
        after = _program()
        after["arrows"] = []
        diff = diff_figure_programs(_program(), after)
        self.assertTrue(diff["changed"])
        self.assertEqual(["edge_1"], diff["changed_object_ids"])

    def test_the_first_round_has_nothing_to_compare_against(self) -> None:
        diff = diff_figure_programs({}, _program())
        self.assertFalse(diff["available"])
        self.assertIsNone(diff["changed"])
        self.assertEqual([], diff["changed_object_ids"])


if __name__ == "__main__":
    unittest.main()
