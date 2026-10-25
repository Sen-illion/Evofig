"""Tests for settling a refused round object by object instead of whole."""

import unittest
from unittest.mock import patch

from rfs.evolution.direct import partial_rollback
from rfs.evolution.direct.partial_rollback import (
    PARTIAL_ROLLBACK_CONTRACT,
    SETTLED_DELETION_KEPT,
    SETTLED_RESTORED,
    blamed_objects,
    faults_of,
    object_moved,
    object_readings,
    plan_partial_rollback,
    validate_partial_rollback,
)


def _box(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}


def _node(node_id, box, *, panel="p1", label=None):
    item = {"id": node_id, "panel_id": panel, "bbox_percent": box}
    if label is not None:
        item["label_bbox_percent"] = label
    return item


def _panel(panel_id="p1", box=None):
    return {"id": panel_id, "bbox_percent": box or _box(0.0, 0.0, 1.0, 1.0)}


def _state(nodes, *, slots=(), panels=None, connectors=()):
    panels = list(panels) if panels is not None else [_panel()]
    layout = {"panels": panels, "nodes": list(nodes), "connectors": list(connectors)}
    program = {
        "panels": panels,
        "semantic_nodes": [dict(item) for item in nodes],
        "slots": [dict(item) for item in slots],
        "arrows": [dict(item) for item in connectors],
    }
    return {"method": {}, "layout": layout, "icons": {}, "program": program}


_REAL_MERGE_OBJECTS = partial_rollback._merge_objects


def _leaking_merge_objects(*, before_items, after_items, keep, additions=None):
    """A merge that hands the candidate back even when nothing was kept.

    This is what the merge used to do on its own -- content that follows no object was carried
    over from the candidate -- and the settlement has to catch it instead of shipping a figure
    nobody settled.
    """

    if not keep and not additions:
        return [dict(item) for item in (after_items or []) if isinstance(item, dict)]
    return _REAL_MERGE_OBJECTS(
        before_items=before_items, after_items=after_items, keep=keep, additions=additions
    )


class PartialRollbackTests(unittest.TestCase):
    def test_the_object_that_came_out_worse_is_given_back_and_the_clean_one_is_kept(self):
        before = _state([
            _node("a", _box(0.1, 0.1, 0.2, 0.2)),
            _node("b", _box(0.6, 0.6, 0.2, 0.2)),
        ])
        after = _state([
            _node("a", _box(0.6, 0.6, 0.2, 0.2)),
            _node("b", _box(0.6, 0.6, 0.2, 0.2)),
        ])
        report = plan_partial_rollback(before, after, round_index=7, action={"type": "resize_nodes"})
        self.assertTrue(report["applied"], report["summary"])
        reverted = {item["object_id"] for item in report["reverted"]}
        self.assertEqual(reverted, {"a", "b"})
        self.assertEqual(report["tier"], 1)
        self.assertEqual(report["before_faults"], 0)
        self.assertLessEqual(report["merged_faults"], report["before_faults"])
        merged_nodes = {item["id"]: item for item in report["merged_state"]["layout"]["nodes"]}
        self.assertEqual(merged_nodes["a"]["bbox_percent"], _box(0.1, 0.1, 0.2, 0.2))
        self.assertEqual(validate_partial_rollback(report), [])

    def test_an_object_that_came_out_better_is_kept_while_its_neighbour_is_given_back(self):
        before = _state(
            [
                _node("a", _box(0.05, 0.05, 0.2, 0.2), label=_box(0.06, 0.06, 0.1, 0.04)),
                _node("b", _box(0.6, 0.6, 0.2, 0.2), label=_box(0.62, 0.62, 0.12, 0.05)),
            ],
            slots=[
                {"id": "slot_a", "parent_id": "a", "bbox_percent": _box(0.3, 0.3, 0.05, 0.05)},
                {"id": "slot_b", "parent_id": "b", "bbox_percent": _box(0.72, 0.72, 0.04, 0.04)},
            ],
        )
        after = _state(
            [
                _node("a", _box(0.05, 0.05, 0.2, 0.2), label=_box(0.06, 0.06, 0.1, 0.04)),
                _node("b", _box(0.6, 0.6, 0.2, 0.2), label=_box(0.9, 0.9, 0.1, 0.06)),
            ],
            slots=[
                {"id": "slot_a", "parent_id": "a", "bbox_percent": _box(0.12, 0.14, 0.05, 0.05)},
                {"id": "slot_b", "parent_id": "b", "bbox_percent": _box(0.72, 0.72, 0.04, 0.04)},
            ],
        )
        reading_before = object_readings(before)
        self.assertEqual(faults_of(reading_before["a"]), 1)
        report = plan_partial_rollback(before, after, round_index=3)
        self.assertTrue(report["applied"], report["summary"])
        self.assertIn("a", {item["object_id"] for item in report["kept"]})
        self.assertIn("b", {item["object_id"] for item in report["reverted"]})
        merged_slots = {
            item["id"]: item["bbox_percent"]
            for item in report["merged_state"]["program"]["slots"]
        }
        self.assertEqual(merged_slots["slot_a"], _box(0.12, 0.14, 0.05, 0.05))
        merged_nodes = {item["id"]: item for item in report["merged_state"]["layout"]["nodes"]}
        self.assertEqual(merged_nodes["b"]["label_bbox_percent"], _box(0.62, 0.62, 0.12, 0.05))

    def test_a_move_that_replanned_the_figure_is_not_settled_object_by_object(self):
        before = _state([_node("a", _box(0.05, 0.05, 0.2, 0.2)), _node("b", _box(0.5, 0.05, 0.2, 0.2))])
        after = _state([_node("x", _box(0.05, 0.05, 0.2, 0.2)), _node("y", _box(0.5, 0.5, 0.2, 0.2))])
        report = plan_partial_rollback(before, after, round_index=2)
        self.assertFalse(report["applied"])
        self.assertIn("re-planned", report["reason"])
        self.assertLess(report["overlap_ratio"], 0.6)
        self.assertFalse(report["merged_state"])
        self.assertEqual(report["tier"], 0)

    def test_a_round_that_moved_nothing_has_nothing_to_settle(self):
        """Nothing moved, nothing was added and nothing was deleted: there is no verdict to take."""

        state = _state([_node("a", _box(0.05, 0.05, 0.2, 0.2)), _node("b", _box(0.5, 0.5, 0.2, 0.2))])
        report = plan_partial_rollback(state, state, round_index=1)
        self.assertFalse(report["applied"])
        self.assertIn("nothing to give back", report["reason"])
        self.assertEqual(report["tier"], 0)

    def test_an_object_that_moved_without_coming_out_worse_is_kept(self):
        """A moved object is not the same object in the same place -- but a moved object that is
        no worse is still worth keeping, and only the objects that came out worse go back."""

        before = _state(
            [
                _node("a", _box(0.05, 0.05, 0.2, 0.2)),
                _node("b", _box(0.35, 0.05, 0.2, 0.2)),
                _node("c", _box(0.65, 0.05, 0.2, 0.2)),
            ],
            slots=[{"id": "slot_c", "parent_id": "c", "bbox_percent": _box(0.66, 0.06, 0.05, 0.05)}],
        )
        after = _state(
            [
                _node("a", _box(0.05, 0.05, 0.2, 0.2)),
                # b moves into the empty half of the panel and stays clean.
                _node("b", _box(0.35, 0.55, 0.2, 0.2)),
                # c keeps its box but its icon is drawn outside it, so c is the one that is worse.
                _node("c", _box(0.65, 0.05, 0.2, 0.2)),
            ],
            slots=[{"id": "slot_c", "parent_id": "c", "bbox_percent": _box(0.90, 0.06, 0.05, 0.05)}],
        )
        report = plan_partial_rollback(before, after, round_index=66)
        self.assertTrue(report["applied"], report["summary"])
        self.assertEqual(report["tier"], 1)
        self.assertEqual({item["object_id"] for item in report["kept"]}, {"a", "b", "p1"})
        self.assertEqual({item["object_id"] for item in report["reverted"]}, {"c"})
        self.assertLessEqual(report["merged_faults"], report["before_faults"])
        merged_nodes = {item["id"]: item for item in report["merged_state"]["layout"]["nodes"]}
        self.assertEqual(merged_nodes["b"]["bbox_percent"], _box(0.35, 0.55, 0.2, 0.2))
        self.assertEqual(merged_nodes["c"]["bbox_percent"], _box(0.65, 0.05, 0.2, 0.2))
        # The settlement is a real merge, not the snapshot with extra bookkeeping.
        self.assertNotEqual(report["merged_state"]["layout"]["nodes"], before["layout"]["nodes"])
        self.assertEqual(validate_partial_rollback(report), [])

    def test_the_objects_the_comparison_blamed_are_given_back(self):
        """The model's own answer about where the figure is worse steers the settlement: the
        objects the named defect points at go back even though they came out no worse, and the
        objects it says nothing about stay where the round put them."""

        before = _state(
            [
                _node("a", _box(0.05, 0.05, 0.3, 0.3)),
                _node("b", _box(0.10, 0.10, 0.3, 0.3)),
                _node("c", _box(0.05, 0.55, 0.2, 0.2)),
            ],
            slots=[{"id": "slot_c", "parent_id": "c", "bbox_percent": _box(0.30, 0.30, 0.05, 0.05)}],
        )
        after = _state(
            [
                _node("a", _box(0.05, 0.05, 0.3, 0.3)),
                _node("b", _box(0.12, 0.12, 0.3, 0.3)),
                _node("c", _box(0.05, 0.55, 0.2, 0.2)),
            ],
            slots=[{"id": "slot_c", "parent_id": "c", "bbox_percent": _box(0.06, 0.56, 0.05, 0.05)}],
        )
        readings_before, readings_after = object_readings(before), object_readings(after)
        self.assertEqual(faults_of(readings_before["a"]), 1)
        self.assertEqual({item["object_id"] for item in plan_partial_rollback(before, after)["reverted"]}, set())
        self.assertEqual(blamed_objects(readings_after, ["overlap"]), {"a", "b"})
        self.assertEqual(blamed_objects(readings_after, ["missing_core_stage"]), set())

        report = plan_partial_rollback(before, after, round_index=9, blame_categories=["overlap"])
        self.assertTrue(report["applied"], report["summary"])
        self.assertEqual(report["blame"], ["a", "b"])
        self.assertIn("c", {item["object_id"] for item in report["kept"]})
        reverted = {item["object_id"]: item for item in report["reverted"]}
        self.assertEqual(set(reverted), {"a", "b"})
        self.assertEqual({item["basis"] for item in reverted.values()}, {"blamed"})
        merged_nodes = {item["id"]: item for item in report["merged_state"]["layout"]["nodes"]}
        self.assertEqual(merged_nodes["b"]["bbox_percent"], _box(0.10, 0.10, 0.3, 0.3))

        unnamed = plan_partial_rollback(before, after, round_index=9, blame_categories=["missing_core_stage"])
        unnamed_nodes = {item["id"]: item for item in unnamed["merged_state"]["layout"]["nodes"]}
        self.assertEqual(unnamed["blame"], [])
        self.assertEqual(unnamed_nodes["b"]["bbox_percent"], _box(0.12, 0.12, 0.3, 0.3))

    def test_the_settlement_drops_a_step_when_the_widest_would_leave_the_figure_worse(self):
        """Keeping a moved object at a place the snapshot still occupies is how a settlement ends
        up worse than the figure it protects; the settlement has to notice and give it back."""

        before = _state(
            [
                _node("a", _box(0.05, 0.05, 0.2, 0.2)),
                _node("c", _box(0.85, 0.05, 0.2, 0.2)),
            ],
            slots=[{"id": "slot_a", "parent_id": "a", "bbox_percent": _box(0.06, 0.06, 0.05, 0.05)}],
        )
        after = _state(
            [
                # a moves away and leaves its icon behind, so a is the one that is worse.
                _node("a", _box(0.05, 0.55, 0.2, 0.2)),
                # c moves into the place a has just left, and is clean there.
                _node("c", _box(0.05, 0.05, 0.2, 0.2)),
            ],
            slots=[{"id": "slot_a", "parent_id": "a", "bbox_percent": _box(0.06, 0.06, 0.05, 0.05)}],
        )
        report = plan_partial_rollback(before, after, round_index=22)
        self.assertFalse(report["tiers"][0]["accepted"])
        self.assertIn("more faults", report["tiers"][0]["why"])
        self.assertEqual(report["tier"], 2)
        self.assertTrue(report["applied"], report["summary"])
        merged_nodes = {item["id"]: item for item in report["merged_state"]["layout"]["nodes"]}
        self.assertEqual(merged_nodes["a"]["bbox_percent"], _box(0.05, 0.05, 0.2, 0.2))
        self.assertEqual(merged_nodes["c"]["bbox_percent"], _box(0.85, 0.05, 0.2, 0.2))
        self.assertEqual(validate_partial_rollback(report), [])

    def test_the_widest_step_keeps_a_figure_that_came_out_better_altogether(self):
        before = _state(
            [
                _node("a", _box(0.05, 0.05, 0.2, 0.2), label=_box(0.06, 0.06, 0.1, 0.04)),
                _node("b", _box(0.55, 0.05, 0.2, 0.2), label=_box(0.56, 0.06, 0.1, 0.04)),
            ],
            slots=[
                {"id": "slot_a", "parent_id": "a", "bbox_percent": _box(0.30, 0.30, 0.05, 0.05)},
                {"id": "slot_b", "parent_id": "b", "bbox_percent": _box(0.80, 0.30, 0.05, 0.05)},
            ],
        )
        after = _state(
            [
                _node("a", _box(0.05, 0.05, 0.2, 0.2), label=_box(0.06, 0.06, 0.1, 0.04)),
                _node("b", _box(0.55, 0.05, 0.2, 0.2), label=_box(0.56, 0.06, 0.1, 0.04)),
            ],
            slots=[
                {"id": "slot_a", "parent_id": "a", "bbox_percent": _box(0.08, 0.14, 0.05, 0.05)},
                {"id": "slot_b", "parent_id": "b", "bbox_percent": _box(0.58, 0.14, 0.05, 0.05)},
            ],
        )
        report = plan_partial_rollback(before, after, round_index=30)
        self.assertEqual(report["before_faults"], 2)
        self.assertEqual(report["merged_faults"], 0)
        self.assertTrue(report["applied"], report["summary"])
        self.assertEqual(report["tier"], 1)
        self.assertEqual(
            [item["object_id"] for item in report["reverted"] if item["kind"] != "panel"], []
        )
        self.assertIn("a", {item["object_id"] for item in report["kept"]})
        self.assertEqual(report["merged_state"]["program"]["slots"], after["program"]["slots"])
        self.assertNotEqual(report["merged_state"]["program"]["slots"], before["program"]["slots"])
        self.assertEqual(validate_partial_rollback(report), [])

    def test_an_object_the_move_deleted_is_settled_by_its_panel(self):
        before = _state([_node("a", _box(0.05, 0.05, 0.4, 0.4)), _node("b", _box(0.5, 0.5, 0.4, 0.4))])
        after = _state([_node("a", _box(0.05, 0.05, 0.4, 0.4))])
        report = plan_partial_rollback(before, after, round_index=5)
        orphaned = {item["object_id"]: item for item in report["orphaned"]}
        self.assertIn("b", orphaned)
        self.assertEqual(orphaned["b"]["settled"], SETTLED_RESTORED)
        self.assertEqual(orphaned["b"]["panel_id"], "p1")
        self.assertIn("panel came out worse", orphaned["b"]["why"])
        self.assertTrue(report["applied"], report["summary"])
        merged_nodes = {item["id"] for item in report["merged_state"]["layout"]["nodes"]}
        self.assertIn("b", merged_nodes)

    def test_a_deleted_object_whose_panel_held_up_keeps_the_deletion(self):
        panels = [_panel("p1", _box(0.0, 0.0, 1.0, 1.0))]
        before = _state(
            [_node("a", _box(0.02, 0.02, 0.3, 0.3)), _node("b", _box(0.4, 0.4, 0.3, 0.3))],
            panels=panels,
        )
        after = _state(
            [
                _node("a", _box(0.02, 0.02, 0.3, 0.3)),
                _node("d", _box(0.4, 0.4, 0.3, 0.3)),
                _node("e", _box(0.8, 0.8, 0.15, 0.15)),
            ],
            panels=panels,
        )
        report = plan_partial_rollback(before, after, round_index=6)
        orphaned = {item["object_id"]: item for item in report["orphaned"]}
        self.assertIn("b", orphaned)
        self.assertEqual(orphaned["b"]["settled"], SETTLED_DELETION_KEPT)

    def test_a_nudge_inside_the_tolerance_is_not_a_fault(self):
        state = _state(
            [_node("a", _box(0.05, 0.05, 0.2, 0.2), label=_box(0.08, 0.08, 0.1, 0.06))],
            slots=[{"id": "slot_a", "parent_id": "a", "bbox_percent": _box(0.2015, 0.1, 0.05, 0.05)}],
        )
        readings = object_readings(state)
        self.assertEqual(faults_of(readings["a"]), 0)

    def test_an_object_that_exists_only_in_the_candidate_is_given_a_verdict(self):
        """An addition is settled on its own, not swept in with the objects that were refused."""

        before = _state([
            _node("a", _box(0.05, 0.05, 0.2, 0.2)),
            _node("b", _box(0.5, 0.5, 0.2, 0.2)),
        ])
        after = _state([
            _node("a", _box(0.05, 0.05, 0.2, 0.2)),
            # b is dropped onto a, so both of them come out worse and are given back.
            _node("b", _box(0.05, 0.05, 0.2, 0.2)),
            # one addition lands in the clear, the other lands on a.
            _node("n_ok", _box(0.75, 0.75, 0.1, 0.1)),
            _node("n_bad", _box(0.05, 0.05, 0.1, 0.1)),
        ])
        report = plan_partial_rollback(before, after, round_index=66)
        additions = {item["object_id"]: item for item in report["additions"]}
        self.assertEqual({"n_ok", "n_bad"}, set(additions))
        self.assertIs(True, additions["n_ok"]["kept"])
        self.assertIs(False, additions["n_bad"]["kept"])
        for item in additions.values():
            self.assertTrue(str(item["why"]).strip())
        self.assertTrue(report["applied"], report["summary"])
        self.assertLessEqual(report["merged_faults"], report["before_faults"])
        merged_nodes = {item["id"] for item in report["merged_state"]["layout"]["nodes"]}
        self.assertIn("n_ok", merged_nodes)
        self.assertNotIn("n_bad", merged_nodes)
        self.assertEqual(validate_partial_rollback(report), [])

    def test_a_merge_that_keeps_nothing_must_come_out_exactly_like_the_snapshot(self):
        """A settlement that kept nothing has to be the snapshot, however the merge was made."""

        before = _state([
            _node("a", _box(0.05, 0.05, 0.4, 0.4)),
            _node("b", _box(0.5, 0.05, 0.4, 0.4)),
        ])
        # Both nodes shrink, so their panels empty out and both of them come out worse.
        after = _state([
            _node("a", _box(0.05, 0.05, 0.2, 0.2)),
            _node("b", _box(0.5, 0.05, 0.2, 0.2)),
        ])
        faithful = plan_partial_rollback(before, after, round_index=41)
        self.assertFalse(faithful["merge_unfaithful"])
        self.assertTrue(faithful["applied"], faithful["summary"])
        self.assertTrue({"a", "b"}.issubset({item["object_id"] for item in faithful["reverted"]}))
        self.assertEqual(
            faithful["merged_state"]["layout"]["nodes"], before["layout"]["nodes"]
        )

        with patch.object(partial_rollback, "_merge_objects", _leaking_merge_objects):
            leaked = plan_partial_rollback(before, after, round_index=41)
        self.assertTrue(leaked["merge_unfaithful"])
        self.assertFalse(leaked["applied"])
        self.assertFalse(leaked["merged_state"])
        self.assertIn("faithful", leaked["summary"])

    def test_a_settlement_whose_merge_is_always_worse_rolls_back_whole(self):
        before = _state([
            _node("a", _box(0.05, 0.05, 0.4, 0.4)),
            _node("b", _box(0.5, 0.5, 0.4, 0.4)),
        ])
        # Both nodes shrink onto one another, so the candidate is worse than the snapshot -- and
        # its panel is emptier -- at every step of the settlement.
        after = _state([
            _node("a", _box(0.05, 0.05, 0.2, 0.2)),
            _node("b", _box(0.05, 0.05, 0.2, 0.2)),
        ])
        with patch.object(partial_rollback, "_merge_objects", _leaking_merge_objects):
            report = plan_partial_rollback(before, after, round_index=42)
        self.assertFalse(report["applied"])
        self.assertEqual(report["tier"], 0)
        self.assertFalse(report["merge_unfaithful"])
        self.assertIn("more faults", report["reason"])
        self.assertEqual([item["accepted"] for item in report["tiers"]], [False, False, False])
        self.assertFalse(report["merged_state"])

    def test_a_report_that_claims_a_merge_it_did_not_make_is_flagged(self):
        errors = validate_partial_rollback(
            {"contract": PARTIAL_ROLLBACK_CONTRACT, "applied": True, "merged_state": {}, "reverted": []}
        )
        self.assertTrue(errors)
        errors_ok = validate_partial_rollback({"contract": "not-ours"})
        self.assertTrue(errors_ok)


if __name__ == "__main__":
    unittest.main()
