"""A composition overlay may restyle objects but never break their geometry."""

from __future__ import annotations

import unittest

from rfs.composition.overlay_geometry import guard_overlay_geometry


def _panel(panel_id: str, box: dict) -> dict:
    return {"id": panel_id, "title": panel_id.title(), "role": "stage", "bbox_percent": dict(box)}


def _node(node_id: str, panel_id: str | None, box: dict) -> dict:
    return {"id": node_id, "label": node_id.title(), "panel_id": panel_id, "bbox_percent": dict(box)}


def _layout() -> dict:
    return {
        "panels": [
            _panel("stage_a", {"x": .10, "y": .15, "w": .30, "h": .70}),
            _panel("stage_b", {"x": .50, "y": .15, "w": .35, "h": .70}),
        ],
        "nodes": [
            _node("first", "stage_a", {"x": .12, "y": .25, "w": .11, "h": .50}),
            _node("second", "stage_a", {"x": .26, "y": .25, "w": .11, "h": .50}),
            _node("third", "stage_b", {"x": .52, "y": .25, "w": .13, "h": .20}),
            _node("fourth", "stage_b", {"x": .52, "y": .55, "w": .13, "h": .20}),
        ],
        "connectors": [],
    }


def _box(plan: dict, node_id: str) -> dict:
    return next(node["bbox_percent"] for node in plan["nodes"] if node["id"] == node_id)


def _panel_box(plan: dict, panel_id: str) -> dict:
    return next(panel["bbox_percent"] for panel in plan["panels"] if panel["id"] == panel_id)


class OverlayGeometryGuardTests(unittest.TestCase):
    def test_an_overlay_that_keeps_the_layout_intact_is_taken_as_is(self) -> None:
        layout = _layout()
        guarded, report = guard_overlay_geometry(
            layout,
            node_overrides={"first": {"label": "First card", "bbox_percent": {"x": .125, "y": .26, "w": .10, "h": .48}}},
        )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(_box(guarded, "first"), {"x": .125, "y": .26, "w": .10, "h": .48})
        self.assertEqual(guarded["nodes"][0]["label"], "First card")

    def test_a_node_box_that_collides_is_rejected_but_its_style_survives(self) -> None:
        layout = _layout()
        guarded, report = guard_overlay_geometry(
            layout,
            node_overrides={"third": {"label": "Renamed", "font_size_pt": 15, "bbox_percent": {"x": .52, "y": .60, "w": .13, "h": .20}}},
        )
        self.assertEqual(report["rejected_count"], 1)
        self.assertEqual(report["rejected"][0]["object_id"], "third")
        self.assertEqual(report["rejected"][0]["kept_bbox_percent"], {"x": .52, "y": .25, "w": .13, "h": .20})
        self.assertEqual(_box(guarded, "third"), {"x": .52, "y": .25, "w": .13, "h": .20})
        third = next(node for node in guarded["nodes"] if node["id"] == "third")
        self.assertEqual(third["label"], "Renamed")
        self.assertEqual(third["font_size_pt"], 15)

    def test_a_node_box_that_escapes_its_panel_is_rejected(self) -> None:
        layout = _layout()
        guarded, report = guard_overlay_geometry(
            layout,
            node_overrides={"fourth": {"bbox_percent": {"x": .52, "y": .55, "w": .13, "h": .35}}},
        )
        self.assertEqual(report["rejected_count"], 1)
        self.assertEqual(_box(guarded, "fourth"), {"x": .52, "y": .55, "w": .13, "h": .20})

    def test_a_node_box_off_the_canvas_is_rejected(self) -> None:
        layout = _layout()
        guarded, report = guard_overlay_geometry(
            layout,
            node_overrides={"first": {"bbox_percent": {"x": .12, "y": .25, "w": .95, "h": .50}}},
        )
        self.assertEqual(report["rejected_count"], 1)
        self.assertEqual(_box(guarded, "first"), {"x": .12, "y": .25, "w": .11, "h": .50})

    def test_a_panel_box_that_would_drop_its_payload_is_rejected(self) -> None:
        layout = _layout()
        guarded, report = guard_overlay_geometry(
            layout,
            panel_overrides={"stage_a": {"bbox_percent": {"x": .10, "y": .15, "w": .30, "h": .30}}},
        )
        self.assertEqual(report["rejected_count"], 1)
        self.assertEqual(report["rejected"][0]["object_type"], "panel")
        self.assertEqual(_panel_box(guarded, "stage_a"), {"x": .10, "y": .15, "w": .30, "h": .70})

    def test_a_panel_box_that_collides_with_its_neighbour_is_rejected(self) -> None:
        layout = _layout()
        guarded, report = guard_overlay_geometry(
            layout,
            panel_overrides={"stage_a": {"bbox_percent": {"x": .10, "y": .15, "w": .60, "h": .70}}},
        )
        self.assertEqual(report["rejected_count"], 1)
        self.assertEqual(_panel_box(guarded, "stage_a"), {"x": .10, "y": .15, "w": .30, "h": .70})

    def test_a_panel_box_that_moves_without_breaking_anything_is_kept(self) -> None:
        layout = _layout()
        guarded, report = guard_overlay_geometry(
            layout,
            panel_overrides={"stage_b": {"bbox_percent": {"x": .49, "y": .15, "w": .36, "h": .72}, "title": "Stage B"}},
        )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(_panel_box(guarded, "stage_b"), {"x": .49, "y": .15, "w": .36, "h": .72})

    def test_no_overlay_leaves_the_layout_untouched(self) -> None:
        layout = _layout()
        guarded, report = guard_overlay_geometry(layout)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(guarded["panels"], layout["panels"])
        self.assertEqual(guarded["nodes"], layout["nodes"])

    def test_objects_without_an_overlay_are_never_blamed(self) -> None:
        layout = _layout()
        guarded, report = guard_overlay_geometry(
            layout,
            node_overrides={"third": {"bbox_percent": {"x": .52, "y": .60, "w": .13, "h": .20}}},
        )
        self.assertEqual({item["object_id"] for item in report["rejected"]}, {"third"})
        self.assertEqual(_box(guarded, "fourth"), {"x": .52, "y": .55, "w": .13, "h": .20})


if __name__ == "__main__":
    unittest.main()
