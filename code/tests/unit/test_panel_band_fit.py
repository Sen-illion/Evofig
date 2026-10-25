"""A stage band keeps the template geometry and spreads its payload inside it."""

from __future__ import annotations

import unittest

from rfs.planning.panel_adaptation import adapt_design_panels

TITLE_LANE = 0.075


def _node(node_id: str, panel_id: str | None, box: dict) -> dict:
    return {"id": node_id, "label": node_id.replace("_", " ").title(), "panel_id": panel_id, "bbox_percent": dict(box)}


def _band(panel_id: str, title: str, box: dict) -> dict:
    return {"id": panel_id, "title": title, "role": "stage", "bbox_percent": dict(box)}


def _row_plan() -> dict:
    """The round_05 geometry: two stage bands roomier than their payload."""

    return {
        "panels": [
            _band("stage_planning", "Planning Phase", {"x": .16, "y": .15, "w": .3, "h": .75}),
            _band("stage_searching", "Searching Phase", {"x": .48, "y": .15, "w": .38, "h": .75}),
        ],
        "nodes": [
            _node("input_instruction_text", None, {"x": .02, "y": .4, "w": .12, "h": .2}),
            _node("module_planner", "stage_planning", {"x": .18, "y": .3, "w": .12, "h": .45}),
            _node("intermediate_plans", "stage_planning", {"x": .32, "y": .3, "w": .12, "h": .45}),
            _node("module_tools", "stage_searching", {"x": .5, "y": .233, "w": .34, "h": .117}),
            _node("module_search", "stage_searching", {"x": .5, "y": .4, "w": .15, "h": .15}),
            _node("module_reviser", "stage_searching", {"x": .69, "y": .4, "w": .15, "h": .3}),
        ],
        "connectors": [],
    }


def _spreads(report: dict) -> list[dict]:
    return [item for item in report["compacted"] if item.get("reason") == "payload_spread_into_panel"]


def _node_of(plan: dict, node_id: str) -> dict:
    return next(node["bbox_percent"] for node in plan["nodes"] if node["id"] == node_id)


def _occupancy(panel: dict, nodes: list[dict]) -> float:
    box = panel["bbox_percent"]
    covered = sum(
        node["bbox_percent"]["w"] * node["bbox_percent"]["h"]
        for node in nodes
        if node["panel_id"] == panel["id"]
    )
    return covered / (box["w"] * box["h"])


class StageBandSpreadTests(unittest.TestCase):
    def test_the_band_keeps_the_template_geometry_and_the_cards_grow_into_it(self) -> None:
        before = _row_plan()
        bands = {panel["id"]: dict(panel["bbox_percent"]) for panel in before["panels"]}
        updated, report = adapt_design_panels(before)
        for panel in updated["panels"]:
            self.assertEqual(panel["bbox_percent"], bands[panel["id"]])
        self.assertEqual(
            {item["panel_id"] for item in _spreads(report)},
            {"stage_planning", "stage_searching"},
        )
        planner = _node_of(updated, "module_planner")
        # The highest card never moves up, so the reserved title lane survives,
        # and the block grows down until it reaches the bottom padding.
        self.assertAlmostEqual(planner["y"], 0.3, places=4)
        self.assertAlmostEqual(planner["h"], 0.57, places=4)
        self.assertAlmostEqual(_node_of(updated, "module_reviser")["y"] + _node_of(updated, "module_reviser")["h"], 0.87, places=4)

    def test_spreading_the_payload_can_only_read_denser(self) -> None:
        before = _row_plan()
        was = {panel["id"]: panel for panel in before["panels"]}
        updated, _ = adapt_design_panels(before)
        for panel in updated["panels"]:
            self.assertGreater(
                _occupancy(panel, updated["nodes"]),
                _occupancy(was[panel["id"]], before["nodes"]),
            )

    def test_every_card_stays_inside_its_own_band_below_the_title_lane(self) -> None:
        updated, _ = adapt_design_panels(_row_plan())
        panels = {panel["id"]: panel["bbox_percent"] for panel in updated["panels"]}
        for node in updated["nodes"]:
            panel = panels.get(str(node["panel_id"]))
            if not panel:
                continue
            box = node["bbox_percent"]
            self.assertGreaterEqual(box["y"] - panel["y"], TITLE_LANE - 1e-6)
            self.assertGreaterEqual(box["x"], panel["x"] - 1e-6)
            self.assertLessEqual(box["x"] + box["w"], panel["x"] + panel["w"] + 1e-6)
            self.assertLessEqual(box["y"] + box["h"], panel["y"] + panel["h"] - 0.03 + 1e-6)

    def test_spread_cards_never_overlap_each_other(self) -> None:
        updated, _ = adapt_design_panels(_row_plan())
        boxes = [node["bbox_percent"] for node in updated["nodes"] if node["panel_id"] == "stage_searching"]
        for index, left in enumerate(boxes):
            for right in boxes[index + 1:]:
                disjoint = (
                    left["x"] + left["w"] <= right["x"] + 1e-9
                    or right["x"] + right["w"] <= left["x"] + 1e-9
                    or left["y"] + left["h"] <= right["y"] + 1e-9
                    or right["y"] + right["h"] <= left["y"] + 1e-9
                )
                self.assertTrue(disjoint)

    def test_a_tiny_payload_is_grown_but_never_past_the_growth_ceiling(self) -> None:
        plan = {
            "panels": [_band("p", "Phase", {"x": .16, "y": .15, "w": .3, "h": .75})],
            "nodes": [
                _node("a", "p", {"x": .18, "y": .3, "w": .12, "h": .04}),
                _node("b", "p", {"x": .32, "y": .3, "w": .12, "h": .04}),
            ],
            "connectors": [],
        }
        updated, report = adapt_design_panels(plan)
        spread = _spreads(report)[0]
        self.assertAlmostEqual(spread["scale_y"], 1.6, places=4)
        self.assertAlmostEqual(_node_of(updated, "a")["h"], 0.064, places=4)

    def test_a_payload_that_already_fills_its_band_is_left_alone(self) -> None:
        plan = {
            "panels": [_band("p", "Phase", {"x": .16, "y": .15, "w": .3, "h": .75})],
            "nodes": [
                _node("a", "p", {"x": .18, "y": .225, "w": .12, "h": .645}),
                _node("b", "p", {"x": .32, "y": .225, "w": .12, "h": .645}),
            ],
            "connectors": [],
        }
        updated, report = adapt_design_panels(plan)
        self.assertEqual(_spreads(report), [])
        self.assertAlmostEqual(_node_of(updated, "a")["h"], 0.645, places=4)

    def test_a_payload_overflowing_the_title_lane_is_left_alone(self) -> None:
        plan = {
            "panels": [_band("p", "Phase", {"x": .16, "y": .15, "w": .3, "h": .75})],
            "nodes": [
                _node("a", "p", {"x": .18, "y": .16, "w": .12, "h": .78}),
                _node("b", "p", {"x": .32, "y": .16, "w": .12, "h": .78}),
            ],
            "connectors": [],
        }
        updated, report = adapt_design_panels(plan)
        self.assertEqual(_spreads(report), [])
        # Only the title-lane reservation moved the card; the spread stayed out.
        self.assertAlmostEqual(_node_of(updated, "a")["y"], 0.233, places=4)

    def test_row_mates_keep_sharing_one_band(self) -> None:
        plan = _row_plan()
        for node in plan["nodes"]:
            if node["panel_id"] == "stage_planning":
                node["bbox_percent"]["h"] = .3
        updated, report = adapt_design_panels(plan)
        bands = {panel["id"]: panel["bbox_percent"] for panel in updated["panels"]}
        self.assertEqual(len({box["y"] for box in bands.values()}), 1)
        self.assertEqual(len({box["h"] for box in bands.values()}), 1)
        self.assertEqual({item["panel_id"] for item in _spreads(report)}, {"stage_planning", "stage_searching"})

    def test_a_single_card_payload_is_not_spread(self) -> None:
        plan = {
            "panels": [_band("p", "Phase", {"x": .16, "y": .15, "w": .3, "h": .75})],
            "nodes": [_node("only", "p", {"x": .18, "y": .3, "w": .12, "h": .2})],
            "connectors": [],
        }
        updated, report = adapt_design_panels(plan)
        self.assertEqual(_spreads(report), [])
        self.assertAlmostEqual(_node_of(updated, "only")["w"] > 0.0, True)


if __name__ == "__main__":
    unittest.main()
