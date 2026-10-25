"""Tests for pre-generation structure density diagnostics and wireframes."""

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rfs.planning.density_advisory import build_density_advisory
from rfs.planning.structure_wireframe import render_structure_wireframe
from rfs.planning.paper_figure_structure import structure_from_design_plan
from rfs.planning.vlm_structure_planner import (
    _build_structure_prompt,
    _is_substantive_density_replan,
    plan_structure_vlm,
)


class StructureDensityTests(unittest.TestCase):
    def test_structure_prompt_treats_title_band_as_limit_not_fill_target(self):
        prompt = _build_structure_prompt(
            {"modules": [{"id": "m", "name": "Method"}]},
            {},
            {},
            None,
            aspect_ratio="16:9",
            content_top=.11,
            content_bottom=.96,
        )
        self.assertIn("containment boundary, not a target panel height", prompt)
        self.assertIn("never stretch a sparse panel", prompt)
        self.assertEqual(prompt.count(" METHOD="), 1)

    def test_oversized_sparse_panels_require_replan(self):
        result = build_density_advisory({
            "panels": [
                {"id": "left", "bbox_percent": {"x": 0, "y": .2, "w": .2, "h": .6}},
                {"id": "main", "bbox_percent": {"x": .25, "y": .1, "w": .5, "h": .8}},
            ],
            "nodes": [
                {"id": "a", "panel_id": "left", "bbox_percent": {"x": .05, "y": .45, "w": .05, "h": .1}},
                {"id": "b", "panel_id": "main", "bbox_percent": {"x": .45, "y": .4, "w": .1, "h": .1}},
            ],
        })
        self.assertEqual(result["status"], "replan_required")
        self.assertTrue(result["sparse_panels"])

    def test_single_large_panel_is_marked_for_compaction(self):
        result = build_density_advisory({
            "panels": [{"id": "input", "bbox_percent": {"x": 0, "y": .2, "w": .2, "h": .6}}],
            "nodes": [{"id": "a", "panel_id": "input", "bbox_percent": {"x": .05, "y": .45, "w": .05, "h": .1}}],
        })
        self.assertTrue(result["compact_panel_candidates"])
        self.assertEqual(result["recommendation"], "compact_panels")

    def test_delivery_threshold_sparse_panel_triggers_structure_replan(self):
        result = build_density_advisory({
            "density_targets": {"panel_occupancy_min": .45},
            "panels": [
                {"id": "planning", "bbox_percent": {"x": .1, "y": .1, "w": .24, "h": .75}},
                {"id": "searching", "bbox_percent": {"x": .4, "y": .1, "w": .44, "h": .75}},
            ],
            "nodes": [
                {"id": "a", "panel_id": "planning", "bbox_percent": {"x": .12, "y": .2, "w": .2, "h": .2}},
                {"id": "b", "panel_id": "planning", "bbox_percent": {"x": .12, "y": .55, "w": .2, "h": .2}},
                {"id": "c", "panel_id": "searching", "bbox_percent": {"x": .45, "y": .2, "w": .2, "h": .2}},
            ],
        })
        self.assertEqual(result["status"], "replan_required")
        self.assertEqual(result["sparse_panels"][0]["panel_id"], "planning")
        self.assertEqual(result["sparse_panels"][0]["minimum_occupancy"], .45)

    def test_vlm_cannot_lower_delivery_panel_occupancy_floor(self):
        result = build_density_advisory({
            "density_targets": {"panel_occupancy_min": .4},
            "panels": [{"id": "p", "bbox_percent": {"x": .1, "y": .1, "w": .4, "h": .5}}],
            "nodes": [{"id": "n", "panel_id": "p", "bbox_percent": {"x": .15, "y": .2, "w": .2, "h": .4}}],
        })
        self.assertEqual(result["experience_rules_applied"]["panel_occupancy_min"], .45)

    def test_node_inflation_without_panel_compaction_is_not_substantive(self):
        before = {
            "panels": [{"id": "p", "bbox_percent": {"x": .1, "y": .1, "w": .8, "h": .8}}],
            "nodes": [{"id": "n", "bbox_percent": {"x": .2, "y": .2, "w": .1, "h": .1}}],
        }
        candidate = {
            "panels": [{"id": "p", "bbox_percent": {"x": .1, "y": .1, "w": .8, "h": .8}}],
            "nodes": [{"id": "n", "bbox_percent": {"x": .2, "y": .2, "w": .5, "h": .5}}],
        }
        substantive, report = _is_substantive_density_replan(before, candidate)
        self.assertFalse(substantive)
        self.assertFalse(report["panel_compacted"])

    def test_wireframe_is_written(self):
        with tempfile.TemporaryDirectory() as temp:
            path = render_structure_wireframe({"panels": [], "nodes": [], "connectors": []}, Path(temp) / "wire.png")
            self.assertTrue(Path(path).exists())

    def test_requested_canvas_ratio_overrides_vlm_ratio(self):
        result = structure_from_design_plan(
            {
                "canvas": {"aspect_ratio": 2.3, "margin": 0.02},
                "panels": [],
                "nodes": [],
                "connectors": [],
            },
            aspect_ratio="16:9",
        )
        canvas = result["figure_layout_plan"]["canvas"]
        self.assertEqual(canvas["aspect_ratio"], "16:9")
        self.assertEqual(canvas["margin"], 0.02)

    def test_a_renamed_node_still_carries_its_contract_id(self):
        result = structure_from_design_plan(
            {
                "canvas": {"aspect_ratio": 16 / 9, "margin": 0.02},
                "panels": [],
                "nodes": [
                    {
                        "id": "node_extractor",
                        "semantic_source_id": "mod_feature_extractor",
                        "label": "Feature Extractor",
                        "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3},
                    }
                ],
                "connectors": [],
            },
            aspect_ratio="16:9",
        )
        icons = result["icon_generation_plan"]["icons"]
        self.assertEqual(icons[0]["asset_id"], "icon_node_extractor")
        self.assertEqual(icons[0]["semantic_source_id"], "mod_feature_extractor")

    def _method(self) -> dict:
        return {
            "modules": [
                {"id": "a", "name": "Input"},
                {"id": "b", "name": "Reasoner"},
                {"id": "c", "name": "Output"},
            ],
            "relations": [
                {"id": "r1", "source": "a", "target": "b", "type": "data_flow"},
                {"id": "r2", "source": "b", "target": "c", "type": "data_flow"},
            ],
        }

    def _model_plan(self, *, b_box: dict) -> dict:
        """A model design whose only defect is the geometry handed in through b_box."""
        return {
            "canvas": {"aspect_ratio": "16:9", "margin": 0.03},
            "panels": [
                {"id": "p1", "title": "Reasoning Loop", "role": "method", "bbox_percent": {"x": .03, "y": .12, "w": .60, "h": .80}},
                {"id": "p2", "title": "Answer Head", "role": "output", "bbox_percent": {"x": .67, "y": .12, "w": .30, "h": .80}},
            ],
            "nodes": [
                {"id": "a", "semantic_source_id": "a", "label": "Input", "panel_id": "p1", "role": "input", "bbox_percent": {"x": .06, "y": .2, "w": .22, "h": .2}, "evidence_ids": []},
                {"id": "b", "semantic_source_id": "b", "label": "Reasoner", "panel_id": "p1", "role": "module", "bbox_percent": b_box, "evidence_ids": ["E1"]},
                {"id": "c", "semantic_source_id": "c", "label": "Output", "panel_id": "p2", "role": "output", "bbox_percent": {"x": .70, "y": .2, "w": .22, "h": .2}, "evidence_ids": []},
            ],
            "connectors": [
                {"id": "e1", "source": "a", "target": "b", "type": "data_flow"},
                {"id": "e2", "source": "b", "target": "c", "type": "data_flow"},
            ],
            "groups": [],
            "reference_adoption": {"adopted": [], "rejected": []},
            "density_targets": {"nodes": 3, "connectors": 2, "text_items": 3},
        }

    def _assert_no_invented_template(self, plan: dict) -> None:
        dumped = json.dumps(plan, ensure_ascii=False)
        for trace in ("Primary Method", "Supporting Evidence", "panel_top", "panel_bottom", "edge_00", "Core Method"):
            self.assertNotIn(trace, dumped)

    def test_the_structure_planner_carries_no_invented_template(self):
        source = Path(inspect.getsourcefile(plan_structure_vlm)).read_text(encoding="utf-8")
        for trace in ("Primary Method", "Supporting Evidence", "panel_top", "panel_bottom", "Core Method"):
            self.assertNotIn(trace, source)

    def test_invalid_initial_vlm_plan_is_sent_back_to_the_model_with_its_errors(self):
        # The deterministic pass already fits every stage band to its payload.
        method = {**self._method(), "optional_details": [{"id": "d", "name": "Evidence Detail"}]}
        method["relations"].append({"id": "r3", "source": "a", "target": "d", "type": "data_flow"})
        compact = {
            "canvas": {"aspect_ratio": "16:9", "margin": 0.03},
            "panels": [
                {"id": "p1", "title": "Method", "role": "method", "bbox_percent": {"x": .03, "y": .12, "w": .64, "h": .60}, "node_ids": ["a", "b", "d"]},
                {"id": "p2", "title": "Results", "role": "output", "bbox_percent": {"x": .60, "y": .74, "w": .37, "h": .20}, "node_ids": ["c"]},
            ],
            "nodes": [
                {"id": "a", "semantic_source_id": "a", "label": "Input", "panel_id": "p1", "role": "input", "bbox_percent": {"x": .05, "y": .14, "w": .28, "h": .30}, "evidence_ids": []},
                {"id": "b", "semantic_source_id": "b", "label": "Reasoner", "panel_id": "p1", "role": "method", "bbox_percent": {"x": .37, "y": .14, "w": .28, "h": .30}, "evidence_ids": []},
                {"id": "d", "semantic_source_id": "d", "label": "Evidence Detail", "panel_id": "p1", "role": "detail", "bbox_percent": {"x": .05, "y": .50, "w": .28, "h": .10}, "evidence_ids": ["ev-detail"]},
                {"id": "c", "semantic_source_id": "c", "label": "Output", "panel_id": "p2", "role": "output", "bbox_percent": {"x": .67, "y": .76, "w": .28, "h": .16}, "evidence_ids": []},
            ],
            "connectors": [
                {"id": "r1", "source": "a", "target": "b", "type": "data_flow", "label": ""},
                {"id": "r2", "source": "b", "target": "c", "type": "data_flow", "label": ""},
                {"id": "r3", "source": "a", "target": "d", "type": "data_flow", "label": ""},
            ],
            "groups": [],
            "reference_adoption": {"adopted": [], "rejected": []},
            "density_targets": {"nodes": 4, "connectors": 3, "text_items": 4},
        }
        with tempfile.TemporaryDirectory() as temp, patch(
            "rfs.planning.vlm_structure_planner.call_vlm_json",
            side_effect=[{"panels": [], "nodes": []}, compact, compact],
        ) as mocked:
            plan, report = plan_structure_vlm(method, {}, None, wireframe_out=Path(temp) / "wire.png")
        self.assertGreaterEqual(mocked.call_count, 2)
        repair_prompt = mocked.call_args_list[1][0][0]
        self.assertIn("VALIDATION_ERRORS", repair_prompt)
        self.assertIn("design plan requires at least two panels", repair_prompt)
        self.assertIn("design plan requires at least one node", repair_prompt)
        self.assertTrue(report["plan_repair_used"])
        self.assertTrue(report["plan_repair_attempts"][0]["accepted"])
        self.assertFalse(report["continued_with_invalid_plan"])
        self.assertEqual([], report["unrepaired_validation_errors"])
        self.assertEqual({"a", "b", "c", "d"}, {node["id"] for node in plan["nodes"]})
        self.assertEqual(
            {("a", "b"), ("b", "c"), ("a", "d")},
            {(item["source"], item["target"]) for item in plan["connectors"]},
        )
        self._assert_no_invented_template(plan)

    def test_a_slightly_out_of_canvas_plan_is_repaired_instead_of_replaced(self):
        repaired = self._model_plan(b_box={"x": .40, "y": .45, "w": .55, "h": .22})
        with tempfile.TemporaryDirectory() as temp, patch(
            "rfs.planning.vlm_structure_planner.call_vlm_json",
            side_effect=[self._model_plan(b_box={"x": .48, "y": .45, "w": .55, "h": .22}), repaired, repaired],
        ) as mocked:
            plan, report = plan_structure_vlm(self._method(), {}, None, wireframe_out=Path(temp) / "wire.png")
        self.assertGreaterEqual(mocked.call_count, 2)
        repair_prompt = mocked.call_args_list[1][0][0]
        self.assertIn("node[1] requires a legal bbox_percent", repair_prompt)
        self.assertTrue(report["plan_repair_used"])
        self.assertEqual(1, len(report["plan_repair_attempts"]))
        self.assertTrue(report["plan_repair_attempts"][0]["accepted"])
        self.assertFalse(report["continued_with_invalid_plan"])
        self.assertEqual([], report["unrepaired_validation_errors"])
        node = next(item for item in plan["nodes"] if item["id"] == "b")
        self.assertLessEqual(node["bbox_percent"]["x"] + node["bbox_percent"]["w"], 1.001)
        self.assertEqual({"a", "b", "c"}, {item["id"] for item in plan["nodes"]})
        self.assertEqual(
            {("a", "b"), ("b", "c")},
            {(item["source"], item["target"]) for item in plan["connectors"]},
        )
        self._assert_no_invented_template(plan)

    def test_an_unrepairable_plan_continues_with_the_model_design(self):
        model_plan = self._model_plan(b_box={"x": .48, "y": .45, "w": .55, "h": .22})
        with tempfile.TemporaryDirectory() as temp, patch(
            "rfs.planning.vlm_structure_planner.call_vlm_json",
            side_effect=[model_plan, model_plan, model_plan],
        ) as mocked:
            plan, report = plan_structure_vlm(self._method(), {}, None, wireframe_out=Path(temp) / "wire.png")
        self.assertGreaterEqual(mocked.call_count, 2)
        self.assertTrue(report["plan_repair_used"])
        self.assertFalse(report["plan_repair_attempts"][0]["accepted"])
        self.assertTrue(report["continued_with_invalid_plan"])
        self.assertTrue(report["fallback_used"])
        self.assertTrue(report["unrepaired_validation_errors"])
        # The model's own science survives: no node and no relation is swapped out.
        self.assertEqual({"a", "b", "c"}, {item["id"] for item in plan["nodes"]})
        self.assertEqual(
            {("a", "b"), ("b", "c")},
            {(item["source"], item["target"]) for item in plan["connectors"]},
        )
        self._assert_no_invented_template(plan)

    def test_an_empty_plan_is_asked_once_more_and_then_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "rfs.planning.vlm_structure_planner.call_vlm_json",
            side_effect=[{}, {}],
        ) as mocked:
            with self.assertRaises(RuntimeError) as raised:
                plan_structure_vlm(self._method(), {}, None, wireframe_out=Path(temp) / "wire.png")
        self.assertEqual(2, mocked.call_count)
        self.assertEqual(mocked.call_args_list[0][0][0], mocked.call_args_list[1][0][0])
        self.assertIn("empty design plan", str(raised.exception))

    def test_bulk_figures_keeps_the_first_plan_without_a_repair_call(self):
        """Bulk corpora buy one design call, not a second one to polish it."""

        model_plan = self._model_plan(b_box={"x": .48, "y": .45, "w": .55, "h": .22})
        with tempfile.TemporaryDirectory() as temp, patch(
            "rfs.planning.vlm_structure_planner.call_vlm_json",
            side_effect=[model_plan],
        ) as mocked:
            plan, report = plan_structure_vlm(
                self._method(), {}, None, wireframe_out=Path(temp) / "wire.png", bulk_figures=True
            )
        self.assertEqual(1, mocked.call_count)
        self.assertTrue(report["bulk_figures"])
        self.assertTrue(report["plan_repair_skipped"])
        self.assertFalse(report["plan_repair_used"])
        self.assertTrue(report["continued_with_invalid_plan"])
        # The model's own science is carried forward unchanged.
        self.assertEqual({"a", "b", "c"}, {item["id"] for item in plan["nodes"]})
        self.assertEqual(
            {("a", "b"), ("b", "c")},
            {(item["source"], item["target"]) for item in plan["connectors"]},
        )
        self._assert_no_invented_template(plan)

    def test_bulk_figures_does_not_pay_for_a_sparse_panel_replan(self):
        replan_required = {"status": "replan_required", "sparse_panels": [{"panel_id": "p2", "occupancy": 0.2}], "node_occupancy_estimate": 0.2}
        with tempfile.TemporaryDirectory() as temp, patch(
            "rfs.planning.vlm_structure_planner.call_vlm_json",
            side_effect=[self._model_plan(b_box={"x": .40, "y": .45, "w": .55, "h": .22})],
        ) as mocked, patch(
            "rfs.planning.vlm_structure_planner.build_density_advisory",
            return_value=dict(replan_required),
        ):
            plan, report = plan_structure_vlm(
                self._method(), {}, None, wireframe_out=Path(temp) / "wire.png", bulk_figures=True
            )
        self.assertEqual(1, mocked.call_count)
        self.assertTrue(report["density_replan_skipped"])
        self.assertEqual([], report["replan_attempts"])
        self.assertIsNone(report["wireframe_path"])
        self.assertEqual({"a", "b", "c"}, {item["id"] for item in plan["nodes"]})

    def test_a_frozen_replay_is_not_replanned_by_a_fresh_model_call(self):
        """A frozen layout is delivered as it was: a new call would unfreeze it."""

        replan_required = {"status": "replan_required", "sparse_panels": [{"panel_id": "p2", "occupancy": 0.2}], "node_occupancy_estimate": 0.2}
        seed = self._model_plan(b_box={"x": .48, "y": .45, "w": .55, "h": .22})
        with tempfile.TemporaryDirectory() as temp, patch(
            "rfs.planning.vlm_structure_planner.call_vlm_json",
            side_effect=AssertionError("a frozen replay must not plan the layout again"),
        ) as mocked, patch(
            "rfs.planning.vlm_structure_planner.build_density_advisory",
            return_value=dict(replan_required),
        ):
            plan, report = plan_structure_vlm(
                self._method(),
                {},
                None,
                wireframe_out=Path(temp) / "wire.png",
                initial_plan=seed,
                frozen_layout=True,
            )
        self.assertEqual(0, mocked.call_count)
        self.assertEqual("replay_seed", report["mode"])
        self.assertTrue(report["initial_plan_replayed"])
        self.assertTrue(report["density_replan_skipped"])
        self.assertEqual([], report["replan_attempts"])
        self.assertIsNone(report["wireframe_path"])
        self.assertEqual({"a", "b", "c"}, {item["id"] for item in plan["nodes"]})

    def test_a_seeded_replan_still_asks_the_model_for_a_denser_layout(self):
        """The upstream retry hands in the current plan and expects a new one back."""

        replan_required = {"status": "replan_required", "sparse_panels": [{"panel_id": "p2", "occupancy": 0.2}], "node_occupancy_estimate": 0.2}
        seed = self._model_plan(b_box={"x": .48, "y": .45, "w": .55, "h": .22})
        denser = self._model_plan(b_box={"x": .40, "y": .45, "w": .55, "h": .30})
        with tempfile.TemporaryDirectory() as temp, patch(
            "rfs.planning.vlm_structure_planner.call_vlm_json",
            side_effect=[denser, denser],
        ) as mocked, patch(
            "rfs.planning.vlm_structure_planner.build_density_advisory",
            return_value=dict(replan_required),
        ):
            plan, report = plan_structure_vlm(
                self._method(), {}, None, wireframe_out=Path(temp) / "wire.png", initial_plan=seed
            )
        self.assertGreaterEqual(mocked.call_count, 1)
        self.assertTrue(report["initial_plan_replayed"])
        self.assertFalse(report["density_replan_skipped"])
        self.assertTrue(report["replan_attempts"])
        self.assertEqual({"a", "b", "c"}, {item["id"] for item in plan["nodes"]})


_LABELS = {"a": "Input", "b": "Reasoner", "c": "Output"}
_BOXES = {
    "a": {"x": .06, "y": .20, "w": .22, "h": .18},
    "b": {"x": .06, "y": .45, "w": .22, "h": .18},
    "c": {"x": .70, "y": .20, "w": .22, "h": .18},
}


class RequiredContentCoverageTests(unittest.TestCase):
    """A thin design is measured against the method contract before it is drawn."""

    def _method(self) -> dict:
        return {
            "modules": [
                {"id": key, "label": label, "must_show": True, "should_render": True}
                for key, label in _LABELS.items()
            ],
            "relations": [{"id": "r1", "source": "a", "target": "b", "type": "data_flow"}],
        }

    def _plan(self, covered) -> dict:
        """A legal plan that draws exactly the named method objects."""
        nodes = [
            {
                "id": key,
                "semantic_source_id": key,
                "label": _LABELS[key],
                "panel_id": "p2" if key == "c" else "p1",
                "bbox_percent": dict(_BOXES[key]),
                "evidence_ids": ["E1"],
            }
            for key in covered
        ]
        connectors = (
            [{"id": "e1", "source": "a", "target": "b", "type": "data_flow"}]
            if {"a", "b"} <= set(covered)
            else []
        )
        return {
            "panels": [
                {"id": "p1", "title": "Method", "bbox_percent": {"x": .03, "y": .12, "w": .60, "h": .80}},
                {"id": "p2", "title": "Answer", "bbox_percent": {"x": .67, "y": .12, "w": .30, "h": .80}},
            ],
            "nodes": nodes,
            "connectors": connectors,
            "groups": [],
            "reference_adoption": {"adopted": [], "rejected": []},
            "density_targets": {},
        }

    def _dense_enough(self):
        return patch(
            "rfs.planning.vlm_structure_planner.build_density_advisory",
            return_value={"status": "pass", "node_occupancy_estimate": 0.6, "sparse_panels": []},
        )

    def _run(self, plans):
        with tempfile.TemporaryDirectory() as temp, patch(
            "rfs.planning.vlm_structure_planner.call_vlm_json",
            side_effect=[self._plan(covered) for covered in plans],
        ) as mocked, self._dense_enough():
            plan, report = plan_structure_vlm(
                self._method(), {}, None, wireframe_out=Path(temp) / "wire.png"
            )
        return plan, report, mocked

    def test_a_design_that_draws_every_required_object_is_not_sent_back(self):
        plan, report, mocked = self._run([["a", "b", "c"]])

        self.assertEqual(1, mocked.call_count)
        self.assertFalse(report["plan_repair_used"])
        self.assertTrue(report["method_coverage_ok"])
        self.assertEqual(report["method_coverage"]["required_count"], 3)
        self.assertFalse(report["continued_with_invalid_plan"])

    def test_a_missing_required_object_reaches_the_repair_prompt_by_name(self):
        plan, report, mocked = self._run([["a", "b"], ["a", "b", "c"]])

        self.assertEqual(2, mocked.call_count)
        repair_prompt = mocked.call_args_list[1][0][0]
        self.assertIn("design plan represents 2 of 3 required method objects; missing: Output", repair_prompt)
        self.assertIn('MISSING_REQUIRED_METHOD_OBJECTS=[{"id":"c","label":"Output"}]', repair_prompt)
        self.assertIn("The only content you may add is a node MISSING_REQUIRED_METHOD_OBJECTS names", repair_prompt)
        self.assertTrue(report["plan_repair_attempts"][0]["accepted"])
        self.assertTrue(report["method_coverage_ok"])
        self.assertEqual({"a", "b", "c"}, {node["id"] for node in plan["nodes"]})

    def test_a_repair_that_draws_more_content_is_kept_and_the_rest_is_drawn_by_the_contract(self):
        plan, report, _mocked = self._run([["a"], ["a", "b"]])

        attempt = report["plan_repair_attempts"][0]
        self.assertTrue(attempt["covered_more_required_content"])
        self.assertTrue(attempt["accepted"])
        # The repair was an improvement but not a complete one, so the object it still owed is
        # drawn from the contract instead of being delivered missing.
        self.assertEqual(1, report["method_object_completion"]["added_count"])
        self.assertEqual(["c"], [item["id"] for item in report["method_object_completion"]["added"]])
        self.assertTrue(report["method_coverage_ok"])
        self.assertEqual(report["method_coverage"]["covered_count"], 3)
        self.assertEqual({"a", "b", "c"}, {node["id"] for node in plan["nodes"]})

    def test_a_repair_that_draws_nothing_more_is_discarded(self):
        plan, report, _mocked = self._run([["a", "b"], ["a", "b"]])

        attempt = report["plan_repair_attempts"][0]
        self.assertFalse(attempt["accepted"])
        self.assertFalse(attempt["covered_more_required_content"])
        # The repair added nothing, so the object the plan still owes is drawn by the contract.
        self.assertEqual(["c"], [item["id"] for item in report["method_object_completion"]["added"]])
        self.assertEqual({"a", "b", "c"}, {node["id"] for node in plan["nodes"]})

    def test_the_object_the_contract_draws_lands_inside_its_panel(self):
        plan, report, _mocked = self._run([["a", "b"], ["a", "b"]])

        added = [node for node in plan["nodes"] if node.get("added_by") == "contract_node_completion"]
        self.assertEqual(["c"], [node["id"] for node in added])
        node = added[0]
        self.assertEqual("Output", node["label"])
        self.assertEqual(node["id"], node["semantic_source_id"])
        panel = next(item for item in plan["panels"] if item["id"] == node["panel_id"])
        box, panel_box = node["bbox_percent"], panel["bbox_percent"]
        centre_x = float(box["x"]) + float(box["w"]) / 2
        centre_y = float(box["y"]) + float(box["h"]) / 2
        self.assertLessEqual(float(panel_box["x"]), centre_x)
        self.assertLessEqual(centre_x, float(panel_box["x"]) + float(panel_box["w"]))
        self.assertLessEqual(float(panel_box["y"]), centre_y)
        self.assertLessEqual(centre_y, float(panel_box["y"]) + float(panel_box["h"]))
        drawn = [item["bbox_percent"] for item in plan["nodes"] if item is not node and item.get("panel_id") == node["panel_id"]]
        for other in drawn:
            apart = (
                float(box["x"]) + float(box["w"]) + 0.012 <= float(other["x"])
                or float(other["x"]) + float(other["w"]) + 0.012 <= float(box["x"])
                or float(box["y"]) + float(box["h"]) + 0.012 <= float(other["y"])
                or float(other["y"]) + float(other["h"]) + 0.012 <= float(box["y"])
            )
            self.assertTrue(apart, "the drawn object was placed on top of a node the model drew")
        self.assertEqual([], report["method_object_completion"]["crowded_ids"])

if __name__ == "__main__":
    unittest.main()
