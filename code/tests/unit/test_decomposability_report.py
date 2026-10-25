"""Tests for the deterministic decomposability measurement of a compiled candidate."""

import unittest

from rfs.contracts import (
    DECOMPOSABILITY_CONTRACT,
    DECOMPOSABILITY_METRICS,
    DECOMPOSABILITY_MIN_PANEL_OCCUPANCY,
    normalize_decomposability_report,
    validate_decomposability_report,
)
from rfs.evaluation.decomposability import build_decomposability_report
from rfs.planning.density_advisory import DELIVERY_MIN_PANEL_OCCUPANCY


def _panels() -> list[dict]:
    return [
        {"id": "panel_a", "bbox_percent": {"x": 0.05, "y": 0.15, "w": 0.42, "h": 0.6}},
        {"id": "panel_b", "bbox_percent": {"x": 0.53, "y": 0.15, "w": 0.42, "h": 0.6}},
    ]


def _nodes() -> list[dict]:
    return [
        {"id": "n1", "panel_id": "panel_a", "bbox_percent": {"x": 0.08, "y": 0.2, "w": 0.3, "h": 0.2}},
        {"id": "n2", "panel_id": "panel_a", "bbox_percent": {"x": 0.08, "y": 0.45, "w": 0.3, "h": 0.2}},
        {"id": "n3", "panel_id": "panel_b", "bbox_percent": {"x": 0.56, "y": 0.2, "w": 0.3, "h": 0.2}},
        {"id": "n4", "panel_id": "panel_b", "bbox_percent": {"x": 0.56, "y": 0.45, "w": 0.3, "h": 0.2}},
    ]


def _program() -> dict:
    return {
        "panels": _panels(),
        "semantic_nodes": _nodes(),
        "slots": [
            {"id": "slot_n1", "parent_id": "n1", "bbox_percent": {"x": 0.1, "y": 0.22, "w": 0.2, "h": 0.1}},
            {"id": "slot_n2", "parent_id": "n2", "bbox_percent": {"x": 0.1, "y": 0.47, "w": 0.2, "h": 0.1}},
            {"id": "slot_n3", "parent_id": "n3", "bbox_percent": {"x": 0.58, "y": 0.22, "w": 0.2, "h": 0.1}},
            {"id": "slot_n4", "parent_id": "n4", "bbox_percent": {"x": 0.58, "y": 0.47, "w": 0.2, "h": 0.1}},
        ],
        "arrows": [
            {"id": "e1", "source": "n1", "target": "n2"},
            {"id": "e2", "source": "n3", "target": "n4"},
        ],
        "text_program": {"items": [{"id": "t1", "owner_node_id": "n1"}]},
    }


def _graphs(*, unbound: bool = False) -> tuple[dict, dict]:
    visual = {
        "graph_id": "visual_graph.test",
        "nodes": [
            {"id": "v1", "scientific_node_ids": ["s1"]},
            {"id": "v2", "scientific_node_ids": ["s2"]},
        ],
    }
    editable = {
        "graph_id": "editable_graph.test",
        "nodes": [{"id": "p1", "scientific_node_ids": ["s1"] if unbound else ["s1", "s2"]}],
    }
    return editable, visual


class DecomposabilityReportTests(unittest.TestCase):
    def test_clean_candidate_is_decomposable(self):
        editable, visual = _graphs()
        report = build_decomposability_report(editable, visual, _program(), round_index=0)
        self.assertEqual(report["contract"], DECOMPOSABILITY_CONTRACT)
        self.assertEqual(report["status"], "decomposable")
        self.assertEqual(report["violations"], [])
        for name in DECOMPOSABILITY_METRICS:
            self.assertIsInstance(report["metrics"][name], (int, float), name)
        self.assertEqual(report["metrics"]["orphan_nodes"], 0)
        self.assertEqual(report["metrics"]["slot_boundary_overflows"], 0)
        self.assertEqual(report["metrics"]["unbound_scientific_nodes"], 0)
        self.assertEqual(report["metrics"]["cross_panel_connectors"], 0)

    def test_panel_occupancy_floor_matches_the_density_advisory(self):
        self.assertEqual(DECOMPOSABILITY_MIN_PANEL_OCCUPANCY, DELIVERY_MIN_PANEL_OCCUPANCY)

    def test_ungrouped_node_is_a_hard_break(self):
        program = _program()
        program["semantic_nodes"][0] = {**program["semantic_nodes"][0], "panel_id": ""}
        report = build_decomposability_report(*_graphs(), program, round_index=1)
        self.assertEqual(report["status"], "hard_violations")
        metrics = [item["metric"] for item in report["violations"]]
        self.assertIn("orphan_nodes", metrics)
        orphan = next(item for item in report["violations"] if item["metric"] == "orphan_nodes")
        self.assertEqual(orphan["object_ids"], ["n1"])

    def test_slot_outside_its_owner_is_a_hard_break(self):
        program = _program()
        program["slots"][0] = {**program["slots"][0], "bbox_percent": {"x": 0.6, "y": 0.2, "w": 0.2, "h": 0.1}}
        report = build_decomposability_report(*_graphs(), program, round_index=1)
        metrics = [item["metric"] for item in report["violations"]]
        self.assertIn("slot_boundary_overflows", metrics)

    def test_unbound_scientific_node_is_a_hard_break(self):
        report = build_decomposability_report(*_graphs(unbound=True), _program(), round_index=2)
        self.assertEqual(report["metrics"]["unbound_scientific_nodes"], 1)
        self.assertEqual(report["status"], "hard_violations")

    def test_cross_panel_connectors_and_sparse_panel_are_soft(self):
        program = _program()
        program["arrows"] = [
            {"id": "e1", "source": "n1", "target": "n3"},
            {"id": "e2", "source": "n1", "target": "n4"},
            {"id": "e3", "source": "n2", "target": "n3"},
        ]
        program["semantic_nodes"] = [
            {**node, "bbox_percent": {**node["bbox_percent"], "w": 0.2, "h": 0.1}}
            for node in program["semantic_nodes"]
        ]
        program["slots"] = [
            {
                **slot,
                "bbox_percent": {
                    "x": next(node["bbox_percent"]["x"] for node in program["semantic_nodes"] if node["id"] == slot["parent_id"]) + 0.01,
                    "y": next(node["bbox_percent"]["y"] for node in program["semantic_nodes"] if node["id"] == slot["parent_id"]) + 0.01,
                    "w": 0.1,
                    "h": 0.05,
                },
            }
            for slot in program["slots"]
        ]
        report = build_decomposability_report(*_graphs(), program, round_index=0)
        self.assertEqual(report["status"], "soft_violations")
        kinds = {item["metric"]: item["kind"] for item in report["violations"]}
        self.assertEqual(kinds.get("cross_panel_connectors"), "soft")
        self.assertEqual(kinds.get("min_panel_occupancy"), "soft")

    def test_explicit_panels_keep_the_layout_panels_in_play(self):
        program = _program()
        program["panels"] = []
        report = build_decomposability_report(*_graphs(), program, panels=_panels(), round_index=0)
        self.assertEqual(report["metrics"]["orphan_nodes"], 0)
        self.assertEqual(report["provenance"]["panel_source"], "declared")

    def test_panels_are_synthesised_when_the_run_declares_none(self):
        program = _program()
        program["panels"] = []
        report = build_decomposability_report(*_graphs(), program, panels=[], round_index=0)
        self.assertEqual(report["provenance"]["panel_source"], "synthesized")
        self.assertEqual(report["provenance"]["panel_count"], 2)
        self.assertLess(report["metrics"]["empty_region_ratio"], 1.0)
        self.assertGreater(report["metrics"]["min_panel_occupancy"], 0.0)
        self.assertEqual(report["metrics"]["orphan_nodes"], 0)

    def test_soft_streak_counts_consecutive_rounds(self):
        program = _program()
        program["arrows"] = [
            {"id": "e1", "source": "n1", "target": "n3"},
            {"id": "e2", "source": "n1", "target": "n4"},
            {"id": "e3", "source": "n2", "target": "n3"},
        ]
        first = build_decomposability_report(*_graphs(), program, round_index=0)
        self.assertEqual(first["repeat"]["soft_streak"], 1)
        second = build_decomposability_report(*_graphs(), program, round_index=1, history=[first])
        self.assertEqual(second["repeat"]["soft_streak"], 2)
        clean = build_decomposability_report(*_graphs(), _program(), round_index=2, history=[first, second])
        self.assertEqual(clean["repeat"]["soft_streak"], 0)

    def test_report_is_deterministic(self):
        args = (*_graphs(), _program())
        self.assertEqual(
            build_decomposability_report(*args, round_index=3),
            build_decomposability_report(*args, round_index=3),
        )

    def test_contract_validation_flags_a_missing_metric(self):
        report = build_decomposability_report(*_graphs(), _program(), round_index=0)
        self.assertTrue(validate_decomposability_report(report)["ok"])
        broken = {**report, "metrics": {key: value for key, value in report["metrics"].items() if key != "orphan_nodes"}}
        result = validate_decomposability_report(broken)
        self.assertFalse(result["ok"])
        self.assertTrue(any("orphan_nodes" in error for error in result["errors"]))

    def test_normalize_fills_every_metric_slot(self):
        normalized = normalize_decomposability_report({"contract": DECOMPOSABILITY_CONTRACT, "metrics": {"orphan_nodes": 1}})
        self.assertEqual(sorted(normalized["metrics"]), sorted(DECOMPOSABILITY_METRICS))
        self.assertEqual(normalized["metrics"]["orphan_nodes"], 1)
        self.assertIsNone(normalized["metrics"]["cross_panel_connectors"])

if __name__ == "__main__":
    unittest.main()
