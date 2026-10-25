"""Tests for deterministic, evidence-grounded scientific graph construction."""

from __future__ import annotations

import unittest

from rfs.analysis.graph import build_scientific_graph


def _evidence(*ids: str) -> dict:
    return {"evidence": [{"id": item, "page": index + 1, "text": item} for index, item in enumerate(ids)]}


class ScientificGraphTests(unittest.TestCase):
    def test_linear_graph_is_deterministic_and_preserves_required_evidence(self) -> None:
        method = {
            "inputs": [{"id": "input", "label": "Paper", "must_show": True, "evidence_ids": ["E1"]}],
            "modules": [{"id": "encode", "label": "Encoder", "should_render": True, "evidence_ids": ["E2"]}],
            "outputs": [{"id": "result", "label": "Result", "evidence_ids": ["E3"]}],
            "relations": [
                {"id": "r1", "source": "input", "target": "encode", "type": "data_flow", "must_show": True, "evidence_ids": ["E4"]},
                {"id": "r2", "source": "encode", "target": "result", "type": "data_flow", "evidence_ids": ["E5"]},
            ],
            "forbidden_inventions": ["Unsupported decoder"],
        }
        first = build_scientific_graph(method, paper_id="paper-1", evidence_map=_evidence("E1", "E2", "E3", "E4", "E5"))
        second = build_scientific_graph(method, paper_id="paper-1", evidence_map=_evidence("E1", "E2", "E3", "E4", "E5"))
        self.assertEqual(first, second)
        graph, truth, report = first
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(2, len(truth["required_node_ids"]))
        self.assertEqual(1, len(truth["required_edge_ids"]))
        self.assertEqual(["Unsupported decoder"], truth["forbidden_labels"])
        self.assertTrue(all(truth["node_evidence"][item] for item in truth["required_node_ids"]))
        self.assertTrue(all(truth["edge_evidence"][item] for item in truth["required_edge_ids"]))
        self.assertEqual(3, len(graph["nodes"]))

    def test_branch_merge_and_training_inference_phases_are_retained(self) -> None:
        method = {
            "modules": [
                {"id": "root", "label": "Shared backbone", "evidence_ids": ["E1"]},
                {"id": "left", "label": "Training branch", "phase": "training", "evidence_ids": ["E2"]},
                {"id": "right", "label": "Inference branch", "phase": "inference", "evidence_ids": ["E3"]},
                {"id": "merge", "label": "Fusion", "evidence_ids": ["E4"]},
            ],
            "relations": [
                {"source": "root", "target": "left", "type": "branch", "evidence_ids": ["E5"]},
                {"source": "root", "target": "right", "type": "branch", "evidence_ids": ["E6"]},
                {"source": "left", "target": "merge", "type": "merge", "evidence_ids": ["E7"]},
                {"source": "right", "target": "merge", "type": "merge", "evidence_ids": ["E8"]},
            ],
        }
        graph, _, report = build_scientific_graph(method, paper_id="paper-2", evidence_map=_evidence(*[f"E{i}" for i in range(1, 9)]))
        self.assertTrue(report["ok"], report["errors"])
        phases = {item["properties"].get("phase") for item in graph["nodes"]}
        self.assertIn("training", phases)
        self.assertIn("inference", phases)
        self.assertEqual(["branch", "branch", "merge", "merge"], [item["relation_type"] for item in graph["edges"]])

    def test_unknown_evidence_fails_report_without_inventing_nodes(self) -> None:
        method = {"modules": [{"id": "only", "label": "Only claim", "must_show": True, "evidence_ids": ["MISSING"]}]}
        graph, _, report = build_scientific_graph(method, paper_id="paper-3", evidence_map=_evidence("E1"))
        self.assertFalse(report["ok"])
        self.assertEqual(["MISSING"], report["unknown_evidence_ids"])
        self.assertEqual(["Only claim"], [item["label"] for item in graph["nodes"]])

    def test_dangling_relation_is_reported(self) -> None:
        method = {
            "modules": [{"id": "known", "label": "Known", "evidence_ids": ["E1"]}],
            "relations": [{"id": "bad", "source": "known", "target": "missing", "type": "data_flow", "evidence_ids": ["E2"]}],
        }
        _, _, report = build_scientific_graph(method, paper_id="paper-4", evidence_map=_evidence("E1", "E2"))
        self.assertFalse(report["ok"])
        self.assertTrue(any("endpoint is not declared" in item for item in report["errors"]))

    def test_should_render_without_evidence_fails(self) -> None:
        method = {"modules": [{"id": "required", "label": "Required", "should_render": True}]}
        _, _, report = build_scientific_graph(method, paper_id="paper-5", evidence_map=_evidence())
        self.assertFalse(report["ok"])
        self.assertTrue(any("has no evidence" in item for item in report["errors"]))

    def test_the_graph_keeps_no_generation_side_planning_fields(self) -> None:
        """The scientific layer copies what the paper states, not how to draw it.

        ``parent_id``, ``input_ids`` and ``output_ids`` are plan-side shortcuts.
        Keeping them would carry one model's reading of the paper into the layer
        that is supposed to record the paper's own facts, so they are dropped and
        the provenance instead names the document the graph was compiled from.
        """

        method = {
            "inputs": [{"id": "input", "label": "Paper", "evidence_ids": ["E2"]}],
            "modules": [{
                "id": "encode",
                "label": "Encoder",
                "parent_id": "input",
                "input_ids": ["input"],
                "output_ids": ["result"],
                "must_show": True,
                "evidence_ids": ["E1"],
            }],
            "outputs": [{"id": "result", "label": "Result", "evidence_ids": ["E3"]}],
            "relations": [{"id": "r1", "source": "input", "target": "encode", "type": "data_flow", "evidence_ids": ["E4"]}],
        }
        graph, _, report = build_scientific_graph(
            method,
            paper_id="paper-6",
            evidence_map=_evidence("E1", "E2", "E3", "E4"),
            source="figure_specification",
        )
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual("rfs.figure_specification", graph["provenance"]["source_contract"])
        self.assertTrue(all(item["provenance"]["source_contract"] == "figure_specification" for item in graph["nodes"]))
        self.assertEqual("figure_specification.relations", graph["edges"][0]["provenance"]["source_contract"])
        for node in graph["nodes"]:
            for field in ("parent_source_id", "input_source_ids", "output_source_ids"):
                self.assertNotIn(field, node["properties"])
                self.assertNotIn(field, node)
        self.assertNotIn("figure_specification_supplied", graph["provenance"])


if __name__ == "__main__":
    unittest.main()
