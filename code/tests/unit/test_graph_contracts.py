"""Tests for the shared Graph Middleware contracts."""

import unittest

from rfs.contracts import (
    MOTIF_KINDS,
    normalize_graph,
    stable_graph_id,
    validate_graph,
    validate_graph_bindings,
    validate_subgraph,
)


def scientific_graph():
    return normalize_graph(
        {
            "graph_id": "graph.scientific.paper-1",
            "paper_id": "paper-1",
            "nodes": [
                {"id": "sci.input", "kind": "input", "required": True, "evidence_ids": ["E1"]},
                {"id": "sci.encoder", "kind": "module", "required": True, "evidence_ids": ["E2"]},
            ],
            "edges": [
                {"id": "sci.edge.input_encoder", "source": "sci.input", "target": "sci.encoder", "required": True, "evidence_ids": ["E3"]}
            ],
            "required_node_ids": ["sci.input", "sci.encoder"],
            "required_edge_ids": ["sci.edge.input_encoder"],
        },
        kind="scientific",
    )


class GraphContractTests(unittest.TestCase):
    def test_stable_ids_are_deterministic_and_namespaced(self):
        left = stable_graph_id("sci.node", "paper-1", "encoder")
        right = stable_graph_id("sci.node", "paper-1", "encoder")
        self.assertEqual(left, right)
        self.assertTrue(left.startswith("sci.node."))
        self.assertNotEqual(left, stable_graph_id("visual.node", "paper-1", "encoder"))

    def test_valid_scientific_graph_preserves_evidence(self):
        result = validate_graph(scientific_graph(), expected_kind="scientific")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["node_count"], 2)

    def test_duplicate_ids_and_dangling_endpoints_fail(self):
        graph = scientific_graph()
        graph["nodes"].append(dict(graph["nodes"][0]))
        graph["edges"][0]["target"] = "sci.missing"
        result = validate_graph(graph, expected_kind="scientific")
        self.assertFalse(result["ok"])
        self.assertTrue(any("unique" in item for item in result["errors"]))
        self.assertTrue(any("not declared" in item for item in result["errors"]))

    def test_required_scientific_evidence_is_enforced(self):
        graph = scientific_graph()
        graph["nodes"][0]["evidence_ids"] = []
        graph["edges"][0]["evidence_ids"] = []
        result = validate_graph(graph, expected_kind="scientific")
        self.assertIn("required scientific node has no evidence: sci.input", result["errors"])
        self.assertIn("required scientific edge has no evidence: sci.edge.input_encoder", result["errors"])

    def test_required_object_and_expected_kind_are_enforced(self):
        graph = scientific_graph()
        graph["required_node_ids"].append("sci.missing")
        result = validate_graph(graph, expected_kind="visual")
        self.assertIn("graph kind must be visual", result["errors"])
        self.assertIn("required graph node is missing: sci.missing", result["errors"])

    def test_bindings_reject_unknown_cross_layer_objects(self):
        visual = normalize_graph({"graph_id": "graph.visual.paper-1", "paper_id": "paper-1", "nodes": [{"id": "visual.input"}], "edges": []}, kind="visual")
        editable = normalize_graph({"graph_id": "graph.editable.paper-1", "paper_id": "paper-1", "nodes": [{"id": "ppt.input"}], "edges": []}, kind="editable")
        bindings = {
            "binding_set_id": "bindings.paper-1",
            "paper_id": "paper-1",
            "node_bindings": [{"binding_id": "binding.input", "scientific_node_id": "sci.input", "visual_node_ids": ["visual.missing"], "ppt_object_ids": ["ppt.missing"]}],
        }
        result = validate_graph_bindings(bindings, scientific_graph=scientific_graph(), visual_graph=visual, editable_graph=editable)
        self.assertFalse(result["ok"])
        self.assertTrue(any("unknown visual node" in item for item in result["errors"]))
        self.assertTrue(any("unknown PPT object" in item for item in result["errors"]))

    def test_subgraph_references_source_graph(self):
        result = validate_subgraph(
            {
                "subgraph_id": "subgraph.linear",
                "graph_id": "graph.scientific.paper-1",
                "paper_id": "paper-1",
                "graph_kind": "scientific",
                "motif": "linear",
                "node_ids": ["sci.input", "sci.encoder"],
                "edge_ids": ["sci.edge.input_encoder"],
            },
            graph=scientific_graph(),
        )
        self.assertTrue(result["ok"], result["errors"])

    def test_the_motif_vocabulary_carries_a_drawing_that_repeats(self):
        self.assertIn("repeated", MOTIF_KINDS)
        result = validate_subgraph(
            {
                "subgraph_id": "subgraph.repeated",
                "graph_id": "graph.visual.paper-1",
                "paper_id": "paper-1",
                "graph_kind": "visual",
                "motif": "repeated",
                "node_ids": ["vnode.pvcnn"],
            }
        )

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["subgraph"]["motif"], "repeated")


if __name__ == "__main__":
    unittest.main()
