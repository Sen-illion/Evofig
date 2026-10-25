"""Cross-layer binding tests for scientific, visual, and editable graphs."""

from __future__ import annotations

import unittest

from rfs.contracts import validate_graph_bindings
from tests.unit.test_editable_graph import _layers
from rfs.composition import build_editable_graph


class GraphBindingTests(unittest.TestCase):
    def test_all_scientific_objects_have_valid_cross_layer_bindings(self) -> None:
        scientific, visual, program = _layers()
        editable, bindings, report = build_editable_graph(program, visual_graph=visual, scientific_graph=scientific)
        self.assertTrue(report["ok"], report["errors"])
        validation = validate_graph_bindings(
            bindings,
            scientific_graph=scientific,
            visual_graph=visual,
            editable_graph=editable,
        )
        self.assertTrue(validation["ok"], validation["errors"])
        self.assertTrue(all(item["status"] == "bound" for item in bindings["node_bindings"]))
        self.assertTrue(all(item["status"] == "bound" for item in bindings["edge_bindings"]))


if __name__ == "__main__":
    unittest.main()
