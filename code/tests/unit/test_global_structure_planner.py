"""Tests for pre-generation global structure planning."""

import unittest

from rfs.planning import build_global_structure_plan
from rfs.planning.layout_variants import build_layout_variants
from rfs.planning.template_mapping import build_template_semantic_mapping


class GlobalStructurePlannerTests(unittest.TestCase):
    def test_builds_shared_nodes_slots_and_relations(self):
        global_plan, segmentation, validation = build_global_structure_plan({
            "topology": "branch",
            "inputs": [{"id": "input", "name": "Input"}],
            "modules": [{"id": "encoder", "name": "Encoder"}],
            "outputs": [{"id": "output", "name": "Output"}],
            "relations": [{"source": "input", "target": "encoder"}, {"source": "encoder", "target": "output"}],
        })
        self.assertTrue(validation["ok"])
        self.assertEqual(len(global_plan["nodes"]), 3)
        self.assertEqual(len(global_plan["relations"]), 2)
        self.assertEqual(len(segmentation["slots"]), 3)
        self.assertTrue(all(slot["text_policy"] == "native_ppt" for slot in segmentation["slots"]))

    def test_incomplete_semantic_plan_cannot_drop_required_entities(self):
        global_plan, _segmentation, validation = build_global_structure_plan(
            {
                "inputs": [{"id": "query", "name": "Query"}],
                "modules": [{"id": "agent_a", "name": "Agent A"}, {"id": "agent_b", "name": "Agent B"}],
                "outputs": [{"id": "answer", "name": "Answer"}],
                "relations": [{"source": "query", "target": "agent_a"}, {"source": "agent_a", "target": "answer"}],
            },
            {"nodes": [{"id": "query", "semantic_source_id": "query", "bbox_percent": {"x": .05, "y": .3, "w": .15, "h": .2}}]},
        )
        self.assertTrue(validation["coverage"]["ok"])
        self.assertEqual({node["id"] for node in global_plan["nodes"]}, {"query", "agent_a", "agent_b", "answer"})
        self.assertEqual({node["group_id"] for node in global_plan["nodes"]}, {"input_group", "parallel_module_group", "output_group"})

    def test_layout_variants_preserve_scientific_graph(self):
        base, _segmentation, _validation = build_global_structure_plan({
            "inputs": [{"id": "input", "name": "Input"}],
            "modules": [{"id": "core", "name": "Core"}],
            "outputs": [{"id": "output", "name": "Output"}],
            "relations": [{"source": "input", "target": "core"}, {"source": "core", "target": "output"}],
        })
        variants = build_layout_variants(base)
        self.assertEqual([item["variant_id"] for item in variants], ["balanced", "compact", "hierarchy_emphasis"])
        for item in variants:
            self.assertEqual(item["plan"]["relations"], base["relations"])
            self.assertFalse(item["density_advisory"]["blocking"])

    def test_template_mapping_reports_canvas_reflow_without_copying_semantics(self):
        base, _segmentation, _validation = build_global_structure_plan({
            "inputs": [{"id": "input", "name": "Input"}],
            "outputs": [{"id": "output", "name": "Output"}],
        })
        mapping, reflow = build_template_semantic_mapping(
            {"template_id": "wide", "source_aspect_ratio": 2.5, "panels": [{"id": "reference_left"}, {"id": "reference_right"}]},
            base,
            "3:2",
        )
        self.assertTrue(reflow["reflow_required"])
        self.assertTrue(mapping["mappings"])
        self.assertIn("scientific_labels", mapping["mappings"][0]["reject"])


if __name__ == "__main__":
    unittest.main()
