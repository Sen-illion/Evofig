"""Tests that the method reading prompt forbids undeclared names in the port lists.

The reader wrote handoff names into ``input_ids``/``output_ids`` without ever
being told to declare them; 180 of the 223 delivered decks carried such a name
and were thrown away whole over it. The prompt now states the rule.
"""

import unittest

from rfs.analysis.paper import method_extractor
from rfs.contracts.vlm_context import compact_method


class MethodExtractionPromptTests(unittest.TestCase):
    def _prompt(self) -> str:
        captured: dict[str, str] = {}

        def fake_call(prompt, images, **kwargs):
            captured["prompt"] = prompt
            return {"figure_goal": "show it", "modules": [{"id": "m", "label": "M", "evidence_ids": ["E1"]}]}

        original = method_extractor.call_vlm_json
        method_extractor.call_vlm_json = fake_call
        try:
            method_extractor.extract_method_figure_content(
                {"figure_goal": "show it", "modules": [{"id": "m", "label": "M", "evidence_ids": ["E1"]}]},
                mode="vlm",
            )
        finally:
            method_extractor.call_vlm_json = original
        return captured["prompt"]

    def test_the_prompt_allows_only_declared_ids_in_the_two_port_lists(self):
        prompt = self._prompt()
        self.assertIn("input_ids and output_ids may only repeat ids declared as method objects", prompt)
        self.assertIn("Never write an undeclared product name", prompt)
        self.assertIn("should_render=false", prompt)


PANEL_TARGET = {
    "available": True,
    "reason": "",
    "source": "figure_specification.panel_graphs",
    "panel_count": 2,
    "label_count": 3,
    "panels": [
        {"panel_id": "panel_a", "panel_label": "a", "title": "Overview", "labels": ["Encoder", "Decoder"], "node_count": 2},
        {"panel_id": "panel_b", "panel_label": "b", "title": "Training", "labels": ["Loss"], "node_count": 1},
    ],
}


class MethodExtractionFigureTargetTests(unittest.TestCase):
    def _prompt_and_report(self, **kwargs):
        captured = {}

        def fake_call(prompt, images, **extra):
            captured["prompt"] = prompt
            return {"figure_goal": "show it", "modules": [{"id": "m", "label": "M", "evidence_ids": ["E1"]}]}

        original = method_extractor.call_vlm_json
        method_extractor.call_vlm_json = fake_call
        try:
            _content, report = method_extractor.extract_method_figure_content(
                {"figure_goal": "show it", "modules": [{"id": "m", "label": "M", "evidence_ids": ["E1"]}]},
                mode="vlm",
                **kwargs,
            )
        finally:
            method_extractor.call_vlm_json = original
        return captured["prompt"], report

    def test_a_benchmark_run_is_told_the_papers_own_panels(self):
        prompt, report = self._prompt_and_report(figure_target="paper", panel_target=PANEL_TARGET)
        self.assertIn("PANEL_TARGET=", prompt)
        self.assertIn('"panel":"a"', prompt)
        self.assertIn("Loss", prompt)
        self.assertIn("panel_target", report["context"]["context_fields"])
        self.assertTrue(report["panel_target_supplied"])
        self.assertEqual(report["figure_target"], "paper")

    def test_the_overview_prompt_is_byte_for_byte_the_same_without_a_target(self):
        overview_prompt, report = self._prompt_and_report()
        paper_prompt, _report = self._prompt_and_report(
            figure_target="paper",
            panel_target={"available": False, "reason": "no_panel_graphs", "panels": []},
        )
        self.assertEqual(overview_prompt, paper_prompt)
        self.assertNotIn("PANEL_TARGET=", overview_prompt)
        self.assertFalse(report["panel_target_supplied"])
        self.assertEqual(report["figure_target"], "overview")


class PanelContractDrillDownTests(unittest.TestCase):
    """The reader must keep every panel node the paper itself named."""

    def test_compact_method_keeps_panel_graphs_and_drops_unnamed_nodes(self):
        compacted = compact_method({
            "figure_goal": "g",
            "modules": [{"id": "m", "name": "M", "evidence_ids": ["E1"]}],
            "panel_graphs": [{
                "id": "panel_a",
                "panel_label": "i",
                "label": "Construction",
                "nodes": [
                    {"instance_id": "a1", "name": "Extraction Agent", "role": "module", "evidence_ids": ["E2"]},
                    {"instance_id": "a2", "name": "   ", "role": "module", "evidence_ids": ["E3"]},
                ],
            }],
        })
        panels = compacted["panel_graphs"]

        self.assertEqual(len(panels), 1)
        self.assertEqual(panels[0]["panel_label"], "i")
        self.assertEqual([node["name"] for node in panels[0]["nodes"]], ["Extraction Agent"])

    def test_a_specification_with_panels_is_told_to_keep_every_panel_node(self):
        captured: dict[str, str] = {}

        def fake_call(prompt, images, **kwargs):
            captured["prompt"] = prompt
            return {"figure_goal": "show it", "modules": [{"id": "m", "label": "M", "evidence_ids": ["E1"]}]}

        original = method_extractor.call_vlm_json
        method_extractor.call_vlm_json = fake_call
        try:
            method_extractor.extract_method_figure_content(
                {
                    "figure_goal": "show it",
                    "modules": [{"id": "m", "label": "M", "evidence_ids": ["E1"]}],
                    "panel_graphs": [{
                        "id": "panel_a",
                        "panel_label": "i",
                        "nodes": [{"instance_id": "a1", "name": "Extraction Agent", "role": "module", "evidence_ids": ["E2"]}],
                    }],
                },
                mode="vlm",
            )
        finally:
            method_extractor.call_vlm_json = original

        prompt = captured["prompt"]
        self.assertIn("REQUIRED_PANEL_NODES=", prompt)
        self.assertIn('"name":"Extraction Agent"', prompt)
        self.assertIn("A panel that lists N names must produce N objects", prompt)

    def test_a_specification_without_panels_keeps_the_plain_prompt(self):
        captured: dict[str, str] = {}

        def fake_call(prompt, images, **kwargs):
            captured["prompt"] = prompt
            return {"figure_goal": "show it", "modules": [{"id": "m", "label": "M", "evidence_ids": ["E1"]}]}

        original = method_extractor.call_vlm_json
        method_extractor.call_vlm_json = fake_call
        try:
            method_extractor.extract_method_figure_content(
                {"figure_goal": "show it", "modules": [{"id": "m", "label": "M", "evidence_ids": ["E1"]}]},
                mode="vlm",
            )
        finally:
            method_extractor.call_vlm_json = original

        self.assertNotIn("REQUIRED_PANEL_NODES=", captured["prompt"])


class PanelNodeMergeTests(unittest.TestCase):
    """A benchmark run must end up with the nodes the paper's panel graphs name."""

    def _extract(self, **kwargs):
        def fake_call(prompt, images, **extra):
            return {"figure_goal": "show it", "modules": [{"id": "m", "label": "Global Memory", "evidence_ids": ["E1"]}]}

        original = method_extractor.call_vlm_json
        method_extractor.call_vlm_json = fake_call
        try:
            return method_extractor.extract_method_figure_content(
                {
                    "figure_goal": "show it",
                    "modules": [{"id": "m", "label": "Global Memory", "evidence_ids": ["E1"]}],
                    "panel_graphs": [{
                        "id": "panel_i",
                        "panel_label": "i",
                        "label": "Construction",
                        "nodes": [
                            {"instance_id": "a1", "name": "Global Memory", "role": "module", "evidence_ids": ["E2"]},
                            {"instance_id": "a2", "name": "Extraction Agent", "role": "module", "evidence_ids": ["E3"]},
                            {"instance_id": "a3", "name": "Unstructured", "role": "input", "evidence_ids": ["E4"]},
                            {"instance_id": "a4", "name": "Answer", "role": "output", "evidence_ids": ["E5"]},
                        ],
                        "relations": [
                            {"id": "p1", "source": "a1", "target": "a2", "type": "data_flow", "evidence_ids": ["E9"]},
                            {"id": "p2", "source": "a3", "target": "a1", "type": "data_flow"},
                            {"id": "p3", "source": "a2", "target": "a2", "type": "self_loop"},
                        ],
                    }],
                },
                mode="vlm",
                **kwargs,
            )
        finally:
            method_extractor.call_vlm_json = original

    def test_a_paper_target_declares_the_panel_nodes_the_reader_left_out(self):
        content, report = self._extract(figure_target="paper")

        self.assertEqual([item["label"] for item in content["modules"]], ["Global Memory", "Extraction Agent"])
        self.assertEqual([item["label"] for item in content["inputs"]], ["Unstructured"])
        self.assertEqual([item["label"] for item in content["outputs"]], ["Answer"])
        self.assertEqual([item["label"] for item in content["stages"]], ["Construction"])
        self.assertEqual(content["modules"][1]["parent_id"], content["stages"][0]["id"])
        self.assertEqual(content["modules"][1]["evidence_ids"], ["E3"])
        self.assertEqual(report["panel_nodes_added"], ["Extraction Agent", "Unstructured", "Answer"])
        self.assertTrue(report["panel_node_merge_applied"])

    def test_a_paper_target_wires_the_panel_edges_the_reader_left_out(self):
        """The paper's own panel edges become relations between the declared objects.

        The reader named four nodes and no relation; the panel graph drew two edges
        between them. Without those relations the design stage lays out the panel
        while being told almost nothing about how it flows, which is how a dense
        figure ends up carrying four arrows for sixty-nine printed ones.
        """

        content, report = self._extract(figure_target="paper")

        pairs = {(item["source"], item["target"]) for item in content["relations"]}
        self.assertIn(("m", "panelnode_extractionagent"), pairs)
        self.assertIn(("panelnode_unstructured", "m"), pairs)
        self.assertEqual(len(content["relations"]), 2)
        self.assertEqual(report["panel_relations_added"], 2)
        self.assertEqual(
            [item["type"] for item in content["relations"]],
            ["data_flow", "data_flow"],
        )

    def test_the_overview_plan_is_left_exactly_as_the_reader_wrote_it(self):
        content, report = self._extract()

        self.assertEqual([item["label"] for item in content["modules"]], ["Global Memory"])
        self.assertEqual(content["stages"], [])
        self.assertEqual(content["relations"], [])
        self.assertEqual(report["panel_nodes_added"], [])
        self.assertEqual(report["panel_relations_added"], 0)
        self.assertFalse(report["panel_node_merge_applied"])


if __name__ == "__main__":
    unittest.main()
