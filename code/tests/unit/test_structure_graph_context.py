"""The structure planner reads the paper's scientific graph only when asked to."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rfs.planning import vlm_structure_planner as planner
from rfs.planning.panel_target import compact_panel_target


METHOD = {
    "figure_goal": "Show the pipeline",
    "inputs": [{"id": "input", "label": "Paper PDF", "must_show": True, "evidence_ids": ["E1"]}],
    "modules": [
        {"id": "encode", "label": "Encoder", "must_show": True, "evidence_ids": ["E2"]},
        {"id": "decode", "label": "Decoder", "should_render": True, "evidence_ids": ["E3"]},
    ],
    "outputs": [{"id": "result", "label": "Figure", "evidence_ids": ["E4"]}],
    "relations": [
        {"id": "r1", "source": "input", "target": "encode", "type": "data_flow", "must_show": True, "evidence_ids": ["E5"]},
        {"id": "r2", "source": "encode", "target": "decode", "type": "data_flow", "must_show": True, "evidence_ids": ["E6"]},
        {"id": "r3", "source": "decode", "target": "result", "type": "data_flow", "evidence_ids": ["E7"]},
    ],
}

#: The instruction the switch appends. The regression test asserts the off prompt
#: is the on prompt with this sentence and the SCIENTIFIC_GRAPH block removed.
GRAPH_SENTENCE = (
    " SCIENTIFIC_GRAPH states this paper's own topology in the same contract-id space as METHOD: follow it when you place and connect nodes. "
    "Draw a feedback motif as a visible loop, a training_inference motif as separated training and inference stages, and a branch or merge motif "
    "with its fan-out or fan-in kept explicit. Every node or relation the graph marks required must appear. Never invent topology the graph does not state."
)


def _valid_plan() -> dict:
    return {
        "canvas": {"aspect_ratio": "16:9", "margin": 0.04},
        "panels": [
            {"id": "p1", "title": "Input", "bbox_percent": {"x": 0.05, "y": 0.15, "w": 0.4, "h": 0.4}},
            {"id": "p2", "title": "Model", "bbox_percent": {"x": 0.55, "y": 0.15, "w": 0.4, "h": 0.4}},
        ],
        "nodes": [
            {"id": "n1", "semantic_source_id": "input", "panel_id": "p1", "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.15}},
            {"id": "n2", "semantic_source_id": "encode", "panel_id": "p2", "bbox_percent": {"x": 0.6, "y": 0.2, "w": 0.2, "h": 0.15}},
            {"id": "n3", "semantic_source_id": "decode", "panel_id": "p2", "bbox_percent": {"x": 0.6, "y": 0.4, "w": 0.2, "h": 0.15}},
        ],
        "connectors": [
            {"id": "c1", "source": "n1", "target": "n2", "type": "data_flow"},
            {"id": "c2", "source": "n2", "target": "n3", "type": "data_flow"},
        ],
    }


def _broken_plan() -> dict:
    """One panel (a contract error) and the required encode->decode relation dropped."""

    plan = _valid_plan()
    plan["panels"] = [plan["panels"][0]]
    plan["connectors"] = [plan["connectors"][0]]
    return plan


class _Recorder:
    """A provider stub that keeps every prompt and replays the given replies."""

    def __init__(self, replies: list[dict]):
        self.replies = replies
        self.prompts: list[str] = []

    def __call__(self, prompt, images, **kwargs):
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]


def _run(
    reply: dict,
    *,
    graph_context: bool,
    bulk_figures: bool = True,
    evidence_structure: bool = True,
    hierarchy: dict | None = None,
    figure_target: str = "overview",
    panel_target: dict | None = None,
):
    recorder = _Recorder([reply])
    with patch.object(planner, "call_vlm_json", recorder):
        design, report = planner.plan_structure_vlm(
            METHOD,
            {},
            None,
            model="fake-model",
            bulk_figures=bulk_figures,
            graph_context=graph_context,
            evidence_structure=evidence_structure,
            scientific_hierarchy=hierarchy,
            figure_target=figure_target,
            panel_target=panel_target,
        )
    return recorder.prompts, design, report


HIERARCHY = {
    "contract": "rfs.scientific_hierarchy",
    "version": 1,
    "sections": [{"id": "s1", "title": "Input", "entity_ids": ["input"]}],
    "submodules": [],
}

#: The inventory sentence the hierarchy block travels with. A row that never
#: planned a hierarchy drops the sentence as well, so a prompt is never left
#: pointing the model at a block that is not there.
HIERARCHY_SENTENCE = (
    "Treat SCIENTIFIC_HIERARCHY as the authoritative candidate inventory: use evidenced sections as macro regions and evidenced submodules as nested nodes when their scientific_importance or visualization_value warrants display. "
)


class StructureGraphPromptTests(unittest.TestCase):
    def test_the_switch_on_sends_the_graph_and_its_motifs(self):
        prompts, _design, report = _run(_valid_plan(), graph_context=True)
        self.assertEqual(len(prompts), 1)
        self.assertIn(" SCIENTIFIC_GRAPH=", prompts[0])
        self.assertIn("linear", prompts[0])
        self.assertIn(GRAPH_SENTENCE, prompts[0])
        self.assertIn("scientific_graph", report["context"]["context_fields"])
        self.assertTrue(report["graph_constraint"]["enabled"])
        self.assertTrue(report["graph_constraint"]["ok"])

    def test_the_switch_off_is_exactly_todays_prompt(self):
        on_prompts, _, _ = _run(_valid_plan(), graph_context=True)
        off_prompts, _design, report = _run(_valid_plan(), graph_context=False)
        on_prompt, off_prompt = on_prompts[0], off_prompts[0]

        block_start = on_prompt.index(" SCIENTIFIC_GRAPH=")
        block_end = on_prompt.index(" METHOD=", block_start)
        without_block = on_prompt[:block_start] + on_prompt[block_end:]
        self.assertEqual(off_prompt, without_block.replace(GRAPH_SENTENCE, ""))
        self.assertNotIn("SCIENTIFIC_GRAPH", off_prompt)
        self.assertEqual(report["context"]["context_fields"], ["compact_method", "compact_reference"])
        self.assertEqual(report["graph_constraint"], {"enabled": False})


class StructureGraphRepairTests(unittest.TestCase):
    def test_a_required_relation_the_design_dropped_is_wired_from_the_contract(self):
        """The paper's own wiring is completed deterministically, not asked for again.

        A design that draws four arrows out of sixty-nine used to be accepted even
        with the graph switch on: the grounding pass only rebinds the connectors a
        plan already carries, so a dropped relation was invisible everywhere. It is
        now restored before the validation and the repair prompt, and the repair
        prompt therefore has nothing to ask for.
        """

        recorder = _Recorder([_broken_plan()])
        with patch.object(planner, "call_vlm_json", recorder):
            design, report = planner.plan_structure_vlm(METHOD, {}, None, model="fake-model", graph_context=True)

        self.assertGreaterEqual(len(recorder.prompts), 2)
        self.assertNotIn("GRAPH_REQUIREMENTS=", recorder.prompts[1])
        pairs = {(str(item.get("source")), str(item.get("target"))) for item in design["connectors"]}
        self.assertIn(("n2", "n3"), pairs)
        completion = report["connector_completion"]
        self.assertEqual(completion["added_count"], 1)
        self.assertEqual(completion["added"][0]["paper_source"], "encode")
        self.assertEqual(completion["added"][0]["paper_target"], "decode")
        self.assertEqual(completion["added"][0]["source"], "n2")
        self.assertEqual(completion["added"][0]["target"], "n3")

    def test_a_relation_the_design_cannot_carry_still_reaches_the_repair_prompt(self):
        """Completion wires placed objects; a missing endpoint is still the model's job."""

        plan = _broken_plan()
        plan["nodes"] = [node for node in plan["nodes"] if str(node.get("id")) != "n3"]
        recorder = _Recorder([plan])
        with patch.object(planner, "call_vlm_json", recorder):
            planner.plan_structure_vlm(METHOD, {}, None, model="fake-model", graph_context=True)

        self.assertGreaterEqual(len(recorder.prompts), 2)
        repair_prompt = recorder.prompts[1]
        self.assertIn("GRAPH_REQUIREMENTS=", repair_prompt)
        self.assertIn('"encode","decode"', repair_prompt)
        self.assertIn("GRAPH_REQUIREMENTS names", repair_prompt)

    def test_the_repair_prompt_keeps_its_add_nothing_rule_when_off(self):
        recorder = _Recorder([_broken_plan()])
        with patch.object(planner, "call_vlm_json", recorder):
            planner.plan_structure_vlm(METHOD, {}, None, model="fake-model", graph_context=False)
        self.assertGreaterEqual(len(recorder.prompts), 2)
        self.assertFalse([prompt for prompt in recorder.prompts if "GRAPH_REQUIREMENTS" in prompt])
        self.assertIn("add or delete no scientific content", recorder.prompts[1])


class StructureEvidenceStructureTests(unittest.TestCase):
    """The middle-layer row drops the hierarchy and the graph together."""

    def test_off_sends_neither_the_hierarchy_nor_the_graph(self):
        prompts, _design, report = _run(
            _valid_plan(),
            graph_context=True,
            evidence_structure=False,
            hierarchy=HIERARCHY,
        )
        self.assertEqual(len(prompts), 1)
        self.assertNotIn("SCIENTIFIC_HIERARCHY", prompts[0])
        self.assertNotIn("SCIENTIFIC_GRAPH", prompts[0])
        self.assertIn(" METHOD=", prompts[0])
        self.assertEqual(report["context"]["context_fields"], ["compact_method", "compact_reference"])
        self.assertEqual(report["graph_constraint"], {"enabled": False})

    def test_off_is_todays_prompt_with_both_evidence_blocks_removed(self):
        on_prompts, _, _ = _run(_valid_plan(), graph_context=True, hierarchy=HIERARCHY)
        off_prompts, _design, _report = _run(
            _valid_plan(), graph_context=True, evidence_structure=False, hierarchy=HIERARCHY
        )
        on_prompt, off_prompt = on_prompts[0], off_prompts[0]

        without_graph = on_prompt[: on_prompt.index(" SCIENTIFIC_GRAPH=")] + on_prompt[
            on_prompt.index(" METHOD=", on_prompt.index(" SCIENTIFIC_GRAPH=")) :
        ]
        without_graph = without_graph.replace(GRAPH_SENTENCE, "")
        without_hierarchy = without_graph[: without_graph.index(" SCIENTIFIC_HIERARCHY=")] + without_graph[
            without_graph.index(" METHOD=", without_graph.index(" SCIENTIFIC_HIERARCHY=")) :
        ]
        self.assertEqual(off_prompt, without_hierarchy.replace(HIERARCHY_SENTENCE, ""))

    def test_on_is_unchanged_when_the_row_keeps_its_structure(self):
        prompts, _design, report = _run(_valid_plan(), graph_context=False, hierarchy=HIERARCHY)
        self.assertIn(" SCIENTIFIC_HIERARCHY=", prompts[0])
        self.assertIn("s1", prompts[0])


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
    "empty": False,
}

#: The sentence the target appends. It travels with the block, so a run that sends
#: no target is not left pointing the model at panels it was never given.
TARGET_SENTENCE = (
    " FIGURE_TARGET_PANELS states the panels of the paper's own framework figure in the paper's own order, "
    "with the labels a reader can see in each. Follow it: keep its panel partitioning and its granularity, "
    "one design panel per target panel, and keep every label it lists. When it names a single panel, that "
    "panel is the whole guide and the canvas still carries its own panel grid inside it. Never invent a "
    "panel or a label the target does not state."
)


class StructureFigureTargetTests(unittest.TestCase):
    def test_paper_target_sends_the_papers_own_panels(self):
        prompts, _design, report = _run(
            _valid_plan(), graph_context=False, figure_target="paper", panel_target=PANEL_TARGET
        )
        self.assertIn(" FIGURE_TARGET_PANELS=", prompts[0])
        self.assertIn(TARGET_SENTENCE, prompts[0])
        self.assertIn('"panel":"a"', prompts[0])
        self.assertIn("Loss", prompts[0])
        self.assertIn("figure_target_panels", report["context"]["context_fields"])
        self.assertEqual(report["figure_target"], "paper")

    def test_the_overview_prompt_is_byte_for_byte_the_same_without_a_target(self):
        overview_prompts, _design, report = _run(_valid_plan(), graph_context=False, figure_target="overview")
        unavailable_prompts, _design, _report = _run(
            _valid_plan(),
            graph_context=False,
            figure_target="paper",
            panel_target={"available": False, "reason": "no_panel_graphs", "panels": []},
        )
        self.assertEqual(overview_prompts[0], unavailable_prompts[0])
        self.assertNotIn("FIGURE_TARGET_PANELS", overview_prompts[0])
        self.assertNotIn("figure_target_panels", report["context"]["context_fields"])
        self.assertEqual(report["figure_target"], "overview")

    def test_the_target_does_not_disturb_the_graph_block(self):
        prompts, _design, _report = _run(
            _valid_plan(), graph_context=True, figure_target="paper", panel_target=PANEL_TARGET
        )
        self.assertIn(" SCIENTIFIC_GRAPH=", prompts[0])
        self.assertIn(" FIGURE_TARGET_PANELS=", prompts[0])
        self.assertLess(prompts[0].index(" SCIENTIFIC_GRAPH="), prompts[0].index(" FIGURE_TARGET_PANELS="))
        self.assertLess(prompts[0].index(" FIGURE_TARGET_PANELS="), prompts[0].index(" METHOD="))


class StructureFigureTargetReplanTests(unittest.TestCase):
    def test_the_density_replan_prompt_carries_the_target(self):
        recorder = _Recorder([_valid_plan()])
        density = {"status": "replan_required", "sparse_panels": [], "node_occupancy_estimate": 0.1}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(planner, "call_vlm_json", recorder):
                planner._attempt_density_replan(
                    _valid_plan(),
                    density,
                    METHOD,
                    {},
                    None,
                    None,
                    Path(tmp) / "wire.png",
                    "fake-model",
                    30,
                    0,
                    "16:9",
                    0.05,
                    0.92,
                    None,
                    None,
                    "",
                    compact_panel_target(PANEL_TARGET),
                    True,
                )
        self.assertTrue(recorder.prompts)
        self.assertIn(" FIGURE_TARGET_PANELS=", recorder.prompts[0])
        self.assertIn('"panel":"b"', recorder.prompts[0])


if __name__ == "__main__":
    unittest.main()
