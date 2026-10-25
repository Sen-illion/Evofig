import re
import tempfile
import unittest
from pathlib import Path

from rfs.planning.paper_contract import validate_plan_grounding
from rfs.analysis.paper.preparation import READ_DEADLINE_SECONDS, _demote_unconnected_completion_entities, _detect_dense_overview_panels, _extract_overview_panels, _fast_cache_path, _merge_isolated_architecture_entities, _normalize_panel_graphs, _repair_cross_script_entity_labels, _repair_relation_evidence_after_grounding, build_overlay_spec, expand_plan_evidence, normalize_figure_contract, prepare_paper_figure_contract, synchronize_plan_to_contract
from rfs.contracts.paper_completion import _overview_candidates


class PaperContractTests(unittest.TestCase):
    def test_panel_graph_normalization_keeps_local_instances_and_repairs_relation_evidence(self):
        parsed = {"evidence": [
            {"id": "E_DATA", "page": 2, "text": "Dataset"},
            {"id": "E_MODEL", "page": 2, "text": "Model"},
        ]}
        spec = {"panel_graphs": [{
            "id": "panel_a",
            "panel_label": "a",
            "label": "Workflow",
            "nodes": [
                {"instance_id": "data", "name": "Dataset", "role": "input", "evidence_ids": ["E_DATA"]},
                {"instance_id": "model", "name": "Model", "role": "module", "evidence_ids": ["E_MODEL"]},
            ],
            "relations": [{"source": "data", "target": "model", "type": "data_flow", "evidence_ids": []}],
        }]}

        report = _normalize_panel_graphs(spec, parsed)

        self.assertEqual(report["node_count"], 2)
        self.assertEqual(report["relation_count"], 1)
        self.assertEqual(spec["panel_graphs"][0]["relations"][0]["evidence_ids"], ["E_DATA", "E_MODEL"])

    def test_two_panel_overview_is_preserved_as_dense_multiframe(self):
        parsed = {
            "document_index": {"figures": [{
                "page": 2,
                "block_id": "P002_B010",
                "caption": "Fig. 1 | Overview of the method. a, High-level pre-training and usage. b, The model architecture.",
            }]},
            "pages": [{
                "page": 2,
                "height": 790,
                "blocks": [
                    {"id": "P002_B010", "reading_order": 10, "bbox": [40, 335, 290, 344], "font_size": 7, "text": "Fig. 1 | Overview of the method. a, High-level pre-training and usage."},
                    {"id": "P002_B011", "reading_order": 11, "bbox": [306, 335, 560, 344], "font_size": 7, "text": "b, The model architecture."},
                ],
            }],
            "evidence": [
                {"id": "E_A", "block_id": "P002_B010", "page": 2, "text": "Fig. 1 | Overview of the method. a, High-level pre-training and usage."},
                {"id": "E_B", "block_id": "P002_B011", "page": 2, "text": "b, The model architecture."},
            ],
        }
        spec = {
            "inputs": [{"id": "input", "name": "Input dataset", "evidence_ids": []}],
            "modules": [{"id": "encoder", "name": "attention over features", "evidence_ids": []}],
            "outputs": [{"id": "prediction", "name": "Prediction", "evidence_ids": []}],
            "innovations": [],
        }

        dense = _detect_dense_overview_panels(parsed)
        panels = _extract_overview_panels(parsed, spec)

        self.assertTrue(dense["detected"])
        self.assertEqual(dense["panel_labels"], ["a", "b"])
        self.assertEqual([item["panel_label"] for item in panels], ["a", "b"])
        self.assertEqual(panels[0]["entity_ids"], ["input", "prediction"])
        self.assertEqual(panels[1]["entity_ids"], ["encoder"])

    def test_extracts_cross_column_figure_one_panel_descriptions(self):
        parsed = {
            "document_index": {"figures": [{
                "page": 3,
                "block_id": "P003_B010",
                "caption": "Fig. 1 | Overview of Example. a, Flow chart showing the model architecture of Example.",
            }]},
            "pages": [{
                "page": 3,
                "height": 790,
                "blocks": [
                    {"id": "P003_B010", "reading_order": 10, "bbox": [40, 365, 290, 374], "font_size": 7, "text": "Fig. 1 | Overview of Example. a, Flow chart showing the model architecture of Example."},
                    {"id": "P003_B011", "reading_order": 11, "bbox": [40, 376, 290, 385], "font_size": 7, "text": "The model maps the input through two encoders."},
                    {"id": "P003_B067", "reading_order": 67, "bbox": [306, 366, 560, 375], "font_size": 7, "text": "b, Image tile-level pretraining using DINOv2."},
                    {"id": "P003_B068", "reading_order": 68, "bbox": [306, 376, 560, 385], "font_size": 7, "text": "c, Slide-level pretraining with LongNet using masked autoencoder."},
                    {"id": "P003_B069", "reading_order": 69, "bbox": [306, 435, 560, 445], "font_size": 8.25, "text": "Body text must not enter the caption."},
                ],
            }],
            "evidence": [
                {"id": "E_A", "block_id": "P003_B010", "text": "Fig. 1 | Overview of Example. a, Flow chart showing the model architecture of Example."},
                {"id": "E_A2", "block_id": "P003_B011", "text": "The model maps the input through two encoders."},
                {"id": "E_B", "block_id": "P003_B067", "text": "b, Image tile-level pretraining using DINOv2."},
                {"id": "E_C", "block_id": "P003_B068", "text": "c, Slide-level pretraining with LongNet using masked autoencoder."},
            ],
        }
        spec = {
            "inputs": [{"id": "input", "name": "Input", "evidence_ids": ["E_A2"]}],
            "modules": [], "outputs": [], "innovations": [],
        }

        panels = _extract_overview_panels(parsed, spec)

        self.assertEqual([item["panel_label"] for item in panels], ["a", "b", "c"])
        self.assertEqual(panels[0]["label"], "Model architecture of Example")
        self.assertEqual(panels[1]["label"], "Image tile-level pretraining using DINOv2")
        self.assertEqual(panels[2]["label"], "Slide-level pretraining with LongNet using masked autoencoder")
        self.assertEqual(panels[0]["entity_ids"], ["input"])

    def test_supporting_details_are_exact_deduplicated_and_weak_verbs_are_dropped(self):
        spec = {
            "inputs": [],
            "modules": [
                {"id": "class_token", "name": "Class Token", "evidence_ids": ["E_CLASS"]},
                {"id": "generate", "name": "Generate", "evidence_ids": ["E_GENERATE"]},
            ],
            "outputs": [], "innovations": [], "relations": [],
            "supporting_details": [
                {"id": "class_token", "name": "Class Token", "evidence_ids": ["E_CLASS"]},
                {"id": "class_token", "name": "Class Token", "evidence_ids": ["E_CLASS"]},
            ],
        }
        completion = {"added_entities": ["class_token", "generate"]}
        parsed = {"evidence": [
            {"id": "E_CLASS", "page": 3, "text": "classification token."},
            {"id": "E_SYMBOL", "page": 3, "text": "[CLS] is the classification token."},
            {"id": "E_GENERATE", "page": 3, "text": "to generate contextualized embeddings, which can serve as the basis for downstream applications"},
        ]}

        _demote_unconnected_completion_entities(spec, completion, parsed)
        _demote_unconnected_completion_entities(spec, completion, parsed)

        self.assertEqual(len(spec["supporting_details"]), 1)
        self.assertEqual(spec["supporting_details"][0]["name"], "classification token")
        self.assertEqual(spec["supporting_details"][0]["visual_cue"], "one distinct highlighted token block among repeated token blocks")
        self.assertEqual(spec["supporting_details"][0]["optional_visible_label"], "[CLS]")

    def test_dense_overview_panels_and_local_details_are_not_flattened_into_peer_nodes(self):
        parsed = {
            "document_index": {
                "figures": [{"page": 3, "caption": "Fig. 1 | Overview of Example. a, Model architecture."}],
            },
            "pages": [{
                "page": 3,
                "blocks": [
                    {"text": "a"},
                    {"text": "b"},
                    {"text": "c"},
                    {"text": "b, Tile-level pretraining."},
                    {"text": "c, Slide-level pretraining."},
                ],
            }],
            "evidence": [
                {"id": "E_MAIN", "page": 3, "text": "Slide Encoder based on the LongNet architecture"},
                {"id": "E_EXACT", "page": 3, "text": "Slide Encoder (LongNet)"},
                {"id": "E_DETAIL", "page": 3, "text": "Classification token"},
            ],
        }
        spec = {
            "inputs": [{"id": "input", "name": "Input", "evidence_ids": ["E_MAIN"]}],
            "modules": [
                {"id": "encoder", "name": "Slide Encoder", "evidence_ids": ["E_MAIN"]},
                {"id": "longnet", "name": "LongNet", "evidence_ids": ["E_MAIN"]},
                {"id": "class_token", "name": "Class Token", "evidence_ids": ["E_DETAIL"]},
            ],
            "outputs": [{"id": "output", "name": "Output", "evidence_ids": ["E_MAIN"]}],
            "innovations": [],
            "relations": [
                {"source": "input", "target": "encoder", "type": "data_flow", "evidence_ids": ["E_MAIN"]},
                {"source": "encoder", "target": "output", "type": "data_flow", "evidence_ids": ["E_MAIN"]},
            ],
        }
        completion = {"added_entities": ["class_token"]}

        dense = _detect_dense_overview_panels(parsed)
        merged = _merge_isolated_architecture_entities(spec, parsed)
        demoted = _demote_unconnected_completion_entities(spec, completion)

        self.assertTrue(dense["detected"])
        self.assertEqual(dense["panel_labels"], ["a", "b", "c"])
        self.assertEqual(merged, ["longnet->encoder"])
        self.assertEqual(spec["modules"][0]["name"], "Slide Encoder (LongNet)")
        self.assertEqual(demoted, ["class_token"])
        self.assertEqual(spec["supporting_details"][0]["visibility"], "unlabeled_visual_detail_not_global_node")

    def test_late_exact_grounding_repairs_boundary_relation_evidence(self):
        parsed = {
            "document_index": {"figures": [{"page": 2, "caption": "Figure 1: Model overview."}]},
            "pages": [],
            "evidence": [
                {"id": "E_INPUT", "page": 2, "text": "Input dataset"},
                {"id": "E_MODEL", "page": 2, "text": "Input dataset enters the model architecture"},
                {"id": "E_OUTPUT", "page": 2, "text": "model architecture produces Predictions"},
            ],
        }
        spec = {
            "inputs": [{"id": "input", "name": "Input dataset", "evidence_ids": ["E_INPUT", "E_MODEL"]}],
            "modules": [{"id": "model", "name": "model architecture", "evidence_ids": ["E_MODEL", "E_OUTPUT"]}],
            "outputs": [{"id": "output", "name": "Predictions", "evidence_ids": ["E_OUTPUT"]}],
            "innovations": [],
            "relations": [
                {"source": "input", "target": "model", "type": "data_flow", "evidence_ids": []},
                {"source": "model", "target": "output", "type": "prediction", "evidence_ids": []},
            ],
        }

        repaired = _repair_relation_evidence_after_grounding(spec, parsed)

        self.assertTrue(all(item["evidence_ids"] for item in spec["relations"]))
        self.assertEqual(repaired, ["input->model", "model->output"])

    def test_peer_generic_inputs_converge_on_existing_shared_target(self):
        evidence = [
            {"id": "E_XTRAIN", "page": 2, "text": "X train"},
            {"id": "E_YTRAIN", "page": 2, "text": "y train"},
            {"id": "E_XTEST", "page": 2, "text": "X test"},
            {"id": "E_LINEAR", "page": 2, "text": "linear encoders"},
            {"id": "E_OUTPUT", "page": 2, "text": "y test"},
        ]
        parsed = {
            "document_index": {"figures": [{"page": 2, "caption": "Figure 1: Model overview."}]},
            "pages": [],
            "evidence": evidence,
        }
        plan = {
            "paper_summary": {},
            "figure_specification": {
                "inputs": [
                    {"id": "xtrain", "name": "X train", "evidence_ids": ["E_XTRAIN"]},
                    {"id": "ytrain", "name": "y train", "evidence_ids": ["E_YTRAIN"]},
                    {"id": "xtest", "name": "X test", "evidence_ids": ["E_XTEST"]},
                ],
                "modules": [{"id": "linear", "name": "linear encoders", "evidence_ids": ["E_LINEAR"]}],
                "outputs": [{"id": "output", "name": "y test", "evidence_ids": ["E_OUTPUT"]}],
                "innovations": [],
                "relations": [
                    {"source": "xtrain", "target": "linear", "type": "encoding", "evidence_ids": ["E_XTRAIN", "E_LINEAR"]},
                    {"source": "ytrain", "target": "linear", "type": "encoding", "evidence_ids": ["E_YTRAIN", "E_LINEAR"]},
                    {"source": "linear", "target": "output", "type": "prediction", "evidence_ids": ["E_LINEAR", "E_OUTPUT"]},
                ],
            },
        }

        normalize_figure_contract(plan, parsed)
        relations = plan["figure_specification"]["relations"]

        self.assertTrue(any(item["source"] == "xtest" and item["target"] == "linear" for item in relations))

    def test_cross_script_symbol_hallucination_is_repaired_from_exact_evidence_span(self):
        spec = {
            "inputs": [{"id": "tiles", "name": "256 脳 256 image tiles", "evidence_ids": ["E1"]}],
            "modules": [],
            "outputs": [],
            "innovations": [],
        }
        parsed = {"evidence": [{"id": "E1", "text": "serializes each WSI into a sequence of 256 × 256 image tiles in row-major order"}]}

        repaired = _repair_cross_script_entity_labels(spec, parsed)

        self.assertEqual(spec["inputs"][0]["name"], "256 × 256 image tiles")
        self.assertEqual(repaired, ["tiles:256 脳 256 image tiles->256 × 256 image tiles"])

    def test_cross_script_repair_restores_exact_multiplication_symbol_label(self):
        parsed = {
            "evidence": [{"id": "E_LAYER", "page": 2, "text": "2D TabPFN layer (12×)"}],
        }
        spec = {
            "inputs": [],
            "modules": [{"id": "layer", "name": "2D TabPFN layer (12脳)", "evidence_ids": ["E_LAYER"]}],
            "outputs": [],
            "innovations": [],
        }

        repaired = _repair_cross_script_entity_labels(spec, parsed)

        self.assertEqual(spec["modules"][0]["name"], "2D TabPFN layer (12×)")
        self.assertTrue(repaired)

    def test_overview_selection_rejects_late_scaling_figure_when_figure_one_is_a_true_model_overview(self):
        parsed = {
            "document_index": {
                "figures": [
                    {"page": 3, "caption": "Figure 1: Model overview. Images become patches and enter a Transformer encoder."},
                    {"page": 16, "caption": "Figure 8: Scaling different model dimensions of the Vision Transformer."},
                ]
            }
        }

        candidates = _overview_candidates(parsed)

        self.assertEqual([item["page"] for item in candidates], [3])

    def test_contract_prunes_appendix_only_diagram_label_and_ungrounded_innovation(self):
        overview = "Figure 1: Model overview. We split an input image into image patches, linearly embed them, add position embeddings and a class token, and feed them to a Transformer Encoder for classification."
        parsed = {
            "page_count": 18,
            "document_index": {
                "figures": [
                    {"page": 3, "caption": overview},
                    {"page": 16, "caption": "Figure 8: Scaling different model dimensions of the Vision Transformer."},
                ]
            },
            "evidence": [
                {"id": "E_OVERVIEW", "page": 3, "kind": "caption", "text": overview, "section_hint": "Figure Captions", "confidence": 1.0},
                {"id": "E_DEPTH", "page": 16, "kind": "paragraph", "text": "Depth", "section_hint": "APPENDIX", "confidence": 1.0},
            ],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "figure_goal": "Show the Vision Transformer model overview.",
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "Images are processed as patch sequences.", "evidence_ids": ["E_OVERVIEW"]},
                "inputs": [
                    {"id": "image", "name": "Input Image", "evidence_ids": ["E_OVERVIEW"]},
                    {"id": "depth", "name": "Depth", "evidence_ids": ["E_DEPTH"]},
                ],
                "modules": [{"id": "encoder", "name": "Transformer Encoder", "evidence_ids": ["E_OVERVIEW"]}],
                "outputs": [{"id": "classification", "name": "Classification", "evidence_ids": ["E_OVERVIEW"]}],
                "innovations": [{"id": "innovation_class_token", "name": "Learnable Class Token", "evidence_ids": []}],
                "relations": [
                    {"source": "image", "target": "encoder", "type": "data_flow", "evidence_ids": ["E_OVERVIEW"]},
                    {"source": "depth", "target": "encoder", "type": "data_flow", "evidence_ids": ["E_DEPTH"]},
                    {"source": "encoder", "target": "classification", "type": "data_flow", "evidence_ids": ["E_OVERVIEW"]},
                ],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        self.assertNotIn("Depth", [item["name"] for item in spec["inputs"]])
        self.assertEqual(spec["innovations"], [])
        self.assertFalse(any(item["source"] == "depth" or item["target"] == "depth" for item in spec["relations"]))
        self.assertIn("Depth", plan["contract_completion_report"]["removed_out_of_scope_entities"])
        self.assertIn("Learnable Class Token", plan["contract_completion_report"]["removed_ungrounded_innovations"])

    def test_vit_mlp_is_upgraded_to_one_mlp_head_instead_of_duplicating_the_prediction_path(self):
        overview = "Figure 1: Model overview. The Transformer encoder representation is passed through an MLP head for classification."
        parsed = {
            "page_count": 4,
            "document_index": {"figures": [{"page": 2, "caption": overview}]},
            "evidence": [{"id": "E_OVERVIEW", "page": 2, "kind": "caption", "text": overview, "section_hint": "Figure Captions", "confidence": 1.0}],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "figure_goal": "Show image classification.",
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [],
                "modules": [
                    {"id": "encoder", "name": "Transformer Encoder", "evidence_ids": ["E_OVERVIEW"]},
                    {"id": "mlp", "name": "MLP", "role": "module", "evidence_ids": ["E_OVERVIEW"]},
                ],
                "outputs": [{"id": "classification", "name": "Classification", "evidence_ids": ["E_OVERVIEW"]}],
                "innovations": [],
                "relations": [
                    {"source": "encoder", "target": "mlp", "type": "data_flow", "evidence_ids": ["E_OVERVIEW"]},
                    {"source": "mlp", "target": "classification", "type": "data_flow", "evidence_ids": ["E_OVERVIEW"]},
                ],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        labels = [item["name"] for item in spec["modules"]]

        self.assertEqual(labels.count("MLP Head"), 1)
        self.assertNotIn("MLP", labels)
        self.assertEqual(len([item for item in spec["relations"] if item["target"] == "classification"]), 1)

    def test_explicit_input_intermediate_path_removes_redundant_input_shortcut(self):
        overview = "Figure 1: Input Image is split into Image Patches, followed by Linear Projection and a Transformer Encoder."
        parsed = {
            "page_count": 3,
            "document_index": {"figures": [{"page": 2, "caption": overview}]},
            "evidence": [{"id": "E_OVERVIEW", "page": 2, "kind": "caption", "text": overview, "section_hint": "Figure Captions", "confidence": 1.0}],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "figure_goal": "Show the patch pipeline.",
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [{"id": "image", "name": "Input Image", "evidence_ids": ["E_OVERVIEW"]}],
                "modules": [
                    {"id": "patches", "name": "Image Patches", "role": "intermediate", "evidence_ids": ["E_OVERVIEW"]},
                    {"id": "projection", "name": "Linear Projection", "evidence_ids": ["E_OVERVIEW"]},
                ],
                "outputs": [],
                "innovations": [],
                "relations": [
                    {"source": "image", "target": "projection", "type": "data_flow", "evidence_ids": ["E_OVERVIEW"]},
                    {"source": "image", "target": "patches", "type": "data_flow", "evidence_ids": ["E_OVERVIEW"]},
                    {"source": "patches", "target": "projection", "type": "data_flow", "evidence_ids": ["E_OVERVIEW"]},
                ],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        self.assertNotIn(("image", "projection"), pairs)
        self.assertIn(("image", "patches"), pairs)
        self.assertIn(("patches", "projection"), pairs)
        self.assertIn("image->projection", plan["contract_completion_report"]["removed_input_shortcuts"])

    def _parsed(self):
        return {
            "evidence": [
                {"id": "E0001", "confidence": 1.0},
                {"id": "E0002", "confidence": 0.7, "source": "easyocr"},
            ]
        }

    def test_contract_adds_topology_labels_and_separate_flows(self):
        plan = {
            "paper_summary": {
                "research_problem": {"text": "Understand an input.", "evidence_ids": ["E0001"]},
                "central_claim": {"text": "The method refines its output.", "evidence_ids": ["E0001"]},
                "inputs": [{"id": "input", "name": "Input", "evidence_ids": ["E0001"]}],
                "outputs": [{"id": "output", "name": "Refined Output", "evidence_ids": ["E0001"]}],
                "training_flow": [{"step": "Train", "evidence_ids": ["E0001"]}],
                "inference_flow": [{"step": "Refine", "evidence_ids": ["E0001"]}],
                "unknowns": [],
            },
            "figure_specification": {
                "modules": [{"id": "refiner", "name": "Refiner", "evidence_ids": ["E0001"]}],
                "relations": [{"source": "output", "target": "refiner", "type": "feedback", "evidence_ids": ["E0001"]}],
                "terminology": {"Refiner": "Refiner"},
            },
            "design_plan": {"reading_order": ["input", "refiner", "output"], "groups": []},
        }

        spec = normalize_figure_contract(plan, self._parsed())
        overlay = build_overlay_spec(plan)

        self.assertEqual(spec["topology"], "feedback")
        self.assertEqual(spec["training_flow"][0]["step"], "Train")
        self.assertEqual(spec["inference_flow"][0]["step"], "Refine")
        self.assertIn("Refiner", spec["required_labels"])
        self.assertEqual(overlay["connectors"][0]["source"], "output")
        self.assertTrue(any(item["text"] == "Refined Output" for item in overlay["labels"]))

    def test_contract_normalizes_scalar_claim_and_grounds_it_from_evidence(self):
        parsed = {
            "evidence": [
                {
                    "id": "E0001",
                    "page": 3,
                    "text": "The same architecture is used for pre-training and fine-tuning with task-specific output layers.",
                    "confidence": 1.0,
                }
            ]
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": "Unified architecture for pre-training and fine-tuning.",
                "central_claim": "The same architecture is used for pre-training and fine-tuning with task-specific output layers.",
                "modules": [{"id": "encoder", "name": "Encoder", "evidence_ids": ["E0001"]}],
                "inputs": [],
                "outputs": [],
                "innovations": [],
                "relations": [],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        validation = validate_plan_grounding(plan, parsed)

        self.assertEqual(spec["central_claim"]["evidence_ids"], ["E0001"])
        self.assertEqual(spec["research_problem"]["evidence_ids"], ["E0001"])
        self.assertTrue(validation["ok"])

    def test_cross_script_translation_of_visible_labels_is_a_recorded_gap(self):
        parsed = {
            "evidence": [
                {"id": "E0001", "text": "输入论文首先进入文档编码器,关系解码器连接实体,最后输出可编辑框架图。", "confidence": 1.0},
            ]
        }
        plan = {
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [{"id": "input", "name": "Input document", "evidence_ids": ["E0001"]}],
                "modules": [{"id": "encoder", "name": "Document encoder", "evidence_ids": ["E0001"]}],
                "outputs": [{"id": "output", "name": "可编辑框架图", "evidence_ids": ["E0001"]}],
                "innovations": [],
                "relations": [
                    {"source": "input", "target": "encoder", "type": "data_flow", "label": "进入", "evidence_ids": ["E0001"]},
                    {"source": "encoder", "target": "output", "type": "prediction", "label": "produces", "evidence_ids": ["E0001"]},
                ],
                "terminology": {},
            }
        }

        validation = validate_plan_grounding(plan, parsed)

        self.assertTrue(validation["ok"], validation["errors"])
        self.assertTrue(any("Input document" in item for item in validation["grounding_gaps"]))
        self.assertTrue(any("Document encoder" in item for item in validation["grounding_gaps"]))
        self.assertTrue(any("produces" in item for item in validation["grounding_gaps"]))

    def test_validation_accepts_verbatim_non_english_visible_labels(self):
        parsed = {
            "evidence": [
                {"id": "E0001", "text": "输入论文首先进入文档编码器,关系解码器连接实体,最后输出可编辑框架图。", "confidence": 1.0},
            ]
        }
        plan = {
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [{"id": "input", "name": "输入论文", "evidence_ids": ["E0001"]}],
                "modules": [{"id": "encoder", "name": "文档编码器", "evidence_ids": ["E0001"]}],
                "outputs": [{"id": "output", "name": "可编辑框架图", "evidence_ids": ["E0001"]}],
                "innovations": [],
                "relations": [
                    {"source": "input", "target": "encoder", "type": "data_flow", "label": "进入", "evidence_ids": ["E0001"]},
                    {"source": "encoder", "target": "output", "type": "prediction", "label": "", "evidence_ids": ["E0001"]},
                ],
                "terminology": {"文档编码器": "文档编码器"},
            }
        }

        self.assertTrue(validate_plan_grounding(plan, parsed)["ok"])

    def test_validation_ignores_mangled_glyphs_when_judging_the_source_script(self):
        # A figure export can leave a couple of glyphs from a symbol range
        # inside an otherwise English span. That noise must not read as a CJK
        # source, which would reject an English label as a translation.
        parsed = {
            "evidence": [
                {"id": "E0001", "text": "Class", "confidence": 1.0},
                {"id": "E0002", "text": "Embedding", "confidence": 1.0},
                {"id": "E0003", "text": "\u00fe \u00d7 \ufffd\u3559\u00d7 \ufffd\u3560", "confidence": 0.2},
                {"id": "E0004", "text": "The class embedding summarises every patch token into one vector.", "confidence": 1.0},
            ]
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [],
                "modules": [{"id": "class_embedding", "name": "Class Token", "evidence_ids": ["E0001", "E0002", "E0003", "E0004"]}],
                "outputs": [],
                "innovations": [],
                "relations": [],
                "terminology": {},
            },
        }

        validation = validate_plan_grounding(plan, parsed)

        self.assertTrue(validation["ok"], validation["errors"])

    def test_contract_repairs_cross_script_terminology_to_verbatim_source_key(self):
        parsed = {
            "evidence": [
                {"id": "E0001", "text": "输入论文首先进入文档编码器,关系解码器连接实体,最后输出可编辑框架图。", "confidence": 1.0},
            ]
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [{"id": "input", "name": "输入论文", "evidence_ids": ["E0001"]}],
                "modules": [{"id": "encoder", "name": "文档编码器", "evidence_ids": ["E0001"]}],
                "outputs": [{"id": "output", "name": "可编辑框架图", "evidence_ids": ["E0001"]}],
                "innovations": [],
                "relations": [{"source": "input", "target": "encoder", "type": "data_flow", "label": "进入", "evidence_ids": ["E0001"]}],
                "terminology": {"文档编码器": "Document encoder"},
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        self.assertEqual(spec["terminology"]["文档编码器"], "文档编码器")
        self.assertIn("文档编码器", spec["required_labels"])
        self.assertTrue(validate_plan_grounding(plan, parsed)["ok"])

    def test_contract_grounds_exact_innovation_only_from_novelty_evidence(self):
        parsed = {
            "evidence": [
                {"id": "E0001", "text": "本文提出论文编码器和关系解码器,生成可编辑框架图。", "section_hint": "摘要", "confidence": 1.0},
                {"id": "E0002", "text": "A baseline also contains a relation decoder.", "section_hint": "Related Work", "confidence": 1.0},
            ]
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [],
                "modules": [{"id": "decoder", "name": "关系解码器", "evidence_ids": ["E0001"]}],
                "outputs": [],
                "innovations": [{"id": "innovation_decoder", "name": "关系解码器", "evidence_ids": []}],
                "relations": [],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        self.assertEqual(spec["innovations"][0]["evidence_ids"], ["E0001"])
        self.assertTrue(validate_plan_grounding(plan, parsed)["ok"])

    def test_a_visible_input_without_evidence_is_a_recorded_gap(self):
        parsed = {"evidence": [{"id": "E0001", "text": "The encoder produces an output.", "confidence": 1.0}]}
        plan = {
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [{"id": "input", "name": "Input Image", "evidence_ids": []}],
                "modules": [{"id": "encoder", "name": "encoder", "evidence_ids": ["E0001"]}],
                "outputs": [],
                "innovations": [],
                "relations": [],
                "terminology": {},
            }
        }

        validation = validate_plan_grounding(plan, parsed)

        self.assertTrue(validation["ok"], validation["errors"])
        self.assertIn("inputs[0] has no evidence_ids", validation["grounding_gaps"])

    def test_contract_grounds_short_diagram_labels_only_on_overview_figure_page(self):
        parsed = {
            "document_index": {"figures": [{"page": 3, "caption": "Figure 1: Overall pre-training and fine-tuning procedures."}]},
            "evidence": [
                {"id": "E0001", "page": 3, "text": "T 1", "confidence": 1.0},
                {"id": "E0002", "page": 3, "text": "C", "confidence": 1.0},
                {"id": "E0003", "page": 8, "text": "C", "confidence": 1.0},
            ],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [],
                "modules": [{"id": "encoder", "name": "Encoder", "evidence_ids": ["E0001"]}],
                "outputs": [
                    {"id": "token_output", "name": "T 1", "evidence_ids": []},
                    {"id": "class_output", "name": "C", "evidence_ids": []},
                ],
                "innovations": [],
                "relations": [],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        self.assertEqual(spec["outputs"][0]["evidence_ids"], ["E0001"])
        self.assertEqual(spec["outputs"][1]["evidence_ids"], ["E0002"])
        self.assertTrue(validate_plan_grounding(plan, parsed)["ok"])

    def test_english_translation_in_a_spanish_paper_is_a_recorded_gap(self):
        evidence_text = "Proponemos un codificador de documentos que transforma el articulo en evidencia estructurada. El documento de entrada pasa al codificador y el decodificador de relaciones produce una salida editable para los resultados."
        parsed = {"evidence": [{"id": "E0001", "text": evidence_text, "confidence": 1.0}]}
        plan = {
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [{"id": "input", "name": "Input Document", "evidence_ids": ["E0001"]}],
                "modules": [{"id": "encoder", "name": "Document Encoder", "evidence_ids": ["E0001"]}],
                "outputs": [{"id": "output", "name": "Editable Output", "evidence_ids": ["E0001"]}],
                "innovations": [],
                "relations": [],
                "terminology": {},
            }
        }

        validation = validate_plan_grounding(plan, parsed)

        self.assertTrue(validation["ok"], validation["errors"])
        self.assertEqual(validation["detected_source_language"], "spanish")
        self.assertTrue(any("translated from spanish" in item for item in validation["grounding_gaps"]))

    def test_validation_accepts_verbatim_spanish_labels(self):
        evidence_text = "Proponemos un codificador de documentos que transforma el articulo en evidencia estructurada. El documento de entrada pasa al codificador y el decodificador de relaciones produce una salida editable para los resultados."
        parsed = {"evidence": [{"id": "E0001", "text": evidence_text, "confidence": 1.0}]}
        plan = {
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [{"id": "input", "name": "documento de entrada", "evidence_ids": ["E0001"]}],
                "modules": [{"id": "encoder", "name": "codificador de documentos", "evidence_ids": ["E0001"]}],
                "outputs": [{"id": "output", "name": "salida editable", "evidence_ids": ["E0001"]}],
                "innovations": [],
                "relations": [],
                "terminology": {},
            }
        }

        validation = validate_plan_grounding(plan, parsed)

        self.assertTrue(validation["ok"])
        self.assertEqual(validation["detected_source_language"], "spanish")

    def test_contract_clears_translated_relation_label_when_direction_is_grounded(self):
        evidence_text = "El codificador de documentos produce evidencia estructurada para el decodificador de relaciones."
        parsed = {"evidence": [{"id": "E0001", "text": evidence_text, "confidence": 1.0}]}
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [],
                "modules": [
                    {"id": "encoder", "name": "codificador de documentos", "evidence_ids": ["E0001"]},
                    {"id": "decoder", "name": "decodificador de relaciones", "evidence_ids": ["E0001"]},
                ],
                "outputs": [],
                "innovations": [],
                "relations": [{"source": "encoder", "target": "decoder", "type": "data_flow", "label": "structured evidence", "evidence_ids": ["E0001"]}],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        self.assertEqual(spec["relations"][0]["label"], "")
        self.assertEqual(plan["contract_completion_report"]["removed_unverbatim_relation_labels"], ["encoder->decoder:structured evidence"])
        self.assertTrue(validate_plan_grounding(plan, parsed)["ok"])

    def test_contract_preserves_verbatim_relation_label(self):
        evidence_text = "El codificador produce evidencia estructurada para el decodificador."
        parsed = {"evidence": [{"id": "E0001", "text": evidence_text, "confidence": 1.0}]}
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [],
                "modules": [
                    {"id": "encoder", "name": "codificador", "evidence_ids": ["E0001"]},
                    {"id": "decoder", "name": "decodificador", "evidence_ids": ["E0001"]},
                ],
                "outputs": [],
                "innovations": [],
                "relations": [{"source": "encoder", "target": "decoder", "type": "data_flow", "label": "evidencia estructurada", "evidence_ids": ["E0001"]}],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        self.assertEqual(spec["relations"][0]["label"], "evidencia estructurada")
        self.assertEqual(plan["contract_completion_report"]["removed_unverbatim_relation_labels"], [])

    def test_overlay_deduplicates_innovation_label_that_repeats_a_module(self):
        parsed = {"evidence": [{"id": "E0001", "text": "Proponemos un codificador de documentos para procesar la entrada.", "confidence": 1.0}]}
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [],
                "modules": [{"id": "encoder", "name": "codificador de documentos", "evidence_ids": ["E0001"]}],
                "outputs": [{"id": "output", "name": "salida", "evidence_ids": ["E0001"]}],
                "innovations": [{"id": "innovation_encoder", "name": "Codificador de documentos", "evidence_ids": ["E0001"]}],
                "relations": [{"source": "innovation_encoder", "target": "output", "type": "data_flow", "label": "", "evidence_ids": ["E0001"]}],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        overlay = build_overlay_spec(plan)

        self.assertEqual(len(spec["innovations"]), 1)
        self.assertEqual([item["text"].casefold() for item in overlay["labels"]].count("codificador de documentos"), 1)
        self.assertEqual(overlay["connectors"][0]["source"], "encoder")
        self.assertEqual(overlay["connectors"][0]["target"], "output")

    def test_contract_grounds_declared_multword_entity_by_exact_paper_term(self):
        parsed = {
            "page_count": 6,
            "evidence": [
                {"id": "E0001", "page": 6, "text": "Transformer encoder. First, a convolution reduces the channel dimension.", "confidence": 0.98},
            ],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "modules": [{"id": "encoder", "name": "Transformer Encoder", "evidence_ids": []}],
                "inputs": [],
                "outputs": [],
                "innovations": [],
                "relations": [],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        self.assertEqual(spec["modules"][0]["evidence_ids"], ["E0001"])
        self.assertIn("encoder", plan["contract_completion_report"]["grounded_entities"])

    def test_contract_does_not_add_cross_modal_edges_to_connected_inputs(self):
        parsed = {"evidence": [{"id": "E0001", "page": 1, "text": "图像进入图像编码器，论文文本进入文本编码器。", "confidence": 1.0}]}
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [
                    {"id": "image", "name": "图像", "evidence_ids": ["E0001"]},
                    {"id": "text", "name": "论文文本", "evidence_ids": ["E0001"]},
                ],
                "modules": [
                    {"id": "image_encoder", "name": "图像编码器", "evidence_ids": ["E0001"]},
                    {"id": "text_encoder", "name": "文本编码器", "evidence_ids": ["E0001"]},
                ],
                "outputs": [],
                "innovations": [],
                "relations": [
                    {"source": "image", "target": "image_encoder", "evidence_ids": ["E0001"]},
                    {"source": "text", "target": "text_encoder", "evidence_ids": ["E0001"]},
                ],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        self.assertEqual(pairs, {("image", "image_encoder"), ("text", "text_encoder")})

    def test_contract_leaves_unknown_input_unconnected_instead_of_guessing(self):
        parsed = {"evidence": [{"id": "E0001", "page": 1, "text": "A scientific system contains two modules.", "confidence": 1.0}]}
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [{"id": "signal", "name": "Scientific Signal", "evidence_ids": ["E0001"]}],
                "modules": [
                    {"id": "first", "name": "First Module", "evidence_ids": ["E0001"]},
                    {"id": "second", "name": "Second Module", "evidence_ids": ["E0001"]},
                ],
                "outputs": [],
                "innovations": [],
                "relations": [],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        self.assertFalse(any(item["source"] == "signal" for item in spec["relations"]))

    def test_contract_normalizes_string_entities_and_relation_alias_fields(self):
        parsed = {
            "evidence": [
                {"id": "E0001", "page": 1, "text": "The Image Encoder sends representations to the Mask Decoder.", "confidence": 1.0},
            ],
        }
        plan = {
            "paper_summary": {
                "unknowns": [],
                "core_modules": ["Image Encoder", "Mask Decoder"],
            },
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [],
                "modules": [],
                "outputs": [],
                "innovations": [],
                "relations": [
                    {"source_id": "Image Encoder", "target_id": "Mask Decoder", "relation_type": "feature_flow", "evidence_ids": ["E0001"]},
                ],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        validation = validate_plan_grounding(plan, parsed)

        self.assertEqual([item["name"] for item in spec["modules"]], ["Image Encoder", "Mask Decoder"])
        self.assertEqual(spec["relations"][0]["source"], spec["modules"][0]["id"])
        self.assertEqual(spec["relations"][0]["target"], spec["modules"][1]["id"])
        self.assertEqual(spec["relations"][0]["type"], "feature_flow")
        self.assertTrue(validation["ok"])

    def test_a_relation_without_evidence_is_a_recorded_gap(self):
        plan = {
            "figure_specification": {
                "research_problem": {"text": "Problem", "evidence_ids": ["E0001"]},
                "central_claim": {"text": "Claim", "evidence_ids": ["E0001"]},
                "modules": [
                    {"id": "a", "name": "A", "evidence_ids": ["E0001"]},
                    {"id": "b", "name": "B", "evidence_ids": ["E0001"]},
                ],
                "relations": [{"source": "a", "target": "b", "type": "data_flow", "evidence_ids": []}],
                "inputs": [],
                "outputs": [],
                "innovations": [],
            }
        }

        result = validate_plan_grounding(plan, self._parsed())

        self.assertTrue(result["ok"], result["errors"])
        self.assertIn("relation a -> b has no evidence_ids", result["grounding_gaps"])

    def test_low_confidence_evidence_is_reported_as_uncertain(self):
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "modules": [{"id": "ocr_module", "name": "OCR Module", "evidence_ids": ["E0002"]}],
                "relations": [],
                "inputs": [],
                "outputs": [],
                "innovations": [],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, self._parsed())

        self.assertIn("ocr_module relies only on low-confidence OCR evidence", spec["uncertainties"])

    def test_review_fact_ids_expand_to_page_evidence_and_encoder_group(self):
        parsed = {"evidence": [{"id": "E0001", "confidence": 1.0}]}
        review = {"modules": [{"id": "review_encoders", "evidence_ids": ["E0001"]}]}
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "modules": [
                    {"id": "image_encoder", "name": "Image Encoder", "evidence_ids": ["review_encoders"]},
                    {"id": "text_encoder", "name": "Text Encoder", "evidence_ids": ["review_encoders"]},
                    {"id": "audio_encoder", "name": "Audio Encoder", "evidence_ids": ["review_encoders"]},
                    {"id": "joint", "name": "Joint Embedding Space", "evidence_ids": ["E0001"]},
                ],
                "inputs": [{"id": "image", "name": "Image", "evidence_ids": ["E0001"]}],
                "outputs": [],
                "innovations": [],
                "relations": [
                    {"source": "image", "target": "image_encoder", "evidence_ids": ["E0001"]},
                    {"source": "image_encoder", "target": "joint", "evidence_ids": ["E0001"]},
                ],
                "terminology": {},
            },
        }

        expand_plan_evidence(plan, review, parsed)
        spec = normalize_figure_contract(plan, parsed)

        self.assertEqual(plan["figure_specification"]["modules"][0]["evidence_ids"], ["E0001"])
        self.assertTrue(any(item.get("id") == "modality_encoders_group" for item in spec["modules"]))
        self.assertTrue(any(item.get("source") == "image" and item.get("target") == "modality_encoders_group" for item in spec["relations"]))

    def test_contract_completion_adds_required_conditioning_and_output_boundary(self):
        parsed = {"evidence": [{"id": "E0001", "confidence": 1.0}]}
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "modules": [
                    {"id": "patches", "name": "Image Patches", "evidence_ids": ["E0001"]},
                    {"id": "encoder", "name": "Transformer Encoder", "evidence_ids": ["E0001"]},
                    {"id": "head", "name": "MLP Head", "evidence_ids": ["E0001"]},
                ],
                "inputs": [{"id": "image", "name": "Input Image", "evidence_ids": ["E0001"]}],
                "outputs": [{"id": "prediction", "name": "Class Prediction", "evidence_ids": ["E0001"]}],
                "relations": [
                    {"source": "patches", "target": "encoder", "evidence_ids": ["E0001"]},
                    {"source": "encoder", "target": "head", "evidence_ids": ["E0001"]},
                ],
                "must_show": [{"text": "Class Token", "evidence_ids": ["E0001"]}],
                "innovations": [],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        class_token = next(item for item in spec["modules"] if item["name"] == "Class Token")
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}
        self.assertIn(("image", "patches"), pairs)
        self.assertIn((class_token["id"], "encoder"), pairs)
        self.assertIn(("head", "prediction"), pairs)

    def test_zero_shot_prediction_is_not_upgraded_to_classifier(self):
        caption = "Fig. 1: The model offers zero-shot prediction of reporter assay readout."
        parsed = {
            "page_count": 2,
            "document_index": {"figures": [{"page": 1, "caption": caption}]},
            "evidence": [
                {"id": "E0001", "page": 1, "kind": "caption", "text": caption, "confidence": 1.0},
            ],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "modules": [{"id": "head", "name": "Prediction Head", "evidence_ids": ["E0001"]}],
                "inputs": [],
                "outputs": [
                    {"id": "output_zero_shot_classifier", "name": "Zero-Shot Classifier", "evidence_ids": ["E0001"]},
                    {"id": "output_zero_shot_prediction", "name": "Zero-shot Prediction", "evidence_ids": ["E0001"]},
                ],
                "relations": [],
                "innovations": [],
                "must_show": [],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        output_names = {item["name"] for item in spec["outputs"]}

        self.assertIn("Zero-shot Prediction", output_names)
        self.assertNotIn("Zero-Shot Classifier", output_names)
        self.assertEqual(sum(item["name"] == "Zero-shot Prediction" for item in spec["outputs"]), 1)

    def test_feedback_contract_merges_formula_and_display_name_duplicates(self):
        caption = "Figure 1: Input x is used to Generate an Initial Output, obtain Self-Feedback, Refine it into a Refined Output, and repeat the feedback loop."
        parsed = {
            "page_count": 3,
            "document_index": {"figures": [{"page": 1, "caption": caption}]},
            "evidence": [{"id": "E0001", "page": 1, "kind": "caption", "text": caption, "confidence": 1.0}],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "inputs": [
                    {"id": "input_x", "name": "input x", "evidence_ids": ["E0001"]},
                    {"id": "input_input", "name": "Input", "evidence_ids": ["E0001"]},
                ],
                "modules": [
                    {"id": "generate", "name": "Generate", "evidence_ids": ["E0001"]},
                    {"id": "initial_output", "name": "Initial Output", "evidence_ids": ["E0001"]},
                    {"id": "feedback", "name": "Self-Feedback", "evidence_ids": ["E0001"]},
                    {"id": "refine", "name": "Refine", "evidence_ids": ["E0001"]},
                ],
                "outputs": [
                    {"id": "y_0", "name": "y 0", "evidence_ids": ["E0001"]},
                    {"id": "y_t_plus_1", "name": "y t + 1", "evidence_ids": ["E0001"]},
                    {"id": "refined_output", "name": "Refined Output", "evidence_ids": ["E0001"]},
                ],
                "relations": [
                    {"source": "input_x", "target": "generate", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "generate", "target": "y_0", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "y_0", "target": "feedback", "type": "feedback", "evidence_ids": ["E0001"]},
                    {"source": "refine", "target": "y_t_plus_1", "type": "data_flow", "evidence_ids": ["E0001"]},
                ],
                "innovations": [],
                "must_show": [],
                "terminology": {"y 0": "Initial Output", "y t + 1": "Refined Output"},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        all_entities = [item for field in ("inputs", "modules", "outputs") for item in spec[field]]
        normalized_names = [re.sub(r"[^a-z0-9]+", "", item["name"].casefold()) for item in all_entities]
        endpoint_ids = {item["id"] for item in all_entities}

        self.assertEqual(len(spec["inputs"]), 1)
        self.assertTrue(normalized_names[0].startswith("input"))
        self.assertEqual(sum(name == "initialoutput" for name in normalized_names), 1)
        self.assertEqual(sum(name == "refinedoutput" for name in normalized_names), 1)
        self.assertTrue(all(item["source"] in endpoint_ids and item["target"] in endpoint_ids for item in spec["relations"]))
        self.assertEqual(len(spec["required_labels"]), len({_normalized.casefold() for _normalized in spec["required_labels"]}))

    def test_feedback_contract_removes_parallel_model_path_and_keeps_one_eight_edge_loop(self):
        caption = "Figure 1: Input x is used to Generate an Initial Output, obtain Self-Feedback from FEEDBACK, REFINE it into a Refined Output, and iterate."
        parsed = {
            "page_count": 3,
            "document_index": {"figures": [{"page": 1, "caption": caption}]},
            "evidence": [{"id": "E0001", "page": 1, "kind": "caption", "text": caption, "confidence": 1.0}],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "topology": "feedback",
                "inputs": [{"id": "input", "name": "input x", "evidence_ids": ["E0001"]}],
                "modules": [
                    {"id": "model", "name": "Model M", "evidence_ids": ["E0001"]},
                    {"id": "generate", "name": "Generate", "evidence_ids": ["E0001"]},
                    {"id": "initial", "name": "Initial Output", "evidence_ids": ["E0001"]},
                    {"id": "feedback", "name": "FEEDBACK", "evidence_ids": ["E0001"]},
                    {"id": "self_feedback", "name": "Self-Feedback", "evidence_ids": ["E0001"]},
                    {"id": "refine", "name": "REFINE", "evidence_ids": ["E0001"]},
                ],
                "outputs": [{"id": "refined", "name": "Refined output", "evidence_ids": ["E0001"]}],
                "innovations": [{"id": "iteration", "name": "Iterative self-refinement", "evidence_ids": ["E0001"]}],
                "relations": [
                    {"source": "input", "target": "model", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "model", "target": "initial", "type": "prediction", "evidence_ids": ["E0001"]},
                    {"source": "feedback", "target": "model", "type": "conditioning", "evidence_ids": ["E0001"]},
                    {"source": "model", "target": "refine", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "input", "target": "generate", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "generate", "target": "initial", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "initial", "target": "feedback", "type": "evaluation", "evidence_ids": ["E0001"]},
                    {"source": "feedback", "target": "self_feedback", "type": "feedback", "evidence_ids": ["E0001"]},
                    {"source": "initial", "target": "refine", "type": "revision_input", "evidence_ids": ["E0001"]},
                    {"source": "self_feedback", "target": "refine", "type": "feedback", "evidence_ids": ["E0001"]},
                    {"source": "refine", "target": "refined", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "refined", "target": "feedback", "type": "feedback_loop", "evidence_ids": ["E0001"]},
                ],
                "must_show": [],
                "terminology": {},
                "repeatable_labels": ["Model M"],
            },
            "design_plan": {"summary": "Old plan", "reading_order": ["iteration"], "preserve": ["Iterative self-refinement"]},
            "layout_intent": {"summary": "Old layout"},
            "visual_metaphors": {"summary": "Old metaphors", "items": [{"module_id": "iteration", "metaphor": "Iterative self-refinement"}]},
        }

        spec = normalize_figure_contract(plan, parsed)
        synchronize_plan_to_contract(plan)
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        self.assertEqual(len(spec["relations"]), 8)
        self.assertEqual(spec["innovations"], [])
        self.assertFalse(any("model" in pair for pair in pairs))
        self.assertIn(("input", "generate"), pairs)
        self.assertIn(("refined", "feedback"), pairs)
        self.assertNotIn("Iterative self-refinement", spec["must_show"])
        self.assertEqual(plan["design_plan"]["preserve"], spec["required_labels"])

    def test_branch_contract_merges_head_output_aliases_and_removes_cross_branch_edges(self):
        caption = "Figure 1: An Input Image passes through a Backbone and RPN to RoIAlign, followed by parallel classification, bounding-box regression, and mask branches."
        parsed = {
            "page_count": 3,
            "document_index": {"figures": [{"page": 1, "caption": caption}]},
            "evidence": [{"id": "E0001", "page": 1, "kind": "caption", "text": caption, "confidence": 1.0}],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "topology": "branch",
                "inputs": [{"id": "image", "name": "Input Image", "evidence_ids": ["E0001"]}],
                "modules": [
                    {"id": "backbone", "name": "Backbone", "evidence_ids": ["E0001"]},
                    {"id": "rpn", "name": "RPN", "evidence_ids": ["E0001"]},
                    {"id": "proposals", "name": "Region Proposals", "evidence_ids": ["E0001"]},
                    {"id": "roi", "name": "RoIAlign", "evidence_ids": ["E0001"]},
                    {"id": "class_branch", "name": "classification branch", "evidence_ids": ["E0001"]},
                    {"id": "box_branch", "name": "bounding-box regression", "evidence_ids": ["E0001"]},
                    {"id": "mask_branch", "name": "mask branch", "evidence_ids": ["E0001"]},
                ],
                "outputs": [
                    {"id": "class_output", "name": "Classification", "evidence_ids": ["E0001"]},
                    {"id": "box_output", "name": "bounding-box offset", "evidence_ids": ["E0001"]},
                    {"id": "mask_output", "name": "binary mask", "evidence_ids": ["E0001"]},
                ],
                "relations": [
                    {"source": "image", "target": "backbone", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "backbone", "target": "rpn", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "proposals", "target": "roi", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "roi", "target": "class_branch", "type": "branch", "evidence_ids": ["E0001"]},
                    {"source": "roi", "target": "box_branch", "type": "branch", "evidence_ids": ["E0001"]},
                    {"source": "roi", "target": "mask_branch", "type": "branch", "evidence_ids": ["E0001"]},
                    {"source": "mask_branch", "target": "box_output", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "mask_branch", "target": "mask_output", "type": "data_flow", "evidence_ids": ["E0001"]},
                ],
                "innovations": [],
                "must_show": [],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        names = [re.sub(r"[^a-z0-9]+", "", item["name"].casefold()) for field in ("inputs", "modules", "outputs") for item in spec[field]]
        branch_names = [name for name in names if name in {"classification", "classificationbranch", "boundingboxregression", "boxbranch", "maskbranch"}]
        endpoint_by_name = {re.sub(r"[^a-z0-9]+", "", item["name"].casefold()): item["id"] for field in ("inputs", "modules", "outputs") for item in spec[field]}
        branch_ids = {endpoint_by_name[name] for name in branch_names}

        self.assertEqual(len(spec["inputs"]), 1)
        self.assertEqual(sum(name in {"rpn", "regionproposals"} for name in names), 1)
        self.assertEqual(len(branch_names), 3)
        self.assertFalse(any(item["source"] in branch_ids and item["target"] in branch_ids and item["source"] != item["target"] for item in spec["relations"]))

    def test_multimodal_contract_keeps_shared_embedding_story_and_removes_training_detail(self):
        caption = "Figure 1: ImageBind uses images to bind text, audio, depth, thermal, and IMU in one joint embedding space with emergent cross-modal alignment."
        parsed = {
            "page_count": 3,
            "document_index": {"figures": [{"page": 1, "caption": caption}]},
            "evidence": [{"id": "E0001", "page": 1, "kind": "caption", "text": caption, "confidence": 1.0}],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "topology": "branch",
                "inputs": [
                    *[{"id": name.lower(), "name": name, "evidence_ids": ["E0001"]} for name in ("Image", "Text", "Audio", "Depth", "Thermal", "IMU")],
                    {"id": "images_alias", "name": "Images", "evidence_ids": ["E0001"]},
                    {"id": "class_descriptions", "name": "Class Descriptions", "evidence_ids": ["E0001"]},
                ],
                "modules": [
                    {"id": "modality_encoders", "name": "Modality encoders", "evidence_ids": ["E0001"]},
                    {"id": "common_space", "name": "Common embedding space", "evidence_ids": ["E0001"]},
                    {"id": "text_encoder", "name": "Text Encoder", "evidence_ids": ["E0001"]},
                    {"id": "contrastive", "name": "Contrastive Learning", "evidence_ids": ["E0001"]},
                ],
                "outputs": [{"id": "joint_space", "name": "Joint Embedding Space", "evidence_ids": ["E0001"]}],
                "innovations": [{"id": "emergent", "name": "Emergent Cross-modal Alignment", "evidence_ids": ["E0001"]}],
                "relations": [],
                "must_show": [],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        labels = [_normalized for field in ("inputs", "modules", "outputs") for item in spec[field] if (_normalized := re.sub(r"[^a-z0-9]+", "", item["name"].casefold()))]

        self.assertEqual(spec["topology"], "multimodal")
        self.assertEqual(len(spec["inputs"]), 6)
        self.assertEqual(labels.count("modalityencoders"), 1)
        self.assertEqual(labels.count("jointembeddingspace"), 1)
        self.assertEqual(labels.count("emergentcrossmodalalignment"), 1)
        self.assertNotIn("classdescriptions", labels)
        self.assertNotIn("textencoder", labels)
        self.assertNotIn("contrastivelearning", labels)
        self.assertEqual(len(spec["relations"]), 8)
        self.assertIn("Emergent Cross-modal Alignment", spec["required_labels"])

    def test_dense_multiframe_contract_keeps_task_model_and_data_engine_panels(self):
        caption = "Figure 1: Three interconnected components: a promptable segmentation task, a segmentation model (SAM), and a data engine for collecting SA-1B."
        parsed = {
            "page_count": 3,
            "document_index": {"figures": [{"page": 1, "caption": caption}]},
            "evidence": [{"id": "E0001", "page": 1, "kind": "caption", "text": caption, "confidence": 1.0}],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "topology": "branch",
                "inputs": [
                    {"id": "image", "name": "image", "evidence_ids": ["E0001"]},
                    {"id": "images_alias", "name": "Images", "evidence_ids": ["E0001"]},
                    {"id": "prompts", "name": "input prompts", "evidence_ids": ["E0001"]},
                    {"id": "prompt", "name": "Prompt", "evidence_ids": ["E0001"]},
                ],
                "modules": [
                    {"id": "task", "name": "Prompt-able segmentation task", "role": "overview_panel", "evidence_ids": ["E0001"]},
                    {"id": "sam", "name": "Segmentation model (SAM)", "role": "overview_panel", "evidence_ids": ["E0001"]},
                    {"id": "engine", "name": "Data engine", "role": "overview_panel", "evidence_ids": ["E0001"]},
                    {"id": "image_encoder", "name": "heavyweight image encoder", "evidence_ids": ["E0001"]},
                    {"id": "prompt_encoder", "name": "prompt encoder", "evidence_ids": ["E0001"]},
                    {"id": "mask_decoder", "name": "lightweight mask decoder", "evidence_ids": ["E0001"]},
                    {"id": "image_embeddings", "name": "Image Embeddings", "evidence_ids": ["E0001"]},
                    {"id": "transformer", "name": "Transformer Encoder", "evidence_ids": ["E0001"]},
                    {"id": "position", "name": "Positional Encoding", "evidence_ids": ["E0001"]},
                    {"id": "modality_group", "name": "Modality Encoders", "evidence_ids": ["E0001"]},
                    {"id": "assisted", "name": "Assisted-Manual", "evidence_ids": ["E0001"]},
                    {"id": "semi", "name": "Semi-Automatic", "evidence_ids": ["E0001"]},
                    {"id": "fully", "name": "Fully Automatic", "evidence_ids": ["E0001"]},
                ],
                "outputs": [
                    {"id": "mask", "name": "Valid Segmentation Mask", "evidence_ids": ["E0001"]},
                    {"id": "sa1b", "name": "SA-1B", "evidence_ids": ["E0001"]},
                    {"id": "confidence", "name": "confidence scores", "evidence_ids": ["E0001"]},
                ],
                "innovations": [],
                "relations": [],
                "must_show": [],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        labels = [re.sub(r"[^a-z0-9]+", "", item["name"].casefold()) for field in ("inputs", "modules", "outputs") for item in spec[field]]

        self.assertEqual(spec["topology"], "dense_multiframe")
        self.assertEqual(len(labels), 13)
        self.assertEqual(len(spec["relations"]), 9)
        self.assertIn("promptablesegmentationtask", labels)
        self.assertIn("segmentanythingmodel", labels)
        self.assertIn("dataengine", labels)
        self.assertIn("imageencoder", labels)
        self.assertIn("maskdecoder", labels)
        self.assertNotIn("imageembeddings", labels)
        self.assertNotIn("transformerencoder", labels)
        self.assertNotIn("positionalencoding", labels)
        self.assertNotIn("modalityencoders", labels)
        self.assertNotIn("confidencescores", labels)

    def test_overview_caption_and_stage_list_complete_missing_panels(self):
        caption = "Figure 1: Three interconnected components: a promptable segmentation task, a segmentation model (SAM) that powers data annotation, and a data engine for collecting SA-1B, our dataset."
        parsed = {
            "document_index": {"figures": [{"page": 1, "caption": caption}]},
            "evidence": [
                {"id": "E0001", "page": 1, "kind": "caption", "text": caption, "confidence": 1.0},
                {"id": "E0002", "page": 2, "kind": "paragraph", "text": "Our data engine has three stages:", "confidence": 1.0},
                {"id": "E0003", "page": 2, "kind": "paragraph", "text": "assisted-manual, semi-automatic, and fully automatic.", "confidence": 1.0},
            ],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "modules": [], "inputs": [], "outputs": [], "innovations": [], "relations": [], "must_show": [], "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        labels = {_item["name"] for _item in spec["modules"]}
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        self.assertIn("Promptable segmentation task", labels)
        self.assertIn("Segmentation model (SAM)", labels)
        self.assertIn("Data engine", labels)
        self.assertIn("Assisted-Manual", labels)
        self.assertTrue(any(item.get("name") == "SA-1B" for item in spec["outputs"]))
        self.assertTrue(any("stage_assisted_manual" in source and "stage_semi_automatic" in target for source, target in pairs))

    def test_generic_completion_recovers_detr_without_bibliography_noise(self):
        captions = [
            "Fig. 1: DETR directly predicts the final set of detections by combining a common CNN with a transformer architecture. During training, bipartite matching uniquely assigns predictions with ground truth boxes and class predictions.",
            "Fig. 2: DETR uses a conventional CNN backbone for an input image, supplements features with a positional encoding, passes them into a transformer encoder, and uses object queries in a transformer decoder. A feed forward network predicts a class and bounding box.",
        ]
        parsed = {
            "page_count": 12,
            "document_index": {"figures": [{"page": index + 1, "caption": text} for index, text in enumerate(captions)]},
            "evidence": [
                {"id": f"E000{index + 1}", "page": index + 1, "kind": "caption", "text": text, "confidence": 1.0}
                for index, text in enumerate(captions)
            ] + [{"id": "E0099", "page": 11, "kind": "heading", "text": "Learning non-maximum suppression. In: ECCV", "section_hint": "References", "confidence": 1.0}],
        }
        plan = {"paper_summary": {"unknowns": []}, "figure_specification": {"modules": [], "inputs": [], "outputs": [], "relations": [], "innovations": [], "must_show": [], "terminology": {}}}

        spec = normalize_figure_contract(plan, parsed)
        labels = {_item.get("name") for field in ("inputs", "modules", "outputs") for _item in spec[field]}
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}
        ids = {_item.get("name"): _item.get("id") for field in ("inputs", "modules", "outputs") for _item in spec[field]}

        self.assertIn("Object Queries", labels)
        self.assertIn("Bipartite Matching", labels)
        self.assertIn("Bounding Box Predictions", labels)
        self.assertNotIn("Non-Maximum Suppression", labels)
        self.assertIn((ids["Object Queries"], ids["Transformer Decoder"]), pairs)
        self.assertIn((ids["Feed Forward Network"], ids["Bounding Box Predictions"]), pairs)

    def test_generic_completion_keeps_clip_modalities_and_encoders_distinct(self):
        caption = "Figure 1. Summary of our approach. CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time the text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target classes."
        parsed = {
            "page_count": 8,
            "document_index": {"figures": [{"page": 2, "caption": caption}]},
            "evidence": [
                {"id": "E0001", "page": 2, "kind": "caption", "text": caption, "section_hint": "Figure Captions", "confidence": 1.0},
                {"id": "E0002", "page": 3, "kind": "paragraph", "text": "The image encoder and text encoder maximize similarity of image and text embeddings using a contrastive objective.", "section_hint": "Method", "confidence": 1.0},
            ],
        }
        plan = {"paper_summary": {"unknowns": []}, "figure_specification": {"modules": [{"id": "fallback_contrastive", "name": "Contrastive Pre-training", "role": "paper-derived stage requiring VLM verification", "evidence_ids": ["E0002"]}], "inputs": [], "outputs": [], "relations": [], "innovations": [], "must_show": [], "terminology": {}}}

        spec = normalize_figure_contract(plan, parsed)
        ids = {_item.get("name"): _item.get("id") for field in ("inputs", "modules", "outputs") for _item in spec[field]}
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        self.assertNotEqual(ids["Text"], ids["Text Encoder"])
        self.assertIn((ids["Text"], ids["Text Encoder"]), pairs)
        self.assertIn((ids["Class Descriptions"], ids["Text Encoder"]), pairs)
        self.assertIn((ids["Text Encoder"], ids["Text Embeddings"]), pairs)
        self.assertTrue(any(item.get("id") == "fallback_contrastive" for item in spec["modules"]))
        self.assertEqual(spec["topology"], "multimodal")

    def test_rich_clip_contract_reuses_singular_embedding_entities(self):
        caption = "Figure 1. CLIP trains an image encoder and a text encoder on paired image and text inputs, then embeds class descriptions for a zero-shot classifier."
        parsed = {
            "page_count": 8,
            "document_index": {"figures": [{"page": 2, "caption": caption}]},
            "evidence": [
                {"id": "E0001", "page": 2, "kind": "caption", "text": caption, "section_hint": "Figure Captions", "confidence": 1.0},
                {"id": "E0002", "page": 3, "kind": "paragraph", "text": "The image and text embeddings are aligned by a contrastive objective.", "section_hint": "Method", "confidence": 1.0},
            ],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "inputs": [
                    {"id": "images", "name": "Image", "evidence_ids": ["E0001"]},
                    {"id": "texts", "name": "Text", "evidence_ids": ["E0001"]},
                    {"id": "classes", "name": "Class Descriptions", "evidence_ids": ["E0001"]},
                ],
                "modules": [
                    {"id": "image_encoder", "name": "Image Encoder", "evidence_ids": ["E0001"]},
                    {"id": "text_encoder", "name": "Text Encoder", "evidence_ids": ["E0001"]},
                    {"id": "contrastive", "name": "Contrastive Learning", "evidence_ids": ["E0002"]},
                ],
                "outputs": [
                    {"id": "image_embedding", "name": "image embedding", "evidence_ids": ["E0002"]},
                    {"id": "text_embedding", "name": "text embedding", "evidence_ids": ["E0002"]},
                    {"id": "classifier", "name": "Zero-Shot Classifier", "evidence_ids": ["E0001"]},
                ],
                "relations": [{"source": "image_embedding", "target": "classifier", "type": "classification", "evidence_ids": ["E0001", "E0002"]}],
                "innovations": [],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        embedding_labels = [
            item.get("name")
            for field in ("inputs", "modules", "outputs")
            for item in spec[field]
            if "embedding" in str(item.get("name") or "").casefold()
        ]
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        self.assertEqual(len([label for label in embedding_labels if "image" in label.casefold()]), 1)
        self.assertEqual(len([label for label in embedding_labels if "text" in label.casefold()]), 1)
        self.assertIn(("text_embedding", "classifier"), pairs)

    def test_rich_bert_contract_uses_selected_caption_for_context_rules(self):
        caption = "Figure 2: BERT input representation. The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings."
        parsed = {
            "page_count": 12,
            "document_index": {"figures": [{"page": 5, "caption": caption}]},
            "evidence": [{"id": "E0001", "page": 5, "kind": "caption", "text": caption, "section_hint": "Figure Captions", "confidence": 1.0}],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "inputs": [{"id": "tokens", "name": "Input Sequence", "evidence_ids": ["E0001"]}],
                "modules": [
                    {"id": "token_embeddings", "name": "Token Embeddings", "evidence_ids": ["E0001"]},
                    {"id": "position_embeddings", "name": "Position Embeddings", "evidence_ids": ["E0001"]},
                    {"id": "bert", "name": "Bidirectional Transformer Encoder", "evidence_ids": ["E0001"]},
                ],
                "outputs": [],
                "innovations": [],
                "relations": [{"source": "tokens", "target": "token_embeddings", "type": "embedding", "evidence_ids": ["E0001"]}],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        ids = {item.get("name"): item.get("id") for field in ("inputs", "modules", "outputs") for item in spec[field]}
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        self.assertIn("Segment Embeddings", ids)
        self.assertIn("Input Representation", ids)
        self.assertIn((ids["Segment Embeddings"], ids["Input Representation"]), pairs)

    def test_rich_nerf_contract_reuses_parenthesized_mlp_and_recovers_distant_method_step(self):
        overview = "Figure 2: A neural radiance field samples 5D coordinates along camera rays and uses an MLP to predict color and volume density for volume rendering."
        encoding = "We transform each sampled 3D point with a positional encoding before passing it into the MLP."
        parsed = {
            "page_count": 10,
            "document_index": {"figures": [{"page": 2, "caption": overview}]},
            "evidence": [
                {"id": "E0001", "page": 2, "kind": "caption", "text": overview, "section_hint": "Figure Captions", "confidence": 1.0},
                {"id": "E0002", "page": 6, "kind": "paragraph", "text": encoding, "section_hint": "Method", "confidence": 1.0},
            ],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "inputs": [{"id": "ray", "name": "Camera Rays", "evidence_ids": ["E0001"]}],
                "modules": [
                    {"id": "points", "name": "Sampled 3D Points", "evidence_ids": ["E0001", "E0002"]},
                    {"id": "field", "name": "MLP (F_theta)", "evidence_ids": ["E0001", "E0002"]},
                    {"id": "render", "name": "Volume Rendering", "evidence_ids": ["E0001"]},
                ],
                "outputs": [{"id": "image", "name": "Rendered Image", "evidence_ids": ["E0001"]}],
                "innovations": [],
                "relations": [{"source": "ray", "target": "points", "type": "sampling", "evidence_ids": ["E0001"]}],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        mlps = [item for item in spec["modules"] if str(item.get("name") or "").casefold().startswith("mlp")]
        positional = next(item for item in spec["modules"] if item.get("name") == "Positional Encoding")
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        self.assertEqual(len(mlps), 1)
        self.assertIn(("points", positional["id"]), pairs)
        self.assertIn((positional["id"], "field"), pairs)

    def test_rich_transformer_contract_recovers_connected_steps_from_distant_method_evidence(self):
        caption = "Figure 1: The Transformer follows an encoder-decoder architecture."
        method = "We use learned embeddings to convert the input tokens and output tokens to vectors. Output embeddings are offset by one position. A linear transformation and softmax convert the decoder output to predicted next-token probabilities."
        parsed = {
            "page_count": 8,
            "document_index": {"figures": [{"page": 2, "caption": caption}]},
            "evidence": [
                {"id": "E0001", "page": 2, "kind": "caption", "text": caption, "section_hint": "Figure Captions", "confidence": 1.0},
                {"id": "E0002", "page": 6, "kind": "paragraph", "text": method, "section_hint": "Method", "confidence": 1.0},
            ],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "inputs": [
                    {"id": "input_embedding", "name": "Input Embedding", "evidence_ids": ["E0002"]},
                    {"id": "output_embedding", "name": "Output Embedding", "evidence_ids": ["E0002"]},
                    {"id": "target_sequence", "name": "output sequence", "evidence_ids": ["E0002"]},
                ],
                "modules": [
                    {"id": "encoder", "name": "Encoder Stack", "evidence_ids": ["E0001"]},
                    {"id": "decoder", "name": "Decoder Stack", "evidence_ids": ["E0001"]},
                    {"id": "softmax", "name": "Softmax", "evidence_ids": ["E0002"]},
                ],
                "outputs": [],
                "innovations": [],
                "relations": [{"source": "encoder", "target": "decoder", "type": "cross_attention", "evidence_ids": ["E0001"]}],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        ids = {item.get("name"): item.get("id") for field in ("inputs", "modules", "outputs") for item in spec[field]}
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        self.assertIn((ids["Inputs"], "input_embedding"), pairs)
        self.assertEqual(ids["Outputs (shifted right)"], "target_sequence")
        self.assertIn(("target_sequence", "output_embedding"), pairs)
        self.assertIn(("softmax", ids["Output Probabilities"]), pairs)

    def test_generic_completion_recovers_nerf_branch_without_false_multimodal_topology(self):
        caption = "Fig. 2: An overview of our neural radiance field and differentiable rendering procedure. We synthesize images by sampling 5D coordinates along camera rays, feeding locations and viewing direction into an MLP to produce a color and volume density, and using volume rendering to composite these values into an image."
        positional = "Fig. 4: Our full model passes input coordinates through a high-frequency positional encoding."
        parsed = {
            "page_count": 10,
            "document_index": {"figures": [{"page": 2, "caption": caption}, {"page": 4, "caption": positional}]},
            "evidence": [
                {"id": "E0001", "page": 2, "kind": "caption", "text": caption, "section_hint": "Method", "confidence": 1.0},
                {"id": "E0002", "page": 4, "kind": "caption", "text": positional, "section_hint": "Method", "confidence": 1.0},
            ],
        }
        plan = {"paper_summary": {"unknowns": []}, "figure_specification": {"modules": [], "inputs": [], "outputs": [], "relations": [], "innovations": [], "must_show": [], "terminology": {}}}

        spec = normalize_figure_contract(plan, parsed)
        ids = {_item.get("name"): _item.get("id") for field in ("inputs", "modules", "outputs") for _item in spec[field]}
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        self.assertIn((ids["Viewing Direction"], ids["MLP"]), pairs)
        self.assertIn((ids["MLP"], ids["Color"]), pairs)
        self.assertIn((ids["Volume Rendering"], ids["Rendered Image"]), pairs)
        self.assertEqual(spec["topology"], "branch")

    def test_generic_completion_recovers_encoder_decoder_from_method_lines(self):
        caption = "Figure 1: The model architecture."
        lines = [
            "Here, the encoder maps an input sequence of symbol representations to continuous representations.",
            "The encoder is composed of a stack of identical layers.",
            "The decoder is also composed of a stack of identical layers and attends over the output of the encoder stack.",
            "We use learned embeddings to convert the input",
            "tokens and output tokens to vectors. We also use the usual learned linear transfor-",
            "mation and softmax function to convert the decoder output to predicted next-token probabilities.",
            "The output embeddings are offset by one position.",
            "We add positional encodings to the input embeddings at the bottoms of the encoder and decoder stacks.",
        ]
        parsed = {
            "page_count": 8,
            "document_index": {"figures": [{"page": 2, "caption": caption}]},
            "evidence": [{"id": "E0001", "page": 2, "kind": "caption", "text": caption, "confidence": 1.0}]
            + [{"id": f"E00{index + 2:02d}", "page": 3 + index // 3, "kind": "paragraph", "text": text, "section_hint": "Method", "confidence": 1.0} for index, text in enumerate(lines)],
        }
        plan = {"paper_summary": {"unknowns": []}, "figure_specification": {"modules": [], "inputs": [], "outputs": [], "relations": [], "innovations": [], "must_show": [], "terminology": {}}}

        spec = normalize_figure_contract(plan, parsed)
        ids = {_item.get("name"): _item.get("id") for field in ("inputs", "modules", "outputs") for _item in spec[field]}
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        for label in ("Inputs", "Input Embedding", "Encoder Stack", "Outputs (shifted right)", "Output Embedding", "Decoder Stack", "Linear", "Softmax", "Output Probabilities"):
            self.assertIn(label, ids)
        self.assertIn((ids["Encoder Stack"], ids["Decoder Stack"]), pairs)
        self.assertIn((ids["Linear"], ids["Softmax"]), pairs)

    def test_generic_completion_treats_refinement_action_as_refined_output(self):
        caption = "Figure 1: Given an input, the system starts by generating an output, gets feedback, and then refines the previously generated output."
        parsed = {
            "page_count": 8,
            "document_index": {"figures": [{"page": 2, "caption": caption}]},
            "evidence": [{"id": "E0001", "page": 2, "kind": "caption", "text": caption, "section_hint": "Figure Captions", "confidence": 1.0}],
        }
        plan = {"paper_summary": {"unknowns": []}, "figure_specification": {"modules": [], "inputs": [], "outputs": [], "relations": [], "innovations": [], "must_show": [], "terminology": {}}}

        spec = normalize_figure_contract(plan, parsed)
        ids = {_item.get("name"): _item.get("id") for field in ("inputs", "modules", "outputs") for _item in spec[field]}
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        self.assertIn("Refine", ids)
        self.assertIn("Refined Output", ids)
        self.assertIn((ids["Refine"], ids["Refined Output"]), pairs)

    def test_generic_completion_recovers_embedding_pretraining_and_finetuning(self):
        captions = [
            "Figure 1: Overall pre-training and fine-tuning procedures. The same pre-trained model parameters are used to initialize models for different down-stream tasks.",
            "Figure 2: Input representation. The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings.",
        ]
        method = [
            "The model architecture is a multi-layer bidirectional Transformer encoder.",
            "Task #1: Masked LM",
            "Task #2: Next Sentence Prediction (NSP)",
            "We represent the input sequence and fine-tune using labeled data from the downstream tasks.",
        ]
        parsed = {
            "page_count": 8,
            "document_index": {"figures": [{"page": index + 2, "caption": text} for index, text in enumerate(captions)]},
            "evidence": [{"id": f"E000{index + 1}", "page": index + 2, "kind": "caption", "text": text, "section_hint": "Figure Captions", "confidence": 1.0} for index, text in enumerate(captions)]
            + [{"id": f"E001{index}", "page": 4, "kind": "paragraph", "text": text, "section_hint": "Method", "confidence": 1.0} for index, text in enumerate(method)],
        }
        plan = {"paper_summary": {"unknowns": []}, "figure_specification": {"modules": [], "inputs": [], "outputs": [], "relations": [], "innovations": [], "must_show": [], "terminology": {}}}

        spec = normalize_figure_contract(plan, parsed)
        ids = {_item.get("name"): _item.get("id") for field in ("inputs", "modules", "outputs") for _item in spec[field]}
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        for label in ("Input Sequence", "Token Embeddings", "Segment Embeddings", "Position Embeddings", "Input Representation", "Bidirectional Transformer Encoder", "Masked LM", "Next Sentence Prediction", "Fine-tuning", "Downstream Tasks"):
            self.assertIn(label, ids)
        self.assertIn((ids["Token Embeddings"], ids["Input Representation"]), pairs)
        self.assertIn((ids["Bidirectional Transformer Encoder"], ids["Fine-tuning"]), pairs)

    def test_generic_completion_recovers_elided_embedding_list_across_ocr_blocks(self):
        caption = "Figure 1: Overall pre-training and fine-tuning procedures."
        parsed = {
            "page_count": 16,
            "document_index": {"figures": [{"page": 3, "caption": caption}]},
            "evidence": [
                {"id": "E0001", "page": 3, "kind": "caption", "text": caption, "section_hint": "Figure Captions", "confidence": 0.99},
                {"id": "E0002", "page": 4, "kind": "paragraph", "text": "For a given token, its input representation is", "section_hint": "Method", "confidence": 0.98},
                {"id": "E0003", "page": 4, "kind": "paragraph", "text": "constructed by summing the corresponding token,", "section_hint": "Method", "confidence": 0.98},
                {"id": "E0004", "page": 4, "kind": "paragraph", "text": "segment, and position embeddings.", "section_hint": "Method", "confidence": 0.98},
                {"id": "E0005", "page": 4, "kind": "paragraph", "text": "The architecture is a multi-layer bidirectional Transformer encoder.", "section_hint": "Method", "confidence": 0.98},
                {"id": "E0006", "page": 4, "kind": "paragraph", "text": "We pre-train with Masked LM and Next Sentence Prediction before fine-tuning on downstream tasks.", "section_hint": "Method", "confidence": 0.98},
            ],
        }
        plan = {"paper_summary": {"unknowns": []}, "figure_specification": {"modules": [], "inputs": [], "outputs": [], "relations": [], "innovations": [], "must_show": [], "terminology": {}}}

        spec = normalize_figure_contract(plan, parsed)
        ids = {_item.get("name"): _item.get("id") for field in ("inputs", "modules", "outputs") for _item in spec[field]}
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        for label in ("Input Sequence", "Token Embeddings", "Segment Embeddings", "Position Embeddings", "Input Representation", "Bidirectional Transformer Encoder"):
            self.assertIn(label, ids)
        self.assertIn((ids["Token Embeddings"], ids["Input Representation"]), pairs)
        self.assertIn((ids["Segment Embeddings"], ids["Input Representation"]), pairs)
        self.assertIn((ids["Position Embeddings"], ids["Input Representation"]), pairs)

    def test_generic_completion_uses_selected_architecture_page_context_and_diagram_labels(self):
        captions = [
            "Figure 1: Direct set prediction combines a CNN with a transformer and bipartite matching.",
            "Figure 10: Architecture of the proposed transformer's subsystem. See the detailed description on this page.",
        ]
        parsed = {
            "page_count": 24,
            "document_index": {"figures": [{"page": 2, "caption": captions[0]}, {"page": 20, "caption": captions[1]}]},
            "evidence": [
                {"id": "E0001", "page": 2, "kind": "caption", "bbox": [40, 300, 500, 320], "text": captions[0], "section_hint": "Figure Captions", "confidence": 0.98},
                {"id": "E0002", "page": 4, "kind": "paragraph", "bbox": [40, 220, 500, 240], "text": "The input image is processed for direct prediction with bipartite matching.", "section_hint": "Object Detection Method", "confidence": 0.98},
                {"id": "E0003", "page": 20, "kind": "caption", "bbox": [40, 600, 500, 620], "text": captions[1], "section_hint": "Figure Captions", "confidence": 0.98},
                {"id": "E0004", "page": 20, "kind": "paragraph", "bbox": [40, 100, 500, 120], "text": "Image features from the CNN backbone are passed through the transformer encoder.", "section_hint": "Detailed Architecture", "confidence": 0.98},
                {"id": "E0005", "page": 20, "kind": "paragraph", "bbox": [40, 150, 500, 170], "text": "The decoder receives object queries and encoder memory.", "section_hint": "Detailed Architecture", "confidence": 0.98},
                {"id": "E0006", "page": 20, "kind": "paragraph", "bbox": [340, 280, 380, 300], "text": "FFN", "section_hint": "Detailed Architecture", "confidence": 0.98},
                {"id": "E0007", "page": 20, "kind": "paragraph", "bbox": [140, 390, 190, 410], "text": "Encoder", "section_hint": "Detailed Architecture", "confidence": 0.98},
                {"id": "E0008", "page": 20, "kind": "paragraph", "bbox": [310, 310, 360, 330], "text": "Add & Norm", "section_hint": "Detailed Architecture", "confidence": 0.98},
                {"id": "E0009", "page": 20, "kind": "paragraph", "bbox": [40, 180, 500, 200], "text": "The final class labels and bounding boxes are predicted in parallel.", "section_hint": "Detailed Architecture", "confidence": 0.98},
                {"id": "E0010", "page": 20, "kind": "paragraph", "bbox": [40, 710, 500, 730], "text": "For example, a separate text encoder uses class descriptions and contrastive learning to create text embeddings.", "section_hint": "Related Work", "confidence": 0.98},
            ],
        }
        plan = {"paper_summary": {"unknowns": []}, "figure_specification": {"modules": [], "inputs": [], "outputs": [], "relations": [], "innovations": [], "must_show": [], "terminology": {}}}

        spec = normalize_figure_contract(plan, parsed)
        ids = {_item.get("name"): _item.get("id") for field in ("inputs", "modules", "outputs") for _item in spec[field]}
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        for label in ("Input Image", "CNN Backbone", "Transformer Encoder", "Object Queries", "Transformer Decoder", "Feed Forward Network", "Class Predictions", "Bounding Box Predictions", "Bipartite Matching"):
            self.assertIn(label, ids)
        self.assertIn((ids["CNN Backbone"], ids["Transformer Encoder"]), pairs)
        self.assertIn((ids["Object Queries"], ids["Transformer Decoder"]), pairs)
        self.assertIn((ids["Feed Forward Network"], ids["Class Predictions"]), pairs)
        self.assertNotIn("Text Encoder", ids)
        self.assertNotIn("Contrastive Learning", ids)

    def test_dataset_sources_are_separated_from_modalities_when_caption_states_provenance(self):
        caption = "Figure 1: Stage one constructs RETFound by means of SSL, using CFP and OCT from MEH-MIDAS and public datasets. Stage two adapts RETFound by supervised fine-tuning for internal and external evaluation."
        parsed = {
            "page_count": 4,
            "document_index": {"figures": [{"page": 2, "caption": caption}]},
            "evidence": [
                {"id": "E0001", "page": 2, "kind": "caption", "text": caption, "section_hint": "Figure Captions", "confidence": 1.0},
                {"id": "E0002", "page": 3, "kind": "paragraph", "text": "Models were adapted with labelled training data and internally evaluated on hold-out test data from the MEH-AlzEye dataset.", "section_hint": "Evaluation", "confidence": 1.0},
            ],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "topology": "linear",
                "inputs": [
                    {"id": "cfp", "name": "CFP", "evidence_ids": ["E0001"]},
                    {"id": "oct", "name": "OCT", "evidence_ids": ["E0001"]},
                ],
                "modules": [
                    {"id": "ssl", "name": "SSL", "evidence_ids": ["E0001"]},
                    {"id": "supervised", "name": "supervised fine-tuning", "evidence_ids": ["E0001"]},
                ],
                "outputs": [],
                "innovations": [],
                "relations": [
                    {"source": "cfp", "target": "ssl", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "oct", "target": "ssl", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "ssl", "target": "supervised", "type": "data_flow", "evidence_ids": ["E0001"]},
                ],
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}
        by_name = {item["name"]: item for item in spec["inputs"]}
        source_id = by_name["MEH-MIDAS + public datasets"]["id"]

        self.assertEqual(by_name["MEH-MIDAS + public datasets"]["role"], "data_source")
        self.assertEqual(by_name["CFP"]["role"], "input modality")
        self.assertEqual(by_name["OCT"]["role"], "input modality")
        self.assertIn((source_id, "cfp"), pairs)
        self.assertIn((source_id, "oct"), pairs)
        self.assertNotIn((source_id, "ssl"), pairs)
        self.assertIn(source_id, plan["contract_completion_report"]["added_data_sources"])
        self.assertNotIn("MEH-MIDAS", by_name)
        self.assertNotIn("public datasets", by_name)
        self.assertNotIn("MEH-AlzEye dataset", by_name)
        output_names = {item["name"]: item["id"] for item in spec["outputs"]}
        self.assertIn(("supervised", output_names["internal evaluation"]), pairs)
        self.assertIn(("supervised", output_names["external evaluation"]), pairs)

    def test_fast_contract_cache_is_scoped_by_scientific_domain_not_visual_preferences(self):
        parsed = {"source_sha256": "abc123"}

        first = _fast_cache_path(parsed, "planner-model", "ai-ml-method")
        second = _fast_cache_path(parsed, "planner-model", "ai-ml-method")

        self.assertEqual(first, second)
        self.assertNotEqual(first, _fast_cache_path(parsed, "planner-model", "empirical-science"))

    def test_generic_foundation_model_label_is_replaced_by_explicit_paper_name(self):
        caption = "Figure 1: Schematic of development and evaluation of the foundation models (RETFound)."
        parsed = {
            "page_count": 4,
            "document_index": {"figures": [{"page": 2, "caption": caption}]},
            "evidence": [{"id": "E0001", "page": 2, "kind": "caption", "text": caption, "section_hint": "Figure Captions", "confidence": 1.0}],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "topology": "linear",
                "inputs": [],
                "modules": [{"id": "model", "name": "foundation model", "role": "module", "evidence_ids": ["E0001"]}],
                "outputs": [],
                "innovations": [],
                "relations": [],
                "terminology": {"RETFound": "foundation model"},
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        self.assertEqual(spec["modules"][0]["name"], "RETFound")
        self.assertEqual(spec["terminology"], {"RETFound": "RETFound"})
        self.assertIn("foundation model->RETFound", plan["contract_completion_report"]["repaired_generic_named_modules"])

    def test_explicit_named_ssl_technique_is_restored_as_required_innovation(self):
        parsed = {
            "page_count": 4,
            "document_index": {"figures": []},
            "evidence": [{"id": "E0001", "page": 2, "kind": "paragraph", "text": "We use an advanced SSL technique (masked autoencoder 15) on retinal images.", "section_hint": "Method", "confidence": 1.0}],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "topology": "linear",
                "inputs": [],
                "modules": [{"id": "ssl", "name": "SSL", "role": "training_objective", "evidence_ids": ["E0001"]}],
                "outputs": [],
                "innovations": [],
                "relations": [],
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        self.assertEqual(spec["innovations"][0]["name"], "masked autoencoder")
        self.assertTrue(spec["innovations"][0]["visible_label_required"])
        self.assertTrue(spec["innovations"][0]["must_appear_in_figure"])
        self.assertTrue(plan["contract_completion_report"]["added_explicit_named_techniques"])

    def test_parallel_leaf_heads_are_promoted_to_consistent_output_roles(self):
        caption = "The mask branch is in parallel with the classification and bounding-box regression heads."
        parsed = {
            "page_count": 3,
            "document_index": {"figures": [{"page": 1, "caption": caption}]},
            "evidence": [{"id": "E0001", "page": 1, "kind": "caption", "text": caption, "section_hint": "Figure Captions", "confidence": 1.0}],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "topology": "branch",
                "inputs": [],
                "modules": [
                    {"id": "roialign", "name": "RoIAlign", "role": "module", "evidence_ids": ["E0001"]},
                    {"id": "box", "name": "bounding-box regression", "role": "module", "evidence_ids": ["E0001"]},
                    {"id": "mask", "name": "mask branch", "role": "module", "evidence_ids": ["E0001"]},
                ],
                "outputs": [{"id": "classification", "name": "Classification", "role": "output", "evidence_ids": ["E0001"]}],
                "innovations": [{"id": "innovation_mask", "name": "mask branch", "evidence_ids": ["E0001"]}],
                "relations": [
                    {"source": "proposal", "target": "roialign", "type": "conditioning", "evidence_ids": ["E0001"]},
                    {"source": "roialign", "target": "classification", "type": "branch", "evidence_ids": ["E0001"]},
                    {"source": "roialign", "target": "box", "type": "branch", "evidence_ids": ["E0001"]},
                    {"source": "roialign", "target": "mask", "type": "branch", "evidence_ids": ["E0001"]},
                    {"source": "proposal", "target": "mask", "type": "data_flow", "evidence_ids": ["E0001"]},
                ],
            },
        }
        plan["figure_specification"]["modules"].insert(0, {"id": "proposal", "name": "Region Proposals", "role": "module", "evidence_ids": ["E0001"]})

        spec = normalize_figure_contract(plan, parsed)
        output_by_id = {item["id"]: item for item in spec["outputs"]}

        self.assertEqual(set(output_by_id), {"classification", "box", "mask"})
        self.assertNotIn("box", {item["id"] for item in spec["modules"]})
        self.assertNotIn("mask", {item["id"] for item in spec["modules"]})
        self.assertTrue(all(output_by_id[item_id]["role"] == "output head" for item_id in ("classification", "box", "mask")))
        self.assertEqual(set(plan["contract_completion_report"]["promoted_parallel_output_heads"]), {"box", "mask"})
        self.assertNotIn(("proposal", "mask"), {(item["source"], item["target"]) for item in spec["relations"]})
        self.assertIn("proposal->mask", plan["contract_completion_report"]["removed_branch_shortcuts"])

    def test_generic_completion_recovers_retrieval_conditioned_generation(self):
        caption = "Figure 1: Overview of our approach. We combine a pre-trained retriever (Query Encoder + Document Index) with a pre-trained seq2seq model (Generator). For query x, we use MIPS to find the top-K documents. For final prediction y, the generator produces the output sequence."
        result_caption = "Figure 2: Posterior for each generated token with five retrieved documents."
        parsed = {
            "page_count": 6,
            "document_index": {"figures": [{"page": 2, "caption": caption}, {"page": 5, "caption": result_caption}]},
            "evidence": [
                {"id": "E0001", "page": 2, "kind": "caption", "text": caption, "section_hint": "Figure Captions", "confidence": 1.0},
                {"id": "E0002", "page": 5, "kind": "caption", "text": result_caption, "section_hint": "Figure Captions", "confidence": 1.0},
            ],
        }
        plan = {"paper_summary": {"unknowns": []}, "figure_specification": {"modules": [], "inputs": [], "outputs": [], "relations": [], "innovations": [], "must_show": [], "terminology": {}}}

        spec = normalize_figure_contract(plan, parsed)
        ids = {_item.get("name"): _item.get("id") for field in ("inputs", "modules", "outputs") for _item in spec[field]}
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        for label in ("Input Query", "Query Encoder", "Document Index", "Retriever", "Top-K Documents", "Generator", "Output Sequence"):
            self.assertIn(label, ids)
        self.assertNotIn("Generate", ids)
        self.assertIn((ids["Document Index"], ids["Retriever"]), pairs)
        self.assertIn((ids["Top-K Documents"], ids["Generator"]), pairs)

    def test_rag_contract_repairs_latent_endpoint_alias_and_acronym_evidence(self):
        caption = "Figure 1: Overview of our approach. Query Encoder and Document Index use Maximum Inner Product Search (MIPS) to find the top-K documents z. The Generator conditions on those latent documents."
        parsed = {
            "page_count": 4,
            "document_index": {"figures": [{"page": 2, "caption": caption}]},
            "evidence": [{"id": "E0001", "page": 2, "kind": "caption", "text": caption, "section_hint": "Figure Captions", "confidence": 1.0}],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [],
                "modules": [
                    {"id": "query_encoder", "name": "Query Encoder", "evidence_ids": ["E0001"]},
                    {"id": "doc_index", "name": "Document Index", "evidence_ids": ["E0001"]},
                    {"id": "generator", "name": "Generator", "evidence_ids": ["E0001"]},
                    {"id": "mips", "name": "MIPS", "evidence_ids": []},
                ],
                "outputs": [],
                "innovations": [],
                "relations": [
                    {"source": "mips", "target": "latent_z", "type": "branch", "label": "Top-K Documents", "evidence_ids": []},
                    {"source": "latent_z", "target": "generator", "type": "conditioning", "label": "context", "evidence_ids": []},
                ],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        validation = validate_plan_grounding(plan, parsed)
        ids = {item["name"]: item["id"] for item in spec["modules"]}
        pairs = {(item["source"], item["target"]): item for item in spec["relations"]}

        self.assertTrue(next(item for item in spec["modules"] if item["name"] == "MIPS")["evidence_ids"])
        self.assertIn(("mips", ids["Top-K Documents"]), pairs)
        self.assertIn((ids["Top-K Documents"], "generator"), pairs)
        self.assertTrue(pairs[("mips", ids["Top-K Documents"])]["evidence_ids"])
        self.assertFalse(any("latent_z" in (item["source"], item["target"]) for item in spec["relations"]))
        self.assertTrue(validation["ok"])

    def test_rich_sam_contract_does_not_import_unrelated_clip_components(self):
        sam_caption = "Figure 1: The promptable segmentation model contains an image encoder, prompt encoder, and mask decoder that produces a valid segmentation mask."
        parsed = {
            "page_count": 8,
            "document_index": {"figures": [{"page": 2, "caption": sam_caption}]},
            "evidence": [
                {"id": "E0001", "page": 2, "kind": "caption", "text": sam_caption, "section_hint": "Figure Captions", "confidence": 1.0},
                {"id": "E0002", "page": 1, "kind": "paragraph", "text": "CLIP uses class descriptions, a text encoder, text embeddings, and contrastive learning.", "section_hint": "Introduction", "confidence": 1.0},
            ],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [{"id": "image", "name": "Image", "evidence_ids": ["E0001"]}],
                "modules": [
                    {"id": "image_encoder", "name": "Image Encoder", "evidence_ids": ["E0001"]},
                    {"id": "prompt_encoder", "name": "Prompt Encoder", "evidence_ids": ["E0001"]},
                    {"id": "mask_decoder", "name": "Mask Decoder", "evidence_ids": ["E0001"]},
                ],
                "outputs": [{"id": "mask", "name": "Valid Segmentation Mask", "evidence_ids": ["E0001"]}],
                "innovations": [],
                "relations": [
                    {"source": "image", "target": "image_encoder", "type": "data_flow", "evidence_ids": ["E0001"]},
                    {"source": "image_encoder", "target": "mask_decoder", "type": "feature_flow", "evidence_ids": ["E0001"]},
                    {"source": "prompt_encoder", "target": "mask_decoder", "type": "conditioning", "evidence_ids": ["E0001"]},
                    {"source": "mask_decoder", "target": "mask", "type": "prediction", "evidence_ids": ["E0001"]},
                ],
                "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        labels = {_item.get("name") for field in ("inputs", "modules", "outputs") for _item in spec[field]}

        self.assertNotIn("Class Descriptions", labels)
        self.assertNotIn("Text Encoder", labels)
        self.assertNotIn("Contrastive Learning", labels)
        self.assertTrue(plan["contract_completion_report"]["conservative_expansion"])

    def test_generic_completion_repairs_vlm_ids_labels_and_relation_grounding(self):
        caption = "Figure 1: Pre-training and fine-tuning procedures."
        parsed = {
            "page_count": 5,
            "document_index": {"figures": [{"page": 2, "caption": caption}]},
            "evidence": [
                {"id": "E0001", "page": 2, "kind": "caption", "text": caption, "section_hint": "Figure Captions", "confidence": 1.0},
                {"id": "E0002", "page": 3, "kind": "paragraph", "text": "The input embeddings are the sum of the token embeddings, segmentation embeddings and position embeddings.", "section_hint": "Method", "confidence": 1.0},
                {"id": "E0003", "page": 3, "kind": "paragraph", "text": "The architecture is a multi-layer bidirectional Transformer encoder.", "section_hint": "Method", "confidence": 1.0},
                {"id": "E0004", "page": 4, "kind": "paragraph", "text": "We train with a masked language model and next sentence prediction.", "section_hint": "Method", "confidence": 1.0},
                {"id": "E0005", "page": 4, "kind": "paragraph", "text": "We represent the input sequence for downstream tasks.", "section_hint": "Method", "confidence": 1.0},
                {"id": "E0006", "page": 1, "kind": "paragraph", "text": "A major limitation is that standard language models are", "section_hint": "Introduction", "confidence": 1.0},
                {"id": "E0007", "page": 1, "kind": "paragraph", "text": "unidirectional.", "section_hint": "Introduction", "confidence": 1.0},
            ],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "inputs": [{"id": "input_tokens", "name": "input example", "evidence_ids": ["E0005"]}],
                "modules": [
                    {"id": "embedding_layer", "name": "input embeddings", "evidence_ids": ["E0002"]},
                    {"id": "transformer_encoder", "name": "Bidirectional Transformer Encoder", "evidence_ids": ["E0003"]},
                    {"id": "mlm_head", "name": "MLM objective", "evidence_ids": []},
                ],
                "research_problem": {"text": "Standard language models are unidirectional.", "evidence_ids": []},
                "outputs": [], "innovations": [], "must_show": [], "terminology": {},
                "relations": [
                    {"source": "embedding_layer", "target": "transformer_encoder", "type": "data_flow", "evidence_ids": []},
                    {"source": "transformer_encoder", "target": "mlm_head", "type": "prediction", "evidence_ids": []},
                ],
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        self.assertEqual(next(item for item in spec["inputs"] if item["id"] == "input_tokens")["name"], "Input Sequence")
        self.assertEqual(next(item for item in spec["modules"] if item["id"] == "mlm_head")["name"], "Masked LM")
        self.assertTrue(next(item for item in spec["modules"] if item["id"] == "mlm_head")["evidence_ids"])
        self.assertTrue(all(item.get("evidence_ids") for item in spec["relations"] if item.get("source") in {"embedding_layer", "transformer_encoder"} and item.get("target") in {"transformer_encoder", "mlm_head"}))
        self.assertEqual(spec["research_problem"]["evidence_ids"], ["E0006", "E0007"])

    def test_rich_contract_expands_newly_grounded_intermediate_to_evidence_neighbors(self):
        caption = "Figure 1: Overall pre-training and fine-tuning procedures."
        parsed = {
            "page_count": 16,
            "document_index": {"figures": [{"page": 3, "caption": caption}]},
            "evidence": [
                {"id": "E0001", "page": 3, "kind": "caption", "text": caption, "section_hint": "Figure Captions", "confidence": 0.99},
                {"id": "E0002", "page": 4, "kind": "paragraph", "text": "For a given token, its input representation is", "section_hint": "Method", "confidence": 0.98},
                {"id": "E0003", "page": 4, "kind": "paragraph", "text": "constructed by summing the corresponding token,", "section_hint": "Method", "confidence": 0.98},
                {"id": "E0004", "page": 4, "kind": "paragraph", "text": "segment, and position embeddings.", "section_hint": "Method", "confidence": 0.98},
                {"id": "E0005", "page": 4, "kind": "paragraph", "text": "The architecture is a multi-layer bidirectional Transformer encoder.", "section_hint": "Method", "confidence": 0.98},
                {"id": "E0006", "page": 4, "kind": "paragraph", "text": "We train with Masked LM and Next Sentence Prediction before fine-tuning on downstream tasks.", "section_hint": "Method", "confidence": 0.98},
            ],
        }
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "inputs": [{"id": "sentence_pair", "name": "Sentence Pair", "evidence_ids": ["E0001"]}],
                "modules": [
                    {"id": "bert", "name": "Bidirectional Transformer Encoder", "evidence_ids": ["E0005"]},
                    {"id": "mlm", "name": "Masked LM", "evidence_ids": ["E0006"]},
                    {"id": "nsp", "name": "Next Sentence Prediction", "evidence_ids": ["E0006"]},
                ],
                "outputs": [{"id": "tasks", "name": "Downstream Tasks", "evidence_ids": ["E0006"]}],
                "relations": [{"source": "bert", "target": "mlm", "type": "training_objective", "evidence_ids": ["E0006"]}],
                "innovations": [], "must_show": [], "terminology": {},
            },
        }

        spec = normalize_figure_contract(plan, parsed)
        ids = {_item.get("name"): _item.get("id") for field in ("inputs", "modules", "outputs") for _item in spec[field]}
        pairs = {(item["source"], item["target"]) for item in spec["relations"]}

        for label in ("Input Sequence", "Token Embeddings", "Segment Embeddings", "Position Embeddings", "Input Representation"):
            self.assertIn(label, ids)
        self.assertIn((ids["Input Representation"], ids["Bidirectional Transformer Encoder"]), pairs)

    def test_missing_citations_are_recorded_instead_of_voiding_the_plan(self):
        """A plan that names the modules but not their sentences is thin, not broken.

        Every complaint here is about evidence, and reading them as fatal threw
        away a paper the extraction had already understood -- after paying for
        it.
        """

        parsed = {"evidence": [{"id": "E0001", "text": "The pipeline encodes the document.", "confidence": 1.0}]}
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "research_problem": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "central_claim": {"text": "unknown", "evidence_ids": [], "status": "unknown"},
                "inputs": [{"id": "in", "name": "Document", "evidence_ids": ["E0001"]}],
                "modules": [{"id": "encoder", "name": "Encoder", "evidence_ids": []}],
                "outputs": [{"id": "out", "name": "Vector", "evidence_ids": []}],
                "innovations": [],
                "relations": [{"source": "in", "target": "encoder", "type": "data_flow", "evidence_ids": []}],
                "terminology": {},
            },
        }

        validation = validate_plan_grounding(plan, parsed)

        self.assertTrue(validation["ok"], validation["errors"])
        self.assertEqual([], validation["errors"])
        self.assertEqual(3, validation["grounding_gap_count"])
        self.assertTrue(all("evidence_ids" in gap for gap in validation["grounding_gaps"]))

    def test_a_plan_without_modules_still_fails(self):
        parsed = {"evidence": [{"id": "E0001", "text": "text", "confidence": 1.0}]}
        plan = {"paper_summary": {"unknowns": []}, "figure_specification": {"modules": [], "inputs": [], "outputs": [], "relations": [], "terminology": {}}}
        validation = validate_plan_grounding(plan, parsed)
        self.assertFalse(validation["ok"])
        self.assertIn("figure_specification.modules is empty", validation["errors"])


    def test_a_relation_whose_endpoints_were_never_declared_is_dropped_and_named(self):
        parsed = {"evidence": [
            {"id": "E1", "page": 2, "text": "A prompt image is attended to by the model.", "confidence": 1.0},
            {"id": "E2", "page": 2, "text": "The encoder reads the image.", "confidence": 1.0},
        ]}
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "inputs": [{"id": "input_image", "name": "Input image", "evidence_ids": ["E1"]}],
                "modules": [{"id": "encoder", "name": "Image encoder", "evidence_ids": ["E2"]}],
                "outputs": [],
                "innovations": [],
                "relations": [
                    {"source": "input_image", "target": "encoder", "type": "data_flow", "evidence_ids": ["E1", "E2"]},
                    {"source": "source_attending_to_question_image_itself_respectively", "target": "input_image",
                     "type": "data_flow", "evidence_ids": ["E1"], "label": "attends"},
                ],
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        self.assertEqual(
            [(item["source"], item["target"]) for item in spec["relations"]],
            [("input_image", "encoder")],
        )
        self.assertEqual(
            plan["contract_completion_report"]["removed_unresolved_relations"],
            ["source_attending_to_question_image_itself_respectively->input_image"],
        )
        self.assertTrue(any("Dropped unresolved relation" in note for note in spec["uncertainties"]))
        validation = validate_plan_grounding(plan, parsed)
        self.assertTrue(validation["ok"], validation["errors"])

    def test_a_relation_the_planner_declared_is_never_dropped_by_the_final_sweep(self):
        parsed = {"evidence": [{"id": "E1", "page": 2, "text": "The encoder reads the image.", "confidence": 1.0}]}
        plan = {
            "paper_summary": {"unknowns": []},
            "figure_specification": {
                "inputs": [{"id": "in", "name": "Image", "evidence_ids": ["E1"]}],
                "modules": [{"id": "encoder", "name": "Image encoder", "evidence_ids": ["E1"]}],
                "outputs": [{"id": "out", "name": "Vector", "evidence_ids": ["E1"]}],
                "innovations": [],
                "relations": [
                    {"source": "in", "target": "encoder", "type": "data_flow", "evidence_ids": ["E1"]},
                    {"source": "encoder", "target": "out", "type": "data_flow", "evidence_ids": ["E1"]},
                ],
            },
        }

        spec = normalize_figure_contract(plan, parsed)

        self.assertEqual(len(spec["relations"]), 2)
        self.assertEqual(plan["contract_completion_report"].get("removed_unresolved_relations", []), [])

    def test_a_named_concept_reuses_the_module_that_already_has_its_id(self):
        from rfs.contracts.paper_completion import augment_contract_from_evidence

        caption = "Figure 1. Overview of the architecture used by the pre-trained retriever."
        parsed = {
            "page_count": 2,
            "document_index": {"figures": [{"page": 1, "caption": caption}]},
            "evidence": [
                {"id": "C1", "page": 1, "kind": "caption", "section_hint": "Figure Captions", "text": caption, "bbox": [40, 400, 500, 440]},
                {"id": "E1", "page": 1, "kind": "paragraph", "section_hint": "Method", "text": "The pre-trained retriever selects passages.", "bbox": [40, 80, 280, 120]},
            ],
        }
        spec = {
            "modules": [
                {"id": "module_retriever", "label": "Dual encoder", "role": "module", "evidence_ids": ["E1"]},
                {"id": "module_llm_generator", "label": "Generator", "role": "module", "evidence_ids": ["E1"]},
                {"id": "module_defense_mechanism", "label": "Defense", "role": "module", "evidence_ids": ["E1"]},
            ],
            "relations": [{"source": "module_retriever", "target": "module_llm_generator", "type": "data_flow", "evidence_ids": ["E1"]}],
            "inputs": [],
            "outputs": [],
            "innovations": [],
        }

        augment_contract_from_evidence(spec, parsed)
        retrievers = [item for item in spec["modules"] if item.get("id") == "module_retriever"]

        self.assertEqual(len(retrievers), 1)
        self.assertEqual(len({item.get("id") for item in spec["modules"]}), len(spec["modules"]))


class ReadModeTests(unittest.TestCase):
    """The read is one of the switches the pipeline takes, so it is checked like one."""

    def test_an_unknown_reading_mode_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "out"
            with self.assertRaises(ValueError) as caught:
                prepare_paper_figure_contract(Path(folder) / "paper.pdf", out, read_mode="lazy")
            self.assertIn("read_mode", str(caught.exception))
            self.assertFalse(out.exists())

    def test_the_full_read_is_given_more_wall_clock_than_the_fast_one(self):
        self.assertEqual(sorted(READ_DEADLINE_SECONDS), ["fast", "full"])
        self.assertLess(READ_DEADLINE_SECONDS["fast"], READ_DEADLINE_SECONDS["full"])


if __name__ == "__main__":
    unittest.main()
