import unittest

from rfs.planning.semantic_blueprint import compile_semantic_blueprint


class SemanticBlueprintTests(unittest.TestCase):
    def test_isolated_required_nodes_are_not_rendered_as_peer_graph_nodes(self):
        specification = {
            "topology": "linear",
            "required_labels": ["Input", "Encoder", "Output", "Local Detail"],
            "inputs": [{"id": "input", "name": "Input"}],
            "modules": [
                {"id": "encoder", "name": "Encoder"},
                {"id": "detail", "name": "Local Detail"},
            ],
            "outputs": [{"id": "output", "name": "Output"}],
            "relations": [
                {"source": "input", "target": "encoder", "type": "data_flow"},
                {"source": "encoder", "target": "output", "type": "data_flow"},
            ],
        }

        report = compile_semantic_blueprint(specification)

        self.assertTrue(report["applied"])
        self.assertEqual(report["isolated_node_ids"], ["detail"])
        self.assertNotIn("detail", {item["id"] for item in report["nodes"]})

    def test_linear_contract_uses_graph_ranks_and_keeps_parallel_outputs_in_one_layer(self):
        specification = {
            "topology": "linear",
            "required_labels": ["Input", "Embed", "Encoder", "Output A", "Output B"],
            "inputs": [{"id": "input", "name": "Input"}],
            "modules": [{"id": "embed", "name": "Embed"}, {"id": "encoder", "name": "Encoder"}],
            "outputs": [{"id": "a", "name": "Output A"}, {"id": "b", "name": "Output B"}],
            "relations": [
                {"source": "input", "target": "embed", "type": "data_flow"},
                {"source": "embed", "target": "encoder", "type": "data_flow"},
                {"source": "encoder", "target": "a", "type": "data_flow"},
                {"source": "encoder", "target": "b", "type": "data_flow"},
            ],
        }

        report = compile_semantic_blueprint(specification)
        nodes = {item["id"]: item for item in report["nodes"]}

        self.assertTrue(report["applied"])
        self.assertEqual(nodes["input"]["rank"], 0)
        self.assertLess(nodes["embed"]["rank"], nodes["encoder"]["rank"])
        self.assertEqual(nodes["a"]["rank"], nodes["b"]["rank"])
        self.assertNotEqual(nodes["a"]["bbox_percent"]["y"], nodes["b"]["bbox_percent"]["y"])
        output_edges = [item for item in report["connectors"] if item["source"] == "encoder"]
        self.assertNotEqual(output_edges[0]["path_percent"][0][1], output_edges[1]["path_percent"][0][1])

    def test_multimodal_inputs_share_one_layer_and_converge(self):
        specification = {
            "topology": "multimodal",
            "required_labels": ["Image", "Text", "Audio", "Encoders", "Joint Space", "Alignment"],
            "inputs": [{"id": "image", "name": "Image"}, {"id": "text", "name": "Text"}, {"id": "audio", "name": "Audio"}],
            "modules": [{"id": "encoders", "name": "Encoders"}, {"id": "joint", "name": "Joint Space", "role": "shared representation"}],
            "outputs": [{"id": "alignment", "name": "Alignment"}],
            "relations": [
                {"source": "image", "target": "encoders", "type": "encoding"},
                {"source": "text", "target": "encoders", "type": "encoding"},
                {"source": "audio", "target": "encoders", "type": "encoding"},
                {"source": "encoders", "target": "joint", "type": "alignment"},
                {"source": "joint", "target": "alignment", "type": "enables"},
            ],
        }

        report = compile_semantic_blueprint(specification)
        nodes = {item["id"]: item for item in report["nodes"]}

        self.assertTrue(report["applied"])
        self.assertEqual({nodes[node_id]["rank"] for node_id in ("image", "text", "audio")}, {0})
        self.assertEqual(len({nodes[node_id]["bbox_percent"]["x"] for node_id in ("image", "text", "audio")}), 1)
        self.assertGreater(nodes["joint"]["rank"], nodes["encoders"]["rank"])
        input_edges = [item for item in report["connectors"] if item["target"] == "encoders"]
        self.assertEqual(len({item["path_percent"][-1][1] for item in input_edges}), 3)
        self.assertEqual(len({item["path_percent"][1][0] for item in input_edges}), 3)

    def test_feedback_loop_is_routed_outside_the_forward_graph(self):
        specification = {
            "topology": "feedback",
            "required_labels": ["Sample", "Measure", "Update", "Estimate", "repeat"],
            "inputs": [{"id": "sample", "name": "Sample"}],
            "modules": [{"id": "measure", "name": "Measure"}, {"id": "update", "name": "Update"}],
            "outputs": [{"id": "estimate", "name": "Estimate"}],
            "relations": [
                {"source": "sample", "target": "measure", "type": "data_flow"},
                {"source": "measure", "target": "update", "type": "data_flow"},
                {"source": "update", "target": "estimate", "type": "data_flow"},
                {"source": "estimate", "target": "measure", "type": "feedback_loop", "label": "repeat"},
            ],
        }

        report = compile_semantic_blueprint(specification)
        loop = next(item for item in report["connectors"] if item["type"] == "feedback_loop")

        self.assertTrue(report["applied"])
        self.assertEqual(loop["route_style"], "outer_feedback")
        self.assertIn(loop["path_percent"][1][1], {0.05, 0.95})
        self.assertEqual(loop["label"], "repeat")

    def test_source_only_conditioning_nodes_are_placed_immediately_before_their_target(self):
        specification = {
            "topology": "linear",
            "required_labels": ["Input", "Patches", "Projection", "Class Token", "Position Embedding", "Encoder"],
            "inputs": [{"id": "input", "name": "Input"}],
            "modules": [
                {"id": "patches", "name": "Patches"},
                {"id": "projection", "name": "Projection"},
                {"id": "class", "name": "Class Token", "role": "conditioning"},
                {"id": "position", "name": "Position Embedding", "role": "conditioning"},
                {"id": "encoder", "name": "Encoder"},
            ],
            "relations": [
                {"source": "input", "target": "patches", "type": "data_flow"},
                {"source": "patches", "target": "projection", "type": "data_flow"},
                {"source": "projection", "target": "encoder", "type": "data_flow"},
                {"source": "class", "target": "encoder", "type": "conditioning"},
                {"source": "position", "target": "encoder", "type": "conditioning"},
            ],
        }

        report = compile_semantic_blueprint(specification)
        nodes = {item["id"]: item for item in report["nodes"]}

        self.assertEqual(nodes["class"]["rank"], nodes["encoder"]["rank"] - 1)
        self.assertEqual(nodes["position"]["rank"], nodes["encoder"]["rank"] - 1)
        self.assertEqual(nodes["projection"]["rank"], nodes["class"]["rank"])

    def test_required_label_whitelist_excludes_out_of_scope_entities(self):
        specification = {
            "topology": "linear",
            "required_labels": ["Input", "Core", "Output"],
            "inputs": [{"id": "input", "name": "Input"}],
            "modules": [{"id": "core", "name": "Core"}, {"id": "detail", "name": "Out-of-scope Detail"}],
            "outputs": [{"id": "output", "name": "Output"}],
            "innovations": [{"id": "core_innovation", "name": "Core"}],
            "relations": [
                {"source": "input", "target": "core", "type": "data_flow"},
                {"source": "core", "target": "output", "type": "data_flow"},
                {"source": "core", "target": "detail", "type": "data_flow"},
            ],
        }

        report = compile_semantic_blueprint(specification)

        self.assertEqual({item["label"] for item in report["nodes"]}, {"Input", "Core", "Output"})
        self.assertFalse(any(item["target"] == "detail" for item in report["connectors"]))

    def test_skip_layer_relation_uses_bypass_lane_instead_of_crossing_middle_node(self):
        specification = {
            "topology": "branch",
            "required_labels": ["Input", "Middle", "Merge"],
            "inputs": [{"id": "input", "name": "Input"}],
            "modules": [{"id": "middle", "name": "Middle"}, {"id": "merge", "name": "Merge"}],
            "relations": [
                {"source": "input", "target": "middle", "type": "data_flow"},
                {"source": "middle", "target": "merge", "type": "data_flow"},
                {"source": "input", "target": "merge", "type": "conditioning"},
            ],
        }

        report = compile_semantic_blueprint(specification)
        bypass = next(item for item in report["connectors"] if item["source"] == "input" and item["target"] == "merge")

        self.assertEqual(bypass["route_style"], "bypass_orthogonal")
        self.assertGreaterEqual(len(bypass["path_percent"]), 4)
        self.assertEqual(bypass["routing_metrics"]["node_obstacle_intersections"], 0)

    def test_global_router_separates_nearby_long_vertical_lanes(self):
        specification = {
            "topology": "linear",
            "required_labels": ["Primary Input", "Auxiliary Input", "Stage One", "Stage Two", "Merger", "Output"],
            "inputs": [
                {"id": "primary", "name": "Primary Input"},
                {"id": "auxiliary", "name": "Auxiliary Input"},
            ],
            "modules": [
                {"id": "stage_one", "name": "Stage One"},
                {"id": "stage_two", "name": "Stage Two"},
                {"id": "merger", "name": "Merger"},
            ],
            "outputs": [{"id": "output", "name": "Output"}],
            "relations": [
                {"source": "primary", "target": "stage_one", "type": "data_flow"},
                {"source": "stage_one", "target": "stage_two", "type": "data_flow"},
                {"source": "stage_two", "target": "merger", "type": "data_flow"},
                {"source": "auxiliary", "target": "merger", "type": "data_flow"},
                {"source": "merger", "target": "output", "type": "prediction"},
            ],
        }

        report = compile_semantic_blueprint(specification)
        primary = next(item for item in report["connectors"] if item["source"] == "primary")
        auxiliary = next(item for item in report["connectors"] if item["source"] == "auxiliary")

        self.assertTrue(report["route_quality"]["passed"])
        self.assertEqual(report["route_quality"]["shared_segment_overlap"], 0.0)
        self.assertNotEqual(primary["lane_assignment"], auxiliary["lane_assignment"])
        self.assertIn("outer_", auxiliary["lane_assignment"])

    def test_fan_in_relations_use_distinct_target_ports(self):
        specification = {
            "topology": "multimodal",
            "required_labels": ["Input A", "Input B", "Input C", "Fusion", "Output"],
            "inputs": [
                {"id": "a", "name": "Input A"},
                {"id": "b", "name": "Input B"},
                {"id": "c", "name": "Input C"},
            ],
            "modules": [{"id": "fusion", "name": "Fusion"}],
            "outputs": [{"id": "output", "name": "Output"}],
            "relations": [
                {"source": "a", "target": "fusion", "type": "data_flow"},
                {"source": "b", "target": "fusion", "type": "data_flow"},
                {"source": "c", "target": "fusion", "type": "data_flow"},
                {"source": "fusion", "target": "output", "type": "prediction"},
            ],
        }

        report = compile_semantic_blueprint(specification)
        incoming = [item for item in report["connectors"] if item["target"] == "fusion"]

        self.assertEqual(len({tuple(item["path_percent"][-1]) for item in incoming}), 3)
        self.assertEqual(report["route_quality"]["node_obstacle_intersections"], 0)
        self.assertEqual(report["route_quality"]["crossing_count"], 0)

    def test_contract_larger_than_blueprint_limit_is_not_forced_into_one_canvas(self):
        modules = [{"id": f"node_{index}", "name": f"Node {index}"} for index in range(18)]
        specification = {
            "topology": "linear",
            "required_labels": [item["name"] for item in modules],
            "modules": modules,
            "relations": [{"source": f"node_{index}", "target": f"node_{index + 1}", "type": "data_flow"} for index in range(17)],
        }

        report = compile_semantic_blueprint(specification)

        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "too_many_nodes")

    def test_two_panel_dense_overview_uses_panel_local_layout(self):
        specification = {
            "topology": "dense_multiframe",
            "required_labels": ["Input", "Model", "Prediction"],
            "inputs": [{"id": "input", "name": "Input", "evidence_ids": ["E_A"]}],
            "modules": [{"id": "model", "name": "Model", "evidence_ids": ["E_B"]}],
            "outputs": [{"id": "output", "name": "Prediction", "evidence_ids": ["E_A"]}],
            "relations": [
                {"source": "input", "target": "model", "type": "data_flow"},
                {"source": "model", "target": "output", "type": "prediction"},
            ],
            "overview_panels": [
                {"id": "panel_a", "panel_label": "a", "label": "Training and usage", "evidence_ids": ["E_A"], "entity_ids": ["input", "output"]},
                {"id": "panel_b", "panel_label": "b", "label": "Architecture", "evidence_ids": ["E_B"], "entity_ids": ["model"]},
            ],
        }

        report = compile_semantic_blueprint(specification)
        nodes = {item["id"]: item for item in report["nodes"]}

        self.assertEqual(len(report["panels"]), 2)
        self.assertEqual(nodes["input"]["panel_id"], "panel_a")
        self.assertEqual(nodes["model"]["panel_id"], "panel_b")
        forward = next(item for item in report["connectors"] if item["source"] == "input")
        self.assertNotEqual(forward["route_style"], "outer_feedback")

    def test_panel_instance_graph_preserves_repeated_labels_and_local_connectors(self):
        specification = {
            "topology": "dense_multiframe",
            "panel_graphs": [
                {
                    "id": "panel_a",
                    "panel_label": "a",
                    "label": "Training and usage",
                    "nodes": [
                        {"instance_id": "a_data", "name": "Dataset", "role": "input", "evidence_ids": ["E1"]},
                        {"instance_id": "a_model_train", "name": "Model", "role": "module", "evidence_ids": ["E2"]},
                        {"instance_id": "a_prediction", "name": "Prediction", "role": "output", "evidence_ids": ["E3"]},
                        {"instance_id": "a_model_use", "name": "Model", "role": "module", "evidence_ids": ["E4"]},
                    ],
                    "relations": [
                        {"source": "a_data", "target": "a_model_train", "type": "data_flow"},
                        {"source": "a_model_train", "target": "a_prediction", "type": "prediction"},
                    ],
                },
                {
                    "id": "panel_b",
                    "panel_label": "b",
                    "label": "Architecture",
                    "nodes": [
                        {"instance_id": "b_model", "name": "Model", "role": "group", "evidence_ids": ["E5"]},
                        {"instance_id": "b_attention", "name": "Attention", "role": "module", "evidence_ids": ["E6"]},
                        {"instance_id": "b_output", "name": "Output", "role": "output", "evidence_ids": ["E7"]},
                    ],
                    "relations": [
                        {"source": "b_attention", "target": "b_output", "type": "prediction"},
                    ],
                },
            ],
        }

        report = compile_semantic_blueprint(specification)

        self.assertTrue(report["applied"])
        self.assertEqual(report["source"], "panel_graphs")
        self.assertEqual(report["node_count"], 6)
        self.assertEqual(sum(item["label"] == "Model" for item in report["nodes"]), 2)
        self.assertEqual(report["panels"][1]["groups"][0]["label"], "Model")
        self.assertTrue(all(item["panel_id"] == "panel_a" for item in report["connectors"][:2]))
        self.assertEqual(report["connectors"][2]["panel_id"], "panel_b")
        self.assertIn("a_model_use", report["isolated_node_ids"])


if __name__ == "__main__":
    unittest.main()
