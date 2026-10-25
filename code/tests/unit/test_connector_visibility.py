import unittest

from rfs.evaluation.connector_visibility import evaluate_connector_visibility


class ConnectorVisibilityTests(unittest.TestCase):
    def test_reports_unrelated_node_intersection(self):
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .05, "y": .1, "w": .1, "h": .1}},
                {"id": "b", "bbox_percent": {"x": .8, "y": .1, "w": .1, "h": .1}},
                {"id": "middle", "bbox_percent": {"x": .4, "y": .08, "w": .2, "h": .14}},
            ],
            "arrows": [{"id": "e", "source": "a", "target": "b", "path_percent": [[.15, .15], [.85, .15]]}],
        }
        report = evaluate_connector_visibility(program)
        self.assertEqual(report["status"], "needs_reroute")
        self.assertEqual(report["issues"][0]["requested_action"], "reroute")

    def test_clear_path_passes(self):
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .05, "y": .1, "w": .1, "h": .1}},
                {"id": "b", "bbox_percent": {"x": .8, "y": .1, "w": .1, "h": .1}},
            ],
            "arrows": [{"id": "e", "source": "a", "target": "b", "path_percent": [[.15, .15], [.85, .15]]}],
        }
        self.assertEqual(evaluate_connector_visibility(program)["status"], "pass")

    def test_a_card_holding_both_endpoints_is_not_an_obstacle(self):
        """Nested submodules: the parent box is where its own links live."""
        program = {
            "semantic_nodes": [
                {"id": "group", "bbox_percent": {"x": .1, "y": .1, "w": .5, "h": .7}},
                {"id": "a", "parent_id": "group", "bbox_percent": {"x": .15, "y": .2, "w": .1, "h": .1}},
                {"id": "b", "parent_id": "group", "bbox_percent": {"x": .15, "y": .6, "w": .1, "h": .1}},
            ],
            "arrows": [{"id": "e", "source": "a", "target": "b", "path_percent": [[.3, .25], [.3, .6]]}],
        }
        self.assertEqual(evaluate_connector_visibility(program)["status"], "pass")

    def test_entering_a_container_to_reach_its_child_is_not_an_obstacle(self):
        program = {
            "semantic_nodes": [
                {"id": "outside", "bbox_percent": {"x": .8, "y": .4, "w": .1, "h": .1}},
                {"id": "group", "bbox_percent": {"x": .1, "y": .1, "w": .5, "h": .7}},
                {"id": "child", "parent_id": "group", "bbox_percent": {"x": .15, "y": .4, "w": .1, "h": .1}},
            ],
            "arrows": [{"id": "e", "source": "outside", "target": "child", "path_percent": [[.8, .45], [.2, .45]]}],
        }
        self.assertEqual(evaluate_connector_visibility(program)["status"], "pass")

    def test_a_sibling_container_in_the_way_is_still_reported(self):
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .05, "y": .45, "w": .1, "h": .1}},
                {"id": "b", "bbox_percent": {"x": .8, "y": .45, "w": .1, "h": .1}},
                {"id": "sibling", "bbox_percent": {"x": .4, "y": .4, "w": .2, "h": .2}},
                {"id": "kept", "parent_id": "sibling", "bbox_percent": {"x": .45, "y": .45, "w": .1, "h": .1}},
            ],
            "arrows": [{"id": "e", "source": "a", "target": "b", "path_percent": [[.15, .5], [.85, .5]]}],
        }
        report = evaluate_connector_visibility(program)
        self.assertEqual(report["status"], "needs_reroute")
        # The sibling container and the card it holds are both in the way.
        self.assertEqual(["sibling", "kept"], report["issues"][0]["covered_by_object_ids"])

    def test_a_link_across_its_own_icon_slot_is_still_reported(self):
        program = {
            "semantic_nodes": [
                {"id": "card", "bbox_percent": {"x": .1, "y": .5, "w": .3, "h": .15}},
                {"id": "above", "bbox_percent": {"x": .1, "y": .1, "w": .3, "h": .15}},
            ],
            "slots": [{"id": "slot_card", "parent_id": "card", "bbox_percent": {"x": .2, "y": .52, "w": .1, "h": .1}}],
            "arrows": [{"id": "e", "source": "card", "target": "above", "path_percent": [[.25, .65], [.25, .1]]}],
        }
        report = evaluate_connector_visibility(program)
        self.assertEqual(report["status"], "needs_reroute")
        self.assertEqual(["slot_card"], report["issues"][0]["covered_by_object_ids"])
