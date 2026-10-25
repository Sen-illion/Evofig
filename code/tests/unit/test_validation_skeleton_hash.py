"""P1: the pair audit compares the figure's skeleton, not the bytes two runs happened to write."""

import unittest

from rfs.evolution.direct import skeleton_state_hash


def layout(**overrides):
    base = {
        "panels": [{"id": "panel_one", "title": "Phase One", "bbox_percent": {"x": 0.05, "y": 0.1, "w": 0.9, "h": 0.8}}],
        "nodes": [
            {"id": "node_a", "label": "Alpha", "panel_id": "panel_one", "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.2}},
            {"id": "node_b", "label": "Beta", "panel_id": "panel_one", "bbox_percent": {"x": 0.4, "y": 0.2, "w": 0.2, "h": 0.2}},
        ],
        "connectors": [{"id": "rel_1", "source": "node_a", "target": "node_b"}],
    }
    base.update(overrides)
    return base


def state(**overrides):
    return {"layout": layout(**overrides)}


class SkeletonStateHashTests(unittest.TestCase):
    def test_a_repainted_figure_hashes_the_same(self):
        """The palette, the font and the label are what each stage plans for itself.

        A hash that counted them would refuse every honest pair, because two runs of one paper
        never write the same colours: that was the whole failure P1 exists to remove.
        """

        first = state()
        first["layout"]["palette"] = {"primary": "#A9C7C7", "surface": "#758690"}
        first["layout"]["font_family"] = "Source Serif"
        second = state()
        second["layout"]["palette"] = {"primary": "#48BB78", "surface": "#D69E2E"}
        second["layout"]["font_family"] = "Inter"
        second["layout"]["nodes"][0]["label"] = "Card A"
        second["layout"]["panels"][0]["title"] = "Phase 1"

        self.assertEqual(skeleton_state_hash(first), skeleton_state_hash(second))

    def test_ids_a_rebuild_rewrites_do_not_decide_the_pair(self):
        renamed = state()
        renamed["layout"]["panels"][0]["id"] = "panel:reporter_assay_stage"
        renamed["layout"]["nodes"][0]["id"] = "node_alpha"
        renamed["layout"]["nodes"][1]["id"] = "node_beta"
        renamed["layout"]["nodes"][0]["panel_id"] = "panel:reporter_assay_stage"
        renamed["layout"]["nodes"][1]["panel_id"] = "panel:reporter_assay_stage"
        renamed["layout"]["connectors"] = [{"id": "rel_99", "source": "node_alpha", "target": "node_beta"}]

        self.assertEqual(skeleton_state_hash(state()), skeleton_state_hash(renamed))

    def test_a_different_structure_hashes_differently(self):
        moved = state()
        moved["layout"]["nodes"][1]["bbox_percent"] = {"x": 0.7, "y": 0.2, "w": 0.2, "h": 0.2}

        added = state()
        added["layout"]["nodes"].append({"id": "node_c", "panel_id": "panel_one", "bbox_percent": {"x": 0.1, "y": 0.6, "w": 0.2, "h": 0.2}})

        reconnected = state()
        reconnected["layout"]["connectors"] = [{"id": "rel_1", "source": "node_b", "target": "node_a"}]

        removed = state()
        removed["layout"]["panels"] = []

        baseline = skeleton_state_hash(state())
        for other in (moved, added, reconnected, removed):
            self.assertNotEqual(baseline, skeleton_state_hash(other))

    def test_a_sub_percent_nudge_is_not_a_structural_change(self):
        """Geometry is quantized to one percent of the canvas, so a hair of drift is the same box."""

        nudged = state()
        nudged["layout"]["nodes"][1]["bbox_percent"] = {"x": 0.4, "y": 0.2, "w": 0.2, "h": 0.2001}

        self.assertEqual(skeleton_state_hash(state()), skeleton_state_hash(nudged))

    def test_an_empty_figure_reads_as_unmeasured(self):
        self.assertEqual("", skeleton_state_hash({}))
        self.assertEqual("", skeleton_state_hash({"layout": {}}))
        self.assertEqual("", skeleton_state_hash(None))


if __name__ == "__main__":
    unittest.main()
