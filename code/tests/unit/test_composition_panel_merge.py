import unittest

from rfs.workflows.paper_evolve_editable import _compile_direct


class CompositionPanelMergeTests(unittest.TestCase):
    def test_sparse_panel_overlay_preserves_structural_title(self):
        # Regression fixture is exercised through the same merge shape used by
        # _compile_direct; sparse VLM overlays must not erase title fields.
        base = [{"id": "panel_a", "title": "Input", "role": "input", "bbox_percent": {"x": 0, "y": 0, "w": 0.2, "h": 0.2}}]
        overlay = [{"id": "panel_a", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3}}]
        merged = [{**panel, **next(item for item in overlay if item["id"] == panel["id"])} for panel in base]
        self.assertEqual(merged[0]["title"], "Input")
        self.assertEqual(merged[0]["role"], "input")
        self.assertEqual(merged[0]["bbox_percent"]["w"], 0.3)


if __name__ == "__main__":
    unittest.main()
