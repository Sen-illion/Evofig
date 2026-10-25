"""Tests for the stable reference PPT layout-facts contract."""

from __future__ import annotations

import unittest

from rfs.contracts import validate_reference_ppt_layout_facts


def _valid_facts() -> dict:
    return {
        "source": {"path": "example.pptx", "sha256": "a" * 64, "total_slide_count": 3},
        "canvas": {"width_inches": 20, "height_inches": 11.25, "aspect_ratio": 16 / 9},
        "slides": [{
            "slide_index": 1,
            "evidence_mode": "editable_objects",
            "trusted_for_layout_geometry": True,
            "root_shape_count": 1,
            "recursive_object_count": 3,
            "maximum_group_depth": 1,
            "object_counts": {"group": 1, "text_box": 2},
            "editable_object_ratio": 1.0,
            "root_region_occupancy": 0.8,
            "root_regions": [{
                "shape_id": 2,
                "name": "Group 1",
                "kind": "group",
                "bbox_percent": {"x": 0.05, "y": 0.1, "w": 0.9, "h": 0.8},
                "text_preview": "",
                "descendant_counts": {"text_box": 2},
            }],
            "native_editable_object_count": 3,
            "picture_object_count": 0,
            "picture_assets": [],
            "typography": {"explicit_run_count": 2, "minimum_pt": 14, "median_pt": 18, "maximum_pt": 28},
            "object_color_frequency": [{"hex": "#263D50", "object_occurrences": 2}],
            "learning_policy": "human_review_required",
        }],
        "safety": {
            "scientific_semantics_learned": False,
            "exact_text_reusable": False,
            "raster_only_geometry_reusable": False,
        },
    }


class ReferencePptLayoutContractTests(unittest.TestCase):
    def test_accepts_auditable_editable_slide_facts(self) -> None:
        result = validate_reference_ppt_layout_facts(_valid_facts())
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["facts"]["contract"], "rfs.reference_ppt_layout_facts")

    def test_rejects_raster_geometry_reuse(self) -> None:
        facts = _valid_facts()
        slide = facts["slides"][0]
        slide["evidence_mode"] = "raster_only"
        result = validate_reference_ppt_layout_facts(facts)
        self.assertFalse(result["ok"])
        self.assertTrue(any("raster_only" in error for error in result["errors"]))

    def test_rejects_semantic_copy_authority(self) -> None:
        facts = _valid_facts()
        facts["safety"]["exact_text_reusable"] = True
        result = validate_reference_ppt_layout_facts(facts)
        self.assertFalse(result["ok"])

    def test_rejects_unproven_picture_provenance(self) -> None:
        facts = _valid_facts()
        slide = facts["slides"][0]
        slide["native_editable_object_count"] = 2
        slide["picture_object_count"] = 1
        slide["picture_assets"] = [{
            "shape_id": 9,
            "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
            "image_sha256": "unknown",
            "pixel_width": 0,
            "pixel_height": 0,
        }]
        result = validate_reference_ppt_layout_facts(facts)
        self.assertFalse(result["ok"])
        self.assertTrue(any("image_sha256" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
