"""Unit tests for the Pilot visual topology reader."""

import json
from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import time
import unittest
from pathlib import Path

import pymupdf
from PIL import Image

from rfs.judge_training.pilot_visual import (
    DEFAULT_FIGURE_DPI,
    PDFIUM_LOCK,
    build_visual_prompt,
    derive_structure,
    load_figure_assets,
    normalize_visual_payload,
    render_figure_image,
    resolve_figure_asset,
)


def write_snapshot(root, *, paper_id="p0001", figure_name="figure.png"):
    """Write a tiny snapshot carrying one figure asset."""
    snapshot = root / "paper-d0-v001"
    directory = snapshot / "assets" / ("paper-" + paper_id) / "figures"
    directory.mkdir(parents=True)
    Image.new("RGB", (8, 4), (10, 20, 30)).save(directory / figure_name)
    relative = "assets/paper-" + paper_id + "/figures/" + figure_name
    with (snapshot / "asset_manifest.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "paper_id": paper_id,
            "asset_id": "asset-1",
            "relative_path": relative,
            "sha256": "0" * 64,
            "suffix": Path(figure_name).suffix,
            "role": "figure",
        }, ensure_ascii=False) + chr(10))
        handle.write(json.dumps({
            "paper_id": paper_id,
            "asset_id": "asset-2",
            "relative_path": "assets/paper-" + paper_id + "/tables/table.png",
            "sha256": "1" * 64,
            "suffix": ".png",
            "role": "table",
        }, ensure_ascii=False) + chr(10))
    return snapshot, relative


def write_pdf(path):
    """Write a tiny one-page PDF figure fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    page = document.new_page(width=72, height=36)
    page.insert_text((8, 24), "framework")
    document.save(str(path))
    document.close()


def two_block_payload(**overrides):
    payload = {
        "summary": "A chain of two modules.",
        "reading_direction": "left_to_right",
        "panels": [{"id": "p1", "label": "pipeline", "box": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}],
        "blocks": [
            {"id": "b1", "label": "Encoder", "panel_id": "p1", "box": {"x": 0.05, "y": 0.1, "w": 0.2, "h": 0.2}, "text_confidence": 0.9, "visual_role": "module"},
            {"id": "b2", "label": "Decoder", "panel_id": "p1", "box": {"x": 0.4, "y": 0.1, "w": 0.2, "h": 0.2}, "text_confidence": 0.9, "visual_role": "module"},
        ],
        "connectors": [{"id": "e1", "source_id": "b1", "target_id": "b2", "relation_type": "forward", "label": "feeds"}],
        "uncertainties": [],
    }
    payload.update(overrides)
    return payload


class FlowReadTests(unittest.TestCase):
    """A figure that draws no connector expresses no module, step or data flow."""

    def normalize(self, payload):
        return normalize_visual_payload(payload, paper_id="p0001")

    def test_a_read_with_connectors_expresses_flow(self):
        _graph, report = self.normalize(two_block_payload())

        self.assertTrue(report["expresses_flow"])
        self.assertEqual(report["warnings"], [])

    def test_a_read_without_connectors_is_reported_not_dropped(self):
        _graph, report = self.normalize(two_block_payload(connectors=[]))

        self.assertFalse(report["expresses_flow"])
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["block_count"], 2)
        self.assertIn("no connector", " ".join(report["warnings"]))

    def test_a_connector_the_reader_could_not_anchor_does_not_count_as_flow(self):
        payload = two_block_payload(connectors=[{"id": "e1", "source_id": "b1", "target_id": "", "relation_type": "forward"}])

        _graph, report = self.normalize(payload)

        self.assertFalse(report["expresses_flow"])
        self.assertEqual([item["reason"] for item in report["removed_connectors"]], ["undeclared_endpoint"])


class FigureAssetTests(unittest.TestCase):
    def test_load_figure_assets_keeps_only_figure_roles(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, relative = write_snapshot(Path(name))
            assets = load_figure_assets(snapshot)

        self.assertEqual(list(assets), ["p0001"])
        self.assertEqual([row["relative_path"] for row in assets["p0001"]], [relative])

    def test_resolve_prefers_the_pilot_ledger_path(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, relative = write_snapshot(Path(name))
            asset = resolve_figure_asset(snapshot, {"paper_id": "p0001", "figure_relative_path": relative, "figure_asset_id": "asset-1"})

        self.assertEqual(asset["relative_path"], relative)
        self.assertEqual(asset["source"], "pilot-ledger")

    def test_resolve_falls_back_to_the_asset_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, relative = write_snapshot(Path(name))
            by_id = resolve_figure_asset(snapshot, {"paper_id": "p0001", "figure_asset_id": "asset-1"})
            first = resolve_figure_asset(snapshot, {"paper_id": "p0001"})
            missing = resolve_figure_asset(snapshot, {"paper_id": "p0002"})

        self.assertEqual(by_id["relative_path"], relative)
        self.assertEqual(by_id["source"], "asset-manifest-by-id")
        self.assertEqual(first["source"], "asset-manifest-first")
        self.assertIsNone(missing)

    def test_resolve_prefers_a_ledger_override_file_outside_the_snapshot(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, relative = write_snapshot(root)
            override = root / "figure_overrides" / "p0001.png"
            override.parent.mkdir(parents=True)
            Image.new("RGB", (10, 6), (30, 40, 50)).save(override)
            asset = resolve_figure_asset(
                snapshot,
                {"paper_id": "p0001", "figure_relative_path": relative, "figure_override_path": str(override)},
            )
            joined = Path(snapshot) / asset["relative_path"]

        self.assertEqual(asset["source"], "ledger-override")
        self.assertEqual(asset["relative_path"], str(override))
        self.assertEqual(joined, override)

    def test_a_missing_override_falls_back_to_the_snapshot_candidates(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, relative = write_snapshot(root)
            asset = resolve_figure_asset(
                snapshot,
                {"paper_id": "p0001", "figure_relative_path": relative, "figure_override_path": str(root / "absent.png")},
            )

        self.assertEqual(asset["source"], "pilot-ledger")
        self.assertEqual(asset["relative_path"], relative)


class RenderTests(unittest.TestCase):
    def test_render_records_provenance_and_scales_the_long_side(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, relative = write_snapshot(root)
            target = root / "figures" / "figure.png"
            provenance = render_figure_image(snapshot / relative, target, dpi=72, max_long_side=4)

            self.assertTrue(target.is_file())
            self.assertEqual((provenance["source_width"], provenance["source_height"]), (8, 4))
            self.assertEqual((provenance["width"], provenance["height"]), (4, 2))
            self.assertTrue(provenance["scaled"])
            self.assertFalse(provenance["reused"])
            self.assertEqual(provenance["dpi"], 72)
            self.assertEqual(provenance["reader"].split("=")[0], "pillow")
            self.assertTrue(provenance["png_sha256"])
            self.assertTrue(target.with_name("figure.png.render.json").is_file())

    def test_render_reuses_only_matching_settings(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, relative = write_snapshot(root)
            target = root / "figures" / "figure.png"
            first = render_figure_image(snapshot / relative, target, dpi=72, max_long_side=64)
            second = render_figure_image(snapshot / relative, target, dpi=72, max_long_side=64)
            third = render_figure_image(snapshot / relative, target, dpi=144, max_long_side=64)

        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])
        self.assertEqual(second["png_sha256"], first["png_sha256"])
        self.assertFalse(third["reused"])
        self.assertEqual(third["dpi"], 144)

    def test_render_rejects_an_unsupported_dpi(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, relative = write_snapshot(root)
            with self.assertRaises(ValueError):
                render_figure_image(snapshot / relative, root / "figure.png", dpi=600)

    def test_render_reports_a_missing_asset(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(FileNotFoundError):
                render_figure_image(Path(name) / "absent.png", Path(name) / "out.png")


class NormalizeTests(unittest.TestCase):
    def normalize(self, payload, **kwargs):
        return normalize_visual_payload(payload, paper_id="p0001", **kwargs)

    def test_boxes_are_clamped_and_reported(self):
        payload = two_block_payload()
        payload["blocks"][0]["box"] = {"x": 0.9, "y": 0.95, "w": 0.4, "h": 0.4}
        graph, report = self.normalize(payload, image={"width": 100, "height": 50})

        self.assertIsNotNone(graph)
        box = graph["nodes"][1]["properties"]["box"]
        self.assertLessEqual(box["x"] + box["w"], 1.0)
        self.assertLessEqual(box["y"] + box["h"], 1.0)
        self.assertEqual(report["clamped_boxes"], ["block:b1"])
        self.assertTrue(any("clamped" in warning for warning in report["warnings"]))
        self.assertEqual(report["image"]["width"], 100)

    def test_connectors_with_undeclared_endpoints_are_dropped(self):
        payload = two_block_payload(connectors=[
            {"id": "e1", "source_id": "b1", "target_id": "b2", "relation_type": "forward"},
            {"id": "e2", "source_id": "b1", "target_id": "b9", "relation_type": "forward"},
        ])
        graph, report = self.normalize(payload)

        self.assertEqual(len(graph["edges"]), 1)
        self.assertEqual([row["id"] for row in report["removed_connectors"]], ["e2"])
        self.assertEqual(report["removed_connectors"][0]["reason"], "undeclared_endpoint")
        self.assertTrue(any("undeclared" in warning for warning in report["warnings"]))

    def test_duplicate_block_ids_are_rejected(self):
        payload = two_block_payload()
        payload["blocks"].append(dict(payload["blocks"][0]))
        graph, report = self.normalize(payload)

        self.assertIsNone(graph)
        self.assertFalse(report["ok"])
        self.assertTrue(any("duplicate block id" in error for error in report["errors"]))

    def test_a_payload_without_blocks_is_rejected(self):
        graph, report = self.normalize({"panels": [], "blocks": [], "connectors": []})

        self.assertIsNone(graph)
        self.assertTrue(any("no readable block" in error for error in report["errors"]))

    def test_reading_order_follows_panels_then_rows(self):
        payload = two_block_payload()
        payload["panels"] = [
            {"id": "p1", "label": "left", "box": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0}},
            {"id": "p2", "label": "right", "box": {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0}},
        ]
        payload["blocks"] = [
            {"id": "b3", "label": "third", "panel_id": "p1", "box": {"x": 0.05, "y": 0.6, "w": 0.2, "h": 0.1}},
            {"id": "b1", "label": "first", "panel_id": "p1", "box": {"x": 0.05, "y": 0.1, "w": 0.2, "h": 0.1}},
            {"id": "b4", "label": "fourth", "panel_id": "p2", "box": {"x": 0.55, "y": 0.1, "w": 0.2, "h": 0.1}},
            {"id": "b2", "label": "second", "panel_id": "p1", "box": {"x": 0.3, "y": 0.12, "w": 0.15, "h": 0.1}},
        ]
        graph, report = self.normalize(payload)

        self.assertEqual(report["reading_order"], ["b1", "b2", "b3", "b4"])
        self.assertEqual(graph["provenance"]["reading_order"], ["b1", "b2", "b3", "b4"])
        reading = {node["properties"]["source_id"]: node["properties"]["reading_index"] for node in graph["nodes"] if node["kind"] == "module"}
        self.assertEqual(reading, {"b1": 0, "b2": 1, "b3": 2, "b4": 3})

    def test_feedback_and_branching_signals_are_detected(self):
        payload = two_block_payload(connectors=[
            {"id": "e1", "source_id": "b1", "target_id": "b2", "relation_type": "forward"},
            {"id": "e2", "source_id": "b1", "target_id": "b3", "relation_type": "branch"},
            {"id": "e3", "source_id": "b3", "target_id": "b1", "relation_type": "feedback"},
        ])
        payload["blocks"].append(
            {"id": "b3", "label": "Memory", "panel_id": "p1", "box": {"x": 0.7, "y": 0.1, "w": 0.2, "h": 0.2}, "text_confidence": 0.6, "visual_role": "module"}
        )
        graph, report = self.normalize(payload)

        self.assertEqual(report["primary_structure"], "feedback_loop")
        self.assertIn("branching", report["structure_signals"])
        self.assertIn("feedback_loop", report["structure_signals"])
        self.assertEqual(len(report["feedback_edges"]), 1)
        self.assertEqual(report["connectors_not_marked_feedback"], [])
        self.assertEqual(report["branching_blocks"], ["b1"])
        self.assertEqual(report["merging_blocks"], [])
        motifs = {node["properties"]["source_id"]: node["properties"]["motif"] for node in graph["nodes"] if node["kind"] == "module"}
        self.assertEqual(motifs["b1"], "feedback")
        self.assertEqual(motifs["b3"], "feedback")

    def test_a_chain_without_signals_is_linear(self):
        graph, report = self.normalize(two_block_payload())

        self.assertEqual(report["primary_structure"], "linear")
        self.assertEqual(report["structure_signals"], ["linear"])
        self.assertEqual(graph["provenance"]["reading_direction"], "left_to_right")

    def test_merging_sources_are_reported_as_converging(self):
        payload = two_block_payload(connectors=[
            {"id": "e1", "source_id": "b1", "target_id": "b3", "relation_type": "forward"},
            {"id": "e2", "source_id": "b2", "target_id": "b3", "relation_type": "merge"},
        ])
        payload["blocks"].append(
            {"id": "b3", "label": "Fusion", "panel_id": "p1", "box": {"x": 0.7, "y": 0.4, "w": 0.2, "h": 0.2}, "text_confidence": 0.7, "visual_role": "module"}
        )
        graph, report = self.normalize(payload)

        self.assertEqual(report["primary_structure"], "converging")
        self.assertEqual(report["merging_blocks"], ["b3"])
        self.assertEqual(report["branching_blocks"], [])
        motifs = {node["properties"]["source_id"]: node["properties"]["motif"] for node in graph["nodes"] if node["kind"] == "module"}
        self.assertEqual(motifs["b3"], "merge")

    def test_pilot_labels_are_only_used_afterwards(self):
        payload = two_block_payload()
        row = {"panel_count": 3, "node_count": 9, "structure": "linear", "has_arrows": False}
        graph, report = self.normalize(payload, pilot_row=row)

        agreement = report["pilot_label_agreement"]
        self.assertFalse(agreement["panel_count"]["agrees"])
        self.assertFalse(agreement["node_count"]["agrees"])
        self.assertFalse(agreement["has_arrows"]["agrees"])
        self.assertTrue(agreement["structure"]["agrees"])
        self.assertTrue(agreement["structure"]["agrees_with_signals"])
        self.assertFalse(graph["provenance"]["pilot_labels_used_in_prompt"])

    def test_cross_panel_and_backwards_edges_are_reported(self):
        payload = two_block_payload()
        payload["panels"] = [
            {"id": "p1", "label": "left", "box": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0}},
            {"id": "p2", "label": "right", "box": {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0}},
        ]
        payload["blocks"][0]["panel_id"] = "p1"
        payload["blocks"][0]["box"] = {"x": 0.05, "y": 0.1, "w": 0.2, "h": 0.2}
        payload["blocks"][1]["panel_id"] = "p2"
        payload["blocks"][1]["box"] = {"x": 0.55, "y": 0.1, "w": 0.2, "h": 0.2}
        payload["connectors"] = [
            {"id": "e1", "source_id": "b2", "target_id": "b1", "relation_type": "cross_panel"},
        ]
        graph, report = self.normalize(payload)

        self.assertEqual(len(report["cross_panel_edges"]), 1)
        self.assertEqual(len(report["backwards_edges"]), 1)
        self.assertEqual(report["connectors_not_marked_feedback"], report["backwards_edges"])
        self.assertTrue(any("backwards" in warning for warning in report["warnings"]))

    def test_unknown_relation_types_are_reported_but_kept(self):
        payload = two_block_payload(connectors=[
            {"id": "e1", "source_id": "b1", "target_id": "b2", "relation_type": "teleports"},
        ])
        graph, report = self.normalize(payload)

        self.assertEqual(report["unknown_relation_types"], ["teleports"])
        self.assertEqual(graph["edges"][0]["relation_type"], "teleports")

    def test_missing_panel_box_is_an_error(self):
        payload = two_block_payload(panels=[{"id": "p1", "label": "broken", "box": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}}])
        graph, report = self.normalize(payload)

        self.assertIsNone(graph)
        self.assertTrue(any("no usable normalized box" in error for error in report["errors"]))


class PromptAndStructureTests(unittest.TestCase):
    def test_prompt_carries_captions_and_never_the_pilot_labels(self):
        prompt = build_visual_prompt(captions=["Overview of the frozen framework.", "Second panel caption."])

        self.assertIn("Overview of the frozen framework.", prompt)
        self.assertIn("reading_direction", prompt)
        self.assertIn("cross_panel", prompt)
        self.assertNotIn("linear", prompt.casefold())
        self.assertNotIn("multi_panel", prompt)
        self.assertNotIn("node_count", prompt)

    def test_structure_derivation_prefers_the_strongest_signal(self):
        self.assertEqual(
            derive_structure(3, converging=False, branching=True, feedback_present=False),
            ("multi_panel", ["multi_panel", "branching"]),
        )
        self.assertEqual(
            derive_structure(1, converging=True, branching=False, feedback_present=False),
            ("converging", ["converging"]),
        )
        self.assertEqual(
            derive_structure(1, converging=False, branching=False, feedback_present=True),
            ("feedback_loop", ["feedback_loop"]),
        )
        self.assertEqual(
            derive_structure(1, converging=False, branching=False, feedback_present=False),
            ("linear", ["linear"]),
        )
        self.assertEqual(
            derive_structure(2, converging=True, branching=True, feedback_present=True),
            ("multi_panel", ["multi_panel", "feedback_loop", "converging", "branching"]),
        )

    def test_default_dpi_is_inside_the_supported_range(self):
        self.assertGreaterEqual(DEFAULT_FIGURE_DPI, 72)
        self.assertLessEqual(DEFAULT_FIGURE_DPI, 400)


class ReadingV2Tests(unittest.TestCase):
    """The second reading answers what each object is, not only what it says."""

    def normalize(self, payload, **kwargs):
        return normalize_visual_payload(payload, paper_id="p0001", **kwargs)

    def module_nodes(self, graph):
        return {node["properties"]["source_id"]: node for node in graph["nodes"] if node["kind"] == "module"}

    def test_prompt_asks_for_roles_repetition_and_notation(self):
        prompt = build_visual_prompt()

        self.assertIn("Read the figure the way its author meant it", prompt)
        self.assertIn("Never put the block's id in the label", prompt)
        for role in ("method_object", "external_method", "stage_annotation", "symbol", "decorative"):
            self.assertIn(role, prompt)
        for key in ("repeat_group", "repeat_index", "repeat_count", "belongs_to", "annotations", "regions", "entries"):
            self.assertIn(key, prompt)
        self.assertIn("the figure explaining itself, not a step of the method", prompt)

    def test_a_block_without_a_readable_label_keeps_the_empty_label(self):
        payload = two_block_payload()
        payload["blocks"][0]["label"] = ""
        graph, _report = self.normalize(payload)

        nodes = self.module_nodes(graph)
        # The graph contract fills a missing label with the node's own id, never with the
        # provider's block id; the binding layer reads a label equal to the id as unreadable.
        self.assertNotEqual(nodes["b1"]["label"], "b1")
        self.assertEqual(nodes["b1"]["label"], nodes["b1"]["id"])
        self.assertEqual(nodes["b2"]["label"], "Decoder")

    def test_a_panel_without_a_readable_label_keeps_the_empty_label(self):
        payload = two_block_payload()
        payload["panels"][0]["label"] = ""
        graph, _report = self.normalize(payload)

        panels = [node for node in graph["nodes"] if node["kind"] == "panel"]
        self.assertNotEqual(panels[0]["label"], "p1")
        self.assertEqual(panels[0]["label"], panels[0]["id"])

    def test_the_declared_role_never_overwrites_the_graphic_role(self):
        payload = two_block_payload()
        payload["blocks"][0]["visual_role"] = "stack"
        payload["blocks"][0]["role"] = "symbol"
        payload["blocks"][1]["role"] = "external_method"
        graph, _report = self.normalize(payload)

        nodes = self.module_nodes(graph)
        self.assertEqual(nodes["b1"]["role"], "stack")
        self.assertEqual(nodes["b1"]["properties"]["role"], "symbol")
        self.assertEqual(nodes["b2"]["role"], "module")
        self.assertEqual(nodes["b2"]["properties"]["role"], "external_method")

    def test_repeated_drawings_keep_their_group_index_and_count(self):
        payload = two_block_payload()
        payload["blocks"][1].update({"repeat_group": "diffusion", "repeat_index": 1, "repeat_count": 3})
        payload["blocks"].append({
            "id": "b3", "label": "Diffusion", "panel_id": "p1",
            "box": {"x": 0.7, "y": 0.1, "w": 0.2, "h": 0.2},
            "repeat_group": "diffusion", "repeat_index": 2, "repeat_count": 3,
        })
        graph, _report = self.normalize(payload)

        nodes = self.module_nodes(graph)
        self.assertEqual(nodes["b2"]["properties"]["repeat_group"], "diffusion")
        self.assertEqual(nodes["b2"]["properties"]["repeat_index"], 1)
        self.assertEqual(nodes["b2"]["properties"]["repeat_count"], 3)
        self.assertEqual(nodes["b3"]["properties"]["repeat_index"], 2)
        self.assertNotIn("repeat_group", nodes["b1"]["properties"])
        self.assertEqual({nodes["b2"]["properties"]["motif"], nodes["b3"]["properties"]["motif"]}, {"repeated"})
        self.assertNotEqual(nodes["b1"]["properties"]["motif"], "repeated")

    def test_annotations_and_the_object_they_belong_to_are_kept(self):
        payload = two_block_payload()
        payload["blocks"][0]["annotations"] = [{"text": "p(c|x)", "kind": "formula"}]
        payload["blocks"][1]["belongs_to"] = "b1"
        payload["blocks"][1]["annotations"] = [{"text": "", "kind": "note"}, "written by hand"]
        graph, _report = self.normalize(payload)

        nodes = self.module_nodes(graph)
        self.assertEqual(nodes["b1"]["properties"]["annotations"], [{"text": "p(c|x)", "kind": "formula"}])
        self.assertEqual(nodes["b2"]["properties"]["belongs_to"], "b1")
        self.assertEqual(nodes["b2"]["properties"]["annotations"], [{"text": "", "kind": "note"}, {"text": "written by hand", "kind": ""}])
        self.assertNotIn("belongs_to", nodes["b1"]["properties"])

    def test_a_notation_region_becomes_a_node_of_its_own_kind(self):
        payload = two_block_payload(regions=[
            {
                "id": "r1", "kind": "Legend", "label": "Legend", "explains": "arrow styles",
                "box": {"x": 0.0, "y": 0.8, "w": 1.0, "h": 0.2},
                "entries": [{"glyph": "conv 3x3", "meaning": "3x3 convolution"}, "bare entry"],
            }
        ])
        payload["connectors"].append({"id": "e2", "source_id": "b2", "target_id": "r1", "relation_type": "forward"})
        graph, report = self.normalize(payload)

        regions = [node for node in graph["nodes"] if node["kind"] == "notation_region"]
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["label"], "Legend")
        self.assertEqual(regions[0]["properties"]["kind"], "legend")
        self.assertEqual(regions[0]["properties"]["explains"], "arrow styles")
        self.assertEqual(regions[0]["properties"]["entries"], [{"glyph": "conv 3x3", "meaning": "3x3 convolution"}, {"glyph": "bare entry", "meaning": ""}])
        self.assertEqual(report["region_count"], 1)
        self.assertIn(regions[0]["id"], [edge["target"] for edge in graph["edges"]])
        self.assertEqual(len(graph["edges"]), 2)
        counted = sum(report["motif_histogram"].values())
        self.assertEqual(counted, len([node for node in graph["nodes"] if node["kind"] == "module"]))

    def test_a_paper_whose_figure_is_a_plot_reads_as_a_plot_not_as_a_failed_read(self):
        payload = {
            "panels": [{"id": "p1", "label": "Scatter plot of pass@1 against cost", "box": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}],
            "blocks": [],
            "connectors": [],
            "regions": [
                {
                    "id": "r1", "kind": "legend", "label": "Lines Legend",
                    "explains": "what each line of the plot is",
                    "box": {"x": 0.7, "y": 0.1, "w": 0.3, "h": 0.1},
                    "entries": [{"glyph": "red dashed line", "meaning": "Pareto Front"}],
                }
            ],
        }
        graph, report = self.normalize(payload)

        self.assertTrue(report["ok"], report["errors"])
        self.assertIsNotNone(graph)
        self.assertEqual(report["block_count"], 0)
        self.assertEqual(report["region_count"], 1)
        self.assertEqual([node["kind"] for node in graph["nodes"] if node["kind"] != "panel"], ["notation_region"])

    def test_an_answer_that_carries_no_shape_at_all_is_still_a_failed_read(self):
        graph, report = self.normalize({"panels": [], "blocks": [], "connectors": [], "regions": [], "summary": "nothing"})

        self.assertIsNone(graph)
        self.assertFalse(report["ok"])
        self.assertTrue(any("no readable block" in error for error in report["errors"]))


class PdfiumSerializationTests(unittest.TestCase):
    def test_pdf_render_waits_for_the_pdfium_lock(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_pdf(root / "figure.pdf")
            target = root / "out.png"
            finished = threading.Event()

            def render():
                render_figure_image(root / "figure.pdf", target, dpi=72, max_long_side=64)
                finished.set()

            with PDFIUM_LOCK:
                worker = threading.Thread(target=render)
                worker.start()
                time.sleep(0.3)
                self.assertFalse(finished.is_set())
                self.assertFalse(target.exists())
                worker.join(0.1)
            worker.join(30)

            self.assertTrue(finished.is_set())
            self.assertTrue(target.is_file())

    def test_concurrent_pdf_renders_all_succeed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_pdf(root / "figure.pdf")
            targets = [root / ("out-" + str(index) + ".png") for index in range(6)]

            def render(target):
                return render_figure_image(root / "figure.pdf", target, dpi=72, max_long_side=64)

            with ThreadPoolExecutor(max_workers=6) as pool:
                provenances = list(pool.map(render, targets))

            self.assertEqual(len(provenances), 6)
            for provenance, target in zip(provenances, targets):
                self.assertTrue(provenance["reader"].startswith("pypdfium2="))
                self.assertEqual(provenance["png_sha256"], provenances[0]["png_sha256"])
                self.assertTrue(target.is_file())
            self.assertEqual(len({provenance["png_sha256"] for provenance in provenances}), 1)


if __name__ == "__main__":
    unittest.main()
