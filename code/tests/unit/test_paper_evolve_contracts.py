import unittest
import json
import tempfile
from pathlib import Path

from rfs.contracts import (
    validate_figure_layout_plan,
    validate_icon_generation_plan,
    validate_method_figure_content,
)
from rfs.contracts.design_plan import validate_design_plan
from rfs.contracts.figure_text import validate_figure_text_plan
from rfs.contracts.ppt_composition import normalize_ppt_composition_plan, validate_ppt_composition_plan
from rfs.contracts.reference_layout_template import normalize_reference_layout_template, validate_reference_layout_template
from rfs.evaluation.interactive_judgement import judge_direct_run
from rfs.evaluation.geometry_quality import evaluate_density_quality, evaluate_geometry_quality
from rfs.evaluation.reference_adoption import evaluate_reference_adoption
from rfs.composition import arrange_node_contents
from rfs.planning.paper_figure_structure import create_layout_candidate
from rfs.analysis.reference import select_positive_framework_reference
from rfs.generation.icon_assets import generate_icon_assets
from rfs.generation.icon_background import normalize_icon_background
from PIL import Image
from rfs.workflows.progress import build_progress_reporter
from rfs.cli import build_parser
from rfs.workflows.paper_evolve_editable import _resolve_paper_id, _text_items_with_planned_roles
from rfs.evolution.direct.controller import compare_candidate


class PaperEvolveContractTests(unittest.TestCase):
    def test_a_composition_text_item_keeps_the_role_the_text_plan_gave_it(self):
        # The composition plan returns the same text ids as the text plan, but
        # only as boxes and styles. Reading those boxes as the whole text
        # program turned a relation label into plain text, so the arrow drew a
        # second copy of it and the copy was fitted below the readable minimum.
        text_plan = {
            "items": [
                {"id": "text_rel_observation_assistant", "text": "reviewed by", "text_role": "connector_label", "role": "connector_label", "target_id": "rel_observation_assistant", "font_size_role": "connector_label"},
                {"id": "text_panel_inputs", "text": "Inputs", "text_role": "panel_title", "role": "panel_title", "panel_id": "panel_inputs"},
            ]
        }
        composition = {
            "text_items": [
                {"id": "text_rel_observation_assistant", "text": "reviewed by", "bbox_percent": {"x": 0.6, "y": 0.3, "w": 0.06, "h": 0.04}, "style": {"font_size": 9}},
                {"id": "text_panel_inputs", "text": "Inputs", "bbox_percent": {"x": 0.02, "y": 0.1, "w": 0.2, "h": 0.04}},
            ]
        }
        merged = _text_items_with_planned_roles(composition, text_plan)
        by_id = {item["id"]: item for item in merged}
        self.assertEqual(by_id["text_rel_observation_assistant"]["role"], "connector_label")
        self.assertEqual(by_id["text_rel_observation_assistant"]["target_id"], "rel_observation_assistant")
        self.assertEqual(by_id["text_panel_inputs"]["role"], "panel_title")
        self.assertEqual(by_id["text_panel_inputs"]["panel_id"], "panel_inputs")

    def test_a_composition_without_text_items_falls_back_to_the_plan(self):
        text_plan = {"items": [{"id": "text_a", "text": "A", "role": "node_label"}]}
        self.assertEqual(_text_items_with_planned_roles({}, text_plan), text_plan["items"])

    def test_a_text_item_the_plan_never_mentioned_is_left_alone(self):
        text_plan = {"items": [{"id": "text_a", "text": "A", "role": "node_label"}]}
        composition = {"text_items": [{"id": "text_x", "text": "X", "bbox_percent": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}]}
        merged = _text_items_with_planned_roles(composition, text_plan)
        self.assertNotIn("role", merged[0])

    def test_frozen_acceptance_requires_significant_improvement(self):
        self.assertFalse(compare_candidate({"overall_score": 0.8}, {"overall_score": 0.805})["accepted"])
        self.assertTrue(compare_candidate({"overall_score": 0.8}, {"overall_score": 0.82})["accepted"])

    def test_method_content_requires_declared_relation_nodes(self):
        result = validate_method_figure_content(
            {
                "modules": [{"id": "encoder"}],
                "relations": [{"source": "encoder", "target": "missing"}],
            }
        )
        self.assertFalse(result["ok"])

    def test_layout_requires_bbox_and_declared_connector_nodes(self):
        result = validate_figure_layout_plan(
            {
                "nodes": [
                    {
                        "id": "encoder",
                        "bbox_percent": {"x": 0, "y": 0, "w": 0.2, "h": 0.2},
                    }
                ],
                "connectors": [{"source": "encoder", "target": "missing"}],
            }
        )
        self.assertFalse(result["ok"])

    def test_icon_requires_semantic_binding(self):
        result = validate_icon_generation_plan(
            {"icons": [{"asset_id": "icon_a", "prompt": "a visual"}]}
        )
        self.assertFalse(result["ok"])

    def test_bootstrap_judge_attributes_invalid_structure(self):
        result = judge_direct_run(
            {"modules": [{"id": "encoder"}]},
            {"nodes": [{"id": "encoder"}], "connectors": []},
            {
                "icons": [
                    {
                        "asset_id": "icon_encoder",
                        "semantic_node_ids": ["encoder"],
                        "prompt": "encoder",
                    }
                ]
            },
        )
        self.assertEqual(result["status"], "fault")
        self.assertEqual(result["stage_feedback"][0]["stage"], "structure")

    def test_design_plan_requires_panels_and_semantic_bindings(self):
        result = validate_design_plan({"nodes": [{"id": "n1"}]})
        self.assertFalse(result["ok"])
        self.assertIn("design plan requires at least two panels", result["errors"])

    def test_empty_text_plan_is_invalid(self):
        result = validate_figure_text_plan({"items": []})
        self.assertFalse(result["ok"])
        self.assertIn("figure text plan requires at least one item", result["errors"])

    def test_composition_requires_all_icon_placements(self):
        result = validate_ppt_composition_plan(
            {"icon_placements": [{"asset_id": "a", "bbox_percent": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}]},
            expected_asset_ids={"a", "b"},
        )
        self.assertFalse(result["ok"])
        self.assertIn("b", result["missing_asset_ids"])

    def test_vlm_array_bboxes_are_normalized(self):
        result = validate_design_plan({
            "panels": [
                {"id": "p", "bbox_percent": [0.0, 0.0, 0.5, 1.0]},
                {"id": "p2", "bbox_percent": [0.5, 0.0, 1.0, 1.0]},
            ],
            "nodes": [{"id": "n", "semantic_source_id": "s", "panel_id": "p", "bbox_percent": [0.1, 0.2, 0.4, 0.6]}],
        })
        self.assertTrue(result["ok"])
        composition = normalize_ppt_composition_plan({"icon_placements": [{"asset_id": "a", "bbox_percent": [0.1, 0.2, 0.4, 0.6]}]})
        self.assertAlmostEqual(composition["icon_placements"][0]["bbox_percent"]["w"], 0.3)

    def test_design_plan_allows_intentional_root_node_outside_panels(self):
        result = validate_design_plan({
            "panels": [
                {"id": "p1", "bbox_percent": {"x": 0.2, "y": 0.1, "w": 0.25, "h": 0.8}},
                {"id": "p2", "bbox_percent": {"x": 0.5, "y": 0.1, "w": 0.3, "h": 0.8}},
            ],
            "nodes": [
                {
                    "id": "input",
                    "semantic_source_id": "input",
                    "panel_id": None,
                    "parent_id": None,
                    "bbox_percent": {"x": 0.02, "y": 0.4, "w": 0.12, "h": 0.2},
                }
            ],
        })
        self.assertTrue(result["ok"], result["errors"])

    def test_design_plan_rejects_panel_orphan_with_missing_panel_id(self):
        result = validate_design_plan({
            "panels": [
                {"id": "p1", "bbox_percent": {"x": 0.2, "y": 0.1, "w": 0.25, "h": 0.8}},
                {"id": "p2", "bbox_percent": {"x": 0.5, "y": 0.1, "w": 0.3, "h": 0.8}},
            ],
            "nodes": [
                {
                    "id": "orphan",
                    "semantic_source_id": "orphan",
                    "panel_id": None,
                    "parent_id": None,
                    "bbox_percent": {"x": 0.25, "y": 0.4, "w": 0.12, "h": 0.2},
                }
            ],
        })
        self.assertFalse(result["ok"])
        self.assertIn(
            "node[0] without panel_id must be a root-level node outside every panel",
            result["errors"],
        )

    def test_sparse_panel_overlay_gets_compiler_safe_title(self):
        composition = normalize_ppt_composition_plan({"panels": [{"id": "panel_loop", "bbox_percent": {"x": 0, "y": 0, "w": 1, "h": 1}}]})
        self.assertEqual(composition["panels"][0]["title"], "Panel Loop")

    def test_bootstrap_judge_faults_incomplete_program(self):
        result = judge_direct_run(
            {"modules": [{"id": "encoder"}]},
            {"nodes": [{"id": "encoder"}], "connectors": []},
            {"icons": [{"asset_id": "icon_encoder", "semantic_node_ids": ["encoder"], "prompt": "encoder"}]},
            program={"panels": [], "semantic_nodes": [], "slots": [], "text_program": {"items": []}},
        )
        self.assertEqual(result["status"], "fault")
        self.assertTrue(any(item["stage"] == "compile" for item in result["stage_feedback"]))

    def test_node_layout_separates_icon_and_label(self):
        program, report = arrange_node_contents({
            "semantic_nodes": [{"id": "n", "label": "A long scientific node label", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3}}],
            "slots": [{"id": "slot_n", "asset_id": "icon_n", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3}}],
        })
        quality = evaluate_geometry_quality(program)
        self.assertEqual(report["node_count"], 1)
        self.assertEqual(quality["icon_text_overlap_count"], 0)
        self.assertEqual(quality["node_boundary_violation_count"], 0)

    def test_geometry_accepts_explicit_parent_child_containment(self):
        quality = evaluate_geometry_quality({
            "semantic_nodes": [
                {"id": "parent", "panel_id": "p", "render_policy": "panel_only", "object_type": "container_anchor", "bbox_percent": {"x": .1, "y": .1, "w": .5, "h": .5}},
                {"id": "child", "panel_id": "p", "parent_id": "parent", "bbox_percent": {"x": .2, "y": .2, "w": .2, "h": .2}},
            ],
            "slots": [],
        })
        self.assertEqual(quality["status"], "pass")
        self.assertEqual(quality["node_overlap_count"], 0)
        self.assertEqual(quality["hierarchy_containment_count"], 1)

    def test_geometry_reads_an_overhanging_child_as_containment(self):
        """A container that was not grown to its payload is not a collision.

        The corpus nested parents left the last child poking past the container
        edge by a couple of percent of the canvas; read as a collision that
        voided the whole figure, while the nesting itself was correct.
        """

        quality = evaluate_geometry_quality({
            "semantic_nodes": [
                {"id": "parent", "panel_id": "p", "render_policy": "panel_only", "object_type": "container_anchor", "bbox_percent": {"x": .1, "y": .1, "w": .5, "h": .5}},
                {"id": "child", "panel_id": "p", "parent_id": "parent", "bbox_percent": {"x": .2, "y": .45, "w": .2, "h": .2}},
            ],
            "slots": [],
        })
        self.assertEqual(quality["node_overlap_count"], 0)
        self.assertEqual(quality["hierarchy_containment_count"], 1)
        row = quality["hierarchy_containments"][0]
        self.assertGreater(row["outside_area"], 0)
        self.assertAlmostEqual(row["outside_area"], 0.2 * 0.05, places=6)

    def test_reference_template_requires_executable_zones(self):
        template = normalize_reference_layout_template({
            "zones": [
                {"id": "left", "relative_width": 0.4, "role": "input", "internal_pattern": "stacked_cards"},
                {"id": "right", "relative_width": 0.6, "role": "method", "internal_pattern": "hierarchy"},
            ],
            "relation_patterns": [{"source_zone": "left", "target_zone": "right", "type": "flow"}],
        })
        self.assertTrue(validate_reference_layout_template(template)["ok"])

    def test_density_report_flags_sparse_panels(self):
        result = evaluate_density_quality({
            "panels": [{"id": "p", "bbox_percent": {"x": 0, "y": 0, "w": 1, "h": 1}}],
            "semantic_nodes": [{"id": "n", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}}],
        })
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["sparse_panel_count"], 1)

    def test_reference_adoption_flags_structure_free_composition(self):
        result = evaluate_reference_adoption(
            {"zones": [{"id": "a"}, {"id": "b"}, {"id": "c"}], "relation_patterns": [{"source_zone": "a", "target_zone": "b"}]},
            {"panels": [], "connectors": []},
        )
        self.assertEqual(result["status"], "blocked")

    def test_bootstrap_judge_rejects_sparse_final_program(self):
        result = judge_direct_run(
            {"modules": [{"id": "n"}]},
            {"nodes": [{"id": "n"}], "connectors": []},
            {"icons": [{"asset_id": "icon_n", "semantic_node_ids": ["n"], "prompt": "n"}]},
            program={
                "panels": [{"id": "p", "bbox_percent": {"x": 0, "y": 0, "w": 1, "h": 1}}],
                "semantic_nodes": [{"id": "n", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}, "label_bbox_percent": {"x": 0.1, "y": 0.17, "w": 0.1, "h": 0.03}}],
                "slots": [{"id": "slot_n", "asset_id": "icon_n", "parent_id": "n", "bbox_percent": {"x": 0.11, "y": 0.11, "w": 0.08, "h": 0.05}}],
                "text_program": {"items": [{"id": "t", "text": "n", "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.03}}]},
            },
        )
        self.assertEqual(result["status"], "fault")
        self.assertTrue(any("sparse panels" in item["problem"] for item in result["stage_feedback"]))
        density_repair = next(
            item for entry in result["stage_feedback"] for item in entry.get("repair") or []
            if isinstance(item, dict) and item.get("action") in {"resize_nodes", "relayout_slots", "relayout_global"}
        )
        self.assertIn(density_repair["action"], {"resize_nodes", "relayout_slots"})
        self.assertTrue(density_repair.get("slot_ids") or density_repair.get("object_ids") or density_repair.get("semantic_node_ids"))

    def test_layout_candidate_preserves_node_identity(self):
        base = {
            "nodes": [
                {"id": "a", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}},
                {"id": "b", "bbox_percent": {"x": 0.4, "y": 0.1, "w": 0.2, "h": 0.2}},
            ],
            "connectors": [{"id": "e", "source": "a", "target": "b"}],
        }
        candidate = create_layout_candidate(base, 1)
        self.assertEqual([item["id"] for item in candidate["nodes"]], ["a", "b"])
        self.assertEqual(candidate["connectors"][0]["source"], "a")

    def test_positive_dataset_selection_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paper = root / "paper.pdf"
            figure = root / "figure.png"
            paper.write_bytes(b"pdf")
            figure.write_bytes(b"png")
            (root / "papers.jsonl").write_text(
                json.dumps(
                    {"paper_id": "p1", "paper_path": str(paper), "title": "T"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "figures.jsonl").write_text(
                json.dumps(
                    {
                        "paper_id": "p1",
                        "figure_path": str(figure),
                        "figure_id": "f1",
                        "final_label": "好（科研）",
                        "candidate_rank": 1,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            first = select_positive_framework_reference(root, seed=7)
            second = select_positive_framework_reference(root, seed=7)
            self.assertEqual(
                first["selected"]["paper_id"], second["selected"]["paper_id"]
            )

    def test_icon_candidate_count_matches_written_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            report = generate_icon_assets(
                {"icons": [{"asset_id": "icon_a", "subject": "A", "prompt": "A"}]},
                temp,
                mode="placeholder",
                candidates=2,
            )
            asset = report["assets"][0]
            self.assertEqual(asset["candidate_count"], 2)
            self.assertEqual(len(asset["candidate_paths"]), 2)
            self.assertTrue(
                all(Path(path).exists() for path in asset["candidate_paths"])
            )

    def test_initial_candidate_can_reuse_frozen_icon_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reusable = root / "frozen" / "icon_a" / "selected.png"
            reusable.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), "white").save(reusable)
            report = generate_icon_assets(
                {"icons": [{"asset_id": "icon_a", "subject": "A", "prompt": "A"}]},
                root / "candidate",
                mode="image2",
                reuse_assets_from=root / "frozen",
            )
            self.assertEqual(report["api_requests_attempted"], 0)
            self.assertEqual(report["assets"][0]["status"], "reused")

    def test_staged_icon_is_reused_under_the_contract_id(self):
        """The layout designer may rename a node, but the paid icon is filed by contract id."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            staged = root / "candidate" / "assets" / "icon_mod_feature_extractor" / "candidate_00.png"
            staged.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), "white").save(staged)
            report = generate_icon_assets(
                {
                    "icons": [
                        {
                            "asset_id": "icon_node_extractor",
                            "semantic_source_id": "mod_feature_extractor",
                            "subject": "Feature Extractor",
                            "prompt": "Feature Extractor",
                        }
                    ]
                },
                root / "candidate",
                mode="image2",
            )
            self.assertEqual(report["api_requests_attempted"], 0)
            self.assertEqual(report["assets"][0]["status"], "reused")

    def test_a_slot_of_this_plan_is_never_borrowed_as_an_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            other = root / "candidate" / "assets" / "icon_mod_feature_extractor" / "candidate_00.png"
            other.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), "white").save(other)
            report = generate_icon_assets(
                {
                    "icons": [
                        {
                            "asset_id": "icon_mod_feature_extractor",
                            "subject": "Feature Extractor",
                            "prompt": "Feature Extractor",
                        },
                        {
                            "asset_id": "icon_node_extractor",
                            "semantic_source_id": "mod_feature_extractor",
                            "subject": "Feature Extractor",
                            "prompt": "Feature Extractor",
                        },
                    ]
                },
                root / "candidate",
                mode="placeholder",
            )
            statuses = {asset["asset_id"]: asset["status"] for asset in report["assets"]}
            self.assertEqual(statuses["icon_mod_feature_extractor"], "reused")
            self.assertNotEqual(statuses["icon_node_extractor"], "reused")

    def test_icon_background_becomes_transparent_at_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "icon.png"
            image = Image.new("RGB", (32, 32), "white")
            for x in range(10, 22):
                for y in range(10, 22):
                    image.putpixel((x, y), (20, 80, 140))
            image.save(path)
            report = normalize_icon_background(path, "#EAF1F8")
            with Image.open(path).convert("RGBA") as result:
                self.assertEqual(result.getpixel((0, 0))[3], 0)
                self.assertEqual(result.getpixel((16, 16))[3], 255)
            self.assertGreater(report["changed_pixel_count"], 0)

    def test_progress_reporter_writes_ordered_jsonl(self):
        with tempfile.TemporaryDirectory() as temp:
            progress = build_progress_reporter(temp, echo=False)
            progress({"stage": "paper", "status": "started"})
            progress({"stage": "paper", "status": "completed"})
            rows = [
                json.loads(line)
                for line in (Path(temp) / "progress.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["seq"] for row in rows], [1, 2])
            self.assertEqual(rows[-1]["status"], "completed")


class PaperIdentityTests(unittest.TestCase):
    """Which paper a run is about when the middle layer kept only a title.

    A paper the middle layer indexed under a path carries no identifier, so a run left to read its
    input path names the paper after the folder it was handed -- one name shared by every paper,
    which names none of them. That is why the continuous loop passes the identifier it already
    holds, and these tests pin both halves: the passed name wins, and the fallback is what it is.
    """

    def test_the_identifier_the_caller_knows_wins(self):
        resolved = _resolve_paper_id(
            "10.1145/3770854.3780184",
            {"paper_id": "from-the-prepared-contract"},
            {"paper_identity": {"title": "T"}, "graph_provenance": {"paper_id": "from-the-provenance"}},
            "/runs/10.1145_3770854.3780184/attempt_01/inputs/paper",
        )
        self.assertEqual(resolved, "10.1145/3770854.3780184")

    def test_a_blank_identifier_does_not_win(self):
        resolved = _resolve_paper_id("   ", {"paper_id": "contract-1"}, {}, "/p/paper.pdf")
        self.assertEqual(resolved, "contract-1")

    def test_the_contract_provenance_names_the_paper_when_the_caller_does_not(self):
        resolved = _resolve_paper_id(
            "",
            {"paper_review": {}},
            {"graph_provenance": {"paper_id": "10.1000/xyz"}},
            "/p/paper.pdf",
        )
        self.assertEqual(resolved, "10.1000/xyz")

    def test_a_directory_input_falls_back_to_its_own_folder_name(self):
        resolved = _resolve_paper_id("", {}, {}, "/runs/10.1145_3770854.3780184/attempt_01/inputs/paper")
        self.assertEqual(resolved, "paper")


class PaperIdCliTests(unittest.TestCase):
    """The product command can be told which paper it is about."""

    def test_it_defaults_to_empty_and_takes_a_doi(self):
        parser = build_parser()
        args = parser.parse_args(["paper-evolve-editable", "--paper", "p.pdf", "--out", "o"])
        self.assertEqual(args.paper_id, "")
        args = parser.parse_args(
            ["paper-evolve-editable", "--paper", "p.pdf", "--out", "o", "--paper-id", "10.1145/3770854.3780184"]
        )
        self.assertEqual(args.paper_id, "10.1145/3770854.3780184")
