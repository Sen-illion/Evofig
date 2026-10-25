"""Integration tests for the judge-data Pilot graph CLI commands."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from rfs.analysis.paper.document import evidence_id

from rfs.cli import _run_judge_data, build_parser, main
from rfs.judge_training.pilot_index import index_all
from rfs.judge_training.preference_sources import SourceUnavailable
from rfs.judge_training.source_snapshot_io import write_jsonl
from rfs.utils import write_json


def prompt_check_ids(prompt):
    """Return the short check id the prompt printed for each label, as a reader would.

    The check stage asks with c1, c2 and r1, r2 instead of the hashed scientific ids,
    so the stand-in below reads those ids straight off the prompt before answering.
    """
    concepts = {}
    relations = {}
    for line in prompt.splitlines():
        text = line.strip()
        if not text.startswith("- ") or " | " not in text:
            continue
        alias, _, rest = text[2:].partition(" | ")
        alias = alias.strip()
        if not rest.startswith('"'):
            continue
        label = rest[1:].split('"', 1)[0]
        if alias.startswith("c"):
            concepts[label] = alias
        elif alias.startswith("r"):
            relations[label] = alias
    return concepts, relations


def write_fixture(root):
    """Write a frozen snapshot and a Pilot version with one paper."""
    snapshot = root / "paper-d0-v001"
    pilot = root / "paper-d0-pilot-v002"
    directory = snapshot / "assets" / "paper-p0001"
    directory.mkdir(parents=True)
    pilot.mkdir(parents=True)
    records = [
        {"page": 1, "text": "A Frozen Study\nAbstract\nWe propose a frozen pipeline."},
        {"page": 2, "text": "1 Introduction\nFrozen snapshots matter."},
    ]
    (directory / "text_locations.jsonl").write_text(
        json.dumps(records[0], ensure_ascii=False) + "\n" + json.dumps(records[1], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (directory / "full_text.txt").write_text(
        "\n\n".join(record["text"] for record in records), encoding="utf-8"
    )
    (directory / "captions.json").write_text(json.dumps({
        "paper_id": "p0001",
        "figures": [{"figure_id": "figure-1", "caption": "Overview of the frozen framework.", "pdf_page": 2}],
    }, ensure_ascii=False), encoding="utf-8")
    (directory / "bindings.json").write_text(json.dumps({"paper_id": "p0001", "bindings": []}), encoding="utf-8")
    figures = directory / "figures"
    figures.mkdir(parents=True)
    Image.new("RGB", (4, 4), (255, 255, 255)).save(figures / "figure.png")
    (snapshot / "asset_manifest.jsonl").write_text(json.dumps({
        "paper_id": "p0001",
        "asset_id": "asset-p0001",
        "relative_path": "assets/paper-p0001/figures/figure.png",
        "sha256": "",
        "suffix": ".png",
        "role": "figure",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    (snapshot / "eligible_papers.jsonl").write_text(json.dumps({
        "paper_id": "p0001",
        "title": "A Frozen Study",
        "primary_domain": "cv",
        "captions_path": "assets/paper-p0001/captions.json",
        "bindings_path": "assets/paper-p0001/bindings.json",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    (snapshot / "snapshot_manifest.json").write_text(json.dumps({"snapshot_id": "paper-d0-v001"}), encoding="utf-8")
    (pilot / "pilot_ledger.jsonl").write_text(json.dumps({
        "paper_id": "p0001", "split": "train", "domain": "cv", "figure_type": "framework", "structure": "linear", "complexity": 2,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    (pilot / "pilot_manifest.json").write_text(json.dumps({"pilot_id": "paper-d0-pilot-v002"}), encoding="utf-8")
    return snapshot, pilot


def _chain(paper_id):
    nodes = [
        {"id": "m1", "kind": "module", "label": "Encoder", "role": "module", "required": True},
        {"id": "m2", "kind": "module", "label": "Decoder", "role": "module", "required": True},
        {"id": "m3", "kind": "module", "label": "Output", "role": "module", "required": True},
    ]
    edges = [
        {"id": "e1", "source": "m1", "target": "m2", "relation_type": "data_flow", "required": True},
        {"id": "e2", "source": "m2", "target": "m3", "relation_type": "data_flow", "required": True},
    ]
    return {
        "graph_id": f"scientific-{paper_id}",
        "paper_id": paper_id,
        "kind": "scientific",
        "nodes": nodes,
        "edges": edges,
        "required_node_ids": [item["id"] for item in nodes],
        "required_edge_ids": [item["id"] for item in edges],
    }


def write_middleware_version(root, papers=("p0001", "p0002")):
    """Write a middleware version whose index already carries subgraph vectors."""

    rows = [{"paper_id": paper_id, "split": "train", "domain": "cv"} for paper_id in papers]
    graphs = {paper_id: {"paper_id": paper_id, "scientific_graph": _chain(paper_id)} for paper_id in papers}
    visuals = {paper_id: {"paper_id": paper_id, "visual_graph": _chain(paper_id)} for paper_id in papers}
    for paper_id in papers:
        folder = root / "graphs" / paper_id
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "scientific_graph.json", graphs[paper_id]["scientific_graph"])
        write_json(folder / "visual_graph.json", visuals[paper_id]["visual_graph"])
    payload, _ = index_all(rows, graphs=graphs, visuals=visuals, graph_id="cli-version", directories={paper_id: paper_id for paper_id in papers})
    write_jsonl(root / "graph_index.jsonl", payload["records"])
    write_jsonl(root / "motif_index.jsonl", payload["motifs"])
    write_jsonl(root / "subgraph_index.jsonl", payload["subgraphs"])
    write_json(root / "retrieval_probe.json", payload["probe"])
    return root


class JudgeDataPilotGraphCliTests(unittest.TestCase):
    def test_commands_parse(self):
        parser = build_parser()
        args = parser.parse_args(["judge-data", "build-pilot-graphs", "--snapshot", "d0", "--pilot", "pilot", "--out", "graphs"])
        self.assertEqual(args.command, "judge-data")
        self.assertEqual(args.judge_data_action, "build-pilot-graphs")
        self.assertEqual(args.stages, "prepare")
        self.assertIsNone(args.limit)
        self.assertIsNone(args.graph_id)
        self.assertFalse(args.resume)
        self.assertFalse(args.strict_review)
        self.assertEqual(args.workers, 4)
        self.assertEqual(args.timeout, 300)
        self.assertEqual(args.figure_dpi, 150)
        self.assertEqual(args.max_long_side, 1600)
        self.assertEqual(args.embedding_model, "")
        named = parser.parse_args([
            "judge-data", "build-pilot-graphs", "--snapshot", "d0", "--pilot", "pilot", "--out", "graphs",
            "--embedding-model", "st:models/bge-small-en-v1.5",
        ])
        self.assertEqual(named.embedding_model, "st:models/bge-small-en-v1.5")
        args = parser.parse_args(["judge-data", "validate-pilot-graphs", "--graphs", "graphs"])
        self.assertEqual(args.judge_data_action, "validate-pilot-graphs")

    def test_build_subgraph_pairs_then_reach_the_preference_pairs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = write_middleware_version(root / "version")
            args = build_parser().parse_args([
                "judge-data", "build-subgraph-pairs",
                "--middleware", str(version), "--out", str(root / "out"), "--split", "train", "--no-swapped",
            ])
            report = _run_judge_data(args)
            pair_args = build_parser().parse_args([
                "judge-data", "build-preference-pairs",
                "--dataset", report["out"], "--out", str(root / "pairs.jsonl"),
            ])
            pairs = _run_judge_data(pair_args)
            examples_written = Path(report["out"]).is_file()
            images_written = sorted((root / "out" / "subgraphs").glob("*.png"))

        self.assertTrue(report["ok"], report["discipline"])
        self.assertEqual(report["discipline"]["candidate_split_leaks"], [])
        self.assertGreater(report["counts"]["examples"], 0)
        self.assertTrue(examples_written)
        self.assertTrue(images_written)
        self.assertTrue(pairs["ok"])
        self.assertGreater(pairs["pairs"], 0)
        self.assertEqual(pairs["by_family"], {"misdiagnosed": pairs["pairs"] // 3, "missed_defect": pairs["pairs"] // 3, "wrong_pick": pairs["pairs"] // 3})

    def test_build_subgraph_pairs_refuses_a_version_that_cannot_pair_anything(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = write_middleware_version(root / "version", papers=("p0001",))
            args = build_parser().parse_args([
                "judge-data", "build-subgraph-pairs", "--middleware", str(version), "--out", str(root / "out"),
            ])
            with self.assertRaises(SourceUnavailable):
                _run_judge_data(args)

        self.assertFalse((root / "out" / "rag_subgraph_examples.jsonl").exists())

    def test_build_then_validate(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            args = build_parser().parse_args([
                "judge-data", "build-pilot-graphs",
                "--snapshot", str(snapshot), "--pilot", str(pilot), "--out", str(root / "graphs"),
            ])
            built = _run_judge_data(args)
            check_args = build_parser().parse_args(["judge-data", "validate-pilot-graphs", "--graphs", str(root / "graphs")])
            checked = _run_judge_data(check_args)

        self.assertTrue(built["ok"])
        self.assertEqual(built["papers"], 1)
        self.assertEqual(built["stages_completed"], ["prepare"])
        self.assertFalse(built["sealed"])
        self.assertGreater(built["estimated_review_tokens"], 0)
        self.assertTrue(checked["ok"])
        self.assertFalse(checked["complete"])

    def test_unknown_stage_fails_the_command(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                status = main([
                    "judge-data", "build-pilot-graphs",
                    "--snapshot", str(snapshot), "--pilot", str(pilot), "--out", str(root / "graphs"),
                    "--stages", "review,nonsense", "--json",
                ])

        self.assertEqual(status, 1)
        self.assertIn("Unknown Pilot graph stages", buffer.getvalue())
        self.assertIn("index", buffer.getvalue())

    def test_review_stage_runs_through_the_cli(self):
        evidence = evidence_id(1, "A Frozen Study Abstract We propose a frozen pipeline.", None, "paragraph")

        def fake_call(prompt, image_paths, model=None, timeout=180, retries=1, call_metadata=None, deadline_at=None):
            if call_metadata is not None:
                call_metadata.update({
                    "attempts": 1, "success": True, "elapsed_seconds": 0.01, "failure_categories": [],
                    "prompt_chars": len(prompt), "estimated_tokens": max(1, round(len(prompt) / 4)),
                    "image_count": 0, "image_bytes": 0, "usage": {"prompt_tokens": 900, "completion_tokens": 100, "total_tokens": 1000},
                    "token_source": "provider", "prompt_tokens": 900, "completion_tokens": 100, "total_tokens": 1000,
                })
            return {
                "summary": "Fake review.",
                "research_questions": [{"id": "rq1", "statement": "Does it work?", "evidence_ids": [evidence]}],
                "modules": [{"id": "m1", "statement": "Frozen encoder", "evidence_ids": [evidence]}],
            }

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            args = build_parser().parse_args([
                "judge-data", "build-pilot-graphs",
                "--snapshot", str(snapshot), "--pilot", str(pilot), "--out", str(root / "graphs"),
                "--stages", "prepare,review", "--workers", "1",
            ])
            with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
                with patch("rfs.analysis.paper.review.call_vlm_json", fake_call):
                    result = _run_judge_data(args)
            check_args = build_parser().parse_args(["judge-data", "validate-pilot-graphs", "--graphs", str(root / "graphs")])
            checked = _run_judge_data(check_args)

        self.assertTrue(result["ok"], result.get("failures"))
        self.assertEqual(result["reviews"], 1)
        self.assertEqual(result["review_tokens"], 1000)
        self.assertEqual(result["review_token_source"], ["provider"])
        self.assertEqual(result["stages_completed"], ["prepare", "review"])
        self.assertTrue(checked["ok"])
        self.assertEqual(checked["checks"]["review_rows"], 1)



    def test_contract_and_graph_stages_run_through_the_cli(self):
        evidence = evidence_id(1, "A Frozen Study Abstract We propose a frozen pipeline.", None, "paragraph")

        def fake_review(prompt, image_paths, model=None, timeout=180, retries=1, call_metadata=None, deadline_at=None):
            if call_metadata is not None:
                call_metadata.update({
                    "attempts": 1, "success": True, "elapsed_seconds": 0.01, "failure_categories": [],
                    "prompt_chars": len(prompt), "estimated_tokens": max(1, round(len(prompt) / 4)),
                    "image_count": 0, "image_bytes": 0, "usage": {"prompt_tokens": 900, "completion_tokens": 100, "total_tokens": 1000},
                    "token_source": "provider", "prompt_tokens": 900, "completion_tokens": 100, "total_tokens": 1000,
                })
            return {
                "summary": "Fake review.",
                "central_claims": [{"id": "c1", "statement": "The frozen pipeline works.", "evidence_ids": [evidence]}],
                "inputs": [{"id": "in1", "statement": "Frozen text", "evidence_ids": [evidence], "must_appear_in_figure": True, "visual_role": "input"}],
                "modules": [{"id": "m1", "statement": "Frozen encoder", "evidence_ids": [evidence], "must_appear_in_figure": True}],
                "outputs": [{"id": "out1", "statement": "Framework figure", "evidence_ids": [evidence], "must_appear_in_figure": True, "visual_role": "output"}],
                "relations": [{"id": "rel1", "source_id": "m1", "target_id": "out1", "relation_type": "data_flow", "statement": "produces", "evidence_ids": [evidence]}],
            }

        def fake_contract(prompt, image_paths, model=None, timeout=180, retries=1, call_metadata=None, deadline_at=None):
            if call_metadata is not None:
                call_metadata.update({
                    "attempts": 1, "success": True, "elapsed_seconds": 0.02, "failure_categories": [],
                    "prompt_chars": len(prompt), "estimated_tokens": max(1, round(len(prompt) / 4)),
                    "image_count": 0, "image_bytes": 0, "usage": {"prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000},
                    "token_source": "provider", "prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000,
                })
            return {
                "figure_goal": "Show the frozen pipeline.",
                "inputs": [{"id": "in1", "label": "Frozen text", "detail_level": "artifact", "must_show": True, "evidence_ids": [evidence]}],
                "modules": [{"id": "m1", "label": "Frozen encoder", "detail_level": "module", "must_show": True, "evidence_ids": [evidence]}],
                "outputs": [{"id": "out1", "label": "Framework figure", "detail_level": "artifact", "must_show": True, "evidence_ids": [evidence]}],
                "relations": [{"id": "r1", "source": "m1", "target": "out1", "type": "data_flow", "label": "produces", "evidence_ids": [evidence]}],
            }

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            args = build_parser().parse_args([
                "judge-data", "build-pilot-graphs",
                "--snapshot", str(snapshot), "--pilot", str(pilot), "--out", str(root / "graphs"),
                "--stages", "prepare,review,contract,graph", "--workers", "1",
            ])
            with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
                with patch("rfs.analysis.paper.review.call_vlm_json", fake_review):
                    with patch("rfs.analysis.paper.method_extractor.call_vlm_json", fake_contract):
                        result = _run_judge_data(args)
            check_args = build_parser().parse_args(["judge-data", "validate-pilot-graphs", "--graphs", str(root / "graphs")])
            checked = _run_judge_data(check_args)
            graph = json.loads((root / "graphs" / "graphs" / "p0001" / "scientific_graph.json").read_text(encoding="utf-8"))
            contract = json.loads((root / "graphs" / "graphs" / "p0001" / "figure_content_plan.json").read_text(encoding="utf-8"))

        self.assertTrue(result["ok"], result.get("failures"))
        self.assertEqual(result["stages_completed"], ["prepare", "review", "contract", "graph"])
        self.assertEqual(result["contracts"], 1)
        self.assertEqual(result["graphs"], 1)
        self.assertEqual(result["scientific_nodes"], 3)
        self.assertEqual(result["scientific_edges"], 1)
        self.assertEqual(result["contract_tokens"], 1000)
        self.assertEqual(result["provider_tokens"], {"review": 1000, "contract": 1000})
        self.assertEqual(result["provider_token_total"], 2000)
        self.assertTrue(checked["ok"])
        self.assertEqual(checked["checks"]["contracts"], 1)
        self.assertEqual(checked["checks"]["graphs"], 1)
        self.assertEqual(checked["checks"]["missing_paper_artifacts"], [])
        self.assertEqual(graph["kind"], "scientific")
        self.assertEqual(contract["contract"], "rfs.method_figure_content")


    def test_visual_stage_runs_through_the_cli(self):
        evidence = evidence_id(1, "A Frozen Study Abstract We propose a frozen pipeline.", None, "paragraph")

        def meta(call_metadata, tokens, images):
            if call_metadata is not None:
                call_metadata.update({
                    "attempts": 1, "success": True, "elapsed_seconds": 0.01, "failure_categories": [],
                    "prompt_chars": 120, "estimated_tokens": 30, "image_count": images, "image_bytes": 0,
                    "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens)},
                    "token_source": "provider", "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens),
                })

        def fake_review(prompt, image_paths, model=None, timeout=180, retries=1, call_metadata=None, deadline_at=None):
            meta(call_metadata, (900, 100), 0)
            return {
                "summary": "Fake review.",
                "central_claims": [{"id": "c1", "statement": "The frozen pipeline works.", "evidence_ids": [evidence]}],
                "inputs": [{"id": "in1", "statement": "Frozen text", "evidence_ids": [evidence], "must_appear_in_figure": True, "visual_role": "input"}],
                "modules": [
                    {"id": "m1", "statement": "Frozen encoder", "evidence_ids": [evidence], "must_appear_in_figure": True, "visual_role": "module"},
                    {"id": "m2", "statement": "Frozen decoder", "evidence_ids": [evidence], "must_appear_in_figure": True, "visual_role": "module"},
                ],
                "outputs": [{"id": "out1", "statement": "Framework figure", "evidence_ids": [evidence], "must_appear_in_figure": True, "visual_role": "output"}],
                "relations": [
                    {"id": "rel1", "source_id": "m1", "target_id": "m2", "relation_type": "data_flow", "statement": "feeds", "evidence_ids": [evidence]},
                    {"id": "rel2", "source_id": "m2", "target_id": "out1", "relation_type": "data_flow", "statement": "produces", "evidence_ids": [evidence]},
                ],
            }

        def fake_contract(prompt, image_paths, model=None, timeout=180, retries=1, call_metadata=None, deadline_at=None):
            meta(call_metadata, (800, 200), 0)
            return {
                "figure_goal": "Show the frozen pipeline.",
                "inputs": [{"id": "in1", "label": "Frozen text", "detail_level": "artifact", "must_show": True, "evidence_ids": [evidence]}],
                "modules": [
                    {"id": "m1", "label": "Frozen encoder", "detail_level": "module", "must_show": True, "evidence_ids": [evidence]},
                    {"id": "m2", "label": "Frozen decoder", "detail_level": "module", "must_show": True, "evidence_ids": [evidence]},
                ],
                "outputs": [{"id": "out1", "label": "Framework figure", "detail_level": "artifact", "must_show": True, "evidence_ids": [evidence]}],
                "relations": [
                    {"id": "r1", "source": "m1", "target": "m2", "type": "data_flow", "label": "feeds", "evidence_ids": [evidence]},
                    {"id": "r2", "source": "m2", "target": "out1", "type": "data_flow", "label": "produces", "evidence_ids": [evidence]},
                ],
            }

        def fake_visual(prompt, image_paths, model=None, timeout=180, retries=1, call_metadata=None, deadline_at=None):
            meta(call_metadata, (2000, 400), len(image_paths))
            return {
                "summary": "One panel with two blocks.",
                "reading_direction": "left_to_right",
                "panels": [{"id": "p1", "label": "pipeline", "box": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}],
                "blocks": [
                    {"id": "b1", "label": "Frozen encoder", "panel_id": "p1", "box": {"x": 0.05, "y": 0.1, "w": 0.2, "h": 0.2}, "text_confidence": 0.9, "visual_role": "module"},
                    {"id": "b2", "label": "Frozen decoder", "panel_id": "p1", "box": {"x": 0.4, "y": 0.1, "w": 0.2, "h": 0.2}, "text_confidence": 0.9, "visual_role": "module"},
                ],
                "connectors": [{"id": "e1", "source_id": "b1", "target_id": "b2", "relation_type": "forward", "label": "feeds"}],
                "uncertainties": [],
            }

        def fake_check(prompt, image_paths, model=None, timeout=180, retries=1, call_metadata=None, deadline_at=None):
            meta(call_metadata, (1500, 500), len(image_paths))
            concepts, relations = prompt_check_ids(prompt)

            def finding(label, *, visible, image_label="", where="", confidence=0.9):
                return {"id": concepts.get(label, ""), "visible": visible, "image_label": image_label, "where": where, "confidence": confidence}

            return {
                "image_summary": "One panel: two blocks and one arrow; the text input and the framework output are not drawn.",
                "concepts": [
                    finding("Frozen encoder", visible=True, image_label="Frozen encoder", where="panel 1, left"),
                    finding("Frozen decoder", visible=True, image_label="Frozen decoder", where="panel 1, right"),
                    finding("Frozen text", visible=False, confidence=0.6),
                    finding("Framework figure", visible=False, confidence=0.6),
                ],
                "relations": [
                    {"id": relations.get("Frozen encoder", ""), "arrow_present": True, "direction_matches": True, "confidence": 0.8},
                    {"id": relations.get("Frozen decoder", ""), "arrow_present": False, "confidence": 0.7},
                ],
                "extra_blocks": [],
                "uncertainties": [],
            }

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            args = build_parser().parse_args([
                "judge-data", "build-pilot-graphs",
                "--snapshot", str(snapshot), "--pilot", str(pilot), "--out", str(root / "graphs"),
                "--stages", "prepare,review,contract,graph,visual,bind,index,check", "--workers", "1", "--figure-dpi", "72",
            ])
            with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
                with patch("rfs.analysis.paper.review.call_vlm_json", fake_review):
                    with patch("rfs.analysis.paper.method_extractor.call_vlm_json", fake_contract):
                        with patch("rfs.judge_training.pilot_graphs.call_vlm_json", fake_visual):
                            with patch("rfs.judge_training.pilot_check.call_vlm_json", fake_check):
                                result = _run_judge_data(args)
            check_args = build_parser().parse_args(["judge-data", "validate-pilot-graphs", "--graphs", str(root / "graphs")])
            checked = _run_judge_data(check_args)
            clean_buffer = io.StringIO()
            with contextlib.redirect_stdout(clean_buffer), contextlib.redirect_stderr(clean_buffer):
                clean_gate = main(["judge-data", "validate-pilot-graphs", "--graphs", str(root / "graphs"), "--json"])
            visual = json.loads((root / "graphs" / "graphs" / "p0001" / "visual_graph.json").read_text(encoding="utf-8"))
            report = json.loads((root / "graphs" / "graphs" / "p0001" / "visual_graph_report.json").read_text(encoding="utf-8"))
            renders = sorted((root / "graphs" / "figures" / "p0001").glob("*.png"))
            sidecar = json.loads((root / "graphs" / "figures" / "p0001" / "figure.png.render.json").read_text(encoding="utf-8"))
            bindings = json.loads((root / "graphs" / "graphs" / "p0001" / "graph_bindings.json").read_text(encoding="utf-8"))
            index_records = [json.loads(line) for line in (root / "graphs" / "graph_index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            motif_rows = [json.loads(line) for line in (root / "graphs" / "motif_index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            subgraph_rows = [json.loads(line) for line in (root / "graphs" / "subgraph_index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            probe = json.loads((root / "graphs" / "retrieval_probe.json").read_text(encoding="utf-8"))
            review = json.loads((root / "graphs" / "graphs" / "p0001" / "figure_text_review.json").read_text(encoding="utf-8"))
            complete_marker = (root / "graphs" / "COMPLETE").is_file()
            stored_graph = root / "graphs" / "graphs" / "p0001" / "scientific_graph.json"
            edited = json.loads(stored_graph.read_text(encoding="utf-8"))
            edited["not_from_the_provider"] = True
            stored_graph.write_text(
                json.dumps(edited, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            recheck_args = build_parser().parse_args([
                "judge-data", "validate-pilot-graphs", "--graphs", str(root / "graphs"),
            ])
            rechecked = _run_judge_data(recheck_args)
            gate_buffer = io.StringIO()
            with contextlib.redirect_stdout(gate_buffer), contextlib.redirect_stderr(gate_buffer):
                tampered_gate = main(["judge-data", "validate-pilot-graphs", "--graphs", str(root / "graphs"), "--json"])

        self.assertTrue(result["ok"], result.get("failures"))
        self.assertEqual(
            result["stages_completed"],
            ["prepare", "review", "contract", "graph", "visual", "check", "bind", "index"],
        )
        self.assertEqual(result["visuals"], 1)
        self.assertEqual(result["visual_tokens"], 2400)
        self.assertEqual(result["visual_panels"], 1)
        self.assertEqual(result["visual_blocks"], 2)
        self.assertEqual(result["visual_edges"], 1)
        self.assertEqual(result["figure_dpi"], 72)
        self.assertEqual(result["provider_tokens"], {"review": 1000, "contract": 1000, "visual": 2400, "check": 2000})
        self.assertTrue(result["sealed"])
        self.assertTrue(checked["ok"], checked["checks"])
        self.assertEqual(checked["checks"]["visuals"], 1)
        self.assertEqual(checked["checks"]["rendered_figures"], 1)
        self.assertEqual(checked["checks"]["missing_figures"], [])
        self.assertEqual(checked["checks"]["missing_paper_artifacts"], [])
        self.assertEqual(visual["kind"], "visual")
        self.assertEqual(report["primary_structure"], "linear")
        self.assertEqual(result["bindings"], 1)
        self.assertEqual(result["bound_nodes"], 2)
        self.assertEqual(result["bound_edges"], 1)
        self.assertEqual(result["diagnostic_required_node_coverage"], 0.5)
        self.assertEqual(result["reversed_edges"], 0)
        self.assertEqual(result["provider_tokens_bind"], 0)
        self.assertEqual(bindings["contract"], "rfs.graph_bindings")
        self.assertEqual(
            sorted(item["status"] for item in bindings["node_bindings"]),
            ["bound", "bound", "unbound", "unbound"],
        )
        self.assertTrue(bindings["provenance"]["check_supplied"], "bind must run after check and read its review")
        self.assertEqual(bindings["provenance"]["provider_calls"], 0)
        self.assertEqual(checked["checks"]["bindings"], 1)
        self.assertEqual(len(renders), 1)
        self.assertEqual(sidecar["dpi"], 72)
        self.assertEqual(sidecar["source_suffix"], ".png")
        self.assertEqual(result["checks"], 1)
        self.assertEqual(result["check_tokens"], 2000)
        self.assertEqual(result["check_confirmed"], 2)
        self.assertEqual(result["check_missed_by_reader"], 0)
        self.assertEqual(result["check_hallucinated_block"], 0)
        self.assertEqual(result["check_direction_conflicts"], 0)
        self.assertEqual(result["check_mean_required_visible_rate"], 0.5)
        self.assertEqual(review["contract"], "rfs.figure_text_review")
        self.assertEqual(review["summary"]["confirmed"], 2)
        self.assertEqual(review["summary"]["not_drawn"], 2)
        self.assertEqual(result["index_records"], 2)
        self.assertEqual(result["index_papers"], 1)
        self.assertEqual(result["index_hidden_test_records"], 0)
        self.assertEqual(result["provider_tokens_index"], 0)
        self.assertEqual(sorted(row["graph_kind"] for row in index_records), ["scientific", "visual"])
        self.assertTrue(all(row["subgraphs"] == [] for row in index_records))
        self.assertTrue(all(row["subgraph_locator"]["count"] > 0 for row in index_records))
        self.assertEqual(len(subgraph_rows), sum(row["subgraph_locator"]["count"] for row in index_records))
        self.assertEqual(sorted({row["motif"] for row in subgraph_rows}), sorted(row["motif"] for row in motif_rows))
        self.assertEqual(probe["guards"]["self_hits"], 0)
        self.assertEqual(probe["guards"]["provider_calls"], 0)
        self.assertEqual(probe["guards"]["splits"], {"train": 2})
        self.assertEqual(len(probe["queries"]), 1)
        self.assertEqual(probe["queries"][0]["hit_count"], 0)
        self.assertTrue(complete_marker)
        self.assertEqual(checked["checks"]["checks"], 1)
        self.assertEqual(checked["checks"]["failed_checks"], [])
        self.assertEqual(checked["checks"]["index_records"], 2)
        self.assertEqual(checked["checks"]["failed_index"], [])
        self.assertEqual(checked["checks"]["missing_index_artifacts"], [])
        self.assertTrue(checked["complete"])
        self.assertTrue(checked["sealed"], checked["checks"])
        self.assertEqual(checked["checks"]["ledger_artifact_mismatches"], [])
        self.assertEqual(checked["checks"]["empty_artifacts"], [])
        self.assertTrue(checked["checks"]["seal"]["valid"], checked["checks"]["seal"])
        self.assertFalse(rechecked["ok"], rechecked["checks"])
        self.assertFalse(rechecked["sealed"], rechecked["checks"])
        mismatch = rechecked["checks"]["ledger_artifact_mismatches"][0]
        self.assertEqual(mismatch["artifact"], "scientific_graph.json")
        self.assertEqual(mismatch["reason"], "content_mismatch")
        self.assertEqual(clean_gate, 0)
        self.assertEqual(tampered_gate, 1)


if __name__ == "__main__":
    unittest.main()
