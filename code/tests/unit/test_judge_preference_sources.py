"""Tests for the read-only Judge preference-pair source ports."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.judge_training.corruption_generator import CORRUPTION_TYPES
from rfs.judge_training.preference_sources import (
    SourceUnavailable,
    defect_distribution,
    open_corruption_source,
    open_generation_source,
    open_middleware_source,
    run_directory_candidates,
    write_source_manifest,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _middleware_version(root):
    root.mkdir(parents=True, exist_ok=True)
    folder = root / "graphs" / "10.1234_example"
    folder.mkdir(parents=True)
    (folder / "scientific_graph.json").write_text(json.dumps({"graph_id": "sg", "paper_id": "2401.00001", "nodes": [], "edges": []}), encoding="utf-8")
    (folder / "visual_graph.json").write_text(json.dumps({"graph_id": "vg", "paper_id": "2401.00001", "nodes": [], "edges": []}), encoding="utf-8")
    (folder / "method_figure_content.json").write_text(json.dumps({"figure_goal": "g"}), encoding="utf-8")
    _write_jsonl(root / "graph_index.jsonl", [
        {"record_id": "r1", "paper_id": "2401.00001", "domain": "cv", "metadata": {"source_graph": "graphs/10.1234_example/scientific_graph.json"}},
        {"record_id": "r2", "paper_id": "2401.00001", "domain": "cv", "metadata": {"source_graph": "graphs/10.1234_example/visual_graph.json"}},
    ])
    _write_jsonl(root / "subgraph_index.jsonl", [{"subgraph_id": "s1", "paper_id": "2401.00001", "motif": "linear"}])
    return root


def _generation_run(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "scientific_graph.json").write_text(json.dumps({"graph_id": "sg"}), encoding="utf-8")
    (root / "evolution_rounds.json").write_text(json.dumps({"rounds": [{"round": 0, "selected_variant": 0, "domain": "cv"}]}), encoding="utf-8")
    (root / "evolution_events.jsonl").write_text(json.dumps({"seq": 1, "event": "candidate_accepted", "round": 0}) + "\n", encoding="utf-8")
    candidate = root / "round_00" / "candidate_00"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "draft_preview.png").write_bytes(b"png")
    return root


def _corruption_source(root):
    root.mkdir(parents=True, exist_ok=True)
    image = root / "figure.png"
    image.write_bytes(b"png")
    _write_jsonl(root / "figures.jsonl", [{"figure_id": "f1", "paper_id": "2401.00001", "figure_path": str(image), "split": "train", "final_label": "好（科研）"}])
    contract = root / "structured" / "2401.00001.json"
    contract.parent.mkdir(parents=True)
    contract.write_text(json.dumps({
        "expected_figure_content": {"required_entities": ["encoder"], "required_relations": ["encoder->head"], "must_not_invent": ["extra"]},
        "entities": [{"entity_id": "encoder"}],
        "relations": [{"source": "encoder", "target": "head"}],
        "training_flow": [{"step_name": "train"}],
        "inference_flow": [{"step_name": "infer"}],
    }), encoding="utf-8")
    _write_jsonl(root / "graph_ledger.jsonl", [{"paper_id": "2401.00001", "domain": "cv"}])
    return root / "figures.jsonl", root / "structured", root / "graph_ledger.jsonl"


class MiddlewareSourceTests(unittest.TestCase):
    def test_manifest_reports_index_domains_and_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = open_middleware_source(_middleware_version(Path(tmp) / "version"))
            manifest = source.manifest()
            self.assertTrue(manifest["ok"])
            self.assertEqual({"cv": 2}, manifest["domain_distribution"])
            self.assertEqual({}, manifest["defect_distribution"])
            self.assertEqual(2, manifest["counts"]["index_records"])
            self.assertEqual(1, manifest["counts"]["paper_folders"])
            self.assertEqual(("2401.00001", "10.1234_example", "recorded_paper_id"), source.folder_for_run(paper_id="2401.00001"))
            self.assertEqual("sg", source.layer_graph("10.1234_example")["graph_id"])
            inventory = source.method_inventory("10.1234_example")
            self.assertEqual("g", inventory["figure_goal"])
            self.assertEqual([], inventory["modules"])

    def test_method_inventory_strips_what_the_figure_chose_to_draw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "version"
            _middleware_version(root)
            folder = root / "graphs" / "10.1234_example"
            (folder / "figure_specification.json").write_text(json.dumps({
                "figure_goal": "g",
                "modules": [{
                    "id": "m1", "label": "encoder", "role": "module", "detail_level": "module",
                    "evidence_ids": ["E1"], "scientific_importance": 1.0, "visualization_value": 1.0,
                    "must_show": True, "should_render": True, "render_priority": 1,
                }],
                "relations": [{"id": "r1", "source": "m1", "target": "m2", "type": "causal", "must_show": True}],
                "seed": {"paper_id": "2401.00001"},
            }), encoding="utf-8")
            source = open_middleware_source(root)

            inventory = source.method_inventory("10.1234_example")

            self.assertEqual("g", inventory["figure_goal"])
            self.assertEqual(["E1"], inventory["modules"][0]["evidence_ids"])
            self.assertNotIn("seed", inventory)
            self.assertEqual(
                ["id", "source", "target", "type"], [name for name in inventory["relations"][0]]
            )

    def test_missing_index_or_folder_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "version"
            _middleware_version(root)
            (root / "subgraph_index.jsonl").unlink()
            with self.assertRaises(SourceUnavailable):
                open_middleware_source(root)
            (root / "subgraph_index.jsonl").write_text("", encoding="utf-8")
            source = open_middleware_source(root)
            with self.assertRaises(SourceUnavailable):
                source.paper_dir("missing")
            with self.assertRaises(SourceUnavailable):
                source.layer_graph("10.1234_example", "editable")

class RunIdentityTests(unittest.TestCase):
    """The generation line drops the paper id, so the run's own path has to supply it."""

    def test_the_recorded_id_wins_when_the_run_has_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = open_middleware_source(_middleware_version(Path(tmp) / "version"))
            self.assertEqual(("2401.00001", "10.1234_example", "recorded_paper_id"), source.folder_for_run(paper_id="2401.00001", run_directories=["something-else"]))

    def test_a_source_label_falls_back_to_the_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = open_middleware_source(_middleware_version(Path(tmp) / "version"))
            resolved = source.folder_for_run(paper_id="arxiv", run_directories=["10.1234_example"])
            self.assertEqual(("2401.00001", "10.1234_example", "run_directory"), resolved)

    def test_a_run_directory_matches_the_sanitized_id_of_a_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "version"
            _middleware_version(root)
            _write_jsonl(root / "graph_index.jsonl", [{"record_id": "r1", "paper_id": "10.5555/abc", "domain": "cv", "metadata": {"source_graph": "graphs/10.5555_abc/scientific_graph.json"}}])
            source = open_middleware_source(root)
            self.assertEqual(("10.5555/abc", "10.5555_abc", "run_directory"), source.folder_for_run(paper_id="official", run_directories=["10.5555_abc"]))

    def test_an_unknown_run_is_refused_with_every_candidate_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = open_middleware_source(_middleware_version(Path(tmp) / "version"))
            with self.assertRaises(SourceUnavailable) as raised:
                source.folder_for_run(paper_id="arxiv", run_directories=["no-such-paper"])
            self.assertIn("no-such-paper", str(raised.exception))
            self.assertIn("recorded_paper_id=arxiv", str(raised.exception))

    def test_the_run_directory_search_skips_attempt_and_container_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "corpus_wave_full" / "runs" / "10.1234_example" / "attempt_01"
            run.mkdir(parents=True)
            self.assertEqual(["10.1234_example"], run_directory_candidates(run))

    def test_a_run_directory_without_an_attempt_folder_is_itself_a_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "runs" / "10.1234_example"
            run.mkdir(parents=True)
            self.assertEqual(["10.1234_example"], run_directory_candidates(run, depth=2))

class GenerationRunSourceTests(unittest.TestCase):
    def test_manifest_lists_rounds_and_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _generation_run(Path(tmp) / "run")
            (run / "editable_composition.pptx").write_bytes(b"pptx")
            (run / "figure_program.json").write_text("{}", encoding="utf-8")
            source = open_generation_source(run)
            manifest = source.manifest()
            self.assertTrue(manifest["ok"])
            self.assertEqual(1, manifest["counts"]["candidates"])
            self.assertEqual({"cv": 1}, manifest["domain_distribution"])
            self.assertEqual(str(run / "round_00" / "candidate_00" / "draft_preview.png"), str(source.candidate_preview(0, 0)))
            self.assertEqual({"editable_composition.pptx", "figure_program.json"}, set(source.deliverables()))

    def test_missing_ledger_or_deliverable_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _generation_run(Path(tmp) / "run")
            (run / "evolution_rounds.json").unlink()
            with self.assertRaises(SourceUnavailable):
                open_generation_source(run)
            _generation_run(run)
            source = open_generation_source(run)
            with self.assertRaises(SourceUnavailable):
                source.deliverables()
            with self.assertRaises(SourceUnavailable):
                source.candidate_preview(0, 3)

    def test_candidate_without_preview_is_reported_and_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _generation_run(root)
            (root / "round_00" / "candidate_00" / "draft_preview.png").unlink()
            source = open_generation_source(root)
            self.assertFalse(source.manifest()["ok"])
            with self.assertRaises(SourceUnavailable):
                source.candidate_preview(0, 0)

    def test_run_without_any_candidate_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _generation_run(root)
            (root / "round_00" / "candidate_00" / "draft_preview.png").unlink()
            (root / "round_00" / "candidate_00").rmdir()
            with self.assertRaises(SourceUnavailable):
                open_generation_source(root)


class CorruptionSourceTests(unittest.TestCase):
    def test_planned_defects_split_into_gate_and_auxiliary(self):
        with tempfile.TemporaryDirectory() as tmp:
            figures, structured, ledger = _corruption_source(Path(tmp))
            source = open_corruption_source(figures, structured, domain_index=ledger)
            manifest = source.manifest()
            self.assertTrue(manifest["ok"])
            self.assertEqual(len(CORRUPTION_TYPES), manifest["counts"]["planned_corruptions"])
            self.assertEqual({"cv": 1}, manifest["domain_distribution"])
            self.assertEqual({"module_order_error": 1, "overlap": 1, "wrong_arrow_target": 1}, manifest["defect_distribution"]["gate"])
            self.assertNotIn("wrong_arrow_target", manifest["defect_distribution"]["auxiliary"])
            self.assertNotIn("module_order_error", manifest["defect_distribution"]["auxiliary"])
            self.assertEqual({}, manifest["defect_distribution"]["unsupported"])

    def test_missing_contract_or_manifest_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            figures, structured, _ = _corruption_source(Path(tmp))
            (structured / "2401.00001.json").unlink()
            with self.assertRaises(SourceUnavailable):
                open_corruption_source(figures, structured).planned_corruptions()
            with self.assertRaises(SourceUnavailable):
                open_corruption_source(Path(tmp) / "missing.jsonl", structured)
            with self.assertRaises(SourceUnavailable):
                open_corruption_source(figures, Path(tmp) / "missing")

    def test_domain_index_is_optional_but_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            figures, structured, _ = _corruption_source(Path(tmp))
            manifest = open_corruption_source(figures, structured).manifest()
            self.assertFalse(manifest["ok"])
            self.assertEqual({"unknown": 1}, manifest["domain_distribution"])


class DefectDistributionTests(unittest.TestCase):
    def test_unsupported_kinds_are_reported_not_hidden(self):
        distribution = defect_distribution(["relation_reversal", "not_a_defect"])
        self.assertEqual({"wrong_arrow_target": 1}, distribution["gate"])
        self.assertEqual({}, distribution["auxiliary"])
        self.assertEqual({"not_a_defect": 1}, distribution["unsupported"])


if __name__ == "__main__":
    unittest.main()


class SourceManifestCommandTests(unittest.TestCase):
    def test_writes_a_hashed_manifest_for_every_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            figures, structured, ledger = _corruption_source(root / "corruption")
            cases = (
                ("middleware", {"root": _middleware_version(root / "version")}),
                ("generation", {"root": _generation_run(root / "run")}),
                ("corruption", {"figures": figures, "structured": structured, "domain_index": ledger}),
            )
            for name, kwargs in cases:
                out = root / f"{name}.manifest.json"
                manifest = write_source_manifest(name, out, **kwargs)
                written = json.loads(out.read_text(encoding="utf-8"))

                self.assertEqual(len(manifest["sha256"]), 64)
                self.assertEqual(written["sha256"], manifest["sha256"])
                self.assertEqual(written["contract"], "rfs.judge_preference_source")
                self.assertTrue(written["domain_distribution"])
                expected_defects = ["auxiliary", "gate", "unsupported"] if name == "corruption" else []
                self.assertEqual(sorted(written["defect_distribution"]), expected_defects)

    def test_the_corruption_manifest_splits_the_gating_defects_from_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            figures, structured, ledger = _corruption_source(root / "corruption")

            manifest = write_source_manifest("corruption", root / "m.json", figures=figures, structured=structured, domain_index=ledger)

            self.assertEqual(sorted(manifest["defect_distribution"]["gate"]), ["module_order_error", "overlap", "wrong_arrow_target"])
            self.assertEqual(manifest["counts"]["planned_corruptions"], len(CORRUPTION_TYPES))

    def test_a_source_that_cannot_be_opened_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "m.json"
            with self.assertRaises(SourceUnavailable):
                write_source_manifest("middleware", out, root=root / "absent-version")
            with self.assertRaises(SourceUnavailable):
                write_source_manifest("generation", out, root=root / "absent-run")
            with self.assertRaises(SourceUnavailable):
                write_source_manifest("corruption", out, figures=None, structured=None)
            with self.assertRaises(SourceUnavailable):
                write_source_manifest("nonsense", out, root=root)
            self.assertFalse(out.exists())
