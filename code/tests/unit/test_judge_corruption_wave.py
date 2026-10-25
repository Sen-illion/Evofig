"""Tests for the controlled-corruption source adapter and its local wave."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from rfs.analysis.paper import ocr as ocr_module
from rfs.judge_training.local_corruptions import _ocr_reader_can_cross_processes, generate_region_candidates
from rfs.judge_training.local_region_reviewer import screen_regions_geometrically
from rfs.judge_training.corruption_wave import (
    CONTRACTS_DIRECTORY,
    FIGURES_FILE,
    RAW_DIRECTORY,
    SOURCE_MANIFEST_FILE,
    build_corruption_source,
    corruption_contract,
    run_local_corruption_wave,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


INVENTORY = {
    "figure_goal": "Pair an encoder with a decoder.",
    "inputs": [{"id": "input_text", "label": "Encoder input text", "role": "input", "detail_level": "artifact", "must_show": False, "evidence_ids": ["E1"]}],
    "modules": [
        {"id": "encoder", "label": "Encoder", "role": "module", "detail_level": "module", "evidence_ids": ["E2"]},
        {"id": "decoder", "label": "Decoder", "role": "module", "detail_level": "module", "evidence_ids": ["E3"]},
    ],
    "outputs": [],
    "innovations": [],
    "stages": [
        {"id": "stage_train", "label": "Training", "role": "stage", "seed_statements": ["Train the encoder end to end."]},
        {"id": "stage_infer", "label": "Inference", "role": "stage", "seed_statements": ["Run inference with the frozen decoder."]},
    ],
    "relations": [
        {"id": "rel_feeds", "source": "encoder", "target": "decoder", "type": "causal", "label": "feeds", "evidence_ids": ["E4"]},
        {"id": "rel_phantom", "source": "encoder", "target": "phantom", "type": "causal"},
    ],
    "forbidden_inventions": ["Do not invent a contrastive loss."],
    "uncertainties": [],
}

REVIEW = {
    "summary": "A two module paper.",
    "paper_identity": {"title": "Encoder and decoder"},
    "concepts": [
        {"evidence_refs": [
            {"evidence_id": "E1", "page": 1, "quote": "quote one", "section": "Introduction"},
            {"evidence_id": "E1", "page": 1, "quote": "quote one", "section": "Introduction"},
        ]},
        {"evidence_refs": [{"evidence_id": "E2", "page": 3, "quote": "quote two", "section": "Method"}]},
    ],
}


def _figure_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 60, 24), fill=(30, 90, 200))
    draw.rectangle((10, 40, 60, 54), fill=(200, 60, 40))
    image.save(path)
    return path


def _middleware_version(root, *, papers=("10.1234/example",), figure_for=("10.1234/example",), inventory=True):
    """Write a small middleware version holding the graph contracts and the figures."""
    graphs = root / "graphs"
    figures = root / "figures"
    index_rows = []
    for paper_id in papers:
        name = paper_id.replace("/", "_")
        if inventory:
            _write_json(graphs / name / "figure_specification.json", INVENTORY)
            _write_json(graphs / name / "paper_review.json", REVIEW)
        else:
            (graphs / name).mkdir(parents=True, exist_ok=True)
        if paper_id in figure_for:
            _figure_image(figures / name / "asset-abc.png")
        index_rows.append({"paper_id": paper_id, "metadata": {"split": "validation", "domain": "rag"}})
    _write_jsonl(root / "graph_index.jsonl", index_rows)
    return root


class CorruptionContractTests(unittest.TestCase):
    def test_contract_states_the_concepts_without_the_selection(self):
        contract = corruption_contract(INVENTORY, REVIEW, paper_id="p1")
        self.assertEqual([row["entity_id"] for row in contract["entities"]], ["input_text", "encoder", "decoder"])
        self.assertEqual(contract["entities"][0]["name"], "Encoder input text")
        self.assertEqual(contract["expected_figure_content"]["required_entities"], ["input_text", "encoder", "decoder"])
        self.assertEqual(contract["expected_figure_content"]["optional_entities"], [])
        self.assertEqual(contract["expected_figure_content"]["must_not_invent"], ["Do not invent a contrastive loss."])
        self.assertEqual(contract["expected_figure_content"]["required_relations"], ["encoder->decoder"])
        self.assertEqual([row["source"] for row in contract["relations"]], ["encoder"])
        self.assertEqual(contract["method_summary"], "Pair an encoder with a decoder.")
        self.assertTrue(contract["provenance"]["selection_keys_stripped"])
        self.assertEqual(contract["provenance"]["relations_dropped_unresolved"], 1)

    def test_flows_split_the_stages_and_keep_the_page_quotes_once(self):
        contract = corruption_contract(INVENTORY, REVIEW, paper_id="p1")
        self.assertEqual(contract["provenance"]["flows_route"], "stages_keyword_split")
        self.assertEqual([row["step_description"] for row in contract["training_flow"]], ["Training \u2014 Train the encoder end to end."])
        self.assertEqual([row["step_description"] for row in contract["inference_flow"]], ["Inference \u2014 Run inference with the frozen decoder."])
        self.assertEqual([row["evidence_id"] for row in contract["evidence_spans"]], ["E1", "E2"])
        self.assertEqual(contract["evidence_spans"][0]["page"], 1)

    def test_review_workflows_win_over_the_stage_words(self):
        review = {"workflows": {"training": [{"label": "Offline fit"}], "inference": [{"label": "Online evaluation"}]}}
        contract = corruption_contract(INVENTORY, review, paper_id="p1")
        self.assertEqual(contract["provenance"]["flows_route"], "review_workflows")
        self.assertEqual([row["step_description"] for row in contract["training_flow"]], ["Offline fit"])
        self.assertEqual([row["step_description"] for row in contract["inference_flow"]], ["Online evaluation"])

    def test_an_absent_inventory_states_nothing(self):
        contract = corruption_contract({}, {}, paper_id="p1")
        self.assertEqual(contract["entities"], [])
        self.assertEqual(contract["expected_figure_content"]["required_entities"], [])
        self.assertEqual(contract["provenance"]["flows_route"], "no_stages")


class CorruptionSourceTests(unittest.TestCase):
    def test_source_writes_figures_contracts_and_raw_titles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = _middleware_version(root / "version")
            manifest = build_corruption_source(version, root / "source")
            figures = _read_jsonl(root / "source" / FIGURES_FILE)
            self.assertTrue(manifest["ok"])
            self.assertEqual(manifest["figures"], 1)
            self.assertEqual(manifest["failures"], [])
            self.assertEqual(manifest["splits"]["validation"], 1)
            self.assertEqual(manifest["domains"], {"rag": 1})
            self.assertEqual(len(figures), 1)
            self.assertEqual(figures[0]["figure_id"], "10.1234_example__reference")
            self.assertEqual(figures[0]["paper_id"], "10.1234_example")
            self.assertEqual(figures[0]["source_paper_id"], "10.1234/example")
            self.assertEqual(figures[0]["split"], "validation")
            self.assertEqual(figures[0]["final_label"], "\u597d\uff08\u79d1\u7814\uff09")
            self.assertEqual(len(figures[0]["image_sha256"]), 64)
            self.assertTrue(str(figures[0]["figure_path"]).endswith("asset-abc.png"))
            contract = json.loads((root / "source" / CONTRACTS_DIRECTORY / "10.1234_example.json").read_text(encoding="utf-8"))
            self.assertEqual(contract["expected_figure_content"]["required_entities"], ["input_text", "encoder", "decoder"])
            raw = json.loads((root / "source" / RAW_DIRECTORY / "10.1234_example.json").read_text(encoding="utf-8"))
            self.assertEqual(raw["title"], "Encoder and decoder")
            self.assertEqual(raw["source_paper_id"], "10.1234/example")
            self.assertTrue((root / "source" / SOURCE_MANIFEST_FILE).is_file())

    def test_source_reports_a_paper_without_a_rendered_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = _middleware_version(root / "version", papers=("10.1234/example", "10.1234/other"), figure_for=("10.1234/example",))
            manifest = build_corruption_source(version, root / "source")
            self.assertFalse(manifest["ok"])
            self.assertEqual(manifest["figures"], 1)
            self.assertEqual(len(manifest["failures"]), 1)
            self.assertIn("no reference figure", manifest["failures"][0]["error"])

    def test_source_refuses_a_label_the_planner_would_gate_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = _middleware_version(root / "version")
            with self.assertRaises(ValueError):
                build_corruption_source(version, root / "source", label="\u574f")


    def test_source_takes_the_split_from_the_frozen_ledger_before_any_index_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = _middleware_version(root / "version")
            (version / "graph_index.jsonl").unlink()
            ledger = root / "pilot_ledger.jsonl"
            _write_jsonl(ledger, [{"paper_id": "10.1234/example", "split": "validation", "domain": "rag", "title": "Encoder and decoder"}])
            manifest = build_corruption_source(version, root / "source", split_ledger=ledger)
            self.assertTrue(manifest["ok"], manifest["failures"])
            self.assertEqual(manifest["splits"]["validation"], 1)
            self.assertEqual(manifest["domains"], {"rag": 1})
            self.assertEqual(manifest["records"][0]["title_source"], "ledger")
            figures = _read_jsonl(root / "source" / FIGURES_FILE)
            self.assertEqual(figures[0]["split"], "validation")
            self.assertEqual(figures[0]["source_paper_id"], "10.1234/example")

    def test_source_reports_a_paper_whose_split_it_cannot_know(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = _middleware_version(root / "version")
            (version / "graph_index.jsonl").unlink()
            _write_json(version / "graphs" / "10.1234_example" / "scientific_graph.json", {"paper_id": "10.1234/example", "nodes": [], "edges": []})
            manifest = build_corruption_source(version, root / "source")
            self.assertFalse(manifest["ok"])
            self.assertEqual(manifest["figures"], 0)
            self.assertIn("no split assignment", manifest["failures"][0]["error"])

    def test_source_reports_a_directory_that_names_no_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = _middleware_version(root / "version")
            (version / "graph_index.jsonl").unlink()
            (version / "graphs" / "10.1234_example" / "paper_review.json").unlink()
            manifest = build_corruption_source(version, root / "source")
            self.assertFalse(manifest["ok"])
            self.assertIn("names no paper identity", manifest["failures"][0]["error"])


class LocalCorruptionWaveTests(unittest.TestCase):
    def _fake_ocr(self, *_args, **_kwargs):
        return [
            {"text": "encoder", "confidence": 0.99, "quad": [[10, 10], [60, 10], [60, 24], [10, 24]]},
            {"text": "decoder", "confidence": 0.98, "quad": [[10, 40], [60, 40], [60, 54], [10, 54]]},
        ]

    def test_wave_compiles_examples_with_the_geometry_screen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = _middleware_version(root / "version")
            build_corruption_source(version, root / "source")
            original = ocr_module.run_rapidocr
            ocr_module.run_rapidocr = self._fake_ocr
            try:
                report = run_local_corruption_wave(root / "source", root / "wave", review="geometry", ocr_engine="rapidocr", per_type=2)
            finally:
                ocr_module.run_rapidocr = original
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["review"], "geometry")
            reviews = _read_jsonl(root / "wave" / "candidates" / "region_reviews.jsonl")
            self.assertTrue(reviews)
            self.assertEqual({row["review_method"] for row in reviews}, {"geometry_screen"})
            self.assertEqual({row["reviewer"] for row in reviews}, {None})
            examples = _read_jsonl(root / "wave" / "examples" / "local_corrupted_examples.jsonl")
            self.assertTrue(examples)
            tasks = {row["task_type"] for row in examples}
            self.assertLessEqual({"scientific_quality", "error_audit", "pairwise_preference"}, tasks)
            self.assertEqual({row["provenance"]["source"] for row in examples}, {"deterministic_local_corruption"})
            self.assertEqual(report["stages"]["candidates"]["counts"]["correct_but_ugly"], 1)
            self.assertEqual({row["provenance"]["human_audited"] for row in examples}, {False})

    def test_every_example_names_the_figure_the_corruption_was_made_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = _middleware_version(root / "version")
            build_corruption_source(version, root / "source")
            original = str(Path(_read_jsonl(root / "source" / FIGURES_FILE)[0]["figure_path"]).resolve())
            original_run = ocr_module.run_rapidocr
            ocr_module.run_rapidocr = self._fake_ocr
            try:
                run_local_corruption_wave(root / "source", root / "wave", review="geometry", ocr_engine="rapidocr", per_type=2)
            finally:
                ocr_module.run_rapidocr = original_run
            examples = _read_jsonl(root / "wave" / "examples" / "local_corrupted_examples.jsonl")

        self.assertTrue(examples)
        for row in examples:
            # A reviewer has to see what was changed, so every row names the untouched figure.
            self.assertEqual(str(row.get("preview") or ""), original, row["example_id"])
            if row["task_type"] in {"scientific_quality", "error_audit"}:
                self.assertNotEqual(row["candidate_a"], original)
            else:
                self.assertEqual(row["candidate_a"], original)

    def test_wave_fails_loudly_when_the_ocr_engine_cannot_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = _middleware_version(root / "version")
            build_corruption_source(version, root / "source")
            original = ocr_module.run_rapidocr

            def broken(*_args, **_kwargs):
                raise RuntimeError("no engine installed")

            ocr_module.run_rapidocr = broken
            try:
                with self.assertRaises(RuntimeError) as caught:
                    run_local_corruption_wave(root / "source", root / "wave", review="geometry", ocr_engine="rapidocr")
            finally:
                ocr_module.run_rapidocr = original
            self.assertIn("rapidocr", str(caught.exception))

    def test_wave_rejects_an_unknown_review_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = _middleware_version(root / "version")
            build_corruption_source(version, root / "source")
            with self.assertRaises(ValueError):
                run_local_corruption_wave(root / "source", root / "wave", review="manual")

def _stub_ocr(image_path, lang):
    """A module-level reader a worker process can import back, unlike a bound method."""
    return [
        {"text": "encoder", "confidence": 0.99, "quad": [[10, 10], [60, 10], [60, 24], [10, 24]]},
        {"text": "decoder", "confidence": 0.98, "quad": [[10, 40], [60, 40], [60, 54], [10, 54]]},
    ]


class ParallelRegionOcrTests(unittest.TestCase):
    """The reader may run in worker processes, but the candidates must not change."""

    def _source(self, root, papers):
        version = _middleware_version(root / "version", papers=papers, figure_for=papers)
        build_corruption_source(version, root / "source")
        return root / "source"

    def test_a_worker_pool_does_not_change_the_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root, ("10.1234/one", "10.1234/two"))
            serial = generate_region_candidates(source / FIGURES_FILE, source / CONTRACTS_DIRECTORY, root / "serial.jsonl", per_type=4, ocr_function=_stub_ocr, workers=1)
            parallel = generate_region_candidates(source / FIGURES_FILE, source / CONTRACTS_DIRECTORY, root / "parallel.jsonl", per_type=4, ocr_function=_stub_ocr, workers=2)
            self.assertEqual(serial["counts"], parallel["counts"])
            self.assertEqual(serial["candidates"], parallel["candidates"])
            self.assertEqual((root / "serial.jsonl").read_bytes(), (root / "parallel.jsonl").read_bytes())

    def test_only_a_module_level_reader_may_cross_processes(self):
        class Reader:
            def read(self, *_args, **_kwargs):
                return []

        self.assertFalse(_ocr_reader_can_cross_processes(Reader().read))
        self.assertFalse(_ocr_reader_can_cross_processes(lambda *_args, **_kwargs: []))
        self.assertTrue(_ocr_reader_can_cross_processes(_stub_ocr))


class GeometryScreenTests(unittest.TestCase):
    def _rows(self, base):
        return [
            {**base, "candidate_id": "c1", "corruption_type": "terminology_corruption", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.1}, "localization_confidence": 0.4},
            {**base, "candidate_id": "c2", "corruption_type": "terminology_corruption", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.9, "h": 0.9}, "localization_confidence": 0.9},
            {**base, "candidate_id": "c3", "corruption_type": "terminology_corruption", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.1}, "localization_confidence": 0.0},
            {**base, "candidate_id": "c4", "corruption_type": "correct_but_ugly", "bbox_percent": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}, "localization_confidence": 1.0},
        ]

    def test_screen_keeps_what_it_can_vouch_for_and_drops_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {"paper_id": "p1", "figure_id": "f1", "split": "train", "final_label": "\u597d\uff08\u79d1\u7814\uff09", "source_image": "figure.png", "target_id": "encoder", "region_source": "ocr"}
            candidates = root / "candidates.jsonl"
            _write_jsonl(candidates, self._rows(base))
            reviews_path = root / "reviews.jsonl"
            report = screen_regions_geometrically(candidates, reviews_path)
            reviews = {row["candidate_id"]: row for row in _read_jsonl(reviews_path)}
            self.assertEqual(reviews["c1"]["review_status"], "approved")
            self.assertEqual(reviews["c2"]["review_status"], "rejected")
            self.assertEqual(reviews["c3"]["review_status"], "rejected")
            self.assertEqual(reviews["c4"]["review_status"], "approved")
            self.assertIsNone(reviews["c1"]["reviewer"])
            self.assertEqual({row["review_method"] for row in reviews.values()}, {"geometry_screen"})
            self.assertEqual(report["approved"], 2)
            self.assertEqual(report["reviewed_this_run"], 4)

    def test_screen_resumes_without_rewriting_what_it_already_decided(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {"paper_id": "p1", "figure_id": "f1", "split": "train", "final_label": "\u597d\uff08\u79d1\u7814\uff09", "source_image": "figure.png", "target_id": "encoder", "region_source": "ocr"}
            candidates = root / "candidates.jsonl"
            _write_jsonl(candidates, self._rows(base))
            reviews_path = root / "reviews.jsonl"
            screen_regions_geometrically(candidates, reviews_path)
            report = screen_regions_geometrically(candidates, reviews_path)
            self.assertEqual(report["reviewed_this_run"], 0)
            self.assertEqual(report["approved"], 2)


if __name__ == "__main__":
    unittest.main()
