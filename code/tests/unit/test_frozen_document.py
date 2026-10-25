"""Unit tests for rebuilding frozen snapshot papers without re-parsing a PDF."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.contracts import validate_method_figure_content

from rfs.judge_training.frozen_document import (
    FROZEN_SPECIFICATION_VERSION,
    build_frozen_document,
    build_frozen_evidence_map,
    build_frozen_figure_specification,
    frozen_document_stats,
    looks_like_heading,
    paper_directory,
    split_flat_document,
)


def write_snapshot(root, *, layout="blocks", duplicate_records=False):
    """Write a minimal frozen snapshot carrying one paper."""
    snapshot = root / "paper-d0-v001"
    paper_dir = snapshot / "assets" / "paper-aaaa"
    paper_dir.mkdir(parents=True)
    (snapshot / "snapshot_manifest.json").write_text(json.dumps({"snapshot_id": "paper-d0-v001"}), encoding="utf-8")
    if layout == "blocks":
        records = [
            {"page": 1, "text": "A Frozen Study of Things\nJane Doe\nAbstract\nWe propose a frozen pipeline."},
            {"page": 2, "text": "1 Introduction\nFrozen snapshots matter."},
            {"page": 3, "text": "3 Method\nOur encoder feeds the decoder."},
        ]
        full_text = "\n\n".join(record["text"] for record in records)
    else:
        full_text = (
            "A Frozen Study of Things Download PDF Abstract Frozen snapshots matter a lot for review work. "
            "Introduction We build on frozen records instead of the live database. "
            "Methods We split the flat document at section headings. "
            "References Smith et al. 2024."
        )
        records = [{
            "char_start": 0,
            "char_end": len(full_text),
            "kind": "source_document",
            "pdf_page": None,
            "precision": "whole_document_only",
            "source_location": {"source_path": "collected/paper.pdf"},
            "text_asset_id": "asset-flat",
        }]
    stream = json.dumps(records[0], ensure_ascii=False)
    for record in records[1:-1]:
        stream += json.dumps(record, ensure_ascii=False)
    if len(records) > 1:
        stream += "\n" + json.dumps(records[-1], ensure_ascii=False, indent=2) + "\n"
    if duplicate_records:
        stream += json.dumps(records[0], ensure_ascii=False)
    (paper_dir / "text_locations.jsonl").write_text(stream, encoding="utf-8")
    (paper_dir / "full_text.txt").write_text(full_text, encoding="utf-8")
    (paper_dir / "captions.json").write_text(json.dumps({
        "paper_id": "p0001",
        "figures": [{"figure_id": "figure-1", "caption": "Overview of the frozen framework.", "figure_label": "Figure 1", "pdf_page": 2}],
    }, ensure_ascii=False), encoding="utf-8")
    (paper_dir / "bindings.json").write_text(json.dumps({"paper_id": "p0001", "bindings": []}), encoding="utf-8")
    paper = {
        "paper_id": "p0001",
        "title": "A Frozen Study of Things",
        "primary_domain": "cv",
        "captions_path": "assets/paper-aaaa/captions.json",
        "bindings_path": "assets/paper-aaaa/bindings.json",
    }
    with (snapshot / "eligible_papers.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
    return snapshot, paper


class FrozenDocumentTests(unittest.TestCase):
    def test_block_records_build_evidence_with_sections(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, paper = write_snapshot(Path(name))
            parsed = build_frozen_document(snapshot, paper)

        evidence = parsed["evidence"]
        self.assertEqual(len(evidence), 4)
        self.assertEqual([item["page"] for item in evidence], [1, 2, 3, 2])
        self.assertEqual([item["section_hint"] for item in evidence], ["Abstract", "Introduction", "Method", "Method"])
        self.assertEqual(evidence[-1]["kind"], "caption")
        self.assertEqual(parsed["headings"], ["Abstract", "Introduction", "Method"])
        self.assertEqual(len(parsed["document_index"]["figures"]), 1)
        self.assertEqual(parsed["page_count"], 3)
        self.assertTrue(parsed["evidence"][0]["id"].startswith("E_P001_"))

    def test_streamed_records_read_concatenated_and_multiline_json(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, paper = write_snapshot(Path(name))
            parsed = build_frozen_document(snapshot, paper)
        self.assertEqual([item["text"] for item in parsed["evidence"][:3]], [
            "A Frozen Study of Things Jane Doe Abstract We propose a frozen pipeline.",
            "1 Introduction Frozen snapshots matter.",
            "3 Method Our encoder feeds the decoder.",
        ])

    def test_flat_document_is_split_by_section_headings(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, paper = write_snapshot(Path(name), layout="flat")
            parsed = build_frozen_document(snapshot, paper)

        sections = [item["section_hint"] for item in parsed["evidence"]]
        self.assertIn("Abstract", sections)
        self.assertIn("Introduction", sections)
        self.assertIn("Methods", sections)
        self.assertGreaterEqual(len(parsed["evidence"]), 5)
        body = [item for item in parsed["evidence"] if item["kind"] != "caption"]
        self.assertTrue(body and all(item["page"] == 0 for item in body))
        captions = [item for item in parsed["evidence"] if item["kind"] == "caption"]
        self.assertEqual([item["page"] for item in captions], [2])
        self.assertTrue(any("flat source document" in warning for warning in parsed["extraction_report"]["warnings"]))
        self.assertFalse(any("no evidence" in warning for warning in parsed["extraction_report"]["warnings"]))

    def test_split_flat_document_stops_at_references(self):
        blocks = split_flat_document("Abstract One. Introduction Two. Methods Three. References Smith 2024.")
        self.assertEqual([block["section"] for block in blocks], ["Abstract", "Introduction", "Methods", "References"])
        self.assertTrue(blocks[-1]["text"].startswith("References"))

    def test_reading_is_deterministic_and_ids_stay_unique(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, paper = write_snapshot(Path(name))
            first = build_frozen_document(snapshot, paper)
            second = build_frozen_document(snapshot, paper)

        self.assertEqual(first["evidence"], second["evidence"])
        stats = frozen_document_stats(first)
        self.assertEqual(stats["evidence_count"], stats["unique_evidence_ids"])
        self.assertEqual(stats["duplicate_evidence_ids"], 0)

    def test_duplicate_records_get_disambiguated_ids(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, paper = write_snapshot(Path(name), duplicate_records=True)
            parsed = build_frozen_document(snapshot, paper)

        stats = frozen_document_stats(parsed)
        self.assertEqual(stats["duplicate_evidence_ids"], 0)
        self.assertEqual(stats["unique_evidence_ids"], stats["evidence_count"])
        first = parsed["evidence"][0]
        repeated = parsed["evidence"][3]
        self.assertEqual(repeated["text"], first["text"])
        self.assertEqual(repeated["id"], first["id"] + "_2")

    def test_missing_frozen_file_raises(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, paper = write_snapshot(Path(name))
            (snapshot / "assets" / "paper-aaaa" / "text_locations.jsonl").unlink()
            with self.assertRaises(FileNotFoundError):
                build_frozen_document(snapshot, paper)

    def test_paper_directory_refuses_to_escape_the_snapshot(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, _paper = write_snapshot(Path(name))
            with self.assertRaises(ValueError):
                paper_directory(snapshot, {"paper_id": "p0001", "captions_path": "../outside/captions.json"})

    def test_looks_like_heading_rules(self):
        self.assertEqual(looks_like_heading("3 Method"), "Method")
        self.assertEqual(looks_like_heading("Abstract:"), "Abstract")
        self.assertIsNone(looks_like_heading("Our method beats the baseline."))
        self.assertIsNone(looks_like_heading(""))
        self.assertIsNone(looks_like_heading("x" * 200))




REVIEW_SEED = {
    "schema_version": "2.0",
    "paper_identity": {"title": "A Frozen Study of Things"},
    "central_claims": [{
        "id": "c1", "statement": "Freezing beats re-parsing.", "evidence_ids": ["E0001"],
        "status": "required", "importance": "critical", "must_appear_in_figure": True,
    }],
    "inputs": [{"id": "in1", "statement": "Frozen text", "evidence_ids": ["E0001"], "must_appear_in_figure": True}],
    "modules": [{
        "id": "m1", "statement": "Frozen encoder", "evidence_ids": ["E0001"],
        "must_appear_in_figure": True, "importance": "high",
    }],
    "outputs": [{"id": "out1", "statement": "Framework figure", "evidence_ids": [], "must_appear_in_figure": True}],
    "innovations": [{"id": "i1", "statement": "Snapshot adapter", "evidence_ids": ["E0002"], "status": "unknown"}],
    "relations": [
        {"id": "r1", "source_id": "m1", "target_id": "out1", "relation_type": "data_flow", "statement": "produces", "evidence_ids": ["E0001"]},
        {"id": "r2", "source_id": "m1", "target_id": "ghost", "relation_type": "data_flow", "statement": "leaks", "evidence_ids": ["E0001"]},
    ],
    "workflows": {"training": [{"id": "w1", "statement": "Train the encoder", "evidence_ids": ["E0001"]}]},
    "forbidden_inventions": ["Do not invent modules."],
}


class FrozenSpecificationSeedTests(unittest.TestCase):
    def test_evidence_map_keeps_the_frozen_evidence_records(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, paper = write_snapshot(Path(name))
            parsed = build_frozen_document(snapshot, paper)
            evidence_map = build_frozen_evidence_map(parsed)

        self.assertEqual(evidence_map["loader"], "frozen-snapshot")
        self.assertTrue(evidence_map["source_path"])
        self.assertEqual([item["id"] for item in evidence_map["evidence"]], [item["id"] for item in parsed["evidence"]])

    def test_seed_keeps_review_facts_and_drops_undeclared_relations(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, paper = write_snapshot(Path(name))
            parsed = build_frozen_document(snapshot, paper)
            specification = build_frozen_figure_specification(
                parsed, REVIEW_SEED, paper_id="p0001", domain="cv", review_mode="vlm"
            )

        self.assertEqual(specification["figure_goal"], "Freezing beats re-parsing.")
        self.assertEqual([item["id"] for item in specification["modules"]], ["m1"])
        self.assertEqual(specification["modules"][0]["label"], "Frozen encoder")
        self.assertEqual(specification["modules"][0]["detail_level"], "module")
        self.assertEqual(specification["modules"][0]["scientific_importance"], 0.75)
        self.assertTrue(specification["modules"][0]["must_show"])
        self.assertTrue(specification["inputs"][0]["should_render"])
        self.assertFalse(specification["outputs"][0]["should_render"])
        self.assertEqual([item["id"] for item in specification["relations"]], ["r1"])
        self.assertEqual(specification["relations"][0]["source"], "m1")
        self.assertEqual([stage["id"] for stage in specification["stages"]], ["stage_training"])
        self.assertEqual(specification["stages"][0]["label"], "Training")
        self.assertEqual(specification["stages"][0]["evidence_ids"], ["E0001"])
        self.assertEqual(specification["uncertainties"], ["Snapshot adapter"])
        self.assertEqual(specification["forbidden_inventions"], ["Do not invent modules."])
        seed = specification["seed"]
        self.assertEqual(seed["specification_version"], FROZEN_SPECIFICATION_VERSION)
        self.assertEqual(seed["source"], "pilot-review-seed")
        self.assertEqual(seed["review_mode"], "vlm")
        self.assertEqual(seed["dropped_relations"], [{
            "id": "r2", "source": "m1", "target": "ghost", "reason": "undeclared_endpoint",
        }])
        self.assertGreater(seed["evidence_count"], 0)
        validation = validate_method_figure_content(specification)
        self.assertTrue(validation["ok"], validation["errors"])

    def test_seed_without_a_review_still_produces_a_valid_goal(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, paper = write_snapshot(Path(name))
            parsed = build_frozen_document(snapshot, paper)
            specification = build_frozen_figure_specification(parsed, None)

        self.assertEqual(specification["figure_goal"], "Explain the paper method.")
        self.assertEqual(specification["modules"], [])
        self.assertEqual(specification["seed"]["declared_entities"], 0)

if __name__ == "__main__":
    unittest.main()
