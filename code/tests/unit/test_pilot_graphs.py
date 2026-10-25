"""Unit tests for the Phase 3 Pilot graph middleware orchestration."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from rfs.analysis.paper.document import evidence_id

from rfs.judge_training.frozen_document import FROZEN_DOCUMENT_VERSION
from rfs.judge_training import pilot_bindings
from rfs.providers.embeddings import DEFAULT_DIMENSION, EmbeddingUnavailable
from rfs.utils import sanitize_surrogates, write_json
from rfs.judge_training.pilot_graphs import (
    ARTIFACT_SCHEMA_VERSION,
    BINDING_VERSION,
    CONTRACT_STEP,
    DEFAULT_GRAPH_ID,
    IMPLEMENTED_STAGES,
    PAPER_ARTIFACTS,
    PAPER_ARTIFACTS_BY_SCHEMA,
    bind_all,
    binding_coverage,
    build_pilot_graph_payloads,
    canonical_sha256,
    contract_all,
    contract_paper,
    expected_paper_artifacts,
    ledger_artifact_mismatches,
    load_pilot_ledger,
    load_resume_state,
    parse_stages,
    review_profile_for,
    safe_paper_directory,
    validate_pilot_graph_version,
    write_pilot_graph_version,
)
from rfs.providers.embeddings import EmbeddingUnavailable

PAPERS = (
    ("p0001", "cv", "train"),
    ("p0002", "biology", "validation"),
    ("p0003", "materials", "train"),
)


def write_fixture(root):
    """Write a frozen snapshot and a sealed Pilot version carrying three papers."""
    snapshot = root / "paper-d0-v001"
    pilot = root / "paper-d0-pilot-v002"
    snapshot.mkdir(parents=True)
    pilot.mkdir(parents=True)
    records = [
        {"page": 1, "text": "A Frozen Study\nAbstract\nWe propose a frozen pipeline."},
        {"page": 2, "text": "1 Introduction\nFrozen snapshots matter."},
        {"page": 3, "text": "3 Method\nOur encoder feeds the decoder."},
    ]
    full_text = "\n\n".join(record["text"] for record in records)
    ledger = []
    papers = []
    asset_rows = []
    for paper_id, domain, split in PAPERS:
        directory = snapshot / "assets" / f"paper-{paper_id}"
        directory.mkdir(parents=True)
        payload = json.dumps(records[0], ensure_ascii=False) + "\n" + json.dumps(records[1], ensure_ascii=False, indent=2) + "\n" + json.dumps(records[2], ensure_ascii=False)
        (directory / "text_locations.jsonl").write_text(payload, encoding="utf-8")
        (directory / "full_text.txt").write_text(full_text, encoding="utf-8")
        (directory / "captions.json").write_text(json.dumps({
            "paper_id": paper_id,
            "figures": [{"figure_id": f"figure-{paper_id}", "caption": "Overview of the frozen framework.", "pdf_page": 2}],
        }, ensure_ascii=False), encoding="utf-8")
        (directory / "bindings.json").write_text(json.dumps({"paper_id": paper_id, "bindings": []}), encoding="utf-8")
        figures = directory / "figures"
        figures.mkdir(parents=True)
        Image.new("RGB", (4, 4), (255, 255, 255)).save(figures / "figure.png")
        asset_rows.append({
            "paper_id": paper_id,
            "asset_id": f"asset-{paper_id}",
            "relative_path": f"assets/paper-{paper_id}/figures/figure.png",
            "sha256": "",
            "suffix": ".png",
            "role": "figure",
        })
        papers.append({
            "paper_id": paper_id,
            "title": f"Frozen paper {paper_id}",
            "primary_domain": domain,
            "captions_path": f"assets/paper-{paper_id}/captions.json",
            "bindings_path": f"assets/paper-{paper_id}/bindings.json",
        })
        ledger.append({
            "paper_id": paper_id,
            "split": split,
            "domain": domain,
            "figure_relative_path": f"assets/paper-{paper_id}/figures/figure.png",
            "figure_type": "framework",
            "structure": "linear",
            "complexity": 3,
        })
    with (snapshot / "asset_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in asset_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (snapshot / "eligible_papers.jsonl").open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
    (snapshot / "snapshot_manifest.json").write_text(json.dumps({"snapshot_id": "paper-d0-v001"}), encoding="utf-8")
    with (pilot / "pilot_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (pilot / "pilot_manifest.json").write_text(json.dumps({"pilot_id": "paper-d0-pilot-v002"}), encoding="utf-8")
    return snapshot, pilot


class UnencodableTextTests(unittest.TestCase):
    """A truncated astral character must not abort a paid stage."""

    def test_sanitize_replaces_lone_surrogates_only(self):
        self.assertEqual(sanitize_surrogates("a\ud835b"), "a\ufffdb")
        self.assertEqual(sanitize_surrogates("x\U0001D434y"), "x\U0001D434y")
        self.assertEqual(sanitize_surrogates({"k": ["v\ud835"]}), {"k": ["v\ufffd"]})

    def test_canonical_sha256_matches_a_written_artifact_with_a_lone_surrogate(self):
        payload = {"label": "half \ud835 symbol"}
        digest = canonical_sha256(payload)
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "artifact.json"
            write_json(path, payload)
            stored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(canonical_sha256(stored), digest)


class PilotGraphStageTests(unittest.TestCase):
    def test_stage_list_defaults_to_prepare(self):
        self.assertEqual(parse_stages(None), ("prepare",))
        self.assertEqual(parse_stages(""), ("prepare",))
        self.assertEqual(parse_stages("all"), ("prepare", "review", "contract", "graph", "visual", "check", "bind", "index"))
        self.assertEqual(parse_stages("prepare,review"), ("prepare", "review"))
        self.assertEqual(parse_stages("contract,graph"), ("contract", "graph"))
        self.assertEqual(parse_stages(["prepare", "prepare"]), ("prepare",))

    def test_unknown_stages_are_rejected_and_every_known_stage_runs(self):
        with self.assertRaises(ValueError):
            parse_stages("nonsense")
        self.assertEqual(parse_stages("index"), ("index",))
        self.assertEqual(set(parse_stages("all")), set(IMPLEMENTED_STAGES))

    def test_review_profile_mapping(self):
        self.assertEqual(review_profile_for("cv"), "ai-ml-method")
        self.assertEqual(review_profile_for("RAG"), "ai-ml-method")
        self.assertEqual(review_profile_for("agent"), "ai-ml-method")
        self.assertEqual(review_profile_for("materials"), "empirical-science")
        self.assertEqual(review_profile_for("biology"), "empirical-science")
        self.assertEqual(review_profile_for("unknown-domain"), "general")

    def test_safe_paper_directory_handles_reserved_characters(self):
        self.assertEqual(safe_paper_directory("iclr:2026:10006567"), "iclr_2026_10006567")
        self.assertEqual(safe_paper_directory("10.1038/s41467-026-68311-9"), "10.1038_s41467-026-68311-9")
        self.assertEqual(safe_paper_directory(""), "unknown-paper")


class PilotGraphPrepareTests(unittest.TestCase):
    def test_prepare_rebuilds_every_pilot_paper_without_provider_calls(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, pilot = write_fixture(Path(name))
            payloads = build_pilot_graph_payloads(snapshot, pilot)

        self.assertTrue(payloads["validation"]["ok"])
        self.assertEqual(len(payloads["reports"]), len(PAPERS))
        self.assertEqual(payloads["manifest"]["provider"]["calls"], 0)
        self.assertEqual(payloads["manifest"]["documents"]["document_version"], FROZEN_DOCUMENT_VERSION)
        self.assertFalse(payloads["complete"])
        self.assertEqual(payloads["stages_completed"], ["prepare"])
        self.assertLessEqual(set(payloads["stages_requested"]), set(payloads["stages_completed"]))
        coverage = payloads["coverage"]
        self.assertEqual(coverage["by_split"], {"train": 2, "validation": 1})
        self.assertEqual(coverage["by_domain"]["cv"]["train"], 1)
        self.assertEqual(coverage["hidden_test_papers"], 0)
        self.assertEqual(coverage["missing_captions"], [])
        self.assertEqual(coverage["empty_evidence"], [])
        self.assertTrue(payloads["manifest"]["inputs"]["pilot_ledger_sha256"])

    def test_prepare_rows_carry_profile_audit_and_usage_fields(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, pilot = write_fixture(Path(name))
            payloads = build_pilot_graph_payloads(snapshot, pilot)

        for report in payloads["reports"]:
            self.assertEqual(report["domain_profile"]["requested"], review_profile_for(report["domain"]))
            self.assertTrue(report["domain_profile"]["detected"])
            self.assertEqual(report["status"], "ok")
            self.assertGreater(report["review_context"]["prompt_chars"], 0)
        for row in payloads["ledger"]:
            self.assertEqual(row["token_source"], "not_called")
            self.assertEqual(row["attempts"], 0)
            self.assertIsNone(row["total_tokens"])
            self.assertTrue(row["artifact_sha256"]["graph_report.json"])

    def test_limit_only_prepares_the_first_papers(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, pilot = write_fixture(Path(name))
            payloads = build_pilot_graph_payloads(snapshot, pilot, limit=2)

        self.assertEqual(len(payloads["reports"]), 2)
        self.assertTrue(payloads["validation"]["ok"])

    def test_pilot_paper_missing_from_snapshot_raises(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot, pilot = write_fixture(Path(name))
            (snapshot / "eligible_papers.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_pilot_graph_payloads(snapshot, pilot)


class PilotGraphWriteTests(unittest.TestCase):
    def test_write_and_validate_round_trip(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = build_pilot_graph_payloads(snapshot, pilot)
            written = write_pilot_graph_version(root / "graphs", payloads)
            checked = validate_pilot_graph_version(root / "graphs")

        self.assertFalse(written["sealed"])
        self.assertIn("graph_manifest.json", written["files"])
        self.assertNotIn("COMPLETE", written["files"])
        self.assertTrue(checked["ok"])
        self.assertFalse(checked["complete"])
        self.assertEqual(checked["papers"], len(PAPERS))
        self.assertEqual(checked["stages_completed"], ["prepare"])
        self.assertEqual(checked["checks"]["reports_missing"], [])
        self.assertEqual(checked["graph_id"], DEFAULT_GRAPH_ID)

    def test_write_writes_one_report_per_paper(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = build_pilot_graph_payloads(snapshot, pilot)
            write_pilot_graph_version(root / "graphs", payloads)
            reports = sorted((root / "graphs" / "graphs").glob("*/graph_report.json"))
            report = json.loads(reports[0].read_text(encoding="utf-8")) if reports else {}

        self.assertEqual(len(reports), len(PAPERS))
        self.assertEqual(report["step"], "prepare")
        self.assertNotIn("source_path", json.dumps(report))

    def test_sealed_version_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = build_pilot_graph_payloads(snapshot, pilot)
            target = root / "graphs"
            write_pilot_graph_version(target, payloads)
            (target / "COMPLETE").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_pilot_graph_version(target, payloads)

    def test_validate_reports_missing_version(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(FileNotFoundError):
                validate_pilot_graph_version(Path(name) / "absent")

    def test_declared_artifact_lists_win_over_the_built_in_default(self):
        # The figure content plan is no longer part of the built-in contract list,
        # so a version that declares it is checked against the list it declares.
        declared = {"contract": ["figure_content_plan.json"]}
        self.assertEqual(
            expected_paper_artifacts({"paper_artifacts": declared}, ["prepare", "contract"]),
            ["figure_content_plan.json"],
        )

    def test_a_manifest_without_a_declared_list_falls_back_to_schema_one(self):
        legacy = list(PAPER_ARTIFACTS_BY_SCHEMA[1][CONTRACT_STEP])
        self.assertEqual(expected_paper_artifacts({}, ["prepare", "contract"]), legacy)
        self.assertEqual(expected_paper_artifacts({"artifact_schema_version": 1}, ["contract"]), legacy)
        self.assertEqual(expected_paper_artifacts({"artifact_schema_version": "unknown"}, ["contract"]), legacy)
        self.assertEqual(expected_paper_artifacts({}, ["prepare"]), [])

    def test_an_older_schema_is_still_checked_under_the_names_it_declared(self):
        """Renaming an artifact must not invalidate a version sealed before it."""
        self.assertEqual(
            list(PAPER_ARTIFACTS_BY_SCHEMA[1][CONTRACT_STEP]),
            ["method_figure_content.json", "method_extraction_report.json"],
        )
        self.assertEqual(
            expected_paper_artifacts({"artifact_schema_version": 3}, ["review", "contract"]),
            [
                "paper_review.json",
                "method_figure_content.json",
                "method_extraction_report.json",
                "figure_specification.json",
            ],
        )

    def test_the_current_contract_list_names_the_inventory_and_not_the_plan(self):
        """A skipped extraction still seals: the inventory is free, the plan is not."""
        self.assertEqual(ARTIFACT_SCHEMA_VERSION, 4)
        contract = list(PAPER_ARTIFACTS[CONTRACT_STEP])
        self.assertIn("paper_method_inventory.json", contract)
        self.assertIn("method_extraction_report.json", contract)
        self.assertNotIn("figure_content_plan.json", contract)

    def test_a_written_version_declares_the_artifacts_validation_expects(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = build_pilot_graph_payloads(snapshot, pilot)
            write_pilot_graph_version(root / "graphs", payloads)
            manifest = json.loads((root / "graphs" / "graph_manifest.json").read_text(encoding="utf-8"))
            checked = validate_pilot_graph_version(root / "graphs")

        self.assertEqual(manifest["artifact_schema_version"], ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(manifest["paper_artifacts"], {step: list(names) for step, names in PAPER_ARTIFACTS.items()})
        self.assertEqual(checked["checks"]["artifact_schema_version"], ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(checked["checks"]["expected_paper_artifacts"], [])


FROZEN_BLOCK = "A Frozen Study Abstract We propose a frozen pipeline."
FROZEN_EVIDENCE_ID = evidence_id(1, FROZEN_BLOCK, None, "paragraph")


def fake_provider(evidence, *, tokens=(1200, 300)):
    """Return a provider stand-in that answers with one grounded review.

    The review names the same four concepts the method contract stand-in names,
    so a test that swaps the scientific layer's source keeps the same cast and
    only the stage it is testing changes behaviour.
    """

    def call(prompt, image_paths, model=None, timeout=180, retries=1, call_metadata=None, deadline_at=None):
        if call_metadata is not None:
            call_metadata.update({
                "attempts": 1, "success": True, "elapsed_seconds": 0.01, "failure_categories": [],
                "prompt_chars": len(prompt), "estimated_tokens": max(1, round(len(prompt) / 4)),
                "image_count": len(image_paths), "image_bytes": 0,
                "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens)},
                "token_source": "provider", "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens),
            })
        return {
            "summary": "Fake grounded review.",
            "paper_identity": {"title": "A Frozen Study", "paper_type": "method", "field": "cv"},
            "research_questions": [{"id": "rq1", "statement": "Does the frozen pipeline work?", "evidence_ids": [evidence]}],
            "inputs": [{"id": "in1", "statement": "Frozen text", "evidence_ids": [evidence], "must_appear_in_figure": True, "visual_role": "input"}],
            "modules": [
                {"id": "m1", "statement": "Frozen encoder", "evidence_ids": [evidence], "must_appear_in_figure": True, "visual_role": "module"},
                {"id": "m2", "statement": "Frozen decoder", "evidence_ids": [evidence], "must_appear_in_figure": True, "visual_role": "module"},
            ],
            "outputs": [{"id": "out1", "statement": "Framework figure", "evidence_ids": [evidence], "must_appear_in_figure": True, "visual_role": "output"}],
            "relations": [{"id": "rel1", "source_id": "m1", "target_id": "m2", "relation_type": "data_flow", "statement": "feeds", "evidence_ids": [evidence]}],
        }

    return call


class ReviewStageTests(unittest.TestCase):
    def test_review_stage_stores_reviews_and_provider_usage(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
                with patch("rfs.analysis.paper.review.call_vlm_json", fake_provider(FROZEN_EVIDENCE_ID)):
                    payloads = build_pilot_graph_payloads(snapshot, pilot, stages="prepare,review", workers=2)

        self.assertTrue(payloads["validation"]["ok"])
        self.assertEqual(len(payloads["reviews"]), len(PAPERS))
        self.assertEqual(payloads["stages_completed"], ["prepare", "review"])
        self.assertFalse(payloads["complete"])
        review_rows = [row for row in payloads["ledger"] if row["step"] == "review"]
        self.assertEqual(len(review_rows), len(PAPERS))
        for row in review_rows:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["token_source"], "provider")
            self.assertEqual(row["total_tokens"], 1500)
        cost = payloads["costs"]["steps"]["review"]
        self.assertEqual(cost["total_tokens"], 1500 * len(PAPERS))
        self.assertEqual(payloads["costs"]["projection"]["papers"], 5732)
        provenance = payloads["reviews"][0]["graph_provenance"]
        self.assertEqual(provenance["review_mode"], "vlm")
        self.assertEqual(provenance["grounding_rate"], 1.0)
        self.assertTrue(provenance["evidence_count"])
        module = payloads["reviews"][0]["modules"][0]
        self.assertEqual(module["evidence_refs"][0]["evidence_id"], FROZEN_EVIDENCE_ID)
        self.assertEqual(module["evidence_refs"][0]["section"], "Abstract")

    def test_the_review_stage_writes_the_review_it_stored_per_paper(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
                with patch("rfs.analysis.paper.review.call_vlm_json", fake_provider(FROZEN_EVIDENCE_ID)):
                    payloads = build_pilot_graph_payloads(snapshot, pilot, stages="prepare,review", workers=2)
            write_pilot_graph_version(root / "graphs", payloads)
            stored = {
                paper_id: json.loads(
                    (root / "graphs" / "graphs" / safe_paper_directory(paper_id) / "paper_review.json")
                    .read_text(encoding="utf-8")
                )
                for paper_id, _domain, _split in PAPERS
            }
            jsonl = {
                row["graph_provenance"]["paper_id"]: row
                for row in (
                    json.loads(line)
                    for line in (root / "graphs" / "paper_reviews.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
            ledger = {row["paper_id"]: row for row in payloads["ledger"] if row["step"] == "review"}
            mismatches = ledger_artifact_mismatches(root / "graphs")
            checked = validate_pilot_graph_version(root / "graphs")

        self.assertEqual(sorted(stored), sorted(jsonl))
        self.assertEqual(mismatches, [])
        self.assertEqual(checked["checks"]["missing_paper_artifacts"], [])
        self.assertTrue(checked["ok"], checked["checks"])
        for paper_id, payload in stored.items():
            self.assertEqual(payload, jsonl[paper_id])
            self.assertEqual(canonical_sha256(payload), ledger[paper_id]["artifact_sha256"]["paper_review.json"])

    def test_review_stage_never_stores_a_heuristic_fallback(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
                with patch("rfs.analysis.paper.review.call_vlm_json", side_effect=RuntimeError("provider down")):
                    payloads = build_pilot_graph_payloads(snapshot, pilot, stages="review", workers=2)

        self.assertEqual(payloads["reviews"], [])
        self.assertFalse(payloads["validation"]["ok"])
        review_rows = [row for row in payloads["ledger"] if row["step"] == "review"]
        self.assertTrue(all(row["status"] == "failed" for row in review_rows))
        self.assertIn("provider down", review_rows[0]["error"])
        self.assertTrue(any("review stage failed" in failure for failure in payloads["validation"]["failures"]))

    def test_review_stage_requires_credentials_before_calling(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            with patch.dict(os.environ, {"API_BASE": "", "API_KEY": "", "GEMINI_API_KEY": ""}, clear=False):
                with self.assertRaises(RuntimeError):
                    build_pilot_graph_payloads(snapshot, pilot, stages="review")

    def test_resume_reuses_stored_reviews_without_new_calls(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
                with patch("rfs.analysis.paper.review.call_vlm_json", fake_provider(FROZEN_EVIDENCE_ID)):
                    first = build_pilot_graph_payloads(snapshot, pilot, stages="prepare,review", workers=2)
                write_pilot_graph_version(root / "graphs", first)
                with patch("rfs.analysis.paper.review.call_vlm_json", side_effect=AssertionError("resume must not call the provider")):
                    second = build_pilot_graph_payloads(snapshot, pilot, stages="prepare,review", workers=2, resume_dir=root / "graphs")

        self.assertEqual(len(second["reviews"]), len(PAPERS))
        before = [row["total_tokens"] for row in first["ledger"] if row["step"] == "review"]
        after = [row["total_tokens"] for row in second["ledger"] if row["step"] == "review"]
        self.assertEqual(after, before)
        self.assertEqual(len(after), len(PAPERS))




def fake_method_provider(evidence, *, tokens=(900, 120), evidence_ids=None, extra_modules=()):
    """Return a provider stand-in that answers with one grounded method contract."""

    def call(prompt, image_paths, model=None, timeout=180, retries=1, call_metadata=None, deadline_at=None):
        if call_metadata is not None:
            call_metadata.update({
                "attempts": 1, "success": True, "elapsed_seconds": 0.02, "failure_categories": [],
                "prompt_chars": len(prompt), "estimated_tokens": max(1, round(len(prompt) / 4)),
                "image_count": len(image_paths), "image_bytes": 0,
                "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens)},
                "token_source": "provider", "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens),
            })
        cited = list(evidence_ids or [evidence])
        modules = [
            {"id": "m1", "label": "Frozen encoder", "detail_level": "module", "must_show": True, "evidence_ids": cited},
            {"id": "m2", "label": "Frozen decoder", "detail_level": "module", "must_show": True, "evidence_ids": cited},
            *[
                {"id": f"extra{index}", "label": label, "detail_level": "module", "must_show": True, "evidence_ids": cited}
                for index, label in enumerate(extra_modules)
            ],
        ]
        return {
            "figure_goal": "Show how the frozen pipeline turns paper text into a figure.",
            "inputs": [{"id": "in1", "label": "Frozen text", "detail_level": "artifact", "must_show": True, "evidence_ids": cited}],
            "modules": modules,
            "outputs": [{"id": "out1", "label": "Framework figure", "detail_level": "artifact", "must_show": True, "evidence_ids": cited}],
            "relations": [{"id": "r1", "source": "m1", "target": "m2", "type": "data_flow", "label": "feeds", "evidence_ids": cited}],
        }

    return call


class ContractAndGraphStageTests(unittest.TestCase):
    def build(self, snapshot, pilot, *, stages, method_provider, review_provider=None, resume_dir=None, workers=2):
        review_provider = review_provider if review_provider is not None else fake_provider(FROZEN_EVIDENCE_ID)
        method_patch = (
            patch("rfs.analysis.paper.method_extractor.call_vlm_json", method_provider)
            if callable(method_provider)
            else patch("rfs.analysis.paper.method_extractor.call_vlm_json", side_effect=method_provider)
        )
        with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.analysis.paper.review.call_vlm_json", review_provider):
                with method_patch:
                    return build_pilot_graph_payloads(snapshot, pilot, stages=stages, workers=workers, resume_dir=resume_dir)

    def test_contract_and_graph_stages_build_the_scientific_graph(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph",
                # The contract stand-in names one concept the review does not, so a
                # graph that grew that concept would prove it was compiled from the
                # contract instead of from the paper review.
                method_provider=fake_method_provider(FROZEN_EVIDENCE_ID, extra_modules=("Frozen reranker",)),
            )

        self.assertTrue(payloads["validation"]["ok"], payloads["validation"]["failures"])
        self.assertEqual(payloads["stages_completed"], ["prepare", "review", "contract", "graph"])
        self.assertEqual(len(payloads["contracts"]), len(PAPERS))
        self.assertEqual(len(payloads["graphs"]), len(PAPERS))
        contract_rows = [row for row in payloads["ledger"] if row["step"] == "contract"]
        self.assertEqual(len(contract_rows), len(PAPERS))
        for row in contract_rows:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["token_source"], "provider")
            self.assertEqual(row["total_tokens"], 1020)
            self.assertEqual(row["node_count"], 5)
            self.assertEqual(row["unknown_evidence_ids"], [])
        for row in [row for row in payloads["ledger"] if row["step"] == "graph"]:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["token_source"], "not_called")
            self.assertEqual(row["node_count"], 4)
            self.assertEqual(row["edge_count"], 1)
            self.assertTrue(row["artifact_sha256"]["scientific_graph.json"])
        graph = payloads["graphs"][0]["scientific_graph"]
        self.assertEqual(graph["kind"], "scientific")
        # The scientific layer is compiled from the stored review, not from the
        # method plan the contract stage asked the provider for, so the two stages
        # answer with different sizes on purpose.
        self.assertEqual(graph["provenance"]["source_contract"], "rfs.paper_method_inventory")
        self.assertEqual(
            [node["label"] for node in graph["nodes"]],
            ["Frozen text", "Framework figure", "Frozen encoder", "Frozen decoder"],
        )
        self.assertNotIn("Frozen reranker", [node["label"] for node in graph["nodes"]])
        self.assertEqual(len(graph["edges"]), 1)
        self.assertTrue(all(node["evidence_ids"] for node in graph["nodes"]))
        for node in graph["nodes"]:
            for field in ("parent_source_id", "input_source_ids", "output_source_ids"):
                self.assertNotIn(field, node["properties"])
        self.assertEqual(payloads["coverage"]["scientific"]["nodes"], 4 * len(PAPERS))
        self.assertEqual(payloads["coverage"]["contracts"], len(PAPERS))
        report = payloads["reports"][0]
        self.assertEqual(report["method_contract"]["mode"], "vlm")
        self.assertFalse(report["method_contract"]["fallback_used"])
        self.assertEqual(report["scientific"]["node_count"], 4)
        self.assertEqual(report["stages"]["graph"], "ok")
        self.assertEqual(payloads["costs"]["steps"]["contract"]["total_tokens"], 1020 * len(PAPERS))
        self.assertEqual(sorted(payloads["costs"]["projection"]["per_step"]), ["contract", "review"])
        self.assertFalse(payloads["complete"])

    def test_the_contract_stage_freezes_the_specification_the_generation_side_reads(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract",
                method_provider=fake_method_provider(FROZEN_EVIDENCE_ID),
            )
            write_pilot_graph_version(root / "graphs", payloads)
            ledger = {row["paper_id"]: row for row in payloads["ledger"] if row["step"] == "contract"}
            payload_specs = {entry["paper_id"]: entry["figure_specification"] for entry in payloads["contracts"]}
            stored = {}
            for paper_id in payload_specs:
                directory = root / "graphs" / "graphs" / safe_paper_directory(paper_id)
                stored[paper_id] = json.loads((directory / "paper_method_inventory.json").read_text(encoding="utf-8"))
            mismatches = ledger_artifact_mismatches(root / "graphs")
            checked = validate_pilot_graph_version(root / "graphs")

        self.assertEqual(sorted(stored), sorted(paper_id for paper_id, _domain, _split in PAPERS))
        self.assertEqual(mismatches, [])
        self.assertEqual(checked["checks"]["missing_paper_artifacts"], [])
        for paper_id, specification in stored.items():
            self.assertEqual(specification, payload_specs[paper_id])
            self.assertTrue(specification["figure_goal"])
            self.assertTrue(specification["modules"])
            self.assertEqual(
                canonical_sha256(specification),
                ledger[paper_id]["artifact_sha256"]["paper_method_inventory.json"],
            )

    def test_a_version_is_checked_against_the_list_it_declares(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract",
                method_provider=fake_method_provider(FROZEN_EVIDENCE_ID),
            )
            write_pilot_graph_version(root / "graphs", payloads)
            stored = validate_pilot_graph_version(root / "graphs")
            # The declared list is the rule the version is read under, so a
            # version asking for one more file than it wrote fails on that file.
            manifest_path = root / "graphs" / "graph_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["paper_artifacts"]["contract"] = list(manifest["paper_artifacts"]["contract"]) + ["figure_program.json"]
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            declared = validate_pilot_graph_version(root / "graphs")

        self.assertEqual(stored["checks"]["missing_paper_artifacts"], [])
        self.assertTrue(stored["ok"], stored["checks"])
        self.assertEqual(stored["checks"]["artifact_schema_version"], ARTIFACT_SCHEMA_VERSION)
        self.assertIn("paper_review.json", stored["checks"]["expected_paper_artifacts"])
        self.assertIn("paper_method_inventory.json", stored["checks"]["expected_paper_artifacts"])
        self.assertNotIn("figure_program.json", stored["checks"]["expected_paper_artifacts"])
        self.assertIn("figure_program.json", declared["checks"]["expected_paper_artifacts"])
        self.assertEqual(
            sorted(declared["checks"]["missing_paper_artifacts"]),
            sorted(f"{paper_id}/figure_program.json" for paper_id, _domain, _split in PAPERS),
        )
        self.assertFalse(declared["ok"])

    def test_contract_stage_never_stores_a_heuristic_fallback(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph",
                method_provider=RuntimeError("contract provider down"),
            )

        self.assertEqual(payloads["contracts"], [])
        self.assertEqual(len(payloads["graphs"]), len(PAPERS))
        self.assertFalse(payloads["validation"]["ok"])
        self.assertEqual(payloads["stages_completed"], ["prepare", "review", "graph"])
        contract_rows = [row for row in payloads["ledger"] if row["step"] == "contract"]
        self.assertTrue(contract_rows)
        self.assertTrue(all(row["status"] == "failed" for row in contract_rows))
        self.assertIn("fell back to the review seed", contract_rows[0]["error"])
        # The graph stage reads the stored review, so a contract that fell back to
        # the review seed can no longer take the scientific layer down with it.
        graph_rows = [row for row in payloads["ledger"] if row["step"] == "graph"]
        self.assertTrue(all(row["status"] == "ok" for row in graph_rows))
        self.assertEqual([row["node_count"] for row in graph_rows], [4] * len(PAPERS))
        self.assertTrue(any("contract stage failed" in failure for failure in payloads["validation"]["failures"]))

    def test_a_carried_ledger_row_finds_the_file_under_its_current_name(self):
        """A resume copies rows that named the earlier file but writes the current one."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph",
                method_provider=fake_method_provider(FROZEN_EVIDENCE_ID),
            )
            target = root / "graphs"
            write_pilot_graph_version(target, payloads)
            ledger_path = target / "graph_ledger.jsonl"
            rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows:
                hashes = row.get("artifact_sha256") or {}
                for current, earlier in (
                    ("figure_content_plan.json", "method_figure_content.json"),
                    ("paper_method_inventory.json", "figure_specification.json"),
                ):
                    if current in hashes:
                        hashes[earlier] = hashes.pop(current)
            ledger_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
            )
            located = ledger_artifact_mismatches(target)
            victim = rows[0]["paper_id"]
            directory = target / "graphs" / safe_paper_directory(victim)
            (directory / "figure_content_plan.json").unlink()
            missing = [
                item for item in ledger_artifact_mismatches(target)
                if item["paper_id"] == victim and item["artifact"] == "method_figure_content.json"
            ]

        self.assertEqual([], located)
        self.assertEqual(1, len(missing))
        self.assertEqual("missing", missing[0]["reason"])
        self.assertEqual("method_figure_content.json", missing[0]["location"])

    def test_a_resumed_version_records_the_current_artifact_names(self):
        """Re-sealing a version written before the rename carries the rows to today's names."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            first = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph",
                method_provider=fake_method_provider(FROZEN_EVIDENCE_ID),
            )
            target = root / "graphs"
            write_pilot_graph_version(target, first)
            # An older version: it stores the earlier names and its rows record them.
            for paper_id, _domain, _split in PAPERS:
                directory = target / "graphs" / safe_paper_directory(paper_id)
                (directory / "figure_content_plan.json").rename(directory / "method_figure_content.json")
                (directory / "paper_method_inventory.json").rename(directory / "figure_specification.json")
            ledger_path = target / "graph_ledger.jsonl"
            rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows:
                hashes = row.get("artifact_sha256") or {}
                for current, earlier in (
                    ("figure_content_plan.json", "method_figure_content.json"),
                    ("paper_method_inventory.json", "figure_specification.json"),
                ):
                    if current in hashes:
                        hashes[earlier] = hashes.pop(current)
            ledger_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
            )
            second = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph",
                method_provider=AssertionError("resume must not call the provider"),
                resume_dir=target,
            )
            write_pilot_graph_version(root / "graphs2", second)
            checked = validate_pilot_graph_version(root / "graphs2")

        contract_rows = [row for row in second["ledger"] if row["step"] == "contract"]
        self.assertTrue(contract_rows)
        for row in contract_rows:
            recorded = row["artifact_sha256"]
            self.assertIn("figure_content_plan.json", recorded)
            self.assertIn("paper_method_inventory.json", recorded)
            self.assertNotIn("method_figure_content.json", recorded)
            self.assertNotIn("figure_specification.json", recorded)
        self.assertEqual([], checked["checks"]["ledger_artifact_mismatches"])
        self.assertEqual([], checked["checks"]["missing_paper_artifacts"])
        self.assertTrue(checked["ok"], checked["checks"])

    def test_ungrounded_contract_evidence_stays_on_the_contract_row(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph",
                method_provider=fake_method_provider(FROZEN_EVIDENCE_ID, evidence_ids=["E9999"]),
            )

        contract_rows = [row for row in payloads["ledger"] if row["step"] == "contract"]
        self.assertEqual(contract_rows[0]["unknown_evidence_ids"], ["E9999"])
        self.assertTrue(any("not part of the frozen document" in warning for warning in contract_rows[0]["warnings"]))
        # The graph stage never reads the contract, so an id the contract made up
        # is reported where it appeared and does not reach the scientific layer.
        graph_rows = [row for row in payloads["ledger"] if row["step"] == "graph"]
        self.assertTrue(all(row["status"] == "ok" for row in graph_rows))
        self.assertEqual([row["unknown_evidence_ids"] for row in graph_rows], [[]] * len(PAPERS))
        self.assertTrue(payloads["validation"]["ok"], payloads["validation"]["failures"])

    def test_ungrounded_review_evidence_fails_the_graph_stage(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,graph",
                method_provider=fake_method_provider(FROZEN_EVIDENCE_ID),
                review_provider=fake_provider("E9999"),
            )

        graph_rows = [row for row in payloads["ledger"] if row["step"] == "graph"]
        self.assertTrue(graph_rows)
        self.assertTrue(all(row["status"] == "failed" for row in graph_rows))
        self.assertEqual(graph_rows[0]["unknown_evidence_ids"], ["E9999"])
        self.assertIn("unknown evidence ID: E9999", graph_rows[0]["error"])
        self.assertFalse(payloads["validation"]["ok"])
        self.assertTrue(any("graph stage failed" in failure for failure in payloads["validation"]["failures"]))

    def test_graph_stage_without_a_stored_review_fails_without_calling_the_provider(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            with patch.dict(os.environ, {"API_BASE": "", "API_KEY": "", "GEMINI_API_KEY": ""}, clear=False):
                payloads = build_pilot_graph_payloads(snapshot, pilot, stages="graph")

        self.assertEqual(payloads["graphs"], [])
        self.assertFalse(payloads["validation"]["ok"])
        graph_rows = [row for row in payloads["ledger"] if row["step"] == "graph"]
        self.assertEqual(len(graph_rows), len(PAPERS))
        self.assertTrue(all(row["status"] == "failed" for row in graph_rows))
        self.assertIn("no stored paper review", graph_rows[0]["error"])

    def test_resume_reuses_stored_contracts_and_graphs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            first = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph",
                method_provider=fake_method_provider(FROZEN_EVIDENCE_ID),
            )
            write_pilot_graph_version(root / "graphs", first)
            second = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph",
                method_provider=AssertionError("resume must not call the provider"),
                review_provider=AssertionError("resume must not call the provider"),
                resume_dir=root / "graphs",
            )
            checked = validate_pilot_graph_version(root / "graphs")

        self.assertEqual(len(second["contracts"]), len(PAPERS))
        self.assertEqual(len(second["graphs"]), len(PAPERS))
        self.assertEqual(second["stages_completed"], ["prepare", "review", "contract", "graph"])
        self.assertTrue(second["validation"]["ok"], second["validation"]["failures"])
        tokens = [row["total_tokens"] for row in second["ledger"] if row["step"] == "contract"]
        self.assertEqual(tokens, [1020] * len(PAPERS))
        self.assertTrue(checked["ok"])
        self.assertEqual(checked["checks"]["contracts"], len(PAPERS))
        self.assertEqual(checked["checks"]["graphs"], len(PAPERS))
        self.assertEqual(checked["checks"]["missing_paper_artifacts"], [])
        self.assertFalse(checked["complete"])

    def test_a_stored_graph_from_an_older_source_is_rebuilt_not_resumed(self):
        """A version bumped to a new graph source must not turn into an empty run.

        The stored graph is still a valid file, so a resume that only asked "is the
        paper present" would skip every paper and the run would claim the new
        source while shipping the old graph. The graph's provenance names the
        document it came from, and that name is what the resume gate reads.
        """

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            first = self.build(
                snapshot, pilot,
                stages="prepare,review,graph",
                method_provider=fake_method_provider(FROZEN_EVIDENCE_ID),
            )
            write_pilot_graph_version(root / "graphs", first)
            for paper_id, _domain, _split in PAPERS:
                path = root / "graphs" / "graphs" / safe_paper_directory(paper_id) / "scientific_graph.json"
                graph = json.loads(path.read_text(encoding="utf-8"))
                graph["provenance"]["source_contract"] = "rfs.figure_specification"
                path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
            second = self.build(
                snapshot, pilot,
                stages="prepare,review,graph",
                method_provider=AssertionError("resume must not call the provider"),
                review_provider=AssertionError("resume must not call the provider"),
                resume_dir=root / "graphs",
            )
            write_pilot_graph_version(root / "graphs2", second)
            # A graph that is already current is left alone: the marker written into
            # the stored file can only survive a resume that rebuilt nothing.
            for paper_id, _domain, _split in PAPERS:
                path = root / "graphs2" / "graphs" / safe_paper_directory(paper_id) / "scientific_graph.json"
                graph = json.loads(path.read_text(encoding="utf-8"))
                graph["provenance"]["resume_marker"] = "untouched"
                path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
            still_current = self.build(
                snapshot, pilot,
                stages="prepare,review,graph",
                method_provider=AssertionError("resume must not call the provider"),
                review_provider=AssertionError("resume must not call the provider"),
                resume_dir=root / "graphs2",
            )

        graph_rows = [row for row in second["ledger"] if row["step"] == "graph"]
        self.assertEqual(len(graph_rows), len(PAPERS))
        self.assertTrue(all(row["status"] == "ok" for row in graph_rows))
        for entry in second["graphs"]:
            provenance = entry["scientific_graph"]["provenance"]
            self.assertEqual(provenance["source_contract"], "rfs.paper_method_inventory")
        # The rebuild wrote a current graph, so the next resume keeps it as it is.
        self.assertEqual(
            {entry["scientific_graph"]["provenance"].get("resume_marker") for entry in still_current["graphs"]},
            {"untouched"},
        )
        self.assertTrue(still_current["validation"]["ok"], still_current["validation"]["failures"])


def fake_visual_provider(*, tokens=(2000, 400)):
    """Return a provider stand-in that reads two panels out of the figure."""

    def call(prompt, image_paths, model=None, timeout=180, retries=1, call_metadata=None, deadline_at=None):
        if call_metadata is not None:
            call_metadata.update({
                "attempts": 1, "success": True, "elapsed_seconds": 0.03, "failure_categories": [],
                "prompt_chars": len(prompt), "estimated_tokens": max(1, round(len(prompt) / 4)),
                "image_count": len(image_paths), "image_bytes": 0,
                "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens)},
                "token_source": "provider", "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens),
            })
        return {
            "summary": "Two panels: a frozen encoder feeding a decoder, then an output.",
            "reading_direction": "left_to_right",
            "panels": [
                {"id": "p1", "label": "training", "box": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0}},
                {"id": "p2", "label": "inference", "box": {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0}},
            ],
            "blocks": [
                {"id": "b1", "label": "Frozen encoder", "panel_id": "p1", "box": {"x": 0.02, "y": 0.1, "w": 0.2, "h": 0.1}, "text_confidence": 0.9, "visual_role": "module"},
                {"id": "b2", "label": "Frozen decoder", "panel_id": "p1", "box": {"x": 0.26, "y": 0.1, "w": 0.2, "h": 0.1}, "text_confidence": 0.9, "visual_role": "module"},
                {"id": "b3", "label": "Output", "panel_id": "p2", "box": {"x": 0.55, "y": 0.1, "w": 0.2, "h": 0.1}, "text_confidence": 0.8, "visual_role": "module"},
                {"id": "b4", "label": "Memory", "panel_id": "p2", "box": {"x": 0.55, "y": 0.5, "w": 0.2, "h": 0.1}, "text_confidence": 0.7, "visual_role": "module"},
            ],
            "connectors": [
                {"id": "e1", "source_id": "b1", "target_id": "b2", "relation_type": "forward", "label": ""},
                {"id": "e2", "source_id": "b2", "target_id": "b3", "relation_type": "cross_panel", "label": "feeds"},
                {"id": "e3", "source_id": "b4", "target_id": "b1", "relation_type": "feedback", "label": "loop"},
                {"id": "e4", "source_id": "b9", "target_id": "b1", "relation_type": "forward", "label": "dangling"},
            ],
            "uncertainties": ["one small arrow label could not be read"],
        }

    return call


def prompt_check_ids(prompt):
    """Return the short check id the prompt printed for each label, as a reader would.

    The check stage asks with c1, c2 and r1, r2 instead of the hashed scientific ids,
    so the stand-in reads those ids straight off the prompt before answering.
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


def fake_check_provider(*, tokens=(1500, 500)):
    """Return a provider stand-in that audits the figure against the contract."""

    def call(prompt, image_paths, model=None, timeout=180, retries=1, call_metadata=None, deadline_at=None):
        if call_metadata is not None:
            call_metadata.update({
                "attempts": 1, "success": True, "elapsed_seconds": 0.02, "failure_categories": [],
                "prompt_chars": len(prompt), "estimated_tokens": max(1, round(len(prompt) / 4)),
                "image_count": len(image_paths), "image_bytes": 0,
                "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens)},
                "token_source": "provider", "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens),
            })
        concepts, relations = prompt_check_ids(prompt)

        def finding(label, *, visible, how="as_block", image_label="", where="", confidence=0.9):
            return {"id": concepts.get(label, ""), "visible": visible, "how": how, "image_label": image_label, "where": where, "confidence": confidence}

        return {
            "image_summary": "Two panels: a frozen encoder feeding a frozen decoder; the text input and the framework output are not drawn.",
            "concepts": [
                finding("Frozen text", visible=False, how="not_drawn", confidence=0.6),
                finding("Frozen encoder", visible=True, image_label="Frozen encoder", where="panel 1, middle"),
                finding("Frozen decoder", visible=True, image_label="Frozen decoder", where="panel 1, right", confidence=0.8),
                finding("Framework figure", visible=False, how="not_drawn", confidence=0.6),
            ],
            "relations": [{"id": relations.get("Frozen encoder", ""), "arrow_present": True, "direction_matches": True, "confidence": 0.8}],
            "region_annotations": [
                {"label": "Legend", "kind": "legend", "explains": "arrow styles", "where": "panel 2, bottom"},
            ],
            "unlisted_content": [
                {"label": "Output", "where": "panel 2, middle"},
                {"label": "Memory", "where": "panel 2, bottom"},
            ],
            "uncertainties": ["one caption is small"],
        }

    return call


class VisualStageTests(unittest.TestCase):
    def build(self, snapshot, pilot, *, stages, visual_provider=None, figure_dir=None, resume_dir=None, workers=2, prompts=None):
        provider = visual_provider if visual_provider is not None else fake_visual_provider()
        if not callable(provider):
            provider = Mock(side_effect=provider)
        elif prompts is not None:
            inner = provider

            def capture(prompt, image_paths, **kwargs):
                prompts.append(prompt)
                return inner(prompt, image_paths, **kwargs)

            provider = capture
        with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.judge_training.pilot_graphs.call_vlm_json", provider):
                return build_pilot_graph_payloads(
                    snapshot, pilot, stages=stages, workers=workers, resume_dir=resume_dir, figure_dir=figure_dir,
                )

    def test_visual_stage_reads_the_framework_figure(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            prompts = []
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,visual",
                figure_dir=root / "graphs" / "figures",
                prompts=prompts,
            )

        self.assertTrue(payloads["validation"]["ok"], payloads["validation"]["failures"])
        self.assertEqual(payloads["stages_completed"], ["prepare", "visual"])
        self.assertEqual(len(payloads["visuals"]), len(PAPERS))
        rows = [row for row in payloads["ledger"] if row["step"] == "visual"]
        self.assertEqual(len(rows), len(PAPERS))
        for row in rows:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["token_source"], "provider")
            self.assertEqual(row["total_tokens"], 2400)
            self.assertEqual(row["panel_count"], 2)
            self.assertEqual(row["block_count"], 4)
            self.assertEqual(row["edge_count"], 3)
            self.assertTrue(row["artifact_sha256"]["visual_graph.json"])
        graph = payloads["visuals"][0]["visual_graph"]
        report = payloads["visuals"][0]["visual_graph_report"]
        self.assertEqual(graph["kind"], "visual")
        self.assertEqual(len(graph["nodes"]), 6)
        self.assertEqual(len(graph["edges"]), 3)
        self.assertFalse(graph["provenance"]["pilot_labels_used_in_prompt"])
        self.assertEqual(report["primary_structure"], "multi_panel")
        self.assertIn("feedback_loop", report["structure_signals"])
        self.assertEqual([row["id"] for row in report["removed_connectors"]], ["e4"])
        self.assertEqual(report["reading_order"], ["b1", "b2", "b3", "b4"])
        self.assertTrue(report["image"]["png_sha256"])
        self.assertTrue(report["pilot_label_agreement"]["panel_count"]["agrees"])
        self.assertEqual(payloads["coverage"]["visual"]["papers"], len(PAPERS))
        self.assertEqual(payloads["coverage"]["visual"]["structures"]["multi_panel"], len(PAPERS))
        self.assertEqual(payloads["coverage"]["visual"]["motif_histogram"]["feedback"], len(PAPERS) * 2)
        self.assertEqual(payloads["reports"][0]["visual"]["panel_count"], 2)
        self.assertEqual(payloads["reports"][0]["stages"]["visual"], "ok")
        self.assertEqual(payloads["costs"]["steps"]["visual"]["total_tokens"], 2400 * len(PAPERS))
        self.assertFalse(payloads["complete"])
        self.assertIn("Overview of the frozen framework.", prompts[0])
        self.assertNotIn("linear", prompts[0].casefold())

    def test_rendered_figures_are_reused_on_a_second_run(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            figure_dir = root / "graphs" / "figures"
            first = self.build(snapshot, pilot, stages="prepare,visual", figure_dir=figure_dir)
            second = self.build(snapshot, pilot, stages="prepare,visual", figure_dir=figure_dir)

        first_image = first["visuals"][0]["visual_graph_report"]["image"]
        second_image = second["visuals"][0]["visual_graph_report"]["image"]
        self.assertFalse(first_image["reused"])
        self.assertTrue(second_image["reused"])
        self.assertEqual(second_image["png_sha256"], first_image["png_sha256"])
        self.assertEqual(second_image["source_sha256"], first_image["source_sha256"])

    def test_visual_stage_failure_is_recorded_and_blocks_the_gate(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,visual",
                visual_provider=RuntimeError("visual provider down"),
                figure_dir=root / "graphs" / "figures",
            )

        self.assertEqual(payloads["visuals"], [])
        self.assertFalse(payloads["validation"]["ok"])
        rows = [row for row in payloads["ledger"] if row["step"] == "visual"]
        self.assertEqual(len(rows), len(PAPERS))
        self.assertTrue(all(row["status"] == "failed" for row in rows))
        self.assertIn("visual provider down", rows[0]["error"])
        self.assertTrue(any("visual stage failed" in failure for failure in payloads["validation"]["failures"]))

    def test_visual_stage_refuses_without_credentials(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            with patch.dict(os.environ, {"API_BASE": "", "API_KEY": "", "GEMINI_API_KEY": ""}, clear=False):
                with self.assertRaises(RuntimeError):
                    build_pilot_graph_payloads(snapshot, pilot, stages="prepare,visual", figure_dir=root / "figures")

    def test_resume_reuses_stored_visual_graphs_and_renders(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            figure_dir = root / "graphs" / "figures"
            first = self.build(snapshot, pilot, stages="prepare,visual", figure_dir=figure_dir)
            write_pilot_graph_version(root / "graphs", first)
            second = self.build(
                snapshot, pilot,
                stages="prepare,visual",
                visual_provider=AssertionError("resume must not call the provider"),
                figure_dir=figure_dir,
                resume_dir=root / "graphs",
            )
            checked = validate_pilot_graph_version(root / "graphs")

        self.assertEqual(len(second["visuals"]), len(PAPERS))
        self.assertEqual(second["stages_completed"], ["prepare", "visual"])
        self.assertTrue(second["validation"]["ok"], second["validation"]["failures"])
        self.assertTrue(checked["ok"], checked["checks"])
        self.assertEqual(checked["checks"]["visuals"], len(PAPERS))
        self.assertEqual(checked["checks"]["rendered_figures"], len(PAPERS))
        self.assertEqual(checked["checks"]["missing_figures"], [])
        self.assertEqual(checked["checks"]["missing_paper_artifacts"], [])
        self.assertFalse(checked["complete"])

    def test_sealed_version_is_not_rendered_into_again(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            figure_dir = root / "graphs" / "figures"
            first = self.build(snapshot, pilot, stages="prepare,visual", figure_dir=figure_dir)
            write_pilot_graph_version(root / "graphs", first)
            (root / "graphs" / "COMPLETE").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.build(snapshot, pilot, stages="prepare,visual", figure_dir=figure_dir)


class BindStageTests(unittest.TestCase):
    def build(self, snapshot, pilot, *, stages, resume_dir=None, figure_dir=None):
        with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.analysis.paper.review.call_vlm_json", fake_provider(FROZEN_EVIDENCE_ID)):
                with patch("rfs.analysis.paper.method_extractor.call_vlm_json", fake_method_provider(FROZEN_EVIDENCE_ID)):
                    with patch("rfs.judge_training.pilot_graphs.call_vlm_json", fake_visual_provider()):
                        return build_pilot_graph_payloads(
                            snapshot, pilot, stages=stages, workers=2, resume_dir=resume_dir, figure_dir=figure_dir,
                        )

    def test_bind_stage_connects_the_two_layers_without_provider_calls(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind",
                figure_dir=root / "graphs" / "figures",
            )

        self.assertTrue(payloads["validation"]["ok"], payloads["validation"]["failures"])
        self.assertEqual(payloads["stages_completed"], ["prepare", "review", "contract", "graph", "visual", "bind"])
        self.assertEqual(len(payloads["bindings"]), len(PAPERS))
        rows = [row for row in payloads["ledger"] if row["step"] == "bind"]
        self.assertEqual(len(rows), len(PAPERS))
        for row in rows:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["token_source"], "not_called")
            self.assertIsNone(row["total_tokens"])
            self.assertEqual(row["bound_nodes"], 2)
            self.assertEqual(row["bound_edges"], 1)
            self.assertEqual(row["diagnostic_required_node_coverage"], 0.5)
            self.assertEqual(row["reversed_edges"], 0)
            self.assertTrue(row["artifact_sha256"]["graph_bindings.json"])
        binding_set = payloads["bindings"][0]["graph_bindings"]
        self.assertEqual(binding_set["contract"], "rfs.graph_bindings")
        self.assertEqual(
            sorted(item["status"] for item in binding_set["node_bindings"]),
            ["bound", "bound", "unbound", "unbound"],
        )
        edge_statuses = {item["scientific_edge_id"]: item["status"] for item in binding_set["edge_bindings"]}
        self.assertEqual(sorted(edge_statuses.values()), ["bound"])
        report = payloads["bindings"][0]["binding_report"]
        self.assertEqual(report["provider_calls"], 0)
        self.assertEqual(report["required_nodes"], 4)
        self.assertEqual(report["mean_binding_score"], 1.0)
        self.assertEqual(report["required_edges_unbound"], [])
        coverage = payloads["coverage"]["bindings"]
        self.assertEqual(coverage["papers"], len(PAPERS))
        self.assertEqual(coverage["bound_nodes"], 2 * len(PAPERS))
        self.assertEqual(coverage["diagnostic_required_node_coverage"], 0.5)
        self.assertEqual(coverage["reversed_edges"], 0)
        self.assertEqual(coverage["papers_with_reversed_edges"], [])
        self.assertEqual(payloads["reports"][0]["binding"]["bound_nodes"], 2)
        self.assertEqual(payloads["reports"][0]["stages"]["bind"], "ok")
        self.assertFalse(payloads["complete"])
        self.assertEqual(payloads["costs"]["steps"]["bind"]["total_tokens"], 0)

    def test_the_claim_audit_is_written_beside_the_bindings(self):
        """Every object of the figure ends with one conclusion, and it is stored."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind",
                figure_dir=root / "graphs" / "figures",
            )
            write_pilot_graph_version(root / "graphs", payloads)
            manifest = json.loads((root / "graphs" / "graph_manifest.json").read_text(encoding="utf-8"))
            checked = validate_pilot_graph_version(root / "graphs")
            claims_files = sorted((root / "graphs" / "graphs").glob("*/figure_claims.json"))
            claims = [json.loads(path.read_text(encoding="utf-8")) for path in claims_files]

        self.assertEqual(len(claims), len(PAPERS))
        self.assertEqual(manifest["artifact_schema_version"], ARTIFACT_SCHEMA_VERSION)
        self.assertIn("figure_claims.json", manifest["paper_artifacts"]["bind"])
        for payload in claims:
            self.assertEqual(payload["contract"], "rfs.figure_claims")
            self.assertEqual(payload["version"], pilot_bindings.FIGURE_CLAIMS_VERSION)
            self.assertEqual(payload["binding_version"], BINDING_VERSION)
            summary = payload["summary"]
            self.assertEqual(
                summary["objects"],
                summary["blocks"] + summary["row_labels"] + summary["arrows"] + summary["other"],
            )
            self.assertEqual(
                sum(
                    summary[key]
                    for key in ("bound", "outside_paper_method", "in_paper_not_in_method", "figure_annotation", "unresolved")
                ),
                summary["objects"],
            )
            for item in payload["objects"]:
                self.assertIn(
                    item["conclusion"],
                    ("bound", "outside_paper_method", "in_paper_not_in_method", "figure_annotation", "unresolved"),
                )
        rows = [row for row in payloads["ledger"] if row["step"] == "bind"]
        for row in rows:
            self.assertTrue(row["artifact_sha256"]["figure_claims.json"])
        coverage = payloads["coverage"]["bindings"]
        self.assertEqual(coverage["claim_objects"], sum(payload["summary"]["objects"] for payload in claims))
        self.assertEqual(coverage["claim_bound"], sum(payload["summary"]["bound"] for payload in claims))
        self.assertEqual(coverage["claim_blocks"], sum(payload["summary"]["blocks"] for payload in claims))
        self.assertEqual(
            coverage["diagnostic_claim_coverage_all_objects"],
            round((coverage["claim_bound"] + coverage["claim_outside_paper_method"]) / coverage["claim_objects"], 4),
        )
        self.assertEqual(
            coverage["claim_coverage"],
            round(
                sum(payload["summary"]["resolved_meaningful"] for payload in claims)
                / sum(payload["summary"]["meaningful_objects"] for payload in claims),
                4,
            ),
        )
        self.assertEqual(coverage["selection_samples"]["as_block"] + coverage["selection_samples"]["inside_other_graphics"], coverage["selection_samples"]["drawn"])
        self.assertTrue(checked["ok"], checked["checks"])
        self.assertEqual(checked["checks"]["missing_paper_artifacts"], [])
        self.assertEqual(checked["checks"]["bindings"], len(PAPERS))

    def test_the_headline_counts_the_meaningful_objects_and_the_all_object_ratio_stays_beside_it(self):
        """A symbol or a decoration is an object of the figure, but never a meaningful one."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind",
                figure_dir=root / "graphs" / "figures",
            )

        coverage = payloads["coverage"]["bindings"]
        summaries = [entry["figure_claims"]["summary"] for entry in payloads["bindings"]]
        self.assertEqual(coverage["claim_meaningful_objects"], sum(item["meaningful_objects"] for item in summaries))
        self.assertEqual(coverage["claim_excluded_from_meaning"], sum(item["excluded_from_meaning"] for item in summaries))
        self.assertEqual(
            coverage["claim_meaningful_objects"] + coverage["claim_excluded_from_meaning"],
            coverage["claim_objects"],
        )
        self.assertEqual(
            coverage["claim_coverage"],
            round(
                sum(item["resolved_meaningful"] for item in summaries) / coverage["claim_meaningful_objects"],
                4,
            ),
        )
        self.assertEqual(
            coverage["diagnostic_claim_coverage_all_objects"],
            round((coverage["claim_bound"] + coverage["claim_outside_paper_method"]) / coverage["claim_objects"], 4),
        )
        self.assertIn("claim_annotation_coverage", coverage)
        self.assertIn("claim_provenance_coverage", coverage)
        # The required-list coverage stays reported, under a name that says it is a
        # diagnostic of the list rather than the acceptance metric of the figure.
        self.assertIn("diagnostic_required_node_coverage", coverage)
        self.assertNotIn("required_node_coverage", coverage)
        self.assertIn("diagnostic_label_match_required_coverage", coverage)
        report = payloads["bindings"][0]["binding_report"]
        self.assertIn("diagnostic_required_node_coverage", report)
        self.assertNotIn("required_node_coverage", report)
        for row in payloads["ledger"]:
            if row["step"] == "bind":
                self.assertIn("diagnostic_required_node_coverage", row)

    def test_a_resumed_version_carries_the_claim_audit_it_stored(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            figure_dir = root / "graphs" / "figures"
            first = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind",
                figure_dir=figure_dir,
            )
            write_pilot_graph_version(root / "graphs", first)
            second = self.build(
                snapshot, pilot,
                stages="prepare,bind",
                figure_dir=figure_dir,
                resume_dir=root / "graphs",
            )

        self.assertEqual(second["bindings"][0]["figure_claims"], first["bindings"][0]["figure_claims"])
        self.assertEqual(second["coverage"]["bindings"]["claim_objects"], first["coverage"]["bindings"]["claim_objects"])

    def test_bind_stage_without_a_visual_graph_fails_and_is_reported(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,bind",
                figure_dir=root / "figures",
            )

        self.assertEqual(payloads["bindings"], [])
        rows = [row for row in payloads["ledger"] if row["step"] == "bind"]
        self.assertEqual(len(rows), len(PAPERS))
        self.assertTrue(all(row["status"] == "failed" for row in rows))
        self.assertIn("no stored visual graph", rows[0]["error"])
        self.assertFalse(payloads["validation"]["ok"])
        self.assertTrue(any("bind stage failed" in failure for failure in payloads["validation"]["failures"]))

    def test_resume_reuses_stored_bindings(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            figure_dir = root / "graphs" / "figures"
            first = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind",
                figure_dir=figure_dir,
            )
            write_pilot_graph_version(root / "graphs", first)
            second = self.build(
                snapshot, pilot,
                stages="prepare,bind",
                figure_dir=figure_dir,
                resume_dir=root / "graphs",
            )
            checked = validate_pilot_graph_version(root / "graphs")

        self.assertEqual(len(second["bindings"]), len(PAPERS))
        self.assertEqual(second["stages_completed"], ["prepare", "review", "contract", "graph", "visual", "bind"])
        self.assertTrue(second["validation"]["ok"], second["validation"]["failures"])
        self.assertEqual(second["bindings"][0]["graph_bindings"], first["bindings"][0]["graph_bindings"])
        self.assertEqual(second["coverage"]["bindings"]["bound_nodes"], 2 * len(PAPERS))
        self.assertTrue(checked["ok"], checked["checks"])
        self.assertEqual(checked["checks"]["bindings"], len(PAPERS))
        self.assertEqual(checked["checks"]["failed_bindings"], [])
        self.assertEqual(checked["checks"]["missing_paper_artifacts"], [])
        self.assertFalse(checked["complete"])

    def test_a_stored_binding_from_another_layer_is_rebuilt_not_resumed(self):
        """A binding names the two layers it read, so a superseded layer invalidates it.

        A graph's id is stable for its paper, so only the recorded hash of the layer
        can tell a current binding from one built against an older revision. The
        untouched papers stay resumed; the one whose layer changed is bound again.
        """

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            figure_dir = root / "graphs" / "figures"
            first = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind",
                figure_dir=figure_dir,
            )
            write_pilot_graph_version(root / "graphs", first)
            changed = PAPERS[0][0]
            # A marker written into the stored files says which of them the resume
            # left alone: only the paper whose layer hash was tampered with is bound
            # again, and a rebuild writes a binding without the marker.
            for paper_id, _domain, _split in PAPERS:
                path = root / "graphs" / "graphs" / safe_paper_directory(paper_id) / "graph_bindings.json"
                bindings = json.loads(path.read_text(encoding="utf-8"))
                bindings["provenance"]["resume_marker"] = "untouched"
                if paper_id == changed:
                    bindings["provenance"]["scientific_graph_sha256"] = "0" * 64
                path.write_text(json.dumps(bindings, ensure_ascii=False), encoding="utf-8")
            second = self.build(
                snapshot, pilot,
                stages="prepare,bind",
                figure_dir=figure_dir,
                resume_dir=root / "graphs",
            )

        rebuilt = {entry["paper_id"]: entry["graph_bindings"] for entry in second["bindings"]}
        self.assertEqual(
            {paper_id for paper_id, value in rebuilt.items() if value["provenance"].get("resume_marker") == "untouched"},
            {paper_id for paper_id, _domain, _split in PAPERS if paper_id != changed},
        )
        self.assertNotIn("resume_marker", rebuilt[changed]["provenance"])
        self.assertNotEqual(rebuilt[changed]["provenance"]["scientific_graph_sha256"], "0" * 64)
        self.assertTrue(rebuilt[changed]["provenance"]["scientific_graph_sha256"])


class StubEmbedder:
    """Deterministic stand-in for the paid endpoint, with the usage a real one reports."""

    name = "stub-embedding-1"
    revision = "stub-v1"
    dimension = 2

    def __init__(self, *, fail=False):
        self.fail = fail
        self.attempts = 0
        self.calls = 0
        self.texts = 0
        self.total_tokens = 0

    def encode(self, texts):
        wanted = [str(text) for text in texts]
        self.attempts += 1
        if self.fail:
            raise EmbeddingUnavailable("stub endpoint is down")
        self.calls += 1
        self.texts += len(wanted)
        self.total_tokens += sum(max(1, len(text) // 4) for text in wanted)
        # Different labels get orthogonal vectors, so this stub cannot invent a match
        # the deterministic layers would not have made: only the accounting and the
        # fuse are under test here.
        return [[1.0, 0.0] if "encoder" in text.casefold() else [0.0, 1.0] for text in wanted]

    def fingerprint(self):
        return {"contract": "rfs.text_embedding", "kind": "api_embeddings", "model": self.name, "revision": self.revision, "dimension": self.dimension}

    def usage(self):
        return {"calls": self.calls, "texts": self.texts, "total_tokens": self.total_tokens, "error": ""}


class BindingSemanticLayerTests(unittest.TestCase):
    """The bind stage's semantic layer is opt-in, fused, and reported per paper."""

    def build(self, snapshot, pilot, *, stages, figure_dir=None, embedder=None, env=None):
        variables = {"API_BASE": "https://example.test/v1", "API_KEY": "key"}
        variables.update(env or {})
        with patch.dict(os.environ, variables, clear=False):
            with patch("rfs.analysis.paper.review.call_vlm_json", fake_provider(FROZEN_EVIDENCE_ID)):
                with patch("rfs.analysis.paper.method_extractor.call_vlm_json", fake_method_provider(FROZEN_EVIDENCE_ID)):
                    with patch("rfs.judge_training.pilot_graphs.call_vlm_json", fake_visual_provider()):
                        if embedder is None:
                            return build_pilot_graph_payloads(
                                snapshot, pilot, stages=stages, workers=2, figure_dir=figure_dir,
                            )
                        with patch(
                            "rfs.judge_training.pilot_graphs.resolve_binding_embedder",
                            return_value=(embedder, {"status": "api_embeddings", "spec": "api:stub", "error": ""}),
                        ):
                            return build_pilot_graph_payloads(
                                snapshot, pilot, stages=stages, workers=2, figure_dir=figure_dir,
                            )

    def rows(self, payloads):
        return [row for row in payloads["ledger"] if row["step"] == "bind"]

    def reports(self, payloads):
        return [entry["binding_report"] for entry in payloads["bindings"]]

    def test_the_semantic_layer_stays_off_until_it_is_named(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind",
                figure_dir=root / "figures",
                env={"RFS_BINDING_EMBEDDER": "off"},
            )

        self.assertTrue(payloads["validation"]["ok"], payloads["validation"]["failures"])
        coverage = payloads["coverage"]["bindings"]
        self.assertEqual(coverage["embedding_statuses"], {"off": len(PAPERS)})
        self.assertEqual(coverage["embedding_fingerprints"], [])
        self.assertEqual(coverage["bound_nodes"], 2 * len(PAPERS))
        self.assertEqual(coverage["diagnostic_required_node_coverage"], 0.5)
        for row in self.rows(payloads):
            self.assertEqual(row["token_source"], "not_called")
            self.assertEqual(row["attempts"], 0)
            self.assertIsNone(row["total_tokens"])
            self.assertEqual(row["embedding_status"], "off")
            self.assertEqual(row["embedding_calls"], 0)
        self.assertEqual(payloads["costs"]["steps"]["bind"]["total_tokens"], 0)

    def test_a_named_endpoint_is_asked_once_per_paper_and_paid_for_once_per_paper(self):
        embedder = StubEmbedder()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind",
                figure_dir=root / "figures",
                embedder=embedder,
            )

        self.assertTrue(payloads["validation"]["ok"], payloads["validation"]["failures"])
        coverage = payloads["coverage"]["bindings"]
        self.assertEqual(coverage["embedding_statuses"], {"api_embeddings": len(PAPERS)})
        self.assertEqual([item["model"] for item in coverage["embedding_fingerprints"]], [embedder.name])
        # The soft layer is added on top: the deterministic layers decide the same nodes.
        self.assertEqual(coverage["bound_nodes"], 2 * len(PAPERS))
        self.assertEqual(coverage["diagnostic_required_node_coverage"], 0.5)
        self.assertEqual(embedder.attempts, len(PAPERS))
        for row in self.rows(payloads):
            self.assertEqual(row["token_source"], "provider")
            self.assertEqual(row["attempts"], 1)
            self.assertEqual(row["embedding_calls"], 1)
            self.assertGreater(row["embedding_texts"], 0)
            self.assertGreater(row["total_tokens"], 0)
        step = payloads["costs"]["steps"]["bind"]
        self.assertEqual(step["token_sources"], {"provider": len(PAPERS)})
        self.assertEqual(step["attempts"], len(PAPERS))
        self.assertEqual(step["total_tokens"], embedder.total_tokens)
        for report in self.reports(payloads):
            self.assertEqual(report["embedding_status"], "api_embeddings")
            self.assertEqual(report["provider_calls"], 1)

    def test_an_endpoint_that_is_down_trips_the_fuse_and_never_fails_the_run(self):
        embedder = StubEmbedder(fail=True)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind",
                figure_dir=root / "figures",
                embedder=embedder,
            )

        self.assertTrue(payloads["validation"]["ok"], payloads["validation"]["failures"])
        self.assertEqual(len(payloads["bindings"]), len(PAPERS))
        # One paper pays for the failure: the fuse stops the rest from asking again.
        self.assertEqual(embedder.attempts, 1)
        self.assertEqual(embedder.calls, 0)
        coverage = payloads["coverage"]["bindings"]
        self.assertEqual(coverage["embedding_statuses"], {"unavailable": len(PAPERS)})
        self.assertEqual(coverage["bound_nodes"], 2 * len(PAPERS))
        self.assertEqual(coverage["diagnostic_required_node_coverage"], 0.5)
        for report in self.reports(payloads):
            self.assertEqual(report["embedding_status"], "unavailable")
            self.assertIn("stub endpoint is down", report["embedding_error"])
            self.assertEqual(report["provider_calls"], 0)
        for row in self.rows(payloads):
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["token_source"], "not_called")
            self.assertEqual(row["embedding_status"], "unavailable")

    def test_a_binder_that_raises_the_unavailable_error_degrades_instead_of_failing(self):
        seen = []

        def flaky(scientific, visual, review=None, *, embedder=None, **kwargs):
            seen.append(embedder is not None)
            if embedder is not None:
                raise EmbeddingUnavailable("endpoint went down mid-run")
            return pilot_bindings.build_layer_bindings(scientific, visual, review, embedder=None, **kwargs)

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            with patch("rfs.judge_training.pilot_graphs.build_layer_bindings", side_effect=flaky):
                payloads = self.build(
                    snapshot, pilot,
                    stages="prepare,review,contract,graph,visual,bind",
                    figure_dir=root / "figures",
                    embedder=StubEmbedder(),
                )

        self.assertTrue(payloads["validation"]["ok"], payloads["validation"]["failures"])
        self.assertEqual(seen[0], True)
        self.assertTrue(all(item is False for item in seen[1:]))
        for report in self.reports(payloads):
            self.assertEqual(report["embedding_status"], "unavailable")
            self.assertIn("endpoint went down mid-run", report["embedding_error"])


class BindingResumeGuardTests(unittest.TestCase):
    """A resumed run rebuilds a set an older binder wrote instead of silently skipping."""

    def build(self, snapshot, pilot, *, stages, figure_dir=None):
        with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.analysis.paper.review.call_vlm_json", fake_provider(FROZEN_EVIDENCE_ID)):
                with patch("rfs.analysis.paper.method_extractor.call_vlm_json", fake_method_provider(FROZEN_EVIDENCE_ID)):
                    with patch("rfs.judge_training.pilot_graphs.call_vlm_json", fake_visual_provider()):
                        return build_pilot_graph_payloads(snapshot, pilot, stages=stages, workers=2, figure_dir=figure_dir)

    def test_a_binding_set_from_an_older_binder_is_rebuilt_instead_of_skipped(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot, stages="prepare,review,contract,graph,visual,bind", figure_dir=root / "figures",
            )
            graphs = {entry["paper_id"]: entry for entry in payloads["graphs"]}
            visuals = {entry["paper_id"]: entry for entry in payloads["visuals"]}
            stored = {entry["paper_id"]: entry for entry in payloads["bindings"]}
            rows = load_pilot_ledger(pilot)
            stale = json.loads(json.dumps(stored))
            for entry in stale.values():
                entry["graph_bindings"]["provenance"]["binding_version"] = "pilot-layer-bindings-v1"

            current_entries, current_ledger, _current = bind_all(
                rows, graphs=graphs, visuals=visuals, existing_bindings=stored,
            )
            stale_entries, stale_ledger, stale_reports = bind_all(
                rows, graphs=graphs, visuals=visuals, existing_bindings=stale,
            )

        self.assertEqual(current_ledger, [])
        self.assertEqual(set(current_entries), set(stored))
        # Raising the version must never turn the stage into a no-op that ships the old
        # semantics: every paper of the stale set is bound again under the current one.
        self.assertEqual(len(stale_ledger), len(PAPERS))
        self.assertTrue(all(row["status"] == "ok" for row in stale_ledger))
        self.assertEqual({report["binding_version"] for report in stale_reports.values()}, {BINDING_VERSION})
        self.assertEqual(
            {entry["graph_bindings"]["provenance"]["binding_version"] for entry in stale_entries.values()},
            {BINDING_VERSION},
        )


class IndexStageTests(unittest.TestCase):
    def build(self, snapshot, pilot, *, stages, resume_dir=None, figure_dir=None, probe_limit=64, embedding_model=""):
        with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.analysis.paper.review.call_vlm_json", fake_provider(FROZEN_EVIDENCE_ID)):
                with patch("rfs.analysis.paper.method_extractor.call_vlm_json", fake_method_provider(FROZEN_EVIDENCE_ID)):
                    with patch("rfs.judge_training.pilot_graphs.call_vlm_json", fake_visual_provider()):
                        return build_pilot_graph_payloads(
                            snapshot, pilot, stages=stages, workers=2, resume_dir=resume_dir,
                            figure_dir=figure_dir, probe_limit=probe_limit, embedding_model=embedding_model,
                        )

    def full(self, root, snapshot, pilot, *, stages="prepare,review,contract,graph,visual,bind,index", resume_dir=None, probe_limit=64):
        return self.build(
            snapshot, pilot, stages=stages, resume_dir=resume_dir,
            figure_dir=root / "graphs" / "figures", probe_limit=probe_limit,
        )

    def test_index_stage_builds_the_retrieval_index_without_provider_calls(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.full(root, snapshot, pilot)
            second = self.full(root, snapshot, pilot)

        self.assertTrue(payloads["validation"]["ok"], payloads["validation"]["failures"])
        self.assertEqual(
            payloads["stages_completed"],
            ["prepare", "review", "contract", "graph", "visual", "bind", "index"],
        )
        self.assertFalse(payloads["complete"])
        self.assertIsNotNone(payloads["manifest"]["index"])
        self.assertEqual(payloads["manifest"]["index"]["provider_calls"], 0)
        index = payloads["index"]
        self.assertEqual(len(index["records"]), 2 * len(PAPERS))
        self.assertEqual(
            sorted(record["graph_kind"] for record in index["records"]),
            ["scientific", "scientific", "scientific", "visual", "visual", "visual"],
        )
        for record in index["records"]:
            self.assertEqual(record["contract"], "rfs.graph_index_record")
            self.assertEqual(record["subgraphs"], [])
            self.assertFalse(record["subgraph_locator"]["materialized"])
            self.assertEqual(record["subgraph_locator"]["index"], "subgraph_index.jsonl")
            self.assertEqual(record["subgraph_locator"]["count"], sum(record["index_report"]["motif_counts"].values()))
            self.assertGreater(record["subgraph_locator"]["count"], 0)
            self.assertEqual(record["metadata"]["layer"], record["graph_kind"])
            self.assertIn(record["metadata"]["split"], ("train", "validation"))
        located = sum(record["subgraph_locator"]["count"] for record in index["records"])
        self.assertEqual(len(index["subgraphs"]), located)
        motifs = {row["motif"] for row in index["subgraphs"]}
        self.assertTrue(motifs)
        for row in index["subgraphs"]:
            self.assertTrue(row["center_node_id"])
            self.assertEqual(row["source_graph"].split("/")[0], "graphs")
            self.assertEqual(row["extractor"], "node_centered_one_hop_v1")
        motif_rows = {row["motif"]: row for row in index["motifs"]}
        self.assertEqual(sorted(motif_rows), sorted(motifs))
        for motif, row in motif_rows.items():
            self.assertGreater(row["subgraph_count"], 0)
            self.assertEqual(row["papers"], sorted(row["papers"]))
            self.assertEqual(sum(row["domains"].values()), row["subgraph_count"])
            self.assertEqual(sum(row["graph_kinds"].values()), row["subgraph_count"])
        probe = index["probe"]
        self.assertEqual(probe["contract"], "rfs.graph_retrieval_probe")
        self.assertEqual(len(probe["queries"]), len(PAPERS))
        self.assertEqual(probe["guards"]["provider_calls"], 0)
        self.assertEqual(probe["guards"]["self_hits"], 0)
        self.assertEqual(probe["guards"]["hidden_test_records"], 0)
        self.assertEqual(probe["guards"]["hidden_test_hits"], 0)
        self.assertEqual(probe["guards"]["unexpected_splits"], [])
        self.assertEqual(probe["guards"]["splits"], {"train": 4, "validation": 2})
        self.assertEqual(probe["summary"]["queries_with_hits"], len(PAPERS))
        self.assertTrue(probe["summary"]["motifs_covered"])
        for row in probe["queries"]:
            self.assertEqual([hit["paper_id"] for hit in row["hits"]].count(row["paper_id"]), 0)
            self.assertTrue(all(hit["graph_kind"] in ("scientific", "visual") for hit in row["hits"]))
        self.assertTrue(probe["summary"]["cross_domain_hits"] > 0)
        rows = [row for row in payloads["ledger"] if row["step"] == "index"]
        self.assertEqual(len(rows), len(PAPERS))
        for row in rows:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["token_source"], "not_called")
            self.assertIsNone(row["total_tokens"])
            self.assertEqual(row["record_count"], 2)
            self.assertGreater(row["subgraph_count"], 0)
            self.assertTrue(row["record_ids"])
        self.assertEqual(payloads["costs"]["steps"]["index"]["total_tokens"], 0)
        self.assertEqual(payloads["coverage"]["index"]["papers"], len(PAPERS))
        self.assertEqual(payloads["coverage"]["index"]["layers"], {"scientific": len(PAPERS), "visual": len(PAPERS)})
        self.assertEqual(payloads["coverage"]["index"]["papers_missing_a_layer"], [])
        self.assertEqual(payloads["coverage"]["index"]["hidden_test_records"], 0)
        self.assertEqual(payloads["reports"][0]["stages"]["index"], "ok")
        self.assertEqual(payloads["reports"][0]["index"]["record_count"], 2)
        self.assertTrue(payloads["reports"][0]["index"]["record_ids"])
        self.assertEqual(
            json.dumps(payloads["index"], ensure_ascii=False, sort_keys=True),
            json.dumps(second["index"], ensure_ascii=False, sort_keys=True),
        )

    def test_every_subgraph_locator_carries_a_vector_and_the_manifest_names_the_encoder(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.full(root, snapshot, pilot)

        self.assertTrue(payloads["index"]["subgraphs"])
        self.assertTrue(all(len(row["embedding"]) == DEFAULT_DIMENSION for row in payloads["index"]["subgraphs"]))
        self.assertTrue(all(row["embedding_fingerprint"]["kind"] == "hashing_ngram" for row in payloads["index"]["subgraphs"]))
        self.assertEqual(payloads["coverage"]["index"]["subgraph_embeddings"], len(payloads["index"]["subgraphs"]))
        self.assertEqual(payloads["manifest"]["index"]["report"]["subgraphs_without_embedding"], 0)
        self.assertEqual(payloads["manifest"]["index"]["report"]["errors"], [])
        self.assertEqual(payloads["manifest"]["index"]["embedding_spec"], "")
        self.assertEqual(payloads["manifest"]["index"]["embedding"]["kind"], "hashing_ngram")

    def test_a_named_encoder_that_cannot_be_loaded_stops_the_stage_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            with self.assertRaises(EmbeddingUnavailable):
                self.build(
                    snapshot, pilot,
                    stages="prepare,review,contract,graph,visual,index",
                    figure_dir=root / "figures",
                    embedding_model=f"st:{root / 'no-such-model'}",
                )

    def test_index_stage_without_a_layer_fails_and_is_reported(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,index",
                figure_dir=root / "figures",
            )

        self.assertEqual(payloads["index"]["records"], [])
        self.assertEqual(payloads["index"]["subgraphs"], [])
        rows = [row for row in payloads["ledger"] if row["step"] == "index"]
        self.assertEqual(len(rows), len(PAPERS))
        self.assertTrue(all(row["status"] == "failed" for row in rows))
        self.assertIn("no stored visual layer", rows[0]["error"])
        self.assertFalse(payloads["validation"]["ok"])
        self.assertTrue(any("no retrieval record" in failure for failure in payloads["validation"]["failures"]))
        self.assertTrue(any("index stage failed" in failure for failure in payloads["validation"]["failures"]))
        self.assertFalse(payloads["complete"])

    def test_written_index_is_validated(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.full(root, snapshot, pilot)
            written = write_pilot_graph_version(root / "graphs", payloads)
            checked = validate_pilot_graph_version(root / "graphs")
            for artifact in ("graph_index.jsonl", "motif_index.jsonl", "subgraph_index.jsonl", "retrieval_probe.json"):
                self.assertIn(artifact, written["files"])
                self.assertTrue((root / "graphs" / artifact).is_file(), artifact)

        self.assertFalse(written["sealed"], written)
        self.assertTrue(checked["ok"], checked["checks"])
        self.assertFalse(checked["complete"])
        self.assertEqual(checked["checks"]["index_records"], 2 * len(PAPERS))
        self.assertEqual(checked["checks"]["index_papers"], len(PAPERS))
        self.assertEqual(checked["checks"]["index_probe_queries"], len(PAPERS))
        self.assertEqual(checked["checks"]["index_missing_papers"], [])
        self.assertEqual(checked["checks"]["missing_index_artifacts"], [])
        self.assertEqual(checked["checks"]["failed_index"], [])
        self.assertTrue(checked["checks"]["index_motifs"])

    def test_resume_reuses_the_stored_index(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            first = self.full(root, snapshot, pilot)
            write_pilot_graph_version(root / "graphs", first)
            second = self.build(
                snapshot, pilot,
                stages="prepare,index",
                resume_dir=root / "graphs",
                figure_dir=root / "graphs" / "figures",
            )
            third = self.build(
                snapshot, pilot,
                stages="prepare",
                resume_dir=root / "graphs",
                figure_dir=root / "graphs" / "figures",
            )

        self.assertEqual(second["stages_completed"], first["stages_completed"])
        self.assertTrue(second["validation"]["ok"], second["validation"]["failures"])
        self.assertEqual(
            json.dumps(second["index"], ensure_ascii=False, sort_keys=True),
            json.dumps(first["index"], ensure_ascii=False, sort_keys=True),
        )
        self.assertEqual(second["coverage"]["index"]["records"], 2 * len(PAPERS))
        self.assertEqual(third["stages_completed"], first["stages_completed"])
        self.assertEqual(third["coverage"]["index"]["records"], 2 * len(PAPERS))


class CheckStageTests(unittest.TestCase):
    def build(self, snapshot, pilot, *, stages, resume_dir=None, figure_dir=None):
        with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.analysis.paper.review.call_vlm_json", fake_provider(FROZEN_EVIDENCE_ID)):
                with patch("rfs.analysis.paper.method_extractor.call_vlm_json", fake_method_provider(FROZEN_EVIDENCE_ID)):
                    with patch("rfs.judge_training.pilot_graphs.call_vlm_json", fake_visual_provider()):
                        with patch("rfs.judge_training.pilot_check.call_vlm_json", fake_check_provider()):
                            return build_pilot_graph_payloads(
                                snapshot, pilot, stages=stages, workers=2, resume_dir=resume_dir, figure_dir=figure_dir,
                            )

    def test_check_stage_audits_the_figure_against_the_contract(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind,index,check",
                figure_dir=root / "figures",
            )
            second = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind,index,check",
                figure_dir=root / "figures-2",
            )

        self.assertTrue(payloads["validation"]["ok"], payloads["validation"]["failures"])
        self.assertEqual(
            payloads["stages_completed"],
            ["prepare", "review", "contract", "graph", "visual", "check", "bind", "index"],
        )
        self.assertTrue(payloads["complete"])
        self.assertEqual(len(payloads["checks"]), len(PAPERS))
        for entry in payloads["checks"]:
            review = entry["figure_text_review"]
            self.assertEqual(review["contract"], "rfs.figure_text_review")
            self.assertEqual(review["prompt_version"], "pilot-figure-check-v2")
            self.assertEqual(review["summary"]["concepts"], 4)
            self.assertEqual(review["summary"]["confirmed"], 2)
            self.assertEqual(review["summary"]["not_drawn"], 2)
            self.assertEqual(review["summary"]["relations_confirmed"], 1)
            self.assertEqual(review["summary"]["extra_blocks"], 2)
            self.assertEqual(review["summary"]["unlisted_content"], 2)
            self.assertEqual(review["summary"]["region_annotations"], 1)
            self.assertEqual(review["summary"]["reader_how_answered"], 4)
            self.assertEqual(review["extra_blocks"], review["unlisted_content"])
            self.assertEqual(review["region_annotations"][0]["explains"], "arrow styles")
            self.assertTrue(review["image"]["png_sha256"])
        rows = [row for row in payloads["ledger"] if row["step"] == "check"]
        self.assertEqual(len(rows), len(PAPERS))
        for row in rows:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["token_source"], "provider")
            self.assertEqual(row["total_tokens"], 2000)
            self.assertEqual(row["image_count"], 1)
            self.assertTrue(row["artifact_sha256"]["figure_text_review.json"])
        self.assertEqual(payloads["costs"]["steps"]["check"]["total_tokens"], 2000 * len(PAPERS))
        coverage = payloads["coverage"]["correspondence"]
        self.assertEqual(coverage["reviewed_papers"], len(PAPERS))
        self.assertEqual(coverage["unreviewed_papers"], 0)
        self.assertEqual(coverage["confirmed"], 2 * len(PAPERS))
        self.assertEqual(coverage["mean_required_visible_rate"], 0.5)
        self.assertEqual(coverage["papers_with_missed_concepts"], [])
        self.assertEqual(coverage["papers_with_hallucinated_blocks"], [])
        self.assertEqual(payloads["reports"][0]["stages"]["check"], "ok")
        self.assertEqual(payloads["reports"][0]["correspondence"]["confirmed"], 2)
        self.assertEqual(payloads["manifest"]["correspondence"]["reviewed_papers"], len(PAPERS))
        self.assertEqual(
            json.dumps([entry["figure_text_review"] for entry in payloads["checks"]], ensure_ascii=False, sort_keys=True),
            json.dumps([entry["figure_text_review"] for entry in second["checks"]], ensure_ascii=False, sort_keys=True),
        )

    def test_the_figure_is_accounted_for_block_by_block(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind,index,check",
                figure_dir=root / "graphs" / "figures",
            )

        coverage = payloads["coverage"]["bindings"]
        # Four blocks are drawn per figure: two carry a contract concept and the two the
        # reading listed outside the contract are declared, so nothing is left over.
        self.assertEqual(coverage["figure_first"], {
            "drawn_blocks": 4 * len(PAPERS),
            "claimed_blocks": 2 * len(PAPERS),
            "declared_outside_contract": 2 * len(PAPERS),
            "unaccounted_blocks": 0,
        })
        # The two concepts each figure does not draw are kept as selection samples.
        self.assertEqual(coverage["selection_samples"], {
            "judged": 4 * len(PAPERS),
            "drawn": 2 * len(PAPERS),
            "not_drawn": 2 * len(PAPERS),
            "as_block": 2 * len(PAPERS),
            "inside_other_graphics": 0,
        })
        for entry in payloads["bindings"]:
            report = entry["binding_report"]
            self.assertEqual(report["figure_first"]["drawn_blocks"], 4)
            self.assertEqual(report["selection_samples"]["not_drawn"], 2)

    def test_the_author_side_is_a_trade_off_not_a_defect(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind,index,check",
                figure_dir=root / "graphs" / "figures",
            )
            write_pilot_graph_version(root / "graphs", payloads)
            checked = validate_pilot_graph_version(root / "graphs")
            written = json.loads((root / "graphs" / "validation_report.json").read_text(encoding="utf-8"))

        self.assertTrue(checked["ok"], checked["checks"])
        reasons = [item["reason"] for item in written["human_review_required"]]
        self.assertTrue(any("selection samples" in reason for reason in reasons), reasons)
        self.assertFalse(any("missing from the figure" in reason for reason in reasons), reasons)
        self.assertFalse(any("against the contract on the figure" in reason for reason in reasons), reasons)
        self.assertFalse(any("against the scientific relation" in reason for reason in reasons), reasons)
        self.assertFalse(any("are drawn and bound" in reason for reason in reasons), reasons)

    def test_check_stage_without_a_visual_layer_fails_and_is_reported(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,check",
                figure_dir=root / "figures",
            )

        self.assertEqual(payloads["checks"], [])
        rows = [row for row in payloads["ledger"] if row["step"] == "check"]
        self.assertEqual(len(rows), len(PAPERS))
        self.assertTrue(all(row["status"] == "failed" for row in rows))
        self.assertIn("no stored visual graph", rows[0]["error"])
        self.assertFalse(payloads["validation"]["ok"])
        self.assertTrue(any("check stage failed" in failure for failure in payloads["validation"]["failures"]))
        self.assertFalse(payloads["complete"])

    def test_check_stage_needs_credentials_before_calling(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            with patch("rfs.judge_training.pilot_check.vlm_credentials_available", return_value=False):
                with self.assertRaises(RuntimeError) as caught:
                    self.build(
                        snapshot, pilot,
                        stages="prepare,review,contract,graph,visual,bind,check",
                        figure_dir=root / "figures",
                    )

        self.assertIn("check stage needs API_BASE", str(caught.exception))

    def test_resume_reuses_the_stored_reviews(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            first = self.build(
                snapshot, pilot,
                stages="prepare,review,contract,graph,visual,bind,index,check",
                figure_dir=root / "graphs" / "figures",
            )
            write_pilot_graph_version(root / "graphs", first)
            second = self.build(
                snapshot, pilot,
                stages="prepare,check",
                resume_dir=root / "graphs",
                figure_dir=root / "graphs" / "figures",
            )
            checked = validate_pilot_graph_version(root / "graphs")

        self.assertEqual(second["stages_completed"], first["stages_completed"])
        self.assertTrue(second["validation"]["ok"], second["validation"]["failures"])
        self.assertEqual(
            json.dumps([entry["figure_text_review"] for entry in second["checks"]], ensure_ascii=False, sort_keys=True),
            json.dumps([entry["figure_text_review"] for entry in first["checks"]], ensure_ascii=False, sort_keys=True),
        )
        self.assertEqual(second["coverage"]["correspondence"]["reviewed_papers"], len(PAPERS))
        self.assertTrue(checked["ok"], checked["checks"])
        self.assertEqual(checked["checks"]["checks"], len(PAPERS))
        self.assertEqual(checked["checks"]["failed_checks"], [])
        self.assertEqual(checked["checks"]["missing_paper_artifacts"], [])


def counting_provider(provider, counts, name):
    """Wrap a provider stand-in so a test can see how often each step called out."""

    def call(prompt, image_paths, *args, **kwargs):
        counts[name] = counts.get(name, 0) + 1
        return provider(prompt, image_paths, *args, **kwargs)

    return call


def drop_render_flags(value, parent=None):
    """Remove the per-run render cache flag from an image record."""
    if isinstance(value, dict):
        return {
            key: drop_render_flags(item, key)
            for key, item in value.items()
            if not (key == "reused" and parent == "image")
        }
    if isinstance(value, list):
        return [drop_render_flags(item, parent) for item in value]
    return value


def drop_volatile(value, fields):
    """Remove the volatile ledger fields wherever they appear in a JSON value."""
    if isinstance(value, dict):
        return {key: drop_volatile(item, fields) for key, item in value.items() if key not in fields}
    if isinstance(value, list):
        return [drop_volatile(item, fields) for item in value]
    return value


class ResumeGapTests(unittest.TestCase):
    """G7: a half written version is continued without redoing the finished work."""

    FULL = "prepare,review,contract,graph,visual,bind,index,check"
    HALF = "prepare,review,contract,graph,visual,bind"
    VOLATILE = ("elapsed_seconds", "prompt_tokens", "completion_tokens", "total_tokens")
    VERSION_FILES = (
        "graph_manifest.json",
        "graph_ledger.jsonl",
        "coverage_report.json",
        "cost_report.json",
        "validation_report.json",
        "paper_reviews.jsonl",
        "graph_index.jsonl",
        "motif_index.jsonl",
        "subgraph_index.jsonl",
        "retrieval_probe.json",
    )
    PAPER_FILES = (
        "graph_report.json",
        "paper_review.json",
        "figure_content_plan.json",
        "method_extraction_report.json",
        "paper_method_inventory.json",
        "scientific_graph.json",
        "scientific_ground_truth.json",
        "scientific_graph_report.json",
        "visual_graph.json",
        "visual_graph_report.json",
        "graph_bindings.json",
        "figure_text_review.json",
    )

    def build(self, snapshot, pilot, *, stages=None, resume_dir=None, figure_dir=None, counts=None):
        counts = {} if counts is None else counts
        with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.analysis.paper.review.call_vlm_json", counting_provider(fake_provider(FROZEN_EVIDENCE_ID), counts, "review")):
                with patch("rfs.analysis.paper.method_extractor.call_vlm_json", counting_provider(fake_method_provider(FROZEN_EVIDENCE_ID), counts, "contract")):
                    with patch("rfs.judge_training.pilot_graphs.call_vlm_json", counting_provider(fake_visual_provider(), counts, "visual")):
                        with patch("rfs.judge_training.pilot_check.call_vlm_json", counting_provider(fake_check_provider(), counts, "check")):
                            return build_pilot_graph_payloads(
                                snapshot, pilot, stages=stages if stages is not None else self.FULL,
                                workers=2, resume_dir=resume_dir, figure_dir=figure_dir,
                            )

    def normalized(self, path):
        """Return one stored file as canonical JSON, volatile fields dropped and rows sorted."""
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            rows = [drop_volatile(json.loads(line), self.VOLATILE) for line in text.splitlines() if line.strip()]
            return json.dumps(sorted(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows), ensure_ascii=False)
        payload = drop_render_flags(drop_volatile(json.loads(text), self.VOLATILE))
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def version_contents(self, directory):
        """Return every stored artifact of a written version, keyed by its relative path."""
        names = [name for name in self.VERSION_FILES if (directory / name).is_file()]
        for paper_id, _domain, _split in PAPERS:
            folder = "graphs/" + safe_paper_directory(paper_id)
            names.extend(folder + "/" + name for name in self.PAPER_FILES if (directory / folder / name).is_file())
        contents = {name: self.normalized(directory / name) for name in names}
        for path in sorted((directory / "figures").glob("*/*.png")):
            contents[path.relative_to(directory).as_posix()] = path.read_bytes()
        return contents

    def stable(self, value):
        """Return one JSON value without the volatile and per-run render fields."""
        return json.dumps(drop_render_flags(drop_volatile(value, self.VOLATILE)), ensure_ascii=False, sort_keys=True)

    def differences(self, left, right):
        """Return the stored artifacts that differ between two written versions."""
        first, second = self.version_contents(left), self.version_contents(right)
        return sorted(name for name in set(first) | set(second) if first.get(name) != second.get(name))

    def test_a_full_run_asks_the_provider_once_per_paper_and_step(self):
        counts = {}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            payloads = self.build(snapshot, pilot, figure_dir=root / "figures", counts=counts)

        self.assertEqual(counts, {"review": len(PAPERS), "contract": len(PAPERS), "visual": len(PAPERS), "check": len(PAPERS)})
        self.assertTrue(payloads["complete"])
        self.assertEqual(
            payloads["stages_completed"],
            ["prepare", "review", "contract", "graph", "visual", "check", "bind", "index"],
        )

    def test_resume_only_fills_the_paper_whose_stored_layer_is_missing(self):
        victim = PAPERS[1][0]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            figures = root / "graphs" / "figures"
            first = self.build(snapshot, pilot, figure_dir=figures, counts={})
            write_pilot_graph_version(root / "graphs", first)
            (root / "graphs" / "COMPLETE").unlink()
            layer = root / "graphs" / "graphs" / safe_paper_directory(victim) / "visual_graph.json"
            stored = json.loads(layer.read_text(encoding="utf-8"))
            layer.unlink()
            counts = {}
            second = self.build(snapshot, pilot, resume_dir=root / "graphs", figure_dir=figures, counts=counts)
            rewritten = {entry["paper_id"]: entry["visual_graph"] for entry in second["visuals"]}
            reused = {
                entry["paper_id"]: ((entry["visual_graph"].get("provenance") or {}).get("image") or {}).get("reused")
                for entry in second["visuals"]
            }

        self.assertEqual(counts, {"visual": 1})
        self.assertTrue(reused[victim], reused)
        self.assertEqual(second["stages_completed"], first["stages_completed"])
        self.assertTrue(second["validation"]["ok"], second["validation"]["failures"])
        self.assertEqual(self.stable(rewritten[victim]), self.stable(stored))

    def test_resume_retries_only_the_missing_review(self):
        victim = PAPERS[2][0]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            figures = root / "graphs" / "figures"
            first = self.build(snapshot, pilot, figure_dir=figures, counts={})
            write_pilot_graph_version(root / "graphs", first)
            (root / "graphs" / "COMPLETE").unlink()
            review = root / "graphs" / "graphs" / safe_paper_directory(victim) / "figure_text_review.json"
            stored = json.loads(review.read_text(encoding="utf-8"))
            review.unlink()
            counts = {}
            second = self.build(snapshot, pilot, resume_dir=root / "graphs", figure_dir=figures, counts=counts)
            rewritten = {entry["paper_id"]: entry["figure_text_review"] for entry in second["checks"]}
            written = write_pilot_graph_version(root / "graphs", second)
            resealed = (root / "graphs" / "COMPLETE").is_file()

        self.assertEqual(counts, {"check": 1})
        self.assertTrue(second["validation"]["ok"], second["validation"]["failures"])
        self.assertTrue(written["sealed"])
        self.assertTrue(resealed)
        self.assertEqual(self.stable(rewritten[victim]), self.stable(stored))

    def test_resume_rebuilds_a_specification_the_version_lost(self):
        victim = PAPERS[2][0]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            figures = root / "graphs" / "figures"
            first = self.build(snapshot, pilot, figure_dir=figures, counts={})
            write_pilot_graph_version(root / "graphs", first)
            (root / "graphs" / "COMPLETE").unlink()
            directory = root / "graphs" / "graphs" / safe_paper_directory(victim)
            stored = canonical_sha256(json.loads((directory / "paper_method_inventory.json").read_text(encoding="utf-8")))
            (directory / "paper_method_inventory.json").unlink()
            counts = {}
            second = self.build(snapshot, pilot, resume_dir=root / "graphs", figure_dir=figures, counts=counts)
            write_pilot_graph_version(root / "graphs", second)
            path = root / "graphs" / "graphs" / safe_paper_directory(victim) / "paper_method_inventory.json"
            rebuilt = canonical_sha256(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else None
            checked = validate_pilot_graph_version(root / "graphs")

        self.assertEqual(counts, {})
        self.assertEqual(rebuilt, stored)
        self.assertEqual(checked["checks"]["ledger_artifact_mismatches"], [])
        self.assertEqual(checked["checks"]["missing_paper_artifacts"], [])
        self.assertTrue(checked["ok"], checked["checks"])

    def test_a_stage_subset_resume_restores_a_specification_it_never_requested(self):
        victim = PAPERS[1][0]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            figures = root / "graphs" / "figures"
            first = self.build(snapshot, pilot, figure_dir=figures, counts={})
            write_pilot_graph_version(root / "graphs", first)
            directory = root / "graphs" / "graphs" / safe_paper_directory(victim)
            stored = canonical_sha256(json.loads((directory / "paper_method_inventory.json").read_text(encoding="utf-8")))
            (root / "graphs" / "COMPLETE").unlink()
            (directory / "paper_method_inventory.json").unlink()
            counts = {}
            second = self.build(snapshot, pilot, stages="index", resume_dir=root / "graphs", figure_dir=figures, counts=counts)
            write_pilot_graph_version(root / "graphs", second)
            path = directory / "paper_method_inventory.json"
            rebuilt = canonical_sha256(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else None
            checked = validate_pilot_graph_version(root / "graphs")
            completed = second["stages_completed"]

        self.assertEqual(counts, {})
        self.assertEqual(rebuilt, stored)
        self.assertIn("contract", completed)
        self.assertEqual(checked["checks"]["missing_paper_artifacts"], [])
        self.assertEqual(checked["checks"]["ledger_artifact_mismatches"], [])
        self.assertTrue(checked["ok"], checked["checks"])

    def test_a_resumed_version_matches_a_fresh_run(self):
        victim = PAPERS[1][0]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            fresh = self.build(snapshot, pilot, figure_dir=root / "fresh" / "figures", counts={})
            write_pilot_graph_version(root / "fresh", fresh)
            version = root / "version"
            half = self.build(snapshot, pilot, stages=self.HALF, figure_dir=version / "figures", counts={})
            write_pilot_graph_version(version, half)
            (version / "graphs" / safe_paper_directory(victim) / "graph_bindings.json").unlink()
            counts = {}
            resumed = self.build(snapshot, pilot, resume_dir=version, figure_dir=version / "figures", counts=counts)
            write_pilot_graph_version(version, resumed)
            differences = self.differences(root / "fresh", version)

        self.assertEqual(counts, {"check": len(PAPERS)})
        self.assertEqual(
            sorted(entry["paper_id"] for entry in resumed["bindings"]),
            sorted(paper_id for paper_id, _domain, _split in PAPERS),
        )
        self.assertEqual(resumed["stages_completed"], fresh["stages_completed"])
        self.assertTrue(resumed["complete"])
        self.assertEqual(differences, [])


class SealVerificationTests(unittest.TestCase):
    """G8: a sealed version proves its stored artifacts are the recorded ones."""

    FULL = "prepare,review,contract,graph,visual,bind,index,check"

    def build(self, snapshot, pilot, *, figure_dir):
        with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.analysis.paper.review.call_vlm_json", fake_provider(FROZEN_EVIDENCE_ID)):
                with patch("rfs.analysis.paper.method_extractor.call_vlm_json", fake_method_provider(FROZEN_EVIDENCE_ID)):
                    with patch("rfs.judge_training.pilot_graphs.call_vlm_json", fake_visual_provider()):
                        with patch("rfs.judge_training.pilot_check.call_vlm_json", fake_check_provider()):
                            return build_pilot_graph_payloads(
                                snapshot, pilot, stages=self.FULL, workers=2, figure_dir=figure_dir,
                            )

    def seal(self, root):
        """Write one sealed full version and return its directory."""
        snapshot, pilot = write_fixture(root)
        payloads = self.build(snapshot, pilot, figure_dir=root / "graphs" / "figures")
        written = write_pilot_graph_version(root / "graphs", payloads)
        return root / "graphs", written

    def rows(self, target):
        return [
            json.loads(line)
            for line in (target / "graph_ledger.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def paper_dir(self, target, index=0):
        return target / "graphs" / safe_paper_directory(self.rows(target)[index]["paper_id"])

    def rewrite(self, path, mutate):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def test_a_sealed_version_verifies_every_recorded_hash(self):
        with tempfile.TemporaryDirectory() as name:
            target, written = self.seal(Path(name))
            checked = validate_pilot_graph_version(target)
            mismatches = ledger_artifact_mismatches(target)
            claimed = {}
            for row in self.rows(target):
                claimed.setdefault(row["step"], set()).update((row.get("artifact_sha256") or {}).keys())

        self.assertTrue(written["sealed"], written)
        self.assertTrue(checked["ok"], checked["checks"])
        self.assertTrue(checked["sealed"], checked["checks"])
        self.assertTrue(checked["complete"], checked["checks"])
        self.assertEqual(checked["checks"]["ledger_artifact_mismatches"], [])
        self.assertEqual(checked["checks"]["empty_artifacts"], [])
        self.assertTrue(checked["checks"]["seal"]["valid"], checked["checks"]["seal"])
        self.assertEqual(mismatches, [])
        self.assertEqual(claimed["prepare"], {"graph_report.json"})
        self.assertEqual(claimed["review"], {"review", "paper_review.json"})
        self.assertEqual(
            claimed["contract"],
            {"figure_content_plan.json", "paper_method_inventory.json"},
        )
        self.assertEqual(
            claimed["graph"],
            {"scientific_graph.json", "scientific_ground_truth.json", "scientific_graph_report.json"},
        )

    def test_a_sealed_version_carries_the_review_and_specification_it_hashed(self):
        with tempfile.TemporaryDirectory() as name:
            target, _written = self.seal(Path(name))
            rows = self.rows(target)
            jsonl = {
                row["graph_provenance"]["paper_id"]: row
                for row in (
                    json.loads(line)
                    for line in (target / "paper_reviews.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
            reviews = {row["paper_id"]: row for row in rows if row["step"] == "review"}
            contracts = {row["paper_id"]: row for row in rows if row["step"] == "contract"}
            stored_reviews = {}
            stored_specifications = {}
            for paper_id in jsonl:
                directory = target / "graphs" / safe_paper_directory(paper_id)
                stored_reviews[paper_id] = json.loads((directory / "paper_review.json").read_text(encoding="utf-8"))
                stored_specifications[paper_id] = json.loads((directory / "paper_method_inventory.json").read_text(encoding="utf-8"))

        self.assertEqual(sorted(stored_reviews), sorted(jsonl))
        self.assertEqual(sorted(stored_specifications), sorted(jsonl))
        for paper_id, payload in stored_reviews.items():
            self.assertEqual(payload, jsonl[paper_id])
            self.assertEqual(canonical_sha256(payload), reviews[paper_id]["artifact_sha256"]["paper_review.json"])
        for paper_id, payload in stored_specifications.items():
            self.assertEqual(
                canonical_sha256(payload),
                contracts[paper_id]["artifact_sha256"]["paper_method_inventory.json"],
            )

    def test_the_prepare_hash_describes_the_merged_report(self):
        with tempfile.TemporaryDirectory() as name:
            target, _written = self.seal(Path(name))
            report = json.loads((self.paper_dir(target) / "graph_report.json").read_text(encoding="utf-8"))
            prepare = [row for row in self.rows(target) if row["step"] == "prepare"][0]

        self.assertEqual(prepare["artifact_sha256"]["graph_report.json"], canonical_sha256(report))
        self.assertEqual(report["step"], "prepare")
        self.assertEqual(sorted(report["stages"]), sorted(IMPLEMENTED_STAGES))

    def test_editing_a_stored_layer_fails_validation(self):
        with tempfile.TemporaryDirectory() as name:
            target, _written = self.seal(Path(name))
            self.rewrite(self.paper_dir(target) / "scientific_graph.json", lambda payload: payload.update({"extra": 1}))
            checked = validate_pilot_graph_version(target)

        self.assertFalse(checked["ok"], checked["checks"])
        self.assertFalse(checked["sealed"], checked["checks"])
        self.assertFalse(checked["complete"], checked["checks"])
        mismatch = checked["checks"]["ledger_artifact_mismatches"][0]
        self.assertEqual(mismatch["artifact"], "scientific_graph.json")
        self.assertEqual(mismatch["step"], "graph")
        self.assertEqual(mismatch["reason"], "content_mismatch")
        self.assertNotEqual(mismatch["expected"], mismatch["actual"])

    def test_an_edited_review_row_fails_validation(self):
        with tempfile.TemporaryDirectory() as name:
            target, _written = self.seal(Path(name))
            reviews = target / "paper_reviews.jsonl"
            lines = [line for line in reviews.read_text(encoding="utf-8").splitlines() if line.strip()]
            first = json.loads(lines[0])
            first["extra"] = 1
            reviews.write_text(
                "\n".join([json.dumps(first, ensure_ascii=False)] + lines[1:]) + "\n", encoding="utf-8",
            )
            checked = validate_pilot_graph_version(target)

        self.assertFalse(checked["ok"], checked["checks"])
        mismatch = checked["checks"]["ledger_artifact_mismatches"][0]
        self.assertEqual(mismatch["artifact"], "review")
        self.assertEqual(mismatch["location"], "paper_reviews.jsonl")
        self.assertEqual(mismatch["reason"], "content_mismatch")

    def test_a_missing_stored_layer_is_reported(self):
        with tempfile.TemporaryDirectory() as name:
            target, _written = self.seal(Path(name))
            (self.paper_dir(target) / "figure_text_review.json").unlink()
            checked = validate_pilot_graph_version(target)

        self.assertFalse(checked["ok"], checked["checks"])
        mismatch = checked["checks"]["ledger_artifact_mismatches"][0]
        self.assertEqual(mismatch["artifact"], "figure_text_review.json")
        self.assertEqual(mismatch["reason"], "missing")
        self.assertIsNone(mismatch["actual"])
        self.assertTrue(checked["checks"]["missing_paper_artifacts"])

    def test_a_stale_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            target, _written = self.seal(Path(name))
            self.rewrite(target / "validation_report.json", lambda payload: payload.update({"ok": False}))
            checked = validate_pilot_graph_version(target)
            seal = checked["checks"]["seal"]

        self.assertFalse(checked["ok"], checked["checks"])
        self.assertFalse(seal["valid"], seal)
        self.assertFalse(seal["validation_report_sha256_matches"], seal)
        self.assertFalse(seal["validation_report_ok"], seal)
        self.assertFalse(checked["sealed"], checked["checks"])

    def test_an_unknown_recorded_artifact_is_reported(self):
        with tempfile.TemporaryDirectory() as name:
            target, _written = self.seal(Path(name))
            ledger = target / "graph_ledger.jsonl"
            rows = self.rows(target)
            rows[0]["artifact_sha256"]["mystery.json"] = "0" * 64
            ledger.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8",
            )
            checked = validate_pilot_graph_version(target)

        self.assertFalse(checked["ok"], checked["checks"])
        mismatch = checked["checks"]["ledger_artifact_mismatches"][0]
        self.assertEqual(mismatch["artifact"], "mystery.json")
        self.assertEqual(mismatch["reason"], "unknown_artifact")

    def test_an_empty_version_artifact_fails_validation(self):
        with tempfile.TemporaryDirectory() as name:
            target, _written = self.seal(Path(name))
            (target / "coverage_report.json").write_text("", encoding="utf-8")
            checked = validate_pilot_graph_version(target)

        self.assertFalse(checked["ok"], checked["checks"])
        self.assertEqual(checked["checks"]["empty_artifacts"], ["coverage_report.json"])

    def test_deleting_the_marker_reports_an_unsealed_version(self):
        with tempfile.TemporaryDirectory() as name:
            target, _written = self.seal(Path(name))
            (target / "COMPLETE").unlink()
            checked = validate_pilot_graph_version(target)

        self.assertTrue(checked["ok"], checked["checks"])
        self.assertFalse(checked["sealed"], checked["checks"])
        self.assertFalse(checked["complete"], checked["checks"])
        self.assertEqual(checked["checks"]["ledger_artifact_mismatches"], [])
        self.assertFalse(checked["checks"]["seal"]["present"])
        self.assertFalse(checked["checks"]["seal"]["valid"])

    def test_every_rendered_figure_ships_a_matching_render_ledger(self):
        with tempfile.TemporaryDirectory() as name:
            target, _written = self.seal(Path(name))
            checked = validate_pilot_graph_version(target)
            pngs = sorted(target.glob("figures/*/*.png"))
            ledgers = sorted(target.glob("figures/*/*.png.render.json"))

        self.assertEqual(len(pngs), len(PAPERS))
        self.assertEqual(len(ledgers), len(PAPERS))
        self.assertEqual(checked["checks"]["figure_render_mismatches"], [])
        self.assertTrue(checked["ok"], checked["checks"])

    def test_a_figure_without_its_render_ledger_fails_validation(self):
        with tempfile.TemporaryDirectory() as name:
            target, _written = self.seal(Path(name))
            ledger = sorted(target.glob("figures/*/*.png.render.json"))[0]
            ledger.unlink()
            checked = validate_pilot_graph_version(target)

        self.assertFalse(checked["ok"], checked["checks"])
        mismatch = checked["checks"]["figure_render_mismatches"][0]
        self.assertEqual(mismatch["reason"], "missing_ledger")
        self.assertTrue(mismatch["figure"].endswith(".png"), mismatch)

    def test_an_edited_render_ledger_fails_validation(self):
        with tempfile.TemporaryDirectory() as name:
            target, _written = self.seal(Path(name))
            ledger = sorted(target.glob("figures/*/*.png.render.json"))[0]
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            payload["png_sha256"] = "0" * 64
            ledger.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            checked = validate_pilot_graph_version(target)

        self.assertFalse(checked["ok"], checked["checks"])
        self.assertFalse(checked["sealed"], checked["checks"])
        mismatch = checked["checks"]["figure_render_mismatches"][0]
        self.assertEqual(mismatch["reason"], "ledger_hash_mismatch")
        self.assertEqual(mismatch["expected"], "0" * 64)
        self.assertNotEqual(mismatch["actual"], "0" * 64)


class ReproducibilityTests(unittest.TestCase):
    """G9: two independent runs must store the same bytes, figures included."""

    FULL = "prepare,review,contract,graph,visual,bind,index,check"
    VOLATILE = ("elapsed_seconds", "prompt_tokens", "completion_tokens", "total_tokens")

    def build(self, snapshot, pilot, *, figure_dir):
        with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.analysis.paper.review.call_vlm_json", fake_provider(FROZEN_EVIDENCE_ID)):
                with patch("rfs.analysis.paper.method_extractor.call_vlm_json", fake_method_provider(FROZEN_EVIDENCE_ID)):
                    with patch("rfs.judge_training.pilot_graphs.call_vlm_json", fake_visual_provider()):
                        with patch("rfs.judge_training.pilot_check.call_vlm_json", fake_check_provider()):
                            return build_pilot_graph_payloads(
                                snapshot, pilot, stages=self.FULL, workers=2, figure_dir=figure_dir,
                            )

    def contents(self, directory):
        """Fingerprint every stored file, so two runs can be compared byte for byte."""
        contents = {}
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(directory).as_posix()
            if path.suffix == ".png":
                contents[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            elif path.suffix == ".jsonl":
                rows = [
                    drop_volatile(json.loads(line), self.VOLATILE)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                contents[relative] = json.dumps(
                    sorted(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows), ensure_ascii=False,
                )
            else:
                payload = drop_render_flags(drop_volatile(json.loads(path.read_text(encoding="utf-8")), self.VOLATILE))
                contents[relative] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return contents

    def test_two_independent_runs_store_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            first = self.build(snapshot, pilot, figure_dir=root / "first" / "figures")
            write_pilot_graph_version(root / "first", first)
            second = self.build(snapshot, pilot, figure_dir=root / "second" / "figures")
            written = write_pilot_graph_version(root / "second", second)
            left = self.contents(root / "first")
            right = self.contents(root / "second")
            sidecar = json.loads(
                (root / "second" / "figures" / safe_paper_directory(PAPERS[0][0]) / "figure.png.render.json")
                .read_text(encoding="utf-8")
            )

        self.assertTrue(written["sealed"], written)
        self.assertEqual(sorted(name for name in set(left) | set(right) if left.get(name) != right.get(name)), [])
        self.assertIn("figures/p0001/figure.png", left)
        self.assertEqual(len([name for name in left if name.endswith(".png")]), len(PAPERS))
        self.assertEqual(sidecar["png_sha256"], left["figures/p0001/figure.png"])
        self.assertFalse(sidecar["reused"], sidecar)


class PilotLedgerGuardTests(unittest.TestCase):
    """G9: a Pilot version may never carry a split that must stay sealed."""

    FULL = "prepare,review,contract,graph,visual,bind,index,check"

    def plant(self, pilot, mutate):
        """Rewrite one field of the sealed Pilot ledger so the guard can be probed."""
        path = pilot / "pilot_ledger.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mutate(rows)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8",
        )

    def test_a_hidden_test_row_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _snapshot, pilot = write_fixture(root)
            self.plant(pilot, lambda rows: rows[1].update({"split": "hidden_test"}))
            with self.assertRaises(ValueError) as caught:
                load_pilot_ledger(pilot)
            message = str(caught.exception)

        self.assertIn("hidden_test", message)
        self.assertIn("must stay sealed", message)
        self.assertIn("pilot_ledger.jsonl", message)

    def test_a_row_without_a_split_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _snapshot, pilot = write_fixture(root)
            self.plant(pilot, lambda rows: rows[0].update({"split": ""}))
            with self.assertRaises(ValueError) as caught:
                load_pilot_ledger(pilot)

        self.assertIn("must stay sealed", str(caught.exception))

    def test_the_build_refuses_before_any_provider_call(self):
        counts = {}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            self.plant(pilot, lambda rows: rows[1].update({"split": "hidden_test"}))
            with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
                with patch("rfs.analysis.paper.review.call_vlm_json", counting_provider(fake_provider(FROZEN_EVIDENCE_ID), counts, "review")):
                    with patch("rfs.analysis.paper.method_extractor.call_vlm_json", counting_provider(fake_method_provider(FROZEN_EVIDENCE_ID), counts, "contract")):
                        with patch("rfs.judge_training.pilot_graphs.call_vlm_json", counting_provider(fake_visual_provider(), counts, "visual")):
                            with patch("rfs.judge_training.pilot_check.call_vlm_json", counting_provider(fake_check_provider(), counts, "check")):
                                with self.assertRaises(ValueError):
                                    build_pilot_graph_payloads(
                                        snapshot, pilot, stages=self.FULL, workers=2, figure_dir=root / "figures",
                                    )

        self.assertEqual(counts, {})

class ResumeConsistencyTests(ResumeGapTests):
    """A stage that failed must never come back as resumed work on the next run."""

    def test_resume_into_a_version_that_does_not_exist_yet_builds_it(self):
        """`--resume` on a version's first run is how a full corpus starts."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            fresh = root / "brand-new-version"
            counts = {}
            payloads = self.build(
                snapshot,
                pilot,
                stages=self.FULL,
                resume_dir=fresh,
                figure_dir=fresh / "figures",
                counts=counts,
            )

        self.assertEqual(set(payloads["stages_completed"]), set(self.FULL.split(",")))
        self.assertTrue(payloads["validation"]["ok"], payloads["validation"]["failures"])
        self.assertEqual(sorted(counts), ["check", "contract", "review", "visual"])

    def test_a_version_nothing_has_been_written_to_resumes_nothing(self):
        with tempfile.TemporaryDirectory() as name:
            state = load_resume_state(Path(name) / "not-written-yet")
        self.assertEqual(state["stages"], [])
        self.assertEqual(state["graphs"], {})
        self.assertEqual(state["visuals"], {})
        self.assertEqual(state["checks"], {})
        self.assertIsNone(state["index"])

    def test_a_failed_graph_row_does_not_resume_from_its_own_leftovers(self):
        victim = PAPERS[1][0]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            figures = root / "graphs" / "figures"
            first = self.build(snapshot, pilot, stages=self.HALF, figure_dir=figures)
            write_pilot_graph_version(root / "graphs", first)

            ledger_path = root / "graphs" / "graph_ledger.jsonl"
            rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            victim_row = [row for row in rows if row["step"] == "graph" and row["paper_id"] == victim][0]
            leftover = root / "graphs" / "graphs" / safe_paper_directory(victim) / "scientific_graph.json"
            self.assertTrue(leftover.is_file())

            rewritten = [row for row in rows if not (row["step"] == "graph" and row["paper_id"] == victim)]
            rewritten.append({**victim_row, "status": "failed", "error": "synthetic graph failure"})
            ledger_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + chr(10) for row in rewritten), encoding="utf-8")

            state = load_resume_state(root / "graphs")
            self.assertIn(PAPERS[0][0], state["graphs"])
            self.assertNotIn(victim, state["graphs"])
            self.assertNotIn(victim, state["graph_ledger"])

            counts = {}
            second = self.build(snapshot, pilot, stages=self.HALF, resume_dir=root / "graphs", figure_dir=figures, counts=counts)

        filled = {row["paper_id"]: row for row in second["ledger"] if row["step"] == "graph"}
        self.assertEqual(len(filled), len(PAPERS))
        self.assertEqual({row["status"] for row in filled.values()}, {"ok"})
        self.assertEqual(counts, {})
        self.assertTrue(second["validation"]["ok"], second["validation"]["failures"])

    def test_a_read_of_another_prompt_is_kept_as_input_but_not_as_completed_work(self):
        def stamp(path, mutate):
            payload = json.loads(path.read_text(encoding="utf-8"))
            mutate(payload)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            version = root / "graphs"
            first = self.build(snapshot, pilot, stages=self.FULL, figure_dir=version / "figures")
            write_pilot_graph_version(version, first)
            for directory in sorted((version / "graphs").iterdir()):
                if not directory.is_dir():
                    continue
                visual_path = directory / "visual_graph.json"
                if visual_path.is_file():
                    stamp(visual_path, lambda payload: payload["provenance"].update({"prompt_version": "pilot-visual-topology-v1"}))
                review_path = directory / "figure_text_review.json"
                if review_path.is_file():
                    stamp(review_path, lambda payload: payload.update({"prompt_version": "pilot-figure-check-v1"}))

            state = load_resume_state(version)
            counts = {}
            second = self.build(snapshot, pilot, stages="visual,check", resume_dir=version, figure_dir=root / "figures-recheck", counts=counts)

        self.assertEqual(sorted(state["visuals"]), [paper[0] for paper in PAPERS])
        self.assertEqual(sorted(state["checks"]), [paper[0] for paper in PAPERS])
        self.assertNotIn("visual", state["stages"])
        self.assertNotIn("check", state["stages"])
        self.assertEqual(sorted(counts), ["check", "visual"])
        rows = {row["step"]: row for row in second["ledger"] if row["paper_id"] == PAPERS[0][0]}
        self.assertEqual(rows["visual"]["status"], "ok")
        self.assertEqual(rows["check"]["status"], "ok")



class BindingCoverageTests(unittest.TestCase):
    """The run summary keeps the claim headline and the older all-object denominator apart."""

    def entry(self, **overrides):
        claims = {
            "objects": 10,
            "bound": 4,
            "outside_paper_method": 1,
            "in_paper_not_in_method": 1,
            "figure_annotation": 2,
            "unresolved": 2,
            "meaningful_objects": 8,
            "resolved_meaningful": 6,
            "roles": {"method_object": 5, "symbol": 2},
            "repeat_groups": 1,
            "repeat_instances": 2,
            "annotations_attached": 2,
            "annotations_unanchored": 0,
            "provenance_searched": 3,
            "provenance_found": 1,
        }
        claims.update(overrides)
        return {"binding_report": {"figure_claims": claims}}

    def test_the_headline_counts_meaningful_objects_while_the_all_object_ratio_stays_readable(self):
        coverage = binding_coverage({"p1": self.entry(), "p2": self.entry()})

        self.assertEqual(coverage["claim_objects"], 20)
        self.assertEqual(coverage["claim_meaningful_objects"], 16)
        self.assertEqual(coverage["claim_excluded_from_meaning"], 4)
        self.assertEqual(coverage["claim_coverage"], 0.75)
        self.assertEqual(coverage["diagnostic_claim_coverage_all_objects"], 0.5)

    def test_the_annotation_provenance_and_repetition_counts_are_carried_through(self):
        coverage = binding_coverage({"p1": self.entry()})

        self.assertEqual(coverage["claim_annotation_coverage"], 1.0)
        self.assertEqual(coverage["claim_annotations_attached"], 2)
        self.assertEqual(coverage["claim_provenance_searched"], 3)
        self.assertEqual(coverage["claim_provenance_found"], 1)
        self.assertEqual(coverage["claim_provenance_coverage"], 0.3333)
        self.assertEqual(coverage["claim_roles"]["symbol"], 2)
        self.assertEqual(coverage["claim_repeat_groups"], 1)
        self.assertEqual(coverage["claim_repeat_instances"], 2)

    def test_a_run_with_nothing_meaningful_never_divides_by_zero(self):
        empty = self.entry(
            objects=0,
            bound=0,
            outside_paper_method=0,
            in_paper_not_in_method=0,
            figure_annotation=0,
            unresolved=0,
            meaningful_objects=0,
            resolved_meaningful=0,
            provenance_searched=0,
            provenance_found=0,
        )
        coverage = binding_coverage({"p1": empty})

        self.assertEqual(coverage["claim_coverage"], 1.0)
        self.assertEqual(coverage["diagnostic_claim_coverage_all_objects"], 1.0)
        self.assertEqual(coverage["claim_provenance_coverage"], 0.0)


class ContractBlockedPaperTests(unittest.TestCase):
    """A paper without a stored review blocks its own row instead of killing the batch."""

    def test_contract_without_a_review_still_returns_the_four_contract_values(self):
        content, ledger_row, report, specification = contract_paper(
            "snapshot-is-not-touched", {}, {"paper_id": "p0001", "split": "train", "domain": "cv"}, None
        )

        self.assertIsNone(content)
        self.assertIsNone(specification)
        self.assertEqual(ledger_row["step"], CONTRACT_STEP)
        self.assertEqual(ledger_row["status"], "failed")
        self.assertEqual(ledger_row["token_source"], "not_called")
        self.assertFalse(report["ok"])

    def test_contract_all_blocks_review_less_papers_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot, pilot = write_fixture(root)
            rows = load_pilot_ledger(pilot)
            with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
                contents, ledger, reports, specifications = contract_all(snapshot, rows, reviews={})

        self.assertEqual(contents, {})
        self.assertEqual(specifications, {})
        self.assertEqual(len(ledger), len(rows))
        self.assertTrue(all(row["status"] == "failed" for row in ledger))
        self.assertTrue(reports and all(not report["ok"] for report in reports.values()))


if __name__ == "__main__":
    unittest.main()
