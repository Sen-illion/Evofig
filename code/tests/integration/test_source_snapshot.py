"""Real local assets exercise immutable capture, relocation and failure recovery."""

import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import pymupdf
from PIL import Image

from rfs.cli import build_parser
from rfs.judge_training.literature_freeze import DOMAINS, build_strict_training_view
from rfs.judge_training.literature_guard import literature_batch_guard
from rfs.judge_training.source_snapshot import freeze_source_snapshot
from rfs.judge_training.source_snapshot_content import DOCUMENTS
from rfs.judge_training.source_snapshot_inventory import audit_source_generation
from rfs.judge_training.source_snapshot_io import archive_path, sha256_file, strict_jsonl, write_jsonl
from rfs.judge_training.source_snapshot_validation import validate_source_snapshot
from rfs.utils import write_json


def _exit_snapshot_worker(_job):
    """Simulate an abrupt native decoder exit inside its isolated process."""
    os._exit(77)


class SourceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.batch = self.root / "collection/batch"
        self.out = self.root / "snapshots/d0"
        self.archive = self.root / "archive"
        self.project = self.root / "project"
        for name in DOCUMENTS:
            p = self.project / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"exact authority bytes")
        (self.project / "rfs").mkdir()
        (self.project / "rfs/example.py").write_text('"""Example."""', encoding="utf-8")
        self.rows = [self.paper("A", "A_source_verified"), self.paper("B", "B_official_standalone"),
                     self.paper("C", "C_pdf_object_extracted")]
        for domain in DOMAINS:
            write_jsonl(self.batch / "verified" / domain / "papers.jsonl", self.rows if domain == "agent" else [])
        self.git_patch = patch("rfs.judge_training.source_snapshot_content.subprocess.run")
        self.git_patch.start().return_value.stdout = "a" * 40
        self.addCleanup(self.git_patch.stop)
        self.writer_patch = patch("rfs.judge_training.source_snapshot.active_legacy_writers", return_value=[])
        self.writer_patch.start()
        self.addCleanup(self.writer_patch.stop)

    def paper(self, name, tier):
        relative = "assets/agent/" + name
        assets = self.batch / relative
        assets.mkdir(parents=True)
        text = assets / "text/full.txt"
        text.parent.mkdir()
        text.write_text("Complete method evidence.\n", encoding="utf-8")
        write_jsonl(text.parent / "pages.jsonl", [{"page": 1, "text": "Complete method evidence.\n"}])
        with pymupdf.open() as doc:
            doc.new_page().insert_text((30, 30), "Complete method evidence.")
            doc.save(assets / "arxiv.pdf")
        image = assets / "figure.png"
        Image.new("RGB", (32, 24), "white").save(image)
        with tarfile.open(assets / "arxiv_source.src", "w:gz") as archive:
            for member, content in (("main.tex", b"source caption"), ("figure.png", image.read_bytes())):
                info = tarfile.TarInfo(member)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        return {"paper_id": name, "arxiv_id": name, "title": "Paper " + name, "domain": "agent",
                "official_record_status": "verified", "official_url": "https://publisher.example/" + name,
                "official_source": "official", "paper_type": "long", "figure_asset_tier": tier,
                "full_text_path": relative + "/text/full.txt",
                "split": "hidden_test", "standalone_asset": tier == "B_official_standalone",
                "figure": {"caption": "Full framework " + name, "image_paths": [relative + "/figure.png"],
                           "tex_file": "main.tex", "source_files": ["figure.png"], "pdf_page": 1,
                           "binding_status": "verified", "asset_source_url": "https://publisher.example/fig"}}

    def freeze(self, **kwargs):
        return freeze_source_snapshot(self.batch, self.out, self.archive, project_root=self.project, **kwargs)

    def test_capture_relocates_and_survives_source_changes(self):
        result = self.freeze()
        self.assertTrue(result["ok"], result)
        self.assertEqual(2, result["strict_unique_papers"])
        self.assertFalse(result["quota_met"])
        papers = strict_jsonl(self.out / "eligible_papers.jsonl")
        self.assertEqual({"A", "B"}, {p["paper_id"] for p in papers})
        self.assertNotIn("split", papers[0])
        self.assertEqual("hidden_test", papers[0]["provenance"]["source_split"])
        before = {p.relative_to(self.out): (sha256_file(p), p.stat().st_mtime_ns) for p in self.out.rglob("*") if p.is_file()}
        (self.batch / "assets/agent/A/text/full.txt").write_text("changed", encoding="utf-8")
        self.assertEqual("already_complete", self.freeze()["status"])
        self.assertEqual(before, {p.relative_to(self.out): (sha256_file(p), p.stat().st_mtime_ns) for p in self.out.rglob("*") if p.is_file()})
        moved, moved_archive = self.root / "moved/snapshot", self.root / "moved/archive"
        shutil.copytree(self.out, moved)
        shutil.copytree(self.archive, moved_archive)
        self.batch.rename(self.root / "unavailable_source")
        self.assertTrue(validate_source_snapshot(moved, moved_archive)["ok"])

    def test_subscription_preview_is_preserved_in_exclusion_evidence(self):
        text = self.batch / self.rows[1]["full_text_path"]
        text.write_text("This is a preview of subscription content. Abstract and references.", encoding="utf-8")
        self.assertTrue(self.freeze()["ok"])
        self.assertEqual({"A"}, {r["paper_id"] for r in strict_jsonl(self.out / "eligible_papers.jsonl")})
        exclusions = json.loads((self.out / "exclusion_summary.json").read_text(encoding="utf-8"))
        self.assertEqual([{"paper_id": "B", "domain": "agent", "status": "needs_complete_paper_text"}],
                         exclusions["pending_full_text"])
        self.assertTrue(text.exists())
        self.assertTrue(validate_source_snapshot(self.out, self.archive)["ok"])

    def test_corruption_deletion_and_incomplete_publication_fail(self):
        self.assertTrue(self.freeze()["ok"])
        marker = (self.out / "COMPLETE").read_bytes()
        (self.out / "COMPLETE").unlink()
        self.assertFalse(validate_source_snapshot(self.out, self.archive)["ok"])
        self.assertFalse(self.freeze()["ok"])
        (self.out / "COMPLETE").write_bytes(marker)
        asset = next(self.archive.glob("sha256/*/*"))
        content = asset.read_bytes()
        asset.unlink()
        self.assertFalse(validate_source_snapshot(self.out, self.archive)["ok"])
        asset.write_bytes(content + b"corrupt")
        self.assertFalse(validate_source_snapshot(self.out, self.archive)["ok"])

    def test_bad_inputs_never_silently_disappear(self):
        ledger = self.batch / "verified/agent/papers.jsonl"
        original = ledger.read_bytes()
        ledger.write_bytes(original + b"{broken}\n")
        self.assertFalse(self.freeze()["ok"])
        self.assertFalse(self.out.exists())
        ledger.write_bytes(original)
        (self.batch / "assets/agent/A/figure.png").write_bytes(b"invalid image")
        result = self.freeze()
        self.assertEqual("source_audit_failed", result["status"], result)
        self.assertFalse(self.out.exists())

    def test_valid_but_wrong_figure_is_rejected_by_source_binding(self):
        Image.new("RGB", (32, 24), "black").save(self.batch / "assets/agent/A/figure.png")
        result = self.freeze()
        self.assertEqual("source_audit_failed", result["status"], result)
        self.assertIn("do not match", result["errors"][0]["error"])

    def test_source_fragments_are_excluded_upstream_with_a_visible_reason(self):
        fragment = self.batch / "assets/agent/A/framework.tex"
        fragment.write_text("source instructions only", encoding="utf-8")
        self.rows[0]["figure"]["image_paths"] = [fragment.relative_to(self.batch).as_posix()]
        write_jsonl(self.batch / "verified/agent/papers.jsonl", self.rows)
        result = self.freeze()
        self.assertTrue(result["ok"], result)
        self.assertEqual(1, result["strict_unique_papers"])
        audit = strict_jsonl(self.out / "eligibility_audit.jsonl")
        row = next(r for r in audit if r["paper_id"] == "A")
        self.assertFalse(row["eligible"])
        self.assertIn("figure_rendering_missing", row["reasons"])
        excluded = json.loads((self.out / "exclusion_summary.json").read_text(encoding="utf-8"))
        self.assertEqual("A", excluded["pending_figure_rendering"][0]["paper_id"])
        self.assertEqual(3, len(strict_jsonl(self.batch / "verified/agent/papers.jsonl")))

    def test_unicode_paper_caption_survives_strict_rebuild_and_capture(self):
        self.rows[0]["figure"]["caption"] = "Framework\u0085with\u2028Unicode\u2029text"
        write_jsonl(self.batch / "verified/agent/papers.jsonl", self.rows)
        result = self.freeze()
        self.assertTrue(result["ok"], result)
        self.assertEqual(2, result["strict_unique_papers"])

    def test_empty_fulltext_and_disk_shortfall_do_not_publish(self):
        with patch("rfs.judge_training.source_snapshot.estimate_storage", return_value={"ok": False}):
            self.assertEqual("insufficient_disk", self.freeze()["status"])
        (self.batch / "assets/agent/A/text/full.txt").write_text("   ", encoding="utf-8")
        self.assertEqual("source_audit_failed", self.freeze()["status"])
        self.assertFalse(self.out.exists())

    def test_busy_writers_return_retry_without_rebuilding(self):
        with literature_batch_guard(self.batch, purpose="collector"):
            self.assertEqual("waiting_for_batch", self.freeze()["status"])
        with patch("rfs.judge_training.source_snapshot.active_legacy_writers", return_value=[{"pid": 123}]):
            self.assertEqual("waiting_for_batch", self.freeze()["status"])
        self.assertFalse((self.batch / "verified/strict_training_papers.jsonl").exists())

    def test_abrupt_decoder_exit_fails_capture_without_publishing(self):
        with patch("rfs.judge_training.source_snapshot_inventory.inspect_paper_assets", _exit_snapshot_worker):
            result = self.freeze()
        self.assertFalse(result["ok"], result)
        self.assertEqual("failed", result["status"])
        self.assertFalse(self.out.exists())
        self.assertTrue(list((self.out.parent / "_work/d0").glob("failure-*.json")))

    def test_overlapped_fulltext_is_not_a_completed_asset_batch(self):
        events = [{"stage": "asset-bind", "status": "started"},
                  {"stage": "fulltext", "status": "completed"}]
        path = self.batch / "manifests/multisource_collection_events.jsonl"
        write_jsonl(path, events)
        self.assertEqual("waiting_for_batch", self.freeze()["status"])
        events.append({"stage": "asset-bind", "status": "completed"})
        write_jsonl(path, events)
        self.assertEqual("waiting_for_batch", self.freeze()["status"])
        events.extend([{"stage": "fulltext", "status": "completed"}, {"stage": "freeze", "status": "completed"}])
        write_jsonl(path, events)
        self.assertTrue(self.freeze()["ok"])

    def test_interrupted_copy_resumes_without_overwriting_published_snapshot(self):
        from rfs.judge_training.source_snapshot_content import copy_verified
        count = 0
        def interrupted(*args):
            nonlocal count
            count += 1
            if count == 3:
                raise OSError("simulated interruption")
            return copy_verified(*args)
        with patch("rfs.judge_training.source_snapshot_content.copy_verified", side_effect=interrupted):
            self.assertFalse(self.freeze()["ok"])
        self.assertFalse(self.out.exists())
        result = self.freeze()
        self.assertTrue(result["ok"], result)

    def test_source_change_during_copy_is_rejected(self):
        from rfs.judge_training.source_snapshot_content import materialize_snapshot
        def changed(*args):
            materialize_snapshot(*args)
            ledger = self.batch / "verified/agent/papers.jsonl"
            ledger.write_bytes(ledger.read_bytes() + b"\n")
        with patch("rfs.judge_training.source_snapshot.materialize_snapshot", side_effect=changed):
            result = self.freeze()
        self.assertFalse(result["ok"], result)
        self.assertIn("changed", result["error"].lower())
        self.assertFalse(self.out.exists())

    def test_alias_overlap_requires_resolution_and_domain_dedupe_is_retained(self):
        self.rows[0]["doi"] = "10.123/example"
        self.rows[1]["doi"] = "https://doi.org/10.123/example"
        write_jsonl(self.batch / "verified/agent/papers.jsonl", self.rows)
        result = self.freeze()
        self.assertFalse(result["ok"])
        self.assertIn("overlap", result["error"])
        self.rows[1].pop("doi")
        write_jsonl(self.batch / "verified/agent/papers.jsonl", self.rows)
        write_jsonl(self.batch / "verified/rag/papers.jsonl", [self.rows[1]])
        result = self.freeze()
        self.assertTrue(result["ok"], result)
        self.assertEqual(2, result["strict_unique_papers"])
        self.assertEqual(1, result["domain_report"]["domain_labels"]["rag"])

    def test_preflight_and_cli_contract(self):
        result = self.freeze(preflight=True)
        self.assertEqual("preflight_passed", result["status"], result)
        self.assertFalse(self.out.exists())
        parser = build_parser()
        args = parser.parse_args(["judge-data", "freeze-source", "--batch-root", "b", "--out", "o", "--archive-root", "a"])
        self.assertEqual("freeze-source", args.judge_data_action)
        self.assertEqual("all", args.scope)
        scoped = parser.parse_args(["judge-data", "freeze-source", "--batch-root", "b", "--out", "o",
                                    "--archive-root", "a", "--scope", "a-source"])
        self.assertEqual("a-source", scoped.scope)
        args = parser.parse_args(["judge-data", "validate-source-snapshot", "--snapshot", "s", "--archive-root", "a"])
        self.assertEqual("validate-source-snapshot", args.judge_data_action)

    def test_named_scope_publishes_its_tiers_and_states_what_it_left_out(self):
        from rfs.contracts.source_snapshot import SCOPE_TIERS
        result = self.freeze(tiers=SCOPE_TIERS["a-source"])
        self.assertTrue(result["ok"], result)
        self.assertEqual(1, result["strict_unique_papers"])
        self.assertEqual({"A"}, {p["paper_id"] for p in strict_jsonl(self.out / "eligible_papers.jsonl")})
        manifest = json.loads((self.out / "snapshot_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(["A_source_verified"], manifest["scope"]["figure_asset_tiers"])
        self.assertEqual(2, manifest["scope"]["strict_view_papers"])
        self.assertEqual(1, manifest["scope"]["out_of_scope_strict_papers"])
        exclusions = json.loads((self.out / "exclusion_summary.json").read_text(encoding="utf-8"))
        self.assertEqual({"B_official_standalone": 1}, exclusions["out_of_scope_records"])
        # The audit still rules on every ledger record, not only the published tier.
        audit = strict_jsonl(self.out / "eligibility_audit.jsonl")
        self.assertEqual({"A", "B", "C"}, {r["paper_id"] for r in audit})
        self.assertTrue(validate_source_snapshot(self.out, self.archive)["ok"])

    def test_exact_binding_repair_retains_old_lists_without_dropping_papers(self):
        extra = self.batch / "assets/agent/A/unrelated.png"
        Image.new("RGB", (32, 24), "red").save(extra)
        self.rows[0]["figure"]["image_paths"].append(extra.relative_to(self.batch).as_posix())
        ledger = self.batch / "verified/agent/papers.jsonl"
        write_jsonl(ledger, self.rows)
        before = ledger.read_bytes()
        result = self.freeze(repair_bindings=True, preflight=True)
        self.assertEqual(1, result["proposed_repairs"], result)
        self.assertEqual(before, ledger.read_bytes())
        result = self.freeze(repair_bindings=True)
        self.assertTrue(result["ok"], result)
        self.assertEqual(2, result["strict_unique_papers"])
        row = strict_jsonl(ledger)[0]
        self.assertEqual(1, len(row["figure"]["image_paths"]))
        self.assertEqual(2, len(row["source_binding_repair"]["previous_image_paths"]))
        backups = list((self.batch / "manifests/source_binding_repairs").glob("*/agent.before.jsonl"))
        self.assertEqual(before, backups[0].read_bytes())
        report = json.loads(backups[0].with_name("report.json").read_text(encoding="utf-8"))
        for item in report["files"]:
            self.assertEqual(item["after_sha256"], sha256_file(self.batch / item["relative_path"]))
        self.assertTrue(extra.is_file())

    def test_repair_journal_blocks_collection_and_recovers_interruption(self):
        from rfs.judge_training.literature_binding_repair import apply_binding_repairs, prepare_binding_repairs
        from rfs.judge_training.literature_guard import guarded_collection
        extra = self.batch / "assets/agent/A/unrelated.png"
        Image.new("RGB", (32, 24), "red").save(extra)
        self.rows[0]["figure"]["image_paths"].append(extra.relative_to(self.batch).as_posix())
        write_jsonl(self.batch / "verified/agent/papers.jsonl", self.rows)
        with literature_batch_guard(self.batch, purpose="repair-test"):
            plan = prepare_binding_repairs(self.batch)
            with patch("rfs.judge_training.literature_binding_repair._resume", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    apply_binding_repairs(self.batch, plan)
        called = []
        result = guarded_collection(lambda *a, **k: called.append(True), self.batch)
        self.assertEqual("waiting_for_binding_repair_recovery", result["status"])
        self.assertFalse(called)
        self.assertEqual("waiting_for_binding_repair_recovery", self.freeze()["status"])
        with literature_batch_guard(self.batch, purpose="repair-recovery"):
            self.assertEqual("complete", apply_binding_repairs(self.batch)["status"])
        self.assertTrue(self.freeze()["ok"])

    def test_wrong_or_missing_extracted_figure_is_recovered_from_declared_source(self):
        image = self.batch / "assets/agent/A/figure.png"
        original = image.read_bytes()
        Image.new("RGB", (32, 24), "black").save(image)
        wrong = image.read_bytes()
        result = self.freeze(repair_bindings=True)
        self.assertTrue(result["ok"], result)
        repaired = strict_jsonl(self.batch / "verified/agent/papers.jsonl")[0]
        selected = self.batch / repaired["figure"]["image_paths"][0]
        self.assertEqual(original, selected.read_bytes())
        self.assertEqual(wrong, image.read_bytes())
        self.assertTrue(repaired["source_binding_repair"]["recovered_source_assets"])

    def test_binding_repair_keeps_the_declared_page_render(self):
        relative = "assets/agent/A"
        render = self.batch / relative / "source_page_003.png"
        Image.new("RGB", (32, 24), "blue").save(render)
        extra = self.batch / relative / "unrelated.png"
        Image.new("RGB", (32, 24), "red").save(extra)
        row = self.rows[0]
        row["figure"]["rendered_page"] = {"method": "pdfium_page_render",
                                            "path": relative + "/source_page_003.png"}
        row["figure"]["image_paths"] = [relative + "/figure.png", relative + "/unrelated.png",
                                          relative + "/source_page_003.png"]
        ledger = self.batch / "verified/agent/papers.jsonl"
        write_jsonl(ledger, self.rows)
        result = self.freeze(repair_bindings=True)
        self.assertTrue(result["ok"], result)
        repaired = strict_jsonl(ledger)[0]["figure"]["image_paths"]
        # The unrelated image is dropped, but the record keeps a decodable figure:
        # a render is derived from the record's own PDF, never a source member.
        self.assertEqual([relative + "/figure.png", relative + "/source_page_003.png"], repaired)
        self.assertEqual({"A", "B"},
                         {p["paper_id"] for p in strict_jsonl(self.out / "eligible_papers.jsonl")})

    def test_case_sensitive_source_members_and_missing_panel_are_preserved(self):
        folder = self.batch / "assets/agent/A"
        figure = (folder / "figure.png").read_bytes()
        panel = io.BytesIO()
        Image.new("RGB", (16, 16), "blue").save(panel, format="PNG")
        with tarfile.open(folder / "arxiv_source.src", "w:gz") as archive:
            for name, content in (("main.tex", b"source caption"), ("Figure.png", b"different case"),
                                  ("figure.png", figure), ("panel2.png", panel.getvalue())):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        self.rows[0]["figure"]["source_files"].append("panel2.png")
        write_jsonl(self.batch / "verified/agent/papers.jsonl", self.rows)
        result = self.freeze(repair_bindings=True)
        self.assertTrue(result["ok"], result)
        repaired = strict_jsonl(self.batch / "verified/agent/papers.jsonl")[0]
        self.assertEqual(2, len(repaired["figure"]["image_paths"]))

    def test_pdf_content_with_png_name_is_frozen_as_pdf(self):
        folder = self.batch / "assets/agent/A"
        pdf_bytes = (folder / "arxiv.pdf").read_bytes()
        (folder / "figure.png").write_bytes(pdf_bytes)
        with tarfile.open(folder / "arxiv_source.src", "w:gz") as archive:
            for name, content in (("main.tex", b"source caption"), ("figure.png", pdf_bytes)):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        result = self.freeze()
        self.assertTrue(result["ok"], result)
        assets = strict_jsonl(self.out / "asset_manifest.jsonl")
        figure = next(a for a in assets if a["paper_id"] == "A" and a["role"] == "figure")
        self.assertEqual(".pdf", figure["suffix"])
        self.assertEqual(".png", figure["source_suffix"])
        self.assertEqual("pdfium", figure["readability"]["decoder"])

    def test_png_content_with_pdf_name_keeps_original_bytes(self):
        folder = self.batch / "assets/agent/A"
        original = folder / "figure.png"
        digest = sha256_file(original)
        mislabeled = original.with_suffix(".pdf")
        original.rename(mislabeled)
        self.rows[0]["figure"]["image_paths"] = ["assets/agent/A/figure.pdf"]
        write_jsonl(self.batch / "verified/agent/papers.jsonl", self.rows)
        result = self.freeze()
        self.assertTrue(result["ok"], result)
        assets = strict_jsonl(self.out / "asset_manifest.jsonl")
        figure = next(a for a in assets if a["paper_id"] == "A" and a["role"] == "figure")
        self.assertEqual(".png", figure["suffix"])
        self.assertEqual(".pdf", figure["source_suffix"])
        self.assertEqual(digest, figure["sha256"])
        self.assertEqual(digest, sha256_file(self.out / figure["relative_path"]))
        self.assertTrue(validate_source_snapshot(self.out, self.archive)["ok"])

    def test_cross_domain_pdf_requires_identical_source_archive(self):
        source = self.batch / "assets/agent/A"
        other = self.batch / "assets/rag/A"
        shutil.copytree(source, other)
        (source / "arxiv.pdf").unlink()
        (other / "arxiv_source.src").write_bytes(b"different source version")
        self.assertEqual("source_audit_failed", self.freeze()["status"])
        shutil.copyfile(source / "arxiv_source.src", other / "arxiv_source.src")
        result = self.freeze()
        self.assertTrue(result["ok"], result)
        assets = strict_jsonl(self.out / "asset_manifest.jsonl")
        pdf = next(a for a in assets if a["paper_id"] == "A" and a["role"] == "paper_pdf")
        self.assertEqual("same_batch_arxiv_id_and_source_archive_sha256", pdf["source_resolution"]["method"])

    def test_archive_object_shared_with_a_collection_entry_is_accepted(self):
        """A collection entry may hold the same inode as its archive object.

        Two names for one immutable object cannot be rewritten in place without
        corrupting the collection too, and the publisher must not require the
        project to spend a second copy of every shared source file.
        """
        self.assertTrue(self.freeze()["ok"])
        shared = self.batch / "assets/agent/A/arxiv_source.src"
        stored = self.archive / archive_path(sha256_file(shared))
        self.assertEqual(1, stored.stat().st_nlink)
        shared.unlink()
        os.link(stored, shared)
        self.assertEqual(2, stored.stat().st_nlink)
        self.assertEqual(stored.stat().st_ino, shared.stat().st_ino)
        self.assertTrue(validate_source_snapshot(self.out, self.archive)["ok"])

    def test_linked_archive_object_is_still_rejected(self):
        """A link, unlike a shared inode, can resolve outside the archive."""
        self.assertTrue(self.freeze()["ok"])
        stored = sorted(p for p in self.archive.rglob("*") if p.is_file())[0]
        target = stored.with_name(stored.name + ".moved")
        stored.rename(target)
        try:
            stored.symlink_to(target)
        except OSError:
            # Creating a symlink needs an extra privilege on Windows, so stand
            # the link in and still exercise the refusal branch here.
            stored.write_bytes(b"placeholder")
            with patch.object(Path, "is_symlink", return_value=True):
                result = validate_source_snapshot(self.out, self.archive)
            self.assertFalse(result["ok"])
            self.assertIn("links", result["error"])
            return
        result = validate_source_snapshot(self.out, self.archive)
        self.assertFalse(result["ok"])
        self.assertIn("links", result["error"])

    def _reseal_test_fixture(self):
        from rfs.judge_training.source_snapshot import _seal
        manifest = json.loads((self.out / "snapshot_manifest.json").read_text(encoding="utf-8"))
        _seal(self.out, manifest)
        write_json(self.out / "COMPLETE", {"manifest_sha256": sha256_file(self.out / "snapshot_manifest.json")})

    def test_valid_hashes_cannot_hide_contradictory_binding_files(self):
        self.assertTrue(self.freeze()["ok"])
        paper = strict_jsonl(self.out / "eligible_papers.jsonl")[0]
        path = self.out / paper["bindings_path"]
        binding = json.loads(path.read_text(encoding="utf-8"))
        binding["asset_ids"] = []
        write_json(path, binding)
        self._reseal_test_fixture()
        result = validate_source_snapshot(self.out, self.archive)
        self.assertFalse(result["ok"])
        self.assertIn("Bindings file", result["error"])

    def test_valid_hashes_cannot_hide_wrong_source_provenance(self):
        self.assertTrue(self.freeze()["ok"])
        papers = strict_jsonl(self.out / "eligible_papers.jsonl")
        papers[0]["provenance"]["source_record"]["title"] = "Wrong paper"
        write_jsonl(self.out / "eligible_papers.jsonl", papers)
        self._reseal_test_fixture()
        result = validate_source_snapshot(self.out, self.archive)
        self.assertFalse(result["ok"])
        self.assertIn("frozen strict input", result["error"])

    def test_valid_hashes_cannot_hide_inconsistent_builder_version(self):
        self.assertTrue(self.freeze()["ok"])
        path = self.out / "schema_versions.json"
        versions = json.loads(path.read_text(encoding="utf-8"))
        versions["builder_git_commit"] = "b" * 40
        write_json(path, versions)
        self._reseal_test_fixture()
        result = validate_source_snapshot(self.out, self.archive)
        self.assertFalse(result["ok"])
        self.assertIn("Builder/schema provenance", result["error"])
