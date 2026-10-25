"""Tests for supplying decodable page renders to fragment-bound records."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pymupdf
from PIL import Image

from rfs.judge_training.figure_rendering import (
    RENDER_METHOD,
    render_pending_source_figures,
    reader_name,
    rendered_page_asset,
)
from rfs.judge_training.literature_freeze import strict_failures
from rfs.judge_training.source_snapshot_assets import paper_asset_specs


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_paper_pdf(path: Path, pages: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    for number in range(1, pages + 1):
        page = document.new_page()
        page.insert_text((72, 72), f"page {number}")
    document.save(str(path))
    document.close()


class FigureRenderingTests(unittest.TestCase):
    def _fragment_row(self, batch: Path, domain: str, arxiv_id: str, *, tier: str, page: int, pdf_pages: int = 4) -> dict:
        asset = batch / "assets" / domain / arxiv_id
        (asset / "figures").mkdir(parents=True, exist_ok=True)
        (asset / "text").mkdir(parents=True, exist_ok=True)
        fragment = asset / "figures" / "framework_01_01.tex"
        fragment.write_text("\\begin{tikzpicture}\\end{tikzpicture}", encoding="utf-8")
        tex_file = asset / "main.tex"
        tex_file.write_text("\\includegraphics{figures/framework_01_01.tex}", encoding="utf-8")
        text = asset / "text" / "full.txt"
        text.write_text("complete paper text", encoding="utf-8")
        write_paper_pdf(asset / "arxiv.pdf", pdf_pages)
        return {
            "domain": domain,
            "paper_id": f"{domain}-{arxiv_id}",
            "arxiv_id": arxiv_id,
            "title": f"Paper {arxiv_id}",
            "official_url": f"https://publisher.test/{arxiv_id}",
            "official_record_status": "verified",
            "paper_type": "full",
            "figure_asset_tier": tier,
            "standalone_asset": False,
            "full_text_path": str(text),
            "figure": {
                "caption": "Overview of the proposed workflow",
                "image_paths": [str(fragment)],
                "binding_status": "verified",
                "tex_file": str(tex_file),
                "source_files": [str(fragment)],
                "pdf_page": page,
            },
        }

    def _batch(self, folder: str, domain: str, arxiv_id: str, **kwargs) -> tuple[Path, dict]:
        batch = Path(folder)
        row = self._fragment_row(batch, domain, arxiv_id, **kwargs)
        write_rows(batch / "verified" / domain / "papers.jsonl", [row])
        return batch, row

    def _standalone_row(self, batch: Path, domain: str, arxiv_id: str, *, tier: str) -> dict:
        asset = batch / "assets" / domain / arxiv_id
        (asset / "figures").mkdir(parents=True, exist_ok=True)
        (asset / "text").mkdir(parents=True, exist_ok=True)
        figure = asset / "figures" / "figure_001.png"
        Image.new("RGB", (4, 4), "white").save(figure)
        text = asset / "text" / "full.txt"
        text.write_text("complete paper text", encoding="utf-8")
        return {
            "domain": domain,
            "paper_id": f"{domain}-{arxiv_id}",
            "arxiv_id": arxiv_id,
            "title": f"Paper {arxiv_id}",
            "official_url": f"https://publisher.test/{arxiv_id}",
            "official_record_status": "verified",
            "paper_type": "full",
            "figure_asset_tier": tier,
            "standalone_asset": True,
            "full_text_path": str(text),
            "figure": {
                "caption": "Overview of the proposed workflow",
                "image_paths": [str(figure)],
                "binding_status": "verified",
                "pdf_page": 2,
            },
        }

    def test_fragment_bound_record_is_excluded_until_a_page_render_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            batch, row = self._batch(folder, "agent", "2501.00001", tier="A_source_verified", page=2)
            self.assertEqual(strict_failures(batch, "agent", row), ["figure_rendering_missing"])

            report = render_pending_source_figures(batch)
            self.assertEqual(report["rendered_total"], 1)
            self.assertEqual(report["skipped_total"], 0)
            self.assertTrue(report["outputs_changed"])

            updated = json.loads((batch / "verified" / "agent" / "papers.jsonl").read_text().splitlines()[0])
            self.assertEqual(strict_failures(batch, "agent", updated), [])

    def test_rendered_page_records_exact_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            batch, _ = self._batch(folder, "rag", "2501.00002", tier="A_source_verified", page=3)
            render_pending_source_figures(batch)

            updated = json.loads((batch / "verified" / "rag" / "papers.jsonl").read_text().splitlines()[0])
            figure = updated["figure"]
            render = figure["rendered_page"]
            self.assertEqual(render["method"], RENDER_METHOD)
            self.assertEqual(render["page"], 3)
            self.assertEqual(render["page_count"], 4)
            self.assertEqual(render["dpi"], 150)
            self.assertTrue(render["reader"].startswith("pypdfium2="))
            self.assertEqual(render["source_pdf_sha256"], render["source_pdf_sha256"].lower())
            self.assertEqual(len(render["source_pdf_sha256"]), 64)
            self.assertEqual(figure["image_paths"], [*figure["image_paths"][:1], render["path"]])
            self.assertEqual(rendered_page_asset(updated), render["path"])
            target = Path(render["path"])
            if not target.is_absolute():
                target = batch / target
            with Image.open(target) as image:
                self.assertGreater(image.width, 0)
                self.assertGreater(image.height, 0)
                self.assertEqual(image.format, "PNG")

    def test_second_run_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            batch, _ = self._batch(folder, "cv", "2501.00003", tier="A_source_verified", page=2)
            render_pending_source_figures(batch)
            ledger = batch / "verified" / "cv" / "papers.jsonl"
            original = ledger.read_bytes()

            second = render_pending_source_figures(batch)

            self.assertEqual(second["rendered_total"], 0)
            self.assertFalse(second["outputs_changed"])
            self.assertEqual(original, ledger.read_bytes())

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            batch, row = self._batch(folder, "cv", "2501.00004", tier="A_source_verified", page=2)
            ledger = batch / "verified" / "cv" / "papers.jsonl"
            original = ledger.read_bytes()

            report = render_pending_source_figures(batch, dry_run=True)

            self.assertEqual(report["rendered_total"], 1)
            self.assertFalse(report["outputs_changed"])
            self.assertEqual(original, ledger.read_bytes())
            self.assertEqual(list((batch / "assets" / "cv" / "2501.00004" / "figures").glob("*.png")), [])

    def test_c_tier_records_are_never_rendered(self):
        with tempfile.TemporaryDirectory() as folder:
            batch, row = self._batch(folder, "agent", "2501.00005", tier="C_pdf_object_extracted", page=2)
            self.assertIn("strict_ab_asset_missing", strict_failures(batch, "agent", row))

            report = render_pending_source_figures(batch)

            self.assertEqual(report["rendered_total"], 0)
            self.assertEqual(list((batch / "assets" / "agent" / "2501.00005" / "figures").glob("*.png")), [])

    def test_declared_page_outside_the_paper_pdf_is_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as folder:
            batch, _ = self._batch(folder, "agent", "2501.00006", tier="A_source_verified", page=9, pdf_pages=2)

            report = render_pending_source_figures(batch)

            self.assertEqual(report["rendered_total"], 0)
            self.assertEqual(report["skipped_total"], 1)
            self.assertIn("outside 2 pages", report["skipped"][0]["reason"])

    def test_source_snapshot_accepts_the_derived_render_role(self):
        with tempfile.TemporaryDirectory() as folder:
            batch, _ = self._batch(folder, "agent", "2501.00007", tier="A_source_verified", page=2)
            render_pending_source_figures(batch)
            updated = json.loads((batch / "verified" / "agent" / "papers.jsonl").read_text().splitlines()[0])

            specs, issues = paper_asset_specs(batch, updated)

            roles = {spec["role"] for spec in specs}
            self.assertIn("figure_page_render", roles)
            self.assertNotIn("figure", roles)
            render_asset = next(spec for spec in specs if spec["role"] == "figure_page_render")
            self.assertEqual(render_asset["storage"], "snapshot")
            self.assertEqual(render_asset["suffix"], ".png")
            self.assertNotIn("source_binding", render_asset)
            self.assertIn("paper_pdf", roles)
            self.assertIn("original_page_mapping_unavailable", issues)

    def test_reader_identity_is_reported(self):
        self.assertTrue(reader_name().startswith("pypdfium2="))

    def test_declared_page_render_path_is_read_across_hosts(self):
        render = {"method": RENDER_METHOD, "path": "assets\\agent\\2501.00009\\figures\\source_page_003.png"}

        self.assertEqual(rendered_page_asset({"figure": {"rendered_page": render}}), render["path"])

        renamed = dict(render, path="assets\\agent\\2501.00009\\figures\\figure_001.png")
        self.assertEqual(rendered_page_asset({"figure": {"rendered_page": renamed}}), "")

    def test_source_slot_holding_the_paper_pdf_is_used_as_the_original(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            row = self._standalone_row(batch, "biology", "2501.00010", tier="B_official_standalone")
            write_paper_pdf(batch / "assets" / "biology" / "2501.00010" / "arxiv_source.src", 4)

            specs, _ = paper_asset_specs(batch, row)

            roles = {spec["role"] for spec in specs}
            self.assertNotIn("source_archive", roles)
            pdf = next(spec for spec in specs if spec["role"] == "paper_pdf")
            self.assertEqual(pdf["source_suffix"], ".src")
            self.assertEqual(pdf["suffix"], ".pdf")
            self.assertEqual(pdf["source_resolution"]["method"],
                             "arxiv_eprint_returned_the_paper_pdf")


if __name__ == "__main__":
    unittest.main()
