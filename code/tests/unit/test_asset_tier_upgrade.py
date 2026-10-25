"""Tests for promoting ledger rows that now have an official standalone asset."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rfs.judge_training.asset_tier_upgrade import upgrade_asset_tiers


def _build(batch: Path, *, tier: str, with_text: bool = True) -> None:
    figure = batch / "assets" / "materials" / "official" / "paper" / "figures" / "figure_001.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)
    text = batch / "assets" / "materials" / "official" / "paper" / "text" / "full.txt"
    text.parent.mkdir(parents=True, exist_ok=True)
    text.write_text("methods " * 2000, encoding="utf-8")

    ledger = batch / "verified" / "materials" / "papers.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "paper_id": "10.1038/x", "doi": "10.1038/x", "title": "T", "year": 2026,
        "domain": "materials", "paper_type": "research_article",
        "official_record_status": "verified", "official_url": "https://doi.org/10.1038/x",
        "official_source": "crossref_journal", "figure_asset_tier": tier,
        "figure": {"caption": "old", "image_paths": []},
        "publisher_landing_evidence": {
            "resolved_url": "https://www.nature.com/articles/x", "title_found": True, "venue_found": True,
        },
    }
    if with_text:
        row["full_text_path"] = str(text.relative_to(batch))
    ledger.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    probe = batch / "manifests" / "official_asset_rows_materials.jsonl"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(
        json.dumps(
            {
                "record_id": "10.1038/x", "identifiers": {"doi": "10.1038/x"},
                "figure_asset_tier": "B_official_standalone", "standalone_asset": True,
                "status": "verified",
                "figure_assets": [str(figure.relative_to(batch))],
                "framework_candidates": [
                    {
                        "caption": "Schematic overview of the device.",
                        "caption_title": "Schematic overview of the device.",
                        "asset_path": str(figure.relative_to(batch)),
                        "asset_source_url": "https://media.springernature.com/x.png",
                        "binding_status": "verified",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


class AssetTierUpgradeTests(unittest.TestCase):
    def test_c_tier_row_is_promoted_to_the_bound_official_asset(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            _build(batch, tier="C_pdf_object_extracted")
            report = upgrade_asset_tiers(batch, domains=("materials",))
            self.assertEqual(report["domains"]["materials"]["upgraded"], 1)
            row = json.loads((batch / "verified" / "materials" / "papers.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["figure_asset_tier"], "B_official_standalone")
            self.assertEqual(row["asset_tier_upgrade"]["from"], "C_pdf_object_extracted")
            self.assertIn("Schematic overview", row["figure"]["caption"])

    def test_a_tier_row_is_never_downgraded(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            _build(batch, tier="A_source_verified")
            report = upgrade_asset_tiers(batch, domains=("materials",))
            self.assertEqual(report["domains"]["materials"]["upgraded"], 0)
            row = json.loads((batch / "verified" / "materials" / "papers.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["figure_asset_tier"], "A_source_verified")

    def test_promotion_that_would_fail_the_gates_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            _build(batch, tier="C_pdf_object_extracted", with_text=False)
            report = upgrade_asset_tiers(batch, domains=("materials",))
            self.assertEqual(report["domains"]["materials"]["upgraded"], 0)
            self.assertIn("fulltext_missing", str(report["domains"]["materials"]["refused"]))

    def test_dry_run_leaves_the_ledger_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            _build(batch, tier="C_pdf_object_extracted")
            ledger = batch / "verified" / "materials" / "papers.jsonl"
            before = ledger.read_bytes()
            report = upgrade_asset_tiers(batch, domains=("materials",), dry_run=True)
            self.assertEqual(report["domains"]["materials"]["upgraded"], 1)
            self.assertEqual(ledger.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
