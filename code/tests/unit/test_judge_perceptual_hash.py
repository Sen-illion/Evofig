"""Unit tests for the rendered-figure perceptual hash report."""

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rfs.judge_training.perceptual_hash import (
    audit_perceptual_leakage,
    build_perceptual_hash_report,
    dhash64,
    figure_assets,
    near_pairs,
    write_perceptual_hash_report,
)


def figure_png(seed, size=(240, 160)):
    """Return a deterministic low-frequency figure with a real dHash pattern."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(8, 9), dtype=np.uint8)
    image = Image.fromarray(small, "L").resize(size, Image.NEAREST).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def write_snapshot(root, figures):
    """Write a snapshot whose asset manifest lists the given figure files."""
    snapshot = root / "paper-d0-v001"
    snapshot.mkdir(parents=True, exist_ok=True)
    with (snapshot / "asset_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for index, item in enumerate(figures):
            target = snapshot / item["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item["payload"])
            stream.write(
                json.dumps(
                    {
                        "asset_id": "asset-%02d" % index,
                        "paper_id": item["paper_id"],
                        "role": item.get("role", "figure"),
                        "relative_path": item["relative_path"],
                        "sha256": item.get("sha256", "0" * 64),
                        "suffix": Path(item["relative_path"]).suffix,
                    }
                )
                + "\n"
            )
    (snapshot / "snapshot_manifest.json").write_text(
        json.dumps({"snapshot_id": "paper-d0-v001"}), encoding="utf-8"
    )
    return snapshot


def figures_for(pairs, *, size=(240, 160)):
    """Return figure rows for (paper_id, seed) pairs, one file each."""
    rows = []
    for index, (paper_id, seed) in enumerate(pairs):
        rows.append(
            {
                "paper_id": paper_id,
                "relative_path": "assets/paper-%02d/figure-%02d.png" % (index, index),
                "payload": figure_png(seed, size),
            }
        )
    return rows


class HashTests(unittest.TestCase):
    def test_dhash_is_stable_and_discriminating(self):
        first = Image.open(io.BytesIO(figure_png(1)))
        second = Image.open(io.BytesIO(figure_png(1)))
        other = Image.open(io.BytesIO(figure_png(2)))
        self.assertEqual(dhash64(first), dhash64(second))
        self.assertNotEqual(dhash64(first), dhash64(other))
        self.assertEqual(len(dhash64(first)), 8)

    def test_flat_images_hash_to_zero_bits(self):
        white = Image.new("RGB", (120, 80), "white")
        black = Image.new("RGB", (120, 80), "black")
        self.assertEqual(dhash64(white), bytes(8))
        self.assertEqual(dhash64(black), bytes(8))

    def test_near_pairs_respects_the_distance(self):
        report = near_pairs(["00" * 8, "00" * 8, "ff" * 8], 4)
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["pairs"][0]["left"], 0)
        self.assertEqual(report["pairs"][0]["right"], 1)
        self.assertEqual(report["pairs"][0]["distance"], 0)
        self.assertFalse(report["truncated"])

    def test_near_pairs_needs_two_hashes(self):
        self.assertEqual(near_pairs(["00" * 8], 4)["total"], 0)


class ReportTests(unittest.TestCase):
    def test_report_is_deterministic_and_finds_identical_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = write_snapshot(
                root,
                figures_for([("p1", 5), ("p2", 5), ("p3", 6)]),
            )
            first = build_perceptual_hash_report(snapshot, workers=1)
            second = build_perceptual_hash_report(snapshot, workers=1)
            parallel = build_perceptual_hash_report(snapshot, workers=2)

            def comparable(report):
                clone = json.loads(json.dumps(report))
                clone["method"].pop("workers")
                return json.dumps(clone, sort_keys=True)

            self.assertEqual(comparable(first), comparable(second))
            self.assertEqual(comparable(first), comparable(parallel))
            self.assertEqual(second["method"]["workers"], 1)
            self.assertEqual(parallel["method"]["workers"], 2)
            self.assertTrue(first["ok"])
            self.assertEqual(first["figures_total"], 3)
            self.assertEqual(first["hashed"], 3)
            self.assertEqual(first["failed"], 0)
            self.assertEqual(first["exact_groups_total"], 1)
            group = first["exact_groups"][0]
            self.assertEqual([asset["paper_id"] for asset in group["assets"]], ["p1", "p2"])
            self.assertEqual(len(group["dhash"]), 16)

    def test_report_fails_loudly_on_an_unreadable_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = figures_for([("p1", 7)])
            rows.append(
                {
                    "paper_id": "p2",
                    "relative_path": "assets/paper-99/figure-99.png",
                    "payload": b"not an image",
                }
            )
            snapshot = write_snapshot(root, rows)
            report = build_perceptual_hash_report(snapshot, workers=1)
            self.assertFalse(report["ok"])
            self.assertEqual(report["failed"], 1)
            self.assertEqual(report["hashed"], 1)
            self.assertEqual(
                report["failures"][0]["relative_path"], "assets/paper-99/figure-99.png"
            )
            self.assertIn("not an image", report["failures"][0]["error"] + " not an image")

    def test_report_only_covers_figure_role_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = figures_for([("p1", 8)])
            rows.append(
                {
                    "paper_id": "p1",
                    "relative_path": "assets/paper-00/text-00.png",
                    "payload": figure_png(9),
                    "role": "full_text",
                }
            )
            snapshot = write_snapshot(root, rows)
            self.assertEqual(
                [row["relative_path"] for row in figure_assets(snapshot)],
                [rows[0]["relative_path"]],
            )
            report = build_perceptual_hash_report(snapshot, workers=1)
            self.assertEqual(report["figures_total"], 1)

    def test_write_report_returns_its_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "phash.json"
            digest = write_perceptual_hash_report(target, {"ok": True})
            self.assertEqual(len(digest), 64)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"ok": True})


class LeakageTests(unittest.TestCase):
    def make_report(self, papers):
        return {
            "ok": True,
            "exact_groups": [{"dhash": "ab" * 8, "assets": [{"paper_id": p} for p in papers]}],
        }

    def test_a_rendered_figure_across_two_splits_is_leakage(self):
        rows = [{"paper_id": "p1", "split": "train"}, {"paper_id": "p2", "split": "hidden_test"}]
        audit = audit_perceptual_leakage(rows, self.make_report(["p1", "p2"]))
        self.assertFalse(audit["ok"])
        self.assertEqual(len(audit["exact_cross_split_groups"]), 1)

    def test_a_rendered_figure_inside_one_split_is_fine(self):
        rows = [{"paper_id": "p1", "split": "train"}, {"paper_id": "p2", "split": "train"}]
        audit = audit_perceptual_leakage(rows, self.make_report(["p1", "p2"]))
        self.assertTrue(audit["ok"])

    def test_a_failing_report_blocks_the_audit(self):
        rows = [{"paper_id": "p1", "split": "train"}]
        audit = audit_perceptual_leakage(rows, {"ok": False, "exact_groups": []})
        self.assertFalse(audit["ok"])
        self.assertTrue(audit["supplied"])
        self.assertFalse(audit["report_ok"])

    def test_a_missing_report_blocks_the_audit(self):
        audit = audit_perceptual_leakage([{"paper_id": "p1", "split": "train"}], None)
        self.assertFalse(audit["ok"])
        self.assertFalse(audit["supplied"])
        self.assertIsNone(audit["report_ok"])


if __name__ == "__main__":
    unittest.main()