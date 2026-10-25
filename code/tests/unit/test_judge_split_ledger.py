"""Unit tests for the deterministic, domain-stratified split ledger."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.judge_training.split_ledger import (
    audit_leakage,
    bucket_u64,
    build_split_payloads,
    build_split_rows,
    domain_distribution,
    normalize_title,
    parse_ratios,
    split_for_bucket,
    split_thresholds,
    validate_split_version,
    write_split_version,
)

ARTIFACTS = (
    "split_manifest.json",
    "split_ledger.jsonl",
    "leakage_report.json",
    "domain_distribution.json",
    "hidden_test_manifest.json",
    "validation_report.json",
)


def make_paper(paper_id, domain="cv", **extra):
    row = {
        "paper_id": paper_id,
        "primary_domain": domain,
        "domain_labels": [domain],
        "title": paper_id,
        "figure_asset_tier": "A_source_verified",
        "identity_aliases": ["paper_id:%s" % paper_id],
    }
    row.update(extra)
    return row


def make_snapshot(root, papers, *, strict_unique=None, complete=True):
    snapshot = root / "paper-d0-v001"
    snapshot.mkdir(parents=True, exist_ok=True)
    with (snapshot / "eligible_papers.jsonl").open("w", encoding="utf-8") as stream:
        for item in papers:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "judge-source-snapshot-v1",
        "snapshot_id": "paper-d0-v001",
        "strict_unique_papers": len(papers) if strict_unique is None else strict_unique,
        "paper_set_sha256": "0" * 64,
    }
    (snapshot / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if complete:
        (snapshot / "COMPLETE").write_text(json.dumps({"manifest_sha256": "0" * 64}), encoding="utf-8")
    return snapshot


class RatioTests(unittest.TestCase):
    def test_parse_ratios_accepts_percent_and_fraction(self):
        self.assertEqual(parse_ratios("65/15/20"), (0.65, 0.15, 0.20))
        self.assertEqual(parse_ratios([0.65, 0.15, 0.2]), (0.65, 0.15, 0.2))
        self.assertEqual(parse_ratios(None), (0.65, 0.15, 0.20))

    def test_parse_ratios_rejects_bad_input(self):
        for bad in ("65/15", "60/15/20", "0/50/50", "abc/15/20"):
            with self.assertRaises(ValueError):
                parse_ratios(bad)


class BucketTests(unittest.TestCase):
    def test_bucket_is_stable_and_seed_sensitive(self):
        first = bucket_u64("2026.acl-long.1619", "seed-a")
        self.assertEqual(first, bucket_u64("2026.acl-long.1619", "seed-a"))
        self.assertNotEqual(first, bucket_u64("2026.acl-long.1619", "seed-b"))
        self.assertNotEqual(first, bucket_u64("2026.acl-long.1618", "seed-a"))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 1 << 64)

    def test_split_for_bucket_uses_absolute_thresholds(self):
        thresholds = split_thresholds((0.65, 0.15, 0.20))
        self.assertEqual(split_for_bucket(0, thresholds), "train")
        self.assertEqual(split_for_bucket(thresholds[0] - 1, thresholds), "train")
        self.assertEqual(split_for_bucket(thresholds[0], thresholds), "validation")
        self.assertEqual(split_for_bucket(thresholds[1] - 1, thresholds), "validation")
        self.assertEqual(split_for_bucket(thresholds[1], thresholds), "hidden_test")
        self.assertEqual(split_for_bucket((1 << 64) - 1, thresholds), "hidden_test")


class TitleTests(unittest.TestCase):
    def test_normalize_title_collapses_noise(self):
        self.assertEqual(normalize_title("  Hello, World!  "), "hello world")
        self.assertEqual(normalize_title("A  B\nC"), "a b c")
        self.assertIsNone(normalize_title("   "))
        self.assertIsNone(normalize_title(None))


class LedgerRowTests(unittest.TestCase):
    def test_rows_are_sorted_and_duplicates_are_rejected(self):
        rows = build_split_rows(
            [make_paper("b"), make_paper("a")], seed="s", ratios=(0.65, 0.15, 0.2)
        )
        self.assertEqual([row["paper_id"] for row in rows], ["a", "b"])
        self.assertTrue(all(row["split"] in ("train", "validation", "hidden_test") for row in rows))
        with self.assertRaises(ValueError):
            build_split_rows([make_paper("a"), make_paper("a")], seed="s", ratios=(0.65, 0.15, 0.2))
        with self.assertRaises(ValueError):
            build_split_rows([{"primary_domain": "cv"}], seed="s", ratios=(0.65, 0.15, 0.2))

    def test_rows_carry_every_identity_key(self):
        record = make_paper(
            "p1",
            identity_aliases=["paper_id:p1", "arxiv_id:2601.00001", "doi:10.1/x"],
        )
        row = build_split_rows([record], seed="s", ratios=(0.65, 0.15, 0.2))[0]
        self.assertEqual(row["identity_keys"]["arxiv_id"], ["2601.00001"])
        self.assertEqual(row["identity_keys"]["doi"], ["10.1/x"])
        self.assertEqual(row["identity_keys"]["title"], ["p1"])


class LeakageTests(unittest.TestCase):
    def test_audit_flags_a_key_spanning_two_splits(self):
        rows = [
            {"paper_id": "p1", "split": "train", "identity_keys": {"title": ["shared title"]}},
            {"paper_id": "p2", "split": "hidden_test", "identity_keys": {"title": ["shared title"]}},
        ]
        report = audit_leakage(rows)
        self.assertFalse(report["ok"])
        self.assertEqual(report["cross_split_duplicates"], 1)
        self.assertEqual(report["cross_split_examples"][0]["splits"], ["hidden_test", "train"])
        self.assertEqual(report["cross_split_examples"][0]["paper_ids"], ["p1", "p2"])

    def test_audit_allows_repeats_inside_one_split(self):
        rows = [
            {"paper_id": "p1", "split": "train", "identity_keys": {"title": ["shared title"]}},
            {"paper_id": "p2", "split": "train", "identity_keys": {"title": ["shared title"]}},
        ]
        report = audit_leakage(rows)
        self.assertTrue(report["ok"])
        self.assertEqual(report["cross_split_duplicates"], 0)
        self.assertEqual(report["within_split_duplicates"]["title"], 1)


class DistributionTests(unittest.TestCase):
    def test_distribution_reports_empty_cells_and_small_domains(self):
        rows = [
            {"paper_id": "p1", "split": "train", "primary_domain": "cv"},
            {"paper_id": "p2", "split": "train", "primary_domain": "biology"},
        ]
        report = domain_distribution(rows)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["totals"], {"train": 2, "validation": 0, "hidden_test": 0})
        self.assertIn({"domain": "biology", "split": "validation"}, report["empty_cells"])
        self.assertIn({"domain": "cv", "split": "hidden_test"}, report["empty_cells"])
        self.assertEqual(report["small_domains"], ["biology", "cv"])


class VersionTests(unittest.TestCase):
    def make_papers(self, count=300):
        return [make_paper("p%04d" % index, "cv" if index % 2 else "rag") for index in range(count)]

    def test_write_and_validate_round_trip_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, self.make_papers())
            report = {"ok": True, "papers": 1}
            first = build_split_payloads(
                snapshot, seed="seed-a", ratios="65/15/20", perceptual_hash_report=report
            )
            second = build_split_payloads(
                snapshot, seed="seed-a", ratios="65/15/20", perceptual_hash_report=report
            )
            left, right = root / "left", root / "right"
            write_split_version(left, first, perceptual_hash=report)
            write_split_version(right, second, perceptual_hash=report)
            for name in ARTIFACTS + ("COMPLETE",):
                self.assertTrue((left / name).is_file(), name)
                self.assertEqual((left / name).read_bytes(), (right / name).read_bytes(), name)
            result = validate_split_version(left)
            self.assertTrue(result["ok"], result)
            self.assertTrue(result["sealed"])
            self.assertEqual(result["papers"], 300)

    def test_complete_is_withheld_until_every_seal_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, self.make_papers(120))
            payloads = build_split_payloads(snapshot, seed="seed-a", ratios="65/15/20")
            self.assertFalse(payloads["seal_ready"])
            self.assertFalse(payloads["seal_gates"]["perceptual_hash_report_ok"])

            unsealed = root / "unsealed"
            written = write_split_version(unsealed, payloads)
            self.assertFalse(written["sealed"])
            self.assertFalse((unsealed / "COMPLETE").exists())

            failing = root / "failing"
            written = write_split_version(failing, payloads, perceptual_hash={"ok": False})
            self.assertFalse(written["sealed"])
            self.assertFalse((failing / "COMPLETE").exists())

            sealed = root / "sealed"
            with_report = build_split_payloads(
                snapshot, seed="seed-a", ratios="65/15/20", perceptual_hash_report={"ok": True}
            )
            written = write_split_version(sealed, with_report, perceptual_hash={"ok": True})
            self.assertTrue(written["sealed"])
            self.assertTrue((sealed / "COMPLETE").exists())
            complete = json.loads((sealed / "COMPLETE").read_text(encoding="utf-8"))
            self.assertEqual(complete["split_id"], "seed-a")

    def test_write_refuses_to_rewrite_a_sealed_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, self.make_papers(60))
            payloads = build_split_payloads(snapshot, seed="seed-a", ratios="65/15/20")
            out = root / "split"
            with_report = build_split_payloads(
                snapshot, seed="seed-a", ratios="65/15/20", perceptual_hash_report={"ok": True}
            )
            write_split_version(out, with_report, perceptual_hash={"ok": True})
            with self.assertRaises(FileExistsError):
                write_split_version(out, with_report, perceptual_hash={"ok": True})

    def test_validation_fails_when_the_source_snapshot_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, self.make_papers(60), complete=False)
            payloads = build_split_payloads(snapshot, seed="seed-a", ratios="65/15/20")
            self.assertFalse(payloads["validation"]["ok"])
            checks = {item["name"]: item["ok"] for item in payloads["validation"]["checks"]}
            self.assertFalse(checks["source_snapshot_complete"])
            self.assertTrue(checks["no_cross_split_leakage"])

    def test_validation_fails_when_the_paper_count_disagrees(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, self.make_papers(60), strict_unique=99)
            payloads = build_split_payloads(snapshot, seed="seed-a", ratios="65/15/20")
            checks = {item["name"]: item["ok"] for item in payloads["validation"]["checks"]}
            self.assertFalse(checks["source_paper_count_matches"])
            self.assertFalse(payloads["validation"]["ok"])


def make_snapshot_with_figure_shares(root, papers, shares):
    snapshot = make_snapshot(root, papers)
    with (snapshot / "asset_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for asset_id, digest in shares.items():
            stream.write(
                json.dumps({"asset_id": asset_id, "role": "figure", "sha256": digest}) + "\n"
            )
        stream.write(json.dumps({"asset_id": "asset-text", "role": "full_text", "sha256": "f" * 64}) + "\n")
    return snapshot


class FigureGroupTests(unittest.TestCase):
    def test_papers_sharing_a_figure_are_bound_to_one_split(self):
        papers = [make_paper("p%03d" % index) for index in range(400)]
        papers[0]["figures"] = [{"figure_id": "f0", "asset_ids": ["asset-a"]}]
        papers[-1]["figures"] = [{"figure_id": "f1", "asset_ids": ["asset-b"]}]
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = make_snapshot_with_figure_shares(
                Path(tmp), papers, {"asset-a": "a" * 64, "asset-b": "a" * 64}
            )
            payloads = build_split_payloads(snapshot, seed="seed-a", ratios="65/15/20")
            rows = {row["paper_id"]: row for row in payloads["rows"]}
            self.assertEqual(rows["p000"]["split"], rows["p399"]["split"])
            self.assertEqual(rows["p399"]["figure_group"], "p000")
            self.assertEqual(rows["p000"]["bucket_id"], "p000")
            self.assertEqual(rows["p399"]["bucket_u64"], rows["p000"]["bucket_u64"])
            self.assertEqual(
                rows["p399"]["identity_keys"]["figure_sha256"], ["a" * 64]
            )
            self.assertTrue(payloads["leakage"]["ok"])
            self.assertEqual(payloads["leakage"]["figure_groups"]["shared_groups"], 1)
            self.assertEqual(payloads["leakage"]["figure_groups"]["figure_assets_mapped"], 2)
            self.assertEqual(payloads["manifest"]["figure_groups_bound"], 1)
            checks = {item["name"]: item["ok"] for item in payloads["validation"]["checks"]}
            self.assertTrue(checks["bucket_rule_reproduced"])
            self.assertTrue(checks["shared_figures_stay_in_one_split"])
            self.assertTrue(checks["no_cross_split_leakage"])

    def test_unrelated_papers_keep_a_pure_hash_bucket_assignment(self):
        papers = [make_paper("p%03d" % index) for index in range(30)]
        for index, paper in enumerate(papers):
            paper["figures"] = [{"figure_id": "f%d" % index, "asset_ids": ["asset-%d" % index]}]
        shares = {"asset-%d" % index: "%064x" % index for index in range(30)}
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = make_snapshot_with_figure_shares(Path(tmp), papers, shares)
            payloads = build_split_payloads(snapshot, seed="seed-a", ratios="65/15/20")
            for row in payloads["rows"]:
                self.assertEqual(row["bucket_id"], row["paper_id"])
                self.assertIsNone(row["figure_group"])
                self.assertEqual(row["split"], split_for_bucket(row["bucket_u64"], payloads["thresholds"]))
            self.assertEqual(payloads["leakage"]["figure_groups"]["shared_groups"], 0)
            self.assertEqual(payloads["manifest"]["figure_groups_bound"], 0)

    def test_papers_sharing_a_rendered_figure_are_bound_to_one_split(self):
        papers = [make_paper("p%03d" % index) for index in range(400)]
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = make_snapshot(Path(tmp), papers)
            report = {
                "ok": True,
                "exact_groups": [
                    {
                        "dhash": "ab" * 8,
                        "assets": [{"paper_id": "p000"}, {"paper_id": "p399"}],
                    }
                ],
            }
            payloads = build_split_payloads(
                snapshot, seed="seed-a", ratios="65/15/20", perceptual_hash_report=report
            )
            rows = {row["paper_id"]: row for row in payloads["rows"]}
            self.assertEqual(rows["p000"]["split"], rows["p399"]["split"])
            self.assertEqual(rows["p399"]["figure_group"], "p000")
            self.assertEqual(payloads["manifest"]["figure_groups_bound"], 1)
            groups = payloads["leakage"]["figure_groups"]["groups"]
            self.assertEqual(groups[0]["reasons"], ["rendered_identical_figure"])
            self.assertEqual(payloads["perceptual"]["exact_cross_split_groups"], [])
            self.assertTrue(payloads["seal_ready"])
            self.assertTrue(all(payloads["seal_gates"].values()), payloads["seal_gates"])

    def test_a_report_is_required_before_any_version_can_be_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = make_snapshot(Path(tmp), [make_paper("p%03d" % index) for index in range(60)])
            payloads = build_split_payloads(snapshot, seed="seed-a", ratios="65/15/20")
            self.assertFalse(payloads["seal_gates"]["perceptual_hash_report_ok"])
            self.assertFalse(payloads["seal_gates"]["rendered_figures_do_not_cross_splits"])
            self.assertFalse(payloads["seal_ready"])
            checks = {item["name"]: item["ok"] for item in payloads["validation"]["checks"]}
            self.assertFalse(checks["perceptual_hash_report_ok"])
            self.assertFalse(checks["no_rendered_figure_crosses_splits"])
            self.assertFalse(payloads["validation"]["ok"])

    def test_snapshot_without_an_asset_manifest_still_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = make_snapshot(Path(tmp), [make_paper("p%03d" % index) for index in range(10)])
            self.assertFalse((snapshot / "asset_manifest.jsonl").exists())
            payloads = build_split_payloads(snapshot, seed="seed-a", ratios="65/15/20")
            self.assertEqual(payloads["leakage"]["figure_groups"]["figure_assets_mapped"], 0)
            self.assertTrue(payloads["leakage"]["ok"])

    def test_leakage_audit_flags_a_shared_figure_across_splits(self):
        report = audit_leakage(
            [
                {"paper_id": "a", "split": "train", "identity_keys": {"figure_sha256": ["d1"]}},
                {"paper_id": "b", "split": "hidden_test", "identity_keys": {"figure_sha256": ["d1"]}},
            ]
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["cross_split_duplicates"], 1)
        self.assertEqual(report["cross_split_examples"][0]["key"], "figure_sha256")


if __name__ == "__main__":
    unittest.main()
