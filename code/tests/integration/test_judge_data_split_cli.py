"""Integration tests for the judge-data split CLI commands."""

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rfs.cli import _run_judge_data, build_parser


def make_figure_png(seed):
    """Return a deterministic low-frequency figure that hashes to a real pattern."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(8, 9), dtype=np.uint8)
    image = Image.fromarray(small, "L").resize((240, 160), Image.NEAREST).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_snapshot(root, count=200, *, figure_assets=None):
    snapshot = root / "paper-d0-v001"
    snapshot.mkdir(parents=True, exist_ok=True)
    with (snapshot / "eligible_papers.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(count):
            domain = ("cv", "rag", "agent", "materials", "biology")[index % 5]
            row = {
                "paper_id": "p%04d" % index,
                "primary_domain": domain,
                "domain_labels": [domain],
                "title": "Paper %04d" % index,
                "figure_asset_tier": "A_source_verified",
                "identity_aliases": ["paper_id:p%04d" % index, "arxiv_id:2601.%05d" % index],
            }
            if figure_assets and index < len(figure_assets):
                row["figures"] = [
                    {"figure_id": "figure-%04d" % index, "asset_ids": [figure_assets[index]]}
                ]
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "judge-source-snapshot-v1",
        "snapshot_id": "paper-d0-v001",
        "strict_unique_papers": count,
        "paper_set_sha256": "1" * 64,
    }
    (snapshot / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (snapshot / "COMPLETE").write_text(json.dumps({"manifest_sha256": "2" * 64}), encoding="utf-8")
    return snapshot


class JudgeDataSplitCliTests(unittest.TestCase):
    def test_split_commands_parse(self):
        parser = build_parser()
        args = parser.parse_args(["judge-data", "split", "--snapshot", "d0", "--out", "split-v001"])
        self.assertEqual(args.command, "judge-data")
        self.assertEqual(args.judge_data_action, "split")
        self.assertEqual(args.ratios, "65/15/20")
        self.assertIsNone(args.perceptual_hash_report)
        args = parser.parse_args(["judge-data", "validate-split", "--split", "split-v001"])
        self.assertEqual(args.judge_data_action, "validate-split")
        args = parser.parse_args(
            ["judge-data", "perceptual-hash", "--snapshot", "d0", "--out", "phash.json"]
        )
        self.assertEqual(args.judge_data_action, "perceptual-hash")
        self.assertEqual(args.dpi, 72)
        self.assertEqual(args.near_distance, 4)

    def test_split_seals_only_with_a_passing_perceptual_hash_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, count=300)

            args = build_parser().parse_args(
                ["judge-data", "split", "--snapshot", str(snapshot), "--out", str(root / "split"), "--json"]
            )
            result = _run_judge_data(args)
            self.assertFalse(result["ok"], result)
            self.assertFalse(result["sealed"])
            self.assertFalse(result["seal_gates"]["perceptual_hash_report_ok"])
            self.assertFalse(result["seal_gates"]["rendered_figures_do_not_cross_splits"])
            self.assertEqual(result["papers"], 300)
            self.assertEqual(
                sum(result["split_counts"].values()),
                300,
            )
            self.assertTrue(result["leakage_ok"])
            self.assertEqual(result["empty_cells"], [])
            split_dir = root / "split"
            for name in (
                "split_manifest.json",
                "split_ledger.jsonl",
                "leakage_report.json",
                "domain_distribution.json",
                "hidden_test_manifest.json",
                "validation_report.json",
            ):
                self.assertTrue((split_dir / name).is_file(), name)
            self.assertFalse((split_dir / "COMPLETE").exists())

            report = root / "phash.json"
            report.write_text(json.dumps({"ok": True, "figures": 300}), encoding="utf-8")
            sealed_dir = root / "split-sealed"
            args = build_parser().parse_args(
                [
                    "judge-data",
                    "split",
                    "--snapshot",
                    str(snapshot),
                    "--out",
                    str(sealed_dir),
                    "--perceptual-hash-report",
                    str(report),
                    "--json",
                ]
            )
            result = _run_judge_data(args)
            self.assertTrue(result["sealed"], result)
            self.assertTrue(result["ok"], result)
            self.assertTrue(result["seal_gates"]["perceptual_hash_report_ok"])
            self.assertTrue((sealed_dir / "COMPLETE").exists())
            self.assertTrue((sealed_dir / "perceptual_hash_report.json").exists())

            check = _run_judge_data(
                build_parser().parse_args(["judge-data", "validate-split", "--split", str(sealed_dir)])
            )
            self.assertTrue(check["ok"], check)
            self.assertTrue(check["sealed"])
            self.assertEqual(check["papers"], 300)

            unsealed = _run_judge_data(
                build_parser().parse_args(["judge-data", "validate-split", "--split", str(root / "split")])
            )
            self.assertFalse(unsealed["sealed"])

    def test_sealed_versions_are_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, count=120)
            report = root / "phash.json"
            report.write_text(json.dumps({"ok": True}), encoding="utf-8")
            argv = [
                "judge-data",
                "split",
                "--snapshot",
                str(snapshot),
                "--out",
                str(root / "split"),
                "--perceptual-hash-report",
                str(report),
                "--json",
            ]
            self.assertTrue(_run_judge_data(build_parser().parse_args(argv))["sealed"])
            with self.assertRaises(FileExistsError):
                _run_judge_data(build_parser().parse_args(argv))

    def test_perceptual_hash_command_renders_figures_and_unlocks_the_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, count=120, figure_assets=["asset-0", "asset-1"])
            assets = snapshot / "assets"
            assets.mkdir(parents=True, exist_ok=True)
            figure = assets / "figure-00.png"
            figure.write_bytes(make_figure_png(11))
            figure_two = assets / "figure-01.png"
            figure_two.write_bytes(make_figure_png(11))
            with (snapshot / "asset_manifest.jsonl").open("w", encoding="utf-8") as stream:
                for index, item in enumerate((figure, figure_two)):
                    stream.write(
                        json.dumps(
                            {
                                "asset_id": "asset-%d" % index,
                                "paper_id": "p%04d" % index,
                                "role": "figure",
                                "relative_path": "assets/%s" % item.name,
                                "sha256": "0" * 64,
                                "suffix": ".png",
                            }
                        )
                        + "\n"
                    )

            report_path = root / "phash.json"
            report = _run_judge_data(
                build_parser().parse_args(
                    [
                        "judge-data",
                        "perceptual-hash",
                        "--snapshot",
                        str(snapshot),
                        "--out",
                        str(report_path),
                        "--workers",
                        "1",
                        "--json",
                    ]
                )
            )
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["figures"], 2)
            self.assertEqual(report["hashed"], 2)
            self.assertEqual(report["exact_groups"], 1)

            split_dir = root / "split"
            result = _run_judge_data(
                build_parser().parse_args(
                    [
                        "judge-data",
                        "split",
                        "--snapshot",
                        str(snapshot),
                        "--out",
                        str(split_dir),
                        "--perceptual-hash-report",
                        str(report_path),
                        "--json",
                    ]
                )
            )
            self.assertTrue(result["sealed"], result)
            self.assertTrue(all(result["seal_gates"].values()), result["seal_gates"])
            self.assertEqual(result["figure_groups_bound"], 1)
            written = json.loads((split_dir / "leakage_report.json").read_text(encoding="utf-8"))
            self.assertTrue(written["perceptual_hash"]["ok"])
            self.assertEqual(len(written["perceptual_hash"]["exact_cross_split_groups"]), 0)

    def test_split_rejects_a_snapshot_without_a_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "empty"
            snapshot.mkdir(parents=True, exist_ok=True)
            args = build_parser().parse_args(
                ["judge-data", "split", "--snapshot", str(snapshot), "--out", str(root / "split"), "--json"]
            )
            with self.assertRaises(FileNotFoundError):
                _run_judge_data(args)


if __name__ == "__main__":
    unittest.main()
