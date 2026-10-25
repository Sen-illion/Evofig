"""Tests for sealing the papers that can carry a reference-versus-generated pair."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.cli import _run_judge_data, build_parser
from rfs.judge_training.pilot_graphs import safe_paper_directory
from rfs.workflows.corpus_subset import (
    SUBSET_CONTRACT,
    build_contract_subset,
    delivered_deck_names,
    reusable_slots_by_paper,
)

PAPER = "2401.00001"
SECOND = "2401.00002"
THIRD = "2401.00003"
FOURTH = "2401.00004"
COLON = "iclr:2026:10006567"


def _jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding="utf-8")
    return path


def _snapshot(root: Path, papers) -> Path:
    _jsonl(
        root / "eligible_papers.jsonl",
        [
            {
                "paper_id": paper_id,
                "primary_domain": domain,
                "title": f"title {paper_id}",
                "figures": [{"asset_ids": asset_ids}],
            }
            for paper_id, domain, asset_ids in papers
        ],
    )
    _jsonl(
        root / "asset_manifest.jsonl",
        [
            {
                "asset_id": asset_id,
                "role": "figure",
                "paper_id": paper_id,
                "relative_path": f"figures/{asset_id}.png",
                "suffix": ".png",
                "sha256": "0" * 64,
            }
            for paper_id, _, asset_ids in papers
            for asset_id in asset_ids
        ],
    )
    return root


def _contract(root: Path, folder: str, nodes=2) -> Path:
    target = root / folder
    target.mkdir(parents=True, exist_ok=True)
    plan = {
        "contract": "rfs.method_figure_content",
        "version": 2,
        "inputs": [{"id": "input_data", "label": "Input Data"}],
        "modules": [{"id": f"mod_{index}", "label": f"Module {index}"} for index in range(nodes)],
    }
    (target / "figure_content_plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    return target


def _legacy(root: Path, folder: str, *, icons=1, size=30_000) -> Path:
    attempt = root / folder / "attempt_01"
    assets = attempt / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    report = {"mode": "image2", "assets": [{"asset_id": f"icon_mod_{index}", "model": "gpt-image-2"} for index in range(icons)]}
    (attempt / "asset_generation_report.json").write_text(json.dumps(report), encoding="utf-8")
    for index in range(icons):
        icon = assets / f"icon_mod_{index}"
        icon.mkdir(parents=True, exist_ok=True)
        (icon / "selected.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * size)
        (icon / "prompt.txt").write_text(f"Create a framework-diagram icon for: Module {index}.", encoding="utf-8")
    return attempt


def _split(root: Path, rows) -> Path:
    _jsonl(root / "split_ledger.jsonl", rows)
    return root


def _fixture(root: Path) -> dict:
    snapshot = _snapshot(
        root / "snapshot",
        [
            (PAPER, "cv", ["fig-1"]),
            (SECOND, "cv", ["fig-2"]),
            (THIRD, "cv", ["fig-3"]),
            (FOURTH, "cv", []),
            (COLON, "rag", ["fig-5"]),
        ],
    )
    contracts = root / "contracts"
    for folder in (PAPER, SECOND, THIRD, FOURTH, "iclr_2026_10006567"):
        _contract(contracts, folder)
    legacy = root / "runs"
    for folder in (PAPER, FOURTH, "iclr_2026_10006567"):
        _legacy(legacy, folder)
    _legacy(legacy, SECOND, icons=1, size=100)
    return {"snapshot": snapshot, "contracts": contracts, "legacy": legacy}


class BuildContractSubsetTest(unittest.TestCase):
    def test_only_papers_with_both_halves_are_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = _fixture(root)
            report = build_contract_subset(
                parts["snapshot"],
                contracts=parts["contracts"],
                legacy_root=parts["legacy"],
                out=root / "subset",
                subset_id="paper-d0-sealed-v001",
            )
            manifest = json.loads((root / "subset" / "subset_manifest.json").read_text(encoding="utf-8"))
            ledger = [json.loads(line) for line in (root / "subset" / "subset_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            complete = (root / "subset" / "COMPLETE").read_text(encoding="utf-8")

        self.assertTrue(report["ok"])
        self.assertEqual(report["papers"], 3)
        self.assertEqual(report["requested"], 5)
        self.assertEqual(report["no_legacy_run"], 1)
        self.assertEqual(report["no_legacy_icon"], 1)
        self.assertEqual(manifest["contract"], SUBSET_CONTRACT)
        self.assertEqual(manifest["subset_id"], "paper-d0-sealed-v001")
        self.assertEqual([entry["paper_id"] for entry in ledger], [PAPER, FOURTH, COLON])
        self.assertEqual(manifest["contract_slots"], 9)
        self.assertIn("papers=3", complete)

    def test_the_names_file_lists_the_run_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = _fixture(root)
            report = build_contract_subset(
                parts["snapshot"],
                contracts=parts["contracts"],
                legacy_root=parts["legacy"],
                out=root / "subset",
                subset_id="paper-d0-sealed-v001",
                names_out=root / "names.txt",
            )
            names = (root / "names.txt").read_text(encoding="utf-8").split()

        self.assertEqual(names, sorted(names))
        self.assertEqual(names, ["2401.00001", "2401.00004", "iclr_2026_10006567"])
        self.assertEqual(report["names_out"], str(root / "names.txt"))

    def test_a_delivered_paper_is_dropped_and_a_half_drawn_one_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = _fixture(root)
            batch = root / "batch"
            delivered = batch / "runs" / PAPER / "attempt_01" / "editable_composition.pptx"
            delivered.parent.mkdir(parents=True, exist_ok=True)
            delivered.write_bytes(b"pptx")
            (batch / "runs" / safe_paper_directory(COLON) / "attempt_01").mkdir(parents=True, exist_ok=True)

            report = build_contract_subset(
                parts["snapshot"],
                contracts=parts["contracts"],
                legacy_root=parts["legacy"],
                out=root / "subset",
                subset_id="paper-d0-sealed-v001",
                exclude_drawn=[batch],
            )
            ledger = [json.loads(line) for line in (root / "subset" / "subset_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(report["already_drawn"], 1)
        self.assertEqual([entry["paper_id"] for entry in ledger], [FOURTH, COLON])

    def test_delivered_deck_names_ignore_an_attempt_without_a_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch" / "runs"
            (root / "10.1_x" / "attempt_01").mkdir(parents=True, exist_ok=True)
            (root / "10.1_y" / "attempt_01").mkdir(parents=True, exist_ok=True)
            (root / "10.1_y" / "attempt_01" / "editable_composition.pptx").write_bytes(b"pptx")

            found = delivered_deck_names([Path(tmp) / "batch"])

        self.assertEqual(found, {"10.1_y"})

    def test_the_split_ledger_labels_travel_with_the_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = _fixture(root)
            split = _split(
                root / "split",
                [
                    {"paper_id": PAPER, "split": "train"},
                    {"paper_id": FOURTH, "split": "hidden_test"},
                    {"paper_id": COLON, "split": "validation"},
                ],
            )
            build_contract_subset(
                parts["snapshot"],
                contracts=parts["contracts"],
                legacy_root=parts["legacy"],
                out=root / "subset",
                subset_id="paper-d0-sealed-v001",
                split=split,
            )
            manifest = json.loads((root / "subset" / "subset_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual({entry["paper_id"]: entry["split"] for entry in manifest["papers"]}, {PAPER: "train", FOURTH: "hidden_test", COLON: "validation"})
        self.assertEqual(manifest["source"]["split"], str(split))

    def test_a_paper_without_a_figure_is_counted_but_still_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = _fixture(root)
            report = build_contract_subset(
                parts["snapshot"],
                contracts=parts["contracts"],
                legacy_root=parts["legacy"],
                out=root / "subset",
                subset_id="paper-d0-sealed-v001",
            )

        self.assertEqual(report["no_figure"], 1)
        self.assertEqual(report["papers"], 3)

    def test_reusable_slots_follow_the_confirmed_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = _fixture(root)
            mapping = root / "out" / "mapping.jsonl"
            _jsonl(
                mapping,
                [
                    {
                        "paper": PAPER,
                        "records": [
                            {"slot": "icon_input_data", "route": "confident", "legacy_png": "/x/icon_mod_0/selected.png"},
                            {"slot": "icon_mod_0", "route": "adjudicated", "legacy_png": "/x/icon_mod_1/selected.png"},
                            {"slot": "icon_mod_1", "route": "review", "legacy_png": "/x/icon_mod_2/selected.png"},
                        ],
                    },
                    {"paper": FOURTH, "records": [{"slot": "icon_mod_0", "route": "confident", "legacy_png": "/x/icon_mod_0/selected.png"}]},
                ],
            )
            report = build_contract_subset(
                parts["snapshot"],
                contracts=parts["contracts"],
                legacy_root=parts["legacy"],
                out=root / "subset",
                subset_id="paper-d0-sealed-v001",
                mapping=mapping,
            )
            counts = reusable_slots_by_paper(mapping)

        self.assertEqual(report["contract_slots"], 9)
        self.assertEqual(report["reusable_slots"], 3)
        self.assertEqual(counts, {PAPER: 2, FOURTH: 1})

    def test_one_paper_cannot_own_two_run_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = _fixture(root)
            report = build_contract_subset(
                parts["snapshot"],
                contracts=parts["contracts"],
                legacy_root=parts["legacy"],
                out=root / "subset",
                subset_id="paper-d0-sealed-v001",
                limit=2,
            )
            ledger = [json.loads(line) for line in (root / "subset" / "subset_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(ledger), 2)
        self.assertEqual(report["papers"], 2)

    def test_a_sealed_subset_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = _fixture(root)
            build_contract_subset(
                parts["snapshot"],
                contracts=parts["contracts"],
                legacy_root=parts["legacy"],
                out=root / "subset",
                subset_id="paper-d0-sealed-v001",
            )
            with self.assertRaises(FileExistsError):
                build_contract_subset(
                    parts["snapshot"],
                    contracts=parts["contracts"],
                    legacy_root=parts["legacy"],
                    out=root / "subset",
                    subset_id="paper-d0-sealed-v001",
                )

    def test_a_missing_contract_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = _fixture(root)
            with self.assertRaises(FileNotFoundError):
                build_contract_subset(
                    parts["snapshot"],
                    contracts=root / "nope",
                    legacy_root=parts["legacy"],
                    out=root / "subset",
                    subset_id="paper-d0-sealed-v001",
                )


    def test_a_paper_without_a_reusable_icon_can_be_kept_on_purpose(self):
        """A wave that accepts paying for every icon keeps the paper instead of the halves."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = _fixture(root)
            report = build_contract_subset(
                parts["snapshot"],
                contracts=parts["contracts"],
                legacy_root=parts["legacy"],
                out=root / "subset",
                subset_id="paper-d0-sealed-v001",
                require_legacy_icons=False,
            )
            ledger = [json.loads(line) for line in (root / "subset" / "subset_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertTrue(report["ok"])
        self.assertEqual(report["papers"], 4)
        self.assertEqual(report["no_legacy_icon"], 0)
        self.assertIn(SECOND, [entry["paper_id"] for entry in ledger])
        self.assertEqual(report["contract_slots"], 12)


class ContractSubsetCommandTest(unittest.TestCase):
    def test_cli_parses_the_command(self):
        args = build_parser().parse_args([
            "judge-data", "build-contract-subset",
            "--snapshot", "s", "--contracts", "c", "--legacy-root", "l", "--out", "o",
            "--subset-id", "paper-d0-sealed-v001", "--names-out", "n.txt", "--exclude-drawn", "b",
        ])

        self.assertEqual(args.judge_data_action, "build-contract-subset")
        self.assertEqual(args.subset_id, "paper-d0-sealed-v001")
        self.assertEqual(args.names_out, "n.txt")
        self.assertEqual(args.exclude_drawn, ["b"])
        self.assertEqual(args.legacy_attempt, "attempt_01")
        self.assertFalse(args.allow_no_reusable_icon)

    def test_cli_dispatches_the_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = _fixture(root)
            args = build_parser().parse_args([
                "judge-data", "build-contract-subset",
                "--snapshot", str(parts["snapshot"]),
                "--contracts", str(parts["contracts"]),
                "--legacy-root", str(parts["legacy"]),
                "--out", str(root / "subset"),
                "--subset-id", "paper-d0-sealed-v001",
                "--names-out", str(root / "names.txt"),
                "--json",
            ])
            report = _run_judge_data(args)
            sealed = (root / "subset" / "COMPLETE").is_file()
            named = (root / "names.txt").is_file()

        self.assertTrue(report["ok"])
        self.assertEqual(report["papers"], 3)
        self.assertTrue(sealed)
        self.assertTrue(named)

    def test_the_cli_can_keep_papers_that_own_no_reusable_icon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = _fixture(root)
            args = build_parser().parse_args([
                "judge-data", "build-contract-subset",
                "--snapshot", str(parts["snapshot"]),
                "--contracts", str(parts["contracts"]),
                "--legacy-root", str(parts["legacy"]),
                "--out", str(root / "subset"),
                "--subset-id", "paper-d0-sealed-v001",
                "--allow-no-reusable-icon",
                "--json",
            ])
            report = _run_judge_data(args)

        self.assertTrue(report["ok"])
        self.assertEqual(report["papers"], 4)
        self.assertEqual(report["no_legacy_icon"], 0)


if __name__ == "__main__":
    unittest.main()
