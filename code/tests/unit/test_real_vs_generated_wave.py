"""Tests for collecting reference pairs across a whole wave of sealed-contract runs."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.cli import _run_judge_data, build_parser
from rfs.judge_training.real_vs_generated_wave import (
    collect_reference_pairs,
    contract_adoption,
    delivered_attempt,
    failure_reason,
    namespace_examples,
    read_examples,
    training_split,
    write_examples,
)
from tests.unit.test_judge_real_vs_generated import (
    BLOCKED_GATE,
    FOLDER,
    GROUND_TRUTH,
    PAPER_ID,
    PASSED_GATE,
    _middleware,
    _run,
)


def _subset(root: Path, rows) -> Path:
    target = root / "subset"
    target.mkdir(parents=True, exist_ok=True)
    (target / "subset_manifest.json").write_text(
        json.dumps({"contract": "rfs.corpus_subset", "version": 1, "subset_id": "paper-d0-sealed-v001"}),
        encoding="utf-8",
    )
    (target / "subset_ledger.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding="utf-8"
    )
    return target


def _entry(paper_id: str = PAPER_ID, directory: str = FOLDER, split: str = "train") -> dict:
    return {"paper_id": paper_id, "directory": directory, "split": split}


def _wave(root: Path, *, gate=BLOCKED_GATE, contract=None, name: str = FOLDER, adoption=None, attempt: str = "attempt_01", delivered: bool = True) -> Path:
    run = root / "wave" / "runs" / name / attempt
    _run(run, gates={0: gate}, contract=contract)
    if adoption is not None:
        (run / "upstream_contract_report.json").write_text(json.dumps(adoption), encoding="utf-8")
    if delivered:
        (run / "editable_composition.pptx").write_bytes(b"pptx")
    return run


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SplitTest(unittest.TestCase):
    def test_a_hidden_test_paper_never_becomes_a_training_split(self):
        self.assertEqual(training_split("hidden_test"), "test")
        self.assertEqual(training_split("test"), "test")
        self.assertEqual(training_split("train"), "train")
        self.assertEqual(training_split("validation"), "validation")
        self.assertEqual(training_split(""), "train")
        self.assertEqual(training_split(None), "train")


class NamespaceTest(unittest.TestCase):
    def test_ids_are_prefixed_with_the_paper(self):
        rows = namespace_examples("paper_a", [{"example_id": "attempt_01-real-vs-generated-r00-c00"}, {"example_id": ""}])

        self.assertEqual([row["example_id"] for row in rows], ["paper_a-attempt_01-real-vs-generated-r00-c00"])
        self.assertEqual(rows[0]["paper_a"] if "paper_a" in rows[0] else "", "")


class FailureReasonTest(unittest.TestCase):
    def test_reasons_are_reduced_to_stable_codes(self):
        self.assertEqual(failure_reason(RuntimeError("zhu: run ab != reference cd, they disagree about the scientific contract of x")), "contract_mismatch")
        self.assertEqual(failure_reason(RuntimeError("nothing here carries a Grounding Gate verdict")), "no_gate_blocked_candidate")
        self.assertEqual(failure_reason(RuntimeError("weird")), "RuntimeError: weird")


class ContractAdoptionTest(unittest.TestCase):
    def test_adoption_reads_the_run_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _wave(Path(tmp), adoption={"used": True})

            self.assertEqual(contract_adoption(run), "adopted")

    def test_a_missing_or_declined_report_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = _wave(root, name=FOLDER)
            declined = _wave(root, name="second_paper", adoption={"used": False})

            self.assertEqual((contract_adoption(missing), contract_adoption(declined)), ("missing", "declined"))


class CollectReferencePairsTest(unittest.TestCase):
    def _collect(self, root: Path, entries, out, **kwargs):
        _middleware(root / "version")
        subset = _subset(root, entries)
        return collect_reference_pairs(
            subset,
            root / "version",
            root / "wave",
            root / out,
            **kwargs,
        )

    def test_a_blocked_candidate_becomes_one_example_and_its_mirror(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _wave(root)
            report = self._collect(root, [_entry()], "pairs")
            merged = _rows(root / "pairs" / "examples.jsonl")
            per_paper = _rows(root / "pairs" / "papers" / f"{FOLDER}.jsonl")
            accounted = (root / "pairs" / "examples_manifest.json").is_file()
            reported = (root / "pairs" / "wave_report.json").is_file()

        self.assertTrue(report["ok"])
        self.assertEqual(report["papers_with_examples"], 1)
        self.assertEqual(report["examples_written"], 2)
        self.assertEqual(report["by_tier"], {"gate_blocked": 1})
        self.assertEqual(report["by_defect"], {"missing_relation": 1})
        self.assertEqual(report["by_split"], {"train": 1})
        self.assertEqual(report["by_contract_route"], {"byte_identical": 1})
        self.assertEqual(len(merged), 2)
        self.assertEqual(len(per_paper), 2)
        self.assertEqual(
            [row["example_id"] for row in merged],
            [f"{FOLDER}-attempt_01-real-vs-generated-r00-c00", f"{FOLDER}-attempt_01-real-vs-generated-r00-c00__swap"],
        )
        self.assertTrue(accounted)
        self.assertTrue(reported)

    def test_two_papers_of_one_wave_never_share_an_example_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _wave(root, name=FOLDER)
            _wave(root, name="2401.00001")
            report = self._collect(root, [_entry(), _entry(directory="2401.00001")], "pairs")
            merged = _rows(root / "pairs" / "examples.jsonl")

        self.assertEqual(len(merged), 4)
        self.assertEqual(len({row["example_id"] for row in merged}), 4)
        self.assertEqual(report["duplicate_example_ids"], 0)

    def test_a_hidden_test_paper_is_collected_as_a_test_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _wave(root)
            report = self._collect(root, [_entry(split="hidden_test")], "pairs")
            merged = _rows(root / "pairs" / "examples.jsonl")
            manifest = json.loads((root / "pairs" / "examples_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(report["by_split"], {"test": 1})
        self.assertEqual({row["split"] for row in merged}, {"test"})
        self.assertEqual(manifest["by_split"], {"test": 1})

    def test_a_candidate_that_passed_the_gate_is_a_named_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _wave(root, gate=PASSED_GATE)
            report = self._collect(root, [_entry()], "pairs")

            written = (root / "pairs" / "papers" / f"{FOLDER}.jsonl").exists()

        self.assertEqual(report["refusals"], {"no_gate_blocked_candidate": 1})
        self.assertEqual(report["examples_written"], 0)
        self.assertFalse(written)

    def test_the_passed_candidates_can_be_counted_as_worse_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _wave(root, gate=PASSED_GATE)
            report = self._collect(root, [_entry()], "pairs", include_passed=True)

        self.assertEqual(report["examples_written"], 2)
        self.assertEqual(report["by_tier"], {"assumed_worse": 1})

    def test_a_run_that_drew_a_different_contract_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _wave(root, contract={**GROUND_TRUTH, "required_edge_ids": []})
            report = self._collect(root, [_entry()], "pairs")

        self.assertEqual(report["refusals"], {"contract_mismatch": 1})
        self.assertEqual(report["examples_written"], 0)

    def test_a_paper_the_wave_never_drew_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self._collect(root, [_entry()], "pairs")

        self.assertEqual(report["refusals"], {"wave_has_no_run": 1})

    def test_merging_keeps_one_row_per_example_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _wave(root)
            base = root / "full-002-examples.jsonl"
            base.write_text(
                chr(10).join([
                    json.dumps({"example_id": "older-example", "task_type": "scientific_quality"}),
                    json.dumps({"example_id": f"{FOLDER}-attempt_01-real-vs-generated-r00-c00", "task_type": "old-copy"}),
                ]) + chr(10),
                encoding="utf-8",
            )
            report = self._collect(root, [_entry()], "pairs", merge_into=base)
            merged = _rows(root / "pairs" / "examples_merged.jsonl")
            manifest = json.loads((root / "pairs" / "examples_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(report["merged"]["base_examples"], 2)
        self.assertEqual(report["merged"]["new_examples"], 2)
        self.assertEqual(report["merged"]["merged_examples"], 3)
        self.assertEqual(merged[0]["example_id"], "older-example")
        self.assertEqual(merged[1]["task_type"], "old-copy")
        self.assertEqual(manifest["contract_adoption"], {"missing": 1})

    def test_the_wave_counts_the_contract_each_run_adopted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _wave(root, adoption={"used": True})
            _wave(root, name="2401.00001", adoption={"used": False})
            report = self._collect(root, [_entry(), _entry(directory="2401.00001")], "pairs")

        self.assertEqual(report["contract_adoption"], {"adopted": 1, "declined": 1})

    def test_a_subset_without_a_ledger_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _middleware(root / "version")
            (root / "subset").mkdir()
            with self.assertRaises(FileNotFoundError):
                collect_reference_pairs(root / "subset", root / "version", root / "wave", root / "pairs")


class DeliveredAttemptTest(unittest.TestCase):
    def test_the_newest_delivered_attempt_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = _wave(root, attempt="attempt_01")
            newer = _wave(root, attempt="attempt_02")

            self.assertEqual(delivered_attempt(root / "wave" / "runs" / FOLDER), newer)
            self.assertNotEqual(older, newer)

    def test_a_staged_directory_is_not_mistaken_for_a_run(self):
        """A directory that only carries pre-staged icons delivered no figure."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged = root / "wave" / "runs" / FOLDER / "attempt_01" / "round_00" / "candidate_00" / "assets" / "icon_x"
            staged.mkdir(parents=True)
            (staged / "candidate_00.png").write_bytes(b"png")
            delivered = _wave(root, attempt="attempt_02")

            self.assertEqual(delivered_attempt(root / "wave" / "runs" / FOLDER), delivered)

    def test_a_paper_that_delivered_nothing_offers_no_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _wave(root, attempt="attempt_01", delivered=False)

            self.assertIsNone(delivered_attempt(root / "wave" / "runs" / FOLDER))
            self.assertIsNone(delivered_attempt(root / "wave" / "runs" / "absent"))


class WaveCommandTest(unittest.TestCase):
    def test_cli_parses_the_command(self):
        args = build_parser().parse_args([
            "judge-data", "build-real-pairs-wave",
            "--subset", "s", "--middleware", "m", "--wave-root", "w", "--out", "o", "--no-swapped", "--merge-into", "base.jsonl",
        ])

        self.assertEqual(args.judge_data_action, "build-real-pairs-wave")
        self.assertFalse(args.include_swapped)
        self.assertEqual(args.merge_into, "base.jsonl")
        self.assertIsNone(args.attempt)

    def test_cli_dispatches_the_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _middleware(root / "version")
            _wave(root)
            subset = _subset(root, [_entry()])
            args = build_parser().parse_args([
                "judge-data", "build-real-pairs-wave",
                "--subset", str(subset),
                "--middleware", str(root / "version"),
                "--wave-root", str(root / "wave"),
                "--out", str(root / "pairs"),
                "--json",
            ])
            report = _run_judge_data(args)
            merged = _rows(root / "pairs" / "examples.jsonl")

        self.assertTrue(report["ok"])
        self.assertEqual(report["examples_written"], 2)
        self.assertEqual(len(merged), 2)


class JsonlLineTest(unittest.TestCase):
    def test_a_line_separator_inside_a_quote_stays_on_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.jsonl"
            write_examples(path, [{"example_id": "paper-1", "quote": "before\u2028after"}])
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("\u2028", text)
            self.assertEqual(read_examples(path)[0]["quote"], "before\u2028after")
            self.assertEqual(_rows(path)[0]["quote"], "before\u2028after")


if __name__ == "__main__":
    unittest.main()
