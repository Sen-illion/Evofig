"""Tests for reusing a legacy run's icons in a run that draws the sealed contract."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.cli import _run_judge_data, build_parser
from rfs.judge_training.icon_reuse import (
    STAGING_ROUTES,
    adjudicate_records,
    contract_icon_slots,
    legacy_run_icons,
    lexical_mapping,
    map_corpus,
    read_mapping,
    stage_from_mapping,
    stage_reuse,
)

FOLDER = "10.1234_example"
PLAN = {
    "contract": "rfs.method_figure_content",
    "version": 2,
    "inputs": [{"id": "input_user_task", "label": "User Task (Tu)"}],
    "outputs": [{"id": "output_detection_decision", "label": "Detection Decision"}],
    "modules": [
        {"id": "mod_original_run", "label": "Original Run Execution"},
        {"id": "mod_ensemble_detector", "label": "Ensemble Detector"},
    ],
}
LEGACY_ASSETS = [
    ("icon_user_task", "User Task"),
    ("icon_original_execution_path", "Original Execution"),
    ("icon_detection_result", "Detection Decision"),
]


def _write_contract(root: Path, plan: dict | None = None, folder: str = FOLDER) -> Path:
    target = root / folder
    target.mkdir(parents=True, exist_ok=True)
    (target / "figure_content_plan.json").write_text(json.dumps(plan or PLAN, ensure_ascii=False), encoding="utf-8")
    return target


def _write_legacy(root: Path, assets=LEGACY_ASSETS, folder: str = FOLDER, size: int = 30_000) -> Path:
    """Write a legacy run: an asset report, one prompt per icon and one PNG per icon."""
    attempt = root / folder / "attempt_01"
    assets_dir = attempt / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    report = {"mode": "image2", "assets": [{"asset_id": asset_id, "model": "gpt-image-2"} for asset_id, _ in assets]}
    (attempt / "asset_generation_report.json").write_text(json.dumps(report), encoding="utf-8")
    for asset_id, subject in assets:
        icon_dir = assets_dir / asset_id
        icon_dir.mkdir(parents=True, exist_ok=True)
        (icon_dir / "selected.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * size)
        (icon_dir / "prompt.txt").write_text(f"Create a framework-diagram icon for: {subject}. Primary visual metaphor: one object.", encoding="utf-8")
    return attempt


class IconReuseSlotsTest(unittest.TestCase):
    def test_contract_slots_read_the_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            slots = contract_icon_slots(_write_contract(Path(tmp)))
        self.assertEqual(
            [slot["slot"] for slot in slots],
            ["icon_input_user_task", "icon_output_detection_decision", "icon_mod_original_run", "icon_mod_ensemble_detector"],
        )
        self.assertEqual(slots[0]["label"], "User Task (Tu)")
        self.assertEqual(slots[0]["id_tokens"], ["user", "task"])
        self.assertTrue(all(slot["primary"] for slot in slots))

    def test_missing_plan_reads_as_no_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(contract_icon_slots(Path(tmp) / "absent"), [])


class LegacyIconTest(unittest.TestCase):
    def test_icons_carry_subject_and_usability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_legacy(root, assets=LEGACY_ASSETS + [("icon_stub", "Stub")], size=30_000)
            icon_dir = root / FOLDER / "attempt_01" / "assets" / "icon_stub"
            (icon_dir / "selected.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)
            icons = legacy_run_icons(root / FOLDER / "attempt_01")
        by_id = {icon["asset_id"]: icon for icon in icons}
        self.assertEqual(by_id["icon_user_task"]["label"], "User Task")
        self.assertTrue(by_id["icon_user_task"]["usable"])
        self.assertFalse(by_id["icon_stub"]["usable"])

    def test_run_without_report_reads_as_no_icons(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(legacy_run_icons(Path(tmp)), [])


class LexicalMappingTest(unittest.TestCase):
    def _records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slots = contract_icon_slots(_write_contract(root))
            icons = legacy_run_icons(_write_legacy(root))
            return lexical_mapping(slots, icons)

    def test_routes_split_identical_similar_and_absent(self):
        by_slot = {record["slot"]: record for record in self._records()}
        self.assertEqual(by_slot["icon_input_user_task"]["route"], "confident")
        self.assertEqual(by_slot["icon_output_detection_decision"]["route"], "confident")
        self.assertEqual(by_slot["icon_mod_original_run"]["route"], "review")
        self.assertEqual(by_slot["icon_mod_original_run"]["legacy_asset"], "icon_original_execution_path")
        self.assertEqual(by_slot["icon_mod_ensemble_detector"]["route"], "missing")
        self.assertIsNone(by_slot["icon_mod_ensemble_detector"]["legacy_asset"])

    def test_every_icon_is_claimed_once(self):
        records = self._records()
        claimed = [record["legacy_asset"] for record in records if record["legacy_asset"]]
        self.assertEqual(len(claimed), len(set(claimed)))


class AdjudicationTest(unittest.TestCase):
    def _fixture(self):
        root = Path(tempfile.mkdtemp())
        slots = contract_icon_slots(_write_contract(root))
        icons = legacy_run_icons(_write_legacy(root))
        return lexical_mapping(slots, icons), icons

    def test_keeps_drops_and_recovers(self):
        records, icons = self._fixture()
        seen = {}

        def fake_call(prompt, images, *, model, timeout, retries):
            seen.update({"model": model, "timeout": timeout, "retries": retries, "prompt": prompt})
            return {
                "mapping": {
                    "icon_input_user_task": "icon_user_task",
                    "icon_mod_original_run": None,
                    "icon_mod_ensemble_detector": "icon_original_execution_path",
                }
            }

        updated, meta = adjudicate_records(records, icons, paper=FOLDER, call=fake_call, model="test-model")
        by_slot = {record["slot"]: record for record in updated}
        self.assertEqual(by_slot["icon_input_user_task"]["legacy_asset"], "icon_user_task")
        # A confirmed pair keeps the route it earned lexically, so it stays on the confirmed sheet.
        self.assertEqual(by_slot["icon_input_user_task"]["route"], "confident")
        self.assertEqual(meta["agreed"], 1)
        self.assertEqual(meta["rejected"], 1)
        self.assertEqual(meta["recovered"], 1)
        self.assertEqual(by_slot["icon_mod_original_run"]["route"], "missing")
        self.assertIsNone(by_slot["icon_mod_original_run"]["legacy_asset"])
        self.assertEqual(by_slot["icon_mod_original_run"]["lexical_route"], "review")
        self.assertEqual(by_slot["icon_mod_ensemble_detector"]["legacy_asset"], "icon_original_execution_path")
        self.assertEqual(by_slot["icon_mod_ensemble_detector"]["route"], "adjudicated")
        self.assertEqual(by_slot["icon_output_detection_decision"]["route"], "confident")
        self.assertTrue(meta["applied"])
        self.assertEqual(meta["model"], "test-model")
        self.assertEqual(seen["model"], "test-model")
        self.assertIn("NEW SLOTS", seen["prompt"])

    def test_a_confirmed_close_match_leaves_the_review_sheet(self):
        records, icons = self._fixture()

        def fake_call(prompt, images, *, model, timeout, retries):
            return {"mapping": {"icon_mod_original_run": "icon_original_execution_path"}}

        updated, meta = adjudicate_records(records, icons, call=fake_call, model="test-model")
        by_slot = {record["slot"]: record for record in updated}
        confirmed = by_slot["icon_mod_original_run"]
        self.assertEqual(confirmed["legacy_asset"], "icon_original_execution_path")
        self.assertEqual(confirmed["route"], "adjudicated")
        self.assertEqual(confirmed["lexical_route"], "review")
        self.assertEqual(meta["agreed"], 1)

    def test_unknown_asset_is_ignored_and_the_lexical_call_survives(self):
        records, icons = self._fixture()

        def fake_call(prompt, images, *, model, timeout, retries):
            return {"mapping": {"icon_input_user_task": "icon_invented"}}

        updated, meta = adjudicate_records(records, icons, call=fake_call, model="test-model")
        record = {item["slot"]: item for item in updated}["icon_input_user_task"]
        self.assertEqual(record["legacy_asset"], "icon_user_task")
        self.assertEqual(record["route"], "confident")
        self.assertEqual(meta["ignored"], 1)

    def test_answer_without_mapping_leaves_the_table_untouched(self):
        records, icons = self._fixture()

        def fake_call(prompt, images, *, model, timeout, retries):
            return {"note": "no mapping here"}

        updated, meta = adjudicate_records(records, icons, call=fake_call, model="test-model")
        self.assertFalse(meta["applied"])
        self.assertEqual([record["legacy_asset"] for record in updated], [record["legacy_asset"] for record in records])

    def test_paper_without_icons_is_never_called(self):
        records, _ = self._fixture()
        called = []

        def fake_call(prompt, images, *, model, timeout, retries):
            called.append(True)
            return {"mapping": {}}

        _, meta = adjudicate_records(records, [], call=fake_call, model="test-model")
        self.assertFalse(meta["applied"])
        self.assertEqual(called, [])


class CorpusMappingTest(unittest.TestCase):
    def test_corpus_writes_the_tables_and_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            legacy = root / "runs"
            _write_contract(contracts)
            _write_legacy(legacy)
            payload = map_corpus(contracts, legacy, root / "out")
            out = root / "out"
            mapping = [json.loads(line) for line in (out / "mapping.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            confident = (out / "contact_sheet_confident.csv").read_text(encoding="utf-8")
            review = (out / "contact_sheet_review.csv").read_text(encoding="utf-8")
            missing = (out / "contact_sheet_missing.csv").read_text(encoding="utf-8")
            report = (out / "coverage_report.md").read_text(encoding="utf-8")
            analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["slots"], 4)
        self.assertEqual(payload["matched"], 3)
        self.assertEqual(payload["missing"], 1)
        self.assertAlmostEqual(payload["coverage"], 0.75, places=4)
        self.assertEqual(len(mapping), 1)
        self.assertEqual(mapping[0]["paper"], FOLDER)
        self.assertIn("icon_input_user_task", confident)
        self.assertIn("icon_original_execution_path", review)
        self.assertIn("icon_mod_ensemble_detector", missing)
        self.assertIn("reusable slots: 3 of 4", report)
        self.assertEqual(analysis["summary"]["matched"], 3)

    def test_limit_reads_only_the_first_papers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            legacy = root / "runs"
            _write_contract(contracts)
            _write_contract(contracts, folder="zz_second")
            _write_legacy(legacy)
            payload = map_corpus(contracts, legacy, root / "out", limit=1)
        self.assertEqual(payload["papers"], 1)
        self.assertEqual(payload["papers_with_icons"], 1)

    def test_stage_into_copies_the_kept_icons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            legacy = root / "runs"
            _write_contract(contracts)
            _write_legacy(legacy)
            payload = map_corpus(contracts, legacy, root / "out", stage_into=root / "batch")
            target = root / "batch" / "runs" / FOLDER / "attempt_01" / "round_00" / "candidate_00" / "assets" / "icon_input_user_task" / "candidate_00.png"
            guess = root / "batch" / "runs" / FOLDER / "attempt_01" / "round_00" / "candidate_00" / "assets" / "icon_mod_original_run" / "candidate_00.png"
            self.assertTrue(target.is_file())
            # The fixture maps three of four slots, but one of them is a close-wording
            # guess the VLM never confirmed: it must not reach a paid run as if it had.
            self.assertFalse(guess.exists())
            self.assertEqual(payload["staging"]["staged"], 2)
            self.assertEqual(payload["staging"]["filtered"], 2)


class ParallelAdjudicationTest(unittest.TestCase):
    def test_workers_produce_the_same_table_in_paper_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            legacy = root / "runs"
            _write_contract(contracts)
            _write_contract(contracts, folder="zz_second")
            _write_legacy(legacy)
            _write_legacy(legacy, folder="zz_second")

            def fake_call(prompt, images, *, model, timeout, retries):
                return {"mapping": {"icon_input_user_task": "icon_user_task"}}

            payload = map_corpus(contracts, legacy, root / "out", adjudicate=True, call=fake_call, model="test-model", workers=2)
            mapping = [json.loads(line) for line in (root / "out" / "mapping.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["papers"], 2)
        self.assertEqual(payload["adjudication_calls"], 2)
        self.assertEqual(payload["matched"], 6)
        self.assertEqual([row["paper"] for row in mapping], [FOLDER, "zz_second"])


class StagingTest(unittest.TestCase):
    def test_staging_skips_a_file_that_is_already_there(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_legacy(root / "runs")
            source = root / "runs" / FOLDER / "attempt_01" / "assets" / "icon_user_task" / "selected.png"
            rows = [{"paper": FOLDER, "records": [{"slot": "icon_input_user_task", "route": "confident", "legacy_png": str(source)}]}]
            first = stage_reuse(rows, root / "batch")
            second = stage_reuse(rows, root / "batch")
            forced = stage_reuse(rows, root / "batch", force=True)
        self.assertEqual(first["staged"], 1)
        self.assertEqual(second["staged"], 0)
        self.assertEqual(second["skipped"], 1)
        self.assertEqual(forced["staged"], 1)

    def test_a_record_without_a_source_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = stage_reuse([{"paper": FOLDER, "records": [{"slot": "icon_x", "route": "confident", "legacy_png": None}]}], Path(tmp) / "batch")
        self.assertEqual(report["staged"], 0)
        self.assertEqual(report["skipped"], 1)
        self.assertEqual(report["filtered"], 0)

    def test_a_guess_is_left_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_legacy(root / "runs")
            source = root / "runs" / FOLDER / "attempt_01" / "assets" / "icon_user_task" / "selected.png"
            rows = [{"paper": FOLDER, "records": [
                {"slot": "icon_input_user_task", "route": "review", "legacy_png": str(source)},
                {"slot": "icon_output_detection_decision", "route": "missing", "legacy_png": str(source)},
            ]}]

            report = stage_reuse(rows, root / "batch")

        self.assertEqual(report["staged"], 0)
        self.assertEqual(report["filtered"], 2)


class CommandTest(unittest.TestCase):
    def test_cli_parses_the_command(self):
        args = build_parser().parse_args(["judge-data", "map-icon-reuse", "--contracts", "c", "--legacy-root", "r", "--out", "o"])
        self.assertEqual(args.judge_data_action, "map-icon-reuse")
        self.assertEqual((args.contracts, args.legacy_root, args.out), ("c", "r", "o"))
        self.assertFalse(args.adjudicate)
        self.assertEqual(args.legacy_attempt, "attempt_01")

    def test_cli_dispatches_the_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_contract(root / "contracts")
            _write_legacy(root / "runs")
            args = build_parser().parse_args(
                ["judge-data", "map-icon-reuse", "--contracts", str(root / "contracts"), "--legacy-root", str(root / "runs"), "--out", str(root / "out"), "--json"]
            )
            payload = _run_judge_data(args)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["matched"], 3)


class StageFromMappingTest(unittest.TestCase):
    """The wave that re-draws a paper reads the mapping an earlier run already paid for."""

    def _mapping(self, root: Path) -> Path:
        _write_legacy(root / "runs")
        assets = root / "runs" / FOLDER / "attempt_01" / "assets"
        records = [
            {"slot": "icon_input_user_task", "route": "confident", "legacy_png": str(assets / "icon_user_task" / "selected.png")},
            {"slot": "icon_output_detection_decision", "route": "adjudicated", "legacy_png": str(assets / "icon_detection_result" / "selected.png")},
            {"slot": "icon_mod_original_run", "route": "review", "legacy_png": str(assets / "icon_original_execution_path" / "selected.png")},
            {"slot": "icon_mod_ensemble_detector", "route": "missing", "legacy_png": ""},
        ]
        path = root / "out" / "mapping.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"paper": FOLDER, "records": records}) + chr(10), encoding="utf-8")
        return path

    def test_only_the_confirmed_routes_reach_the_new_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = self._mapping(root)
            report = stage_from_mapping(mapping, root / "batch")
            staged = root / "batch" / "runs" / FOLDER / "attempt_01" / "round_00" / "candidate_00" / "assets"

            self.assertTrue((staged / "icon_input_user_task" / "candidate_00.png").is_file())
            self.assertTrue((staged / "icon_output_detection_decision" / "candidate_00.png").is_file())
            self.assertFalse((staged / "icon_mod_original_run").exists())
            self.assertFalse((staged / "icon_mod_ensemble_detector").exists())

        self.assertTrue(report["ok"])
        self.assertEqual(report["staged"], 2)
        self.assertEqual(report["filtered"], 2)
        self.assertEqual(report["routes"], list(STAGING_ROUTES))
        self.assertEqual(report["routing"], {"adjudicated": 1, "confident": 1, "missing": 1, "review": 1})
        self.assertEqual(report["records_total"], 2)

    def test_the_report_is_written_next_to_the_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = self._mapping(root)
            report = stage_from_mapping(mapping, root / "batch")
            written = json.loads(Path(report["report"]).read_text(encoding="utf-8"))

        self.assertEqual(written["staged"], 2)
        self.assertEqual(len(written["records"]), 2)
        self.assertEqual(written["records"][0]["target"].endswith("candidate_00.png"), True)

    def test_a_dry_run_reports_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = self._mapping(root)

            report = stage_from_mapping(mapping, root / "batch", dry_run=True)

            self.assertEqual(report["staged"], 2)
            self.assertTrue(report["dry_run"])
            self.assertNotIn("report", report)
            self.assertFalse((root / "batch").exists())

    def test_a_second_run_skips_what_is_already_staged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = self._mapping(root)
            first = stage_from_mapping(mapping, root / "batch")
            second = stage_from_mapping(mapping, root / "batch")
            forced = stage_from_mapping(mapping, root / "batch", force=True)

        self.assertEqual(first["staged"], 2)
        self.assertEqual(second["staged"], 0)
        self.assertEqual(second["skipped"], 2)
        self.assertEqual(forced["staged"], 2)

    def test_the_routes_can_be_widened_by_the_caller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = self._mapping(root)
            report = stage_from_mapping(mapping, root / "batch", routes=("confident", "adjudicated", "review"))

        self.assertEqual(report["staged"], 3)
        self.assertEqual(report["filtered"], 1)


class StagingCommandTest(unittest.TestCase):
    def test_cli_parses_the_command(self):
        args = build_parser().parse_args([
            "judge-data", "stage-icon-reuse", "--mapping", "m.jsonl", "--stage-into", "batch", "--dry-run",
        ])

        self.assertEqual(args.judge_data_action, "stage-icon-reuse")
        self.assertEqual((args.mapping, args.stage_into), ("m.jsonl", "batch"))
        self.assertEqual(args.routes, "confident,adjudicated")
        self.assertTrue(args.dry_run)
        self.assertFalse(args.force)

    def test_cli_dispatches_the_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = StageFromMappingTest()._mapping(Path(tmp))
            args = build_parser().parse_args([
                "judge-data", "stage-icon-reuse", "--mapping", str(mapping), "--stage-into", str(root / "batch"), "--json",
            ])
            report = _run_judge_data(args)

        self.assertTrue(report["ok"])
        self.assertEqual(report["staged"], 2)

    def test_read_mapping_tolerates_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.jsonl"
            path.write_text(chr(10) + json.dumps({"paper": FOLDER, "records": []}) + chr(10) + chr(10), encoding="utf-8")

            rows = read_mapping(path)

        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
