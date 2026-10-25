import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rfs.utils import write_json

from rfs.workflows.supervised_batch import (
    FAILURE_FILE,
    LEDGER_FILE,
    SUMMARY_FILE,
    WITHHELD_DELIVERED,
    attempt_directory,
    merge_shard_ledgers,
    next_attempt_number,
    previous_paper_state,
    record_attempt_failure,
    run_supervised_batch,
    sum_provider_tokens,
)

ENTRY = {
    "paper_id": "2025.emnlp-main.892",
    "domain": "agent",
    "split": "train",
    "pdf": "<PATH>",
    "reference": "<PATH>",
    "missing": [],
}


def clean_reports() -> dict:
    return {
        "paper_evolve_editable_report.json": {
            "generation_status": "completed",
            "quality_status": "needs_repair",
            "delivery_status": "available",
            "visual_review_mode": "vlm",
            "visual_review_status": "needs_replan",
            "node_count": 20,
            "connector_count": 22,
            "blocking_generation_errors": [],
        },
        "geometry_quality_report.json": {
            "status": "blocked",
            "node_overlap_count": 1,
            "node_overlaps": [{"node_a": "a", "node_b": "b", "overlap_area": 0.0004}],
        },
        "connector_visibility_report.json": {"status": "needs_reroute", "issue_count": 1, "connector_count": 22},
    }


def broken_reports() -> dict:
    reports = clean_reports()
    reports["geometry_quality_report.json"]["node_boundary_violation_count"] = 3
    return reports


def write_reports(out_dir: Path, reports: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in reports.items():
        (out_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def scripted_runner(script: list):
    """A runner that plays one scripted answer per attempt and records the calls."""
    calls: list[Path] = []
    entries: list[dict] = []

    def run(entry, out_dir):
        calls.append(Path(out_dir))
        entries.append(dict(entry))
        step = script[min(len(calls) - 1, len(script) - 1)]
        if isinstance(step, str):
            raise RuntimeError(step)
        if step is not None:
            write_reports(Path(out_dir), step)
        return {"ok": True, "error": None}

    run.calls = calls
    run.entries = entries
    return run


class SupervisedBatchTests(unittest.TestCase):
    def test_a_clean_attempt_is_not_paid_for_twice(self):
        runner = scripted_runner([clean_reports()])
        with TemporaryDirectory() as temp:
            summary = run_supervised_batch([ENTRY], out_root=temp, runner=runner)
            self.assertEqual(summary["attempts"], 1)
            self.assertEqual(summary["kept_clean"], 1)
            self.assertEqual(summary["extra_attempts"], 0)
            self.assertEqual(len(runner.calls), 1)

    def test_breakage_buys_another_attempt(self):
        runner = scripted_runner([broken_reports(), clean_reports()])
        with TemporaryDirectory() as temp:
            summary = run_supervised_batch([ENTRY], out_root=temp, runner=runner)
            self.assertEqual(summary["attempts"], 2)
            self.assertEqual(summary["extra_attempts"], 1)
            self.assertEqual(summary["kept_clean"], 1)
            self.assertEqual(summary["by_attempt_category"], {"clean": 1, "mechanical": 1})
            self.assertEqual(len(runner.calls), 2)

    def test_the_second_attempt_lands_in_its_own_directory(self):
        runner = scripted_runner([broken_reports(), clean_reports()])
        with TemporaryDirectory() as temp:
            run_supervised_batch([ENTRY], out_root=temp, runner=runner)
            self.assertEqual(runner.calls[0].name, "attempt_01")
            self.assertEqual(runner.calls[1].name, "attempt_02")
            self.assertNotEqual(runner.calls[0], runner.calls[1])

    def test_a_retry_continues_the_attempt_that_already_paid_for_the_paper(self):
        # The reading, the contracts and the Icons are on disk once the first
        # attempt is over; a retry that started from zero would buy them twice.
        runner = scripted_runner([broken_reports(), clean_reports()])
        with TemporaryDirectory() as temp:
            run_supervised_batch([ENTRY], out_root=temp, runner=runner)
            rows = [
                json.loads(line)
                for line in (Path(temp) / LEDGER_FILE).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertNotIn("reuse_upstream_from", runner.entries[0])
        self.assertEqual(Path(runner.entries[1]["reuse_upstream_from"]).name, "attempt_01")
        self.assertEqual(Path(rows[1]["reused_upstream_from"]).name, "attempt_01")

    def test_a_restart_continues_the_attempt_an_earlier_run_left_on_disk(self):
        with TemporaryDirectory() as temp:
            run_supervised_batch([ENTRY], out_root=temp, runner=scripted_runner([broken_reports()]))
            left_on_disk = next_attempt_number(temp, ENTRY["paper_id"])
            second = scripted_runner([clean_reports()])
            run_supervised_batch([ENTRY], out_root=temp, runner=second, resume=True)

        self.assertEqual(Path(second.entries[0]["reuse_upstream_from"]).name, f"attempt_{left_on_disk:02d}")

    def test_a_batch_can_be_told_to_retry_from_the_start(self):
        runner = scripted_runner([broken_reports(), clean_reports()])
        with TemporaryDirectory() as temp:
            run_supervised_batch([ENTRY], out_root=temp, runner=runner, reuse_upstream=False)

        self.assertTrue(all("reuse_upstream_from" not in entry for entry in runner.entries))

    def test_a_resumed_run_skips_what_was_already_kept(self):
        # The second pass exists to repair the papers that came back empty. Paying
        # for the ones that already worked again would spend the budget twice.
        with TemporaryDirectory() as temp:
            first = scripted_runner([clean_reports()])
            run_supervised_batch([ENTRY], out_root=temp, runner=first)
            second = scripted_runner([clean_reports()])
            summary = run_supervised_batch([ENTRY], out_root=temp, runner=second, resume=True)
        self.assertEqual(second.calls, [])
        self.assertEqual(summary["papers"], 1)
        self.assertEqual(summary["already_kept"], 1)
        self.assertEqual(summary["kept_clean"], 1)
        self.assertEqual(summary["attempts"], 0)

    def test_a_resumed_run_does_not_buy_a_kept_first_draft_again(self):
        with TemporaryDirectory() as temp:
            run_supervised_batch(
                [ENTRY],
                out_root=temp,
                runner=scripted_runner([broken_reports(), clean_reports()]),
                keep_first_draft=True,
            )
            second = scripted_runner([clean_reports()])
            summary = run_supervised_batch(
                [ENTRY],
                out_root=temp,
                runner=second,
                resume=True,
                keep_first_draft=True,
            )
        self.assertEqual(second.calls, [])
        self.assertEqual(summary["already_kept"], 1)
        self.assertEqual(summary["attempts"], 0)

    def test_a_resumed_run_continues_the_attempt_numbering(self):
        with TemporaryDirectory() as temp:
            run_supervised_batch([ENTRY], out_root=temp, runner=scripted_runner([broken_reports()]))
            before = next_attempt_number(temp, ENTRY["paper_id"])
            second = scripted_runner([clean_reports()])
            summary = run_supervised_batch([ENTRY], out_root=temp, runner=second, resume=True)
        self.assertEqual([path.name for path in second.calls], [f"attempt_{before + 1:02d}"])
        self.assertEqual(summary["kept_clean"], 1)
        self.assertEqual(summary["attempts"], 1)

    def test_the_previous_state_reads_the_latest_attempt_of_each_paper(self):
        with TemporaryDirectory() as temp:
            run_supervised_batch([ENTRY], out_root=temp, runner=scripted_runner([broken_reports(), clean_reports()]))
            state = previous_paper_state(temp)
            self.assertEqual(state[ENTRY["paper_id"]]["attempt"], 2)
            self.assertEqual(state[ENTRY["paper_id"]]["status"], "clean")
            self.assertEqual(next_attempt_number(temp, ENTRY["paper_id"]), 2)
            self.assertEqual(next_attempt_number(temp, "an_unknown_paper"), 0)

    def test_a_delivered_figure_is_kept_when_the_batch_is_told_to_keep_it(self):
        # Bulk work wants the flawed figure next to the note that says why, not a
        # second bill: the verdict asks for a retry and the batch keeps the first
        # attempt anyway, recording that it did.
        runner = scripted_runner([broken_reports(), clean_reports()])
        with TemporaryDirectory() as temp:
            summary = run_supervised_batch([ENTRY], out_root=temp, runner=runner, keep_first_draft=True)
            rows = [
                json.loads(line)
                for line in (Path(temp) / LEDGER_FILE).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(summary["by_final_category"], {"mechanical": 1})
        self.assertEqual(summary["kept_first_draft"], 1)
        self.assertEqual(summary["kept_first_draft_paper_ids"], [ENTRY["paper_id"]])
        self.assertEqual(summary["extra_attempts"], 0)
        self.assertEqual(rows[0]["retry_withheld"], WITHHELD_DELIVERED)

    def test_an_attempt_that_delivered_nothing_is_still_retried(self):
        # A crash is not a flaw in a figure: there is no figure to keep, so the
        # paper keeps its retry even in the middle of a bulk run.
        runner = scripted_runner(["RuntimeError: provider closed the connection", clean_reports()])
        with TemporaryDirectory() as temp:
            summary = run_supervised_batch([ENTRY], out_root=temp, runner=runner, keep_first_draft=True)

        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(summary["kept_clean"], 1)
        self.assertEqual(summary["kept_first_draft"], 0)

    def test_keeping_the_first_draft_is_off_by_default(self):
        runner = scripted_runner([broken_reports(), clean_reports()])
        with TemporaryDirectory() as temp:
            summary = run_supervised_batch([ENTRY], out_root=temp, runner=runner)
            rows = [
                json.loads(line)
                for line in (Path(temp) / LEDGER_FILE).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(summary["kept_first_draft"], 0)
        self.assertNotIn("retry_withheld", rows[0])

    def test_a_provider_outage_is_retried_and_then_recorded(self):
        runner = scripted_runner(["RuntimeError: provider closed the connection"])
        with TemporaryDirectory() as temp:
            summary = run_supervised_batch([ENTRY], out_root=temp, runner=runner)
            self.assertEqual(summary["attempts"], 3)
            self.assertEqual(summary["by_final_category"], {"failure": 1})
            self.assertEqual(len(runner.calls), 3)

    def test_an_unusable_contract_is_recorded_without_a_retry(self):
        failed = clean_reports()
        failed["paper_evolve_editable_report.json"].update(
            {"generation_status": "failed", "delivery_status": "unavailable", "blocking_generation_errors": ["method_contract_invalid"]}
        )
        runner = scripted_runner([failed])
        with TemporaryDirectory() as temp:
            summary = run_supervised_batch([ENTRY], out_root=temp, runner=runner)
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(summary["by_final_category"], {"upstream": 1})
            self.assertEqual(summary["needs_attention"][0]["codes"], ["blocking:method_contract_invalid"])

    def test_a_paper_without_inputs_is_skipped_and_named(self):
        runner = scripted_runner([clean_reports()])
        entry = dict(ENTRY, paper_id="no-file", pdf=None, missing=["paper_pdf"])
        with TemporaryDirectory() as temp:
            summary = run_supervised_batch([entry], out_root=temp, runner=runner)
            self.assertEqual(len(runner.calls), 0)
            self.assertEqual(summary["by_final_category"], {"unresolved_input": 1})
            self.assertEqual(summary["unresolved"], 1)
            self.assertEqual(summary["needs_attention"][0]["paper_id"], "no-file")

    def test_a_crash_says_which_stage_it_happened_in(self):
        # "TypeError: float() argument must be..." in the ledger is not a fix.
        # The stage and the frame are what make the next occurrence cheap.
        def crashing_runner(entry, out_dir):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "progress.jsonl").write_text(
                json.dumps({"stage": "structure_planning", "status": "started"}) + "\n", encoding="utf-8"
            )
            float({})

        with TemporaryDirectory() as temp:
            run_supervised_batch([ENTRY], out_root=temp, runner=crashing_runner)
            row = json.loads((Path(temp) / LEDGER_FILE).read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["failure"]["error_type"], "TypeError")
            self.assertEqual(row["failure"]["stage"], "structure_planning")
            self.assertEqual(row["failure"]["report"], FAILURE_FILE)
            self.assertTrue(row["failure"]["location"])
            report = json.loads(
                (attempt_directory(temp, ENTRY["paper_id"], 1) / FAILURE_FILE).read_text(encoding="utf-8")
            )
            self.assertEqual(report["contract"], "rfs.supervised_batch_attempt_error")
            self.assertIn("TypeError", report["traceback"])
            self.assertIn("float({})", report["traceback"])
            # The innermost pipeline frame is where the batch loop handed
            # control away, so a crash inside the product workflow is named.
            self.assertIn("supervised_batch.py", report["location"])

    def test_the_recorded_location_names_the_package_frame(self):
        with TemporaryDirectory() as temp:
            try:
                write_json(Path(temp) / "unserializable.json", object())
            except TypeError as exc:
                report = record_attempt_failure(temp, paper_id="p", attempt=1, error=exc)
            else:  # pragma: no cover - the fixture has to raise
                self.fail("write_json accepted an unserializable object")
            self.assertIn("utils.py", report["location"])
            self.assertIn("rfs", report["location"])

    def test_every_attempt_is_written_to_the_ledger_as_it_finishes(self):
        runner = scripted_runner([broken_reports(), clean_reports()])
        with TemporaryDirectory() as temp:
            run_supervised_batch([ENTRY], out_root=temp, runner=runner)
            rows = [json.loads(line) for line in (Path(temp) / LEDGER_FILE).read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["attempt"] for row in rows], [1, 2])
            self.assertEqual(rows[0]["status"], "retry")
            self.assertEqual(rows[1]["status"], "clean")
            self.assertIn("elapsed_seconds", rows[0])

    def test_the_summary_is_written_next_to_the_ledger(self):
        runner = scripted_runner([clean_reports()])
        with TemporaryDirectory() as temp:
            run_supervised_batch([ENTRY], out_root=temp, runner=runner, batch_id="subset-100")
            summary = json.loads((Path(temp) / SUMMARY_FILE).read_text(encoding="utf-8"))
            self.assertEqual(summary["batch_id"], "subset-100")
            self.assertEqual(summary["papers"], 1)
            self.assertEqual(summary["token_source"], "run_reports")

    def test_what_the_run_reported_is_counted_back(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "design_plan_validation.json").write_text(
                json.dumps({"provider": {"usage": {"prompt_tokens": 100, "completion_tokens": 20}}}),
                encoding="utf-8",
            )
            (root / "run_report.json").write_text(
                json.dumps({"provider": {"planner": {"usage": {"total_tokens": 500}}}}),
                encoding="utf-8",
            )
            (root / "paper_evolve_editable_report.json").write_text(
                json.dumps({"structure_report": {"provider": {"usage": {"total_tokens": 120}}}}),
                encoding="utf-8",
            )
            totals = sum_provider_tokens(root)
            self.assertEqual(totals["total"], 620)
            self.assertEqual(totals["source"], "run_reports")

    def test_a_windows_hostile_paper_id_becomes_a_directory_name(self):
        directory = attempt_directory("<PATH>", "iclr:2026:10006567", 2)
        self.assertEqual(directory.parent.name, "iclr_2026_10006567")
        self.assertEqual(directory.name, "attempt_02")

    def test_an_attempt_id_is_not_reused_across_papers(self):
        first = attempt_directory("<PATH>", "a/b", 1)
        second = attempt_directory("<PATH>", "a:b", 1)
        self.assertEqual(first.parent.name, second.parent.name)

    def test_a_shard_runs_every_nth_paper(self):
        entries = [dict(ENTRY, paper_id=f"paper-{index}") for index in range(4)]
        runner = scripted_runner([clean_reports()])
        with TemporaryDirectory() as temp:
            summary = run_supervised_batch(entries, out_root=temp, runner=runner, shard=(1, 2))
            self.assertEqual(summary["papers"], 2)
            self.assertEqual(sorted(path.parent.name for path in runner.calls), ["paper-1", "paper-3"])
            self.assertEqual(summary["shard"], "1/2")

    def test_shards_keep_their_ledgers_apart(self):
        entries = [dict(ENTRY, paper_id=f"paper-{index}") for index in range(2)]
        runner = scripted_runner([clean_reports()])
        with TemporaryDirectory() as temp:
            run_supervised_batch(entries, out_root=temp, runner=runner, shard=(0, 2))
            run_supervised_batch(entries, out_root=temp, runner=runner, shard=(1, 2))
            self.assertTrue((Path(temp) / "attempts_shard_00.jsonl").is_file())
            self.assertTrue((Path(temp) / "attempts_shard_01.jsonl").is_file())
            self.assertFalse((Path(temp) / LEDGER_FILE).exists())

    def test_the_shard_ledgers_fold_into_one_batch_summary(self):
        entries = [dict(ENTRY, paper_id=f"paper-{index}") for index in range(4)]
        with TemporaryDirectory() as temp:
            run_supervised_batch(entries, out_root=temp, runner=scripted_runner([clean_reports()]), shard=(0, 2))
            run_supervised_batch(entries, out_root=temp, runner=scripted_runner([broken_reports()]), shard=(1, 2))
            merged = merge_shard_ledgers(temp, batch_id="subset-100")
            self.assertEqual(merged["papers"], 4)
            # The broken shard spends its whole budget: three attempts per paper.
            self.assertEqual(merged["attempts"], 8)
            self.assertEqual(merged["shards"], 2)
            self.assertEqual(merged["kept_clean"], 2)
            self.assertEqual(len(merged["needs_attention"]), 2)
            rows = [json.loads(line) for line in (Path(temp) / LEDGER_FILE).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 8)

    def test_a_shard_out_of_range_is_refused(self):
        with TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                run_supervised_batch([ENTRY], out_root=temp, runner=scripted_runner([clean_reports()]), shard=(2, 2))

    def test_a_spent_account_stops_the_batch_instead_of_walking_the_queue(self):
        # Every paper left would fail the same way within seconds. Walking the
        # queue anyway would spend the whole corpus's retry budget on questions
        # the provider refused to answer (it did, on 2026-09-21).
        entries = [dict(ENTRY, paper_id=f"paper-{index}") for index in range(6)]
        runner = scripted_runner(["RuntimeError: HTTP 403 insufficient_quota"])
        with TemporaryDirectory() as temp:
            summary = run_supervised_batch(entries, out_root=temp, runner=runner)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(summary["attempts"], 1)
        self.assertEqual(summary["halted_reason"], "provider_account_exhausted")
        self.assertEqual(summary["halted_after"], 1)

    def test_a_stage_that_swallowed_the_quota_error_still_stops_the_batch(self):
        # The pipeline catches its own failures and records them as fallbacks, so
        # the refusal arrives inside a report rather than as an exception.
        swallowed = clean_reports()
        swallowed["scientific_hierarchy_validation.json"] = {
            "fallback_used": True,
            "fallback_reason": 'provider rejected the request with HTTP status 403: {"message":"user quota is not enough"}',
        }
        entries = [dict(ENTRY, paper_id=f"paper-{index}") for index in range(4)]
        runner = scripted_runner([swallowed])
        with TemporaryDirectory() as temp:
            summary = run_supervised_batch(entries, out_root=temp, runner=runner)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(summary["halted_reason"], "provider_account_exhausted")

    def test_the_refusal_is_found_even_in_a_report_the_verdict_never_reads(self):
        # Seven of the hundred outage attempts recorded the refusal only in a
        # file the verdict does not load, and each of those kept the queue going.
        def runner(entry, out_dir):
            directory = Path(out_dir)
            write_reports(directory, clean_reports())
            (directory / "figure_text_plan_validation.json").write_text(
                json.dumps({"fallback_used": True, "fallback_reason": "HTTP 403: user quota is not enough"}),
                encoding="utf-8",
            )
            return {"ok": True, "error": None}

        entries = [dict(ENTRY, paper_id=f"paper-{index}") for index in range(4)]
        with TemporaryDirectory() as temp:
            summary = run_supervised_batch(entries, out_root=temp, runner=runner)
            self.assertEqual(summary["halted_reason"], "provider_account_exhausted")
            self.assertEqual(summary["attempts"], 1)

    def test_the_merged_summary_carries_the_halt_up(self):
        entries = [dict(ENTRY, paper_id=f"paper-{index}") for index in range(4)]
        with TemporaryDirectory() as temp:
            run_supervised_batch(entries, out_root=temp, runner=scripted_runner(["RuntimeError: insufficient_quota"]), shard=(0, 2))
            run_supervised_batch(entries, out_root=temp, runner=scripted_runner([clean_reports()]), shard=(1, 2))
            merged = merge_shard_ledgers(temp, batch_id="subset-100")
        self.assertEqual(merged["halted_reason"], "provider_account_exhausted")
        self.assertEqual([item["shard"] for item in merged["halted_shards"]], ["0/2"])


class BulkFiguresRunnerTests(unittest.TestCase):
    """The corpus command tells the workflow it is filling a bulk dataset."""

    def _run(self, **factory_kwargs):
        from unittest.mock import patch

        from rfs.workflows.supervised_batch import make_paper_evolve_runner

        seen: dict = {}

        def fake(**options):
            seen.update(options)
            return {"ok": True}

        with TemporaryDirectory() as temp, patch(
            "rfs.workflows.paper_evolve_editable.run_paper_evolve_editable", side_effect=fake
        ):
            runner = make_paper_evolve_runner(**factory_kwargs)
            runner({"pdf": "paper.pdf"}, Path(temp))
        return seen

    def test_the_runner_passes_the_bulk_flag_to_the_workflow(self):
        seen = self._run(bulk_figures=True)
        self.assertTrue(seen["bulk_figures"])
        self.assertEqual("vlm", seen["structure_mode"])

    def test_the_runner_leaves_the_flag_off_by_default(self):
        self.assertFalse(self._run()["bulk_figures"])


class ContractAdoptionRunnerTests(unittest.TestCase):
    """A paper whose contract the middleware exported is drawn from that contract."""

    def _run(self, entry, **factory_kwargs):
        from unittest.mock import patch

        from rfs.workflows.supervised_batch import make_paper_evolve_runner

        seen: dict = {}

        def fake(**options):
            seen.update(options)
            return {"ok": True}

        with TemporaryDirectory() as temp, patch(
            "rfs.workflows.paper_evolve_editable.run_paper_evolve_editable", side_effect=fake
        ):
            runner = make_paper_evolve_runner(**factory_kwargs)
            captured = runner(entry, Path(temp))
        return seen, captured

    def test_a_paper_the_export_covers_is_drawn_from_the_exported_contract(self):
        from rfs.judge_training.pilot_graphs import safe_paper_directory

        entry = dict(ENTRY, paper_id="10.1038/s41467-025-61026-3")
        with TemporaryDirectory() as temp:
            contract = Path(temp) / safe_paper_directory(entry["paper_id"])
            contract.mkdir(parents=True)
            seen, result = self._run(entry, contracts_from=temp)

        self.assertEqual(seen["upstream_from"], str(contract))
        self.assertEqual(result["metrics"]["upstream_contract"], str(contract))

    def test_a_paper_the_export_does_not_cover_is_analysed_as_before(self):
        with TemporaryDirectory() as temp:
            seen, result = self._run(dict(ENTRY), contracts_from=temp)

        self.assertNotIn("upstream_from", seen)
        self.assertEqual(result["metrics"]["upstream_contract"], "")

    def test_an_entry_may_name_its_own_contract_directory(self):
        with TemporaryDirectory() as temp:
            named = Path(temp) / "chosen"
            named.mkdir()
            seen, _ = self._run(dict(ENTRY, upstream_contract_dir=str(named)))

        self.assertEqual(seen["upstream_from"], str(named))

    def test_without_a_contract_root_nothing_is_adopted(self):
        seen, _ = self._run(dict(ENTRY))
        self.assertNotIn("upstream_from", seen)


if __name__ == "__main__":
    unittest.main()
