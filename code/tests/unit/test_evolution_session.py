"""Tests for the wall-clock evolution session: its queue, its attempts, its report."""

from __future__ import annotations

import contextlib
import io
import json
import ssl
import tempfile
import unittest
from pathlib import Path

from rfs.cli import build_parser, main
from rfs.contracts.evolution_session import validate_evolution_session_report
from rfs.evolution.direct import ExperienceStore
from rfs.utils import write_json
from rfs.workflows import run_evolution_session
from tests.unit.test_memory_sweep import _row as _experience_row


def _rate_limited() -> RuntimeError:
    """The failure the provider raises once its own short retries are spent on a 429."""

    return RuntimeError("provider rejected the request with HTTP status 429: slow down")


def _paper(root: Path, name: str) -> Path:
    path = root / name
    path.write_bytes(b"%PDF-1.4\n")
    return path


def _suite(root: Path) -> Path:
    """A validation suite shaped like the one the paired harness reads."""

    return write_json(
        root / "suite.json",
        {
            "target_metric": "overall_score",
            "tolerance": 0.01,
            "cases": [
                {"case_id": "same-1", "domain": "ai_method", "split": "same_domain", "paper": str(_paper(root, "same-1.pdf"))},
                {"case_id": "same-2", "domain": "ai_method", "split": "same_domain", "paper": str(_paper(root, "same-2.pdf"))},
                {"case_id": "hold-1", "domain": "computer_system", "split": "holdout", "paper": str(_paper(root, "hold-1.pdf"))},
            ],
        },
    )


def _config(root: Path, stations: list[dict], **overrides) -> Path:
    body = {"session_id": "session-test", "stations": stations, **overrides}
    return write_json(root / "session.json", body)


class _Clock:
    """A wall clock the test drives, so a budget can be spent without waiting for it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeEvolve:
    """A stand-in for the paid run: it writes the artifacts the evidence reader looks for."""

    def __init__(self, *, clock: "_Clock | None" = None, spend: float = 0.0, failures: int = 0, error: BaseException | None = None, experience_id: str = "", applicability: str | None = None, injected: list | None = None, retrieved: list | None = None, ok: bool = True) -> None:
        self.calls: list[dict] = []
        self.clock = clock
        self.spend = spend
        self.failures = failures
        self.error = error
        self.experience_id = experience_id
        self.applicability = applicability
        self.injected = list(injected or [])
        self.retrieved = list(retrieved or [])
        self.ok = ok

    def __call__(self, **options) -> dict:
        self.calls.append(dict(options))
        if self.error is not None and len(self.calls) <= self.failures:
            raise self.error
        out = Path(str(options["out"]))
        out.mkdir(parents=True, exist_ok=True)
        (out / "final_draft_preview.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        trace = {
            "contract": "rfs.memory_application_trace",
            "version": 1,
            "totals": {"patches_applied": ["patch-1"], "patches_rejected": 1, "rules_applied": ["rule-1"], "rules_withheld": 0, "rules_refuted": 0},
            "rounds": [
                {
                    "round": 0,
                    "retrieval": {"candidates": 2, "reason": "two rows outranked the rest"},
                    "injected_experience": {"experience_id": self.injected[0] if self.injected else "", "status": "same_domain_validated" if self.injected else ""},
                    "rejected_experiences": [{"experience_id": "direct-exp-lost", "reason": "lost the precedence decision"}],
                }
            ],
        }
        write_json(out / "memory_application_trace.json", trace)
        library = str(options.get("experience_library") or "")
        if self.experience_id:
            overrides = {"status": "current_case_validated"}
            if self.applicability is not None:
                overrides["applicability"] = self.applicability
                if self.applicability == "domain":
                    overrides["applicability_domain"] = "ai_method"
            row = _experience_row(self.experience_id, **overrides)
            write_json(out / "round_00" / "experience_candidate.json", row)
            if library:
                ExperienceStore(library).append_result(row)
        if self.clock is not None:
            self.clock.advance(self.spend)
        return {
            "summary": "a run that returned",
            "ok": self.ok,
            "out_dir": str(out),
            "rounds_completed": 1,
            "experience_library": library,
            "retrieved_strategy_ids": list(self.retrieved),
            "injected_experience_ids": list(self.injected),
            "prompt_patch_manifest": {"stages": {}, "applied": []},
            "memory_application_trace": trace,
        }


class FakeValidate:
    """A stand-in for the paired ON/OFF harness: it writes the ladder report and moves the row."""

    def __init__(self, *, clock: "_Clock | None" = None, spend: float = 0.0, status: str = "cross_domain_validated", failures: int = 0, error: BaseException | None = None) -> None:
        self.calls: list[dict] = []
        self.clock = clock
        self.spend = spend
        self.status = status
        self.failures = failures
        self.error = error

    def __call__(self, **options) -> dict:
        self.calls.append(dict(options))
        if self.error is not None and len(self.calls) <= self.failures:
            raise self.error
        out = Path(str(options["out"]))
        out.mkdir(parents=True, exist_ok=True)
        report = {
            "experience_id": str(options.get("experience_id") or ""),
            "status": self.status,
            "ladder": {"stopped_because": "not_every_case_improved"},
            "summary": {"cases": 3, "measured_cases": 3, "passed_cases": 3},
            "target_metric": "overall_score",
            "all_pairs_isolated": True,
            "all_cases_measured": True,
            "results": [
                {
                    "case_id": "same-1",
                    "domain": "ai_method",
                    "split": "same_domain",
                    "measured": True,
                    "target_delta": 0.3,
                    "passed": True,
                    "new_blockers": 0,
                    "off": {"overall_score": 0.4, "action_sequence": ["relayout_global"]},
                    "on": {"overall_score": 0.7, "action_sequence": ["relayout_global"]},
                }
            ],
        }
        write_json(out / "cross_paper_validation.json", report)
        library = str(options.get("experience_library") or "")
        if library:
            ExperienceStore(library).transition(str(options.get("experience_id") or ""), self.status, validation_report=str(out / "cross_paper_validation.json"))
        if self.clock is not None:
            self.clock.advance(self.spend)
        return report


class SessionTestCase(unittest.TestCase):
    def assertValid(self, report: dict) -> dict:
        check = validate_evolution_session_report(report)
        self.assertTrue(check["ok"], check["errors"])
        return report


class SessionContractTests(unittest.TestCase):
    def _report(self, **overrides) -> dict:
        report = {
            "session_id": "session-x",
            "summary": "a session",
            "ok": True,
            "out_dir": "out",
            "budget": {"hours": 5.0, "used_seconds": 60.0, "exhausted": False},
            "stations": [
                {"kind": "evolve", "subject": "paper-a", "status": "completed", "attempts": [{"attempt": 1, "status": "ok"}]},
                {"kind": "validate", "subject": "direct-exp-1", "status": "completed", "attempts": [{"attempt": 1, "status": "ok"}]},
            ],
            "blockers": [],
        }
        report.update(overrides)
        return report

    def test_a_completed_session_validates(self):
        check = validate_evolution_session_report(self._report())
        self.assertTrue(check["ok"], check["errors"])

    def test_a_session_without_a_station_is_rejected(self):
        check = validate_evolution_session_report(self._report(stations=[]))
        self.assertFalse(check["ok"])
        self.assertTrue(any("at least one station" in error for error in check["errors"]))

    def test_an_unknown_status_is_rejected(self):
        report = self._report()
        report["stations"][0]["status"] = "finished"
        check = validate_evolution_session_report(report)
        self.assertFalse(check["ok"])
        self.assertTrue(any("unknown status" in error for error in check["errors"]), check["errors"])

    def test_an_unknown_reason_is_rejected(self):
        report = self._report(
            ok=False,
            blockers=[{"station_id": "evolve:paper-a", "kind": "evolve", "reason": "no_endpoint", "detail": "x"}],
            stations=[
                {"kind": "evolve", "subject": "paper-a", "status": "failed", "reason": "no_endpoint", "detail": "x", "attempts": [{"attempt": 1, "status": "failed", "reason": "no_endpoint"}]},
            ],
        )
        check = validate_evolution_session_report(report)
        self.assertFalse(check["ok"])
        self.assertTrue(any("unknown reason" in error for error in check["errors"]))

    def test_every_reason_the_session_may_use_is_accepted(self):
        for reason in ("run_failed", "validation_failed", "rate_limited", "paper_unreadable"):
            with self.subTest(reason=reason):
                report = self._report(
                    ok=False,
                    blockers=[{"station_id": "evolve:paper-a", "kind": "evolve", "reason": reason, "detail": "d"}],
                    stations=[{"kind": "evolve", "subject": "paper-a", "status": "failed", "reason": reason, "detail": "d", "attempts": [{"attempt": 1, "status": "failed", "reason": reason}]}],
                )
                check = validate_evolution_session_report(report)
                self.assertTrue(check["ok"], check["errors"])

    def test_a_budget_that_ran_out_may_not_hide_a_station_that_never_ran(self):
        report = self._report(budget={"hours": 5.0, "used_seconds": 18000.0, "exhausted": True})
        report["stations"][1] = {"kind": "validate", "subject": "direct-exp-1", "status": "completed", "attempts": []}
        check = validate_evolution_session_report(report)
        self.assertFalse(check["ok"])
        self.assertTrue(any("without recording the attempt" in error for error in check["errors"]))

    def test_a_session_that_ran_out_of_budget_may_not_leave_stations_pending(self):
        report = self._report(
            budget={"hours": 5.0, "used_seconds": 18000.0, "exhausted": True},
            stations=[{"kind": "evolve", "subject": "paper-a", "status": "completed", "attempts": [{"attempt": 1, "status": "ok"}]}, {"kind": "validate", "subject": "direct-exp-1", "status": "pending"}],
        )
        check = validate_evolution_session_report(report)
        self.assertFalse(check["ok"])
        self.assertTrue(any("must say which stations it dropped" in error for error in check["errors"]))

    def test_a_station_dropped_for_a_budget_that_was_not_exhausted_is_rejected(self):
        report = self._report(
            stations=[
                {"kind": "evolve", "subject": "paper-a", "status": "completed", "attempts": [{"attempt": 1, "status": "ok"}]},
                {"kind": "validate", "subject": "direct-exp-1", "status": "skipped", "reason": "budget_exhausted"},
            ]
        )
        check = validate_evolution_session_report(report)
        self.assertFalse(check["ok"])
        self.assertTrue(any("budget the report says was not exhausted" in error for error in check["errors"]))

    def test_failure_without_a_failed_station_is_rejected(self):
        check = validate_evolution_session_report(self._report(ok=False))
        self.assertFalse(check["ok"])
        self.assertTrue(any("without naming a failed station" in error for error in check["errors"]))

    def test_success_with_a_failed_station_is_rejected(self):
        report = self._report(
            ok=True,
            stations=[
                {"kind": "evolve", "subject": "paper-a", "status": "completed", "attempts": [{"attempt": 1, "status": "ok"}]},
                {"kind": "evolve", "subject": "paper-b", "status": "failed", "reason": "run_failed", "detail": "d", "attempts": [{"attempt": 1, "status": "failed", "reason": "run_failed"}]},
            ],
        )
        check = validate_evolution_session_report(report)
        self.assertFalse(check["ok"])
        self.assertTrue(any("claims success" in error for error in check["errors"]))

    def test_a_station_that_never_ran_may_not_report_attempts(self):
        report = self._report(
            stations=[{"kind": "validate", "subject": "direct-exp-1", "status": "skipped", "reason": "no_candidate_experience", "attempts": [{"attempt": 1, "status": "ok"}]}]
        )
        check = validate_evolution_session_report(report)
        self.assertFalse(check["ok"])
        self.assertTrue(any("skipped after it had already run an attempt" in error for error in check["errors"]))

    def test_attempts_have_to_be_numbered_in_order(self):
        report = self._report(
            stations=[{"kind": "evolve", "subject": "paper-a", "status": "completed", "attempts": [{"attempt": 2, "status": "ok"}]}]
        )
        check = validate_evolution_session_report(report)
        self.assertFalse(check["ok"])
        self.assertTrue(any("out of order" in error for error in check["errors"]))

    def test_the_same_station_may_not_appear_twice(self):
        report = self._report(
            stations=[
                {"kind": "evolve", "subject": "paper-a", "status": "completed", "attempts": [{"attempt": 1, "status": "ok"}]},
                {"kind": "evolve", "subject": "paper-a", "status": "completed", "attempts": [{"attempt": 1, "status": "ok"}]},
            ]
        )
        check = validate_evolution_session_report(report)
        self.assertFalse(check["ok"])
        self.assertTrue(any("appears more than once" in error for error in check["errors"]))

    def test_a_dry_run_may_not_claim_it_ran_anything(self):
        report = self._report(dry_run=True)
        check = validate_evolution_session_report(report)
        self.assertFalse(check["ok"])
        self.assertTrue(any("dry run cannot have carried out" in error for error in check["errors"]))

    def test_a_session_needs_a_positive_budget(self):
        check = validate_evolution_session_report(self._report(budget={"hours": 0.0, "used_seconds": 0.0, "exhausted": False}))
        self.assertFalse(check["ok"])
        self.assertTrue(any("positive wall-clock budget" in error for error in check["errors"]))


class SessionRunnerTests(SessionTestCase):
    def _evolve_station(self, root: Path, name: str, **overrides) -> dict:
        station = {"kind": "evolve", "paper": str(_paper(root, f"{name}.pdf")), "paper_id": name, "max_rounds": 3, "candidates_per_round": 2}
        station.update(overrides)
        return station

    def test_the_queue_runs_in_order_and_gives_each_paper_one_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "session"
            config = _config(root, [self._evolve_station(root, "paper-a"), self._evolve_station(root, "paper-b")])
            evolve = FakeEvolve()
            report = self.assertValid(run_evolution_session(config, out, evolve_runner=evolve, clock=_Clock(), sleeper=lambda _s: None))
            self.assertTrue(report["ok"])
            self.assertEqual(["evolve:paper-a", "evolve:paper-b"], [item["station_id"] for item in report["stations"]])
            self.assertEqual(["completed", "completed"], [item["status"] for item in report["stations"]])
            self.assertEqual([1, 1], [len(item["attempts"]) for item in report["stations"]])
            self.assertEqual(str((out / "papers" / "paper-a" / "attempt_01").resolve()), str(Path(report["stations"][0]["out_dir"]).resolve()))
            self.assertTrue((out / "papers" / "paper-a" / "attempt_01" / "final_draft_preview.png").is_file())
            self.assertEqual([3, 3], [call["max_rounds"] for call in evolve.calls])

    def test_the_report_records_what_each_station_recalled_and_injected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root, [self._evolve_station(root, "paper-a")])
            evolve = FakeEvolve(injected=["direct-exp-7e93"], retrieved=["direct-exp-7e93"])
            report = self.assertValid(run_evolution_session(config, root / "session", evolve_runner=evolve, clock=_Clock(), sleeper=lambda _s: None))
            evidence = report["stations"][0]["evidence"]
            self.assertEqual(["direct-exp-7e93"], evidence["experiences_injected"])
            self.assertEqual(["direct-exp-7e93"], evidence["experiences_retrieved"])
            self.assertTrue(evidence["memory_application_trace"].endswith("memory_application_trace.json"))
            trace = json.loads(Path(evidence["memory_application_trace"]).read_text(encoding="utf-8"))
            self.assertEqual(["direct-exp-7e93"], [row["injected_experience"]["experience_id"] for row in trace["rounds"]])
            self.assertEqual(["same_domain_validated"], [row["status"] for row in evidence["recall"]])
            self.assertEqual(["patch-1"], evidence["patches_applied"])
            self.assertEqual(1, evidence["patches_rejected"])
            self.assertEqual("direct-exp-lost", evidence["rejected_experiences"][0]["experience_id"])

    def test_a_new_experience_is_validated_after_the_paper_that_earned_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            config = _config(root, [self._evolve_station(root, "paper-a")], validation_suite=str(_suite(root)))
            evolve = FakeEvolve(experience_id="direct-exp-new")
            validate = FakeValidate()
            report = self.assertValid(
                run_evolution_session(config, root / "session", experience_library=library, evolve_runner=evolve, validate_runner=validate, clock=_Clock(), sleeper=lambda _s: None)
            )
            self.assertEqual(["evolve:paper-a", "validate:direct-exp-new"], [item["station_id"] for item in report["stations"]])
            self.assertEqual(["direct-exp-new"], [call["experience_id"] for call in validate.calls])
            self.assertEqual("cross_domain_validated", report["stations"][1]["evidence"]["status_after"])
            self.assertEqual("cross_domain_validated", ExperienceStore(library).lifecycle_status("direct-exp-new"))
            self.assertEqual(3, report["stations"][1]["evidence"]["summary"]["cases"])

    def test_the_queues_own_suite_is_used_when_the_session_named_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            suite = _suite(root)
            config = _config(
                root,
                [self._evolve_station(root, "paper-a"), {"kind": "validate", "experience_id": "direct-exp-known", "suite": str(suite)}],
            )
            ExperienceStore(library).append_result(_experience_row("direct-exp-known"))
            evolve = FakeEvolve(experience_id="direct-exp-new")
            validate = FakeValidate()
            report = self.assertValid(
                run_evolution_session(config, root / "session", experience_library=library, evolve_runner=evolve, validate_runner=validate, clock=_Clock(), sleeper=lambda _s: None)
            )
            self.assertEqual(["evolve:paper-a", "validate:direct-exp-known", "validate:direct-exp-new"], [item["station_id"] for item in report["stations"]])
            self.assertEqual([str(suite), str(suite)], [call["suite"] for call in validate.calls])

    def test_a_row_earned_without_any_suite_is_not_validated_against_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "session"
            config = _config(root, [self._evolve_station(root, "paper-a")])
            validate = FakeValidate()
            report = self.assertValid(
                run_evolution_session(config, out, experience_library=root / "library", evolve_runner=FakeEvolve(experience_id="direct-exp-new"), validate_runner=validate, clock=_Clock(), sleeper=lambda _s: None)
            )
            self.assertEqual([], validate.calls)
            self.assertEqual(1, report["counts"]["stations"])
            events = [json.loads(line) for line in (out / "evolution_session_events.jsonl").read_text(encoding="utf-8").splitlines()]
            declined = [item for item in events if item["event"] == "auto_validate_declined"]
            self.assertEqual(1, len(declined))
            self.assertIn("no validation suite", declined[0]["detail"])

    def test_every_paper_receives_the_session_acceptance_checkpoint_and_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root, [self._evolve_station(root, "paper-a"), self._evolve_station(root, "paper-b")])
            evolve = FakeEvolve()
            self.assertValid(
                run_evolution_session(
                    config,
                    root / "session",
                    acceptance_checkpoint="ckpt",
                    acceptance_base_model="base",
                    judge_holdout_paths=("holdout.jsonl", "exam.jsonl"),
                    evolve_runner=evolve,
                    clock=_Clock(),
                    sleeper=lambda _s: None,
                )
            )
            self.assertEqual(["ckpt", "ckpt"], [call["acceptance_checkpoint"] for call in evolve.calls])
            self.assertEqual(["base", "base"], [call["acceptance_base_model"] for call in evolve.calls])
            self.assertEqual([("holdout.jsonl", "exam.jsonl"), ("holdout.jsonl", "exam.jsonl")], [call["judge_holdout_paths"] for call in evolve.calls])

    def test_a_paper_that_names_its_own_checkpoint_keeps_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            station = self._evolve_station(root, "paper-a", acceptance_checkpoint="paper-ckpt")
            config = _config(root, [station])
            evolve = FakeEvolve()
            self.assertValid(
                run_evolution_session(
                    config,
                    root / "session",
                    acceptance_checkpoint="session-ckpt",
                    acceptance_base_model="base",
                    evolve_runner=evolve,
                    clock=_Clock(),
                    sleeper=lambda _s: None,
                )
            )
            self.assertEqual("paper-ckpt", evolve.calls[0]["acceptance_checkpoint"])
            self.assertEqual("base", evolve.calls[0]["acceptance_base_model"])

    def test_a_config_level_checkpoint_reaches_a_paper_that_named_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(
                root,
                [self._evolve_station(root, "paper-a")],
                acceptance_checkpoint="from-config",
                acceptance_base_model="from-base",
                judge_holdout_paths=["exam.jsonl"],
            )
            evolve = FakeEvolve()
            self.assertValid(run_evolution_session(config, root / "session", evolve_runner=evolve, clock=_Clock(), sleeper=lambda _s: None))
            self.assertEqual("from-config", evolve.calls[0]["acceptance_checkpoint"])
            self.assertEqual("from-base", evolve.calls[0]["acceptance_base_model"])
            self.assertEqual(("exam.jsonl",), evolve.calls[0]["judge_holdout_paths"])

    def test_a_domain_validated_row_counts_as_a_passed_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            config = _config(root, [self._evolve_station(root, "paper-a")], validation_suite=str(_suite(root)))
            evolve = FakeEvolve(experience_id="direct-exp-new", applicability="domain")
            validate = FakeValidate(status="domain_validated")
            report = self.assertValid(
                run_evolution_session(config, root / "session", experience_library=library, evolve_runner=evolve, validate_runner=validate, clock=_Clock(), sleeper=lambda _s: None)
            )
            self.assertEqual(["evolve:paper-a", "validate:direct-exp-new"], [item["station_id"] for item in report["stations"]])
            self.assertTrue(report["stations"][1]["ok"])
            self.assertEqual("domain_validated", report["stations"][1]["evidence"]["status_after"])

    def test_an_unlabeled_experience_is_saved_and_not_queued_for_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "session"
            library = root / "library"
            config = _config(root, [self._evolve_station(root, "paper-a")], validation_suite=str(_suite(root)))
            validate = FakeValidate()
            report = self.assertValid(
                run_evolution_session(
                    config,
                    out,
                    experience_library=library,
                    evolve_runner=FakeEvolve(experience_id="direct-exp-new", applicability=""),
                    validate_runner=validate,
                    clock=_Clock(),
                    sleeper=lambda _s: None,
                )
            )
            self.assertEqual([], validate.calls)
            self.assertEqual(["evolve:paper-a"], [item["station_id"] for item in report["stations"]])
            self.assertEqual("direct-exp-new", ExperienceStore(library).get("direct-exp-new")["experience_id"])
            events = [json.loads(line) for line in (out / "evolution_session_events.jsonl").read_text(encoding="utf-8").splitlines()]
            declined = [item for item in events if item["event"] == "auto_validate_declined"]
            self.assertEqual(["applicability was not written"], [item["detail"] for item in declined])
            # The dead end is named rather than left looking like an ordinary candidate: the
            # library says the row cannot be examined, and the session says it out loud.
            self.assertEqual("unverifiable", ExperienceStore(library).lifecycle_status("direct-exp-new", ""))
            unverifiable = [item for item in events if item["event"] == "experience_unverifiable"]
            self.assertEqual(["direct-exp-new"], [item["experience_id"] for item in unverifiable])

    def test_a_paper_run_that_earned_nothing_queues_no_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root, [self._evolve_station(root, "paper-a")], validation_suite=str(_suite(root)))
            validate = FakeValidate()
            report = self.assertValid(
                run_evolution_session(config, root / "session", experience_library=root / "library", evolve_runner=FakeEvolve(), validate_runner=validate, clock=_Clock(), sleeper=lambda _s: None)
            )
            self.assertEqual([], validate.calls)
            self.assertEqual(1, report["counts"]["stations"])

    def test_a_validation_with_nothing_to_validate_says_so_instead_of_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root, [{"kind": "validate", "experience_id": "direct-exp-missing"}], validation_suite=str(_suite(root)))
            validate = FakeValidate()
            report = self.assertValid(
                run_evolution_session(config, root / "session", experience_library=root / "library", evolve_runner=FakeEvolve(), validate_runner=validate, clock=_Clock(), sleeper=lambda _s: None)
            )
            self.assertEqual([], validate.calls)
            self.assertEqual("skipped", report["stations"][0]["status"])
            self.assertEqual("no_candidate_experience", report["stations"][0]["reason"])
            self.assertTrue(report["ok"])

    def test_an_experience_already_promoted_is_not_validated_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            ExperienceStore(library).append_result(_experience_row("direct-exp-done", status="promoted"))
            config = _config(root, [{"kind": "validate", "experience_id": "direct-exp-done"}], validation_suite=str(_suite(root)))
            validate = FakeValidate()
            report = self.assertValid(
                run_evolution_session(config, root / "session", experience_library=library, evolve_runner=FakeEvolve(), validate_runner=validate, clock=_Clock(), sleeper=lambda _s: None)
            )
            self.assertEqual([], validate.calls)
            self.assertEqual("no_candidate_experience", report["stations"][0]["reason"])

    def test_validation_can_be_switched_off_and_the_report_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root, [{"kind": "validate", "experience_id": "direct-exp-new"}, self._evolve_station(root, "paper-a")])
            validate = FakeValidate()
            report = self.assertValid(
                run_evolution_session(config, root / "session", evolve_runner=FakeEvolve(), validate_runner=validate, validate=False, clock=_Clock(), sleeper=lambda _s: None)
            )
            self.assertEqual([], validate.calls)
            self.assertEqual(["skipped", "completed"], [item["status"] for item in report["stations"]])
            self.assertEqual("validation_disabled", report["stations"][0]["reason"])

    def test_a_rate_limited_station_is_retried_into_a_fresh_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "session"
            clock = _Clock()
            config = _config(root, [self._evolve_station(root, "paper-a")])
            evolve = FakeEvolve(clock=clock, failures=2, error=_rate_limited())
            sleeps: list[float] = []
            report = self.assertValid(
                run_evolution_session(config, out, evolve_runner=evolve, clock=clock, sleeper=lambda seconds: sleeps.append(seconds), rate_limit_cooldown=45.0, rate_limit_retries=3)
            )
            self.assertEqual(3, len(evolve.calls))
            self.assertEqual([45.0, 45.0], sleeps)
            self.assertEqual("completed", report["stations"][0]["status"])
            self.assertEqual([1, 2, 3], [item["attempt"] for item in report["stations"][0]["attempts"]])
            self.assertEqual(["failed", "failed", "ok"], [item["status"] for item in report["stations"][0]["attempts"]])
            self.assertTrue(str(report["stations"][0]["out_dir"]).endswith("attempt_03"))
            # A retry writes a new directory each time; the two the provider refused are
            # still named, so nothing the earlier attempt wrote can be overwritten.
            self.assertEqual(3, len({item["out_dir"] for item in report["stations"][0]["attempts"]}))
            self.assertTrue((out / "papers" / "paper-a" / "attempt_03").is_dir())
            self.assertTrue(all(str(row["out_dir"]).endswith(f"attempt_0{number}") for number, row in enumerate(report["stations"][0]["attempts"], start=1)))

    def test_a_requests_style_image_429_is_retried_like_vlm_http_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clock = _Clock()
            config = _config(root, [self._evolve_station(root, "paper-a")])
            evolve = FakeEvolve(clock=clock, failures=1, error=RuntimeError("429 Client Error: Too Many Requests for url: https://api.example/v1/images/generations"))
            sleeps: list[float] = []
            report = self.assertValid(
                run_evolution_session(config, root / "session", evolve_runner=evolve, clock=clock, sleeper=lambda seconds: sleeps.append(seconds), rate_limit_cooldown=45.0, rate_limit_retries=3)
            )
            self.assertEqual(2, len(evolve.calls))
            self.assertEqual([45.0], sleeps)
            self.assertEqual("completed", report["stations"][0]["status"])
            self.assertEqual("rate_limited", report["stations"][0]["attempts"][0]["reason"])

    def test_a_dropped_tls_session_is_retried_into_a_fresh_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clock = _Clock()
            config = _config(root, [self._evolve_station(root, "paper-a")])
            evolve = FakeEvolve(clock=clock, failures=1, error=ssl.SSLError(8, "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol"))
            sleeps: list[float] = []
            report = self.assertValid(
                run_evolution_session(config, root / "session", evolve_runner=evolve, clock=clock, sleeper=lambda seconds: sleeps.append(seconds), rate_limit_cooldown=45.0, rate_limit_retries=3)
            )
            self.assertEqual(2, len(evolve.calls))
            self.assertEqual([45.0], sleeps)
            self.assertEqual("completed", report["stations"][0]["status"])
            self.assertEqual("run_failed", report["stations"][0]["attempts"][0]["reason"])

    def test_a_station_that_keeps_getting_rate_limited_fails_and_the_queue_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clock = _Clock()
            config = _config(root, [self._evolve_station(root, "paper-a"), self._evolve_station(root, "paper-b")])
            failures = FakeEvolve(clock=clock, failures=99, error=_rate_limited())
            following = FakeEvolve()
            calls: list[str] = []

            def runner(**options):
                if str(options["paper"]).endswith("paper-a.pdf"):
                    return failures(**options)
                calls.append(str(options["paper"]))
                return following(**options)

            report = run_evolution_session(config, root / "session", evolve_runner=runner, clock=clock, sleeper=lambda _s: None, rate_limit_retries=3)
            self.assertValid(report)
            self.assertFalse(report["ok"])
            self.assertEqual("failed", report["stations"][0]["status"])
            self.assertEqual("rate_limited", report["stations"][0]["reason"])
            self.assertEqual(4, len(report["stations"][0]["attempts"]))
            self.assertEqual(["completed"], [item["status"] for item in report["stations"][1:]])
            self.assertEqual(1, len(calls))
            self.assertEqual("rate_limited", report["blockers"][0]["reason"])

    def test_a_real_defect_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clock = _Clock()
            config = _config(root, [self._evolve_station(root, "paper-a")])
            evolve = FakeEvolve(failures=99, error=RuntimeError("a defect in our own code"))
            report = run_evolution_session(config, root / "session", evolve_runner=evolve, clock=clock, sleeper=lambda _s: None, rate_limit_retries=3)
            self.assertValid(report)
            self.assertEqual(1, len(evolve.calls))
            self.assertEqual("run_failed", report["stations"][0]["reason"])
            self.assertEqual(1, len(report["stations"][0]["attempts"]))

    def test_no_retries_means_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root, [self._evolve_station(root, "paper-a")])
            evolve = FakeEvolve(failures=99, error=_rate_limited())
            report = self.assertValid(
                run_evolution_session(config, root / "session", evolve_runner=evolve, clock=_Clock(), sleeper=lambda _s: None, rate_limit_retries=0)
            )
            self.assertEqual(1, len(evolve.calls))
            self.assertEqual("rate_limited", report["stations"][0]["reason"])

    def test_a_validation_that_fails_names_its_own_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            ExperienceStore(library).append_result(_experience_row("direct-exp-x"))
            config = _config(root, [{"kind": "validate", "experience_id": "direct-exp-x"}], validation_suite=str(_suite(root)))
            validate = FakeValidate(failures=99, error=RuntimeError("the harness fell over"))
            report = run_evolution_session(config, root / "session", experience_library=library, evolve_runner=FakeEvolve(), validate_runner=validate, clock=_Clock(), sleeper=lambda _s: None)
            self.assertValid(report)
            self.assertEqual("validation_failed", report["stations"][0]["reason"])
            self.assertEqual(1, len(validate.calls))

    def test_a_paper_that_cannot_be_read_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root, [{"kind": "evolve", "paper": str(root / "gone.pdf"), "paper_id": "paper-gone"}])
            evolve = FakeEvolve()
            report = self.assertValid(run_evolution_session(config, root / "session", evolve_runner=evolve, clock=_Clock(), sleeper=lambda _s: None))
            self.assertEqual([], evolve.calls)
            self.assertEqual("failed", report["stations"][0]["status"])
            self.assertEqual("paper_unreadable", report["stations"][0]["reason"])
            self.assertFalse(report["ok"])

    def test_the_budget_drops_the_rest_of_the_queue_without_cutting_the_running_station(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clock = _Clock()
            config = _config(root, [self._evolve_station(root, "paper-a"), self._evolve_station(root, "paper-b"), self._evolve_station(root, "paper-c")])
            evolve = FakeEvolve(clock=clock, spend=2.0)
            report = self.assertValid(
                run_evolution_session(config, root / "session", evolve_runner=evolve, clock=clock, sleeper=lambda _s: None, hours=1.0 / 3600.0)
            )
            self.assertEqual(1, len(evolve.calls))
            self.assertEqual(["completed", "skipped", "skipped"], [item["status"] for item in report["stations"]])
            self.assertEqual(["", "budget_exhausted", "budget_exhausted"], [item["reason"] for item in report["stations"]])
            self.assertTrue(report["budget"]["exhausted"])
            self.assertTrue(report["ok"])

    def test_a_dry_run_lists_the_queue_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "session"
            config = _config(root, [self._evolve_station(root, "paper-a"), {"kind": "validate", "experience_id": "direct-exp-x"}], validation_suite=str(_suite(root)))
            evolve, validate = FakeEvolve(), FakeValidate()
            report = self.assertValid(
                run_evolution_session(config, out, evolve_runner=evolve, validate_runner=validate, clock=_Clock(), sleeper=lambda _s: None, dry_run=True)
            )
            self.assertEqual([], evolve.calls)
            self.assertEqual([], validate.calls)
            self.assertEqual(["pending", "pending"], [item["status"] for item in report["stations"]])
            self.assertEqual(str((out / "papers" / "paper-a" / "attempt_01").resolve()), str(Path(report["stations"][0]["out_dir"]).resolve()))
            self.assertFalse(out.exists())

    def test_a_run_that_reported_its_own_quality_failure_is_still_recorded_as_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root, [self._evolve_station(root, "paper-a")])
            report = self.assertValid(run_evolution_session(config, root / "session", evolve_runner=FakeEvolve(ok=False), clock=_Clock(), sleeper=lambda _s: None))
            self.assertEqual("completed", report["stations"][0]["status"])
            self.assertFalse(report["stations"][0]["ok"])
            self.assertFalse(report["stations"][0]["evidence"]["run_ok"])
            self.assertTrue(report["ok"])

    def test_a_validate_station_without_a_suite_is_refused_before_anything_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root, [{"kind": "validate", "experience_id": "direct-exp-x"}])
            with self.assertRaises(ValueError) as caught:
                run_evolution_session(config, root / "session", evolve_runner=FakeEvolve(), clock=_Clock(), sleeper=lambda _s: None)
            self.assertIn("suite", str(caught.exception))

    def test_the_queue_is_read_from_the_shorthand_lists_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            ExperienceStore(library).append_result(_experience_row("direct-exp-x"))
            config = write_json(
                root / "session.json",
                {
                    "session_id": "shorthand",
                    "validations": [{"experience_id": "direct-exp-x"}],
                    "papers": [{"paper": str(_paper(root, "paper-a.pdf")), "paper_id": "paper-a"}],
                },
            )
            validate = FakeValidate()
            report = self.assertValid(
                run_evolution_session(config, root / "session", validation_suite=str(_suite(root)), experience_library=library, evolve_runner=FakeEvolve(), validate_runner=validate, clock=_Clock(), sleeper=lambda _s: None)
            )
            self.assertEqual(["validate:direct-exp-x", "evolve:paper-a"], [item["station_id"] for item in report["stations"]])

    def test_the_session_writes_a_report_a_markdown_and_an_event_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "session"
            config = _config(root, [self._evolve_station(root, "paper-a")])
            report = self.assertValid(run_evolution_session(config, out, evolve_runner=FakeEvolve(), clock=_Clock(), sleeper=lambda _s: None))
            written = json.loads((out / "evolution_session_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report, written)
            self.assertIn("Evolution session report", (out / "evolution_session_report.md").read_text(encoding="utf-8"))
            events = [json.loads(line) for line in (out / "evolution_session_events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(["station_started", "attempt_started", "attempt_completed", "station_finished"], [item["event"] for item in events])


class SessionCliTests(SessionTestCase):
    def test_the_parser_accepts_the_acceptance_checkpoint_and_holdout_lists(self):
        args = build_parser().parse_args(
            [
                "paper-evolve-session",
                "--config",
                "c.json",
                "--out",
                "o",
                "--acceptance-checkpoint",
                "ckpt",
                "--acceptance-base-model",
                "base",
                "--judge-holdout",
                "a.jsonl",
                "--judge-holdout",
                "b.jsonl",
            ]
        )
        self.assertEqual("ckpt", args.acceptance_checkpoint)
        self.assertEqual("base", args.acceptance_base_model)
        self.assertEqual(["a.jsonl", "b.jsonl"], args.judge_holdout)

    def test_the_parser_exposes_the_session_command(self):
        parser = build_parser()
        self.assertIn("paper-evolve-session", parser._subparsers._group_actions[0].choices)

    def test_a_dry_run_prints_the_queue_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "session"
            config = _config(root, [{"kind": "evolve", "paper": str(root / "paper-a.pdf"), "paper_id": "paper-a"}, {"kind": "evolve", "paper": str(root / "paper-b.pdf"), "paper_id": "paper-b"}])
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = main(["paper-evolve-session", "--config", str(config), "--out", str(out), "--hours", "5", "--dry-run", "--json"])
            payload = json.loads(buffer.getvalue())
            self.assertEqual(0, code)
            self.assertTrue(payload["ok"])
            self.assertEqual(["pending", "pending"], [item["status"] for item in payload["stations"]])
            self.assertEqual(5.0, payload["budget"]["hours"])
            self.assertFalse(out.exists())

    def test_a_validate_station_without_a_suite_exits_non_zero_and_says_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root, [{"kind": "validate", "experience_id": "direct-exp-x"}])
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                code = main(["paper-evolve-session", "--config", str(config), "--out", str(root / "session"), "--json"])
            self.assertEqual(1, code)
            self.assertIn("suite", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
