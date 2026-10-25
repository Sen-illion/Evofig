"""The loop measures the candidates it earns, in the background, once each.

A new experience is inert until another paper has measured it, and that measurement is a full
paired ablation -- hours of work. These tests pin the queue: which candidates are handed over,
what the measurement is told about the Judge seats, what happens when it fails, and what a
restart does with a measurement that was interrupted.
"""

import json
import time
import unittest
from pathlib import Path
import tempfile

from rfs.contracts.prompt_patch import PROMPT_PATCH_REFUTATION_LIMIT
from rfs.evolution.direct import PromptPatchStore
from rfs.utils import write_json, write_text
from rfs.workflows import continuous_evolution as ce
from rfs.workflows.continuous_evolution import (
    CONTINUOUS_EVENTS,
    CONTINUOUS_STATE,
    MAX_UNMEASURED_REQUEUES,
    load_continuous_state,
    read_continuous_config,
    run_continuous_evolution,
)

EXPERIENCE = "direct-exp-5075381877abebb7"
PATCH = "prompt-patch-0123456789abcdef"


def _config(**overrides) -> dict:
    block = {
        "base_papers": [{"paper": "/papers/one.pdf", "paper_id": "one", "domain": "biology"}],
        "budget_hours": 24.0,
        "iteration_rounds": 2,
        "evolve": {"judge_mode": "vlm", "online_judge_checkpoint": "/ckpt/step-003174", "online_judge_base_model": "/models/Qwen3.5-9B"},
    }
    block.update(overrides)
    return {"continuous": block}


def _events(out: Path) -> list[dict]:
    path = out / CONTINUOUS_EVENTS
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _file_candidate(out_dir: Path, experience_id: str, options: dict) -> None:
    """File a candidate the way a real iteration does: the round's own file, and the library row.

    The loop only measures a row the library still holds -- a candidate file that names a row no
    library ever heard of is what used to buy a wave that died on its first step -- so a stand-in
    that files one without the other is modelling a run that cannot happen.
    """

    write_json(out_dir / "round_00" / "experience_candidate.json", {"experience_id": experience_id, "applicability": "common"})
    library = Path(str(options.get("experience_library") or ""))
    if not library:
        return
    library.mkdir(parents=True, exist_ok=True)
    with (library / "judge_experiences.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"experience_id": experience_id, "applicability": "common"}) + "\n")


def _evolve_runner(calls: list, *, experience_id: str = EXPERIENCE):
    """A stand-in run: it files an iteration that earned one candidate experience."""

    def runner(**options):
        calls.append(dict(options))
        out = Path(options["out"])
        out.mkdir(parents=True, exist_ok=True)
        best = out / "round_00" / "candidate_00"
        best.mkdir(parents=True, exist_ok=True)
        write_json(out / "evolution_state.json", {"best_candidate_dir": str(best), "best_score": 1.0})
        write_text(out / "final_draft_preview.png", "png")
        if experience_id:
            _file_candidate(out, experience_id, options)
        if options.get("reuse_upstream_from"):
            write_json(out / "stage_reuse_report.json", {"source": str(options["reuse_upstream_from"]), "reused_stages": ["design_seed"]})
        return {"ok": True, "out_dir": str(out), "rounds_completed": 2}

    return runner


def _file_patch(library: Path, patch_id: str, *, status: str = "current_case_validated", directive: str = "Route connectors through a shared vertical channel.") -> None:
    """File a patch the way a real round does: the library holds the row the round permitted."""

    row = {
        "patch_id": patch_id,
        "version": 1,
        "apply_at": "ppt_compilation",
        "class": "advisory",
        "action_type": "reroute",
        "directive": directive,
        "condition": {"topology": "pipeline"},
        "source_experience_ids": ["direct-exp-source"],
        "median_repair_gain": 0.2,
        "provenance": {"source": "direct_evolution_process", "run": "one", "round": 1, "measured": True},
        "status": status,
    }
    stored = PromptPatchStore(library).append_result(row)
    assert stored["ok"], stored


def _evolve_runner_with_patch(calls: list, *, experience_id: str = "", patch_id: str = PATCH):
    """A stand-in run that earned one prompt patch this round."""

    def runner(**options):
        calls.append(dict(options))
        out = Path(options["out"])
        out.mkdir(parents=True, exist_ok=True)
        best = out / "round_00" / "candidate_00"
        best.mkdir(parents=True, exist_ok=True)
        write_json(out / "evolution_state.json", {"best_candidate_dir": str(best), "best_score": 1.0})
        write_text(out / "final_draft_preview.png", "png")
        if experience_id:
            _file_candidate(out, experience_id, options)
        return {
            "ok": True,
            "out_dir": str(out),
            "rounds_completed": 2,
            "rounds": ([{"prompt_patch_recorded": patch_id}] if patch_id else []),
        }

    return runner


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class RowsOwedASecondPaperTests(unittest.TestCase):
    """A row that passed its first measurement is handed the paper that tests whether it travels.

    The ladder asks two questions of a row: does it reproduce where it came from, and does it
    travel to a paper that did not write it. Only a row born on a paper that had already measured
    it was ever asked the second one, so a row measured for the first time was written down as
    ``quick_validated`` and left with nothing behind it: it stalled one rung short of travel for
    ever, and ``cross_domain_follow_ups`` stayed empty. These tests pin the queue that asks the
    second question, the rows it is owed for, and the patch that has to arrive as a patch.
    """

    def _loop(self, root: Path, library: Path) -> "ce._Loop":
        return ce._Loop(
            config=read_continuous_config(_config(validation={"suite": "/suites/four_groups.json"})),
            out=root,
            library=str(library),
            stop_file="",
            evolve_runner=lambda **options: {},
            review_runner=lambda previous, spec: {"ok": False},
            clock=lambda: 0.0,
            disk_usage=lambda path: {"ratio": 0.9, "free_gb": 900.0},
            max_waves=1,
        )

    def _library_row(self, library: Path, experience_id: str, status: str) -> None:
        library.mkdir(parents=True, exist_ok=True)
        row = {"experience_id": experience_id, "status": status, "applicability": "common"}
        with (library / "judge_experiences.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def _queue(self, loop: "ce._Loop") -> list[dict]:
        return [dict(item) for item in list(loop._validation_queue.queue)]

    def test_a_row_measured_for_the_first_time_is_handed_the_paper_that_tests_travel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = self._loop(root, root / "library")
            record = loop.settle_measurement_rung(
                {
                    "experience_id": EXPERIENCE,
                    "status": "quick_validated",
                    "purpose": "candidate",
                    "starting_status": "candidate",
                    "starting_applicability": "common",
                }
            )
            self.assertEqual(record["rung_awarded"], "quick_validated")
            self.assertEqual(loop.state["cross_domain_follow_ups"][EXPERIENCE], "queued")
            self.assertIn("cross_domain_follow_up_queued", [event["event"] for event in _events(root)])
            self.assertEqual(self._queue(loop)[0]["purpose"], "cross_domain")

    def test_a_row_already_measured_is_picked_up_from_the_library_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            self._library_row(library, EXPERIENCE, "quick_validated")
            loop = self._loop(root, library)
            self.assertEqual(loop.resume_rows_owed_a_second_paper(), [EXPERIENCE])
            self.assertEqual(loop.state["cross_domain_follow_ups"][EXPERIENCE], "queued")
            self.assertEqual(loop.resume_rows_owed_a_second_paper(), [])
            self.assertEqual(len(self._queue(loop)), 1)

    def test_a_licensed_patch_is_handed_another_paper_as_a_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            library.mkdir(parents=True, exist_ok=True)
            _file_patch(library, PATCH)
            loop = self._loop(root, library)
            self.assertEqual(loop.resume_rows_owed_a_second_paper(), [PATCH])
            item = self._queue(loop)[0]
            self.assertEqual(item["kind"], "patch")
            self.assertTrue(item["prompt_patch"])
            self.assertEqual(item["prompt_patch_id"], PATCH)

    def test_a_row_that_already_travelled_or_lost_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            self._library_row(library, "direct-exp-travelled", "cross_domain_validated")
            self._library_row(library, "direct-exp-refuted", "refuted")
            self._library_row(library, "direct-exp-quiet", "quick_validated")
            loop = self._loop(root, library)
            loop.state.setdefault("cross_domain_follow_ups", {})["direct-exp-quiet"] = "done"
            self.assertEqual(loop.resume_rows_owed_a_second_paper(), [])
            self.assertEqual(self._queue(loop), [])


class ValidationQueueTests(unittest.TestCase):
    def test_a_new_candidate_is_measured_with_the_seats_the_loop_runs_with(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            seen: list[dict] = []

            def validate(**options):
                seen.append(dict(options))
                return {"status": "same_domain_validated", "ladder": {"stopped_because": "the ladder stopped at the same-domain rung"}, "summary": {"passed": 2}, "all_pairs_isolated": True, "all_cases_measured": True}

            report = run_continuous_evolution(
                _config(validation={"suite": "/suites/four_groups.json"}),
                out,
                experience_library=str(out / "library"),
                evolve_runner=_evolve_runner([]),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=validate,
                clock=lambda: 0.0,
                max_waves=1,
            )
            self.assertTrue(_wait_for(lambda: seen))

            # The row stops at the same-domain rung, so the loop also queues its one other-domain
            # paper behind it; the assertion is about the first hand-over, not about the count.
            self.assertGreaterEqual(len(seen), 1)
            # The suite was written with whatever seats were seated when it was built; the loop
            # overrides them with the seats this run actually uses, or the ablation would be
            # measured by a different model than the round that earned the experience.
            self.assertEqual("/suites/four_groups.json", seen[0]["suite"])
            self.assertEqual(EXPERIENCE, seen[0]["experience_id"])
            self.assertEqual(str(out / "library"), seen[0]["experience_library"])
            self.assertEqual("/ckpt/step-003174", seen[0]["run_options"]["online_judge_checkpoint"])

            self.assertTrue(
                _wait_for(lambda: any(item["event"] == "validation_finished" for item in _events(out)))
            )
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
            self.assertEqual("same_domain_validated", state["validations"][EXPERIENCE]["status"])
            self.assertEqual("the ladder stopped at the same-domain rung", state["validations"][EXPERIENCE]["stopped_because"])
            self.assertTrue(report["validation"]["enabled"])
            self.assertEqual("/suites/four_groups.json", report["validation"]["suite"])

    def test_the_same_candidate_is_not_measured_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            seen: list[dict] = []

            def validate(**options):
                seen.append(dict(options))
                return {"status": "promoted", "ladder": {}, "summary": {}}

            run_continuous_evolution(
                _config(validation={"suite": "/suites/four_groups.json"}),
                out,
                experience_library=str(out / "library"),
                evolve_runner=_evolve_runner([]),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=validate,
                clock=lambda: 0.0,
                max_waves=2,
            )
            self.assertTrue(_wait_for(lambda: seen))
            time.sleep(0.2)

            self.assertEqual(1, len(seen))
            queued = [item for item in _events(out) if item["event"] == "validation_queued"]
            self.assertEqual(1, len(queued))

    def test_validation_off_names_the_candidate_it_left_unmeasured(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            seen: list[dict] = []

            run_continuous_evolution(
                _config(),
                out,
                experience_library=str(out / "library"),
                evolve_runner=_evolve_runner([]),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=lambda **options: seen.append(dict(options)) or {"status": "promoted"},
                clock=lambda: 0.0,
                max_waves=1,
            )

            self.assertEqual([], seen)
            skipped = [item for item in _events(out) if item["event"] == "validation_skipped"]
            self.assertEqual(1, len(skipped))
            self.assertEqual(EXPERIENCE, skipped[0]["experience_id"])
            self.assertIn("no validation block", skipped[0]["reason"])

    def test_a_measurement_that_failed_is_recorded_and_the_loop_carries_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            def validate(**options):
                raise RuntimeError("the provider refused the run")

            report = run_continuous_evolution(
                _config(validation={"suite": "/suites/four_groups.json"}),
                out,
                experience_library=str(out / "library"),
                evolve_runner=_evolve_runner([]),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=validate,
                clock=lambda: 0.0,
                max_waves=1,
            )
            self.assertTrue(_wait_for(lambda: any(item["event"] == "validation_finished" for item in _events(out))))

            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
            self.assertEqual("failed", state["validations"][EXPERIENCE]["status"])
            self.assertIn("the provider refused the run", state["validations"][EXPERIENCE]["error"])
            self.assertNotIn("error", report["stop_detail"] or "")

    def test_an_interrupted_measurement_is_taken_up_again_by_the_next_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_json(
                out / CONTINUOUS_STATE,
                {
                    "schema": "rfs.continuous_evolution_state",
                    "version": 1,
                    "papers": {},
                    "validations": {EXPERIENCE: {"status": "running", "paper": "one", "iteration": 1, "out": ""}},
                    "waves": 3,
                },
            )
            # The row the interrupted measurement was about is still in the library, which is
            # what makes it a measurement to take up rather than an entry to drop.
            _file_candidate(out / "iter", EXPERIENCE, {"experience_library": str(out / "library")})
            seen: list[dict] = []

            def validate(**options):
                seen.append(dict(options))
                return {"status": "promoted", "ladder": {}, "summary": {}}

            run_continuous_evolution(
                _config(validation={"suite": "/suites/four_groups.json"}),
                out,
                experience_library=str(out / "library"),
                evolve_runner=_evolve_runner([], experience_id=""),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=validate,
                clock=lambda: 0.0,
                max_waves=1,
            )

            self.assertTrue(_wait_for(lambda: seen))
            self.assertEqual([EXPERIENCE], [item["experience_id"] for item in seen])
            requeued = [item for item in _events(out) if item["event"] == "validation_requeued"]
            self.assertEqual([[EXPERIENCE]], [item["experience_ids"] for item in requeued])

    def test_a_measurement_left_queued_by_an_earlier_call_is_taken_up_again(self):
        # The previous call ended between putting this row back on the queue and the worker
        # reaching it. The queue itself died with that process, so unless a restart takes
        # ``requeued`` up as well the row waits for ever -- which is what the rows that had
        # passed qualification did.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_json(
                out / CONTINUOUS_STATE,
                {
                    "schema": "rfs.continuous_evolution_state",
                    "version": 1,
                    "papers": {},
                    "validations": {EXPERIENCE: {"status": "requeued", "paper": "one", "iteration": 1, "out": ""}},
                    "waves": 3,
                },
            )
            _file_candidate(out / "iter", EXPERIENCE, {"experience_library": str(out / "library")})
            seen: list[dict] = []

            def validate(**options):
                seen.append(dict(options))
                return {"status": "promoted", "ladder": {}, "summary": {}}

            run_continuous_evolution(
                _config(validation={"suite": "/suites/four_groups.json"}),
                out,
                experience_library=str(out / "library"),
                evolve_runner=_evolve_runner([], experience_id=""),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=validate,
                clock=lambda: 0.0,
                max_waves=1,
            )

            self.assertTrue(_wait_for(lambda: seen))
            self.assertEqual([EXPERIENCE], [item["experience_id"] for item in seen])

    def test_a_row_the_library_no_longer_holds_is_dropped_instead_of_retried(self):
        # The library is not frozen while a measurement waits its turn: rows are archived,
        # retired and rewritten underneath it. Asking for a row nothing holds buys a wave that
        # can only end in "unknown experience", so it is dropped once and named.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_json(
                out / CONTINUOUS_STATE,
                {
                    "schema": "rfs.continuous_evolution_state",
                    "version": 1,
                    "papers": {},
                    "validations": {EXPERIENCE: {"status": "requeued", "paper": "one", "iteration": 1, "out": ""}},
                    "waves": 3,
                },
            )
            library = out / "library"
            library.mkdir(parents=True, exist_ok=True)
            (library / "judge_experiences.jsonl").write_text(
                json.dumps({"experience_id": "direct-exp-someone-else", "status": "candidate"}) + "\n",
                encoding="utf-8",
            )
            seen: list[dict] = []

            run_continuous_evolution(
                _config(validation={"suite": "/suites/four_groups.json"}),
                out,
                experience_library=str(library),
                evolve_runner=_evolve_runner([], experience_id=""),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=lambda **options: seen.append(dict(options)) or {"status": "promoted"},
                clock=lambda: 0.0,
                max_waves=1,
            )

            # The row the library does hold is still measured; the one it does not is not.
            self.assertTrue(_wait_for(lambda: seen))
            self.assertNotIn(EXPERIENCE, [item["experience_id"] for item in seen])
            dropped = [item for item in _events(out) if item["event"] == "validation_dropped"]
            self.assertEqual([EXPERIENCE], [item["experience_id"] for item in dropped])
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
            self.assertEqual("dropped", state["validations"][EXPERIENCE]["status"])
            self.assertEqual("row_no_longer_in_library", state["validations"][EXPERIENCE]["blocked_because"])

    def test_a_config_without_a_validation_block_queues_nothing(self):
        config = read_continuous_config(_config())
        self.assertFalse(config["validation"]["enabled"])
        self.assertEqual({}, load_continuous_state(Path(tempfile.mkdtemp()), config).get("validations"))


class UnmeasuredCandidateTests(unittest.TestCase):
    """A candidate nothing measured is queued again, not left measured.

    A validation that ends with zero cases read nothing: the case set could not be drawn, or the
    exam was refused before a paper ran. Recording that as a decision is how a candidate the code
    could not measure stayed unmeasured for ever -- the state called it measured, the queue never
    saw it again, the library only ever grew, and nothing was ever retrieved from it.
    """

    def _library(self, out: Path, *, status: str = "candidate") -> str:
        library = out / "library"
        library.mkdir(parents=True, exist_ok=True)
        row = {"domain": "direct_editable", "experience_id": EXPERIENCE, "status": status}
        (library / "judge_experiences.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        return str(library)

    def _state(self, out: Path, *, measured: int, status: str, counters: dict | None = None) -> None:
        state = {
            "schema": "rfs.continuous_evolution_state",
            "version": 1,
            "papers": {},
            "waves": 3,
            "validations": {
                EXPERIENCE: {
                    "status": status,
                    "paper": "one",
                    "iteration": 1,
                    "out": "",
                    "summary": {"cases": measured, "measured_cases": measured},
                }
            },
        }
        if counters is not None:
            state["unmeasured_requeues"] = counters
        write_json(out / CONTINUOUS_STATE, state)

    def _run(self, out: Path, library: str, validate) -> None:
        run_continuous_evolution(
            _config(validation={"suite": "/suites/four_groups.json"}),
            out,
            experience_library=library,
            evolve_runner=_evolve_runner([], experience_id=""),
            review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
            validate_runner=validate,
            clock=lambda: 0.0,
            max_waves=1,
        )

    def test_a_candidate_measured_with_zero_cases_is_queued_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = self._library(out)
            self._state(out, measured=0, status="current_case_validated")
            seen: list[dict] = []

            def validate(**options):
                seen.append(dict(options))
                return {"status": "promoted", "ladder": {}, "summary": {"measured_cases": 4}}

            self._run(out, library, validate)

            self.assertTrue(_wait_for(lambda: seen))
            self.assertEqual([EXPERIENCE], [item["experience_id"] for item in seen])
            queued = [item for item in _events(out) if item["event"] == "validation_queued"]
            self.assertEqual([EXPERIENCE], [item["experience_id"] for item in queued])
            self.assertIn("no case was ever measured", queued[0]["because"])

    def test_a_candidate_that_really_was_measured_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = self._library(out)
            self._state(out, measured=4, status="same_domain_validated")
            seen: list[dict] = []

            self._run(out, library, lambda **options: seen.append(dict(options)) or {"status": "promoted"})

            self.assertEqual([], seen)
            self.assertEqual([], [item for item in _events(out) if item["event"] == "validation_queued"])

    def test_a_row_that_is_not_a_candidate_is_not_queued(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = self._library(out, status="refuted")
            self._state(out, measured=0, status="refuted")
            seen: list[dict] = []

            self._run(out, library, lambda **options: seen.append(dict(options)) or {"status": "promoted"})

            self.assertEqual([], seen)

    def test_the_loop_stops_asking_after_three_unmeasured_tries(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = self._library(out)
            self._state(out, measured=0, status="current_case_validated", counters={EXPERIENCE: MAX_UNMEASURED_REQUEUES})
            seen: list[dict] = []

            self._run(out, library, lambda **options: seen.append(dict(options)) or {"status": "promoted"})

            self.assertEqual([], seen)
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
            self.assertEqual(MAX_UNMEASURED_REQUEUES, state["unmeasured_requeues"][EXPERIENCE])


class HandoverTimingTests(unittest.TestCase):
    def test_a_finished_paper_hands_its_candidate_over_without_waiting_for_a_slow_one(self):
        """The wave is as slow as its slowest paper; the measurement must not be.

        A wave moves two papers at once and an iteration is hours long, so handing the candidates
        over at the end of the wave would delay a measurement that is already ready by as long as
        the other paper takes. The candidate is handed over by the paper that earned it.
        """

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            started = time.time()
            finished: list[tuple[str, float]] = []
            measured: list[tuple[str, float]] = []

            def evolve(**options):
                paper = Path(options["paper"]).stem
                out_dir = Path(options["out"])
                out_dir.mkdir(parents=True, exist_ok=True)
                best = out_dir / "round_00" / "candidate_00"
                best.mkdir(parents=True, exist_ok=True)
                write_json(out_dir / "evolution_state.json", {"best_candidate_dir": str(best), "best_score": 1.0})
                write_text(out_dir / "final_draft_preview.png", "png")
                if paper == "slow":
                    time.sleep(1.5)
                _file_candidate(out_dir, f"direct-exp-{paper}", options)
                finished.append((paper, time.time() - started))
                return {"ok": True, "out_dir": str(out_dir), "rounds_completed": 1}

            def validate(**options):
                measured.append((options["experience_id"], time.time() - started))
                return {"status": "same_domain_validated", "ladder": {}, "summary": {}, "all_pairs_isolated": True, "all_cases_measured": True}

            run_continuous_evolution(
                _config(
                    base_papers=[
                        {"paper": "/papers/fast.pdf", "paper_id": "fast", "domain": "biology"},
                        {"paper": "/papers/slow.pdf", "paper_id": "slow", "domain": "biology"},
                    ],
                    parallel=2,
                    validation={"suite": "/suites/four_groups.json"},
                ),
                out,
                experience_library=str(out / "library"),
                evolve_runner=evolve,
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=validate,
                clock=lambda: 0.0,
                max_waves=1,
            )
            # Both candidates are measured before the wave's own clock matters. A row that stops
            # at the same-domain rung also gets its one other-domain paper queued behind it, so
            # the count is "at least two rows measured" rather than "exactly two calls".
            self.assertTrue(_wait_for(lambda: len({name for name, _ in measured}) >= 2, timeout=30))

            by_paper = dict(finished)
            handed = dict(measured)
            self.assertEqual({"fast", "slow"}, set(by_paper))
            # The fast paper's candidate went to the validation worker before the slow paper had
            # even finished its iteration, so the wave no longer sets the pace of the measuring.
            self.assertLess(handed["direct-exp-fast"], by_paper["slow"])



class RetiredRowTests(unittest.TestCase):
    """A row the exam cannot hold is retired, not handed back for ever.

    Three dead ends end a candidate without a case being generated: no applicability written, a
    stage a pair cannot fork at, and a pool that cannot supply a required group. Each is written
    into the report as ``unverifiable`` and the row is moved out of ``candidate``, so the loop
    stops spending its budget re-queueing a measurement it will never be able to make.
    """

    def _library(self, out: Path, *, status: str = "candidate", rows: list[dict] | None = None) -> str:
        library = out / "library"
        library.mkdir(parents=True, exist_ok=True)
        body = rows if rows is not None else [{"domain": "direct_editable", "experience_id": EXPERIENCE, "status": status}]
        (library / "judge_experiences.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in body), encoding="utf-8")
        return str(library)

    def _state(self, out: Path, *, status: str, cases: int = 0, counters: dict | None = None) -> None:
        state = {
            "schema": "rfs.continuous_evolution_state",
            "version": 1,
            "papers": {},
            "waves": 3,
            "validations": {
                EXPERIENCE: {
                    "status": status,
                    "paper": "one",
                    "iteration": 1,
                    "out": "",
                    "summary": {"cases": cases, "measured_cases": cases},
                }
            },
        }
        if counters is not None:
            state["unmeasured_requeues"] = counters
        write_json(out / CONTINUOUS_STATE, state)

    def _run(self, out: Path, library: str, validate) -> dict:
        return run_continuous_evolution(
            _config(validation={"suite": "/suites/four_groups.json"}),
            out,
            experience_library=library,
            evolve_runner=_evolve_runner([], experience_id=""),
            review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
            validate_runner=validate,
            clock=lambda: 0.0,
            max_waves=1,
        )

    def test_a_row_retired_as_unverifiable_is_not_queued_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = self._library(out, status="unverifiable")
            self._state(out, status="unverifiable")
            seen: list[dict] = []

            self._run(out, library, lambda **options: seen.append(dict(options)) or {"status": "promoted"})

            self.assertEqual([], seen)
            self.assertEqual([], [item for item in _events(out) if item["event"] == "validation_queued"])

    def test_a_set_that_is_making_progress_is_not_charged_to_the_retry_budget(self):
        """A resumed set is continuing, not failing again, so the budget must not count it.

        The counter used to live in session state and was incremented on every restart, so a set
        that was too long to finish in one call ran out of tries for the only reason that it was
        long. With progress on disk the budget is at its limit here and the row is still handed
        over.
        """

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = self._library(out, status="candidate")
            self._state(out, status="running", cases=2, counters={EXPERIENCE: MAX_UNMEASURED_REQUEUES})
            seen: list[dict] = []

            self._run(out, library, lambda **options: seen.append(dict(options)) or {"status": "promoted", "summary": {"cases": 2, "measured_cases": 2}})

            self.assertTrue(_wait_for(lambda: seen))
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
            # The budget is untouched: the row advanced, so nothing is charged for this restart.
            self.assertEqual(MAX_UNMEASURED_REQUEUES, state["unmeasured_requeues"][EXPERIENCE])

    def test_a_row_with_no_progress_that_runs_out_of_tries_is_retired(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = self._library(out, status="candidate")
            self._state(out, status="running", cases=0, counters={EXPERIENCE: MAX_UNMEASURED_REQUEUES})
            seen: list[dict] = []

            self._run(out, library, lambda **options: seen.append(dict(options)) or {"status": "promoted"})

            self.assertEqual([], seen)
            events = _events(out)
            self.assertIn("validation_parked", [item["event"] for item in events])
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
            self.assertEqual("unverifiable", state["validations"][EXPERIENCE]["status"])
            self.assertEqual("requeue_budget_exhausted", state["validations"][EXPERIENCE]["blocked_because"])

    def test_the_measurement_section_reports_what_the_library_holds(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = self._library(
                out,
                rows=[
                    {"domain": "direct_editable", "experience_id": EXPERIENCE, "status": "candidate"},
                    {"domain": "direct_editable", "experience_id": "direct-exp-promoted", "status": "promoted"},
                    {"domain": "direct_editable", "experience_id": "direct-exp-refuted", "status": "refuted"},
                ],
            )
            self._state(out, status="candidate", cases=1)
            report = self._run(out, library, lambda **options: {"status": "promoted", "summary": {"cases": 3, "measured_cases": 3}})

            measurement = report["measurement"]

        self.assertEqual(3, measurement["library_rows"])
        self.assertEqual(1, measurement["candidates"])
        self.assertEqual(1, measurement["promoted"])
        self.assertEqual(1, measurement["retrievable"])
        self.assertEqual(1, measurement["building"])
        self.assertEqual({"candidate": 1, "promoted": 1, "refuted": 1}, measurement["by_status"])
        row = next(item for item in measurement["rows"] if item["experience_id"] == EXPERIENCE)
        self.assertGreaterEqual(row["cases_decided"], 1)
        self.assertEqual("", row["blocked_because"])


class PromptPatchValidationTests(unittest.TestCase):
    """A patch reaches another paper only through the same paired ON/OFF an experience does."""

    def test_a_patch_holding_its_own_case_rung_is_measured_on_another_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = out / "library"
            _file_patch(library, PATCH)
            seen: list[dict] = []

            def validate(**options):
                seen.append(dict(options))
                return {"status": "cross_domain_validated", "ladder": {}, "summary": {"cases": 2, "measured_cases": 2}}

            run_continuous_evolution(
                _config(validation={"suite": "/suites/four_groups.json"}),
                out,
                experience_library=str(library),
                evolve_runner=_evolve_runner_with_patch([], patch_id=PATCH),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=validate,
                clock=lambda: 0.0,
                max_waves=1,
            )

            self.assertTrue(_wait_for(lambda: seen))
            self.assertEqual([PATCH], [item["experience_id"] for item in seen])
            self.assertEqual(PATCH, seen[0]["prompt_patch_id"], "the paired run carries the patch, not an experience")
            self.assertTrue(_wait_for(lambda: any(item["event"] == "prompt_patch_cross_domain" for item in _events(out))))
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
            self.assertEqual("patch", state["validations"][PATCH]["kind"])
            self.assertEqual("cross_domain_validated", state["validations"][PATCH]["rung_awarded"])

    def test_a_patch_still_in_candidate_is_not_handed_to_another_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = out / "library"
            _file_patch(library, PATCH, status="candidate")
            seen: list[dict] = []

            run_continuous_evolution(
                _config(validation={"suite": "/suites/four_groups.json"}),
                out,
                experience_library=str(library),
                evolve_runner=_evolve_runner_with_patch([], patch_id=PATCH),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=lambda **options: seen.append(dict(options)) or {"status": "promoted"},
                clock=lambda: 0.0,
                max_waves=1,
            )

            self.assertEqual([], seen, "a patch with nothing measured behind it is not asked to travel")
            self.assertEqual([], [item for item in _events(out) if item["event"] == "validation_queued"])

    def test_a_patch_that_measures_worse_here_too_is_retired(self):
        """A measured loss on another paper spends the patch: it lost a round and then lost here."""

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = out / "library"
            _file_patch(library, PATCH)
            store = PromptPatchStore(library)
            store.append_event(PATCH, "refuted", round=1, reason="the round drawn under this patch did not come out better")

            run_continuous_evolution(
                _config(validation={"suite": "/suites/four_groups.json"}),
                out,
                experience_library=str(library),
                evolve_runner=_evolve_runner_with_patch([], patch_id=PATCH),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=lambda **options: {"status": "current_case_validated", "ladder": {"stopped_because": "same_domain_reproduction_not_shown"}, "summary": {"cases": 2, "measured_cases": 2}},
                clock=lambda: 0.0,
                max_waves=1,
            )

            self.assertTrue(_wait_for(lambda: any(item["event"] == "prompt_patch_refuted" for item in _events(out))))
            self.assertEqual(PROMPT_PATCH_REFUTATION_LIMIT, store.refutation_streak(PATCH))
            self.assertIn("disabled", [str(event.get("event") or "") for event in store.events(PATCH)])

    def test_a_set_that_could_not_be_measured_does_not_spend_the_patch(self):
        """A measurement nobody could take is not a measurement of failure.

        The ladder stops ``overfit_or_failed`` for a pair whose two arms were never shown to start
        from the same figure, and for a pair nobody drew at all. Charging those against a patch
        retired a sentence that had never actually been asked a question, so they are recorded as
        neutral: no refutation, no disable, and the rung the patch already held stands.
        """

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = out / "library"
            _file_patch(library, PATCH)
            store = PromptPatchStore(library)

            run_continuous_evolution(
                _config(validation={"suite": "/suites/four_groups.json"}),
                out,
                experience_library=str(library),
                evolve_runner=_evolve_runner_with_patch([], patch_id=PATCH),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=lambda **options: {"status": "overfit_or_failed", "ladder": {"stopped_because": "pair_not_measured"}, "summary": {"cases": 2, "measured_cases": 0}},
                clock=lambda: 0.0,
                max_waves=1,
            )

            self.assertTrue(_wait_for(lambda: any(item["event"] == "prompt_patch_not_promoted" for item in _events(out))))
            self.assertEqual(0, store.refutation_streak(PATCH))
            self.assertNotIn("disabled", [str(event.get("event") or "") for event in store.events(PATCH)])


if __name__ == "__main__":
    unittest.main()
