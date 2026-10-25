"""The continuous loop: fixed base papers, inherit the last draft, never stop.

The loop draws nothing, so it is driven here with a fake runner, a fake reviewer, a fake clock
and a fake disk probe. What is worth asserting is the accounting: which stage a review forks at,
what a restart remembers, which options the second iteration passes, and every way the loop is
allowed to stop.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rfs.evaluation.direct_judges import SEAT_NO_BACKEND, SEAT_UNREADABLE
from rfs.utils import write_json, write_text
from rfs.workflows import continuous_evolution as ce
from rfs.evolution.direct.human_preference import HUMAN_REVIEW_DIR, HumanReviewStore

from rfs.workflows.continuous_evolution import (
    CONTINUOUS_EVENTS,
    CONTINUOUS_MARKDOWN,
    CONTINUOUS_REPORT,
    CONTINUOUS_STATE,
    HUMAN_VERDICTS_FILE,
    DEFAULT_FORK_STAGE,
    DEFAULT_ITERATION_ROUNDS,
    DEFAULT_PARALLEL,
    INCREMENTAL_FORK_STAGE,
    CONTINUOUS_STATE_SCHEMA,
    DEFAULT_REOPEN_AFTER_STALLED_ROUNDS,
    MAX_CONSECUTIVE_FAILURES,
    READING_ATTRIBUTION_CONFIRMATIONS,
    REVIEW_CLEAN,
    REVIEW_ERROR,
    REVIEW_UNDEFINED,
    SILENT_REVIEW_FORK_STAGE,
    UPSTREAM_EVIDENCE_RETRY,
    deadlock_hint,
    fork_before_from_feedback,
    fork_from_review,
    is_upstream,
    iteration_defects,
    stage_index,
    upstream_allowed,
    work_face,
    load_continuous_state,
    opening_move_from_feedback,
    read_continuous_config,
    requested_action_types,
    requested_actions,
    run_continuous_evolution,
)


def _roomy_disk(_path) -> dict:
    """A volume with room to spare, whatever drive the tests happen to run on.

    These tests assert which guards the loop hands over, not whether the machine is full,
    so they must not read the real disk: a 40 GB floor on a small /tmp would stop the loop
    before its first wave and the test would fail for the machine's reasons.
    """

    return {"ratio": 0.9, "free_gb": 900.0}


def _paper(name: str) -> dict:
    return {"paper": f"/papers/{name}.pdf", "paper_id": name, "domain": "biology"}


def _config(**overrides) -> dict:
    block = {"base_papers": [_paper("paper_one"), _paper("paper_two")], "budget_hours": 24.0, "iteration_rounds": 3}
    block.update(overrides)
    return {"continuous": block}


def _seed_iteration(out: Path, name: str, number: int, *, score: float = 0.5) -> Path:
    """A finished iteration directory with the artifacts the loop reads back."""

    iter_dir = out / "papers" / name / f"iter_{number:02d}"
    best = iter_dir / "round_00" / "candidate_00"
    best.mkdir(parents=True, exist_ok=True)
    write_json(iter_dir / "evolution_state.json", {"best_candidate_dir": str(best), "best_score": score})
    write_text(iter_dir / "final_draft_preview.png", "png")
    write_json(iter_dir / "figure_program.json", {"panels": [], "semantic_nodes": [], "arrows": []})
    return iter_dir


class ForkStageTests(unittest.TestCase):
    def test_each_repair_forks_at_its_own_stage(self):
        for action, stage in (
            ("reanalyze_semantics", "global_planning"),
            ("resegment", "segmentation"),
            ("replan_theme", "theme_planning"),
            ("regen_assets", "icon_generation"),
            ("resize_nodes", "ppt_compilation"),
        ):
            with self.subTest(action=action):
                stage_name, reason = fork_before_from_feedback([{"requested_actions": [action]}])
                self.assertEqual(stage_name, stage)
                self.assertIn(stage, reason)

    def test_one_reading_attribution_does_not_re_read_the_paper(self):
        # No repair reaches the reading, which is why the attribution used to outrank the repair
        # the same call named. That ranking costs the whole figure, and it was being spent on one
        # verdict: the review both blamed the reading and asked for a repair below it. A single
        # attribution now leaves the fork to that repair.
        for stage in ("paper_reading", "content_planning"):
            with self.subTest(stage=stage):
                fork, reason = fork_from_review(
                    {
                        "ok": True,
                        "stage_feedback": [{"requested_actions": ["replan_structure"]}],
                        "judgement": {"fault_stage": stage},
                    },
                    attribution_streak=1,
                )
                self.assertEqual(fork, "global_planning")
                self.assertIn(stage, reason)
                self.assertNotIn("reads the paper again", reason)

    def test_a_reading_attribution_with_no_repair_still_keeps_the_reading(self):
        # It blamed the reading and asked for nothing, so there is no repair to run instead. What
        # it does not get is the paper: one verdict is not enough to throw the reading away, and
        # re-cutting the structure is the change that can help a figure nobody complained about.
        fork, reason = fork_from_review(
            {"ok": True, "stage_feedback": [], "judgement": {"fault_stage": "content_planning"}},
            attribution_streak=1,
        )
        self.assertEqual(fork, DEFAULT_FORK_STAGE)
        self.assertIn("content_planning", reason)
        self.assertNotIn("reads the paper again", reason)

    def test_two_reading_attributions_in_a_row_read_the_paper_again(self):
        for stage in ("paper_reading", "content_planning"):
            with self.subTest(stage=stage):
                fork, reason = fork_from_review(
                    {
                        "ok": True,
                        "stage_feedback": [{"requested_actions": ["replan_structure"]}],
                        "judgement": {"fault_stage": stage},
                    },
                    attribution_streak=READING_ATTRIBUTION_CONFIRMATIONS,
                )
                self.assertEqual(fork, stage)
                self.assertIn(stage, reason)
                self.assertIn("reads the paper again", reason)

    def test_an_attribution_to_a_later_stage_leaves_the_fork_to_the_repair(self):
        # The reading stages are the only exception: for every later stage the repair the review
        # named is the earlier and therefore the correct fork point.
        fork, _ = fork_from_review(
            {
                "ok": True,
                "stage_feedback": [{"requested_actions": ["replan_theme"]}],
                "judgement": {"fault_stage": "ppt_compilation"},
            }
        )
        self.assertEqual(fork, "theme_planning")

    def test_the_earliest_stage_named_wins(self):
        feedback = [{"requested_actions": ["reroute"]}, {"requested_actions": ["replan_theme"]}]
        self.assertEqual(fork_before_from_feedback(feedback)[0], "theme_planning")

    def test_a_repair_the_alias_does_not_recognise_is_not_a_stage(self):
        # A name nothing recognises asks for nothing this loop can act on, so it must not drag
        # the fork stage to whatever stage_for_action falls back to.
        self.assertEqual(requested_action_types([{"requested_actions": ["reroute_arrows"]}]), [])
        self.assertEqual(fork_before_from_feedback([{"requested_actions": ["reroute_arrows"]}])[0], DEFAULT_FORK_STAGE)

    def test_a_review_that_asks_for_nothing_recuts_structure_and_layout(self):
        self.assertEqual(fork_before_from_feedback([])[0], DEFAULT_FORK_STAGE)
        self.assertEqual(fork_before_from_feedback(None)[0], DEFAULT_FORK_STAGE)

    def test_actions_are_read_from_both_vocabularies_and_deduped(self):
        feedback = [{"requested_actions": ["reroute"]}, {"repair": [{"type": "reroute"}, "resize_nodes"]}]
        self.assertEqual(requested_action_types(feedback), ["reroute", "resize_nodes"])


class SilentReviewTests(unittest.TestCase):
    """A review that did not answer is not a review that asked for nothing.

    Both used to arrive here as an empty list of repairs, and both were answered the same
    way: the structure and the layout were planned again. For a review that failed, that was
    an hour of drawing spent on a crash -- the default fork stage, reached 107 times.
    """

    def test_a_review_that_failed_keeps_the_figure_instead_of_recutting_the_structure(self):
        stage, reason = fork_from_review(
            {"ok": False, "reason": "review_failed: ValueError: could not convert string to float: 'critical'", "stage_feedback": []}
        )
        self.assertEqual(stage, SILENT_REVIEW_FORK_STAGE)
        self.assertNotEqual(stage, DEFAULT_FORK_STAGE)
        self.assertIn("did not answer", reason)
        self.assertNotIn("asked for no repair", reason)

    def test_every_way_a_review_can_fail_to_answer_reads_the_same_way(self):
        for reason in ("no_advisory_judge", "no_kept_figure", "review_failed: TimeoutError: slow"):
            stage, sentence = fork_from_review({"ok": False, "reason": reason, "stage_feedback": []})
            self.assertEqual(stage, SILENT_REVIEW_FORK_STAGE, reason)
            self.assertIn(reason, sentence)

    def test_a_review_that_ran_and_asked_for_nothing_still_recuts_the_structure(self):
        stage, reason = fork_from_review({"ok": True, "reason": "", "stage_feedback": []})
        self.assertEqual(stage, DEFAULT_FORK_STAGE)
        self.assertIn("asked for no repair", reason)

    def test_a_named_repair_is_still_what_decides_the_fork(self):
        stage, _reason = fork_from_review({"ok": True, "reason": "", "stage_feedback": [{"requested_actions": ["replan_theme"]}]})
        self.assertEqual(stage, "theme_planning")


class BoundaryReviewRunnerTests(unittest.TestCase):
    """The three silences of the boundary review are three different things.

    A seat with no weights, an answer that could not be parsed and a call that raised all used to
    arrive at the fork as an empty list of repairs, which read as "the figure is fine" and was
    answered by re-cutting the structure -- an hour of drawing spent on a crash. The seat that
    read the figure and found nothing to repair is the one case where that is a real answer, and
    even it does not get the structure re-planned.
    """

    class _Seat:
        """A stand-in advisory seat that replays a scripted list of answers."""

        model = "fake-seat"

        def __init__(self, answers):
            self.answers = list(answers)
            self.calls = []

        def evaluate(self, _judge_input, *, extra_instruction=""):
            self.calls.append(extra_instruction)
            answer = self.answers.pop(0) if self.answers else {"status": "pass", "stage_feedback": []}
            if isinstance(answer, Exception):
                raise answer
            return answer

    def _review(self, answers):
        seat = self._Seat(answers)
        with mock.patch.object(ce, "build_direct_judges", return_value=(seat, None)), mock.patch.object(
            ce, "kept_figure_judge_input", return_value={"domain": "biology"}
        ):
            runner = ce._default_review_runner({"evolve": {"judge_mode": "vlm"}})
            result = runner(Path("."), {"name": "paper_one", "domain": "biology"})
        return seat, result

    @staticmethod
    def _silence(kind):
        return {
            "stage_feedback": [],
            "blocking_issues": ["nothing readable"],
            "advisory_unavailable": "nothing readable",
            "advisory_unavailable_kind": kind,
            "status": "fault",
        }

    @staticmethod
    def _localised():
        return {"status": "fault", "fault_stage": "segmentation", "fault_requirement": "r", "fault_evidence": [{"field": "overlap_count", "value": 3}]}

    def test_a_seat_with_no_weights_is_not_asked_twice(self):
        seat, result = self._review([self._silence(SEAT_NO_BACKEND)])
        self.assertEqual(len(seat.calls), 1)
        self.assertEqual(result["exit"], REVIEW_ERROR)
        self.assertFalse(result["ok"])
        self.assertFalse(result["retried"])

    def test_an_answer_that_could_not_be_read_is_asked_for_once_more(self):
        seat, result = self._review([self._silence(SEAT_UNREADABLE), self._silence(SEAT_UNREADABLE)])
        self.assertEqual(len(seat.calls), 2)
        self.assertEqual(result["exit"], REVIEW_ERROR)
        self.assertTrue(result["retried"])
        self.assertEqual(result["retries"], 1)

    def test_a_call_that_raised_is_retried_once_and_still_reports_the_error(self):
        seat, result = self._review([RuntimeError("device fell over"), RuntimeError("device fell over")])
        self.assertEqual(len(seat.calls), 2)
        self.assertEqual(result["exit"], REVIEW_ERROR)
        self.assertIn("RuntimeError", result["reason"])

    def test_an_answer_that_named_a_defect_without_a_stage_is_asked_again(self):
        second = dict(self._localised())
        second["stage_feedback"] = [{"problem": "overlap", "requested_actions": ["resegment"]}]
        seat, result = self._review([{"status": "fault", "stage_feedback": [{"problem": "overlap"}], "blocking_issues": ["overlap"]}, second])
        self.assertEqual(len(seat.calls), 2)
        self.assertTrue(seat.calls[1])
        self.assertEqual(result["exit"], "")
        self.assertTrue(result["ok"])
        self.assertTrue(result["localisation_retry"])
        stage, _reason = fork_from_review(result)
        self.assertEqual(stage, "segmentation")

    def test_two_answers_without_a_stage_are_recorded_as_undefined(self):
        _, result = self._review([{"status": "fault", "blocking_issues": ["overlap"]}, {"status": "fault", "blocking_issues": ["overlap"]}])
        self.assertEqual(result["exit"], REVIEW_UNDEFINED)
        self.assertFalse(result["ok"])

    def test_a_figure_the_seat_found_nothing_wrong_with_is_not_a_error(self):
        seat, result = self._review([{"status": "pass", "stage_feedback": []}])
        self.assertEqual(len(seat.calls), 1)
        self.assertEqual(result["exit"], REVIEW_CLEAN)
        self.assertTrue(result["ok"])
        self.assertFalse(result["retried"])

    def test_a_localised_answer_is_not_asked_again(self):
        seat, result = self._review([self._localised()])
        self.assertEqual(len(seat.calls), 1)
        self.assertEqual(result["exit"], "")
        self.assertTrue(result["ok"])


class BoundaryReviewForkTests(unittest.TestCase):
    """What the loop does with each of the three silences."""

    def _forks(self, exit_reason, **extra):
        record = {"ok": exit_reason in {"", REVIEW_CLEAN}, "reason": "the seat said nothing", "exit": exit_reason, "stage_feedback": []}
        record.update(extra)
        return fork_from_review(record)

    def test_no_silence_sends_the_round_back_to_the_structure(self):
        for exit_reason in (REVIEW_ERROR, REVIEW_UNDEFINED, REVIEW_CLEAN):
            with self.subTest(exit_reason=exit_reason):
                stage, reason = self._forks(exit_reason)
                self.assertEqual(stage, SILENT_REVIEW_FORK_STAGE)
                self.assertNotEqual(stage, DEFAULT_FORK_STAGE)
                self.assertNotIn("asked for no repair", reason)

    def test_the_exit_reason_is_readable_afterwards(self):
        _stage, reason = self._forks(REVIEW_UNDEFINED)
        self.assertIn(REVIEW_UNDEFINED, reason)

    def test_a_localised_review_still_reads_its_repair(self):
        stage, _reason = fork_from_review({"ok": True, "reason": "", "exit": "", "stage_feedback": [{"requested_actions": ["replan_theme"]}]})
        self.assertEqual(stage, "theme_planning")


class BoundaryReviewFileTests(unittest.TestCase):
    """The round writes down what the review was asked, answered and cost."""

    def test_the_round_leaves_the_review_beside_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []
            runner = LoopTests()._runner(calls)

            def review(_previous: Path, _spec: dict):
                return {
                    "ok": False,
                    "reason": "the advisory answer could not be read: boom",
                    "exit": REVIEW_ERROR,
                    "retried": True,
                    "retries": 1,
                    "stage_feedback": [],
                    "judgement": {"status": "fault"},
                    "model": "fake-seat",
                }

            run_continuous_evolution(_config(), out, evolve_runner=runner, review_runner=review, clock=lambda: 0.0, max_waves=2)

            payload = json.loads((out / "papers" / "paper_one" / "iter_02" / "boundary_review.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["exit_reason"], REVIEW_ERROR)
            self.assertEqual(payload["model"], "fake-seat")
            self.assertTrue(payload["retried"])
            self.assertEqual(payload["retries"], 1)
            self.assertEqual(payload["answer"], {"status": "fault"})
            events = (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8")
            self.assertIn(REVIEW_ERROR, events)


def _layers(*settled: str) -> dict:
    """A layer ledger with the named stages settled and no reading behind them."""

    return {stage: {"settled_at": 1, "settled_by": "model"} for stage in settled}


class WorkFaceTests(unittest.TestCase):
    """The layer a round may work on is the earliest one the model has not called finished."""

    def test_the_open_layer_is_the_earliest_unsettled_one(self):
        self.assertEqual("paper_reading", work_face({}))
        self.assertEqual("content_planning", work_face(_layers("paper_reading")))
        self.assertEqual("theme_planning", work_face(_layers("paper_reading", "content_planning", "global_planning", "segmentation")))

    def test_a_settle_with_no_stamp_does_not_count(self):
        self.assertEqual("paper_reading", work_face({"paper_reading": {"settled_at": 0}}))

    def test_every_layer_settled_leaves_the_last_one_open(self):
        self.assertEqual("repair", work_face(_layers(*("paper_reading", "content_planning", "global_planning", "segmentation", "theme_planning", "icon_generation", "ppt_compilation", "repair"))))

    def test_a_name_that_is_not_a_stage_is_upstream_of_everything(self):
        self.assertEqual(-1, stage_index("not_a_stage"))
        self.assertTrue(is_upstream("not_a_stage", "paper_reading"))
        self.assertFalse(is_upstream("ppt_compilation", "paper_reading"))
        self.assertTrue(is_upstream("global_planning", "theme_planning"))

    def test_a_reading_verdict_the_loop_confirmed_may_go_back(self):
        self.assertTrue(upstream_allowed(fork="paper_reading", work="icon_generation", attributed="paper_reading", streak=2, review={}))
        self.assertFalse(upstream_allowed(fork="paper_reading", work="icon_generation", attributed="paper_reading", streak=1, review={}))
        self.assertFalse(upstream_allowed(fork="global_planning", work="icon_generation", attributed="", streak=0, review={}))

    def test_working_below_the_work_face_is_never_an_upstream_jump(self):
        self.assertTrue(upstream_allowed(fork="ppt_compilation", work="theme_planning", attributed="", streak=0, review={}))

    def test_the_stall_hint_only_appears_after_the_limit(self):
        self.assertEqual("", deadlock_hint(3, DEFAULT_REOPEN_AFTER_STALLED_ROUNDS, "theme_planning"))
        self.assertEqual("", deadlock_hint(0, DEFAULT_REOPEN_AFTER_STALLED_ROUNDS, "theme_planning"))
        hint = deadlock_hint(4, DEFAULT_REOPEN_AFTER_STALLED_ROUNDS, "theme_planning")
        self.assertIn("theme_planning", hint)
        self.assertIn("4 rounds", hint)


class WorkFaceLoopTests(unittest.TestCase):
    """A round is held on the layer it is working on, and the model is asked before it is."""

    def _state(self, out: Path, paper: str, entry: dict) -> None:
        write_json(out / CONTINUOUS_STATE, {"schema": CONTINUOUS_STATE_SCHEMA, "version": 1, "papers": {paper: entry}, "waves": 1})

    def _accepted(self):
        """A stand-in run whose single repair stuck, so the iteration moved the figure forward."""

        def runner(**options):
            out = Path(options["out"])
            number = int(out.name.split("_")[-1])
            iter_dir = _seed_iteration(out.parents[2], out.parent.name, number)
            write_json(iter_dir / "evolution_rounds.json", {"rounds": [{"round": 1, "accepted": True, "repair_decision": {"reason": "repair"}}]})
            return {"ok": True, "out_dir": str(iter_dir), "rounds_completed": number}

        return runner

    def _run(self, out: Path, review, *, instructions: list | None = None, runner=None):
        calls: list[dict] = []
        seen = instructions if instructions is not None else []

        def wrapped(previous: Path, spec: dict):
            seen.append((str(spec.get("name") or ""), str(spec.get("review_instruction") or "")))
            return review(previous, spec)

        report = run_continuous_evolution(
            _config(),
            out,
            evolve_runner=runner or LoopTests()._runner(calls),
            review_runner=wrapped,
            clock=lambda: 0.0,
            max_waves=2,
        )
        return report, calls, seen

    def _second_iteration(self, calls: list[dict], paper: str) -> list[dict]:
        # Compared by path parts rather than by a string a separator was written into: a hand-written
        # separator is what makes a test pass on the machine it was written on and nowhere else.
        want = ("papers", paper, "iter_02")
        return [item for item in calls if Path(item["out"]).parts[-3:] == want]

    def _events(self, out: Path) -> list[dict]:
        return [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines() if line.strip()]

    def _state_of(self, out: Path, paper: str) -> dict:
        return json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))["papers"][paper]

    def test_a_jump_over_a_finished_layer_is_held_where_the_round_is_working(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._state(out, "paper_one", {"layers": _layers("paper_reading", "content_planning", "global_planning", "segmentation")})
            seen: list[tuple] = []
            _report, calls, _seen = self._run(
                out,
                lambda previous, spec: {"ok": True, "reason": "", "stage_feedback": [{"requested_actions": ["reanalyze_semantics"]}]},
                instructions=seen,
            )

            self.assertEqual(["theme_planning"], [item["fork_before"] for item in self._second_iteration(calls, "paper_one")])
            capped = [item for item in self._events(out) if item["event"] == "fork_capped"]
            self.assertEqual(1, len(capped))
            self.assertEqual("global_planning", capped[0]["requested"])
            self.assertEqual("theme_planning", capped[0]["held_at"])
            # The seat was asked once more before the fork was held, and the rule went with the ask.
            self.assertEqual(1, sum(1 for _paper, text in seen if UPSTREAM_EVIDENCE_RETRY in text))
            payload = json.loads((out / "papers" / "paper_one" / "iter_02" / "boundary_review.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["capped"])
            self.assertIn("reask", payload)

    def test_a_case_for_the_layer_it_names_reopens_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._state(out, "paper_one", {"layers": _layers("paper_reading", "content_planning", "global_planning", "segmentation")})
            review = {
                "ok": True,
                "reason": "",
                "stage_feedback": [{"requested_actions": ["resegment"]}],
                "judgement": {
                    "fault_stage": "segmentation",
                    "fault_requirement": "the panel titles",
                    "fault_evidence": [{"field": "panel_occupancy", "value": 0.2}],
                    "advisory_shown_fields": ["panel_occupancy"],
                },
            }
            _report, calls, _seen = self._run(out, lambda previous, spec: review)

            self.assertEqual(["segmentation"], [item["fork_before"] for item in self._second_iteration(calls, "paper_one")])
            self.assertEqual([], [item for item in self._events(out) if item["event"] == "fork_capped"])
            reopened = [item for item in self._events(out) if item["event"] == "layer_reopened"]
            self.assertEqual(["segmentation"], [item["stage"] for item in reopened])
            payload = json.loads((out / "papers" / "paper_one" / "iter_02" / "boundary_review.json").read_text(encoding="utf-8"))
            self.assertEqual("segmentation", payload["upstream_case"]["fault_stage"])

    def test_a_case_citing_nothing_the_seat_was_shown_is_held_at_the_open_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._state(out, "paper_one", {"layers": _layers("paper_reading", "content_planning", "global_planning", "segmentation")})
            review = {
                "ok": True,
                "reason": "",
                "stage_feedback": [{"requested_actions": ["resegment"]}],
                "judgement": {
                    "fault_stage": "segmentation",
                    "fault_requirement": "the panel titles",
                    "fault_evidence": [{"field": "overlap_count", "value": 3}],
                    "advisory_shown_fields": ["panel_occupancy"],
                },
            }
            _report, calls, seen = self._run(out, lambda previous, spec: review, instructions=[])

            self.assertEqual(["theme_planning"], [item["fork_before"] for item in self._second_iteration(calls, "paper_one")])
            capped = [item for item in self._events(out) if item["event"] == "fork_capped"]
            self.assertEqual(1, len(capped))
            self.assertEqual("fault_evidence_unreadable", capped[0]["upstream_case_rejected"])
            payload = json.loads((out / "papers" / "paper_one" / "iter_02" / "boundary_review.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["capped"])
            self.assertEqual("fault_evidence_unreadable", payload["upstream_case_rejected"])
            self.assertEqual(1, sum(1 for _paper, text in seen if UPSTREAM_EVIDENCE_RETRY in text))


    def test_working_on_the_open_layer_is_not_an_upstream_jump(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._state(out, "paper_one", {"layers": _layers("paper_reading", "content_planning", "global_planning", "segmentation")})
            seen: list[tuple] = []
            _report, calls, _seen = self._run(
                out,
                lambda previous, spec: {"ok": True, "reason": "", "stage_feedback": [{"requested_actions": ["replan_theme"]}]},
                instructions=seen,
            )
            self.assertEqual(["theme_planning"], [item["fork_before"] for item in self._second_iteration(calls, "paper_one")])
            self.assertEqual([], [item for item in self._events(out) if item["event"] == "fork_capped"])
            self.assertEqual([], [text for _paper, text in seen if UPSTREAM_EVIDENCE_RETRY in text])

    def test_a_confirmed_reading_verdict_reopens_the_layer_it_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._state(
                out,
                "paper_one",
                {
                    "layers": _layers("paper_reading", "content_planning", "global_planning", "segmentation", "theme_planning"),
                    "last_reading_attribution": {"iteration": 1, "stage": "paper_reading", "streak": 1},
                },
            )
            _report, calls, _seen = self._run(
                out,
                lambda previous, spec: {
                    "ok": True,
                    "reason": "",
                    "stage_feedback": [{"requested_actions": ["resegment"]}],
                    "judgement": {"fault_stage": "paper_reading"},
                },
            )
            self.assertEqual(["paper_reading"], [item["fork_before"] for item in self._second_iteration(calls, "paper_one")])
            reopened = [item for item in self._events(out) if item["event"] == "layer_reopened"]
            self.assertEqual(1, len(reopened))
            self.assertEqual("paper_reading", reopened[0]["stage"])
            self.assertEqual(1, reopened[0]["reopen_count"])
            self.assertEqual([], [item for item in self._events(out) if item["event"] == "fork_capped"])

    def test_a_round_stuck_too_long_tells_the_review_what_it_watched(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._state(
                out,
                "paper_one",
                {"layers": _layers("paper_reading", "content_planning"), "stalled_streak": DEFAULT_REOPEN_AFTER_STALLED_ROUNDS},
            )
            _report, _calls, seen = self._run(out, lambda previous, spec: {"ok": True, "reason": "", "stage_feedback": []})
            hints = [text for paper, text in seen if paper == "paper_one" and text]
            self.assertEqual(1, len(hints))
            self.assertIn("has not moved forward", hints[0])
            started = [item for item in self._events(out) if item["event"] == "iteration_started" and item["iteration"] == 2]
            self.assertEqual([True], [item["stalled_hint"] for item in started if item["paper"] == "paper_one"])

    def test_a_round_that_moves_the_figure_forward_clears_the_stall(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._state(out, "paper_one", {"stalled_streak": 3})
            self._run(out, lambda previous, spec: {"ok": True, "reason": "", "stage_feedback": []}, runner=self._accepted())
            self.assertEqual(0, self._state_of(out, "paper_one")["stalled_streak"])

    def test_a_round_that_keeps_introducing_the_same_defect_also_counts_as_stalled(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            iter_dir = out / "papers" / "paper_one" / "iter_01"
            iter_dir.mkdir(parents=True)
            write_json(
                iter_dir / "evolution_rounds.json",
                {"rounds": [{"round": 1, "accepted": False, "repair_decision": {"new_blocking_defects": ["overlap", "crossing"]}}]},
            )
            self.assertEqual(["crossing", "overlap"], iteration_defects(iter_dir))


class LayerLedgerTests(unittest.TestCase):
    """A layer is settled by an answer, and the ledger only records what happened."""

    def _gates(self, iter_dir: Path) -> None:
        write_json(
            iter_dir / "round_00" / "stage_gate.json",
            {
                "round": 0,
                "stages": [
                    {"stage": "paper_reading", "changed": True, "reasons": ["the plan was read"], "reading": {"declared_nodes": 7}},
                    {"stage": "content_planning", "changed": False, "reasons": [], "reading": {}},
                ],
                "figure_metrics": {"after": {"node_count": 5}, "delta": {"node_count": 1}},
            },
        )

    def test_the_moves_come_from_the_round_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp)
            self._gates(iter_dir)
            moves = ce.stage_moves(iter_dir)
            self.assertTrue(moves["paper_reading"]["moved"])
            self.assertEqual([0], moves["paper_reading"]["rounds"])
            self.assertEqual({"declared_nodes": 7}, moves["paper_reading"]["reading"])
            self.assertFalse(moves["content_planning"]["moved"])

    def test_only_a_reflections_own_answer_settles_a_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp)
            write_json(
                iter_dir / "evolution_rounds.json",
                {
                    "rounds": [
                        {"round": 0, "reflection": {"layer_settled": {"stage": "paper_reading", "because": "first"}}},
                        {"round": 1, "reflection": {"layer_settled": {"stage": "paper_reading", "because": "again"}}},
                        {"round": 2, "reflection": {"layer_settled": {"stage": "nonsense", "because": "x"}}},
                        {"round": 3, "reflection": {"outcome": "improved"}},
                    ]
                },
            )
            self.assertEqual({"paper_reading": "again"}, ce.iteration_layer_settled(iter_dir))

    def test_the_ledger_reads_a_settled_layer_as_settled_and_an_idle_one_as_idle(self):
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp)
            self._gates(iter_dir)
            payload = ce.layer_ledger(iter_dir, layers=_layers("paper_reading"), iteration=3)
            self.assertEqual("rfs.layer_ledger", payload["contract"])
            self.assertEqual(3, payload["iteration"])
            self.assertEqual("content_planning", payload["work_face"])
            self.assertEqual(["paper_reading"], payload["settled"])
            rows = {item["stage"]: item for item in payload["layers"]}
            self.assertEqual(list(ce.STAGE_ORDER), [item["stage"] for item in payload["layers"]])
            self.assertTrue(rows["paper_reading"]["settled"])
            self.assertTrue(rows["paper_reading"]["moved"])
            self.assertEqual({"declared_nodes": 7}, rows["paper_reading"]["reading"])
            self.assertFalse(rows["content_planning"]["settled"])
            self.assertEqual({"node_count": 5}, payload["figure_metrics"]["after"])

    def test_the_first_answer_stands_until_the_layer_is_opened_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            loop = ce._Loop(
                config=read_continuous_config(_config()),
                out=out,
                library="",
                stop_file="",
                evolve_runner=lambda **options: {},
                review_runner=lambda previous, spec: {"ok": False},
                clock=lambda: 0.0,
                disk_usage=_roomy_disk,
                max_waves=1,
            )
            entry: dict = {"layers": {}}
            self.assertTrue(loop.settle_layer(entry, "paper_reading", paper="p", number=2, because="the plan is complete"))
            self.assertFalse(loop.settle_layer(entry, "paper_reading", paper="p", number=3, because="a second opinion"))
            self.assertEqual(2, entry["layers"]["paper_reading"]["settled_at"])
            self.assertEqual("content_planning", entry["work_face"])
            self.assertTrue(loop.reopen_layer(entry, "paper_reading", paper="p", number=4, why="a person says so"))
            self.assertEqual("paper_reading", entry["work_face"])
            self.assertEqual(1, entry["layers"]["paper_reading"]["reopen_count"])
            self.assertTrue(loop.settle_layer(entry, "paper_reading", paper="p", number=5, because="settled again", by="human"))
            self.assertEqual("human", entry["layers"]["paper_reading"]["settled_by"])


class LayerSettleLoopTests(unittest.TestCase):
    """The loop writes down the layer the model finished and moves the work face on by itself."""

    def _runner(self):
        def runner(**options):
            target = Path(options["out"])
            number = int(target.name.split("_")[-1])
            iter_dir = _seed_iteration(target.parents[2], target.parent.name, number)
            write_json(
                iter_dir / "evolution_rounds.json",
                {
                    "rounds": [
                        {
                            "round": 1,
                            "accepted": True,
                            "repair_decision": {"reason": "repair"},
                            "reflection": {"layer_settled": {"stage": "paper_reading", "because": "the plan lists every module"}},
                        }
                    ]
                },
            )
            write_json(
                iter_dir / "round_00" / "stage_gate.json",
                {
                    "round": 0,
                    "stages": [
                        {"stage": "paper_reading", "changed": True, "reasons": ["the plan was read"], "reading": {"declared_nodes": 7}},
                    ],
                    "figure_metrics": {"after": {"node_count": 5}, "delta": {"node_count": 1}},
                },
            )
            return {"ok": True, "out_dir": str(iter_dir), "rounds_completed": number}

        return runner

    def _run(self, out: Path, runner=None):
        return run_continuous_evolution(
            _config(),
            out,
            evolve_runner=runner or self._runner(),
            review_runner=lambda previous, spec: {"ok": True, "reason": "", "stage_feedback": []},
            clock=lambda: 0.0,
            max_waves=1,
        )

    def _events(self, out: Path) -> list[dict]:
        return [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_a_reflection_that_settles_a_layer_moves_the_work_face(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._run(out)
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))["papers"]["paper_one"]
            self.assertEqual(1, state["layers"]["paper_reading"]["settled_at"])
            self.assertEqual("model", state["layers"]["paper_reading"]["settled_by"])
            self.assertEqual("the plan lists every module", state["layers"]["paper_reading"]["because"])
            self.assertEqual("content_planning", state["work_face"])
            settled = [item for item in self._events(out) if item["event"] == "layer_settled" and item["paper"] == "paper_one"]
            self.assertEqual(["paper_reading"], [item["stage"] for item in settled])
            ledger = json.loads((out / "papers" / "paper_one" / "iter_01" / "layer_ledger.json").read_text(encoding="utf-8"))
            self.assertEqual(["paper_reading"], ledger["settled"])
            self.assertEqual("content_planning", ledger["work_face"])
            row = {item["stage"]: item for item in ledger["layers"]}["paper_reading"]
            self.assertEqual(1, row["last_move_at"])
            self.assertEqual(1, row["settled_at"])

    def test_an_ordinary_reflection_settles_no_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._run(out, runner=LoopTests()._runner([]))
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))["papers"]["paper_one"]
            self.assertEqual({}, {stage: item for stage, item in state["layers"].items() if item.get("settled_at")})
            self.assertEqual("paper_reading", state["work_face"])
            self.assertEqual([], [item for item in self._events(out) if item["event"] == "layer_settled"])


class HumanVerdictLoopTests(unittest.TestCase):
    """A person's own verdict is read once, applied, and never read a second time."""

    def _runner(self, calls: list, *, accepted: bool = False):
        def runner(**options):
            calls.append(dict(options))
            out = Path(options["out"])
            number = int(out.name.split("_")[-1])
            iter_dir = _seed_iteration(out.parents[2], out.parent.name, number)
            write_json(
                iter_dir / "evolution_rounds.json",
                {"rounds": [{"round": 1, "accepted": accepted, "repair_decision": {"reason": "repair"}}]},
            )
            return {"ok": True, "out_dir": str(iter_dir), "rounds_completed": number}

        return runner

    def _file(self, out: Path, payload: dict) -> dict:
        return HumanReviewStore(out / HUMAN_REVIEW_DIR).record(payload, defect_menu=[])

    def _events(self, out: Path) -> list[dict]:
        return [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines() if line.strip()]

    def _run(self, out: Path, calls: list, *, waves: int, accepted: bool = False):
        return run_continuous_evolution(
            _config(base_papers=[_paper("paper_one")]),
            out,
            evolve_runner=self._runner(calls, accepted=accepted),
            review_runner=lambda previous, spec: {"ok": True, "reason": "", "stage_feedback": []},
            clock=lambda: 0.0,
            max_waves=waves,
        )

    def test_a_person_saying_a_layer_is_finished_settles_it_and_moves_the_work_face(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []
            filed = self._file(
                out,
                {"kind": "layer_settled", "paper": "paper_one", "stage": "paper_reading", "human_note": "every module is on the figure", "reviewer": "<AUTHOR>"},
            )
            self.assertTrue(filed["ok"], filed)
            self._run(out, calls, waves=1)
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))["papers"]["paper_one"]
            self.assertEqual("human", state["layers"]["paper_reading"]["settled_by"])
            self.assertEqual("every module is on the figure", state["layers"]["paper_reading"]["because"])
            self.assertEqual("content_planning", state["work_face"])
            self.assertEqual([filed["review"]["review_id"]], state["human_verdicts_seen"])
            applied = [item for item in self._events(out) if item["event"] == "layer_settled"]
            self.assertEqual(["human"], [item["by"] for item in applied])
            consumed = [item for item in self._events(out) if item["event"] == "human_verdict_consumed"]
            self.assertEqual(["layer_settled"], [item["applied"] for item in consumed])
            receipt = json.loads((out / "papers" / "paper_one" / "iter_01" / HUMAN_VERDICTS_FILE).read_text(encoding="utf-8"))
            self.assertEqual(["layer_settled"], [item["applied"] for item in receipt["verdicts"]])

    def test_a_person_sealing_a_layer_holds_against_the_model_sealing_it_in_the_same_iteration(self):
        """A layer is sealed once, and the person's answer is the one that says who sealed it.

        A layer a person sealed is theirs to open; a layer the model sealed is one the model may
        open again on its own word. Reading the model's reflection first therefore threw the
        person's seal away and handed the layer back to the model.
        """

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []

            def runner(**options):
                calls.append(dict(options))
                target = Path(options["out"])
                number = int(target.name.split("_")[-1])
                iter_dir = _seed_iteration(target.parents[2], target.parent.name, number)
                write_json(
                    iter_dir / "evolution_rounds.json",
                    {
                        "rounds": [
                            {
                                "round": 1,
                                "accepted": False,
                                "repair_decision": {"reason": "repair"},
                                "reflection": {"layer_settled": {"stage": "paper_reading", "because": "the seat's own word"}},
                            }
                        ]
                    },
                )
                return {"ok": True, "out_dir": str(iter_dir), "rounds_completed": number}

            self._file(
                out,
                {"kind": "layer_settled", "paper": "paper_one", "stage": "paper_reading", "human_note": "every module is on the figure", "reviewer": "<AUTHOR>"},
            )
            run_continuous_evolution(
                _config(base_papers=[_paper("paper_one")]),
                out,
                evolve_runner=runner,
                review_runner=lambda previous, spec: {"ok": True, "reason": "", "stage_feedback": []},
                clock=lambda: 0.0,
                max_waves=1,
            )
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))["papers"]["paper_one"]
            self.assertEqual("human", state["layers"]["paper_reading"]["settled_by"])
            self.assertEqual("every module is on the figure", state["layers"]["paper_reading"]["because"])
            sealed = [item for item in self._events(out) if item["event"] == "layer_settled"]
            self.assertEqual(["human"], [item["by"] for item in sealed])

    def test_a_verdict_is_read_once_however_many_rounds_follow_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []
            self._file(
                out,
                {"kind": "layer_settled", "paper": "paper_one", "stage": "paper_reading", "human_note": "done", "reviewer": "me"},
            )
            self._run(out, calls, waves=3)
            settled = [item for item in self._events(out) if item["event"] == "layer_settled"]
            consumed = [item for item in self._events(out) if item["event"] == "human_verdict_consumed"]
            self.assertEqual(1, len(settled))
            self.assertEqual(1, len(consumed))

    def test_a_preference_reaches_the_round_and_stands_once_a_round_kept_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []
            filed = self._file(
                out,
                {
                    "kind": "preference",
                    "paper": "paper_one",
                    "apply_at": "theme_planning",
                    "directive": "Keep the palette calm and let the panels carry the contrast.",
                    "condition": {"defect_patterns": ["clashing_palette"]},
                    "reviewer": "<AUTHOR>",
                    "human_note": "the colours fight each other",
                },
            )
            self.assertTrue(filed["ok"], filed)
            self._run(out, calls, waves=1, accepted=True)
            handed = [item for item in calls if item.get("human_preferences")]
            self.assertEqual(1, len(handed))
            entry = handed[0]["human_preferences"][0]
            self.assertEqual("candidate", entry["status"])
            self.assertEqual("theme_planning", entry["apply_at"])
            self.assertEqual("Keep the palette calm and let the panels carry the contrast.", entry["directive"])
            # The round kept the change the sentence asked for, so the trial now stands.
            rows = json.loads((out / HUMAN_REVIEW_DIR / "human_preference_entries.json").read_text(encoding="utf-8"))["entries"]
            self.assertEqual(["resident"], [row["status"] for row in rows])
            promoted = [item for item in self._events(out) if item["event"] == "human_preference_promoted"]
            self.assertEqual(1, len(promoted))

    def test_a_round_that_kept_nothing_leaves_the_preference_a_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []
            self._file(
                out,
                {"kind": "preference", "paper": "paper_one", "apply_at": "theme_planning", "directive": "Keep it calm.", "reviewer": "me"},
            )
            self._run(out, calls, waves=1, accepted=False)
            rows = json.loads((out / HUMAN_REVIEW_DIR / "human_preference_entries.json").read_text(encoding="utf-8"))["entries"]
            self.assertEqual(["candidate"], [row["status"] for row in rows])
            self.assertEqual([], [item for item in self._events(out) if item["event"] == "human_preference_promoted"])

    def test_a_verdict_about_another_paper_is_not_read_here(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []
            self._file(
                out,
                {"kind": "layer_settled", "paper": "paper_two", "stage": "paper_reading", "human_note": "done", "reviewer": "me"},
            )
            self._run(out, calls, waves=1)
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))["papers"]["paper_one"]
            self.assertEqual({}, {stage: row for stage, row in state["layers"].items() if row.get("settled_at")})
            self.assertEqual([], state["human_verdicts_seen"])
            self.assertEqual([], [item for item in self._events(out) if item["event"] == "human_verdict_consumed"])

    def test_a_round_without_a_verdict_is_handed_no_preferences(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []
            self._run(out, calls, waves=1)
            self.assertTrue(all("human_preferences" not in item for item in calls))
            self.assertEqual([], [item for item in self._events(out) if item["event"] == "human_verdict_consumed"])


class ProgressSheetTests(unittest.TestCase):
    """Where the paper stands is written down and put in front of the seat that decides the fork."""

    def test_the_sheet_names_the_goal_the_order_and_each_layer(self):
        sheet = ce.progress_sheet({"layers": _layers("paper_reading")}, {"name": "paper_one"}, iteration=3)
        self.assertEqual("rfs.progress_sheet", sheet["contract"])
        self.assertEqual(3, sheet["iteration"])
        self.assertEqual(list(ce.STAGE_ORDER), sheet["layer_order"])
        self.assertEqual(ce.PROGRESS_SHEET_GOAL, sheet["goal"])
        self.assertEqual("content_planning", sheet["now_working_on"])
        rows = {item["stage"]: item for item in sheet["layers"]}
        self.assertTrue(rows["paper_reading"]["settled"])
        self.assertFalse(rows["content_planning"]["settled"])

    def test_the_sheet_carries_what_the_round_before_it_changed(self):
        entry = {
            "history": [
                {
                    "iteration": 2,
                    "status": "completed",
                    "fork_before": "segmentation",
                    "accepted_rounds": 1,
                    "best_score": 0.42,
                    "reuse": {"accepted": True},
                }
            ],
            "last_defect_ids": ["overlap"],
            "stalled_streak": 2,
        }
        self.assertEqual(
            {
                "iteration": 2,
                "status": "completed",
                "fork_before": "segmentation",
                "accepted_rounds": 1,
                "best_score": 0.42,
                "inherited": True,
                "defects_this_round": ["overlap"],
            },
            ce.previous_round_summary(entry),
        )
        self.assertEqual({}, ce.previous_round_summary({"history": []}))
        self.assertEqual(2, ce.progress_sheet(entry, {"name": "p"}, iteration=3)["stalled_streak"])

    def test_the_loop_writes_a_sheet_for_every_round_and_hands_it_to_the_seat(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            seen: list[dict] = []

            def review(previous: Path, spec: dict) -> dict:
                seen.append(dict(spec.get("progress_sheet") or {}))
                return {"ok": True, "reason": "", "stage_feedback": []}

            run_continuous_evolution(
                _config(),
                out,
                evolve_runner=LoopTests()._runner([]),
                review_runner=review,
                clock=lambda: 0.0,
                max_waves=2,
            )
            paper_dir = out / "papers" / "paper_one"
            first = json.loads((paper_dir / "iter_01" / "progress_sheet.json").read_text(encoding="utf-8"))
            self.assertEqual("paper_reading", first["now_working_on"])
            self.assertEqual({}, first["last_round"])
            latest = json.loads((paper_dir / ce.PROGRESS_SHEET_LATEST).read_text(encoding="utf-8"))
            self.assertEqual(2, latest["iteration"])
            self.assertEqual(1, latest["last_round"]["iteration"])
            self.assertEqual(1, json.loads((paper_dir / "iter_02" / "progress_sheet.json").read_text(encoding="utf-8"))["last_round"]["iteration"])
            # Every seat that was asked to decide a fork had a sheet in front of it.
            self.assertTrue(seen)
            self.assertTrue(all(item.get("now_working_on") in ce.STAGE_ORDER for item in seen))


class ProgressDirectionTests(unittest.TestCase):
    """The layer a round is on is asked about, and the figure is handed its own standing."""

    def _events(self, out: Path) -> list[dict]:
        return [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_the_sheet_asks_about_the_layer_the_run_is_on(self):
        sheet = ce.progress_sheet({"layers": _layers("paper_reading")}, {"name": "paper_one"}, iteration=4)
        self.assertEqual("content_planning", sheet["now_working_on"])
        self.assertIn("content_planning", sheet["ask"])
        self.assertIn("layer_settled", sheet["ask"])
        # The reason a good layer moves the round on has to be said, not implied: the seat read
        # "is this layer finished with" as a verdict to withhold, which is how the work face stayed
        # on the first layer.
        self.assertIn("already good", sheet["ask"])
        self.assertIn("next round", sheet["ask"])

    def test_a_layer_the_state_stopped_reading_still_arrives_with_its_reading(self):
        readings = {"segmentation": {"reading": {"box_pass": 2}, "iteration": 128}}
        sheet = ce.progress_sheet(
            {"layers": _layers("paper_reading")}, {"name": "paper_one"}, iteration=132, readings=readings
        )
        row = next(item for item in sheet["layers"] if item["stage"] == "segmentation")
        self.assertEqual({"box_pass": 2}, row["last_reading"])
        self.assertEqual(128, row["reading_from"])

    def test_the_reading_the_state_kept_is_not_replaced_by_an_older_one(self):
        entry = {
            "layers": {
                "segmentation": {"settled_at": 1, "settled_by": "model", "last_move_at": 131, "last_reading": {"box_pass": 5}}
            }
        }
        sheet = ce.progress_sheet(
            entry,
            {"name": "paper_one"},
            iteration=132,
            readings={"segmentation": {"reading": {"box_pass": 2}, "iteration": 128}},
        )
        row = next(item for item in sheet["layers"] if item["stage"] == "segmentation")
        self.assertEqual({"box_pass": 5}, row["last_reading"])
        self.assertEqual(131, row["reading_from"])

    def test_the_sheet_says_whether_this_round_inherits_the_best_version(self):
        entry = {"iteration": 131, "best": "/out/papers/p/iter_129/round_00/candidate_00", "best_score": 0.8057}
        inherited = ce.progress_sheet(entry, {"name": "paper_one"}, iteration=132, inherited=entry["best"])
        self.assertTrue(inherited["history"]["inherits_the_best"])
        self.assertEqual(129, inherited["history"]["best_iteration"])
        self.assertIn("best version this paper has reached", inherited["history"]["note"])
        other = ce.progress_sheet(entry, {"name": "paper_one"}, iteration=132, inherited="/out/papers/p/iter_131")
        self.assertFalse(other["history"]["inherits_the_best"])
        self.assertIn("iteration 129", other["history"]["note"])
        self.assertEqual(
            "no version of this paper has been measured against another one yet",
            ce.progress_sheet({"layers": {}}, {"name": "paper_one"}, iteration=1)["history"]["note"],
        )

    def test_the_readings_are_read_back_off_the_newest_ledgers_below_the_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp)
            for name, rows in (
                (
                    "iter_04",
                    [
                        {"stage": "segmentation", "reading": {"box_pass": 1}},
                        {"stage": "theme_planning", "reading": {}},
                    ],
                ),
                ("iter_03", [{"stage": "theme_planning", "reading": {"palette": 3}}]),
                ("iter_09", [{"stage": "segmentation", "reading": {"box_pass": 9}}]),
            ):
                (paper / name).mkdir()
                (paper / name / "layer_ledger.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
            found = ce.older_layer_readings(paper, before=5)
            self.assertEqual({"box_pass": 1}, found["segmentation"]["reading"])
            self.assertEqual(4, found["segmentation"]["iteration"])
            self.assertEqual({"palette": 3}, found["theme_planning"]["reading"])
            self.assertEqual(3, found["theme_planning"]["iteration"])

    def test_the_plan_names_the_layer_and_who_finished_the_others(self):
        entry = {
            "layers": {
                **_layers("paper_reading"),
                "content_planning": {"settled_at": 3, "settled_by": "human", "because": "a person closed it"},
            }
        }
        plan = ce.layer_plan(entry)
        self.assertEqual("global_planning", plan["working_on"])
        self.assertEqual(list(ce.STAGE_ORDER), plan["layer_order"])
        by_stage = {item["stage"]: item["by"] for item in plan["settled"]}
        self.assertEqual("human", by_stage["content_planning"])
        self.assertEqual("model", by_stage["paper_reading"])
        self.assertIn("global_planning", plan["still_open"])
        self.assertNotIn("content_planning", plan["still_open"])
        self.assertNotIn("paper_reading", plan["still_open"])
        self.assertEqual(ce.progress_ask("global_planning"), plan["ask"])
        self.assertEqual("paper_reading", ce.layer_plan({})["working_on"])

    def test_the_round_hands_the_plan_into_the_run_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []

            def review(previous: Path, spec: dict) -> dict:
                return {"ok": True, "reason": "", "stage_feedback": []}

            run_continuous_evolution(
                _config(),
                out,
                evolve_runner=LoopTests()._runner(calls),
                review_runner=review,
                clock=lambda: 0.0,
                max_waves=1,
            )
            self.assertTrue(calls)
            for options in calls:
                self.assertIn("layer_plan", options)
                self.assertEqual("paper_reading", options["layer_plan"]["working_on"])
                self.assertIn("layer_settled", options["layer_plan"]["ask"])
            handed = [item for item in self._events(out) if item["event"] == "layer_plan_handed_down"]
            self.assertTrue(handed)
            self.assertEqual("paper_reading", handed[0]["working_on"])
            self.assertEqual([], handed[0]["settled"])


class UpstreamCaseTests(unittest.TestCase):
    """Going back over a finished layer costs the argument every other claim costs."""

    def _review(self, *, stage: str, evidence: list, shown_fields: list | None = None) -> dict:
        judgement = {"fault_stage": stage, "fault_requirement": "the panel titles", "fault_evidence": evidence}
        if shown_fields is not None:
            judgement["advisory_shown_fields"] = shown_fields
        return {"ok": True, "reason": "", "stage_feedback": [], "judgement": judgement}

    def test_a_case_with_a_reading_the_seat_was_shown_is_an_argument(self):
        review = self._review(stage="global_planning", evidence=[{"field": "node_occupancy", "value": 0.9}], shown_fields=["node_occupancy"])
        verdict = ce.upstream_case(review)
        self.assertTrue(verdict["ok"])
        self.assertEqual("global_planning", verdict["case"]["fault_stage"])
        self.assertEqual([{"field": "node_occupancy", "value": 0.9}], verdict["case"]["fault_evidence"])

    def test_a_number_no_block_showed_the_seat_buys_nothing(self):
        review = self._review(stage="global_planning", evidence=[{"field": "invented_metric", "value": 3}], shown_fields=["node_occupancy"])
        verdict = ce.upstream_case(review)
        self.assertFalse(verdict["ok"])
        self.assertEqual("fault_evidence_unreadable", verdict["reason"])

    def test_a_layer_the_pipeline_does_not_have_buys_nothing(self):
        verdict = ce.upstream_case(self._review(stage="polishing", evidence=[{"field": "node_occupancy", "value": 3}], shown_fields=["node_occupancy"]))
        self.assertFalse(verdict["ok"])

    def test_a_reading_of_the_sheet_the_seat_read_is_citable_without_a_field_list(self):
        review = {"ok": True, "judgement": {"fault_stage": "content_planning", "fault_evidence": [{"field": "declared_nodes", "value": 4}], "progress_sheet": {"layers": [{"last_reading": {"declared_nodes": 4}}]}}}
        self.assertTrue(ce.upstream_case(review)["ok"])

    def test_the_allowed_check_asks_for_the_layer_the_fork_names(self):
        case = ce.upstream_case(self._review(stage="global_planning", evidence=[{"field": "node_occupancy", "value": 1}], shown_fields=["node_occupancy"]))
        self.assertTrue(ce.upstream_allowed(fork="global_planning", work="theme_planning", attributed="", streak=0, review={}, case=case))
        self.assertFalse(ce.upstream_allowed(fork="segmentation", work="theme_planning", attributed="", streak=0, review={}, case=case))
        # Working below the open layer is always free, case or no case.
        self.assertTrue(ce.upstream_allowed(fork="ppt_compilation", work="theme_planning", attributed="", streak=0, review={}, case={}))


class CrossRoundTests(unittest.TestCase):
    """This round put beside the best the paper had reached, as evidence and never as a gate."""

    def test_a_settled_layer_and_the_cadence_both_ask_for_a_comparison(self):
        settled = {"layers": {"paper_reading": {"settled_at": 3}}}
        self.assertTrue(ce.cross_round_due(settled, 3, 5))
        self.assertFalse(ce.cross_round_due({}, 3, 5))
        self.assertTrue(ce.cross_round_due({}, 5, 5))
        self.assertFalse(ce.cross_round_due({}, 0, 5))

    def test_the_baseline_is_the_best_recent_version_with_a_reading(self):
        archive = [
            {"iter": 1, "round": 0, "candidate": 0, "dir": "a", "absolute_score": 0.4},
            {"iter": 2, "round": 1, "candidate": 0, "dir": "b", "absolute_score": 0.7},
            {"iter": 2, "round": 0, "candidate": 1, "dir": "c", "absolute_score": None},
            {"iter": 3, "round": 0, "candidate": 0, "dir": "d", "absolute_score": 0.9},
        ]
        chosen = ce.cross_round_baseline(archive, before=4)
        self.assertEqual("d", chosen["dir"])
        self.assertEqual(3, chosen["iter"])
        # A version with no absolute reading cannot be the reference, and the current round cannot
        # be its own baseline.
        self.assertEqual("b", ce.cross_round_baseline(archive, before=3)["dir"])
        self.assertEqual({}, ce.cross_round_baseline(archive[:2], before=1))

    def test_an_old_version_is_not_the_baseline_any_more(self):
        archive = [{"iter": 1, "round": 0, "candidate": 0, "dir": "a", "absolute_score": 0.9}]
        self.assertEqual({}, ce.cross_round_baseline(archive, before=40, window=8))

    def test_the_verdict_names_which_version_the_seat_preferred(self):
        base = ce.cross_round_record(paper="p", iteration=4, current={"dir": "c"}, baseline={"dir": "b"}, answer={"preferred_candidate": "A"})
        self.assertEqual("history_better", base["verdict"])
        self.assertFalse(base["gate"])
        self.assertEqual("current_better", ce.cross_round_record(paper="p", iteration=4, current={}, baseline={}, answer={"preferred_candidate": "B"})["verdict"])
        self.assertEqual("tie", ce.cross_round_record(paper="p", iteration=4, current={}, baseline={}, answer={"preferred_candidate": "tie"})["verdict"])
        unanswered = ce.cross_round_record(paper="p", iteration=4, current={}, baseline={}, reason="nothing to compare against")
        self.assertEqual("not_compared", unanswered["verdict"])
        self.assertEqual("nothing to compare against", unanswered["reason"])


class CrossRoundLoopTests(unittest.TestCase):
    """The loop writes the comparison beside the round and lets nothing else read it."""

    def _run(self, out: Path, compare, *, every: int = 1, runner=None):
        return run_continuous_evolution(
            _config(cross_round_every=every),
            out,
            evolve_runner=runner or WorkFaceLoopTests()._accepted(),
            review_runner=lambda previous, spec: {"ok": True, "reason": "", "stage_feedback": []},
            cross_round_runner=compare,
            clock=lambda: 0.0,
            max_waves=1,
        )

    def test_the_comparison_is_written_and_decides_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            earlier = out / "papers" / "paper_one" / "iter_01"
            earlier.mkdir(parents=True)
            write_json(
                out / CONTINUOUS_STATE,
                {
                    "schema": CONTINUOUS_STATE_SCHEMA,
                    "version": 1,
                    "waves": 1,
                    "papers": {
                        "paper_one": {
                            "iteration": 1,
                            "archive": [{"iter": 1, "round": 0, "candidate": 0, "dir": str(earlier), "source": str(earlier), "absolute_score": 0.61}],
                            "best": str(earlier),
                        }
                    },
                },
            )
            calls: list[tuple] = []

            def compare(current: Path, history: Path, spec: dict) -> dict:
                calls.append((str(current), str(history)))
                return {"preferred_candidate": "A", "status": "fault"}

            report = self._run(out, compare)
            self.assertEqual(1, len(calls))
            self.assertTrue(calls[0][1].endswith("iter_01"))
            payload = json.loads((out / "papers" / "paper_one" / "iter_02" / "cross_round_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual("history_better", payload["verdict"])
            self.assertFalse(payload["gate"])
            self.assertEqual(1, payload["history_best"]["iter"])
            iteration = report["papers"][0]["iterations"][-1]
            self.assertEqual("history_better", iteration["cross_round"]["verdict"])
            self.assertEqual("completed", iteration["status"])
            self.assertEqual(1, iteration["accepted_rounds"])
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))["papers"]["paper_one"]
            self.assertEqual(1, len(state["cross_round"]))
            # The comparison is one of the round's own numbers, so the trend has to be written
            # after it. Counted before it, every round reads as one nobody had compared yet, and
            # the one number that says whether the paper is walking anywhere stays at zero.
            ledger = [json.loads(line) for line in (out / ce.ROUND_METRICS_LEDGER).read_text(encoding="utf-8").splitlines()]
            counted = next(item for item in ledger if item["paper"] == "paper_one" and item["iteration"] == 2)
            self.assertEqual(
                {"comparisons": 1, "wins": 0, "losses": 1, "ties": 0, "undecided": 0},
                counted["progress"]["cross_round"],
            )

    def test_a_round_with_no_reading_to_compare_against_says_so_and_asks_nobody(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[tuple] = []
            self._run(out, lambda current, history, spec: calls.append((current, history)) or {})
            payload = json.loads((out / "papers" / "paper_one" / "iter_01" / "cross_round_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual("not_compared", payload["verdict"])
            self.assertIn("absolute reading", payload["reason"])
            self.assertEqual([], calls)


class RotatedRowRecoveryTests(unittest.TestCase):
    """The library's own rotation must not throw a queued measurement away."""

    def _loop(self, root: Path, library: Path) -> "ce._Loop":
        config = read_continuous_config(
            _config(base_papers=[_paper("paper_one")], validation={"enabled": True, "suite": str(root / "suite.json")})
        )
        loop = ce._Loop(
            config=config,
            out=root / "continuous",
            library=str(library),
            stop_file="",
            evolve_runner=lambda **_: {"ok": True},
            review_runner=lambda *_a: {"ok": False, "stage_feedback": []},
            clock=lambda: 0.0,
            disk_usage=_roomy_disk,
            max_waves=0,
        )
        loop.state["validations"] = loop.validations
        return loop

    def _archive(self, library: Path, experience_ids: list[str]) -> Path:
        archive = library.parent / "library_archived_20261006"
        archive.mkdir(parents=True, exist_ok=True)
        with (archive / "judge_experiences.jsonl").open("w", encoding="utf-8") as handle:
            for identifier in experience_ids:
                handle.write(json.dumps({"experience_id": identifier, "status": "candidate"}) + "\n")
        return archive

    def test_a_row_that_only_the_archive_holds_is_measured_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            library.mkdir()
            (library / "judge_experiences.jsonl").write_text("", encoding="utf-8")
            self._archive(library, ["direct-exp-rotated"])
            loop = self._loop(root, library)
            loop.validations["direct-exp-rotated"] = {
                "status": "dropped",
                "blocked_because": "row_no_longer_in_library",
                "paper": "paper_one",
                "iteration": 3,
            }

            recovered = loop.recover_rotated_rows()

            self.assertEqual(["direct-exp-rotated"], recovered)
            self.assertEqual("requeued", loop.validations["direct-exp-rotated"]["status"])
            queued = loop._validation_queue.get_nowait()
            self.assertEqual("direct-exp-rotated", queued["experience_id"])
            events = [json.loads(line) for line in (loop.out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines()]
            self.assertIn("validation_row_recovered", [item["event"] for item in events])

    def test_a_row_no_archive_holds_is_left_where_it_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            library.mkdir()
            (library / "judge_experiences.jsonl").write_text("", encoding="utf-8")
            self._archive(library, ["direct-exp-something-else"])
            loop = self._loop(root, library)
            loop.validations["direct-exp-gone"] = {"status": "dropped", "blocked_because": "row_no_longer_in_library"}
            loop.validations["direct-exp-unknown"] = {"status": "failed", "error": "ValueError: Unknown experience: direct-exp-unknown"}

            recovered = loop.recover_rotated_rows()

            self.assertEqual([], recovered)
            self.assertEqual("dropped", loop.validations["direct-exp-gone"]["status"])
            self.assertEqual("failed", loop.validations["direct-exp-unknown"]["status"])
            self.assertTrue(loop._validation_queue.empty())

    def test_the_archive_makes_the_queue_stop_calling_a_row_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            library.mkdir()
            (library / "judge_experiences.jsonl").write_text(
                json.dumps({"experience_id": "direct-exp-live", "status": "candidate"}) + "\n", encoding="utf-8"
            )
            self._archive(library, ["direct-exp-archived"])
            loop = self._loop(root, library)

            held = loop._library_row_ids()

            self.assertEqual({"direct-exp-live", "direct-exp-archived"}, held)
            self.assertEqual("direct-exp-archived", loop._library_row("direct-exp-archived")["experience_id"])

    def test_a_row_only_the_archive_holds_is_pinned_before_the_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            library.mkdir()
            (library / "judge_experiences.jsonl").write_text("", encoding="utf-8")
            self._archive(library, ["direct-exp-rotated"])
            loop = self._loop(root, library)
            out = loop.validation_attempt_dir("direct-exp-rotated")

            self.assertTrue(loop._pin_archived_row("experience", "direct-exp-rotated", out))

            pinned = json.loads((out / "row_under_test.json").read_text(encoding="utf-8"))
            self.assertEqual("rfs.validation_row_under_test", pinned["contract"])
            self.assertEqual("direct-exp-rotated", pinned["experience"]["experience_id"])
            self.assertIsNone(pinned["prompt_patch"])
            events = [json.loads(line) for line in (loop.out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines()]
            self.assertIn("validation_row_pinned_from_archive", [item["event"] for item in events])

    def test_a_row_the_library_still_holds_is_left_for_the_validator_to_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            library.mkdir()
            (library / "judge_experiences.jsonl").write_text(
                json.dumps({"experience_id": "direct-exp-live", "status": "candidate"}) + "\n", encoding="utf-8"
            )
            loop = self._loop(root, library)
            out = loop.validation_attempt_dir("direct-exp-live")

            self.assertFalse(loop._pin_archived_row("experience", "direct-exp-live", out))
            self.assertFalse((out / "row_under_test.json").exists())

    def test_a_row_no_archive_holds_is_not_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            library.mkdir()
            (library / "judge_experiences.jsonl").write_text("", encoding="utf-8")
            self._archive(library, ["direct-exp-something-else"])
            loop = self._loop(root, library)
            out = loop.validation_attempt_dir("direct-exp-gone")

            self.assertFalse(loop._pin_archived_row("experience", "direct-exp-gone", out))
            self.assertFalse((out / "row_under_test.json").exists())

    def test_the_measurement_of_an_archived_row_reads_the_row_it_was_queued_for(self):
        """Requeueing a rotated row is only a rescue when the measurement can read that row."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            library.mkdir()
            (library / "judge_experiences.jsonl").write_text("", encoding="utf-8")
            self._archive(library, ["direct-exp-rotated"])
            seen: list[dict] = []

            def fake_validate(**options):
                seen.append(options)
                pinned = json.loads((Path(options["out"]) / "row_under_test.json").read_text(encoding="utf-8"))
                self.assertEqual(options["experience_id"], pinned["experience"]["experience_id"])
                return {"status": "inconclusive", "summary": {"cases": 1}}

            config = read_continuous_config(_config(base_papers=[_paper("paper_one")], validation={"enabled": True, "suite": str(root / "suite.json")}))
            loop = ce._Loop(
                config=config,
                out=root / "continuous",
                library=str(library),
                stop_file="",
                evolve_runner=lambda **_: {"ok": True},
                review_runner=lambda *_a: {"ok": False, "stage_feedback": []},
                clock=lambda: 0.0,
                disk_usage=_roomy_disk,
                max_waves=0,
                validate_runner=fake_validate,
            )
            loop.state["validations"] = loop.validations

            loop.run_validation({"experience_id": "direct-exp-rotated", "paper": "paper_one", "iteration": 3})

            self.assertEqual(1, len(seen))
            self.assertEqual("inconclusive", loop.validations["direct-exp-rotated"]["status"])


class KeptFigureTests(unittest.TestCase):
    """A round always inherits a figure, even when the iteration before it kept none."""

    def test_a_directory_only_counts_when_it_holds_a_showable_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(ce.holds_kept_figure(root))
            write_text(root / "final_draft_preview.png", "png")
            self.assertFalse(ce.holds_kept_figure(root), "a preview with no program is not a figure")
            write_json(root / "figure_program.json", {})
            self.assertTrue(ce.holds_kept_figure(root))

    def test_the_newest_earlier_iteration_that_left_a_figure_is_the_one_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper_dir = Path(tmp) / "papers" / "paper_one"
            paper_dir.mkdir(parents=True)
            for number in (1, 2, 3, 5):
                _seed_iteration(Path(tmp), "paper_one", number)
            _seed_iteration(Path(tmp), "paper_one", 4)
            (paper_dir / "iter_04" / "final_draft_preview.png").unlink()
            (paper_dir / "iter_04" / "figure_program.json").unlink()
            self.assertEqual("iter_03", ce.newest_kept_figure_dir(paper_dir, 4).name)
            self.assertEqual("iter_05", ce.newest_kept_figure_dir(paper_dir, 6).name)
            self.assertIsNone(ce.newest_kept_figure_dir(paper_dir, 1))

    def test_an_empty_previous_iteration_does_not_leave_the_round_without_a_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []

            def runner(**options):
                calls.append(dict(options))
                iter_dir = Path(options["out"])
                number = int(iter_dir.name.split("_")[-1])
                _seed_iteration(out, iter_dir.parent.name, number, score=0.5 + 0.1 * number)
                if number == 2:
                    # Every repair rolled back, so this iteration published no figure of its own.
                    (iter_dir / "final_draft_preview.png").unlink()
                    (iter_dir / "figure_program.json").unlink()
                return {"ok": True, "out_dir": str(iter_dir), "rounds_completed": number}

            run_continuous_evolution(
                _config(),
                out,
                evolve_runner=runner,
                review_runner=lambda *_a: {"ok": False, "stage_feedback": []},
                clock=lambda: 0.0,
                max_waves=3,
            )

            third = [item for item in calls if item["out"].endswith("iter_03")]
            self.assertTrue(third)
            for options in third:
                self.assertTrue(options["reuse_upstream_from"].endswith("iter_01"), options["reuse_upstream_from"])
                self.assertTrue(options["baseline_from"].endswith("iter_01"), options["baseline_from"])
            events = [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines()]
            walked = [item for item in events if item["event"] == "baseline_walked_back"]
            self.assertTrue(walked, "the substitution was not recorded")
            self.assertTrue(str(walked[0]["from_dir"]).endswith("iter_02"))
            self.assertTrue(str(walked[0]["to_dir"]).endswith("iter_01"))


class OpeningMoveTests(unittest.TestCase):
    """A review that named a bounded repair becomes the next iteration's first move.

    The iteration inherits the very figure the review was written about, so running the repair it
    named is a real change to that figure. Planning the same stages again is the work the earlier
    iteration already paid for, and it leaves the review unread.
    """

    def test_a_bounded_repair_opens_the_iteration(self):
        repair = {"type": "resize_nodes", "slot_ids": ["slot_1"], "factor": 1.08, "dx": 0.0, "dy": 0.0}
        feedback = [{"requested_actions": [repair]}]
        self.assertEqual(fork_before_from_feedback(feedback)[0], INCREMENTAL_FORK_STAGE)

        opening = opening_move_from_feedback(feedback, INCREMENTAL_FORK_STAGE)

        self.assertEqual(opening["type"], "resize_nodes")
        self.assertEqual(opening["factor"], 1.08)
        self.assertEqual(opening["slot_ids"], ["slot_1"])

    def test_a_stage_rerun_is_not_an_opening_change(self):
        # A repair that names a stage says the draft has to be planned again, which is a redraw
        # and not an incremental change to the inherited figure.
        feedback = [{"requested_actions": ["replan_structure"]}]
        fork_stage, _reason = fork_before_from_feedback(feedback)
        self.assertNotEqual(fork_stage, INCREMENTAL_FORK_STAGE)
        self.assertEqual(opening_move_from_feedback(feedback, fork_stage), {})

    def test_a_review_that_asked_for_nothing_leaves_the_first_round_alone(self):
        self.assertEqual(opening_move_from_feedback([], DEFAULT_FORK_STAGE), {})
        self.assertEqual(opening_move_from_feedback(None, DEFAULT_FORK_STAGE), {})

    def test_a_repair_without_its_parameters_is_not_run(self):
        # The executor reads factor/dx/dy, so a geometric move that names none of them is the
        # shipped default in disguise rather than the change the reviewer asked for.
        feedback = [{"requested_actions": ["resize_nodes"]}]
        self.assertEqual(fork_before_from_feedback(feedback)[0], INCREMENTAL_FORK_STAGE)
        self.assertEqual(opening_move_from_feedback(feedback, INCREMENTAL_FORK_STAGE), {})

    def test_a_name_the_alias_does_not_recognise_is_dropped(self):
        self.assertEqual(requested_actions([{"requested_actions": ["reroute_arrows"]}]), [])
        self.assertEqual(opening_move_from_feedback([{"requested_actions": ["reroute_arrows"]}], DEFAULT_FORK_STAGE), {})

    def test_the_earliest_stage_named_is_still_what_decides_the_fork(self):
        # A review that named a bounded repair beside a stage re-run asks for a redraw, so the
        # bounded half must not be run against a figure that is about to be planned again.
        feedback = [{"requested_actions": [{"type": "resize_nodes", "factor": 1.08, "dx": 0.0, "dy": 0.0}]}, {"requested_actions": ["replan_structure"]}]
        fork_stage, _reason = fork_before_from_feedback(feedback)
        self.assertEqual(fork_stage, "global_planning")
        self.assertEqual(opening_move_from_feedback(feedback, fork_stage), {})


class ConfigTests(unittest.TestCase):
    def test_a_paper_list_is_normalised_and_deduplicated(self):
        config = read_continuous_config({"continuous": {"base_papers": ["/papers/one.pdf", {"paper": "/papers/two.pdf", "domain": "materials"}]}})
        self.assertEqual([spec["name"] for spec in config["papers"]], ["one", "two"])
        self.assertEqual(config["papers"][1]["domain"], "materials")
        self.assertEqual(config["parallel"], DEFAULT_PARALLEL)
        self.assertEqual(config["iteration_rounds"], DEFAULT_ITERATION_ROUNDS)

    def test_two_papers_may_not_resolve_to_one_directory(self):
        with self.assertRaises(ValueError):
            read_continuous_config({"continuous": {"base_papers": [{"paper": "/a/one.pdf", "paper_id": "same"}, {"paper": "/b/other.pdf", "paper_id": "same"}]}})

    def test_evolve_options_come_from_inside_and_outside_the_block(self):
        config = read_continuous_config({"continuous": {"base_papers": [_paper("p")], "evolve": {"icon_mode": "image2"}}, "judge_mode": "vlm"})
        self.assertEqual(config["evolve"]["icon_mode"], "image2")
        self.assertEqual(config["evolve"]["judge_mode"], "vlm")


class DiskFloorConfigTests(unittest.TestCase):
    """The floor is a number the caller writes down, not one the loop quietly lowers."""

    def test_the_default_floor_is_the_shipped_share(self):
        config = read_continuous_config(_config())
        self.assertEqual(config["disk_floor"], 0.10)
        self.assertEqual(config["disk_floor_gb"], 0.0)

    def test_the_configured_floors_are_read_and_clamped(self):
        config = read_continuous_config(_config(disk_floor=0.02, disk_floor_gb=40))
        self.assertEqual(config["disk_floor"], 0.02)
        self.assertEqual(config["disk_floor_gb"], 40.0)
        self.assertEqual(read_continuous_config(_config(disk_floor=5))["disk_floor"], 0.9)
        self.assertEqual(read_continuous_config(_config(disk_floor=-1))["disk_floor"], 0.0)
        self.assertEqual(read_continuous_config(_config(disk_floor_gb=-5))["disk_floor_gb"], 0.0)


class StateTests(unittest.TestCase):
    def test_a_restart_keeps_what_the_config_still_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            config = _config()
            state = load_continuous_state(out, read_continuous_config(config))
            state["papers"]["paper_one"]["iteration"] = 4
            state["papers"]["paper_one"]["best"] = "/somewhere/best"
            write_json(out / CONTINUOUS_STATE, state)

            resumed = load_continuous_state(out, read_continuous_config(config))
            self.assertEqual(resumed["papers"]["paper_one"]["iteration"], 4)
            self.assertEqual(resumed["papers"]["paper_one"]["best"], "/somewhere/best")
            self.assertEqual(resumed["papers"]["paper_two"]["iteration"], 0)

    def test_a_paper_the_config_dropped_disappears(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_json(out / CONTINUOUS_STATE, {"papers": {"gone": {"iteration": 9}, "paper_one": {"iteration": 2}}})
            config = read_continuous_config(_config())
            resumed = load_continuous_state(out, config)
            self.assertEqual(sorted(resumed["papers"]), ["paper_one", "paper_two"])
            self.assertEqual(resumed["papers"]["paper_one"]["iteration"], 2)


class LoopTests(unittest.TestCase):
    def _runner(self, calls: list, *, decline_reuse: str = ""):
        """A stand-in for the real run: it files an iteration and says what it did with the reuse."""

        def runner(**options):
            calls.append(dict(options))
            out = Path(options["out"])
            number = int(out.name.split("_")[-1])
            iter_dir = _seed_iteration(out.parents[2], out.parent.name, number, score=0.5 + 0.1 * number)
            if options.get("reuse_upstream_from"):
                reuse = {
                    "source": str(options["reuse_upstream_from"]),
                    "fork_before": str(options.get("fork_before") or ""),
                    "reused_stages": ["design_seed"],
                }
                if decline_reuse:
                    reuse["declined_reason"] = decline_reuse
                write_json(iter_dir / "stage_reuse_report.json", reuse)
            return {"ok": True, "out_dir": str(iter_dir), "rounds_completed": number}

        return runner

    def test_the_second_iteration_inherits_the_first_and_forks_where_the_review_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []
            reviews: list = []

            def review(previous: Path, spec: dict):
                reviews.append(previous)
                return {"ok": True, "reason": "", "stage_feedback": [{"requested_actions": ["replan_theme"]}]}

            report = run_continuous_evolution(
                _config(), out, evolve_runner=self._runner(calls), review_runner=review, clock=lambda: 0.0, max_waves=2
            )

            self.assertEqual(report["waves"], 2)
            first = [item for item in calls if item["out"].endswith("iter_01")]
            second = [item for item in calls if item["out"].endswith("iter_02")]
            self.assertEqual(len(first), 2)
            self.assertEqual(len(second), 2)
            for options in first:
                self.assertNotIn("reuse_upstream_from", options)
                self.assertNotIn("fork_before", options)
            for options in second:
                self.assertTrue(options["reuse_upstream_from"].endswith("iter_01"))
                self.assertEqual(options["fork_before"], "theme_planning")
            self.assertEqual(len(reviews), 2)
            self.assertTrue(all(path.name == "iter_01" for path in reviews))
            self.assertEqual(report["papers"][0]["iteration"], 2)
            self.assertTrue(report["papers"][0]["iterations"][1]["reuse"]["accepted"])
            self.assertEqual(report["papers"][0]["reuse_declined"], [])
            self.assertIn("inherited the previous version", (out / CONTINUOUS_MARKDOWN).read_text(encoding="utf-8"))

    def test_a_fault_in_the_reading_that_named_a_repair_is_answered_by_it(self):
        # The attribution outranks the repair only because no repair reaches the reading. That
        # ranking costs the whole figure, and the same call that blamed the reading also named a
        # repair below it -- so on one verdict the repair is what this iteration runs, and the
        # reading the inherited figure was built on is kept.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []

            def review(_previous: Path, _spec: dict):
                return {
                    "ok": True,
                    "reason": "",
                    "stage_feedback": [{"requested_actions": ["replan_structure"]}],
                    "judgement": {"fault_stage": "paper_reading"},
                }

            run_continuous_evolution(
                _config(), out, evolve_runner=self._runner(calls), review_runner=review, clock=lambda: 0.0, max_waves=2
            )

            second = [item for item in calls if item["out"].endswith("iter_02")]
            self.assertTrue(second)
            for options in second:
                self.assertEqual(options["fork_before"], "global_planning")
                self.assertTrue(options["reuse_upstream_from"].endswith("iter_01"))
            events = [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines()]
            started = [item for item in events if item["event"] == "iteration_started" and item["iteration"] == 2][0]
            self.assertEqual(started["fork_before"], "global_planning")
            self.assertEqual(started["reading_attribution"], "paper_reading")
            self.assertEqual(started["reading_attribution_streak"], 1)
            self.assertNotIn("reads the paper again", started["fork_reason"])

    def test_a_reading_blamed_twice_in_a_row_inherits_nothing_and_reads_the_paper_again(self):
        # A review that says the reading itself was wrong cannot be answered from the reading the
        # figure was built on: inheriting it would inherit the thing the review complains about.
        # Two iterations in a row that place the fault there is a statement about the reading
        # rather than about one call, so the second of them takes nothing from the source -- while
        # still being measured against the figure it continues from, because a re-read has a
        # previous version like every other round.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []

            def review(_previous: Path, _spec: dict):
                return {
                    "ok": True,
                    "reason": "",
                    "stage_feedback": [{"requested_actions": ["replan_structure"]}],
                    "judgement": {"fault_stage": "paper_reading"},
                }

            run_continuous_evolution(
                _config(), out, evolve_runner=self._runner(calls), review_runner=review, clock=lambda: 0.0, max_waves=3
            )

            third = [item for item in calls if item["out"].endswith("iter_03")]
            self.assertTrue(third)
            for options in third:
                self.assertNotIn("reuse_upstream_from", options)
                self.assertEqual(options["fork_before"], "paper_reading")
                self.assertTrue(str(options["baseline_from"]).endswith("iter_02"))
            events = [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines()]
            started = [
                item
                for item in events
                if item["event"] == "iteration_started" and item["iteration"] == 3 and item["paper"] == "paper_one"
            ][0]
            self.assertEqual(started["fork_before"], "paper_reading")
            self.assertEqual(started["reuse_from"], "")
            self.assertEqual(started["reading_attribution_streak"], READING_ATTRIBUTION_CONFIRMATIONS)
            self.assertIn("reads the paper again", started["fork_reason"])

    def test_the_iteration_that_inherits_opens_on_the_repair_the_review_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []
            repair = {"type": "reroute", "edge_ids": ["rel_1"], "problem": "two arrows cross"}

            def review(_previous: Path, _spec: dict):
                return {"ok": True, "reason": "", "stage_feedback": [{"requested_actions": [repair]}]}

            report = run_continuous_evolution(
                _config(), out, evolve_runner=self._runner(calls), review_runner=review, clock=lambda: 0.0, max_waves=2
            )

            first = [item for item in calls if item["out"].endswith("iter_01")]
            second = [item for item in calls if item["out"].endswith("iter_02")]
            for options in first:
                self.assertNotIn("first_round_action", options, "a first iteration has nothing to inherit")
            for options in second:
                self.assertEqual(options["fork_before"], INCREMENTAL_FORK_STAGE)
                self.assertEqual(options["first_round_action"]["type"], "reroute")
                self.assertEqual(options["first_round_action"]["edge_ids"], ["rel_1"])
            self.assertEqual(report["papers"][0]["iterations"][1]["first_round_action"]["type"], "reroute")
            events = [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines()]
            started = {item["iteration"]: item for item in events if item["event"] == "iteration_started" and item["paper"] == "paper_one"}
            self.assertIsNone(started[1]["first_round_action"])
            self.assertEqual(started[2]["first_round_action"]["type"], "reroute")

    def test_each_run_is_handed_the_identifier_its_config_already_knows(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[dict] = []
            run_continuous_evolution(
                _config(),
                Path(tmp),
                evolve_runner=self._runner(calls, decline_reuse="no upstream"),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                clock=lambda: 0.0,
                max_waves=1,
            )

            self.assertTrue(calls)
            for options in calls:
                self.assertEqual(options["paper_id"], Path(options["out"]).parents[0].name)

    def test_a_config_that_names_no_identifier_passes_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[dict] = []
            run_continuous_evolution(
                _config(base_papers=[{"paper": "/papers/one.pdf"}]),
                Path(tmp),
                evolve_runner=self._runner(calls, decline_reuse="no upstream"),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                clock=lambda: 0.0,
                max_waves=1,
            )

            self.assertTrue(calls)
            for options in calls:
                self.assertNotIn("paper_id", options)

    def test_a_review_whose_repair_names_a_stage_leaves_the_first_round_as_it_was(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []
            report = run_continuous_evolution(
                _config(),
                out,
                evolve_runner=self._runner(calls),
                review_runner=lambda *_a: {"ok": True, "reason": "", "stage_feedback": [{"requested_actions": ["replan_structure"]}]},
                clock=lambda: 0.0,
                max_waves=2,
            )

            second = [item for item in calls if item["out"].endswith("iter_02")]
            for options in second:
                self.assertEqual(options["fork_before"], "global_planning")
                self.assertNotIn("first_round_action", options)
            self.assertEqual(report["papers"][0]["iterations"][1]["first_round_action"], {})

    def test_a_declined_reuse_is_reported_as_a_replanned_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            reason = "stage replay requires a complete reference analysis"
            report = run_continuous_evolution(
                _config(),
                out,
                evolve_runner=self._runner([], decline_reuse=reason),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                clock=lambda: 0.0,
                max_waves=2,
            )

            for paper in report["papers"]:
                self.assertEqual(paper["reuse_declined"], [2])
                self.assertFalse(paper["iterations"][1]["reuse"]["accepted"])
                self.assertEqual(paper["iterations"][1]["reuse"]["declined_reason"], reason)
            events = [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines()]
            declined = [item for item in events if item["event"] == "reuse_declined"]
            self.assertEqual(len(declined), 2)
            self.assertEqual(declined[0]["reason"], reason)
            markdown = (out / CONTINUOUS_MARKDOWN).read_text(encoding="utf-8")
            self.assertIn("re-planned everything", markdown)
            self.assertIn("started fresh", markdown)

    def test_a_restart_continues_from_the_last_iteration_instead_of_starting_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first_calls: list[dict] = []
            run_continuous_evolution(_config(), out, evolve_runner=self._runner(first_calls), review_runner=lambda *_a: {"ok": False, "stage_feedback": []}, clock=lambda: 0.0, max_waves=1)
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
            self.assertEqual(state["papers"]["paper_one"]["iteration"], 1)

            second_calls: list[dict] = []
            report = run_continuous_evolution(_config(), out, evolve_runner=self._runner(second_calls), review_runner=lambda *_a: {"ok": True, "reason": "", "stage_feedback": []}, clock=lambda: 0.0, max_waves=1)
            self.assertEqual(report["waves"], 2)
            self.assertEqual(report["waves_this_call"], 1)
            self.assertEqual([Path(item["out"]).name for item in second_calls], ["iter_02", "iter_02"])
            self.assertEqual(report["papers"][0]["iteration"], 2)
            for options in second_calls:
                self.assertTrue(options["reuse_upstream_from"].endswith("iter_01"))
                self.assertEqual(options["fork_before"], DEFAULT_FORK_STAGE)

    def test_the_reading_is_blamed_twice_before_the_paper_is_read_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []
            report = run_continuous_evolution(
                _config(),
                out,
                evolve_runner=self._runner(calls),
                review_runner=lambda *_a: {
                    "ok": True,
                    "reason": "",
                    "stage_feedback": [{"requested_actions": ["replan_structure"]}],
                    "judgement": {"fault_stage": "content_planning"},
                },
                clock=lambda: 0.0,
                max_waves=3,
            )

            for options in [item for item in calls if item["out"].endswith("iter_02")]:
                self.assertEqual(options["fork_before"], "global_planning")
            for options in [item for item in calls if item["out"].endswith("iter_03")]:
                self.assertEqual(options["fork_before"], "content_planning")
            self.assertEqual(report["papers"][0]["iteration"], 3)
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
            stamped = state["papers"]["paper_one"]["last_reading_attribution"]
            self.assertEqual(stamped["stage"], "content_planning")
            self.assertEqual(stamped["iteration"], 3)
            self.assertGreaterEqual(int(stamped["streak"]), READING_ATTRIBUTION_CONFIRMATIONS)
            events = [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines()]
            third = [
                item
                for item in events
                if item["event"] == "iteration_started" and item["iteration"] == 3 and item["paper"] == "paper_one"
            ][0]
            self.assertEqual(third["reading_attribution"], "content_planning")
            self.assertIn("reads the paper again", third["fork_reason"])

    def test_an_iteration_that_blamed_no_reading_breaks_the_run(self):
        # The run of attributions is counted per iteration, and only against the one before it: a
        # review that blamed no reading -- or never ran -- breaks the run rather than letting two
        # unrelated verdicts add up to a re-read. Iteration 03's review did not answer, so
        # iteration 04's verdict is a first one again and may not send the run back to the paper.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []

            def review(previous: Path, _spec: dict):
                if previous.name == "iter_02":
                    return {"ok": False, "reason": "no_kept_figure", "stage_feedback": []}
                return {"ok": True, "reason": "", "stage_feedback": [], "judgement": {"fault_stage": "content_planning"}}

            run_continuous_evolution(
                _config(base_papers=[_paper("paper_one")]),
                out,
                evolve_runner=self._runner(calls),
                review_runner=review,
                clock=lambda: 0.0,
                max_waves=4,
            )

            fourth = [item for item in calls if item["out"].endswith("iter_04")]
            self.assertTrue(fourth)
            for options in fourth:
                self.assertEqual(options["fork_before"], DEFAULT_FORK_STAGE)
                self.assertNotEqual(options["fork_before"], "content_planning")
            events = [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines()]
            silent = [item for item in events if item["event"] == "iteration_started" and item["iteration"] == 3][0]
            self.assertEqual(silent["reading_attribution"], "")
            restarted = [item for item in events if item["event"] == "iteration_started" and item["iteration"] == 4][0]
            self.assertEqual(restarted["reading_attribution_streak"], 1)
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
            stamped = state["papers"]["paper_one"]["last_reading_attribution"]
            self.assertEqual(stamped["stage"], "content_planning")
            self.assertEqual(int(stamped["streak"]), 1)

    def test_a_review_that_failed_does_not_restart_the_iteration_from_the_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []
            report = run_continuous_evolution(
                _config(),
                out,
                evolve_runner=self._runner(calls),
                review_runner=lambda *_a: {
                    "ok": False,
                    "reason": "review_failed: ValueError: could not convert string to float: 'critical'",
                    "stage_feedback": [],
                },
                clock=lambda: 0.0,
                max_waves=2,
            )

            second = [item for item in calls if item["out"].endswith("iter_02")]
            self.assertTrue(second)
            for options in second:
                self.assertEqual(options["fork_before"], INCREMENTAL_FORK_STAGE)
                self.assertNotEqual(options["fork_before"], DEFAULT_FORK_STAGE)
            self.assertEqual(report["papers"][0]["iteration"], 2)
            events = [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines()]
            started = [item for item in events if item["event"] == "iteration_started" and item["iteration"] == 2 and item["paper"] == "paper_one"]
            self.assertTrue(started)
            self.assertFalse(started[0]["review_ok"])
            self.assertIn("review_failed", started[0]["review_reason"])
            self.assertIn("did not answer", started[0]["fork_reason"])

    def test_the_stop_file_ends_the_loop_after_the_wave_in_flight(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stop = out / "STOP"
            calls: list[dict] = []

            def runner(**options):
                calls.append(dict(options))
                Path(stop).write_text("stop", encoding="utf-8")
                _seed_iteration(out, Path(options["out"]).parent.name, 1)
                return {"ok": True}

            report = run_continuous_evolution(_config(), out, evolve_runner=runner, review_runner=lambda *_a: {"ok": False, "stage_feedback": []}, stop_file=stop, clock=lambda: 0.0)
            self.assertEqual(report["stop_reason"], "stop_requested")
            self.assertEqual(len(calls), 2)
            self.assertEqual(report["waves"], 1)

    def test_the_budget_stops_the_loop_between_waves(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            clock = {"now": 0.0}

            def runner(**_options):
                clock["now"] += 4000.0
                name = Path(_options["out"]).parent.name
                _seed_iteration(out, name, 1)
                return {"ok": True}

            report = run_continuous_evolution(_config(budget_hours=1.0), out, evolve_runner=runner, review_runner=lambda *_a: {"ok": False, "stage_feedback": []}, clock=lambda: clock["now"])
            self.assertEqual(report["stop_reason"], "budget_exhausted")
            self.assertEqual(report["waves"], 1)

    def test_the_disk_floor_stops_the_loop_between_waves(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report = run_continuous_evolution(_config(), out, evolve_runner=lambda **_o: {"ok": True}, review_runner=lambda *_a: {"ok": False, "stage_feedback": []}, clock=lambda: 0.0, disk_usage=lambda _p: 0.01)
            self.assertEqual(report["stop_reason"], "disk_below_floor")
            self.assertEqual(report["waves"], 0)

    def test_an_absolute_floor_is_what_stops_the_wave(self):
        # 5% of this volume is still 20 GB, which a share floor of 1% would wave through. The
        # caller asked to be stopped at 40 GB, so that is the number that decides, and the report
        # says which one it was.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report = run_continuous_evolution(
                _config(disk_floor=0.01, disk_floor_gb=40.0),
                out,
                evolve_runner=lambda **_o: {"ok": True},
                review_runner=lambda *_a: {"ok": False, "stage_feedback": []},
                clock=lambda: 0.0,
                disk_usage=lambda _p: {"ratio": 0.05, "free_gb": 20.0},
            )
            self.assertEqual(report["stop_reason"], "disk_below_floor")
            self.assertEqual(report["waves"], 0)
            self.assertIn("40 GB", report["stop_detail"])
            self.assertEqual(report["disk_floor"]["gb"], 40.0)
            self.assertEqual(report["disk_floor"]["reading"]["free_gb"], 20.0)

    def test_an_absolute_floor_with_room_lets_the_wave_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report = run_continuous_evolution(
                _config(disk_floor=0.10, disk_floor_gb=40.0),
                out,
                evolve_runner=self._runner([]),
                review_runner=lambda *_a: {"ok": False, "stage_feedback": []},
                clock=lambda: 0.0,
                disk_usage=lambda _p: {"ratio": 0.001, "free_gb": 200.0},
                max_waves=1,
            )
            self.assertEqual(report["stop_reason"], "wave_limit")
            self.assertEqual(report["waves_this_call"], 1)

    def test_a_paper_that_keeps_failing_is_parked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            def runner(**_options):
                raise RuntimeError("the provider refused")

            report = run_continuous_evolution(_config(), out, evolve_runner=runner, review_runner=lambda *_a: {"ok": False, "stage_feedback": []}, clock=lambda: 0.0)
            self.assertEqual(report["stop_reason"], "all_papers_paused")
            self.assertEqual(report["waves"], MAX_CONSECUTIVE_FAILURES)
            self.assertTrue(all(paper["paused"] for paper in report["papers"]))
            events = (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines()
            self.assertTrue(any(json.loads(line)["event"] == "paper_paused" for line in events))

    def test_a_new_call_gives_a_parked_paper_another_chance(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            def failing(**_options):
                raise RuntimeError("the provider refused")

            first = run_continuous_evolution(_config(), out, evolve_runner=failing, review_runner=lambda *_a: {"ok": False, "stage_feedback": []}, clock=lambda: 0.0)
            self.assertEqual(first["stop_reason"], "all_papers_paused")

            # Parking is scoped to the call that set it: a caller that relaunches the loop has
            # said it wants the paper tried again, so the flag must not silently persist.
            second = run_continuous_evolution(_config(), out, evolve_runner=self._runner([]), review_runner=lambda *_a: {"ok": False, "stage_feedback": []}, clock=lambda: 0.0, max_waves=1)

            self.assertEqual(second["waves_this_call"], 1)
            self.assertFalse(any(paper["paused"] for paper in second["papers"]))
            self.assertEqual(second["papers"][0]["iteration"], 1)
            events = [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len([item for item in events if item["event"] == "paper_resumed"]), 2)

    def test_a_wave_moves_every_paper_at_the_same_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls: list[dict] = []
            report = run_continuous_evolution(
                _config(base_papers=[_paper("a"), _paper("b"), _paper("c")], parallel=3),
                out,
                evolve_runner=self._runner(calls),
                review_runner=lambda *_a: {"ok": False, "stage_feedback": []},
                clock=lambda: 0.0,
                max_waves=1,
            )
            self.assertEqual(report["waves"], 1)
            self.assertEqual(sorted(Path(item["out"]).name for item in calls), ["iter_01", "iter_01", "iter_01"])
            self.assertEqual(sorted(Path(item["out"]).parent.name for item in calls), ["a", "b", "c"])

    def test_the_report_lands_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_continuous_evolution(_config(), out, evolve_runner=self._runner([]), review_runner=lambda *_a: {"ok": False, "stage_feedback": []}, clock=lambda: 0.0, max_waves=1)
            self.assertTrue((out / CONTINUOUS_STATE).is_file())
            self.assertTrue((out / CONTINUOUS_EVENTS).is_file())
            self.assertTrue((out / CONTINUOUS_REPORT).is_file())
            self.assertTrue((out / CONTINUOUS_MARKDOWN).is_file())
            report = json.loads((out / CONTINUOUS_REPORT).read_text(encoding="utf-8"))
            self.assertEqual(report["contract"], "rfs.continuous_evolution_report")
            self.assertEqual(len(report["papers"]), 2)


class RoundMetricsTests(unittest.TestCase):
    """The numbers a round leaves about its own progress: the walk down the layers, and its trend.

    ``rounds_to_settle`` and ``rounds_to_settle_all`` are the two numbers the direction change
    exists to move, so they are asserted from the record the loop writes rather than recomputed in
    the test. The run's own metrics file must survive the loop's addition to it: two writers of one
    file is exactly how the walk would read as the five numbers and nothing else.
    """

    def _run(self, tmp, *, waves=2, settle=""):
        out = Path(tmp)

        def runner(**options):
            iter_dir = Path(options["out"])
            number = int(iter_dir.name.split("_")[-1])
            _seed_iteration(out, iter_dir.parent.name, number, score=0.5 + 0.1 * number)
            # The run writes its own five numbers beside itself; the loop must add to this file.
            write_json(
                iter_dir / ce.ROUND_METRICS_FILE,
                {
                    "contract": "rfs.evolution_metrics",
                    "version": 1,
                    "headline": {"location_accuracy": 1.0},
                    "claims": {"rate": 0.5, "settled": 1, "open": 1},
                },
            )
            write_json(
                iter_dir / "round_00" / "stage_gate.json",
                {"round": 0, "stages": [{"stage": "content_planning", "changed": True, "reasons": ["the plan grew"], "reading": {"modules": 9}}]},
            )
            if settle and number == 1:
                reflection = {"layer_settled": {"stage": settle, "because": "the content plan is complete"}}
                write_json(iter_dir / "round_00" / "reflection.json", reflection)
                write_json(iter_dir / "evolution_rounds.json", {"rounds": [{"round": 0, "reflection": reflection}]})
            return {"ok": True, "out_dir": str(iter_dir), "rounds_completed": number}

        report = run_continuous_evolution(
            _config(),
            out,
            evolve_runner=runner,
            review_runner=lambda *_a: {"ok": False, "stage_feedback": []},
            clock=lambda: 0.0,
            max_waves=waves,
        )
        return out, report

    def test_the_runs_own_numbers_are_kept_and_the_walk_is_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, report = self._run(tmp, waves=2)
            payload = json.loads((out / "papers" / "paper_one" / "iter_01" / ce.ROUND_METRICS_FILE).read_text(encoding="utf-8"))
            self.assertEqual(payload["contract"], "rfs.evolution_metrics", "the run's own file was replaced")
            self.assertEqual(payload["headline"]["location_accuracy"], 1.0)
            progress = payload["progress"]
            self.assertEqual(progress["contract"], "rfs.evolution_round_metrics")
            self.assertEqual(progress["iteration"], 1)
            self.assertEqual(progress["layers"]["total"], len(ce.STAGE_ORDER))
            # This round's own share of the counters, not the running total it inherited.
            self.assertEqual(progress["upstream"], {"requested": 0, "passed": 0, "rate": None})
            self.assertEqual(progress["settled_this_round"], [])
            # The run's claim count reaches the report without opening every iteration again.
            self.assertEqual(report["papers"][0]["claims"]["rate"], 0.5)

    def test_the_trend_ledger_gets_one_line_per_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self._run(tmp, waves=2)
            lines = [json.loads(line) for line in (out / ce.ROUND_METRICS_LEDGER).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sorted(item["iteration"] for item in lines), [1, 1, 2, 2])
            self.assertTrue(all(item["progress"]["layers_total"] == len(ce.STAGE_ORDER) for item in lines))

    def test_a_settled_layer_is_counted_with_the_rounds_it_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, report = self._run(tmp, waves=2, settle="content_planning")
            walk = report["papers"][0]["layer_walk"]
            row = walk["layers"]["content_planning"]
            self.assertEqual(row["settled_at"], 1)
            self.assertEqual(row["settled_by"], "model")
            self.assertEqual(row["first_move_at"], 1)
            self.assertEqual(row["rounds_to_settle"], 1)
            # Nothing has settled the reading yet, so the paper has not walked the whole order and
            # the total is absent rather than zero.
            self.assertFalse(walk["all_settled"])
            self.assertIsNone(walk["rounds_to_settle_all"])

    def test_a_layer_nobody_moved_has_no_cost_to_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, report = self._run(tmp, waves=2, settle="theme_planning")
            row = report["papers"][0]["layer_walk"]["layers"]["theme_planning"]
            self.assertEqual(row["settled_at"], 1)
            self.assertEqual(row["first_move_at"], 0)
            self.assertIsNone(row["rounds_to_settle"], "a layer nobody moved did not take no rounds")

    def test_the_patch_library_reports_sentences_not_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = out / "lib"
            library.mkdir()
            sentence = "route the crossing connectors through one shared vertical channel"
            rows = [
                {"patch_id": "p1", "apply_at": "ppt_compilation", "directive": sentence, "class": "preference", "support": 4, "status": "candidate"},
                {"patch_id": "p2", "apply_at": "ppt_compilation", "directive": sentence, "class": "preference", "support": 7, "status": "candidate"},
                {"patch_id": "p3", "apply_at": "theme_planning", "directive": "warm the palette toward the reference", "class": "preference", "support": 1, "status": "promoted"},
            ]
            write_text(library / "prompt_patches.jsonl", "\n".join(json.dumps(row) for row in rows) + "\n")
            report = run_continuous_evolution(
                _config(),
                out,
                experience_library=str(library),
                evolve_runner=self._runner_with(out),
                review_runner=lambda *_a: {"ok": False, "stage_feedback": []},
                clock=lambda: 0.0,
                max_waves=1,
            )
            yield_report = report["patch_yield"]
            self.assertEqual(yield_report["patches"], 3)
            self.assertEqual(yield_report["distinct_sentences"], 2)
            self.assertEqual(yield_report["asked"], 12)
            self.assertEqual(yield_report["licensed"], 1)
            self.assertAlmostEqual(yield_report["novelty"], 0.6667, places=4)

    def _runner_with(self, out: Path):
        def runner(**options):
            iter_dir = Path(options["out"])
            number = int(iter_dir.name.split("_")[-1])
            _seed_iteration(out, iter_dir.parent.name, number, score=0.5)
            return {"ok": True, "out_dir": str(iter_dir), "rounds_completed": number}

        return runner


class LayerWalkTests(unittest.TestCase):
    """The two numbers the direction change is judged on, computed from the record rather than state.

    A walk read off the live state changes every time the state moves, so the number a round wrote
    and the number a reader recomputes disagree. These tests fix the rule instead: the answers and
    the rounds worked in decide the walk, and recomputing from the same record gives the same
    answer.
    """

    def _layers(self, *, first=1):
        return {stage: {"first_move_at": first, "last_move_at": first} for stage in ce.STAGE_ORDER}

    def _settled(self, *, at):
        return [{"stage": stage, "iteration": at, "by": "model", "because": "done"} for stage in ce.STAGE_ORDER]

    def test_the_last_answer_for_a_layer_is_the_round_it_was_finished_in(self):
        events = [
            {"stage": "segmentation", "iteration": 2, "by": "model"},
            {"stage": "segmentation", "iteration": 6, "by": "human"},
        ]
        walk = ce.layer_walk(self._layers(), stage_order=ce.STAGE_ORDER, settle_events=events)
        row = walk["layers"]["segmentation"]
        self.assertEqual(6, row["settled_at"])
        self.assertEqual("human", row["settled_by"], "a person's later answer is the one that stands")
        self.assertEqual(6, row["rounds_to_settle"])

    def test_every_layer_settled_counts_from_the_round_the_paper_worked_in(self):
        walk = ce.layer_walk(
            self._layers(),
            stage_order=ce.STAGE_ORDER,
            settle_events=self._settled(at=3),
            worked_rounds=[1, 2, 4, 5],
        )
        self.assertTrue(walk["all_settled"])
        # The answers all came in round 3, which moved nothing, so the paper is not recorded as
        # having finished every layer until round 4.
        self.assertEqual(4, walk["rounds_to_settle_all"])
        again = ce.layer_walk(
            self._layers(),
            stage_order=ce.STAGE_ORDER,
            settle_events=self._settled(at=3),
            worked_rounds=[1, 2, 4, 5],
        )
        self.assertEqual(walk, again, "the same record has to give the same walk")

    def test_a_layer_opened_again_is_not_settled_and_pushes_the_total_out(self):
        events = self._settled(at=3)
        reopened = [{"stage": "segmentation", "iteration": 4, "by": "model"}]
        walk = ce.layer_walk(self._layers(), stage_order=ce.STAGE_ORDER, settle_events=events, reopen_events=reopened)
        self.assertEqual(0, walk["layers"]["segmentation"]["settled_at"])
        self.assertFalse(walk["all_settled"])
        self.assertIsNone(walk["rounds_to_settle_all"])
        # Settled again after the reopen, and the paper has finished every layer only now.
        walk = ce.layer_walk(
            self._layers(),
            stage_order=ce.STAGE_ORDER,
            settle_events=events + [{"stage": "segmentation", "iteration": 7, "by": "model"}],
            reopen_events=reopened,
            worked_rounds=[1, 2, 3, 4, 7],
        )
        self.assertEqual(7, walk["layers"]["segmentation"]["settled_at"])
        self.assertEqual(7, walk["rounds_to_settle_all"])

    def test_a_paper_with_no_answers_at_all_reads_as_not_settled(self):
        walk = ce.layer_walk(self._layers(), stage_order=ce.STAGE_ORDER)
        self.assertEqual(0, walk["settled"])
        self.assertFalse(walk["all_settled"])
        self.assertIsNone(walk["rounds_to_settle_all"])

    def test_an_old_run_still_reads_as_far_as_it_got(self):
        layers = {"paper_reading": {"first_move_at": 1, "settled_at": 1, "settled_by": "model"}}
        walk = ce.layer_walk(layers, stage_order=("paper_reading", "content_planning"))
        self.assertEqual(1, walk["layers"]["paper_reading"]["settled_at"])
        self.assertEqual(1, walk["layers"]["paper_reading"]["rounds_to_settle"])
        self.assertEqual(0, walk["layers"]["content_planning"]["settled_at"])


class WiringTests(unittest.TestCase):
    def test_the_workflow_is_exported_and_registered(self):
        from rfs.workflows import run_continuous_evolution as exported

        self.assertTrue(callable(exported))
        from rfs.cli import build_parser

        args = build_parser().parse_args(["paper-evolve-continuous", "--config", "c.json", "--out", "o", "--max-waves", "2"])
        self.assertEqual(args.command, "paper-evolve-continuous")
        self.assertEqual(args.max_waves, 2)


class RoundGuardWiringTests(unittest.TestCase):
    """A round that spends two moves must ask the same guards the loop asks between waves.

    Both halves of this were reachable only from a unit test until the loop handed them over:
    the figure a first round is measured against, and the stop file / disk floor / budget a
    second move is asked about. The wiring is what the assertions are for.
    """

    def _calls_for_one_wave(self, **overrides):
        calls: list[dict] = []

        def runner(**options):
            calls.append(dict(options))
            out = Path(options["out"])
            number = int(out.name.split("_")[-1])
            iter_dir = _seed_iteration(out.parents[2], out.parent.name, number, score=0.5)
            return {"ok": True, "out_dir": str(iter_dir), "rounds_completed": number}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_continuous_evolution(
                _config(**overrides),
                out,
                evolve_runner=runner,
                review_runner=lambda *_a: {"ok": False, "stage_feedback": []},
                clock=lambda: 0.0,
                disk_usage=_roomy_disk,
                max_waves=2,
            )
        return [item for item in calls if item["out"].endswith("iter_01")][0]

    def test_the_first_round_is_told_which_figure_it_continues_from(self):
        calls: list[dict] = []

        def runner(**options):
            calls.append(dict(options))
            out = Path(options["out"])
            number = int(out.name.split("_")[-1])
            iter_dir = _seed_iteration(out.parents[2], out.parent.name, number, score=0.5 + 0.1 * number)
            return {"ok": True, "out_dir": str(iter_dir), "rounds_completed": number}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_continuous_evolution(
                _config(),
                out,
                evolve_runner=runner,
                review_runner=lambda *_a: {"ok": False, "stage_feedback": []},
                clock=lambda: 0.0,
                disk_usage=_roomy_disk,
                max_waves=2,
            )
        first = [item for item in calls if item["out"].endswith("iter_01")][0]
        second = [item for item in calls if item["out"].endswith("iter_02")][0]
        self.assertNotIn("baseline_from", first)
        self.assertTrue(second["baseline_from"].endswith("iter_01"))
        self.assertEqual(second["baseline_from"], second["reuse_upstream_from"])

    def test_every_move_guard_the_loop_owns_is_handed_to_the_round(self):
        options = self._calls_for_one_wave(disk_floor=0.05, disk_floor_gb=40)
        guard = options["move_guard"]
        self.assertEqual(guard["stop_file"], "")
        self.assertEqual(guard["disk_path"], str(Path(options["out"]).parents[2]))
        self.assertEqual(guard["disk_floor_gb"], 40.0)
        self.assertEqual(guard["disk_floor_ratio"], 0.05)
        # 24h of budget measured from the clock the loop was given, so a round can tell
        # whether it would overrun it before it spends a second move.
        self.assertEqual(guard["budget_deadline"], 86400.0)

    def test_a_zero_budget_is_handed_over_as_no_deadline_at_all(self):
        options = self._calls_for_one_wave(budget_hours=0.0)
        self.assertEqual(options["move_guard"]["budget_deadline"], 0.0)




class LayerSealGuardTests(unittest.TestCase):
    """Only the layer the round is on may be sealed, and a person's seal is theirs to open."""

    def _loop(self, tmp: str):
        return ce._Loop(
            config=read_continuous_config(_config()),
            out=Path(tmp),
            library="",
            stop_file="",
            evolve_runner=lambda **options: {},
            review_runner=lambda previous, spec: {"ok": False},
            clock=lambda: 0.0,
            disk_usage=_roomy_disk,
            max_waves=1,
        )

    def test_the_model_may_not_seal_a_layer_two_steps_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = self._loop(tmp)
            entry: dict = {"layers": {}}
            self.assertFalse(loop.settle_layer(entry, "ppt_compilation", paper="p", number=2, because="looks fine"))
            self.assertEqual("paper_reading", ce.work_face(entry["layers"]))
            self.assertNotIn("ppt_compilation", entry["layers"])

    def test_the_model_seals_the_layer_it_is_on_and_the_face_moves_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = self._loop(tmp)
            entry: dict = {"layers": {}}
            self.assertTrue(loop.settle_layer(entry, "paper_reading", paper="p", number=2, because="the plan is complete"))
            self.assertEqual("content_planning", entry["work_face"])
            self.assertTrue(loop.settle_layer(entry, "content_planning", paper="p", number=3, because="the content is chosen"))
            self.assertEqual("global_planning", entry["work_face"])

    def test_a_person_may_seal_a_layer_the_loop_has_not_reached(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = self._loop(tmp)
            entry: dict = {"layers": {}}
            self.assertTrue(loop.settle_layer(entry, "segmentation", paper="p", number=2, because="a person says so", by="human"))
            self.assertEqual("human", entry["layers"]["segmentation"]["settled_by"])
            # The face still stands at the first layer nobody has answered for.
            self.assertEqual("paper_reading", entry["work_face"])

    def test_the_model_may_not_open_a_layer_a_person_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = self._loop(tmp)
            entry: dict = {"layers": {}}
            loop.settle_layer(entry, "paper_reading", paper="p", number=2, because="a person read it", by="human")
            self.assertFalse(loop.reopen_layer(entry, "paper_reading", paper="p", number=3, why="stuck for eleven rounds"))
            self.assertEqual(2, entry["layers"]["paper_reading"]["settled_at"])
            self.assertEqual("human", entry["layers"]["paper_reading"]["settled_by"])
            self.assertEqual(0, int(entry["layers"]["paper_reading"].get("reopen_count") or 0))

    def test_a_case_naming_the_unwritten_requirement_opens_a_persons_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = self._loop(tmp)
            entry: dict = {"layers": {}}
            loop.settle_layer(entry, "paper_reading", paper="p", number=2, because="a person read it", by="human")
            case = {
                "ok": True,
                "case": {"fault_stage": "paper_reading", "fault_requirement": "the training loop is missing from the plan", "fault_evidence": [{"field": "declared_nodes", "value": 7}]},
            }
            self.assertTrue(loop.reopen_layer(entry, "paper_reading", paper="p", number=3, why="the case", case=case))
            self.assertEqual("paper_reading", entry["work_face"])
            self.assertEqual(1, entry["layers"]["paper_reading"]["reopen_count"])

    def test_a_person_opens_their_own_seal_without_a_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = self._loop(tmp)
            entry: dict = {"layers": {}}
            loop.settle_layer(entry, "paper_reading", paper="p", number=2, because="a person read it", by="human")
            self.assertTrue(loop.reopen_layer(entry, "paper_reading", paper="p", number=3, why="a person changed their mind", by="human"))

    def test_the_loops_own_seal_is_still_opened_by_the_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = self._loop(tmp)
            entry: dict = {"layers": {}}
            loop.settle_layer(entry, "paper_reading", paper="p", number=2, because="the plan is complete")
            self.assertTrue(loop.reopen_layer(entry, "paper_reading", paper="p", number=3, why="the figure is stuck"))

    def test_the_stalled_hint_states_the_fact_and_asks_for_nothing(self):
        hint = ce.deadlock_hint(7, 4, "icon_generation")
        self.assertIn("7 rounds in a row the figure has not moved forward from icon_generation", hint)
        self.assertNotIn("say so", hint)
        self.assertNotIn("reopen", hint)
        self.assertEqual("", ce.deadlock_hint(3, 4, "icon_generation"))


class PaperRecordTests(unittest.TestCase):
    """An interrupted wave must not cost the rounds it had already finished.

    The wave's state file is written once per wave, so a process that dies inside a wave used to
    lose the loop's account of every round in it -- the rounds were on disk and paid for, and the
    restart bought them again. Two records say otherwise, and both are on disk: the account each
    paper writes the moment its own round ends, and the round's own output.
    """

    def setUp(self) -> None:
        self.calls: list[str] = []
        #: Whether the fake round keeps the candidate it drew. A round whose repairs all rolled back
        #: keeps nothing, and that is the round the claim has to reconstruct too.
        self.kept = True

    def _claim_loop(self, out: Path):
        """The loop, without a wave: enough to ask what it makes of the papers' own records."""

        return ce._Loop(
            config=read_continuous_config(_config(base_papers=[_paper("paper_one")])),
            out=out,
            library=str(out / "library"),
            stop_file="",
            evolve_runner=lambda **_: {},
            review_runner=lambda *_a: {},
            clock=lambda: 0.0,
            disk_usage=_roomy_disk,
            max_waves=0,
        )

    def _runner(self):
        def runner(**options):
            target = Path(options["out"])
            self.calls.append(target.name)
            number = int(target.name.split("_")[-1])
            iter_dir = _seed_iteration(target.parents[2], target.parent.name, number)
            if not self.kept:
                write_json(iter_dir / "evolution_state.json", {"best_candidate_dir": None, "best_score": None})
            write_json(iter_dir / "judgement.json", {"absolute_score": 0.5, "gate_score": 0.5})
            # A round that settled a layer and moved one, so the ledger a claim is rebuilt from has
            # something in it and the reconstruction can be read back.
            write_json(
                iter_dir / "evolution_rounds.json",
                {
                    "rounds": [
                        {
                            "round": 1,
                            "accepted": True,
                            "repair_decision": {"reason": "repair"},
                            "reflection": {"layer_settled": {"stage": "paper_reading", "because": "the plan lists every module"}},
                        }
                    ]
                },
            )
            write_json(
                iter_dir / "round_00" / "stage_gate.json",
                {
                    "round": 0,
                    "stages": [{"stage": "paper_reading", "changed": True, "reasons": ["the plan was read"], "reading": {"declared_nodes": 7}}],
                    "figure_metrics": {"after": {"node_count": 5}, "delta": {"node_count": 1}},
                },
            )
            return {"ok": True, "out_dir": str(iter_dir), "rounds_completed": number}

        return runner

    def _run(self, out: Path, *, waves: int = 1, **overrides):
        block = {"base_papers": [_paper("paper_one")]}
        block.update(overrides)
        return run_continuous_evolution(
            _config(**block),
            out,
            evolve_runner=self._runner(),
            review_runner=lambda previous, spec: {"ok": True, "reason": "", "stage_feedback": []},
            clock=lambda: 0.0,
            max_waves=waves,
        )

    def _events(self, out: Path) -> list[dict]:
        return [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines() if line.strip()]

    def _entry(self, out: Path) -> dict:
        return json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))["papers"]["paper_one"]

    def _rewind(self, out: Path, iteration: int) -> None:
        """Put the wave's accounting back the way an interrupted wave leaves it."""

        payload = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
        payload["papers"]["paper_one"]["iteration"] = iteration
        write_json(out / CONTINUOUS_STATE, payload)

    def test_every_round_is_written_beside_its_paper_the_moment_it_ends(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._run(out)

            record = json.loads((out / "papers" / "paper_one" / ce.PAPER_RECORD_FILE).read_text(encoding="utf-8"))

            self.assertEqual("paper_one", record["paper"])
            self.assertEqual(1, record["iteration"])
            self.assertEqual(1, record["entry"]["iteration"])
            self.assertTrue(str(record["entry"]["best"]).endswith("candidate_00"))
            self.assertTrue(str(record["entry"]["reuse_from"]).endswith("iter_01"))

    def test_a_restart_believes_the_paper_over_the_wave(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._run(out)
            self._rewind(out, 0)
            self.calls.clear()

            self._run(out)

            self.assertIn("paper_record_recovered", [item["event"] for item in self._events(out)])
            self.assertEqual(["iter_02"], self.calls)
            self.assertEqual(2, self._entry(out)["iteration"])

    def test_a_round_that_finished_before_the_wave_was_written_is_claimed(self):
        # The wave died before it wrote anything at all: no per-paper account, no state. The only
        # account of the round is the round itself, and re-running it would be paying twice.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._run(out)
            self._rewind(out, 0)
            (out / "papers" / "paper_one" / ce.PAPER_RECORD_FILE).unlink()
            self.calls.clear()

            self._run(out)

            self.assertIn("round_claimed_from_output", [item["event"] for item in self._events(out)])
            self.assertEqual(["iter_02"], self.calls)
            entry = self._entry(out)
            self.assertIn("claimed_from_output", [row.get("status") for row in entry["history"]])
            # The claim rebuilt the layer record off the round itself: the reading layer reads as
            # settled in the first round -- a number only the claimed round could have said -- and
            # the work face stands where that answer left it.
            self.assertEqual(1, entry["layers"]["paper_reading"]["settled_at"])
            self.assertEqual("content_planning", entry["work_face"])
            self.assertTrue(str(entry["best"]).endswith("candidate_00"))

    def test_a_round_that_left_no_readable_record_is_run_again(self):
        # A claim is built out of the round's own records, and half of one would decide the next
        # round's fork on a premise nobody answered. So an unreadable round is run again, and said
        # to be run again, rather than inherited.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._run(out)
            self._rewind(out, 0)
            (out / "papers" / "paper_one" / ce.PAPER_RECORD_FILE).unlink()
            (out / "papers" / "paper_one" / "iter_01" / "layer_ledger.json").unlink()
            self.calls.clear()

            self._run(out)

            self.assertIn("round_on_disk_without_record", [item["event"] for item in self._events(out)])
            self.assertEqual(["iter_01"], self.calls)

    def test_a_paper_that_reached_the_target_is_left_out_of_the_waves(self):
        # Without a target the waves run the full budget for every paper, which is how the papers
        # that were ahead after an interruption came to be carried past the round the sweep reads.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.calls.clear()

            report = self._run(out, waves=3, target_iterations=1)

            self.assertEqual(["iter_01"], self.calls)
            # Having run what it was asked for is not a failure: a finished batch reports ok.
            self.assertTrue(report["ok"])
            stopped = next(item for item in self._events(out) if item["event"] == "continuous_stopped")
            self.assertEqual("rounds_complete", stopped["reason"])
    def test_a_round_that_kept_no_candidate_is_claimed_with_the_one_it_drew(self):
        # The repairs all rolled back, so the round published nothing and recorded no best
        # candidate. The round still ran, still settled what it settled and still drew a figure,
        # and the loop's own accounting falls back to the first candidate it drew -- so does this.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.kept = False
            self._run(out)
            self._rewind(out, 0)
            (out / "papers" / "paper_one" / ce.PAPER_RECORD_FILE).unlink()

            loop = self._claim_loop(out)

            self.assertEqual(["paper_one"], loop.reconcile_paper_records())
            entry = loop.state["papers"]["paper_one"]
            self.assertEqual(1, entry["iteration"])
            self.assertIn(str(out / "papers" / "paper_one" / "iter_01"), str(entry["best"]))
            self.assertTrue(str(entry["best"]).endswith("candidate_00"))

    def test_a_rebuilt_record_that_disagrees_with_its_ledger_is_not_claimed(self):
        # The claim is only safe while it can be checked against what the round wrote. A ledger
        # that says the work stands somewhere else than the rebuilt record does is a record this
        # loop did not write, and the round is run again rather than inherited on it.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._run(out)
            self._rewind(out, 0)
            (out / "papers" / "paper_one" / ce.PAPER_RECORD_FILE).unlink()
            ledger_path = out / "papers" / "paper_one" / "iter_01" / ce.LAYER_LEDGER_FILE
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["work_face"] = "ppt_compilation"
            write_json(ledger_path, ledger)

            loop = self._claim_loop(out)

            self.assertEqual([], loop.reconcile_paper_records())
            self.assertEqual(0, loop.state["papers"]["paper_one"]["iteration"])


if __name__ == "__main__":
    unittest.main()
