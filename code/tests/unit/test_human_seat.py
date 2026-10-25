"""The seats a person stands at: the card each question leaves, and what the answer decides.

The loop already asks which of two figures is better and whether a layer is finished with. When a
person is sitting at that seat the question is written down as a card, the chain stops on it, and
the answer decides -- while the model's own answer is kept beside it as a shadow, so "how far apart
were the two" is a number rather than an impression. These tests pin the card, the answers, and
what each one is allowed to move.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct.human_queue import ANSWER_FILE, MODEL_ANSWER_FILE, QUESTION_FILE, HumanQueue
from rfs.utils import read_json, write_json
from rfs.workflows import continuous_evolution as ce
from rfs.workflows import human_review
from rfs.workflows.direct_experience_validation import ROW_UNDER_TEST_CONTRACT, ROW_UNDER_TEST_FILE


def _roomy_disk() -> dict:
    return {"ratio": 0.9, "free_gb": 900.0}


def _paper(name: str) -> dict:
    return {"paper": f"/papers/{name}.pdf", "paper_id": name, "domain": "biology"}


def _call_arguments(text: str, start: int) -> str:
    """The argument text of the call whose opening parenthesis sits just before ``start``."""

    depth = 1
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return text[start:]


def _config(**overrides) -> dict:
    block = {"base_papers": [_paper("paper_one"), _paper("paper_two")], "budget_hours": 24.0, "iteration_rounds": 3}
    block.update(overrides)
    return {"continuous": block}


def _loop(tmp: str, **overrides):
    return ce._Loop(
        config=ce.read_continuous_config(_config(**overrides)),
        out=Path(tmp),
        library="",
        stop_file="",
        evolve_runner=lambda **options: {},
        review_runner=lambda previous, spec: {"ok": False},
        clock=lambda: 0.0,
        disk_usage=_roomy_disk,
        max_waves=1,
    )


class QueueClock:
    """A clock a test drives by hand, so a hold can expire without waiting an hour for it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        # Every poll moves the clock past the hold, so a card nobody answers is a card whose hold
        # runs out on the second look rather than one the test waits through.
        self.now += max(seconds, 1.0)


class TheCardTests(unittest.TestCase):
    def test_a_judgement_point_nobody_sits_at_writes_no_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = HumanQueue(root=Path(tmp), seats=("figure_choice",), enabled=False)
            answer = queue.ask(kind="figure_choice", paper="p", iteration=1, round_index=0, question="?", model_answer={"choice": "B"})
            self.assertFalse(answer.answered)
            self.assertEqual("model", answer.by)
            self.assertEqual("B", answer.choice)
            self.assertEqual([], list((Path(tmp) / "human_queue").glob("**/*")) if (Path(tmp) / "human_queue").exists() else [])

    def test_the_card_does_not_carry_the_models_own_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clock = QueueClock()
            queue = HumanQueue(root=Path(tmp), seats=("figure_choice",), hold_hours=0.001, clock=clock, sleeper=clock.sleep, poll_seconds=1.0)
            queue.ask(
                kind="figure_choice",
                paper="p",
                iteration=3,
                round_index=0,
                question="which is better",
                model_answer={"choice": "B", "reason": "the new one is denser"},
                figures={"A": "a.png", "B": "b.png"},
                readings={"before": 0.5, "after": 0.7},
            )
            card = Path(tmp) / "human_queue" / "p" / "iter_03" / "round_00" / "figure_choice"
            shown = read_json(card / QUESTION_FILE)
            self.assertEqual("which is better", shown["question"])
            self.assertEqual("a.png", shown["figures"]["A"])
            self.assertNotIn("B", json.dumps({key: value for key, value in shown.items() if key != "figures"}))
            shadow = read_json(card / MODEL_ANSWER_FILE)
            self.assertEqual("B", shadow["choice"])

    def test_an_answer_that_never_comes_leaves_the_model_answer_and_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clock = QueueClock()
            queue = HumanQueue(root=Path(tmp), seats=("figure_choice",), hold_hours=0.001, clock=clock, sleeper=clock.sleep, poll_seconds=1.0)
            answer = queue.ask(kind="figure_choice", paper="p", iteration=1, round_index=0, question="?", model_answer={"choice": "A"})
            self.assertFalse(answer.answered)
            self.assertEqual("timeout", answer.by)
            self.assertEqual("A", answer.choice)
            self.assertIn("nobody answered", answer.reason)

    def test_a_second_answer_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = HumanQueue(root=Path(tmp), seats=("figure_choice",))
            card = queue.card_dir(paper="p", iteration=1, round_index=0, kind="figure_choice")
            card.mkdir(parents=True, exist_ok=True)
            write_json(card / QUESTION_FILE, {"kind": "figure_choice"})
            self.assertTrue(queue.answer(card, choice="A")["ok"])
            second = queue.answer(card, choice="B")
            self.assertFalse(second["ok"])
            self.assertEqual("A", read_json(card / ANSWER_FILE)["choice"])

    def test_the_model_answer_and_the_persons_answer_are_both_on_the_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clock = QueueClock()
            queue = HumanQueue(root=Path(tmp), seats=("figure_choice",), hold_hours=0.001, clock=clock, sleeper=clock.sleep, poll_seconds=1.0)

            def answer_it(payload: dict) -> None:
                HumanQueue(root=Path(tmp)).answer(payload["card"], choice="A", note="the older one reads better")

            queue.on_question = answer_it
            got = queue.ask(kind="figure_choice", paper="p", iteration=1, round_index=0, question="?", model_answer={"choice": "B"})
            self.assertTrue(got.answered)
            self.assertEqual("A", got.choice)
            card = Path(tmp) / "human_queue" / "p" / "iter_01" / "round_00" / "figure_choice"
            self.assertEqual("B", read_json(card / MODEL_ANSWER_FILE)["choice"])
            self.assertEqual("A", read_json(card / ANSWER_FILE)["choice"])

    def test_the_review_commands_read_the_same_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _loop(tmp, human_seat={"enabled": True, "seats": ["acceptance"], "papers": ["paper_one"]})
            self.assertTrue(loop.human_queue.sits_at("figure_choice"))
            card = loop.human_queue.card_dir(paper="paper_one", iteration=2, round_index=0, kind="figure_choice")
            card.mkdir(parents=True, exist_ok=True)
            write_json(card / QUESTION_FILE, {"contract": "rfs.human_question", "version": 1, "kind": "figure_choice", "paper": "paper_one"})
            self.assertEqual(1, human_review.review_queue(run_root=tmp)["open"])
            self.assertTrue(human_review.review_answer(card=str(card), choice="B", note="mine", reviewer="<AUTHOR>")["ok"])
            self.assertEqual(0, human_review.review_queue(run_root=tmp)["open"])


class WhatAPersonsAnswerMovesTests(unittest.TestCase):
    def test_a_person_can_seal_a_layer_the_model_could_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _loop(tmp)
            entry: dict = {"layers": {}}
            self.assertTrue(loop.settle_layer(entry, "segmentation", paper="p", number=2, because="reads fine", by="human"))
            self.assertEqual("human", entry["layers"]["segmentation"]["settled_by"])

    def test_a_persons_answer_is_read_off_the_iteration_it_answered_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _loop(tmp, human_seat={"enabled": True, "seats": ["acceptance"], "papers": ["paper_one"]})
            entry = loop.state["papers"]["paper_one"]
            entry["layers"] = {}
            iter_dir = Path(tmp) / "papers" / "paper_one" / "iter_02"
            round_dir = iter_dir / "round_00"
            round_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                round_dir / "human_verdict.json",
                {
                    "contract": "rfs.human_verdict",
                    "version": 1,
                    "round": 0,
                    "kind": "figure_choice",
                    "stage": "global_planning",
                    "settled_layer": "global_planning",
                    "direction_layer": "",
                    "answer": {"answered": True, "by": "human", "choice": "B", "note": "this plan is good"},
                },
            )
            rows = ce.iteration_human_answers(iter_dir)
            self.assertEqual(["global_planning"], [row["settled_layer"] for row in rows])
            for row in rows:
                if row["settled_layer"]:
                    loop.settle_layer(entry, row["settled_layer"], paper="paper_one", number=2, because=row["settled_because"], by="human")
            self.assertEqual("human", entry["layers"]["global_planning"]["settled_by"])

    def test_a_person_naming_an_earlier_layer_opens_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _loop(tmp)
            entry: dict = {"layers": {}}
            loop.settle_layer(entry, "paper_reading", paper="p", number=2, because="read", by="human")
            loop.settle_layer(entry, "content_planning", paper="p", number=3, because="planned", by="human")
            self.assertEqual("global_planning", ce.work_face(entry["layers"]))
            self.assertTrue(loop.reopen_layer(entry, "content_planning", paper="p", number=4, why="a person said the fault is still in this layer", by="human"))
            self.assertEqual("content_planning", entry["work_face"])

    def test_a_card_nobody_answered_is_not_read_as_an_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "papers" / "p" / "iter_01" / "round_00"
            iter_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                iter_dir / "human_verdict.json",
                {
                    "contract": "rfs.human_verdict",
                    "version": 1,
                    "round": 0,
                    "stage": "segmentation",
                    "settled_layer": "",
                    "direction_layer": "",
                    "answer": {"answered": False, "by": "timeout", "choice": "A"},
                },
            )
            self.assertEqual([], ce.iteration_human_answers(iter_dir.parent.parent))

    def test_the_seat_is_held_for_the_named_papers_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _loop(tmp, human_seat={"enabled": True, "seats": ["acceptance"], "papers": ["paper_one"]})
            self.assertTrue(loop.human_paper({"name": "paper_one"}))
            self.assertFalse(loop.human_paper({"name": "paper_two"}))
            everywhere = _loop(tmp, human_seat={"enabled": True, "seats": ["acceptance"]})
            self.assertTrue(everywhere.human_paper({"name": "paper_two"}))


class ThePersonsVerdictDecidesTheRoundTests(unittest.TestCase):
    """The round's own accept/rollback call, when a person is sitting at it."""

    def _run(self, root: Path, gate, capture: list[str] | None = None, max_rounds: int = 2, **extra):
        import contextlib

        import fitz
        from unittest.mock import patch

        from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable

        target = root / "paper.pdf"
        document = fitz.open()
        page = document.new_page(width=600, height=800)
        page.insert_text((50, 60), "A Comparable Study", fontsize=14)
        page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
        page.insert_textbox((50, 160, 550, 240), "We propose a frozen encoder that turns a paper into a symbolic figure plan.", fontsize=10)
        page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
        document.save(target)
        document.close()

        def answer(prompt: str, images: list[str] | None = None, **kwargs: object) -> dict:
            if capture is not None:
                capture.append(str(prompt))
            if "the state it has just produced" in prompt:
                return {"preferred_candidate": "B", "margin": "clear", "confidence": 0.7, "blocking_issues": []}
            if "why_changed" in prompt or "reflection" in prompt.casefold():
                return {"outcome": "improved", "failure_type": "none", "why_changed": "the repair helped", "scope_assessment": "appropriate", "next_policy": "keep"}
            if "stage_feedback" in prompt:
                return {
                    "status": "fault",
                    "stage_feedback": [{"scope_type": "global", "problem": "too sparse", "requested_actions": [{"action": "relayout_global", "factor": 1.03, "dx": 0.01, "dy": 0.0}]}],
                    "blocking_issues": [],
                }
            return {"status": "pass", "overall_score": 0.6, "dimension_scores": {}, "blocking_issues": []}

        out = root / "run"
        progress = patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None))
        vlm = patch("rfs.evaluation.direct_judges.call_vlm_json", side_effect=answer)
        with progress, vlm:
            run_paper_evolve_editable(
                paper=target,
                out=out,
                planner_mode="heuristic",
                method_mode="heuristic",
                structure_mode="heuristic",
                text_plan_mode="heuristic",
                ppt_plan_mode="heuristic",
                color_theme_mode="fallback",
                icon_mode="placeholder",
                visual_review_mode="heuristic",
                judge_mode="bootstrap",
                max_rounds=max_rounds,
                candidates_per_round=1,
                experience_enabled=False,
                human_gate=gate,
                **extra,
            )
        return out

    def test_the_model_accepts_and_the_person_overrules_it(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            calls: list[dict] = []

            def gate(payload: dict) -> dict:
                calls.append(dict(payload))
                if int(payload.get("round") or 0) == 0:
                    # The first round of a run with nothing before it is not a question a person can
                    # answer, and the round does not ask: only the second round's pair is real.
                    return {"answered": False, "by": "model"}
                return {"answered": True, "by": "human", "choice": "A", "note": "the older one reads better", "card": "/cards/one"}

            out = self._run(root, gate)
            self.assertTrue(calls)
            self.assertEqual("B", calls[0]["model_answer"]["choice"])
            decision = json.loads((out / "round_01" / "repair_decision.json").read_text(encoding="utf-8"))
            self.assertTrue(decision["model_accepted"])
            self.assertFalse(decision["accepted"])
            self.assertEqual("human", decision["decided_by"])
            self.assertEqual("A", decision["human_choice"])
            self.assertEqual("human_rejected", decision["reason"])
            verdict = json.loads((out / "round_01" / "human_verdict.json").read_text(encoding="utf-8"))
            self.assertEqual("A", verdict["answer"]["choice"])

    def test_a_round_nobody_answered_is_left_as_the_model_decided_it(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            out = self._run(root, lambda payload: {"answered": False, "by": "timeout", "choice": "B"})
            decision = json.loads((out / "round_01" / "repair_decision.json").read_text(encoding="utf-8"))
            self.assertTrue(decision["accepted"])
            self.assertNotIn("decided_by", decision)


class WhatTheCardShowsTests(unittest.TestCase):
    """The two things a card has to get right for a person to be able to answer it at all."""

    def test_the_before_figure_is_the_one_the_round_inherited(self) -> None:
        from rfs.workflows.paper_evolve_editable import _before_preview

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            inherited = root / "iter_134"
            inherited.mkdir()
            (inherited / "final_draft_preview.png").write_bytes(b"png")
            accepted = root / "candidate_00"
            accepted.mkdir()
            (accepted / "final_draft_preview.png").write_bytes(b"png")

            # Nothing accepted yet in this run: the inherited version is the before.
            self.assertEqual(str(inherited / "final_draft_preview.png"), _before_preview(None, inherited))
            # Once this run has accepted something, that is the before.
            self.assertEqual(str(accepted / "final_draft_preview.png"), _before_preview(accepted, inherited))
            # Neither: the question falls back to the single-figure form rather than inventing one.
            self.assertEqual("", _before_preview(None, ""))

    def test_a_person_is_asked_about_the_layer_the_round_is_on(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            seen: list[dict] = []

            def gate(payload: dict) -> dict:
                seen.append(dict(payload))
                return {"answered": False, "by": "model"}

            self_prep = "theme_planning"
            ThePersonsVerdictDecidesTheRoundTests()._run(
                root, gate, layer_plan={"working_on": self_prep, "layer_order": list(ce.STAGE_ORDER)}
            )
            self.assertTrue(seen)
            self.assertTrue(all(item["stage"] == self_prep for item in seen), [item["stage"] for item in seen])


class TheRuleReachesTheRoundAfterItTests(unittest.TestCase):
    """A sentence a person files while the run is going belongs to the round that follows it.

    Read once for the iteration, that sentence sat out the rest of the iteration -- three more rounds
    spent under an instruction the person had already replaced, which is the whole of what this
    channel was for. These tests pin the read to the round: the rule is filed from inside the first
    round's own judgement point, and the second round's question has to carry it.
    """

    RULE = "Between boxes that sit close together, draw one direct connector."

    def _run_with_a_rule_filed_mid_run(self, root: Path) -> tuple[Path, list[str], list[dict]]:
        review_dir = root / "human_review"
        prompts: list[str] = []
        filed_after: dict[str, int] = {}

        def gate(payload: dict) -> dict:
            if not filed_after:
                # The person answers the first card, and their sentence is filed while the chain is
                # still standing at that card -- which is exactly when a live person writes one.
                human_review.review_record(
                    bundle=review_dir,
                    kind="preference",
                    paper="paper_one",
                    apply_at="ppt_compilation",
                    directive=self.RULE,
                    note="filed between this round's figure and the next",
                    reviewer="test",
                )
                filed_after["prompts"] = len(prompts)
            return {"answered": False, "by": "model"}

        out = ThePersonsVerdictDecidesTheRoundTests()._run(
            root,
            gate,
            capture=prompts,
            # Three rounds, because the first round of a run with nothing before it asks nobody:
            # the rule is filed from the second round's card, and the third round is the one that
            # has to carry it.
            max_rounds=3,
            human_preferences_dir=str(review_dir),
        )
        events = [
            json.loads(line)
            for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return out, prompts, events, int(filed_after.get("prompts") or 0)

    def test_the_rule_is_read_again_for_the_round_that_follows_it(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            _, prompts, events, filed_at = self._run_with_a_rule_filed_mid_run(Path(name))
            refreshed = [item for item in events if item.get("event") == "human_preferences_refreshed"]
            self.assertTrue(refreshed, "the round after the filing has to record that it read the list again")
            self.assertTrue(all(int(item.get("round") or 0) >= 1 for item in refreshed))
            self.assertTrue(all(item.get("entry_ids") for item in refreshed))
            # The sentence reaches a seat that ran after the person filed it, and no seat that ran
            # before: read once for the iteration, the second half of this test fails.
            self.assertTrue(any(self.RULE in prompt for prompt in prompts[filed_at:]))
            self.assertTrue(all(self.RULE not in prompt for prompt in prompts[:filed_at]))
            self.assertTrue(prompts[filed_at:], "the round after the filing has to ask a seat something")


class ThePromptCarriesThePersonsInstructionTests(unittest.TestCase):
    """Which seat is told what a person asked for, and which seat is not.

    The advisory seat proposes the next move and may be told. The acceptance seat and the pairwise
    question measure what a figure is worth, and a preference in their mouth would stop being a
    measurement -- so the key is refused there, exactly the way the library's advice is.
    """

    def _preference(self) -> dict:
        return {"apply_at": "ppt_compilation", "directive": "Put the icon beside the label.", "source": "human"}

    def test_the_advisory_seat_is_told(self) -> None:
        from rfs.evaluation.direct_judges import DirectJudgeInput
        from rfs.evaluation.judge_prompt import online_judge_row, canonical_prompt

        row = online_judge_row(
            DirectJudgeInput(method={}, layout={}, icons={}, program={}, human_preferences=[self._preference()]),
            role="online_trained",
        )
        self.assertEqual(1, len(row["process_context"]["human_preferences"]))
        self.assertIn("Put the icon beside the label.", canonical_prompt(row))

    def test_the_measuring_seats_are_not(self) -> None:
        from rfs.evaluation.direct_judges import DirectJudgeInput
        from rfs.evaluation.judge_prompt import online_judge_row

        for role, pairwise in (("frozen_choice", False), ("online_trained", True)):
            row = online_judge_row(
                DirectJudgeInput(method={}, layout={}, icons={}, program={}, human_preferences=[self._preference()]),
                role=role,
                pairwise=pairwise,
            )
            self.assertNotIn("human_preferences", row["process_context"], role)

    def test_a_round_without_a_rule_renders_as_before(self) -> None:
        from rfs.evaluation.direct_judges import DirectJudgeInput
        from rfs.evaluation.judge_prompt import canonical_prompt, online_judge_row

        plain = online_judge_row(DirectJudgeInput(method={}, layout={}, icons={}, program={}), role="online_trained")
        empty = online_judge_row(
            DirectJudgeInput(method={}, layout={}, icons={}, program={}, human_preferences=[]), role="online_trained"
        )
        self.assertNotIn("human_preferences", plain["process_context"])
        self.assertEqual(canonical_prompt(plain), canonical_prompt(empty))


class RoundHumanPreferencesReadTests(unittest.TestCase):
    """The reader itself: a named folder is read fresh, and a caller with none is unchanged."""

    def test_the_folder_wins_over_the_list_the_caller_handed_in(self) -> None:
        from rfs.evolution.direct import HumanReviewStore
        from rfs.workflows.paper_evolve_editable import _round_human_preferences

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name) / "human_review"
            store = HumanReviewStore(directory)
            store.install_entry({"kind": "preference", "paper": "p", "apply_at": "ppt_compilation", "directive": "fresh one", "source": "human"}, status="candidate")
            rows = _round_human_preferences(directory, [{"directive": "stale one"}])
            self.assertEqual(["fresh one"], [row.get("directive") for row in rows])

    def test_no_folder_keeps_the_list_it_was_handed(self) -> None:
        from rfs.workflows.paper_evolve_editable import _round_human_preferences

        given = [{"directive": "the one the caller passed"}]
        self.assertEqual(given, _round_human_preferences(None, given))
        self.assertEqual(given, _round_human_preferences("", given))

    def test_a_folder_nobody_can_read_does_not_stop_the_round(self) -> None:
        from rfs.workflows.paper_evolve_editable import _round_human_preferences

        missing = Path(tempfile.gettempdir()) / "no_such_human_review_folder_20261014"
        self.assertEqual([{"directive": "kept"}], _round_human_preferences(missing, [{"directive": "kept"}]))


class TheCardIsAskedAgainAfterARestartTests(unittest.TestCase):
    """A restart inside an iteration poses the same questions again.

    The candidate is rebuilt, so the figures on the new card are not the figures the person answered
    about. The old answer has to be set aside: read as if it were fresh, it would decide a round the
    person never judged, and the round would be filed as a judgement they made.
    """

    def _card(self, tmp: str) -> Path:
        return Path(tmp) / "human_queue" / "p" / "iter_01" / "round_00" / "figure_choice"

    def test_the_answer_of_the_abandoned_attempt_is_set_aside(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clock = QueueClock()
            queue = HumanQueue(
                root=Path(tmp), seats=("figure_choice",), hold_hours=0.001, clock=clock, sleeper=clock.sleep, poll_seconds=1.0
            )
            card = self._card(tmp)
            card.mkdir(parents=True, exist_ok=True)
            write_json(card / ANSWER_FILE, {"choice": "A", "note": "about the figure of the first attempt"})

            got = queue.ask(kind="figure_choice", paper="p", iteration=1, round_index=0, question="?", model_answer={"choice": "B"})

            self.assertFalse(got.answered)
            self.assertEqual("timeout", got.by)
            self.assertFalse((card / ANSWER_FILE).is_file())
            stale = sorted(card.glob("answer.stale.*.json"))
            self.assertEqual(1, len(stale), stale)
            self.assertEqual("A", read_json(stale[0])["choice"])
            self.assertEqual(str(stale[0]), read_json(card / QUESTION_FILE)["rotated_previous_answer"])

    def test_the_answer_given_to_the_new_card_still_decides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clock = QueueClock()
            queue = HumanQueue(
                root=Path(tmp), seats=("figure_choice",), hold_hours=0.001, clock=clock, sleeper=clock.sleep, poll_seconds=1.0
            )
            card = self._card(tmp)
            card.mkdir(parents=True, exist_ok=True)
            write_json(card / ANSWER_FILE, {"choice": "A", "note": "about the figure of the first attempt"})

            def answer_it(payload: dict) -> None:
                HumanQueue(root=Path(tmp)).answer(payload["card"], choice="B", note="the figure in front of me now")

            queue.on_question = answer_it
            got = queue.ask(kind="figure_choice", paper="p", iteration=1, round_index=0, question="?", model_answer={"choice": "A"})

            self.assertTrue(got.answered)
            self.assertEqual("B", got.choice)
            self.assertEqual("the figure in front of me now", got.note)
            self.assertEqual("B", read_json(card / ANSWER_FILE)["choice"])


class ThePersonsSentenceReachesTheDrawingTests(unittest.TestCase):
    """A rule filed for a stage rides the block that stage already reads.

    Before this the sentence reached the advisory seat and the reflection and nothing that decides
    geometry -- so a figure came back carrying the detour the person had objected to, while the
    round's own account said the rule had been handed to it.
    """

    def _preference(self) -> dict:
        return {
            "entry_id": "human-pref-1",
            "review_id": "human-review-1",
            "directive": "Between boxes that sit close together, draw one direct connector.",
            "apply_at": "ppt_compilation",
            "reviewer": "<AUTHOR>",
        }

    def test_only_the_stage_it_names_is_told(self) -> None:
        from rfs.workflows.paper_evolve_editable import person_rule_patches

        rows = person_rule_patches([self._preference()], "ppt_compilation")
        self.assertEqual(1, len(rows))
        self.assertEqual("ppt_compilation", rows[0]["apply_at"])
        self.assertIn("one direct connector", rows[0]["directive"])
        self.assertEqual("person", rows[0]["source"])
        self.assertEqual([], person_rule_patches([self._preference()], "theme_planning"))

    def test_a_row_with_no_sentence_is_not_injected(self) -> None:
        from rfs.workflows.paper_evolve_editable import person_rule_patches

        self.assertEqual([], person_rule_patches([{"entry_id": "e", "apply_at": "ppt_compilation"}], "ppt_compilation"))
        self.assertEqual([], person_rule_patches(None, "ppt_compilation"))

    def test_a_rule_that_names_no_stage_applies_everywhere(self) -> None:
        from rfs.workflows.paper_evolve_editable import person_rule_patches

        row = {"entry_id": "e", "directive": "Never route a link out around the outside of a panel."}
        for stage in ("global_planning", "ppt_compilation", "repair"):
            self.assertEqual(1, len(person_rule_patches([row], stage)), stage)

    def test_the_machine_form_of_a_persons_rule_reaches_the_router(self) -> None:
        from rfs.workflows.paper_evolve_editable import person_router_rules

        row = {
            "entry_id": "e",
            "directive": "Between boxes that sit close together, draw one direct connector.",
            "apply_at": "ppt_compilation",
            "condition": {"router": {"direct_when_close": True, "stay_inside_parent": True}},
        }
        self.assertEqual(
            {"direct_when_close": True, "stay_inside_parent": True},
            person_router_rules([row], "ppt_compilation"),
        )
        self.assertEqual({}, person_router_rules([row], "theme_planning"))

    def test_a_sentence_with_no_machine_form_changes_no_router_rule(self) -> None:
        from rfs.workflows.paper_evolve_editable import person_router_rules

        row = {"entry_id": "e", "directive": "Put the icon beside the label.", "apply_at": "ppt_compilation"}
        self.assertEqual({}, person_router_rules([row], "ppt_compilation"))

    def test_the_same_sentence_filed_twice_is_one_rule(self) -> None:
        from rfs.workflows.paper_evolve_editable import person_rule_patches

        first = {"entry_id": "e1", "directive": "Close boxes get one direct connector.", "apply_at": "ppt_compilation"}
        second = {
            "entry_id": "e2",
            "directive": "  Close boxes get one DIRECT connector.  ",
            "apply_at": "ppt_compilation",
            "condition": {"router": {"direct_when_close": True}},
        }
        rows = person_rule_patches([first, second], "ppt_compilation")
        self.assertEqual(1, len(rows))
        self.assertIn("one DIRECT connector", rows[0]["directive"])
        self.assertEqual({"direct_when_close": True}, (rows[0].get("condition") or {}).get("router"))


class EveryCompileCarriesThePersonsRulesTests(unittest.TestCase):
    """Every compile in the product loop is handed the person's rules.

    Four compiles draw a figure inside one round and a round normally reaches only one of them. The
    person's sentence reached three of the four, so on the round that took the fourth the figure
    came back carrying the detour the person had just objected to -- while that round's own account
    said the sentence had been handed to the stage that draws. Reading the call sites is what
    catches the one that was missed, because no single round exercises all four.
    """

    def test_every_compile_in_the_round_is_handed_the_persons_rules(self):
        source = Path(__file__).resolve().parents[2] / "rfs" / "workflows" / "paper_evolve_editable.py"
        text = source.read_text(encoding="utf-8")
        calls: list[str] = []
        index = text.find("_compile_direct(")
        while index != -1:
            if text[max(0, index - 4):index].strip() != "def":
                calls.append(_call_arguments(text, index + len("_compile_direct(")))
            index = text.find("_compile_direct(", index + 1)
        self.assertEqual(4, len(calls), "the round's compiles moved; this guard follows them")
        self.assertEqual(
            [],
            [" ".join(call.split())[:120] for call in calls if "person_rules=" not in call],
            "a compile the person's rules never reach draws the figure without them",
        )


def _loop_with_validation(tmp: str, runner, **overrides):
    """A loop whose measurement is a stub, so a verdict exists without a paired run being paid for."""

    return ce._Loop(
        config=ce.read_continuous_config(_config(**overrides)),
        out=Path(tmp),
        library="",
        stop_file="",
        evolve_runner=lambda **options: {},
        review_runner=lambda previous, spec: {"ok": False},
        validate_runner=runner,
        clock=lambda: 0.0,
        disk_usage=_roomy_disk,
        max_waves=1,
    )


class ThePersonsWordOnWhereTheWorkIsTests(unittest.TestCase):
    """The answer a person already gave, read where the next fork is decided.

    The accusation this settles: a person answered the figure card, said the layer was not good
    enough yet, and then watched the next iteration go and polish the layer below it -- because the
    fork was taken from the seat's own review and the person's word never reached it, and because
    going back over a layer wanted the same answer twice. Now the word *is* the fork: it is the
    case for planning that layer again, and it is read without a second card or a second round.
    """

    def _verdict(self, tmp: str, *, stage: str, settled: str, answered: bool = True) -> Path:
        iter_dir = Path(tmp) / "papers" / "paper_one" / "iter_01"
        round_dir = iter_dir / "round_00"
        round_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            round_dir / "human_verdict.json",
            {
                "contract": "rfs.human_verdict",
                "version": 1,
                "round": 0,
                "kind": "figure_choice",
                "stage": stage,
                "settled_layer": settled,
                "direction_layer": "",
                "answer": {
                    "answered": answered,
                    "by": "human" if answered else "timeout",
                    "choice": "A",
                    "note": "this layer is not good enough yet",
                },
            },
        )
        return iter_dir

    def test_a_person_leaving_a_layer_open_sends_the_next_iteration_back_to_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = self._verdict(tmp, stage="content_planning", settled="")
            self.assertEqual("content_planning", ce.person_open_layer(iter_dir)["stage"])
            loop = _loop(tmp)
            got = loop.person_direction({"name": "paper_one"}, 2, model_stage="ppt_compilation")
            self.assertEqual("content_planning", got["stage"])
            self.assertEqual("person", got["by"])

    def test_the_persons_own_answer_moves_the_fork_with_no_seat_and_no_second_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._verdict(tmp, stage="content_planning", settled="")
            # No human_seat block at all: the seat is off, and the answer the person already gave
            # still names the layer -- reading it needs no second card standing open.
            loop = _loop(tmp)
            got = loop.person_direction({"name": "paper_one"}, 2, model_stage="ppt_compilation")
            self.assertEqual("content_planning", got["stage"])
            cards = list((Path(tmp) / "human_queue").glob("**/*")) if (Path(tmp) / "human_queue").exists() else []
            self.assertEqual([], cards)

    def test_a_person_sealing_the_layer_is_not_read_as_leaving_it_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._verdict(tmp, stage="content_planning", settled="content_planning")
            loop = _loop(tmp)
            got = loop.person_direction({"name": "paper_one"}, 2, model_stage="ppt_compilation", model_reason="the seat said so")
            self.assertEqual("ppt_compilation", got["stage"])
            self.assertEqual("model", got["by"])

    def test_a_card_nobody_answered_names_no_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = self._verdict(tmp, stage="content_planning", settled="", answered=False)
            self.assertEqual({}, ce.person_open_layer(iter_dir))

    def test_the_last_word_about_a_layer_is_the_one_that_stands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = self._verdict(tmp, stage="content_planning", settled="")
            later = iter_dir / "round_01"
            later.mkdir(parents=True, exist_ok=True)
            write_json(
                later / "human_verdict.json",
                {
                    "contract": "rfs.human_verdict",
                    "version": 1,
                    "round": 1,
                    "kind": "figure_choice",
                    "stage": "content_planning",
                    "settled_layer": "content_planning",
                    "direction_layer": "",
                    "answer": {"answered": True, "by": "human", "choice": "A", "note": "this is good now"},
                },
            )
            # The person first said the layer was not good enough and then called the same layer
            # finished with: only the later answer is what the person thinks now, so the next
            # iteration is not sent back to a layer they have since sealed.
            self.assertEqual({}, ce.person_open_layer(iter_dir))


class TheDirectionCardTests(unittest.TestCase):
    """The seat a person stands at when the question is which layer the next iteration works on."""

    def test_a_person_sitting_at_the_direction_seat_answers_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _loop(tmp, human_seat={"enabled": True, "seats": ["direction"], "papers": ["paper_one"]})
            asked: list[dict] = []

            def answer_it(payload: dict) -> None:
                asked.append(dict(payload))
                loop.human_queue.answer(payload["card"], choice="global_planning", note="the fault is in the plan")

            loop.human_queue.on_question = answer_it
            got = loop.person_direction({"name": "paper_one"}, 3, model_stage="ppt_compilation", model_reason="the seat said so")
            self.assertEqual("global_planning", got["stage"])
            self.assertEqual("person", got["by"])
            self.assertEqual(["direction"], [item["kind"] for item in asked])
            card = Path(tmp) / "human_queue" / "paper_one" / "iter_03" / "round_00" / "direction"
            self.assertEqual("global_planning", read_json(card / ANSWER_FILE)["choice"])
            self.assertEqual("ppt_compilation", read_json(card / MODEL_ANSWER_FILE)["choice"])
            # The card carries the question, and the seat's own answer sits beside it only after the
            # person has answered: the question the person read is the question the loop wrote.
            self.assertEqual(ce.DIRECTION_QUESTION, read_json(card / QUESTION_FILE)["question"])

    def test_a_direction_card_left_unanswered_keeps_the_seats_own_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _loop(
                tmp,
                human_seat={"enabled": True, "seats": ["direction"], "papers": ["paper_one"], "hold_hours": 0.0},
            )
            got = loop.person_direction({"name": "paper_one"}, 3, model_stage="ppt_compilation", model_reason="the seat said so")
            self.assertEqual("ppt_compilation", got["stage"])
            self.assertEqual("model", got["by"])

    def test_the_paper_the_seat_is_not_held_for_is_never_asked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _loop(tmp, human_seat={"enabled": True, "seats": ["direction"], "papers": ["paper_two"]})
            got = loop.person_direction({"name": "paper_one"}, 3, model_stage="ppt_compilation")
            self.assertEqual("ppt_compilation", got["stage"])
            self.assertFalse((Path(tmp) / "human_queue").exists())


class TheValidationSeatTests(unittest.TestCase):
    """The last seat: whether a measurement that just finished may carry a row into the library."""

    def _seat(self) -> dict:
        return {"enabled": True, "seats": ["validation"], "papers": ["paper_one"]}

    def _measured(self, loop, experience_id: str = "exp-1") -> None:
        """Pin the row, so the measurement is not dropped as a row the library no longer holds."""

        write_json(
            loop.validation_attempt_dir(experience_id) / ROW_UNDER_TEST_FILE,
            {"contract": ROW_UNDER_TEST_CONTRACT, "version": 1, "experience_id": experience_id, "experience": {}},
        )

    def test_a_verdict_that_moves_nothing_is_put_to_nobody(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _loop_with_validation(tmp, lambda **options: {"status": "overfit_or_failed"}, human_seat=self._seat())
            self._measured(loop)
            record = loop.run_validation({"experience_id": "exp-1", "paper": "paper_one", "iteration": 3})
            self.assertEqual("overfit_or_failed", record["status"])
            self.assertFalse((Path(tmp) / "human_queue").exists())

    def test_a_person_holding_a_measurement_keeps_the_row_where_it_was(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _loop_with_validation(
                tmp, lambda **options: {"status": "quick_validated", "summary": {"cases": 1}}, human_seat=self._seat()
            )
            self._measured(loop)
            asked: list[dict] = []

            def hold_it(payload: dict) -> None:
                asked.append(dict(payload))
                loop.human_queue.answer(payload["card"], choice="hold", note="this was not measured well enough")

            loop.human_queue.on_question = hold_it
            record = loop.run_validation({"experience_id": "exp-1", "paper": "paper_one", "iteration": 3})
            self.assertEqual(["validation"], [item["kind"] for item in asked])
            self.assertTrue(record["held_by_person"])
            self.assertEqual("", record["rung_awarded"])
            self.assertEqual("this was not measured well enough", record["person_note"])

    def test_a_measurement_a_person_lets_stand_is_settled_by_the_ladder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _loop_with_validation(
                tmp, lambda **options: {"status": "quick_validated", "summary": {"cases": 1}}, human_seat=self._seat()
            )
            self._measured(loop)
            loop.human_queue.on_question = lambda payload: loop.human_queue.answer(payload["card"], choice="yes")
            record = loop.run_validation({"experience_id": "exp-1", "paper": "paper_one", "iteration": 3})
            self.assertFalse(record.get("held_by_person"))
            self.assertEqual("quick_validated", record["rung_awarded"])



if __name__ == "__main__":
    unittest.main()
