"""The advisory seat can be seated with a local checkpoint instead of the served model.

Seating a checkpoint in the seat that proposes advice is what makes that advice measurable:
each round records what was advised, what the loop did with it and what the comparison said
afterwards. These tests pin the wiring, not the quality of any particular checkpoint.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rfs.contracts.evolution import ADVISORY_ROLE_FAMILY, judge_role_family
from rfs.evaluation.direct_judges import (
    ADVISORY_LOCALISATION_RETRY,
    ADVISORY_MAX_NEW_TOKENS,
    CheckpointAdvisoryJudge,
    ChoiceAcceptanceJudge,
    DirectJudgeInput,
    build_direct_judges,
    local_judge_generator,
)
from rfs.workflows.continuous_evolution import read_continuous_config
from rfs.workflows.evolution_session import EVOLVE_OPTION_KEYS

ADVICE = {
    "status": "fault",
    "applicability": "common",
    "stage_feedback": [
        {
            "scope_type": "global",
            "problem": "the arrows leaving the encoder cross the panel border",
            "evidence": ["rel_12 leaves its panel"],
            "expected_effect": "the arrows stay inside the panel they belong to",
            "requested_actions": [{"action": "relayout_global", "factor": 1.08, "dx": 0.02}],
        }
    ],
}

REFLECTION = {
    "outcome": "improved",
    "failure_type": "none",
    "why_changed": "the arrows were re-routed and the comparison preferred the result",
    "scope_assessment": "appropriate",
    "next_policy": "keep",
    "rule": {
        "type": "router_constraint",
        "constraints": {"shared_vertical_channel": True, "forbid_crossing": True},
    },
    "prompt_patch": {
        "apply_at": "ppt_compilation",
        "condition": {"defect_patterns": ["cross_panel"]},
        "directive": "Run a connector down a channel shared with its neighbours.",
    },
}


def _input(preview=None):
    return DirectJudgeInput(method={"title": "paper"}, layout={"nodes": []}, icons={}, program={}, preview=preview)


def _png(root):
    path = Path(root) / "figure.png"
    path.write_bytes(b"png")
    return path


class CheckpointAdvisorySeatTests(unittest.TestCase):
    def test_a_seated_checkpoint_advises_in_the_advisory_family(self):
        seen = []

        def ask(prompt, images):
            seen.append((prompt, images))
            return copy.deepcopy(ADVICE)

        advisory, frozen = build_direct_judges(
            mode="bootstrap",
            online_model="served-name",
            frozen_model="served-name",
            advisory_checkpoint="/checkpoints/local-step",
            advisory_ask=ask,
        )
        result = advisory.evaluate(_input())

        self.assertEqual("online_trained", advisory.role)
        self.assertEqual(ADVISORY_ROLE_FAMILY, judge_role_family(advisory.role))
        self.assertTrue(result["stage_feedback"])
        self.assertEqual([{"action": "relayout_global", "factor": 1.08, "dx": 0.02}], result["repair"])
        self.assertEqual("common", result["applicability"])
        self.assertFalse(result["gate_only"])
        # The prompt the seat sends is the advisory one, which carries the action catalog.
        self.assertIn("relayout_global", seen[0][0])
        # The other seat is untouched by this decision.
        self.assertEqual("frozen_bootstrap", frozen.role)

    def test_the_seated_seat_never_reaches_for_the_endpoint(self):
        def ask(prompt, images):
            return copy.deepcopy(ADVICE)

        with mock.patch("rfs.evaluation.direct_judges.call_vlm_json", side_effect=AssertionError("the endpoint must not be called")):
            advisory, _frozen = build_direct_judges(mode="bootstrap", online_model="served-name", advisory_ask=ask)
            result = advisory.evaluate(_input())

        self.assertTrue(result["stage_feedback"])

    def test_a_named_checkpoint_and_an_injected_runner_skip_the_load_entirely(self):
        # The two arguments answer different questions: ``advisory_checkpoint`` says which
        # weights the seat is, ``advisory_ask`` says how they are run. A caller that supplies
        # the runner has already loaded them, so nothing is loaded a second time here.
        def ask(prompt, images):
            return copy.deepcopy(ADVICE)

        with mock.patch("rfs.evaluation.direct_judges.load_local_judge_backend", side_effect=AssertionError("nothing should be loaded")):
            advisory, _frozen = build_direct_judges(mode="bootstrap", online_model="served-name", advisory_checkpoint="/checkpoints/local-step", advisory_ask=ask)

        self.assertEqual("online_trained", advisory.role)
        self.assertEqual("", advisory._load_error)

    def test_the_seat_asks_about_the_preview_it_was_given(self):
        with tempfile.TemporaryDirectory() as temp:
            preview = _png(temp)
            seen = []

            def ask(prompt, images):
                seen.append(images)
                return copy.deepcopy(ADVICE)

            judge = CheckpointAdvisoryJudge(ask_json=ask)
            judge.evaluate(_input(preview=preview))

        self.assertEqual([[preview]], seen)

    def test_a_checkpoint_that_cannot_load_says_so_instead_of_serving(self):
        advisory, _frozen = build_direct_judges(mode="bootstrap", online_model="served-name", advisory_checkpoint="/no/such/checkpoint", advisory_base_model="/no/such/base")
        result = advisory.evaluate(_input())

        self.assertTrue(advisory._load_error)
        self.assertTrue(result["advisory_unavailable"])
        self.assertTrue(result["gate_only"])
        self.assertEqual([], result["stage_feedback"])
        self.assertEqual([], result["repair"])

    def test_a_reflection_from_a_seated_checkpoint_is_the_local_answer(self):
        calls = []

        def ask(prompt, images):
            calls.append((prompt, images))
            return dict(REFLECTION)

        judge = CheckpointAdvisoryJudge(ask_json=ask)
        out = judge.reflect({"decision": {"accepted": True}})

        self.assertEqual(1, len(calls))
        self.assertEqual([], calls[0][1])
        self.assertEqual("improved", out["outcome"])
        self.assertEqual("keep", out["next_policy"])

    def test_a_reflection_that_failed_is_recorded_rather_than_raised(self):
        def ask(prompt, images):
            raise RuntimeError("the checkpoint said nothing usable")

        out = CheckpointAdvisoryJudge(ask_json=ask).reflect({"decision": {"accepted": False}})

        self.assertIn("provider_error", out)
        self.assertEqual("regressed", out["outcome"])

    def test_a_heuristic_run_can_still_seat_a_checkpoint_in_the_advisory_seat(self):
        advisory, frozen = build_direct_judges(
            mode="heuristic",
            advisory_checkpoint="/checkpoints/local-step",
            advisory_ask=lambda prompt, images: copy.deepcopy(ADVICE),
        )

        self.assertEqual("online_trained", advisory.role)
        self.assertEqual("frozen_heuristic", frozen.role)
        self.assertTrue(advisory.evaluate(_input())["stage_feedback"])


class LocalGeneratorTests(unittest.TestCase):
    def test_the_local_arm_builds_the_record_the_backend_reads(self):
        class _Backend:
            def __init__(self):
                self.records = []
                self.budgets = []

            def generate(self, record, max_new_tokens=None):
                self.records.append(record)
                self.budgets.append(max_new_tokens)
                return json.dumps({"preferred_candidate": "B", "defect": "overlap"})

        with tempfile.TemporaryDirectory() as temp:
            preview = _png(temp)
            backend = _Backend()
            raw = local_judge_generator(backend, max_new_tokens=64)("ask about this figure", [preview, "/missing.png"])

        self.assertEqual({"preferred_candidate": "B", "defect": "overlap"}, raw)
        content = backend.records[0]["messages"][0]["content"]
        self.assertEqual({"type": "text", "text": "ask about this figure"}, content[0])
        self.assertEqual([{"type": "image", "image": str(preview)}], content[1:])
        self.assertEqual([str(preview)], backend.records[0]["images"])
        self.assertEqual([64], backend.budgets)


class SeatOptionWiringTests(unittest.TestCase):
    def test_the_two_seat_options_travel_with_every_station(self):
        self.assertIn("online_judge_checkpoint", EVOLVE_OPTION_KEYS)
        self.assertIn("online_judge_base_model", EVOLVE_OPTION_KEYS)

    def test_the_structure_switches_travel_with_every_station_too(self):
        """A round has to be able to say which structure switch it ran under."""

        config = read_continuous_config(
            {
                "continuous": {
                    "base_papers": [{"paper": "/p/one.pdf", "paper_id": "one"}],
                    "evolve": {
                        "middle_layer": "off",
                        "figure_target": "paper",
                        "graph_context": False,
                    },
                }
            }
        )

        for key in ("middle_layer", "figure_target", "graph_context"):
            self.assertIn(key, EVOLVE_OPTION_KEYS, key)
        self.assertEqual("off", config["evolve"]["middle_layer"])
        self.assertEqual("paper", config["evolve"]["figure_target"])
        self.assertIs(False, config["evolve"]["graph_context"])

    def test_a_continuous_config_can_name_the_advisory_checkpoint(self):
        config = read_continuous_config(
            {
                "continuous": {
                    "base_papers": [{"paper": "/p/one.pdf", "paper_id": "one"}],
                    "evolve": {
                        "judge_mode": "vlm",
                        "online_judge_checkpoint": "/checkpoints/step-000001",
                        "online_judge_base_model": "/models/Qwen3.5-9B",
                    },
                }
            }
        )

        self.assertEqual("/checkpoints/step-000001", config["evolve"]["online_judge_checkpoint"])
        self.assertEqual("/models/Qwen3.5-9B", config["evolve"]["online_judge_base_model"])


class AcceptanceSeatUnchangedTests(unittest.TestCase):
    def test_the_acceptance_seat_still_names_its_own_missing_checkpoint(self):
        judge = ChoiceAcceptanceJudge()

        self.assertEqual("no acceptance checkpoint was given", judge._load_error)
        self.assertEqual("frozen_choice", judge.role)


class UnreadableAdvisoryAnswerTests(unittest.TestCase):
    """A seat that cannot be read must not be able to end the iteration that asked it.

    Measured on the live run: the pair question produced an answer long enough to be cut off
    inside its object, the parse raised, and a whole iteration -- figure, review and hours --
    was lost to a judge that simply had nothing readable to say. The round has to record that
    state instead, and it has to keep the deterministic findings it already had.
    """

    def test_an_unreadable_answer_is_recorded_instead_of_raised(self):
        def ask(prompt, images):
            raise json.JSONDecodeError("Unterminated string starting at", "{", 4510)

        advisory = CheckpointAdvisoryJudge(checkpoint="/checkpoints/local-step", base_model="/models/base", ask_json=ask)
        result = advisory.evaluate(_input())

        self.assertIn("could not be read", result["advisory_unavailable"])
        self.assertIn("Unterminated string", result["advisory_unavailable"])
        self.assertTrue(result["gate_only"])
        self.assertEqual([], result["stage_feedback"])
        self.assertEqual("online_trained", result["judge_role"])

    def test_the_round_keeps_the_gate_findings_when_the_seat_cannot_be_read(self):
        def ask(prompt, images):
            raise RuntimeError("CUDA out of memory")

        advisory = CheckpointAdvisoryJudge(checkpoint="/checkpoints/local-step", base_model="/models/base", ask_json=ask)
        result = advisory.evaluate(_input())

        self.assertIn("CUDA out of memory", result["advisory_unavailable"])
        self.assertIn("deterministic_blockers", result)

    def test_a_reflection_that_cannot_be_read_still_returns_a_reflection(self):
        def ask(prompt, images):
            raise json.JSONDecodeError("Unterminated string starting at", "{", 12)

        advisory = CheckpointAdvisoryJudge(checkpoint="/checkpoints/local-step", base_model="/models/base", ask_json=ask)
        reflection = advisory.reflect({"decision": {"accepted": False, "reason": "new_blocker"}})

        self.assertEqual("regressed", reflection["outcome"])
        self.assertEqual("new_blocker", reflection["failure_type"])


class AdvisoryRetryTests(unittest.TestCase):
    """A cut-off answer is retried once, and both failures are reported when neither is readable."""

    def test_a_cut_off_answer_is_asked_for_again_and_the_second_one_is_used(self):
        calls = []

        def ask(prompt, images):
            calls.append(prompt)
            if len(calls) == 1:
                raise json.JSONDecodeError("Unterminated string starting at", "{", 4510)
            return copy.deepcopy(ADVICE)

        advisory = CheckpointAdvisoryJudge(checkpoint="/checkpoints/local-step", base_model="/models/base", ask_json=ask)
        result = advisory.evaluate(_input())

        self.assertEqual(2, len(calls))
        self.assertIn("cut off before it finished", calls[1])
        self.assertTrue(result["stage_feedback"])
        self.assertTrue(result["advisory_parse_retry"])
        self.assertFalse(result["gate_only"])

    def test_two_unreadable_answers_report_both_reasons(self):
        def ask(prompt, images):
            raise json.JSONDecodeError("Expecting ',' delimiter", "{", 6001)

        advisory = CheckpointAdvisoryJudge(checkpoint="/checkpoints/local-step", base_model="/models/base", ask_json=ask)
        result = advisory.evaluate(_input())

        self.assertIn("Expecting ',' delimiter", result["advisory_unavailable"])
        self.assertIn("retry:", result["advisory_unavailable"])
        self.assertTrue(result["advisory_parse_retry"])
        self.assertTrue(result["gate_only"])


class AdvisoryLocalisationHookTests(unittest.TestCase):
    """The seat renders one prompt, and a question asked again is that prompt plus one line.

    The boundary review reads the answer for the stage a fault entered and asks again when the
    answer named no stage. The re-ask is the same question with the three fields spelled out, so
    the seat owns the text and the caller owns the half that was missing -- and a call that carries
    no instruction renders the prompt byte for byte as it did before.
    """

    def _seat(self, seen):
        def ask(prompt, images):
            seen.append(prompt)
            return copy.deepcopy(ADVICE)

        return CheckpointAdvisoryJudge(checkpoint="/checkpoints/local-step", base_model="/models/base", ask_json=ask)

    def test_no_instruction_renders_the_prompt_byte_for_byte(self):
        plain: list = []
        empty: list = []
        self._seat(plain).evaluate(_input())
        self._seat(empty).evaluate(_input(), extra_instruction="")
        self.assertEqual(plain, empty)

    def test_an_instruction_is_appended_after_the_prompt(self):
        seen: list = []
        self._seat(seen).evaluate(_input(), extra_instruction=ADVISORY_LOCALISATION_RETRY)
        plain: list = []
        self._seat(plain).evaluate(_input())
        self.assertEqual(plain[0] + ADVISORY_LOCALISATION_RETRY, seen[0])

    def test_the_answer_carries_the_fields_the_prompt_showed(self):
        result = self._seat([]).evaluate(_input())
        shown = result["advisory_shown_fields"]
        self.assertEqual(sorted(shown), shown)
        self.assertIn("method", shown)
        self.assertIn("process_context", shown)

    def test_a_missing_checkpoint_says_which_silence_it_was(self):
        advisory = CheckpointAdvisoryJudge(checkpoint="/no/such/checkpoint", base_model="/no/such/base")
        result = advisory.evaluate(_input())
        self.assertEqual("no_backend", result["advisory_unavailable_kind"])

    def test_an_unreadable_answer_says_which_silence_it_was(self):
        def ask(prompt, images):
            raise json.JSONDecodeError("Unterminated string starting at", "{", 12)

        advisory = CheckpointAdvisoryJudge(checkpoint="/checkpoints/local-step", base_model="/models/base", ask_json=ask)
        result = advisory.evaluate(_input())
        self.assertEqual("unreadable", result["advisory_unavailable_kind"])


class AdvisoryBudgetTests(unittest.TestCase):
    def test_the_advisory_seat_generates_with_the_advisory_budget(self):
        class _Backend:
            def __init__(self):
                self.budgets = []

            def generate(self, record, max_new_tokens=None):
                self.budgets.append(max_new_tokens)
                return "{}"

        backend = _Backend()
        with mock.patch("rfs.evaluation.direct_judges.load_local_judge_backend", return_value=(backend, "")):
            advisory = CheckpointAdvisoryJudge(checkpoint="/checkpoints/local-step", base_model="/models/base")
            advisory._ask_json("ask about this figure", [])

        self.assertEqual([ADVISORY_MAX_NEW_TOKENS], backend.budgets)
        self.assertGreater(ADVISORY_MAX_NEW_TOKENS, 1024)


class AdvisoryCheckpointCommandLineTests(unittest.TestCase):
    """The product command can seat a checkpoint in the advisory seat, like the acceptance one.

    A batch that seats only the acceptance seat is measured by the trained weights while it is
    advised by whatever the endpoint serves, and no flag said so; the ablation and the live loop
    both need one model in both seats.
    """

    def test_the_product_command_seats_an_advisory_checkpoint(self):
        from rfs.cli import build_parser

        args = build_parser().parse_args([
            "paper-evolve-editable", "--paper", "p.pdf", "--out", "out",
            "--advisory-checkpoint", "/checkpoints/local-step",
            "--advisory-base-model", "/models/base",
        ])

        self.assertEqual(args.online_judge_checkpoint, "/checkpoints/local-step")
        self.assertEqual(args.online_judge_base_model, "/models/base")

    def test_the_legacy_online_flag_names_the_same_dest(self):
        from rfs.cli import build_parser

        args = build_parser().parse_args([
            "paper-evolve-editable", "--paper", "p.pdf", "--out", "out",
            "--online-judge-checkpoint", "/checkpoints/local-step",
            "--online-judge-base-model", "/models/base",
        ])

        self.assertEqual(args.online_judge_checkpoint, "/checkpoints/local-step")
        self.assertEqual(args.online_judge_base_model, "/models/base")

    def test_a_run_that_seats_neither_leaves_both_seats_empty(self):
        from rfs.cli import build_parser

        args = build_parser().parse_args(["paper-evolve-editable", "--paper", "p.pdf", "--out", "out"])

        self.assertEqual(args.online_judge_checkpoint, "")
        self.assertEqual(args.online_judge_base_model, "")

    def test_the_command_hands_both_seats_to_the_workflow(self):
        from rfs import cli

        args = cli.build_parser().parse_args([
            "paper-evolve-editable", "--paper", "p.pdf", "--out", "out",
            "--acceptance-checkpoint", "/checkpoints/local-step",
            "--acceptance-base-model", "/models/base",
            "--advisory-checkpoint", "/checkpoints/local-step",
            "--advisory-base-model", "/models/base",
        ])
        with mock.patch("rfs.cli.run_paper_evolve_editable", return_value={}) as runner:
            cli._run_direct(args)

        self.assertEqual(runner.call_args.kwargs["acceptance_checkpoint"], "/checkpoints/local-step")
        self.assertEqual(runner.call_args.kwargs["online_judge_checkpoint"], "/checkpoints/local-step")
        self.assertEqual(runner.call_args.kwargs["online_judge_base_model"], "/models/base")

if __name__ == "__main__":
    unittest.main()
