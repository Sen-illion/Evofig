"""A reflection states outcome and defect without contradicting itself."""

from __future__ import annotations

import copy
import unittest

from rfs.contracts.evolution import REFLECTION_CONTRACT_VERSION, normalize_reflection, reflection_conflict
from rfs.evaluation.direct_judges import build_direct_judges
from rfs.judge_training.prompts import REFLECTION_PROMPT


def _advisory(answers):
    seen: list[str] = []
    pending = [copy.deepcopy(item) for item in answers]

    def ask(prompt, images):
        seen.append(prompt)
        return pending.pop(0) if pending else {}

    advisory, _ = build_direct_judges(
        mode="bootstrap",
        online_model="served-name",
        frozen_model="served-name",
        advisory_checkpoint="/checkpoints/local-step",
        advisory_ask=ask,
    )
    return advisory, seen


def _outcome():
    return {"decision": {"accepted": True, "reason": "preferred"}, "before": {}, "after": {}}


#: The mechanism-level rule a well-formed answer carries. The question asks for one whenever the
#: round taught something about how figures are drawn, so an answer without it is a missing answer
#: and is asked for again; the tests below that count asks therefore bring their own rule.
RULE = {"type": "layout_constraint", "params": {"align_grid": 0.05}}

#: The patch a well-formed answer carries beside its rule. The question asks for both whenever the
#: round taught something, so the tests below that count asks bring both unless they are about one
#: of the two being missing.
PATCH = {
    "apply_at": "ppt_compilation",
    "condition": {"defect_patterns": ["cross_panel"]},
    "directive": "Run a connector down a channel shared with its neighbours.",
}


class ReflectionContractV11Tests(unittest.TestCase):
    def test_the_reflection_contract_is_one_version_ahead_of_the_shared_one(self):
        reflection = normalize_reflection({"outcome": "improved"})
        self.assertEqual("rfs.judge_reflection", reflection["contract"])
        # 1.2 adds the three fields a losing round uses to say whether the rule it ran was wrong or
        # was carried out badly, and what to do about the second.
        self.assertEqual("1.2", REFLECTION_CONTRACT_VERSION)
        self.assertEqual(REFLECTION_CONTRACT_VERSION, reflection["version"])

    def test_the_reflection_carries_the_patch_the_seat_wrote(self):
        # The patch's wording is the seat's answer; whether any prompt ever sees it is decided by
        # the measurement behind the row, so the reflection only has to carry it faithfully.
        reflection = normalize_reflection(
            {
                "outcome": "regressed",
                "prompt_patch": {
                    "apply_at": "ppt_compilation",
                    "condition": {"defect_patterns": ["cross_panel"]},
                    "directive": "Run the connector down a shared channel.",
                    "something_else": 1,
                },
            }
        )

        self.assertEqual(
            {
                "apply_at": "ppt_compilation",
                "condition": {"defect_patterns": ["cross_panel"]},
                "directive": "Run the connector down a shared channel.",
            },
            reflection["prompt_patch"],
        )

    def test_a_reflection_that_wrote_no_patch_carries_an_empty_one(self):
        reflection = normalize_reflection({"outcome": "improved"})

        self.assertEqual({"apply_at": "", "condition": {}, "directive": ""}, reflection["prompt_patch"])
        self.assertEqual({"apply_at": "", "condition": {}, "directive": ""}, normalize_reflection({"prompt_patch": "nope"})["prompt_patch"])

    def test_the_reflection_prompt_asks_for_the_patch_and_says_what_it_is_for(self):
        self.assertIn("prompt_patch", REFLECTION_PROMPT)
        self.assertIn("injected into a later prompt on the strength of the measurement", REFLECTION_PROMPT)
        # The sentence has to come from somewhere, and where it comes from is the review's own
        # attribution: the block the round hands the reflection when its answer named a fault.
        self.assertIn("fault_attribution", REFLECTION_PROMPT)

    def test_a_row_written_before_the_split_reads_as_an_empty_defect_pair(self):
        legacy = normalize_reflection({"outcome": "regressed", "failure_type": "global_regression", "why_changed": "x"})
        self.assertEqual({"cleared": [], "introduced": []}, legacy["defects"])
        self.assertEqual("regressed", legacy["outcome"])
        self.assertEqual("global_regression", legacy["failure_type"])
        self.assertEqual("", legacy["reflection_conflict"])

    def test_an_improvement_that_also_names_a_failure_is_a_contradiction(self):
        reflection = normalize_reflection({"outcome": "improved", "failure_type": "global_regression"})
        self.assertEqual("outcome is improved but failure_type is global_regression", reflection["reflection_conflict"])

    def test_an_improvement_that_also_introduces_a_defect_is_a_contradiction(self):
        reflection = normalize_reflection(
            {"outcome": "improved", "failure_type": "none", "defects": {"cleared": ["overlap"], "introduced": ["crossing"]}}
        )
        self.assertEqual("outcome is improved but defects.introduced is not empty", reflection["reflection_conflict"])

    def test_a_consistent_row_carries_no_conflict(self):
        self.assertEqual(
            "",
            reflection_conflict({"outcome": "improved", "failure_type": "none", "defects": {"cleared": ["overlap"], "introduced": []}}),
        )
        self.assertEqual(
            "",
            reflection_conflict({"outcome": "regressed", "failure_type": "global_regression", "defects": {"cleared": [], "introduced": ["crossing"]}}),
        )

    def test_the_prompt_asks_for_defects_and_keeps_the_two_statements_apart(self):
        self.assertIn("defects", REFLECTION_PROMPT)
        self.assertIn("defects.cleared", REFLECTION_PROMPT)
        self.assertIn("defects.introduced", REFLECTION_PROMPT)
        self.assertIn("failure_type is only meaningful when the round did not improve", REFLECTION_PROMPT)
        # The examples are written in the vocabulary the compiler reads, not in prose.
        self.assertIn('"shared_vertical_channel": true', REFLECTION_PROMPT)
        self.assertIn('"gutter_min": 0.02', REFLECTION_PROMPT)


    def test_an_improvement_that_also_calls_the_measurement_noise_is_a_contradiction(self):
        reflection = normalize_reflection({"outcome": "improved", "failure_type": "none", "measurement_noise_risk": True})
        self.assertEqual("outcome is improved but measurement_noise_risk is set", reflection["reflection_conflict"])

    def test_a_stochastic_drawing_is_not_a_contradiction_with_an_improvement(self):
        reflection = normalize_reflection({"outcome": "improved", "failure_type": "none", "randomness_risk": True})
        self.assertEqual("", reflection["reflection_conflict"])

    def test_a_doubted_measurement_beside_a_regression_is_consistent(self):
        reflection = normalize_reflection({"outcome": "regressed", "failure_type": "none", "measurement_noise_risk": True})
        self.assertEqual("", reflection["reflection_conflict"])


class ReflectionConflictRetryTests(unittest.TestCase):
    def test_a_contradictory_reflection_is_asked_once_more_and_the_second_answer_is_kept(self):
        advisory, seen = _advisory(
            [
                {"outcome": "improved", "failure_type": "global_regression", "why_changed": "a", "rule": RULE, "prompt_patch": PATCH},
                {
                    "outcome": "regressed",
                    "failure_type": "global_regression",
                    "defects": {"cleared": [], "introduced": ["crossing"]},
                    "why_changed": "b",
                    "rule": RULE,
                    "prompt_patch": PATCH,
                },
            ]
        )
        reflection = advisory.reflect(_outcome())
        self.assertEqual(2, len(seen))
        self.assertIn("contradicted itself", seen[1])
        self.assertEqual("regressed", reflection["outcome"])
        self.assertEqual("", reflection["reflection_conflict"])
        self.assertTrue(reflection["reflection_parse_retry"])

    def test_a_row_that_contradicts_itself_twice_is_kept_and_named(self):
        advisory, seen = _advisory(
            [
                {"outcome": "improved", "failure_type": "global_regression", "rule": RULE, "prompt_patch": PATCH},
                {"outcome": "improved", "failure_type": "new_blocker", "rule": RULE, "prompt_patch": PATCH},
            ]
        )
        reflection = advisory.reflect(_outcome())
        self.assertEqual(2, len(seen))
        self.assertEqual("outcome is improved but failure_type is new_blocker", reflection["reflection_conflict"])

    def test_a_consistent_reflection_is_not_asked_twice(self):
        advisory, seen = _advisory([{"outcome": "improved", "failure_type": "none", "next_policy": "keep", "rule": RULE, "prompt_patch": PATCH}])
        reflection = advisory.reflect(_outcome())
        self.assertEqual(1, len(seen))
        self.assertEqual("", reflection["reflection_conflict"])


    def test_a_doubted_measurement_beside_an_improvement_is_asked_once_more(self):
        advisory, seen = _advisory(
            [
                {"outcome": "improved", "failure_type": "none", "measurement_noise_risk": True, "rule": RULE, "prompt_patch": PATCH},
                {"outcome": "improved", "failure_type": "none", "measurement_noise_risk": False, "next_policy": "keep", "rule": RULE, "prompt_patch": PATCH},
            ]
        )
        reflection = advisory.reflect(_outcome())
        self.assertEqual(2, len(seen))
        self.assertIn("cannot be trusted", seen[1])
        self.assertFalse(reflection["measurement_noise_risk"])
        self.assertEqual("", reflection["reflection_conflict"])

    def test_a_regression_beside_a_doubted_measurement_is_not_asked_again(self):
        advisory, seen = _advisory(
            [{"outcome": "regressed", "failure_type": "none", "measurement_noise_risk": True, "next_policy": "keep", "rule": RULE, "prompt_patch": PATCH}]
        )
        advisory.reflect(_outcome())
        self.assertEqual(1, len(seen))


class ReflectionRuleMissingRetryTests(unittest.TestCase):
    """A round that drew or moved a figure and stated no rule is asked once more for one."""

    def test_a_reflection_that_states_no_rule_is_asked_once_more(self):
        advisory, seen = _advisory(
            [
                {"outcome": "regressed", "failure_type": "none", "next_policy": "narrow_scope"},
                {"outcome": "regressed", "failure_type": "none", "next_policy": "revise_rule", "rule": RULE, "prompt_patch": PATCH},
            ]
        )
        reflection = advisory.reflect(_outcome())
        self.assertEqual(2, len(seen))
        self.assertIn("stated no rule", seen[1])
        self.assertEqual(RULE["type"], reflection["rule"]["type"])
        self.assertNotIn("rule_missing", reflection)

    def test_a_second_answer_that_states_no_rule_is_kept_and_named(self):
        # The retry is a question, not a fill-in: a seat that answers nothing twice is recorded as
        # having taught nothing, rather than handed a rule it never wrote.
        advisory, seen = _advisory(
            [
                {"outcome": "regressed", "failure_type": "none", "next_policy": "narrow_scope"},
                {"outcome": "regressed", "failure_type": "none", "next_policy": "narrow_scope"},
            ]
        )
        reflection = advisory.reflect(_outcome())
        self.assertEqual(2, len(seen))
        self.assertEqual("regressed", reflection["outcome"])
        self.assertTrue(reflection["rule_missing"])

    def test_a_silent_second_answer_leaves_the_first_standing(self):
        # Only an answer that arrived replaces the one before it: a seat that says nothing the
        # second time has not taken back what it already said.
        advisory, seen = _advisory([{"outcome": "regressed", "failure_type": "none", "next_policy": "narrow_scope"}])
        reflection = advisory.reflect(_outcome())
        self.assertEqual(2, len(seen))
        self.assertEqual("regressed", reflection["outcome"])
        self.assertEqual("narrow_scope", reflection["next_policy"])
        self.assertTrue(reflection["rule_missing"])


class ReflectionPatchMissingRetryTests(unittest.TestCase):
    """A round that named a fault and wrote no patch is asked once more for one."""

    def test_a_reflection_that_states_a_rule_but_no_patch_is_asked_once_more(self):
        advisory, seen = _advisory(
            [
                {"outcome": "regressed", "failure_type": "none", "next_policy": "revise_rule", "rule": RULE},
                {"outcome": "regressed", "failure_type": "none", "next_policy": "revise_rule", "rule": RULE, "prompt_patch": PATCH},
            ]
        )
        reflection = advisory.reflect(_outcome())

        self.assertEqual(2, len(seen))
        self.assertIn("wrote no prompt_patch", seen[1])
        self.assertEqual(PATCH["directive"], reflection["prompt_patch"]["directive"])
        self.assertEqual(PATCH["apply_at"], reflection["prompt_patch"]["apply_at"])
        self.assertNotIn("prompt_patch_missing", reflection)

    def test_a_second_answer_that_wrote_no_patch_is_kept_and_named(self):
        # The retry is a question, not a fill-in: a seat that answers nothing twice is recorded as
        # having written no patch rather than handed the sentence it never wrote.
        advisory, seen = _advisory(
            [
                {"outcome": "regressed", "failure_type": "none", "next_policy": "revise_rule", "rule": RULE},
                {"outcome": "regressed", "failure_type": "none", "next_policy": "revise_rule", "rule": RULE},
            ]
        )
        reflection = advisory.reflect(_outcome())

        self.assertEqual(2, len(seen))
        self.assertTrue(reflection["prompt_patch_missing"])
        self.assertEqual("", reflection["prompt_patch"]["directive"])

    def test_the_two_missing_fields_share_one_retry(self):
        # One round, one second ask: a seat that left out both is not questioned twice for it.
        advisory, seen = _advisory([{"outcome": "regressed", "failure_type": "none", "next_policy": "narrow_scope"}])

        reflection = advisory.reflect(_outcome())

        self.assertEqual(2, len(seen))
        self.assertIn("stated no rule", seen[1])
        self.assertIn("wrote no prompt_patch", seen[1])
        self.assertTrue(reflection["rule_missing"])
        self.assertTrue(reflection["prompt_patch_missing"])

    def test_the_retry_names_the_fields_the_patch_carries(self):
        advisory, seen = _advisory(
            [
                {"outcome": "regressed", "failure_type": "none", "next_policy": "revise_rule", "rule": RULE},
                {"outcome": "regressed", "failure_type": "none", "next_policy": "revise_rule", "rule": RULE, "prompt_patch": PATCH},
            ]
        )

        advisory.reflect(_outcome())

        self.assertIn("apply_at", seen[1])
        self.assertIn("directive", seen[1])
        self.assertIn("condition", seen[1])


if __name__ == "__main__":
    unittest.main()
