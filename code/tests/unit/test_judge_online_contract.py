"""The online Judge and Judge training must speak one prompt and one contract."""

import json
import unittest
from unittest.mock import patch

from rfs.contracts.evolution import MECHANISM_ACTIONS, REPAIR_ACTIONS, normalize_judge_result, validate_judge_result
from rfs.evaluation.direct_judges import DirectJudgeInput, build_direct_judges
from rfs.evaluation.judge_prompt import canonical_pairwise_prompt, canonical_prompt, canonical_reflection_prompt, method_contract, online_judge_row, prompt_version
from rfs.judge_training.prompts import ACCEPTANCE_OUTPUT_SCHEMA, ADVISORY_OUTPUT_SCHEMA, PROMPT_VERSION, REFLECTION_PROMPT, build_prompt, judgement_target


def _judge_input(**overrides) -> DirectJudgeInput:
    """The smallest candidate the deterministic gate accepts, so the model path is reached."""
    values = dict(
        method={"modules": [{"id": "n"}], "relations": []},
        layout={"panels": [{"id": "p", "bbox_percent": {"x": 0, "y": 0, "w": 1, "h": 1}}], "nodes": [{"id": "n", "panel_id": "p", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6}}], "connectors": []},
        icons={"icons": [{"asset_id": "icon_n", "semantic_node_ids": ["n"], "prompt": "n"}]},
        program={"panels": [{"id": "p", "bbox_percent": {"x": 0, "y": 0, "w": 1, "h": 1}}], "semantic_nodes": [{"id": "n", "panel_id": "p", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6}}], "slots": [{"id": "slot_n", "asset_id": "icon_n", "bbox_percent": {"x": 0.15, "y": 0.15, "w": 0.3, "h": 0.3}}], "text_program": {"items": [{"id": "t", "text": "n"}]}, "arrows": []},
    )
    values.update(overrides)
    return DirectJudgeInput(**values)


def _method_contract() -> dict:
    return {
        "figure_goal": "Rebuild a scene from one image.",
        "modules": [{"id": "enc", "label": "Encoder", "must_show": True}, {"id": "dec", "label": "Decoder", "must_show": False}],
        "relations": [{"source": "enc", "target": "dec", "type": "data_flow"}],
    }


class _PassingGate:
    """Stub for the deterministic gate, which has its own tests, so the model path runs."""

    def evaluate(self, judge_input):
        return {"status": "pass"}


class OnlineJudgeContractTests(unittest.TestCase):
    def test_the_online_judge_renders_through_the_training_renderer(self):
        row = online_judge_row(_judge_input(), role="online_bootstrap")
        self.assertEqual(build_prompt(row), canonical_prompt(row))

    def test_the_prompt_asks_for_the_contract_the_loop_consumes(self):
        prompt = canonical_prompt(online_judge_row(_judge_input(), role="online_bootstrap"))
        for field in ("stage_feedback", "requested_actions", "forbidden_changes", "scope_type"):
            self.assertIn(field, prompt)
        self.assertIn("reroute", prompt)

    def test_the_advisory_prompt_never_asks_for_a_score(self):
        prompt = canonical_prompt(online_judge_row(_judge_input(), role="online_bootstrap"))
        for absent in ("overall_score", "dimension_scores"):
            self.assertNotIn(absent, prompt)

    def test_the_acceptance_prompt_asks_for_a_score_and_no_repair(self):
        prompt = canonical_prompt(online_judge_row(_judge_input(), role="frozen_bootstrap"))
        self.assertIn("overall_score", prompt)
        self.assertIn("blocking_issues", prompt)
        for absent in ("stage_feedback", "requested_actions", "reroute"):
            self.assertNotIn(absent, prompt)

    def test_the_progress_sheet_travels_to_the_advisory_seat_alone(self):
        sheet = {"contract": "rfs.progress_sheet", "version": 1, "now_working_on": "segmentation", "layers": []}
        without = canonical_prompt(online_judge_row(_judge_input(), role="online_bootstrap"))
        with_sheet = canonical_prompt(online_judge_row(_judge_input(progress_sheet=sheet), role="online_bootstrap"))
        # A round that carries no sheet renders exactly as it did before the sheet existed.
        self.assertNotIn("progress_sheet", without)
        self.assertIn('\"progress_sheet\"', with_sheet)
        self.assertIn("segmentation", with_sheet)
        # A measurement told where the pipeline stands is no longer measuring.
        acceptance = canonical_prompt(online_judge_row(_judge_input(progress_sheet=sheet), role="frozen_bootstrap"))
        self.assertNotIn("progress_sheet", acceptance)

    def test_both_questions_that_ask_which_figure_is_better_are_judged_by_one_rule(self):
        """A well-planned figure must not lose to a worse one because it carries a crossed line."""

        from rfs.judge_training.prompts import WHOLE_FIRST_RULE

        advisory = canonical_prompt(online_judge_row(_judge_input(), role="online_bootstrap", comparison=True))
        pairwise = canonical_pairwise_prompt(online_judge_row(_judge_input(), role="frozen_bootstrap"))
        self.assertIn(WHOLE_FIRST_RULE, advisory)
        self.assertIn(WHOLE_FIRST_RULE, pairwise)
        # The order is stated, not merely implied: the whole figure first, the parts after it.
        self.assertIn("Judge the whole figure before you judge anything in it", advisory)
        self.assertIn("outweighs clean lines", pairwise)

    def test_a_round_with_a_persons_instruction_hands_it_to_the_seat(self):
        prompt = canonical_reflection_prompt(
            {"outcome": "improved", "human_preferences": [{"apply_at": "theme_planning", "directive": "Keep the palette calm."}]}
        )
        self.assertIn("Keep the palette calm.", prompt)
        # The instruction travels in the round's own outcome, so a round with none carries no key:
        # the sentence describing the field is the same text either way.
        self.assertIn('\"human_preferences\"', prompt.split("OUTCOME=")[-1])
        self.assertNotIn('\"human_preferences\"', canonical_reflection_prompt({"outcome": "improved"}).split("OUTCOME=")[-1])

    def test_the_row_carries_the_paper_contract_and_the_candidate(self):
        row = online_judge_row(_judge_input(method=_method_contract()), role="online_bootstrap")
        self.assertEqual(["Encoder"], row["ground_truth"]["required_entities"])
        self.assertEqual(["Encoder -> Decoder (data_flow)"], row["ground_truth"]["required_relations"])
        self.assertEqual("Rebuild a scene from one image.", row["ground_truth"]["method_summary"])
        self.assertIn("program", row["candidate"])
        self.assertIn("method", row["candidate"])

    def test_the_method_contract_prefers_must_show_modules(self):
        contract = method_contract(_method_contract())
        self.assertEqual(["Encoder"], contract["required_entities"])
        self.assertEqual(["A"], method_contract({"modules": [{"id": "a", "label": "A"}]})["required_entities"])

    def test_a_prompt_never_leaks_an_unseen_consequence(self):
        prompt = canonical_prompt(online_judge_row(_judge_input(), role="online_bootstrap"))
        for leaked in ("repair_gain", "action_execution", "state_after", "audit_status", "decision"):
            self.assertNotIn(leaked, prompt)

    def test_the_two_roles_are_told_two_different_things(self):
        advisory = canonical_prompt(online_judge_row(_judge_input(), role="online_bootstrap"))
        acceptance = canonical_prompt(online_judge_row(_judge_input(), role="frozen_bootstrap"))
        self.assertIn("Judge role: advisory.", advisory)
        self.assertIn("Judge role: acceptance.", acceptance)
        self.assertIn("you are not asked to score it", advisory)
        self.assertIn("you are not asked which repair to run", acceptance)
        self.assertNotEqual(advisory, acceptance)

    def test_the_served_prompt_version_is_the_trained_one(self):
        self.assertEqual(PROMPT_VERSION, prompt_version())

    def test_the_blocker_tally_is_derived_once_from_the_answer(self):
        # The controller and the validators used to count blockers differently, so "no new
        # blocking issue" could hold while the Judge had listed some. One writer, one number.
        scored = normalize_judge_result({
            "judge_role": "frozen_bootstrap",
            "overall_score": 0.6,
            "status": "fault",
            "blocking_issues": ["panel is empty"],
            "stage_feedback": [{"severity": "blocker", "problem": "connector has no target"}, {"severity": "minor", "problem": "label is short"}],
        }, role="frozen_bootstrap")
        self.assertEqual(2, scored["blocking_issue_count"])
        advised = normalize_judge_result({
            "judge_role": "online_bootstrap",
            "status": "fault",
            "stage_feedback": [{"severity": "blocker", "problem": "connector has no target"}],
        }, role="online_bootstrap")
        self.assertEqual(1, advised["blocking_issue_count"])

    def test_only_the_advisory_answer_keeps_a_legal_applicability(self):
        advised = normalize_judge_result({"judge_role": "online_bootstrap", "applicability": "domain", "status": "fault"}, role="online_bootstrap")
        self.assertEqual("domain", advised["applicability"])
        dropped = normalize_judge_result({"judge_role": "online_bootstrap", "applicability": "everywhere", "status": "fault"}, role="online_bootstrap")
        self.assertEqual("", dropped["applicability"])
        scored = normalize_judge_result({"judge_role": "frozen_bootstrap", "applicability": "common", "overall_score": 0.4, "status": "fault"}, role="frozen_bootstrap")
        self.assertEqual("", scored["applicability"])

    def test_the_reflection_prompt_names_the_actions_the_loop_may_run(self):
        # A revision the controller cannot run used to be silently dropped. The prompt now asks
        # in the loop's own vocabulary, and the alias reader covers the spellings it may still use.
        for action in sorted(REPAIR_ACTIONS):
            self.assertIn(action, REFLECTION_PROMPT)

    def test_the_reflection_prompt_asks_for_the_rule_that_travels(self):
        # A lesson worth keeping is in two parts: the binding of this figure and the rule another
        # paper could be drawn by. The prompt names the mechanism vocabulary and the policy that
        # keeps such a rule, so a reflection can state one instead of only naming a bounded move.
        self.assertIn("revise_rule", REFLECTION_PROMPT)
        for action in sorted(MECHANISM_ACTIONS):
            self.assertIn(action, REFLECTION_PROMPT)
        # Filling the field is asked for outright, and stating a rule is separated from "this
        # round went wrong" so the model does not withhold a rule to avoid blaming the round.
        self.assertIn("Always fill both fields", REFLECTION_PROMPT)
        self.assertIn("not a claim that this round went wrong", REFLECTION_PROMPT)
        for parameter in ("shared_vertical_channel", "forbid_crossing", "min_bend", "max_segments"):
            self.assertIn(parameter, REFLECTION_PROMPT)
        for parameter in ("align_grid", "gutter_min", "panel_occupancy_min"):
            self.assertIn(parameter, REFLECTION_PROMPT)
        self.assertIn("binding", REFLECTION_PROMPT)

    def test_each_served_schema_matches_the_dataset_label_of_its_role(self):
        prompt = canonical_prompt(online_judge_row(_judge_input(), role="online_bootstrap"))
        self.assertIn(json.dumps(ADVISORY_OUTPUT_SCHEMA, ensure_ascii=False), prompt)
        self.assertEqual("common|domain", ADVISORY_OUTPUT_SCHEMA["applicability"])
        self.assertIn("do not write domain", prompt)
        self.assertEqual(set(ADVISORY_OUTPUT_SCHEMA), set(judgement_target({})))
        self.assertEqual(set(ACCEPTANCE_OUTPUT_SCHEMA), set(judgement_target({}, role="frozen_bootstrap")))
        self.assertNotIn("overall_score", judgement_target({"overall_score": 0.5}))
        self.assertEqual(0.5, judgement_target({"overall_score": 0.5}, role="frozen_bootstrap")["overall_score"])
        self.assertNotIn("stage_feedback", judgement_target({"stage_feedback": [{"problem": "x"}]}, role="frozen_bootstrap"))
        self.assertNotIn("repair", judgement_target({"overall_score": 0.5, "repair": [{"action": "reroute"}]}, role="frozen_bootstrap"))

    def test_a_model_answer_in_the_process_dialect_reaches_the_controller(self):
        online, _ = build_direct_judges(mode="bootstrap", online_model="ordinary-vlm")
        answer = {
            "overall_score": 0.4,
            "status": "fault",
            "dimension_scores": {"connector_clarity": 0.3},
            "confidence": 0.6,
            "stage_feedback": [
                {
                    "scope_type": "connector",
                    "problem": "Two connectors cross the panel border.",
                    "evidence": ["edge e1"],
                    "edge_ids": ["e1"],
                    "requested_actions": [{"action": "reroute"}],
                }
            ],
        }
        with patch.object(online, "gate", _PassingGate()), patch("rfs.evaluation.direct_judges.call_vlm_json", return_value=answer):
            result = online.evaluate(_judge_input())
        self.assertEqual("rfs.judge_result", result["contract"])
        self.assertTrue(validate_judge_result(result)["ok"], validate_judge_result(result)["errors"])
        self.assertEqual("reroute", result["repair"][0]["action"])
        self.assertIsNone(result["overall_score"])
        self.assertFalse(result["score_measured"])
        self.assertEqual(0.4, result["reported_score"])
        self.assertEqual(PROMPT_VERSION, result["prompt_version"])

    def test_the_scoring_role_answer_carries_no_repair_action(self):
        _, frozen = build_direct_judges(mode="bootstrap", frozen_model="ordinary-vlm")
        answer = {
            "overall_score": 0.9,
            "status": "pass",
            "dimension_scores": {"connector_clarity": 0.8},
            "stage_feedback": [{"scope_type": "slot", "slot_ids": ["slot_n"], "edge_ids": ["e1"], "problem": "icon too small", "requested_actions": [{"action": "reroute"}]}],
        }
        with patch.object(frozen, "gate", _PassingGate()), patch("rfs.evaluation.direct_judges.call_vlm_json", return_value=answer):
            result = frozen.evaluate(_judge_input())
        self.assertEqual(0.9, result["overall_score"])
        self.assertTrue(result["score_measured"])
        self.assertEqual([], result["repair"])
        self.assertEqual("reroute", result["reported_actions"][0]["action"])

    def test_a_gate_fault_no_longer_silences_the_advice(self):
        online, frozen = build_direct_judges(mode="bootstrap", online_model="ordinary-vlm", frozen_model="ordinary-vlm")
        broken = _judge_input(program={"panels": [], "semantic_nodes": [], "slots": [], "arrows": []})
        answer = {
            "status": "fault",
            "stage_feedback": [{"scope_type": "global", "problem": "the middle band is empty", "requested_actions": [{"action": "relayout_global"}]}],
        }
        with patch("rfs.evaluation.direct_judges.call_vlm_json", return_value=answer) as call:
            advised = online.evaluate(broken)
        # The worst figures are the ones advice is for. The deterministic quality check's
        # findings are merged into the answer rather than replacing it.
        self.assertEqual(1, call.call_count)
        self.assertFalse(advised["gate_only"])
        self.assertIsNone(advised["overall_score"])
        self.assertFalse(advised["score_measured"])
        self.assertEqual("relayout_global", advised["repair"][0]["action"])
        self.assertGreaterEqual(advised["blocking_issue_count"], 1)
        with patch("rfs.evaluation.direct_judges.call_vlm_json", return_value={"overall_score": 0.6, "status": "pass"}) as call:
            measured = frozen.evaluate(broken)
        self.assertEqual(1, call.call_count)
        self.assertEqual(0.6, measured["overall_score"])
        self.assertFalse(measured["gate_only"])
        self.assertGreaterEqual(measured["blocking_issue_count"], 1)
        self.assertEqual("fault", measured["status"])
        with patch("rfs.evaluation.direct_judges.call_vlm_json", return_value={"overall_score": 0.6, "status": "pass"}) as call:
            measured = frozen.evaluate(broken)
        self.assertEqual(1, call.call_count)
        self.assertEqual(0.6, measured["overall_score"])
        self.assertFalse(measured["gate_only"])
        self.assertGreaterEqual(measured["blocking_issue_count"], 1)
        self.assertEqual("fault", measured["status"])

    def test_a_gate_blocked_candidate_still_reaches_the_advisory_model(self):
        online, frozen = build_direct_judges(mode="bootstrap", online_model="ordinary-vlm", frozen_model="ordinary-vlm")
        blocked = _judge_input(grounding_gate={"status": "blocked", "blocking_issues": ["missing required node"]})
        answer = {
            "status": "fault",
            "stage_feedback": [{
                "scope_type": "interface",
                "problem": "arrows cross",
                "edge_ids": ["e1"],
                "requested_actions": [{"action": "reroute", "edge_ids": ["e1"]}],
            }],
        }
        with patch("rfs.evaluation.direct_judges.call_vlm_json", return_value=answer) as call:
            result = online.evaluate(blocked)
        self.assertEqual(1, call.call_count)
        self.assertFalse(result["gate_only"])
        self.assertEqual("reroute", result["repair"][0]["action"])
        self.assertNotIn("reanalyze_semantics", [item.get("action") for item in result["repair"]])
        self.assertIn("missing required node", result["blocking_issues"])
        with patch("rfs.evaluation.direct_judges.call_vlm_json", return_value={"overall_score": 0.4, "status": "blocked"}) as call:
            measured = frozen.evaluate(blocked)
        self.assertEqual(1, call.call_count)
        self.assertFalse(measured["gate_only"])
        self.assertEqual(0.4, measured["overall_score"])
        self.assertIn("missing required node", measured["blocking_issues"])


if __name__ == "__main__":
    unittest.main()
