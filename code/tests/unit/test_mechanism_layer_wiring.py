"""The mechanism-layer wiring: the parts that were written but never reached the loop.

Every refusal in the rule/binding gate and every mechanism-level action had a definition, a
contract value and a unit test, and none of them could fire in a real run: the gate was never
handed the fingerprint of the move the mechanism would have made anyway, the mechanism actions
were absent from the table ``action_is_executable`` reads, a rule the reflection stated was
dropped unless the model also changed an unrelated policy field, and the binding was never
split off the row that entered the library. This module holds the loop shut: it asserts each of
those channels is reachable through the code path a run actually takes, not only through a
hand-called helper.
"""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.contracts.evolution import validate_experience
from rfs.evolution.direct import ExperienceStore
from rfs.evolution.direct.controller import ACTION_PRIORITY, action_execution_fingerprint, action_is_executable, apply_reflection_policy, mechanism_rule_fingerprint, next_rule_turn, select_next_round_proposal
from rfs.evolution.direct.experience import experience_binding_record, experience_library_record
from rfs.evolution.direct.rule_binding import rule_binding_gate, split_rule_binding
from rfs.evolution.direct.store import ALLOWED_APPEND_STATUSES, DEFAULT_RETRIEVAL_STATUSES


def measured_row(**overrides) -> dict:
    base = {
        "experience_id": "direct-exp-wiring",
        "domain": "direct_editable",
        "schema_version": "3.7",
        "scope": "global",
        "apply_at": "ppt_compilation",
        "memory_type": "connector",
        "hypothesis": "a shared vertical channel keeps a one-to-many fan-out readable",
        "guidance": "Give every branch of a fan-out its own vertical channel and keep the entry points aligned so the flow reads as one direction.",
        "action": {"type": "router_constraint", "constraints": {"shared_vertical_channel": True}, "edge_ids": ["rel_12"]},
        "condition": {"topology": "pipeline", "node_count_bucket": "high", "defect_patterns": ["fanout_1_to_n", "cross_panel"]},
        "required_motifs": [],
        "excluded_conditions": [],
        "graph_scope": {"edge_ids": ["rel_12"], "object_ids": ["node_alpha"]},
        "condition_evidence": {"changed_object_ids": ["rel_12"], "measured_by": "judge_score_delta"},
        "observed_effect": {"changed_object_ids": ["rel_12"], "overall_score_delta": 0.12},
        "accepted": True,
        "repair_gain": 0.12,
        "comparison": "",
        "pairwise_verdict": "",
        "status": "candidate",
        "applicability": "common",
        "strategy_id": "direct-strategy-abc",
        "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.62},
        "outcome": {"accepted": True, "before_score": 0.5, "after_score": 0.62, "score_delta": 0.12, "new_blockers": [], "comparison": "", "pairwise_verdict": ""},
        "validated_cases": ["a.pdf"],
        "failed_cases": [],
        "side_effects": [],
        "trigger": {},
        "preserve": [],
        "forbidden_changes": [],
        "judge_fingerprint": {},
        "expected_effect": {"overall_score": "increase", "new_blockers": 0},
        "source_case": "a.pdf",
        "source_round": 1,
    }
    base.update(overrides)
    return base


class TheGateIsReachableThroughTheStoreTests(unittest.TestCase):
    def test_a_row_that_restates_the_rounds_default_move_is_refused_on_write(self):
        # The caller stamps the mechanism's own policy move onto the row, which is the only way
        # the gate can tell a finding from a restatement. Without that stamp the branch is dead.
        action = {"type": "resize_nodes", "factor": 1.08, "object_ids": ["node_alpha"]}
        record = measured_row(action=action, default_action_fingerprint=action_execution_fingerprint({"type": "resize_nodes", "factor": 1.08}))

        with tempfile.TemporaryDirectory() as tmp:
            stored = ExperienceStore(Path(tmp)).append_result(record)
            rows = ExperienceStore(Path(tmp)).records()

        self.assertFalse(stored["ok"])
        self.assertEqual("common_knowledge", stored["reason"])
        self.assertEqual([], rows)

    def test_a_row_that_differs_from_the_default_move_is_kept(self):
        record = measured_row(default_action_fingerprint=action_execution_fingerprint({"type": "relayout_global", "factor": 1.02, "dx": 0.008}))

        with tempfile.TemporaryDirectory() as tmp:
            stored = ExperienceStore(Path(tmp)).append_result(record)
            rows = ExperienceStore(Path(tmp)).records()

        self.assertTrue(stored["ok"], stored)
        self.assertEqual(1, len(rows))


class TheLibraryKeepsTheRuleNotTheFigureTests(unittest.TestCase):
    def test_the_row_kept_is_a_rule_and_names_no_object_of_the_figure(self):
        row = experience_library_record(measured_row(), binding_ref="/runs/round_01/experience_binding.json")

        self.assertTrue(row["hypothesis"])
        self.assertTrue(row["action"])
        self.assertNotIn("graph_scope", row)
        self.assertNotIn("edge_ids", row["action"])
        self.assertEqual({"shared_vertical_channel": True}, row["action"]["constraints"])
        self.assertEqual("/runs/round_01/experience_binding.json", row["binding_ref"])
        # Nothing that names an object of the source figure may survive anywhere in the rule text.
        rule, _binding = split_rule_binding(row)
        self.assertNotIn("rel_12", json.dumps(rule, ensure_ascii=False))
        self.assertNotIn("node_alpha", json.dumps(rule, ensure_ascii=False))

    def test_the_binding_keeps_the_objects_the_rule_gave_up(self):
        binding = experience_binding_record(measured_row())

        self.assertEqual(["rel_12"], binding["graph_scope"]["edge_ids"])
        self.assertEqual(["rel_12"], binding["action"]["edge_ids"])
        self.assertEqual("direct-exp-wiring", binding["experience_id"])

    def test_a_rule_row_is_valid_because_it_points_at_its_binding(self):
        row = experience_library_record(measured_row(), binding_ref="/runs/round_01/experience_binding.json")

        self.assertTrue(validate_experience(row)["ok"], validate_experience(row)["errors"])

    def test_a_rule_row_without_a_binding_and_without_ids_is_refused(self):
        row = experience_library_record(measured_row())

        self.assertFalse(validate_experience(row)["ok"])

    def test_a_row_written_the_old_way_still_validates(self):
        self.assertTrue(validate_experience(measured_row())["ok"])


class MechanismSelfRepairIsNotALessonTests(unittest.TestCase):
    def _build(self, action):
        from rfs.evolution.direct.experience import build_direct_experience_result

        return build_direct_experience_result(
            round_index=1,
            previous={"overall_score": 0.5},
            current={"overall_score": 0.62},
            layout={"topology": "pipeline", "nodes": [], "connectors": []},
            action=action,
            source_case="a.pdf",
        )

    def test_a_round_the_mechanism_repaired_itself_is_not_memorized(self):
        built = self._build({"type": "resize_nodes", "factor": 1.08, "source": "decomposition_policy"})

        self.assertFalse(built["ok"])
        self.assertEqual("mechanism_self_repair", built["reason"])
        self.assertIsNone(built["experience"])

    def test_a_colour_round_the_planner_took_on_its_own_is_not_memorized(self):
        # The theme planner reacts to the colour it just measured on its own policy, so the move
        # is made with or without Memory. It used to be filed as an experience, and because no
        # proposal wrote an applicability for it, every such row stopped the validation ladder at
        # ``applicability_unspecified`` instead of testing anything.
        built = self._build({"type": "replan_theme", "color_theme": {"primary": "#111111"}, "source": "color_feedback"})

        self.assertFalse(built["ok"])
        self.assertEqual("mechanism_self_repair", built["reason"])
        self.assertIsNone(built["experience"])

    def test_a_round_the_advisor_asked_for_is_still_memorized(self):
        built = self._build({"type": "reroute", "edge_ids": ["rel_12"]})

        self.assertTrue(built["ok"], built)
        self.assertEqual(
            {"topology": "pipeline", "node_count_bucket": "low", "connector_count_bucket": "low"},
            built["experience"]["condition"],
        )

    def test_the_condition_names_the_shape_the_row_can_recognise_again(self):
        # One node feeding four others across a panel boundary: the two size buckets both read
        # "low", so without the pattern the row could not have recognised this illness on a
        # different paper. The shape has to reach the stored condition, not only the reader.
        layout = {
            "topology": "pipeline",
            "panels": [{"id": "p1"}, {"id": "p2"}],
            "nodes": [
                {"id": "n1", "panel_id": "p1"},
                {"id": "n2", "panel_id": "p2"},
                {"id": "n3", "panel_id": "p2"},
                {"id": "n4", "panel_id": "p2"},
                {"id": "n5", "panel_id": "p2"},
            ],
            "connectors": [
                {"id": "c1", "source": "n1", "target": "n2"},
                {"id": "c2", "source": "n1", "target": "n3"},
                {"id": "c3", "source": "n1", "target": "n4"},
                {"id": "c4", "source": "n1", "target": "n5"},
            ],
        }
        from rfs.evolution.direct.experience import build_direct_experience_result

        built = build_direct_experience_result(
            round_index=1,
            previous={"overall_score": 0.4},
            current={"overall_score": 0.6},
            layout=layout,
            action={"type": "reroute", "edge_ids": ["c1"]},
            source_case="a.pdf",
        )

        self.assertTrue(built["ok"], built)
        self.assertEqual(["cross_panel", "fanout_1_to_n", "port_collision"], built["experience"]["condition"]["defect_patterns"])


class AMechanismRuleRoundBecomesALessonTests(unittest.TestCase):
    """A round the loop ran on a reflection's rule enters the library like any other move.

    The library kept the statement and nothing ever ran it, so the whole path from "the round
    won" to "the row is examinable" was never walked on a mechanism action. This walks it: the
    statement a later paper is handed is rendered from the rule rather than from how this round
    scored, the row passes the same gate any row passes, and the cheap exam recognises it.
    """

    LAYOUT = {
        "topology": "pipeline",
        "panels": [{"id": "p1"}],
        "nodes": [{"id": "n1", "panel_id": "p1", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.1}}],
        "connectors": [],
    }

    def _row(self, action: dict) -> tuple[dict, dict]:
        from rfs.evolution.direct.experience import build_direct_experience_result

        built = build_direct_experience_result(
            round_index=1,
            previous={"overall_score": 0.5},
            current={"overall_score": 0.65},
            layout=self.LAYOUT,
            action=action,
            source_case="a.pdf",
        )
        self.assertTrue(built["ok"], built)
        return built["experience"], experience_library_record(built["experience"])

    def test_a_layout_rule_round_becomes_a_row_the_cheap_exam_can_measure(self):
        from rfs.workflows.direct_experience_validation import _compile_only_reason

        experience, row = self._row({"type": "layout_constraint", "params": {"gutter_min": 0.02}, "source": "reflection_rule"})

        # The instruction a later paper is handed is rendered from the rule, not from how this
        # round scored: a row whose instruction says "the checkpoint preferred the revised figure"
        # tells the next paper nothing it can do.
        self.assertTrue(str(experience["hypothesis"]).strip())
        self.assertIn("gutter_min", str(experience["hypothesis"]))
        self.assertEqual(experience["hypothesis"], row["strategy"]["instruction"])
        self.assertTrue(rule_binding_gate(row)["ok"], rule_binding_gate(row))
        self.assertEqual({"params": {"gutter_min": 0.02}}, row["action"]["layout_constraints"])
        # The exam that compiles one figure twice can only run on a row it recognises as a rule.
        self.assertEqual("", _compile_only_reason(row))

    def test_a_router_rule_round_is_recognised_the_same_way(self):
        from rfs.workflows.direct_experience_validation import _compile_only_reason

        _experience, row = self._row({"type": "router_constraint", "constraints": {"forbid_crossing": True}, "source": "reflection_rule"})

        self.assertEqual({"router": {"forbid_crossing": True}}, row["action"]["routing_constraints"])
        self.assertTrue(rule_binding_gate(row)["ok"], rule_binding_gate(row))
        self.assertEqual("", _compile_only_reason(row))

    def test_the_same_rule_driven_by_the_mechanism_itself_is_still_not_a_lesson(self):
        # What makes the row a lesson is that the reflection stated it: the same parameters taken
        # on the mechanism's own policy would have been done with or without Memory.
        from rfs.evolution.direct.experience import build_direct_experience_result

        built = build_direct_experience_result(
            round_index=1,
            previous={"overall_score": 0.5},
            current={"overall_score": 0.65},
            layout=self.LAYOUT,
            action={"type": "layout_constraint", "params": {"gutter_min": 0.02}, "source": "decomposition_policy"},
            source_case="a.pdf",
        )

        self.assertFalse(built["ok"])
        self.assertEqual("mechanism_self_repair", built["reason"])


class MechanismActionsCanActuallyRunTests(unittest.TestCase):
    def test_the_two_mechanism_actions_are_ordered_and_executable(self):
        for name in ("router_constraint", "layout_constraint"):
            self.assertIn(name, ACTION_PRIORITY)
            action = {"type": name, "constraints": {"forbid_crossing": True}} if name == "router_constraint" else {"type": name, "params": {"gutter_min": 0.02}}
            self.assertTrue(action_is_executable(action), name)

    def test_the_judge_may_name_them_without_them_counting_as_repairs(self):
        from rfs.contracts.evolution import REPAIR_ACTIONS
        from rfs.evolution.direct.actions import SUPPORTED_ACTIONS

        for name in ("router_constraint", "layout_constraint"):
            self.assertIn(name, REPAIR_ACTIONS)
            self.assertNotIn(name, SUPPORTED_ACTIONS)


class AStatedRuleSurvivesAPolicyThatDidNotNameItTests(unittest.TestCase):
    def test_a_rule_is_kept_when_next_policy_stayed_at_keep(self):
        reflection = {"next_policy": "keep", "rule": {"type": "router_constraint", "constraints": {"forbid_crossing": True}}}

        policy = apply_reflection_policy({"accepted": True, "reason": "ok"}, {"type": "reroute"}, reflection)

        self.assertTrue(policy["revised_rule"])
        self.assertEqual("router_constraint", policy["revised_rule"]["type"])
        self.assertIn("revise_rule", policy["policy"]["applied"])
        self.assertEqual("revise_rule", policy["policy"]["next_policy"])

    def test_a_rule_is_not_kept_from_a_round_whose_guidance_was_disabled(self):
        reflection = {"next_policy": "disable", "rule": {"type": "router_constraint", "constraints": {"forbid_crossing": True}}}

        policy = apply_reflection_policy({"accepted": True, "reason": "ok"}, {"type": "reroute"}, reflection, injected_experience={"experience_id": "direct-exp-old"})

        self.assertFalse(policy["revised_rule"])
        self.assertIn("disable", policy["policy"]["applied"])

    def test_an_unusable_rule_is_still_refused(self):
        policy = apply_reflection_policy({"accepted": True, "reason": "ok"}, {"type": "reroute"}, {"next_policy": "keep", "rule": {"type": "reroute", "edge_ids": ["rel_12"]}})

        self.assertFalse(policy["revised_rule"])


class AStatedRuleGetsOneTurnInTheLoopTests(unittest.TestCase):
    """A rule the reflection wrote is proposed once, instead of only being written down.

    The library kept the statement and nothing ever ran it, so a rule that would have helped and
    a rule that would not were indistinguishable: neither was ever put to the loop. The next
    round tries it, once per iteration, through the ordinary judgement and the ordinary gates.
    """

    RULE = {"type": "layout_constraint", "params": {"align_grid": 0.05}}

    def test_a_stated_rule_comes_back_as_an_action_the_loop_may_run(self):
        policy = apply_reflection_policy({"accepted": True, "reason": "ok"}, {"type": "reroute"}, {"next_policy": "keep", "rule": self.RULE})

        self.assertTrue(policy["revised_rule_action"])
        self.assertEqual("layout_constraint", policy["revised_rule_action"]["type"])
        self.assertEqual({"align_grid": 0.05}, policy["revised_rule_action"]["params"])
        self.assertEqual("reflection_rule", policy["revised_rule_action"]["source"])
        # The policy record carries it too, so a run's own ledger names what it was about to try.
        self.assertEqual(policy["revised_rule_action"], policy["policy"]["revised_rule_action"])
        self.assertTrue(action_is_executable(policy["revised_rule_action"]))

    def test_a_record_without_the_field_reads_as_no_rule_to_try(self):
        policy = apply_reflection_policy({"accepted": True, "reason": "ok"}, {"type": "reroute"}, {"next_policy": "keep"})

        self.assertEqual({}, policy["revised_rule_action"])
        self.assertIsNone(policy["policy"]["revised_rule_action"])

    def test_the_rule_is_offered_after_a_hard_break_and_before_everything_else(self):
        soft = {"required": True, "violations": [{"kind": "soft", "metric": "panel_occupancy"}]}
        soft_proposal = {"type": "relayout_global", "scope_type": "global", "factor": 1.02, "dx": 0.0, "dy": 0.0, "source": "decomposition_policy"}
        hard = {"required": True, "violations": [{"kind": "hard", "metric": "cross_panel"}]}
        hard_proposal = {"type": "replan_structure", "scope_type": "global", "source": "decomposition_policy"}
        rule, _ = next_rule_turn(self.RULE)

        # A figure whose structure does not hold has to be fixed before a drawing rule can be
        # measured on it.
        self.assertEqual("replan_structure", select_next_round_proposal(rule_action=rule, decomposition_action=hard, decomposition_proposal=hard_proposal)["type"])
        # With the structure sound, a rule nobody has ever run is not left waiting behind a
        # colour repair the loop would have proposed anyway.
        self.assertEqual("layout_constraint", select_next_round_proposal(rule_action=rule, decomposition_action=soft, decomposition_proposal=soft_proposal)["type"])
        self.assertEqual("layout_constraint", select_next_round_proposal(rule_action=rule, pending_color_feedback={"repair_required": True})["type"])
        self.assertEqual("replan_theme", select_next_round_proposal(pending_color_feedback={"repair_required": True})["type"])
        # A reflection's own revision still outranks everything.
        self.assertEqual("reroute", select_next_round_proposal(revised_action={"type": "reroute", "edge_ids": ["e1"]}, rule_action=rule)["type"])

    def test_the_same_rule_is_offered_once_and_two_spellings_are_one_rule(self):
        other_spelling = {"type": "layout_constraint", "params": {"align_grid": 0.05}, "apply_at": "ppt_compilation", "source": "reflection_rule"}
        self.assertEqual(mechanism_rule_fingerprint(self.RULE), mechanism_rule_fingerprint(other_spelling))

        offered, fingerprint = next_rule_turn(self.RULE)
        self.assertTrue(offered)
        self.assertEqual({}, next_rule_turn(other_spelling, tried={fingerprint})[0])

        # A different rule is a different turn.
        self.assertNotEqual(fingerprint, mechanism_rule_fingerprint({"type": "layout_constraint", "params": {"gutter_min": 0.02}}))
        self.assertTrue(next_rule_turn({"type": "layout_constraint", "params": {"gutter_min": 0.02}}, tried={fingerprint})[0])

    def test_a_name_without_its_parameters_is_never_offered(self):
        self.assertEqual(({}, ""), next_rule_turn({"type": "layout_constraint"}))
        self.assertEqual(({}, ""), next_rule_turn({}))
        self.assertEqual("", mechanism_rule_fingerprint({"type": "reroute", "edge_ids": ["e1"]}))


class TheUnverifiableDeadEndIsNamedTests(unittest.TestCase):
    def test_the_status_is_allowed_but_never_retrieved(self):
        self.assertIn("unverifiable", ALLOWED_APPEND_STATUSES)
        self.assertNotIn("unverifiable", DEFAULT_RETRIEVAL_STATUSES)


if __name__ == "__main__":
    unittest.main()
