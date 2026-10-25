"""Tests for Memory construction and the reasons a record is refused."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import (
    ExperienceStore,
    build_direct_experience_result,
    normalize_experience_scope,
)
from rfs.evolution.direct.experience import experience_library_record, written_applicability
from rfs.evolution.direct.store import DEFAULT_RETRIEVAL_STATUSES, resolve_retrieval_statuses
from rfs.contracts.evolution import validate_experience

LAYOUT = {"topology": "pipeline", "nodes": [{"id": "node_a"}, {"id": "node_b"}], "connectors": [{"id": "edge_a"}]}
VISUAL_GRAPH = {
    "contract": "rfs.graph",
    "version": 1,
    "paper_id": "paper",
    "kind": "visual",
    "graph_id": "visual_graph.test",
    "nodes": [
        {"id": "node_a", "kind": "visual_node", "label": "A", "role": "stage"},
        {"id": "node_b", "kind": "visual_node", "label": "B", "role": "stage"},
        {"id": "node_c", "kind": "visual_node", "label": "C", "role": "stage"},
    ],
    "edges": [
        {"id": "edge_a", "source": "node_a", "target": "node_b", "relation_type": "data_flow"},
        {"id": "edge_b", "source": "node_b", "target": "node_c", "relation_type": "data_flow"},
    ],
}


#: The prose half of a rule: how to apply a move, and when not to. It is the only part of a row a
#: model writes, so a row without it carries nothing a later paper could act on and the gate
#: refuses it. The tests that do not exercise that refusal carry it.
ADVICE = "Reserve one vertical channel per connector so the flow reads left to right."


def _build(action, **overrides):
    payload = {
        "round_index": 1,
        "previous": {"overall_score": overrides.pop("before", 0.5)},
        "current": {"overall_score": overrides.pop("after", 0.75), "blocking_issue_count": overrides.pop("blocking", 0)},
        "layout": LAYOUT,
        "action": action,
        "source_case": "paper.pdf",
    }
    payload.update(overrides)
    return build_direct_experience_result(**payload)


class ExperienceScopeTests(unittest.TestCase):
    def test_stage_derived_scopes_map_onto_the_contract_vocabulary(self):
        self.assertEqual(normalize_experience_scope("compile"), "global")
        self.assertEqual(normalize_experience_scope("semantic"), "global")
        self.assertEqual(normalize_experience_scope("slot"), "slot")
        self.assertEqual(normalize_experience_scope(None), "global")
        self.assertEqual(normalize_experience_scope("nonsense"), "global")

    def test_applicability_is_copied_only_from_the_action(self):
        unlabeled = _build({"type": "regen_assets", "scope_type": "slot", "slot_ids": ["node_a"]})
        self.assertEqual("", unlabeled["experience"]["applicability"])
        self.assertEqual("no proposal wrote an applicability", unlabeled["experience"]["applicability_reason"])
        common = _build({"type": "regen_assets", "scope_type": "slot", "slot_ids": ["node_a"], "applicability": "common"})
        self.assertEqual("common", common["experience"]["applicability"])
        self.assertEqual("", common["experience"]["applicability_domain"])
        named = _build(
            {"type": "regen_assets", "scope_type": "slot", "slot_ids": ["node_a"], "applicability": "domain"},
            domain_profile={"domain": "ai_method", "domain_missing": False},
        )
        self.assertEqual("domain", named["experience"]["applicability"])
        self.assertEqual("ai_method", named["experience"]["applicability_domain"])
        missing = written_applicability({"applicability": "domain"}, {"domain": "", "domain_missing": True})
        self.assertEqual("", missing["applicability"])
        self.assertIn("named paper domain", missing["applicability_reason"])

    def test_a_single_paper_save_keeps_the_row_as_a_candidate(self):
        built = _build({"type": "regen_assets", "scope_type": "slot", "slot_ids": ["node_a"], "applicability": "common"})
        self.assertEqual("current_case_validated", built["experience"]["status"])
        saved = experience_library_record(built["experience"])
        self.assertEqual("candidate", saved["status"])
        self.assertEqual("refuted", experience_library_record({"status": "refuted"})["status"])
        self.assertEqual("candidate", experience_library_record({"status": "overfit_or_failed"})["status"])

    def test_a_composition_repair_that_wrote_no_applicability_still_says_where_it_applies(self):
        """A move that acts on the composition answers the same defect on any paper.

        Leaving the field blank made the validation harness refuse the row before it drew a case,
        and that refusal was stored as the row's own status -- "no measurement was possible" read
        as "this lesson does not work". The stage the move acts at is enough to say where it
        applies, and it is read through the same function on both sides.
        """

        built = _build({"type": "reroute", "scope_type": "global", "reroute_edges": [{"edge_id": "rel_1"}]})
        self.assertEqual("common", built["experience"]["applicability"])
        self.assertIn("composition", built["experience"]["applicability_reason"])

    def test_a_content_repair_nobody_labelled_stays_unlabelled(self):
        # Nothing up here can say which papers a content claim travels to, so the row keeps the
        # blank the proposal left rather than a label the mechanism invented for it.
        built = _build({"type": "reanalyze_semantics", "scope_type": "semantic", "transferable_advice": ADVICE})
        self.assertEqual("", built["experience"]["applicability"])
        self.assertEqual("no proposal wrote an applicability", built["experience"]["applicability_reason"])

    def test_semantic_scope_repair_produces_a_storable_record(self):
        result = _build({"type": "reanalyze_semantics", "scope_type": "semantic", "transferable_advice": ADVICE})
        self.assertTrue(result["ok"])
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(Path(temp))
            self.assertTrue(store.append(result["experience"]))


class ReflectionPatchProseTests(unittest.TestCase):
    """A reflection that states a mechanism rule states no guidance, so the patch is its prose."""

    PATCH_DIRECTIVE = (
        "Lead every connector out of the side of its node that faces its target, and give the "
        "connectors of one panel a channel they share instead of letting them cross it."
    )

    def _result(self, directive):
        return _build(
            {"type": "relayout_global", "scope_type": "global", "factor": 1.0},
            reflection={
                "outcome": "improved",
                "next_policy": "revise_rule",
                "rule": {
                    "type": "layout_constraint",
                    "apply_at": "ppt_compilation",
                    "params": {"align_grid": 0.05, "gutter_min": 0.02},
                },
                "prompt_patch": {"apply_at": "ppt_compilation", "condition": {}, "directive": directive},
            },
        )

    def test_the_patch_directive_becomes_the_rows_transferable_prose(self):
        # The row used to be filed with an empty prose half whenever the reflection answered in the
        # mechanism vocabulary, and the library gate refused it as having nothing that travels. The
        # patch's own sentence is that half, so the row is kept and reaches the library.
        result = self._result(self.PATCH_DIRECTIVE)

        self.assertEqual(self.PATCH_DIRECTIVE, result["experience"]["guidance"])
        # The row says the sentence is the model's own, so the mechanical fallback below is
        # never mistaken for something the model wrote.
        self.assertEqual("model", result["experience"]["guidance_source"])
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(Path(temp))
            stored = store.append_result(result["experience"], figure=LAYOUT)
            self.assertTrue(stored["ok"], stored)

    def test_a_patch_that_names_the_figures_own_objects_is_refused_like_any_other_rule(self):
        # The patch's sentence travels the same gate as the guidance always did: a sentence about
        # this figure's own boxes is evidence about a measurement, not a rule another paper can act
        # on, and reading it as prose must not open a way around that.
        result = self._result("Move node_a to the left so its arrows stop crossing the panel.")

        self.assertEqual("node_a", result["experience"]["guidance"].split()[1])
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(Path(temp))
            stored = store.append_result(result["experience"], figure=LAYOUT)
            self.assertFalse(stored["ok"])
            self.assertEqual("rule_binds_to_case_objects", stored["reason"])


class ExperienceRejectionReasonTests(unittest.TestCase):
    def test_a_repair_that_changed_nothing_becomes_a_refuted_experience(self):
        result = _build({"type": "relayout_global", "scope_type": "global"}, before=0.75, after=0.75)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "refuted")
        experience = result["experience"]
        self.assertEqual("refuted", experience["status"])
        self.assertFalse(experience["accepted"])
        self.assertEqual(0.0, experience["repair_gain"])
        self.assertEqual(0.0, experience["outcome"]["score_delta"])
        self.assertTrue(validate_experience(experience)["ok"], validate_experience(experience)["errors"])

    def test_a_worse_repair_records_the_negative_gain_and_the_condition_it_failed_under(self):
        result = _build(
            {"type": "reroute", "scope_type": "connector", "edge_ids": ["edge_a"]},
            before=0.75,
            after=0.5,
            bounded_change={"object_diff": {"changed_fields": ["layout"], "changed": True}},
            compiled_state_diff={"changed_fields": ["layout", "program"]},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("refuted", result["reason"])
        experience = result["experience"]
        self.assertEqual(-0.25, experience["repair_gain"])
        self.assertEqual("refuted", experience["status"])
        self.assertEqual(["paper.pdf"], experience["failed_cases"])
        self.assertEqual([], experience["validated_cases"])
        self.assertEqual(["layout"], experience["object_diff"]["changed_fields"])
        self.assertEqual(["layout", "program"], experience["compiled_object_diff"]["changed_fields"])
        self.assertEqual([{"topology": "pipeline", "node_count_bucket": "low", "connector_count_bucket": "low", "reason": "repair_gain_not_positive", "repair_gain": -0.25}], experience["excluded_conditions"])
        self.assertEqual({"source": "direct_evolution_process", "run": "paper.pdf", "round": 1, "measured": True, "instrument": "acceptance_score", "before_score": 0.75, "after_score": 0.5}, experience["provenance"])
        self.assertIn("Do not reuse this move", experience["hypothesis"])
        self.assertTrue(validate_experience(experience)["ok"], validate_experience(experience)["errors"])

    def test_a_refuted_experience_can_be_stored_and_transitioned(self):
        result = _build({"type": "relayout_global", "scope_type": "global"}, before=0.75, after=0.5)
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(Path(temp))
            stored = store.append_result(result["experience"])
            self.assertTrue(stored["ok"], stored)
            store.transition(result["experience"]["experience_id"], "refuted", reason="rolled_back")
            self.assertEqual("refuted", store.lifecycle_status(result["experience"]["experience_id"]))
            self.assertEqual("refuted", store.get(result["experience"]["experience_id"])["status"])
            self.assertEqual([], store.retrieve(limit=5, context={"topology": "pipeline"}))

    def test_a_round_with_nothing_to_compare_against_is_refused_rather_than_credited(self):
        # The score of a first draft is not a gain: with no earlier accepted state there is
        # nothing to subtract, and the record may not present the candidate score as one.
        result = _build({"type": "reroute", "scope_type": "connector"}, previous={})
        self.assertFalse(result["ok"])
        self.assertEqual("no_previous_accepted_round", result["reason"])
        self.assertIsNone(result["experience"])
        self.assertIsNone(result["before_score"])
        self.assertEqual(0.75, result["after_score"])

    def test_a_score_less_seat_is_measured_by_its_own_gate(self):
        # The product's acceptance seat answers the pair question and reports no score of its
        # own. Its deterministic gate still measured both states, so the round carries a
        # measurement -- and says which instrument took it, because the two are not the same
        # evidence.
        result = _build(
            {"type": "relayout_global", "scope_type": "global", "factor": 1.05, "dx": 0.0, "dy": 0.0},
            previous={"overall_score": None, "score_measured": False, "gate_score": 0.4},
            current={"overall_score": None, "score_measured": False, "gate_score": 0.7, "blocking_issue_count": 0},
        )
        self.assertTrue(result["ok"], result)
        experience = result["experience"]
        self.assertEqual("deterministic_gate", experience["measurement_instrument"])
        self.assertEqual(0.3, experience["repair_gain"])
        self.assertEqual("deterministic_gate", experience["provenance"]["instrument"])
        self.assertEqual(0.4, experience["provenance"]["before_score"])

    def test_a_gain_taken_with_two_instruments_is_labelled_as_such(self):
        # A number the acceptance role reported and a gate score are different instruments, so a
        # difference taken across the two is kept but never presented as one reading.
        result = _build(
            {"type": "relayout_global", "scope_type": "global", "transferable_advice": ADVICE},
            previous={"overall_score": 0.5},
            current={"overall_score": None, "score_measured": False, "gate_score": 1.0, "blocking_issue_count": 0},
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual("acceptance_score_to_deterministic_gate", result["experience"]["measurement_instrument"])

    def test_a_round_no_instrument_measured_is_still_refused(self):
        result = _build(
            {"type": "relayout_global", "scope_type": "global", "transferable_advice": ADVICE},
            previous={"overall_score": None, "score_measured": False},
            current={"overall_score": None, "score_measured": False, "blocking_issue_count": 0},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("no_after_measurement", result["reason"])

    def test_leftover_blockers_do_not_drop_a_measured_repair(self):
        result = _build(
            {"type": "relayout_global", "scope_type": "global", "factor": 1.05, "dx": 0.012, "dy": 0.0},
            previous={"overall_score": 0.5, "blocking_issue_count": 2},
            blocking=2,
        )
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["experience"]["accepted"])

    def test_a_redraw_is_not_filed_as_a_refusal(self):
        """The round won by drawing the figure again, which no catalogue can hand to another round.

        A row carrying that move could only ever be refused at the store, so every accepted redraw
        wrote a refusal notice and a run that kept moving forward recorded nothing else. There is
        no rule to keep, and this says so instead of asking the store to reject the row.
        """
        result = _build({"type": "initial", "scope_type": "global"})
        self.assertFalse(result["ok"])
        self.assertEqual("not_a_catalogue_repair", result["reason"])
        self.assertEqual("initial", result["action"])
        self.assertIsNone(result["experience"])
        self.assertIn("no rule to keep", result["note"])

    def test_a_defect_this_round_introduced_is_carried_on_the_row_not_refused(self):
        """The gain was measured, so the row is kept; the defect rides on it instead of dropping it.

        Refusing the row over the defect threw the whole reading away: the library stayed empty
        while the figures kept improving. The row is filed with the defect written on it and the
        exclusion that goes with it, and the validation ladder is what spends the mark.
        """

        result = _build(
            {"type": "relayout_global", "scope_type": "global", "factor": 1.05, "dx": 0.012, "dy": 0.0},
            previous={"overall_score": 0.5, "blocking_issue_count": 0},
            blocking=2,
        )
        self.assertTrue(result["ok"], result)
        row = result["experience"]
        self.assertIsNotNone(row)
        self.assertTrue(row["accepted"])
        self.assertTrue(row["carried_new_defect"])
        self.assertEqual(row["new_blocking_defect_count"], 2)
        self.assertEqual(
            [item["reason"] for item in row["excluded_conditions"] if item["reason"] == "carried_new_defect"],
            ["carried_new_defect"],
        )

    def test_the_defect_a_round_introduced_is_named_when_the_gate_separated_it(self):
        result = _build(
            {"type": "relayout_global", "scope_type": "global", "factor": 1.05, "dx": 0.012, "dy": 0.0},
            previous={
                "overall_score": 0.5,
                "blocking_issue_count": 0,
                "deterministic_blockers": [],
                "deterministic_blocker_counts": {},
            },
            current={
                "overall_score": 0.75,
                "blocking_issue_count": 2,
                "deterministic_blockers": ["interface:box_pass"],
                "deterministic_blocker_counts": {"interface:box_pass": 2},
            },
        )
        row = result["experience"]
        self.assertEqual(row["new_blocking_defects"], ["interface:box_pass"])
        self.assertEqual(row["new_blocking_defect_count"], 1)
        condition = next(item for item in row["excluded_conditions"] if item["reason"] == "carried_new_defect")
        self.assertEqual(condition["defects"], ["interface:box_pass"])

    def test_a_round_that_introduced_nothing_carries_no_mark(self):
        result = _build({"type": "relayout_global", "scope_type": "global", "factor": 1.05, "dx": 0.012, "dy": 0.0})
        row = result["experience"]
        self.assertFalse(row["carried_new_defect"])
        self.assertEqual(row["new_blocking_defects"], [])
        self.assertEqual([item for item in row["excluded_conditions"] if item["reason"] == "carried_new_defect"], [])

    def test_pairwise_preference_credits_memory_when_absolute_scores_do_not_move(self):
        result = _build(
            {"type": "relayout_global", "scope_type": "global", "factor": 0.9, "dx": 0.0, "dy": 0.0},
            before=1.0,
            after=1.0,
            round_decision={"accepted": True, "comparison": "pairwise", "pairwise_verdict": "after"},
        )
        self.assertTrue(result["ok"], result)
        experience = result["experience"]
        self.assertTrue(experience["accepted"])
        self.assertEqual(0.0, experience["repair_gain"])
        self.assertEqual("pairwise", experience["comparison"])
        self.assertEqual("after", experience["pairwise_verdict"])
        self.assertEqual("current_case_validated", experience["status"])
        self.assertEqual([], experience["excluded_conditions"])
        self.assertTrue(validate_experience(experience)["ok"], validate_experience(experience)["errors"])

    def test_pairwise_preference_credits_memory_when_absolute_scores_fall(self):
        result = _build(
            {"type": "regen_assets", "scope_type": "slot", "slot_ids": ["slot_a"]},
            before=1.0,
            after=0.8,
            round_decision={"accepted": True, "comparison": "pairwise", "pairwise_verdict": "after"},
        )
        self.assertTrue(result["ok"], result)
        experience = result["experience"]
        self.assertTrue(experience["accepted"])
        self.assertEqual(-0.2, experience["repair_gain"])
        self.assertEqual("current_case_validated", experience["status"])
        self.assertTrue(validate_experience(experience)["ok"], validate_experience(experience)["errors"])

    def test_pairwise_refusal_records_a_refuted_row_even_if_scores_rose(self):
        result = _build(
            {"type": "relayout_global", "scope_type": "global", "factor": 0.9},
            before=0.5,
            after=0.7,
            round_decision={"accepted": False, "comparison": "pairwise", "pairwise_verdict": "before"},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("refuted", result["reason"])
        experience = result["experience"]
        self.assertFalse(experience["accepted"])
        self.assertEqual(0.2, experience["repair_gain"])
        self.assertEqual("refuted", experience["status"])
        self.assertEqual("pairwise_not_preferred", experience["excluded_conditions"][0]["reason"])
        self.assertTrue(validate_experience(experience)["ok"], validate_experience(experience)["errors"])


class ExperienceTypingTests(unittest.TestCase):
    def test_action_kind_decides_memory_type_and_router_stage(self):
        theme = _build({"type": "regen_assets", "scope_type": "slot"})["experience"]
        self.assertEqual(theme["memory_type"], "icon")
        self.assertEqual(theme["apply_at"], "icon_generation")
        routing = _build({"type": "reroute", "scope_type": "connector"})["experience"]
        self.assertEqual(routing["memory_type"], "connector")
        self.assertEqual(routing["apply_at"], "ppt_compilation")

    def test_required_motifs_come_from_the_touched_subgraphs(self):
        experience = _build(
            {"type": "relayout_global", "scope_type": "global"},
            visual_graph=VISUAL_GRAPH,
            bounded_change={"changed_node_ids": ["node_b"], "changed_asset_ids": []},
        )["experience"]
        self.assertEqual(experience["required_motifs"], ["linear"])
        self.assertEqual(experience["trigger"]["motifs"], ["linear"])

    def test_trigger_records_the_state_the_repair_was_applied_to(self):
        experience = _build(
            {"type": "relayout_global", "scope_type": "global"},
            previous={"overall_score": 0.5, "dimension_scores": {"density_fitness": 0.4}},
            current={"overall_score": 0.75, "blocking_issue_count": 0, "dimension_scores": {"density_fitness": 0.9}},
            domain_profile={"domain": "ai_method"},
            bounded_change={"changed_node_ids": ["node_a"], "object_diff": {"changed_fields": ["layout", "program"], "changed": True}},
        )["experience"]
        trigger = experience["trigger"]
        self.assertEqual(trigger["domain"], "ai_method")
        self.assertEqual(trigger["topology"], "pipeline")
        self.assertEqual(trigger["problem_metrics"], {"density_fitness": 0.4})
        self.assertEqual(trigger["object_state"]["changed_fields"], ["layout", "program"])
        self.assertEqual(trigger["object_state"]["changed_object_count"], 1)

    def test_observed_effect_keeps_sub_quantum_dimension_movement(self):
        experience = _build(
            {"type": "relayout_global", "scope_type": "global"},
            previous={"overall_score": 0.5, "dimension_scores": {"density_fitness": 0.40, "geometry_cleanliness": 1.0}},
            current={"overall_score": 0.75, "blocking_issue_count": 0, "dimension_scores": {"density_fitness": 0.46, "geometry_cleanliness": 1.0}},
        )["experience"]
        self.assertEqual(experience["observed_effect"]["dimension_deltas"], {"density_fitness": 0.06, "geometry_cleanliness": 0.0})

    def test_graph_scope_prefers_ids_the_executor_actually_touched(self):
        experience = _build(
            {"type": "relayout_global", "scope_type": "global"},
            bounded_change={"changed_node_ids": ["node_b"], "changed_asset_ids": ["icon_b"]},
        )["experience"]
        self.assertEqual(experience["graph_scope"]["object_ids"], ["icon_b", "node_b"])

    def test_judge_fingerprint_is_carried_onto_the_record(self):
        fingerprint = {"role": "frozen_heuristic", "mode": "heuristic", "trained": False, "model": ""}
        experience = _build({"type": "relayout_global", "scope_type": "global"}, judge_fingerprint=fingerprint)["experience"]
        self.assertEqual(experience["judge_fingerprint"], fingerprint)


class ExperienceLedgerHonestyTests(unittest.TestCase):
    """What a repair is recorded as having done has to be what it measurably did."""

    def test_the_object_diff_comes_from_the_action_and_the_compile_diff_stays_apart(self):
        experience = _build(
            {"type": "relayout_slots", "scope_type": "slot", "semantic_node_ids": ["node_a"]},
            bounded_change={"object_diff": {"changed_fields": [], "changed": False}, "changed_node_ids": []},
            compiled_state_diff={"changed_fields": ["program", "graphs"], "changed": True},
        )["experience"]
        self.assertFalse(experience["object_diff"]["changed"])
        self.assertEqual([], experience["observed_effect"]["changed_object_ids"])
        self.assertTrue(experience["compiled_object_diff"]["changed"])
        self.assertEqual(["program", "graphs"], experience["observed_effect"]["compiled_changed_fields"])

    def test_a_requested_id_is_never_reported_as_a_touched_id(self):
        experience = _build(
            {"type": "relayout_slots", "scope_type": "slot", "object_ids": ["conn_1", "conn_2"]},
            before=0.4,
            after=1.0,
        )["experience"]
        self.assertEqual([], experience["observed_effect"]["changed_object_ids"])
        self.assertEqual(0, experience["observed_effect"]["changed_object_count"])
        self.assertEqual(["conn_1", "conn_2"], experience["observed_effect"]["requested_object_ids"])
        self.assertEqual(["conn_1", "conn_2"], experience["graph_scope"]["object_ids"])
        self.assertEqual(0.4, experience["observed_effect"]["before_score"])
        self.assertEqual(1.0, experience["observed_effect"]["after_score"])
        self.assertTrue(validate_experience(experience)["ok"], validate_experience(experience)["errors"])

    def test_the_reflection_is_kept_as_an_unverified_claim(self):
        reflection = {"outcome": "improved", "why_changed": "the reflow cleared the crowd", "scope_assessment": "appropriate", "next_policy": "keep"}
        experience = _build({"type": "relayout_global", "scope_type": "global"}, reflection=reflection)["experience"]
        self.assertEqual("keep", experience["reported_reflection"]["next_policy"])
        self.assertEqual(reflection["why_changed"], experience["reported_reflection"]["why_changed"])
        self.assertFalse(experience["reported_reflection"]["verified"])
        self.assertTrue(validate_experience(experience)["ok"], validate_experience(experience)["errors"])

    def test_a_stored_reflection_cannot_claim_to_be_verified(self):
        experience = _build({"type": "relayout_global", "scope_type": "global"})["experience"]
        break_it = {**experience, "reported_reflection": {**experience["reported_reflection"], "verified": True}}
        result = validate_experience(break_it)
        self.assertFalse(result["ok"])
        self.assertTrue(any("cannot be stored as verified" in item for item in result["errors"]))

    def test_a_too_broad_claim_narrows_the_scope_only_when_the_footprint_supports_it(self):
        one_panel = {
            "topology": "pipeline",
            "nodes": [{"id": "node_a", "panel_id": "panel_1"}, {"id": "node_b", "panel_id": "panel_1"}],
            "connectors": [],
        }
        two_panels = {
            "topology": "pipeline",
            "nodes": [{"id": "node_a", "panel_id": "panel_1"}, {"id": "node_b", "panel_id": "panel_2"}],
            "connectors": [],
        }
        claim = {"scope_assessment": "too_broad", "next_policy": "narrow_scope"}
        narrowed = _build(
            {"type": "relayout_global", "scope_type": "global"},
            reflection=claim,
            layout=one_panel,
            bounded_change={"changed_node_ids": ["node_a", "node_b"]},
        )["experience"]
        self.assertEqual("slot", narrowed["scope"])
        self.assertTrue(narrowed["condition_evidence"]["scope_evidence"]["narrowed"])
        refused = _build(
            {"type": "relayout_global", "scope_type": "global"},
            reflection=claim,
            layout=two_panels,
            bounded_change={"changed_node_ids": ["node_a", "node_b"]},
        )["experience"]
        self.assertEqual("global", refused["scope"])
        self.assertFalse(refused["condition_evidence"]["scope_evidence"]["narrowed"])

    def test_a_domain_mismatch_claim_becomes_an_excluded_condition(self):
        experience = _build(
            {"type": "relayout_global", "scope_type": "global"},
            reflection={"domain_mismatch": True, "next_policy": "narrow_scope"},
            domain_profile={"domain": "ai_method"},
        )["experience"]
        self.assertIn({"dimension": "domain", "value": "ai_method", "reason": "reflection_domain_mismatch"}, experience["excluded_conditions"])
        self.assertTrue(validate_experience(experience)["ok"], validate_experience(experience)["errors"])

    def test_the_measured_budget_is_written_onto_the_action_another_paper_will_replay(self):
        result = _build(
            {"type": "relayout_global", "scope_type": "global"},
            bounded_change={"applied_budget": {"factor": 1.02, "dx": 0.016, "dy": 0.0}, "changed_node_ids": ["node_a", "node_b"]},
        )
        self.assertTrue(result["ok"])
        action = result["experience"]["action"]
        self.assertEqual(1.02, action["factor"])
        self.assertEqual(0.016, action["dx"])
        self.assertEqual(0.0, action["dy"])
        self.assertTrue(validate_experience(result["experience"])["ok"], validate_experience(result["experience"])["errors"])

    def test_the_shipped_nudge_is_not_written_as_memory(self):
        result = _build(
            {"type": "relayout_global", "scope_type": "global"},
            bounded_change={"applied_budget": {"factor": 1.02, "dx": 0.008, "dy": 0.0}, "changed_node_ids": ["node_a", "node_b"]},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("indistinguishable_from_default_policy", result["reason"])
        self.assertEqual("common_knowledge", result["mark"])
        self.assertIsNone(result["experience"])

    def test_a_named_slot_nudge_with_the_catalog_budget_is_still_memory(self):
        result = _build(
            {
                "type": "relayout_slots",
                "scope_type": "slot",
                "semantic_node_ids": ["node_a"],
                "factor": 1.08,
                "dx": 0.0,
                "dy": 0.0,
            },
            bounded_change={"applied_budget": {"factor": 1.08, "dx": 0.0, "dy": 0.0}, "changed_node_ids": ["node_a"]},
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(["node_a"], result["experience"]["action"]["semantic_node_ids"])
        self.assertTrue(validate_experience(result["experience"])["ok"], validate_experience(result["experience"])["errors"])

    def test_an_unnamed_slot_nudge_with_the_catalog_budget_is_still_refused(self):
        result = _build(
            {"type": "relayout_slots", "scope_type": "slot", "factor": 1.08, "dx": 0.0, "dy": 0.0},
            bounded_change={"applied_budget": {"factor": 1.08, "dx": 0.0, "dy": 0.0}, "changed_node_ids": ["node_a"]},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("indistinguishable_from_default_policy", result["reason"])
        self.assertEqual("common_knowledge", result["mark"])


class ExperienceStoreReasonTests(unittest.TestCase):
    def setUp(self):
        self.experience = _build({"type": "relayout_global", "scope_type": "global", "transferable_advice": ADVICE})["experience"]

    def test_unknown_status_is_refused_with_a_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(Path(temp))
            result = store.append_result({**self.experience, "status": "probably_fine"})
            self.assertFalse(result["ok"])
            self.assertEqual(result["reason"], "status_not_allowed")

    def test_inconclusive_is_an_allowed_status(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(Path(temp))
            result = store.append_result({**self.experience, "status": "inconclusive"})
            self.assertTrue(result["ok"], result)
            self.assertEqual("inconclusive", store.lifecycle_status(self.experience["experience_id"]))

    def test_contract_failure_carries_the_validator_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(Path(temp))
            result = store.append_result({**self.experience, "hypothesis": ""})
            self.assertFalse(result["ok"])
            self.assertEqual(result["reason"], "contract_validation_failed")
            self.assertTrue(result["errors"])

    def test_duplicate_identifier_is_refused_with_a_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(Path(temp))
            self.assertTrue(store.append_result(self.experience)["ok"])
            repeat = store.append_result(self.experience)
            self.assertFalse(repeat["ok"])
            self.assertEqual(repeat["reason"], "duplicate_experience_id")

    def test_integrity_report_surfaces_rows_the_store_skips(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(Path(temp))
            store.append(self.experience)
            with store.path.open("a", encoding="utf-8") as handle:
                handle.write("{not json\n")
                handle.write(json.dumps({"experience_id": "other", "domain": "some_other_domain"}) + "\n")
            report = store.integrity_report()
            self.assertEqual(report["readable_records"], 1)
            self.assertEqual(report["malformed_lines"], [2])
            self.assertEqual([item["domain"] for item in report["foreign_domain_rows"]], ["some_other_domain"])


class RetrievalBarTests(unittest.TestCase):
    """What a run is allowed to draw from, and how a bar it named is reported."""

    def _candidate(self, temp: str):
        built = _build(
            {"type": "relayout_global", "scope_type": "global", "transferable_advice": ADVICE},
            before=0.4,
            after=0.7,
        )
        store = ExperienceStore(Path(temp))
        # The ids and the boxes stay beside the row in a binding, and the reference to it is what
        # makes the stripped row readable as a lesson at all; this is the same split the run makes.
        binding = Path(temp) / "experience_binding.json"
        binding.write_text("{}", encoding="utf-8")
        row = experience_library_record(built["experience"], binding_ref=str(binding))
        stored = store.append_result(row, figure=LAYOUT)
        self.assertTrue(stored["ok"], stored)
        return store, row

    def test_the_default_bar_keeps_a_row_measured_on_its_own_paper_out(self):
        with tempfile.TemporaryDirectory() as temp:
            store, row = self._candidate(temp)
            self.assertEqual("candidate", store.lifecycle_status(row["experience_id"]))
            self.assertEqual([], store.retrieve(context={"topology": "pipeline", "node_count": 2, "connector_count": 1}))

    def test_the_relaxed_bar_offers_it_and_the_report_names_the_bar(self):
        with tempfile.TemporaryDirectory() as temp:
            store, row = self._candidate(temp)
            report = store.retrieve_with_report(
                context={"topology": "pipeline", "node_count": 2, "connector_count": 1},
                statuses=resolve_retrieval_statuses("relaxed"),
            )
            self.assertEqual([row["experience_id"]], [item["evidence"]["experience_id"] for item in report["rows"]])
            self.assertIn("candidate", report["report"]["statuses"])

    def test_a_bar_naming_a_rung_the_store_does_not_know_is_refused(self):
        with self.assertRaises(ValueError):
            resolve_retrieval_statuses("definitely_a_real_rung")
        self.assertIsNone(resolve_retrieval_statuses(""))
        self.assertEqual(set(DEFAULT_RETRIEVAL_STATUSES), resolve_retrieval_statuses("strict"))
        self.assertEqual({"refuted"}, resolve_retrieval_statuses(["refuted"]))


class UnmeasuredRollbackTests(unittest.TestCase):
    """The Gate decides what is kept; the round decides what the move showed.

    A rollback on a reading the seat itself doubts, or on a rule the figure never obeyed, has
    shown nothing about the move -- and a loss that was never measured is not a loss.
    """

    ACTION = {"type": "relayout_global", "scope_type": "global", "factor": 0.9}

    def test_a_kept_round_is_recorded_as_improved(self):
        result = _build(self.ACTION, round_decision={"accepted": True})
        self.assertEqual("improved", result["verdict"])
        self.assertEqual("improved", result["experience"]["verdict"])

    def test_a_reading_the_seat_doubts_is_recorded_as_unmeasured(self):
        result = _build(
            self.ACTION,
            before=0.6,
            after=1.0,
            round_decision={"accepted": False, "comparison": "pairwise", "pairwise_verdict": "after"},
            reflection={"outcome": "improved", "measurement_noise_risk": True},
        )
        self.assertEqual("unmeasured", result["verdict"])
        self.assertEqual("unmeasured", result["reason"])
        self.assertEqual("measurement_noise", result["experience"]["status"])
        self.assertEqual([], result["experience"]["excluded_conditions"])
        self.assertEqual([], result["experience"]["failed_cases"])
        self.assertNotIn("Do not reuse this move", result["experience"]["guidance"])
        self.assertTrue(validate_experience(result["experience"])["ok"], validate_experience(result["experience"])["errors"])

    def test_a_rule_the_figure_never_obeyed_is_recorded_as_unmeasured(self):
        result = _build(
            self.ACTION,
            round_decision={"accepted": False, "comparison": "pairwise", "pairwise_verdict": "before"},
            reflection={"outcome": "regressed", "attribution": "not_carried_out"},
        )
        self.assertEqual("unmeasured", result["verdict"])
        self.assertEqual("measurement_noise", result["experience"]["status"])
        self.assertEqual([], result["experience"]["failed_cases"])

    def test_a_round_that_cannot_tell_which_way_the_figure_moved_is_recorded_as_unmeasured(self):
        result = _build(
            self.ACTION,
            round_decision={"accepted": False, "comparison": "pairwise", "pairwise_verdict": "before"},
            reflection={"outcome": "inconclusive"},
        )
        self.assertEqual("unmeasured", result["verdict"])
        self.assertEqual("measurement_noise", result["experience"]["status"])

    def test_a_round_that_says_the_figure_got_worse_is_still_refuted(self):
        result = _build(
            self.ACTION,
            round_decision={"accepted": False, "comparison": "pairwise", "pairwise_verdict": "before"},
            reflection={"outcome": "regressed"},
        )
        self.assertEqual("worse", result["verdict"])
        self.assertEqual("refuted", result["reason"])
        self.assertEqual("refuted", result["experience"]["status"])
        self.assertEqual(["paper.pdf"], result["experience"]["failed_cases"])

    def test_a_rollback_with_no_reading_at_all_is_still_refuted(self):
        result = _build(self.ACTION, round_decision={"accepted": False, "comparison": "pairwise", "pairwise_verdict": "before"})
        self.assertEqual("worse", result["verdict"])
        self.assertEqual("refuted", result["experience"]["status"])

    def test_the_models_own_rule_is_kept_when_the_round_says_the_figure_got_worse(self):
        result = _build(
            self.ACTION,
            round_decision={"accepted": False, "comparison": "pairwise", "pairwise_verdict": "before"},
            reflection={"outcome": "regressed", "rule": {"guidance": "give the wider panel the row it needs"}},
        )
        self.assertEqual("worse", result["verdict"])
        self.assertEqual("give the wider panel the row it needs", result["experience"]["guidance"])

if __name__ == "__main__":
    unittest.main()
