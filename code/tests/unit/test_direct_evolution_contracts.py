"""Tests for direct self-evolution contracts."""

import unittest

from rfs.contracts.evolution import (
    EXPERIENCE_SCHEMA_VERSION,
    normalize_experience,
    normalize_judge_result,
    validate_experience,
    validate_judge_result,
)


MEASURED = {
    "repair_gain": 0.25,
    "provenance": {"source": "direct_evolution_process", "run": "paper.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.75},
}


class DirectEvolutionContractTests(unittest.TestCase):
    def test_judge_feedback_requires_object_binding_for_slot_scope(self):
        result = validate_judge_result({
            "status": "fault",
            "stage_feedback": [{"scope_type": "slot", "problem": "icon too small"}],
        })
        self.assertFalse(result["ok"])

    def test_judge_actions_are_limited_to_bounded_vocabulary(self):
        result = normalize_judge_result({
            "stage_feedback": [{
                "scope_type": "slot",
                "slot_ids": ["slot_a"],
                "problem": "wrong icon",
                "repair": [{"action": "regen_assets"}, {"action": "write_arbitrary_code"}],
            }]
        })
        self.assertEqual([item["action"] for item in result["stage_feedback"][0]["requested_actions"]], ["regen_assets"])

    def test_experience_has_explicit_lifecycle_status(self):
        experience = normalize_experience({"experience_id": "e1", "status": "cross_domain_validated"})
        self.assertEqual(experience["status"], "cross_domain_validated")

    def test_the_retired_cross_paper_rung_is_no_longer_a_status(self):
        # cross_paper_validated named the rung cross_domain_validated now names, so the name is
        # gone: a row still carrying it reads as the first rung, not as evidence.
        experience = normalize_experience({"experience_id": "e1", "status": "cross_paper_validated"})
        self.assertEqual(experience["status"], "candidate")

    def test_a_domain_experience_without_a_paper_domain_is_rejected(self):
        result = validate_experience({
            "experience_id": "e1",
            "scope": "slot",
            "hypothesis": "h",
            "action": {"type": "regen_assets"},
            "applicability": "domain",
            "applicability_domain": "",
            **MEASURED,
        })
        self.assertFalse(result["ok"])
        self.assertTrue(any("named paper domain" in item for item in result["errors"]))

    def test_experience_requires_hypothesis_and_bounded_action(self):
        result = validate_experience({"experience_id": "e1", "scope": "slot", "action": {"type": "regen_assets"}})
        self.assertFalse(result["ok"])

    def test_typed_memory_fields_default_to_a_valid_repair_record(self):
        experience = normalize_experience({"experience_id": "e1"})
        self.assertEqual(experience["memory_type"], "repair")
        self.assertEqual(experience["apply_at"], "repair")
        self.assertEqual(experience["required_motifs"], [])
        self.assertEqual(experience["judge_fingerprint"], {"role": "", "mode": "", "trained": False, "model": "", "checkpoint": ""})

    def test_records_written_before_the_contract_grew_normalise_without_new_fields(self):
        experience = normalize_experience({"experience_id": "e1", "status": "cross_domain_validated"})
        self.assertEqual(EXPERIENCE_SCHEMA_VERSION, experience["schema_version"])
        self.assertEqual([], experience["excluded_conditions"])
        self.assertEqual({}, experience["object_diff"])
        self.assertEqual({}, experience["compiled_object_diff"])
        self.assertEqual({}, experience["reported_reflection"])
        self.assertEqual({}, experience["condition_evidence"])
        self.assertIsNone(experience["repair_gain"])
        self.assertFalse(experience["accepted"])
        self.assertEqual("", experience["provenance"]["source"])
        self.assertFalse(experience["provenance"]["measured"])

    def test_unsupported_memory_type_and_stage_are_rejected(self):
        base = {"experience_id": "e1", "scope": "slot", "hypothesis": "h", "action": {"type": "regen_assets"}, "graph_scope": {"slot_ids": ["s"]}, **MEASURED}
        self.assertTrue(validate_experience(base)["ok"])
        bad_type = validate_experience({**base, "memory_type": "vibes"})
        self.assertFalse(bad_type["ok"])
        self.assertTrue(any("memory_type" in item for item in bad_type["errors"]))
        bad_stage = validate_experience({**base, "apply_at": "whenever"})
        self.assertFalse(bad_stage["ok"])
        self.assertTrue(any("apply_at" in item for item in bad_stage["errors"]))

    def test_a_selection_memory_is_accepted_and_plans_globally(self):
        """The author's trade-off is a memory type, applied where content is chosen."""
        base = {"experience_id": "e1", "scope": "slot", "hypothesis": "h", "action": {"type": "regen_assets"}, "graph_scope": {"slot_ids": ["s"]}, **MEASURED}
        accepted = validate_experience({**base, "memory_type": "selection", "apply_at": "global_planning"})

        self.assertTrue(accepted["ok"], accepted["errors"])
        self.assertEqual("selection", normalize_experience({**base, "memory_type": "selection"})["memory_type"])

    def test_required_motifs_must_come_from_the_motif_vocabulary(self):
        base = {"experience_id": "e1", "scope": "slot", "hypothesis": "h", "action": {"type": "regen_assets"}, "graph_scope": {"slot_ids": ["s"]}, **MEASURED}
        self.assertTrue(validate_experience({**base, "required_motifs": ["linear", "feedback"]})["ok"])
        result = validate_experience({**base, "required_motifs": ["spaghetti"]})
        self.assertFalse(result["ok"])
        self.assertTrue(any("required_motifs" in item for item in result["errors"]))

    def test_experience_rejects_legacy_whole_image_fields(self):
        legacy_field = "approved_" + "image"
        result = validate_experience({"experience_id": "e1", "scope": "slot", "hypothesis": "targeted repair", "action": {"type": "regen_assets"}, "graph_scope": {"slot_ids": ["slot_a"]}, legacy_field: "legacy.png"})
        self.assertFalse(result["ok"])
        self.assertTrue(any("legacy whole-image" in item for item in result["errors"]))

    def test_experience_requires_a_measured_repair_gain_and_a_traceable_run(self):
        base = {"experience_id": "e1", "scope": "slot", "hypothesis": "h", "action": {"type": "regen_assets"}, "graph_scope": {"slot_ids": ["s"]}}
        unmeasured = validate_experience(base)
        self.assertFalse(unmeasured["ok"])
        self.assertTrue(any("repair_gain is required" in item for item in unmeasured["errors"]))
        unmeasured = validate_experience({**base, "repair_gain": 0.25})
        self.assertTrue(any("provenance must record a measurement" in item for item in unmeasured["errors"]))
        self.assertTrue(any("provenance must name the source" in item for item in unmeasured["errors"]))
        mismatched = validate_experience({**base, "repair_gain": 0.4, **{key: value for key, value in MEASURED.items() if key == "provenance"}})
        self.assertTrue(any("disagrees with its measured provenance" in item for item in mismatched["errors"]))
        self.assertTrue(validate_experience({**base, **MEASURED})["ok"])

    def test_a_refuted_experience_must_name_the_condition_it_excludes(self):
        measured = {"repair_gain": -0.25, "provenance": {"source": "controlled_negative_round", "run": "run/round_01", "round": 1, "measured": True, "before_score": 0.75, "after_score": 0.5}}
        base = {"experience_id": "e1", "scope": "slot", "hypothesis": "h", "action": {"type": "regen_assets"}, "graph_scope": {"slot_ids": ["s"]}, "status": "refuted", **measured}
        self.assertTrue(any("must name the conditions it excludes" in item for item in validate_experience(base)["errors"]))
        refuted = validate_experience({**base, "excluded_conditions": [{"topology": "pipeline", "reason": "repair_gain_not_positive"}]})
        self.assertTrue(refuted["ok"], refuted["errors"])
        self.assertEqual("refuted", refuted["experience"]["status"])
        forged = validate_experience({**base, "excluded_conditions": [{"topology": "pipeline"}], "repair_gain": 0.25, "provenance": {**measured["provenance"], "before_score": 0.5, "after_score": 0.75}})
        self.assertTrue(any("cannot record a positive repair_gain" in item for item in forged["errors"]))

    def test_an_accepted_experience_must_record_a_positive_gain(self):
        base = {"experience_id": "e1", "scope": "slot", "hypothesis": "h", "action": {"type": "regen_assets"}, "graph_scope": {"slot_ids": ["s"]}, "accepted": True}
        errors = validate_experience({**base, "repair_gain": 0.0, "provenance": {"source": "s", "run": "r", "round": 0, "measured": True}})["errors"]
        self.assertTrue(any("accepted experience must record a positive" in item for item in errors), errors)

    def test_pairwise_acceptance_may_keep_a_non_positive_gain(self):
        base = {
            "experience_id": "e1",
            "scope": "slot",
            "hypothesis": "h",
            "action": {"type": "relayout_global"},
            "graph_scope": {"object_ids": ["n1"]},
            "accepted": True,
            "comparison": "pairwise",
            "pairwise_verdict": "after",
            "repair_gain": 0.0,
            "provenance": {"source": "s", "run": "r", "round": 1, "measured": True, "before_score": 1.0, "after_score": 1.0},
        }
        result = validate_experience(base)
        self.assertTrue(result["ok"], result["errors"])
        missing_verdict = validate_experience({**base, "pairwise_verdict": "before"})
        self.assertTrue(any("pairwise_verdict=after" in item for item in missing_verdict["errors"]), missing_verdict["errors"])

    def test_a_pairwise_refuted_row_may_carry_a_positive_score_delta(self):
        result = validate_experience({
            "experience_id": "e1",
            "scope": "slot",
            "hypothesis": "h",
            "action": {"type": "relayout_global"},
            "graph_scope": {"object_ids": ["n1"]},
            "accepted": False,
            "status": "refuted",
            "comparison": "pairwise",
            "pairwise_verdict": "before",
            "excluded_conditions": [{"topology": "pipeline", "reason": "pairwise_not_preferred"}],
            "repair_gain": 0.2,
            "provenance": {"source": "s", "run": "r", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.7},
        })
        self.assertTrue(result["ok"], result["errors"])


if __name__ == "__main__":
    unittest.main()
