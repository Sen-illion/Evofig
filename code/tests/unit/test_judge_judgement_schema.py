"""Tests for the unified Judge judgement contract and its deterministic labels."""

import unittest

from rfs.judge_training.judge_schema import (
    DIAGNOSIS_TYPES,
    GATE_DIAGNOSIS_TYPES,
    JUDGE_CONTRACT_VERSION,
    QUALITY_DIMENSIONS,
    defect_label,
    entity_ids,
    grounding_field,
    has_blocking_diagnosis,
    human_verdict_quality_scores,
    is_legacy_dataset_version,
    normalize_judgement,
    pairwise_choice,
    scientific_score,
    validate_judgement,
)
from rfs.judge_training.prompts import PROMPT_VERSION, build_prompt, output_schema
from rfs.judge_training.schemas import normalize_example, validate_example


def _judgement(**overrides):
    value = {
        "preferred_candidate": "b",
        "quality_scores": {dimension: 0.8 for dimension in QUALITY_DIMENSIONS},
        "diagnosis": [{"figure": "a", "type": "Wrong_Arrow_Target", "target": "encoder_to_fusion", "severity": "HIGH", "evidence": "arrow ends at decoder"}],
        "grounding": {"grounded_entities": ["encoder"]},
        "confidence": 0.7,
    }
    value.update(overrides)
    return value


class JudgeJudgementContractTests(unittest.TestCase):
    def test_contract_version_and_vocabulary_are_frozen(self):
        self.assertEqual(JUDGE_CONTRACT_VERSION, "judge-judgement-v1")
        self.assertEqual(GATE_DIAGNOSIS_TYPES, ("module_order_error", "wrong_arrow_target", "overlap"))
        self.assertEqual(len(QUALITY_DIMENSIONS), 5)
        self.assertTrue(set(GATE_DIAGNOSIS_TYPES) <= set(DIAGNOSIS_TYPES))

    def test_valid_pairwise_judgement_passes(self):
        self.assertEqual([], validate_judgement(_judgement(), task_type="pairwise_preference"))
        self.assertEqual([], validate_judgement(_judgement(), task_type="error_audit"))

    def test_quality_scores_and_confidence_are_ranged(self):
        judgement = _judgement(quality_scores={"scientific_fidelity": 1.4, "logical_flow": 0.5})
        errors = validate_judgement(judgement, task_type="scientific_quality")
        self.assertIn("quality_scores.scientific_fidelity must be a number 0..1", errors)
        self.assertIn("quality_scores.layout_quality must be a number 0..1", errors)
        self.assertIn("confidence must be a number 0..1", validate_judgement(_judgement(confidence="high"), task_type="scientific_quality"))
        self.assertIn("quality_scores is required", validate_judgement(_judgement(quality_scores=[]), task_type="scientific_quality"))

    def test_diagnosis_entries_are_validated(self):
        errors = validate_judgement(_judgement(diagnosis="order is wrong"), task_type="scientific_quality")
        self.assertIn("diagnosis must be an array", errors)
        errors = validate_judgement(_judgement(diagnosis=["order is wrong"]), task_type="scientific_quality")
        self.assertIn("diagnosis entries must be objects", errors)
        errors = validate_judgement(_judgement(diagnosis=[{"type": "looks_off", "target": "x", "severity": "high", "evidence": "e"}]), task_type="scientific_quality")
        self.assertIn("diagnosis type is unsupported: 'looks_off'", errors)
        errors = validate_judgement(_judgement(diagnosis=[{"type": "overlap", "target": "", "severity": "severe", "evidence": ""}]), task_type="scientific_quality")
        for expected in ("diagnosis entry requires a target", "diagnosis severity is unsupported: 'severe'", "diagnosis entry requires evidence"):
            self.assertIn(expected, errors)
        errors = validate_judgement(_judgement(diagnosis=[{"type": "overlap", "target": "n1", "severity": "low", "evidence": "e", "figure": "C"}]), task_type="scientific_quality")
        self.assertIn("diagnosis figure must be A or B: 'C'", errors)

    def test_pairwise_and_grounding_requirements(self):
        self.assertIn("pairwise judgements require preferred_candidate A, B or tie", validate_judgement(_judgement(preferred_candidate=None), task_type="image_pairwise"))
        self.assertEqual([], validate_judgement(_judgement(preferred_candidate="tie"), task_type="image_pairwise"))
        errors = validate_judgement(_judgement(grounding=None), task_type="method_grounding")
        self.assertIn("grounding is required for grounding tasks", errors)
        errors = validate_judgement(_judgement(grounding={"grounded_entities": "encoder", "relation_directions": []}), task_type="error_audit")
        self.assertIn("grounding.grounded_entities must be an array", errors)
        self.assertIn("grounding.relation_directions must be an object", errors)

    def test_normalization_keeps_metadata_and_case_folds(self):
        judgement = normalize_judgement(_judgement(pair_group="g1", quality_label="坏"))
        self.assertEqual("B", judgement["preferred_candidate"])
        self.assertEqual("wrong_arrow_target", judgement["diagnosis"][0]["type"])
        self.assertEqual("high", judgement["diagnosis"][0]["severity"])
        self.assertEqual("A", judgement["diagnosis"][0]["figure"])
        self.assertEqual("g1", judgement["pair_group"])
        self.assertEqual("坏", judgement["quality_label"])
        self.assertEqual({}, normalize_judgement("not a judgement"))

    def test_accessors_read_only_the_unified_fields(self):
        judgement = _judgement()
        self.assertEqual(0.8, scientific_score(judgement))
        self.assertEqual("B", pairwise_choice(judgement))
        self.assertEqual(["encoder"], grounding_field(judgement, "grounded_entities"))
        self.assertEqual({}, grounding_field(judgement, "relation_directions"))
        self.assertIsNone(scientific_score({"scientific_score": 0.1}))
        self.assertEqual("", pairwise_choice({"aesthetic_preference": "A"}))

    def test_blocking_diagnosis_excludes_style_only_defects(self):
        self.assertTrue(has_blocking_diagnosis(_judgement()))
        ugly = {"type": "correct_but_ugly", "target": "panel", "severity": "low", "evidence": "tiny text"}
        self.assertFalse(has_blocking_diagnosis(_judgement(diagnosis=[ugly])))
        self.assertFalse(has_blocking_diagnosis(_judgement(diagnosis=[])))
        self.assertTrue(has_blocking_diagnosis(_judgement(diagnosis=[ugly, {"type": "overlap", "target": "panel", "severity": "high", "evidence": "labels collide"}])))
        self.assertFalse(has_blocking_diagnosis({"error_types": ["relation_reversal"]}))

    def test_defect_labels_map_sources_onto_the_gate_vocabulary(self):
        self.assertEqual("wrong_arrow_target", defect_label("relation_reversal")["diagnosis_type"])
        self.assertEqual("wrong_arrow_target", defect_label("missing_relation")["diagnosis_type"])
        self.assertEqual("module_order_error", defect_label("module_order_error")["diagnosis_type"])
        self.assertEqual("overlap", defect_label("overlap")["diagnosis_type"])
        self.assertEqual("low", defect_label("correct_but_ugly")["severity"])
        self.assertEqual(set(QUALITY_DIMENSIONS), set(defect_label("overlap")["quality_scores"]))
        with self.assertRaises(KeyError):
            defect_label("not_a_defect")

    def test_human_verdicts_expand_to_contract_scores(self):
        self.assertTrue(all(value == 1.0 for value in human_verdict_quality_scores("好（科研）").values()))
        self.assertEqual(0.0, human_verdict_quality_scores("坏")["scientific_fidelity"])
        self.assertEqual(1.0, human_verdict_quality_scores("坏")["readability"])
        with self.assertRaises(KeyError):
            human_verdict_quality_scores("maybe")

    def test_entity_ids_accepts_ids_and_entity_objects(self):
        self.assertEqual(["a", "b", "c"], entity_ids([{"entity_id": "a"}, "b", {"name": "c"}, {"entity_id": "a"}]))

    def test_legacy_dataset_versions_are_flagged(self):
        self.assertTrue(is_legacy_dataset_version("judge-d1-v001"))
        self.assertTrue(is_legacy_dataset_version("judge-dataset-v001"))
        self.assertFalse(is_legacy_dataset_version("judge-dataset-v002"))


class JudgePromptContractTests(unittest.TestCase):
    def test_prompt_uses_the_shared_contract(self):
        prompt = build_prompt({"task_type": "pairwise_preference", "paper": {"title": "t"}, "ground_truth": {}, "domain": "cv", "domain_rubric": {"required_dimensions": ["scientific_fidelity"]}})
        self.assertIn("quality_scores", prompt)
        self.assertIn("preferred_candidate", prompt)
        self.assertIn("domain_rubric", prompt)
        self.assertNotIn("scientific_score", prompt)
        self.assertEqual("judge-online-v031", PROMPT_VERSION)
        self.assertEqual({"state_after", "repair_gain", "grounding_gate", "decision", "audit_status"}, set(output_schema("outcome")))
        self.assertIn("preferred_candidate", output_schema("style_preference"))
        self.assertIn("grounding", output_schema("error_audit"))
        self.assertNotIn("grounding", output_schema("scientific_quality"))

    def test_the_reflection_asks_for_the_fields_that_carry_a_lesson_onward(self):
        # The field list is what a model answers to: the paragraphs below it described
        # ``prompt_patch`` and ``claims`` from v021 on, and every round of that run still wrote an
        # empty patch and a bare claims list, so a patch could never be injected and a claim could
        # never be settled. The list itself has to name them.
        from rfs.judge_training.prompts import build_reflection_prompt

        prompt = build_reflection_prompt({"outcome": "regressed"})
        enumerated = [sentence for sentence in prompt.split(". ") if "Return JSON only with" in sentence]
        self.assertEqual(1, len(enumerated))
        self.assertIn("prompt_patch", enumerated[0])
        self.assertIn("claims", enumerated[0])
        # v027: a reflection is the only thing that can finish with a layer, so the field
        # that says so has to be named where the seat reads the answer it must write.
        self.assertIn("layer_settled", enumerated[0])

    def test_a_reference_figure_block_rides_along_only_when_the_sample_carries_one(self):
        block = {
            "available": True,
            "roles": {"method_object": 3, "stage_annotation": 1},
            "repeat_drawn": [{"label": ["PVCNN Diffusion"], "group": "g1", "index": 1, "count": 2}],
            "notation_regions": [{"kind": "legend", "explains": "conv 3x3", "label": [], "entries": 4}],
            "objects": 4,
        }
        base = {"task_type": "pairwise", "judge_role": "frozen_bootstrap", "paper": {"title": "t"}, "domain": "cv"}
        with_figure = build_prompt({**base, "ground_truth": {"reference_figure": block}})
        without = build_prompt({**base, "ground_truth": {}})
        empty = build_prompt({**base, "ground_truth": {"reference_figure": {}}})

        self.assertIn('"reference_figure"', with_figure)
        self.assertIn('"method_object"', with_figure)
        self.assertIn("PVCNN Diffusion", with_figure)
        self.assertNotIn("reference_figure", without)
        self.assertEqual(without, empty)

    def test_the_two_state_question_is_asked_in_the_training_vocabulary(self):
        # The loop decides a round on a comparison, so the served question has to be the one the
        # training contract can supervise: the same field name, the same two answers, no score.
        schema = output_schema("pairwise")
        self.assertIn("preferred_candidate", schema)
        self.assertIn("margin", schema)
        self.assertNotIn("overall_score", schema)
        prompt = build_prompt({
            "task_type": "pairwise",
            "judge_role": "frozen_bootstrap",
            "paper": {"title": "t"},
            "ground_truth": {},
            "domain": "cv",
            "domain_rubric": {"required_dimensions": ["scientific_fidelity"]},
        })
        self.assertIn("preferred_candidate", prompt)
        self.assertIn("the first is A", prompt)
        self.assertIn("margin", prompt)
        self.assertNotIn("overall_score", prompt)

    def test_each_role_is_asked_only_for_its_own_answer(self):
        advisory = output_schema("repair", role="online_bootstrap")
        acceptance = output_schema("repair", role="frozen_bootstrap")
        self.assertNotIn("overall_score", advisory)
        self.assertIn("stage_feedback", advisory)
        self.assertIn("overall_score", acceptance)
        self.assertNotIn("stage_feedback", acceptance)

    def test_the_advisory_prompt_carries_the_action_catalog_and_the_others_do_not(self):
        advisory = build_prompt({"task_type": "repair", "judge_role": "online_bootstrap", "paper": {"title": "t"}, "ground_truth": {}, "domain": "cv"})
        acceptance = build_prompt({"task_type": "repair", "judge_role": "frozen_bootstrap", "paper": {"title": "t"}, "ground_truth": {}, "domain": "cv"})
        pairwise = build_prompt({"task_type": "pairwise", "judge_role": "frozen_bootstrap", "paper": {"title": "t"}, "ground_truth": {}, "domain": "cv"})
        self.assertIn("relayout_global:", advisory)
        self.assertIn("does not re-run structure planning", advisory)
        self.assertIn("empty requested_actions list", advisory)
        self.assertIn("factor", advisory)
        self.assertNotIn("relayout_global:", acceptance)
        self.assertNotIn("relayout_global:", pairwise)
        schema = output_schema("repair", role="online_bootstrap")
        keys = schema["stage_feedback"][0]["requested_actions"][0]
        self.assertIn("factor", keys)
        self.assertIn("dx", keys)
        self.assertNotIn("requested_actions", output_schema("repair", role="frozen_bootstrap"))
        self.assertNotIn("requested_actions", output_schema("pairwise"))


class JudgeExampleContractTests(unittest.TestCase):
    def test_normalize_example_keeps_group_keys_and_checks_judgement_labels(self):
        example = normalize_example({
            "example_id": "error_audit-1",
            "ground_truth_id": "arxiv-1",
            "task_type": "error_audit",
            "split": "validation",
            "paper": {"title": "t"},
            "ground_truth": {"required_entities": ["encoder"]},
            "candidate_a": "a.png",
            "pair_group": "pair-1",
            "repeat_group": "repeat-1",
            "labels": _judgement(),
        })
        self.assertEqual("pair-1", example["pair_group"])
        self.assertEqual("repeat-1", example["repeat_group"])
        self.assertEqual([], validate_example(example))

    def test_validate_example_rejects_retired_label_fields(self):
        example = normalize_example({
            "example_id": "scientific_quality-1",
            "ground_truth_id": "arxiv-1",
            "task_type": "scientific_quality",
            "split": "train",
            "paper": {"title": "t"},
            "ground_truth": {"required_entities": ["encoder"]},
            "candidate_a": "a.png",
            "labels": {"error_types": ["relation_reversal"], "scientific_score": 0.1, "blocking_issues": ["arrow reversed"]},
        })
        errors = validate_example(example)
        self.assertTrue(any(error.startswith("labels:") for error in errors))
        self.assertIn("labels: quality_scores is required", errors)

    def test_normalize_example_still_accepts_process_tasks(self):
        example = normalize_example({"task_type": "outcome", "example_id": "x", "ground_truth_id": "g", "preview": "p.png", "labels": {"decision": "accept"}})
        self.assertEqual("outcome", example["task_type"])
        self.assertEqual([], validate_example(example))


if __name__ == "__main__":
    unittest.main()
