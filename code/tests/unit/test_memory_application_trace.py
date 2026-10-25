"""The per-round audit of which Memory, patch and rule a run actually used."""

from __future__ import annotations

import unittest

from rfs.contracts.prompt_patch import PROMPT_ASSEMBLY_ORDER
from rfs.evolution.direct.trace import (
    MEMORY_APPLICATION_TRACE_CONTRACT,
    build_memory_application_trace,
)


ROUNDS = [
    {
        "round": 0,
        "accepted": True,
        "injected_experience": None,
        "applied_rules": {"segmentation": ["creator-rule-1"]},
        "applied_prompt_patches": {"ppt_compilation": ["prompt-patch-1"]},
        "prompt_patch_check": {"ok": True, "checked": [{"patch_id": "prompt-patch-1", "metric": "cross_panel_connectors", "op": "<=", "value": 2, "measured": 1.0}], "violations": [], "unchecked": []},
    },
    {
        "round": 1,
        "accepted": False,
        "injected_experience": {"experience_id": "direct-exp-injected", "stage": "segmentation", "source": "memory_retrieval"},
        "applied_rules": {"segmentation": ["creator-rule-1"]},
        "applied_prompt_patches": {"ppt_compilation": ["prompt-patch-1"]},
        "prompt_patch_check": {
            "ok": False,
            "checked": [{"patch_id": "prompt-patch-1", "metric": "unbound_scientific_nodes", "op": "<=", "value": 0, "measured": 1.0}],
            "violations": [{"patch_id": "prompt-patch-1", "metric": "unbound_scientific_nodes", "measured": 1.0, "reason": "unbound_scientific_nodes=1.0 breaks the prohibition unbound_scientific_nodes <= 0"}],
            "unchecked": [{"patch_id": "prompt-patch-3", "reason": "the prohibition names no measurable metric bound"}],
        },
    },
]

RETRIEVAL = [
    {"round": 0, "mode": "structure_only", "degraded": True, "candidates": 3, "statuses": ["promoted"], "reason": "no query or record text could be embedded"},
    {"round": 1, "mode": "structure_and_embedding", "degraded": False, "candidates": 2, "statuses": ["cross_domain_validated", "promoted", "same_domain_validated"], "reason": ""},
]

ROUTING = [
    {
        "round": 1,
        "stage": "segmentation",
        "selected": ["direct-exp-injected"],
        "reason": "1 experience applies at segmentation",
        "conflicts": [
            {"experience_id": "direct-exp-injected", "boundary_tier": 2, "median_repair_gain": 0.2, "support": 2, "selected": True},
            {"experience_id": "direct-exp-loser", "boundary_tier": 0, "median_repair_gain": -0.05, "support": 1, "selected": False},
        ],
    }
]

STAGE_PATCHES = {
    "ppt_compilation": {
        "considered": 2,
        "patches": [{"patch_id": "prompt-patch-1", "class": "executable_geometry"}],
        "rejected": [{"patch_id": "prompt-patch-2", "version": 1, "class": "advisory", "reason": "conflicts with prompt-patch-1 for relayout_slots, which declares the safer boundary"}],
        "reason": "1 patch(es) assembled for ppt_compilation",
    },
    "segmentation": {"considered": 0, "patches": [], "rejected": [], "reason": "no patch applies here"},
}

CREATOR_RULES = {
    "injected": {"segmentation": ["creator-rule-1"]},
    "withheld": [{"source_id": "direct-exp-thin", "apply_at": "ppt_compilation", "reason": "support_below_minimum", "support": 1}],
    "refuted_rules": [{"rule_id": "creator-rule-2", "status": "refuted", "refutations": 1}],
    "refutations": {"creator-rule-2": 1},
    "min_support": 2,
}


def _trace(**overrides) -> dict:
    arguments = {
        "rounds": ROUNDS,
        "retrieval_reports": RETRIEVAL,
        "routing_reports": ROUTING,
        "stage_patches": STAGE_PATCHES,
        "creator_rules": CREATOR_RULES,
        "experience_library": "<PATH>",
        "patch_library": "<PATH>",
    }
    arguments.update(overrides)
    return build_memory_application_trace(**arguments)


class MemoryApplicationTraceTests(unittest.TestCase):
    def test_the_trace_names_the_assembly_order_and_both_libraries(self):
        trace = _trace()

        self.assertEqual(trace["contract"], MEMORY_APPLICATION_TRACE_CONTRACT)
        self.assertEqual(trace["assembly_order"], list(PROMPT_ASSEMBLY_ORDER))
        self.assertEqual(trace["libraries"], {"experience": "<PATH>", "prompt_patch": "<PATH>"})

    def test_every_round_says_what_it_retrieved_injected_and_ran_under(self):
        trace = _trace()
        first, second = trace["rounds"]

        self.assertEqual(first["retrieval"]["mode"], "structure_only")
        self.assertTrue(first["retrieval"]["degraded"])
        self.assertEqual(first["retrieval"]["candidates"], 3)
        self.assertEqual(first["retrieval"]["statuses"], ["promoted"])
        self.assertEqual(second["retrieval"]["statuses"], ["cross_domain_validated", "promoted", "same_domain_validated"])
        self.assertIsNone(first["injected_experience"])
        self.assertEqual(first["applied_rules"], {"segmentation": ["creator-rule-1"]})
        self.assertEqual(second["injected_experience"]["experience_id"], "direct-exp-injected")
        self.assertTrue(first["prohibition_check"]["ok"])
        self.assertEqual(first["prohibition_check"]["checked"], ["prompt-patch-1"])
        self.assertFalse(second["prohibition_check"]["ok"])
        self.assertEqual([item["patch_id"] for item in second["prohibition_check"]["violations"]], ["prompt-patch-1"])
        self.assertEqual([item["patch_id"] for item in second["prohibition_check"]["unchecked"]], ["prompt-patch-3"])
        self.assertEqual(second["applied_prompt_patches"], {"ppt_compilation": ["prompt-patch-1"]})

    def test_a_loser_of_the_precedence_decision_is_recorded_with_its_reason(self):
        trace = _trace()

        self.assertEqual(trace["rounds"][0]["rejected_experiences"], [])
        rejected = trace["rounds"][1]["rejected_experiences"]
        self.assertEqual([item["experience_id"] for item in rejected], ["direct-exp-loser"])
        self.assertEqual(rejected[0]["reason"], "lost the precedence decision at segmentation")
        self.assertEqual(rejected[0]["support"], 1)

    def test_a_retrieval_report_without_a_status_list_reports_an_empty_set(self):
        trace = _trace(retrieval_reports=[{"round": 0, "mode": "structure_only", "degraded": True, "candidates": 1, "reason": ""}])

        self.assertEqual(trace["rounds"][0]["retrieval"]["statuses"], [])

    def test_a_patch_that_lost_its_stage_is_recorded_with_its_reason(self):
        trace = _trace()

        self.assertEqual(trace["prompt_patches"]["applied"], ["prompt-patch-1"])
        self.assertEqual(len(trace["prompt_patches"]["rejected"]), 1)
        loss = trace["prompt_patches"]["rejected"][0]
        self.assertEqual((loss["stage"], loss["patch_id"], loss["class"]), ("ppt_compilation", "prompt-patch-2", "advisory"))
        self.assertIn("safer boundary", loss["reason"])
        self.assertEqual(trace["prompt_patches"]["stages"]["segmentation"]["applied"], [])
        self.assertEqual(trace["prompt_patches"]["stages"]["ppt_compilation"]["considered"], 2)

    def test_the_rule_section_keeps_the_withheld_and_the_refuted_rules(self):
        trace = _trace()

        self.assertEqual(trace["rules"]["applied"], {"segmentation": ["creator-rule-1"]})
        self.assertEqual([item["source_id"] for item in trace["rules"]["withheld"]], ["direct-exp-thin"])
        self.assertEqual(trace["rules"]["refuted"], [{"rule_id": "creator-rule-2", "status": "refuted", "refutations": 1}])
        self.assertEqual(trace["rules"]["min_support"], 2)

    def test_the_totals_summarise_the_run(self):
        trace = _trace()

        self.assertEqual(
            trace["totals"],
            {
                "rounds": 2,
                "experiences_injected": ["direct-exp-injected"],
                "patches_applied": ["prompt-patch-1"],
                "patches_rejected": 1,
                "rules_applied": ["creator-rule-1"],
                "rules_withheld": 1,
                "rules_refuted": 1,
            },
        )

    def test_an_empty_run_traces_nothing_but_keeps_the_shape(self):
        trace = build_memory_application_trace()

        self.assertEqual(trace["rounds"], [])
        self.assertEqual(trace["prompt_patches"]["applied"], [])
        self.assertEqual(trace["rules"]["withheld"], [])
        self.assertEqual(trace["totals"]["rounds"], 0)
        self.assertEqual(trace["assembly_order"], list(PROMPT_ASSEMBLY_ORDER))


if __name__ == "__main__":
    unittest.main()
