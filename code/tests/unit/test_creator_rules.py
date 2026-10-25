"""Aggregation, precedence and refutation behaviour of the measured Creator rules."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rfs.contracts.creator_rules import (
    CREATOR_RULE_MANIFEST_CONTRACT,
    CREATOR_RULE_MIN_SUPPORT,
    CREATOR_RULE_REFUTATION_LIMIT,
    CREATOR_RULE_STAGE4_KEYS,
    creator_rule_id,
    normalize_creator_rule_manifest,
    validate_creator_rule_manifest,
)
from rfs.contracts.prompt_patch import validate_prompt_patch
from rfs.evolution.direct.creator_rules import (
    CREATOR_RULE_REFUTING_REASONS,
    CreatorRuleLedger,
    build_creator_rule_manifest,
    read_creator_rule_manifest,
    refuting_rule_ids,
)
from rfs.evolution.direct.prompt_patch import build_prompt_patch


CONNECTOR_CONDITION = {"metric": "cross_panel_connectors", "op": "<=", "value": 2}
OCCUPANCY_CONDITION = {"metric": "min_panel_occupancy", "op": ">=", "value": 0.5}


def _experience(identifier: str, **overrides) -> dict:
    """Build a measured Memory row in the shape the store writes."""

    row = {
        "experience_id": identifier,
        "domain": "direct_editable",
        "apply_at": "ppt_compilation",
        "action": {"type": "relayout_slots", "scope_type": "slot", "factor": 0.9},
        "condition": dict(CONNECTOR_CONDITION),
        "accepted": True,
        "repair_gain": 0.12,
        "validated_cases": ["a.pdf", "b.pdf"],
        "status": "cross_domain_validated",
        "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True},
    }
    row.update(overrides)
    return row


class CreatorRuleAggregationTests(unittest.TestCase):
    def test_rows_claiming_one_stage_and_action_aggregate_into_one_rule(self):
        manifest = build_creator_rule_manifest([_experience("direct-exp-a"), _experience("direct-exp-b", repair_gain=0.2)])

        self.assertEqual(len(manifest["rules"]), 1)
        rule = manifest["rules"][0]
        self.assertEqual(rule["contract"], CREATOR_RULE_MANIFEST_CONTRACT)
        self.assertEqual(rule["apply_at"], "ppt_compilation")
        self.assertEqual(rule["class"], "executable_geometry")
        self.assertEqual(rule["support"], 2)
        self.assertEqual(rule["median_repair_gain"], 0.2)
        self.assertEqual(rule["source_experience_ids"], ["direct-exp-a", "direct-exp-b"])
        self.assertEqual(manifest["buckets"], {"ppt_compilation": [rule["rule_id"]]})

    def test_two_stages_produce_two_rules_and_two_buckets(self):
        manifest = build_creator_rule_manifest(
            [
                _experience("direct-exp-seg", apply_at="segmentation", action={"type": "resegment"}),
                _experience("direct-exp-ppt"),
            ]
        )

        self.assertEqual(sorted(manifest["buckets"]), ["ppt_compilation", "segmentation"])
        self.assertEqual(len(manifest["rules"]), 2)

    def test_a_thin_row_stays_out_and_is_reported_as_withheld(self):
        thin = _experience("direct-exp-thin", status="candidate", validated_cases=[], repair_gain=0.05)

        manifest = build_creator_rule_manifest([thin])

        self.assertEqual(manifest["rules"], [])
        self.assertEqual(manifest["withheld"], [{"source_id": "direct-exp-thin", "apply_at": "ppt_compilation", "reason": "support_below_minimum", "support": 1}])
        self.assertEqual(manifest["min_support"], CREATOR_RULE_MIN_SUPPORT)

    def test_a_validated_rung_aggregates_on_its_own_evidence(self):
        single = _experience("direct-exp-one", status="promoted", validated_cases=["a.pdf"])

        manifest = build_creator_rule_manifest([single])

        self.assertEqual(len(manifest["rules"]), 1)
        self.assertEqual(manifest["rules"][0]["support"], 1)
        self.assertEqual(manifest["withheld"], [])

    def test_a_withdrawn_row_never_reaches_a_rule(self):
        healthy = _experience("direct-exp-a")
        quarantined = _experience("direct-exp-b", status="retrying", validated_cases=[f"case-{index}.pdf" for index in range(5)])

        manifest = build_creator_rule_manifest([healthy, quarantined])

        self.assertEqual([rule["source_experience_ids"] for rule in manifest["rules"]], [["direct-exp-a"]])
        self.assertEqual(manifest["withheld"][0], {"source_id": "direct-exp-b", "apply_at": "ppt_compilation", "reason": "status_withdrawn", "status": "retrying"})

    def test_a_prohibition_wins_its_group_against_an_advisory_claim(self):
        advisory = _experience("direct-exp-advisory", action={"type": "resegment"}, repair_gain=0.3)
        prohibition = _experience("direct-exp-prohibition", action={"type": "resegment"}, accepted=False, repair_gain=-0.05)

        manifest = build_creator_rule_manifest([advisory, prohibition])

        self.assertEqual(len(manifest["rules"]), 1)
        self.assertEqual(manifest["rules"][0]["class"], "prohibited")
        self.assertEqual(
            manifest["rules"][0]["rule_id"],
            creator_rule_id(apply_at="ppt_compilation", rule_class="prohibited", action_type="resegment", condition=CONNECTOR_CONDITION),
        )

    def test_the_same_rows_always_produce_the_same_rule_ids(self):
        rows = [_experience("direct-exp-a"), _experience("direct-exp-b", condition=dict(OCCUPANCY_CONDITION))]

        first = build_creator_rule_manifest(rows)
        second = build_creator_rule_manifest(rows)

        self.assertEqual([rule["rule_id"] for rule in first["rules"]], [rule["rule_id"] for rule in second["rules"]])
        self.assertEqual(len({rule["rule_id"] for rule in first["rules"]}), 2)


class CreatorRuleRefutationTests(unittest.TestCase):
    def test_a_refuted_rule_leaves_the_active_list_and_names_its_condition(self):
        rows = [_experience("direct-exp-a")]
        rule_id = build_creator_rule_manifest(rows)["rules"][0]["rule_id"]

        once = build_creator_rule_manifest(rows, refutations={rule_id: 1})
        twice = build_creator_rule_manifest(rows, refutations={rule_id: CREATOR_RULE_REFUTATION_LIMIT})

        self.assertEqual(once["rules"], [])
        self.assertEqual(once["refuted_rules"][0]["status"], "refuted")
        self.assertTrue(once["refuted_rules"][0]["excluded_conditions"])
        self.assertEqual(twice["refuted_rules"][0]["status"], "disabled")
        self.assertEqual(twice["refuted_rules"][0]["refutations"], 2)
        self.assertEqual(once["stage4_manifest"]["evidence_references"], [])

    def test_a_confirmation_resets_the_refutation_streak(self):
        with tempfile.TemporaryDirectory() as name:
            ledger = CreatorRuleLedger(Path(name) / "library")

            self.assertEqual(ledger.refutation_count("creator-rule-x"), 0)
            ledger.record_refutation("creator-rule-x", round_index=1, reason="score dropped")
            ledger.record_refutation("creator-rule-x", round_index=2, reason="score dropped")
            self.assertEqual(ledger.refutation_count("creator-rule-x"), 2)
            self.assertEqual(ledger.counters(), {"creator-rule-x": 2})
            ledger.record_confirmation("creator-rule-x", round_index=3)
            self.assertEqual(ledger.refutation_count("creator-rule-x"), 0)
            self.assertEqual(ledger.counters(), {})
            ledger.record_refutation("creator-rule-x", round_index=4, reason="score dropped")
            self.assertEqual(ledger.refutation_count("creator-rule-x"), 1)
            self.assertTrue((Path(name) / "library" / "creator_rule_events.jsonl").exists())


    def test_a_release_clears_the_streak_a_confirmation_could_not(self):
        with tempfile.TemporaryDirectory() as name:
            ledger = CreatorRuleLedger(Path(name) / "library")
            ledger.record_refutation("creator-rule-x", round_index=1, reason="the guided round lost")
            ledger.record_refutation("creator-rule-x", round_index=2, reason="the guided round lost")

            ledger.release("creator-rule-x", round_index=3, reason="the operator let it back in")

            self.assertEqual(ledger.refutation_count("creator-rule-x"), 0)
            self.assertEqual(ledger.counters(), {})
            tally = ledger.tally()["creator-rule-x"]
            self.assertEqual(tally["refuted"], 2)
            self.assertEqual(tally["released"], 1)


class CreatorRuleRefutationScopeTests(unittest.TestCase):
    """A rollback is charged only where the reason and the stage both place it on one rule."""

    def test_the_ten_rule_shapes_do_not_all_share_one_rollback(self):
        stages = {
            "curated-planning-direction": "global_planning",
            "curated-segmentation-panels": "segmentation",
            "curated-layout-occupancy": "ppt_compilation",
            "curated-route-shared-channel": "ppt_compilation",
        }

        charged, spared = refuting_rule_ids(rule_stages=stages, reason="pairwise_not_preferred", round_stages={"ppt_compilation"})

        self.assertEqual(sorted(charged), ["curated-layout-occupancy", "curated-route-shared-channel"])
        self.assertEqual({item["rule_id"] for item in spared}, {"curated-planning-direction", "curated-segmentation-panels"})
        self.assertTrue(all(item["reason"] for item in spared))

    def test_a_reason_about_something_else_charges_nobody(self):
        stages = {"curated-layout-occupancy": "ppt_compilation"}
        for reason in ("measurement_flagged_randomness", "new_blocker", "fault_evidence_unreadable", "no_after_measurement"):
            charged, spared = refuting_rule_ids(rule_stages=stages, reason=reason, round_stages={"ppt_compilation"})
            self.assertEqual(charged, [], reason)
            self.assertEqual(len(spared), 1, reason)

    def test_a_round_that_names_no_stage_charges_nobody(self):
        charged, spared = refuting_rule_ids(
            rule_stages={"curated-layout-occupancy": "ppt_compilation"},
            reason="pairwise_not_preferred",
            round_stages=set(),
        )
        self.assertEqual(charged, [])
        self.assertEqual([item["rule_id"] for item in spared], ["curated-layout-occupancy"])

    def test_every_spared_reason_and_the_charged_reason_are_named(self):
        self.assertIn("pairwise_not_preferred", CREATOR_RULE_REFUTING_REASONS)
        self.assertNotIn("new_blocker", CREATOR_RULE_REFUTING_REASONS)
        self.assertNotIn("measurement_flagged_randomness", CREATOR_RULE_REFUTING_REASONS)


class CreatorRuleManifestTests(unittest.TestCase):
    def test_the_rule_list_renders_into_the_manifest_the_planners_read(self):
        manifest = build_creator_rule_manifest(
            [
                _experience("direct-exp-advisory", action={"type": "resegment"}, condition=dict(OCCUPANCY_CONDITION)),
                _experience("direct-exp-prohibition", action={"type": "resegment"}, accepted=False, repair_gain=-0.05),
            ]
        )
        stage4 = manifest["stage4_manifest"]

        self.assertTrue(set(CREATOR_RULE_STAGE4_KEYS) <= set(stage4))
        self.assertEqual(stage4["global_geometry_rules"], {})
        self.assertEqual(stage4["panel_sizing_rules"], {})
        self.assertEqual(len(stage4["composition_patterns"]), 1)
        self.assertEqual(len(stage4["anti_patterns"]), 1)
        self.assertEqual(stage4["evidence_references"], ["direct-exp-advisory", "direct-exp-prohibition"])
        self.assertEqual([rule["rule_id"] for rule in stage4["creator_rules"]], sorted(rule["rule_id"] for rule in manifest["rules"]))
        self.assertTrue(validate_creator_rule_manifest(manifest)["ok"])

    def test_an_empty_library_produces_an_empty_manifest(self):
        manifest = normalize_creator_rule_manifest({})

        self.assertEqual(manifest["rules"], [])
        self.assertEqual(manifest["buckets"], {})
        self.assertFalse(manifest["stage4_execution_policy"]["enabled"])
        self.assertEqual(manifest["stage4_manifest"]["anti_patterns"], [])
        self.assertTrue(validate_creator_rule_manifest(manifest)["ok"])

    def test_the_manifest_reads_both_libraries_and_ignores_other_domains(self):
        patch = build_prompt_patch(
            experience=_experience("direct-exp-theme", apply_at="theme_planning", action={"type": "replan_theme"}),
            status="promoted",
        )
        self.assertTrue(validate_prompt_patch(patch)["ok"])

        manifest = read_creator_rule_manifest(
            experiences=[_experience("direct-exp-seg", apply_at="segmentation", action={"type": "resegment"}), _experience("direct-exp-elsewhere", domain="figure_review")],
            patches=[patch],
        )

        self.assertEqual(sorted(manifest["buckets"]), ["segmentation", "theme_planning"])
        themed = next(rule for rule in manifest["rules"] if rule["apply_at"] == "theme_planning")
        self.assertEqual(themed["source_patch_ids"], [patch["patch_id"]])
        self.assertEqual(themed["source_experience_ids"], ["direct-exp-theme"])
        self.assertEqual(themed["support"], 2)
        self.assertEqual(manifest["provenance"]["source"], "measured_memory_aggregation")

    def test_the_ledger_disables_a_rule_the_manifest_reads_back(self):
        rows = [_experience("direct-exp-a")]
        rule_id = build_creator_rule_manifest(rows)["rules"][0]["rule_id"]
        with tempfile.TemporaryDirectory() as name:
            ledger = CreatorRuleLedger(Path(name) / "library")
            for index in range(CREATOR_RULE_REFUTATION_LIMIT):
                ledger.record_refutation(rule_id, round_index=index, reason="the guided round rolled back")

            manifest = read_creator_rule_manifest(experiences=rows, ledger=ledger)

        self.assertEqual(manifest["rules"], [])
        self.assertEqual(manifest["refuted_rules"][0]["status"], "disabled")
        self.assertTrue(manifest["validation"]["ok"])


if __name__ == "__main__":
    unittest.main()
