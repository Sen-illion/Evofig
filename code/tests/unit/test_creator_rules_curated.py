"""The hand-written standard travels the rule channel without looking measured."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.contracts import (
    CREATOR_RULE_CURATED_SOURCE,
    CURATED_STANDARD_RULES,
    load_curated_standard,
    merge_creator_rule_manifests,
    normalize_creator_rule_manifest,
    validate_curated_standard,
)
from rfs.contracts.evolution import APPLY_AT_STAGES
from rfs.contracts.vlm_context import compact_reference


class CuratedStandardTests(unittest.TestCase):
    def test_the_builtin_standard_validates_and_only_advises(self):
        manifest = load_curated_standard()
        self.assertTrue(manifest["validation"]["ok"], manifest["validation"]["errors"])
        self.assertEqual(manifest["provenance"]["source"], CREATOR_RULE_CURATED_SOURCE)
        self.assertFalse(manifest["provenance"]["measured"])
        self.assertEqual(len(manifest["rules"]), len(CURATED_STANDARD_RULES))
        for rule in manifest["rules"]:
            self.assertEqual(rule["class"], "advisory")
            self.assertIn(rule["apply_at"], APPLY_AT_STAGES)
            self.assertTrue(rule["directive"].strip())

    def test_the_retired_three_colour_advisory_is_gone(self):
        # The hand-written "at most three base colours" row was pulled: the palette belongs to
        # the theme planner, and the advisory kept a model-authored theme from being used.
        manifest = load_curated_standard()
        ids = [rule["rule_id"] for rule in manifest["rules"]]
        self.assertNotIn("curated-theme-accent", ids)
        self.assertFalse(any("three base colours" in rule["directive"] for rule in manifest["rules"]))

    def test_the_validator_refuses_a_non_advisory_row_and_the_loader_forces_advisory(self):
        forged = normalize_creator_rule_manifest(
            {"rules": [{"rule_id": "x", "class": "executable_geometry", "apply_at": "ppt_compilation", "directive": "d"}]}
        )
        self.assertFalse(validate_curated_standard(forged)["ok"])
        # The loader is what forces the class, so read a forged row back through a file.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "standard.json"
            path.write_text(
                json.dumps({"rules": [{"rule_id": "x", "class": "executable_geometry", "apply_at": "ppt_compilation", "directive": "d"}]}),
                encoding="utf-8",
            )
            loaded = load_curated_standard(path)
        self.assertEqual(loaded["rules"][0]["class"], "advisory")
        self.assertTrue(loaded["validation"]["ok"])
        self.assertEqual(loaded["provenance"]["path"], str(path))

    def test_merging_keeps_the_earned_rule_and_the_authored_one_apart(self):
        earned = normalize_creator_rule_manifest(
            {"rules": [{"rule_id": "measured-1", "class": "advisory", "apply_at": "segmentation", "directive": "measured"}]}
        )
        earned["refuted_rules"] = [{"rule_id": "measured-dead"}]
        merged = merge_creator_rule_manifests(earned, load_curated_standard())
        ids = [rule["rule_id"] for rule in merged["rules"]]
        self.assertIn("measured-1", ids)
        self.assertIn("curated-layout-occupancy", ids)
        self.assertEqual(merged["curated_rule_ids"], [rid for rid in ids if rid.startswith("curated-")])
        self.assertEqual(len(merged["rules"]), len(set(ids)))
        self.assertEqual(merged["refuted_rules"], [{"rule_id": "measured-dead"}])
        self.assertIn("curated-layout-occupancy", merged["buckets"]["ppt_compilation"])

    def test_a_written_off_standard_leaves_the_list_every_stage_injects(self):
        merged = merge_creator_rule_manifests(
            normalize_creator_rule_manifest({}),
            load_curated_standard(),
            refutations={"curated-layout-occupancy": 2},
        )

        ids = [rule["rule_id"] for rule in merged["rules"]]
        self.assertNotIn("curated-layout-occupancy", ids)
        self.assertNotIn("curated-layout-occupancy", merged["curated_rule_ids"])
        self.assertNotIn("curated-layout-occupancy", merged["buckets"].get("ppt_compilation", []))
        self.assertNotIn("curated-layout-occupancy", merged["stage4_manifest"]["composition_patterns"])
        written_off = next(rule for rule in merged["refuted_rules"] if rule["rule_id"] == "curated-layout-occupancy")
        self.assertEqual(written_off["status"], "disabled")
        # The rest of the standard is untouched: one rule being written off is not the whole
        # standard being withdrawn.
        self.assertIn("curated-layout-gutter", ids)

    def test_a_duplicate_id_keeps_the_first_row(self):
        first = normalize_creator_rule_manifest({"rules": [{"rule_id": "same", "apply_at": "segmentation", "directive": "first"}]})
        second = normalize_creator_rule_manifest({"rules": [{"rule_id": "same", "apply_at": "segmentation", "directive": "second"}]})
        merged = merge_creator_rule_manifests(first, second)
        self.assertEqual([rule["directive"] for rule in merged["rules"]], ["first"])


class CuratedRulePromptTests(unittest.TestCase):
    def _brief(self):
        merged = merge_creator_rule_manifests(normalize_creator_rule_manifest({}), load_curated_standard())
        return {"creator_rules": {"rules": merged["rules"]}}

    def test_the_directives_reach_the_planner_prompt_under_their_own_source(self):
        reference = compact_reference({}, self._brief())
        rows = reference["creator_rules"]
        self.assertTrue(rows)
        self.assertTrue(all(row["source"] == CREATOR_RULE_CURATED_SOURCE for row in rows))
        self.assertTrue(any("panel_occupancy_min" in row["directive"] for row in rows))

    def test_a_run_without_rules_keeps_the_prompt_it_had(self):
        self.assertNotIn("creator_rules", compact_reference({}, {}))
        self.assertNotIn("creator_rules", compact_reference({"layout": {}}, {"layout_rules": ["a"]}))


if __name__ == "__main__":
    unittest.main()
