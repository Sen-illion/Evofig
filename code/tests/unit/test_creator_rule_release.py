"""Releasing a written-off creator rule: the streak is shown, and cleared on request."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.contracts.creator_rules import (
    CREATOR_RULE_REFUTATION_LIMIT,
    CURATED_STANDARD_RULES,
    load_curated_standard,
    merge_creator_rule_manifests,
    normalize_creator_rule_manifest,
)
from rfs.evolution.direct import CreatorRuleLedger


def write_refutations(library: Path, rule_id: str, times: int) -> None:
    ledger = CreatorRuleLedger(library)
    for index in range(times):
        ledger.record_refutation(rule_id, round_index=index, reason="the guided round lost the comparison")


class TheCreatorRulesCommandTests(unittest.TestCase):
    def test_the_command_is_registered_in_the_product_category(self):
        from rfs.cli import build_parser
        from rfs.cli_commands.categories import COMMAND_CATEGORIES

        registered = set(build_parser()._subparsers._group_actions[0].choices)
        self.assertIn("creator-rules", registered)
        self.assertEqual("product", COMMAND_CATEGORIES["creator-rules"])

    def test_the_report_names_every_rule_at_the_limit(self):
        from rfs.workflows import report_creator_rule_ledger

        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp, "library")
            write_refutations(library, "curated-layout-gutter", CREATOR_RULE_REFUTATION_LIMIT)

            report = report_creator_rule_ledger(library)

        self.assertTrue(report["ok"])
        self.assertEqual(report["disabled_rule_ids"], ["curated-layout-gutter"])
        self.assertEqual(report["released"], [])
        row = next(item for item in report["rules"] if item["rule_id"] == "curated-layout-gutter")
        self.assertTrue(row["disabled"])
        self.assertEqual(row["refuted"], CREATOR_RULE_REFUTATION_LIMIT)

    def test_release_all_clears_every_streak_and_writes_the_report(self):
        from rfs.workflows import report_creator_rule_ledger

        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp, "library")
            write_refutations(library, "curated-layout-gutter", CREATOR_RULE_REFUTATION_LIMIT)
            write_refutations(library, "curated-layout-grid", 1)
            out = Path(tmp, "report.json")

            report = report_creator_rule_ledger(library, release_all=True, out=out)

            self.assertEqual(sorted(report["released"]), ["curated-layout-grid", "curated-layout-gutter"])
            self.assertEqual(report["disabled_rule_ids"], [])
            self.assertEqual([row["refutations"] for row in report["rules"]], [0, 0])
            self.assertTrue(json.loads(out.read_text(encoding="utf-8"))["released"])

    def test_a_released_standard_is_injected_again(self):
        # The whole point of the release: the next run's manifest carries the rule once more.
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp, "library")
            write_refutations(library, "curated-layout-gutter", CREATOR_RULE_REFUTATION_LIMIT)
            ledger = CreatorRuleLedger(library)

            written_off = merge_creator_rule_manifests(
                normalize_creator_rule_manifest({}),
                load_curated_standard(),
                refutations=ledger.counters(),
            )
            ledger.release("curated-layout-gutter", reason="the operator let it back in")
            restored = merge_creator_rule_manifests(
                normalize_creator_rule_manifest({}),
                load_curated_standard(),
                refutations=ledger.counters(),
            )

        self.assertNotIn("curated-layout-gutter", [rule["rule_id"] for rule in written_off["rules"]])
        self.assertIn("curated-layout-gutter", [rule["rule_id"] for rule in restored["rules"]])
        self.assertEqual(restored["curated_rule_ids"].count("curated-layout-gutter"), 1)
        self.assertEqual(len(CURATED_STANDARD_RULES), len(restored["rules"]))


if __name__ == "__main__":
    unittest.main()
