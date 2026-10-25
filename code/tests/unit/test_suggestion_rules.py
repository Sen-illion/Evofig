"""Advice a reflection stated is kept, counted, and offered to the advisory role alone.

P9 let a reflection state a mechanism-level rule. This suite covers where that sentence goes
afterwards: how many witnesses it needs, which move vocabulary it may use, which figure it is
offered for, and which role is allowed to read it. The last point is the one that keeps a
measurement a measurement.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.analysis.graph.defect_patterns import DEFECT_PATTERNS
from rfs.contracts.evolution import normalize_reflection
from rfs.contracts.suggestion_rule import MIN_SUPPORT, USABLE_STATUS, suggestion_rule_usable
from rfs.evaluation.direct_judges import DirectJudgeInput
from rfs.evaluation.judge_prompt import canonical_prompt, injectable_suggestion_rules, online_judge_row
from rfs.evolution.direct import SuggestionRuleStore, apply_reflection_policy
from rfs.evolution.direct.suggestion_rules import RULE_LIBRARY_FILE
from tests.unit.test_prompt_patch_wiring import run_workflow, write_paper


ROUTER_RULE = {
    "type": "router_constraint",
    "apply_at": "ppt_compilation",
    "constraints": {"shared_vertical_channel": True, "forbid_crossing": True},
}
LAYOUT_RULE = {"type": "layout_constraint", "apply_at": "ppt_compilation", "params": {"align_grid": 4}}

FANOUT = ["fanout_1_to_n"]
LONG_EDGE = ["long_edge"]


def _store(root: Path) -> SuggestionRuleStore:
    return SuggestionRuleStore(root)


def _judge_input(**overrides) -> DirectJudgeInput:
    values = dict(method={"modules": [{"id": "n", "label": "N", "must_show": True}], "relations": []}, layout={}, icons={}, program={})
    values.update(overrides)
    return DirectJudgeInput(**values)


def _offered_rule() -> dict:
    """One rule past the support bar, as the store hands it to a prompt."""

    return {
        "rule_id": "suggestion-rule-test",
        "apply_at": "ppt_compilation",
        "defect_pattern": FANOUT,
        "rule_text": "router_constraint(forbid_crossing=true, shared_vertical_channel=true)",
        "rule": ROUTER_RULE,
        "support": MIN_SUPPORT,
        "status": "candidate",
    }


class SuggestionRuleStoreTests(unittest.TestCase):
    def test_a_rule_stated_once_is_one_witness_not_advice(self):
        with tempfile.TemporaryDirectory() as name:
            store = _store(Path(name))
            recorded = store.record_statement(rule=ROUTER_RULE, defect_pattern=FANOUT)
            selection = store.select(apply_at="ppt_compilation", defect_pattern=FANOUT)

        self.assertTrue(recorded["ok"])
        self.assertEqual(1, recorded["support"])
        self.assertFalse(recorded["usable"])
        self.assertFalse(suggestion_rule_usable(recorded["rule"]))
        self.assertEqual([], selection["rules"])
        self.assertEqual(["below_support"], [item["reason"] for item in selection["rejected"]])

    def test_two_rounds_reaching_the_same_rule_is_what_earns_it(self):
        with tempfile.TemporaryDirectory() as name:
            store = _store(Path(name))
            store.record_statement(rule=ROUTER_RULE, defect_pattern=FANOUT, source_run="a.pdf", round_index=1)
            second = store.record_statement(rule=ROUTER_RULE, defect_pattern=FANOUT, source_run="b.pdf", round_index=1)
            selection = store.select(apply_at="ppt_compilation", defect_pattern=FANOUT)

        self.assertEqual(MIN_SUPPORT, second["support"])
        self.assertTrue(second["usable"])
        self.assertEqual(1, len(selection["rules"]))
        self.assertEqual([], selection["rejected"])
        self.assertIn("shared_vertical_channel", selection["rules"][0]["rule_text"])

    def test_the_status_says_the_same_thing_the_support_count_says(self):
        """The two fields a reader can ask "is this rule in use" of agree, always.

        The stored status used to say ``candidate`` for ever while the count decided selection, so
        a reader who asked the status was told a rule was unused while its count had already put it
        in prompts. Reaching the bar now writes the rung, and a row written before that is read on
        the rung its own count earns.
        """

        with tempfile.TemporaryDirectory() as name:
            store = _store(Path(name))
            first = store.record_statement(rule=ROUTER_RULE, defect_pattern=FANOUT)
            second = store.record_statement(rule=ROUTER_RULE, defect_pattern=FANOUT)
            listed = next(row for row in store.latest() if row["rule_id"] == first["rule"]["rule_id"])
            streamed = store.status_of(first["rule"]["rule_id"])

        self.assertEqual("candidate", first["rule"]["status"])
        self.assertEqual("usable", second["rule"]["status"])
        self.assertEqual(USABLE_STATUS, listed["status"])
        self.assertEqual(USABLE_STATUS, streamed)
        self.assertTrue(suggestion_rule_usable(listed))

    def test_a_row_written_before_the_status_followed_the_count_is_read_on_its_own_rung(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            store = _store(root)
            store.append_result(
                {
                    "rule_id": "suggestion-rule-legacy",
                    "apply_at": "ppt_compilation",
                    "rule": ROUTER_RULE,
                    "support": MIN_SUPPORT,
                    "status": "candidate",
                }
            )
            listed = store.latest()[0]
            streamed = store.status_of("suggestion-rule-legacy")

        self.assertEqual(USABLE_STATUS, listed["status"])
        self.assertEqual(USABLE_STATUS, streamed)

    def test_two_different_rules_do_not_count_as_two_witnesses_of_one(self):
        with tempfile.TemporaryDirectory() as name:
            store = _store(Path(name))
            store.record_statement(rule=ROUTER_RULE, defect_pattern=FANOUT)
            store.record_statement(rule=LAYOUT_RULE, defect_pattern=LONG_EDGE)
            selection = store.select(apply_at="ppt_compilation")

        self.assertEqual([], selection["rules"])
        self.assertEqual(["below_support", "below_support"], sorted(item["reason"] for item in selection["rejected"]))

    def test_a_move_outside_the_mechanism_vocabulary_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            store = _store(Path(name))
            refused = store.record_statement(rule={"type": "resize_nodes", "node_ids": ["n1"], "factor": 1.1})
            empty = store.record_statement(rule={"type": "router_constraint", "constraints": {}})
            rows = store.records()

        for result in (refused, empty):
            self.assertFalse(result["ok"])
            self.assertIn("no mechanism-level rule", result["reason"])
        self.assertEqual([], rows)

    def test_a_rule_the_figure_does_not_have_is_not_offered(self):
        with tempfile.TemporaryDirectory() as name:
            store = _store(Path(name))
            store.record_statement(rule=ROUTER_RULE, defect_pattern=FANOUT)
            store.record_statement(rule=ROUTER_RULE, defect_pattern=FANOUT)
            selection = store.select(apply_at="ppt_compilation", defect_pattern=LONG_EDGE)

        self.assertEqual([], selection["rules"])
        self.assertEqual(["defect_pattern_mismatch"], [item["reason"] for item in selection["rejected"]])

    def test_a_restatement_keeps_every_pattern_it_was_heard_with(self):
        with tempfile.TemporaryDirectory() as name:
            store = _store(Path(name))
            store.record_statement(rule=ROUTER_RULE, defect_pattern=FANOUT)
            store.record_statement(rule=ROUTER_RULE, defect_pattern=LONG_EDGE)
            stored = store.latest()[0]

        self.assertEqual(sorted(FANOUT + LONG_EDGE), stored["defect_pattern"])

    def test_a_disabled_rule_is_never_offered_again(self):
        with tempfile.TemporaryDirectory() as name:
            store = _store(Path(name))
            store.record_statement(rule=ROUTER_RULE, defect_pattern=FANOUT)
            recorded = store.record_statement(rule=ROUTER_RULE, defect_pattern=FANOUT)
            store.disable(recorded["rule"]["rule_id"], reason="measured worse on a second pair")
            selection = store.select(apply_at="ppt_compilation")
            status = store.status_of(recorded["rule"]["rule_id"])

        self.assertEqual([], selection["rules"])
        self.assertEqual(["status_refused"], [item["reason"] for item in selection["rejected"]])
        self.assertEqual("disabled", status)

    def test_the_library_keeps_every_statement_as_the_support_history(self):
        with tempfile.TemporaryDirectory() as name:
            store = _store(Path(name))
            store.record_statement(rule=ROUTER_RULE, defect_pattern=FANOUT)
            store.record_statement(rule=ROUTER_RULE, defect_pattern=FANOUT)
            rows = (Path(name) / RULE_LIBRARY_FILE).read_text(encoding="utf-8").splitlines()

        self.assertEqual([1, 2], [json.loads(line)["support"] for line in rows if line.strip()])


class SuggestionRulePromptTests(unittest.TestCase):
    def test_only_the_advisory_row_carries_advice(self):
        row = online_judge_row(_judge_input(suggestion_rules=[_offered_rule()]), role="online_bootstrap")
        acceptance = online_judge_row(_judge_input(suggestion_rules=[_offered_rule()]), role="frozen_bootstrap")

        self.assertEqual(["suggestion-rule-test"], [item["rule_id"] for item in row["process_context"]["suggestion_rules"]])
        self.assertNotIn("suggestion_rules", acceptance["process_context"])
        self.assertNotIn("judge_metadata", acceptance)

    def test_a_pairwise_row_never_carries_advice(self):
        row = online_judge_row(_judge_input(suggestion_rules=[_offered_rule()]), role="online_bootstrap", pairwise=True)

        self.assertNotIn("suggestion_rules", row["process_context"])
        self.assertNotIn("judge_metadata", row)

    def test_a_rule_below_the_support_bar_is_not_injected(self):
        below = {**_offered_rule(), "support": MIN_SUPPORT - 1}
        row = online_judge_row(_judge_input(suggestion_rules=[below]), role="online_bootstrap")

        self.assertNotIn("suggestion_rules", row["process_context"])
        self.assertEqual([], injectable_suggestion_rules([below], role="online_bootstrap"))

    def test_the_injected_ids_travel_in_the_row_metadata_and_not_in_the_text(self):
        row = online_judge_row(_judge_input(suggestion_rules=[_offered_rule()]), role="online_bootstrap")

        self.assertEqual({"suggestion_rule_ids": ["suggestion-rule-test"]}, row["judge_metadata"])
        self.assertNotIn("judge_metadata", canonical_prompt(row))

    def test_the_advisory_prompt_quotes_advice_and_the_acceptance_one_does_not(self):
        inputs = _judge_input(suggestion_rules=[_offered_rule()])
        advised = canonical_prompt(online_judge_row(inputs, role="online_bootstrap"))
        measured = canonical_prompt(online_judge_row(inputs, role="frozen_bootstrap"))

        self.assertIn("forbid_crossing", advised)
        self.assertNotIn("forbid_crossing", measured)

    def test_an_empty_library_renders_the_row_as_before(self):
        row = online_judge_row(_judge_input(), role="online_bootstrap")

        self.assertNotIn("suggestion_rules", row["process_context"])
        self.assertNotIn("judge_metadata", row)


class ReflectionToAdviceTests(unittest.TestCase):
    def test_a_reflection_that_states_a_rule_is_kept_after_two_rounds(self):
        reflection = normalize_reflection({"outcome": "regressed", "next_policy": "revise_rule", "rule": ROUTER_RULE})
        policy = apply_reflection_policy({"accepted": False}, {"type": "reroute"}, reflection)
        with tempfile.TemporaryDirectory() as name:
            store = _store(Path(name))
            first = store.record_statement(rule=policy["revised_rule"], defect_pattern=FANOUT, source_run="a.pdf", round_index=2)
            second = store.record_statement(rule=policy["revised_rule"], defect_pattern=FANOUT, source_run="b.pdf", round_index=2)

        self.assertEqual("revise_rule", policy["policy"]["next_policy"])
        self.assertTrue(first["ok"])
        self.assertEqual(1, first["support"])
        self.assertFalse(first["usable"])
        self.assertEqual(2, second["support"])
        self.assertTrue(second["usable"])


class SuggestionRuleWiringTests(unittest.TestCase):
    def test_the_loop_offers_earned_advice_to_the_advisory_step(self):
        """The run opens the library itself: a rule on disk reaches the prompt step of a real run."""

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            library = root / "library"
            store = _store(library)
            rule = {"type": "router_constraint", "constraints": {"forbid_crossing": True}}
            store.record_statement(rule=rule, defect_pattern=list(DEFECT_PATTERNS), source_run="a.pdf", round_index=1)
            store.record_statement(rule=rule, defect_pattern=list(DEFECT_PATTERNS), source_run="b.pdf", round_index=1)
            rule_id = store.latest()[0]["rule_id"]

            out = run_workflow(root, library=library)
            events = [json.loads(line) for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            report = json.loads((out / "paper_evolve_editable_report.json").read_text(encoding="utf-8"))

        offered = [event for event in events if event.get("event") == "suggestion_rules_offered"]
        self.assertTrue(offered)
        self.assertEqual([rule_id], offered[0]["rule_ids"])
        self.assertEqual(str(library.resolve()), report["suggestion_rule_library"])


if __name__ == "__main__":
    unittest.main()
