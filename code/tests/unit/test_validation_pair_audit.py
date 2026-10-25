"""P2: a pair whose arms never showed the same start state is unmeasured, not passing.

The harness used to fall back to the configuration hash when a start snapshot was missing, and
two arms sharing a configuration always share that hash -- so a pair nobody could measure passed
the isolation check on a technicality. These tests pin the refusal in place.
"""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import ExperienceStore, ValidationCase, validate_experience_cases
from rfs.workflows.direct_experience_validation import run_direct_experience_validation


MEASURED = {
    "repair_gain": 0.25,
    "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.75},
}

LINEAR = {"linear": 2}
CYCLIC = {"cyclic": 2}


def graph_record(paper_id, domain, motif=LINEAR, **extra):
    return {"paper_id": paper_id, "domain": domain, "features": {"node_count": 4, "edge_count": 3, "motif_histogram": motif}, **extra}


def four_group_suite():
    return {
        "query_graph_record": graph_record("query", "ai_method"),
        "cases": [
            {"case_id": "a", "paper_id": "a", "paper": "a.pdf", "domain": "ai_method", "graph_record": graph_record("a", "ai_method")},
            {"case_id": "b", "paper_id": "b", "paper": "b.pdf", "domain": "ai_method", "graph_record": graph_record("b", "ai_method")},
            {"case_id": "c", "paper_id": "c", "paper": "c.pdf", "domain": "biology", "graph_record": graph_record("c", "biology")},
            {"case_id": "d", "paper_id": "d", "paper": "d.pdf", "domain": "ai_method", "graph_record": graph_record("d", "ai_method", CYCLIC)},
            {"case_id": "e", "paper_id": "e", "paper": "e.pdf", "domain": "ai_method", "graph_record": graph_record("e", "ai_method", topology="reference_guided_multiframe")},
        ],
    }


EXPERIENCE = {
    "experience_id": "e1",
    "status": "current_case_validated",
    "domain": "direct_editable",
    "scope": "slot",
    "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
    "hypothesis": "larger icon improves clarity",
    "action": {"type": "regen_assets", "slot_ids": ["icon_n"]},
    "graph_scope": {"slot_ids": ["icon_n"]},
    "excluded_conditions": [{"dimension": "topology", "value": "reference_guided_multiframe"}],
    "applicability": "common",
    "apply_at": "ppt_compilation",
    **MEASURED,
}


#: The figure both arms of a pair have to have started from. Only the skeleton is compared, so
#: a run is free to plan its own colours and names -- what it may not do is lay out a different
#: figure and still be read as the same experiment.
START_STATE = {
    "layout": {
        "panels": [{"id": "p1", "bbox_percent": {"x": 0.05, "y": 0.1, "w": 0.9, "h": 0.8}}],
        "nodes": [
            {"id": "n1", "panel_id": "p1", "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.2}},
            {"id": "n2", "panel_id": "p1", "bbox_percent": {"x": 0.4, "y": 0.2, "w": 0.2, "h": 0.2}},
        ],
        "connectors": [{"id": "c1", "source": "n1", "target": "n2"}],
    }
}


def leave_start_state(options):
    """Write the snapshot a real generation run leaves behind before its first round."""

    target = Path(options["out"]) / "round_00" / "state_before.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"state_hash": "start", "state": START_STATE}), encoding="utf-8")


def leave_painted_start_state(options, *, palette: str, name: str) -> None:
    """Leave a start state painted the way one arm would paint it: same boxes, other names.

    The raw hash is deliberately different in each arm, the way it is in a real pair: the
    stage that plans colours writes them itself, so only the skeleton can show that both arms
    began from the same figure.
    """

    target = Path(options["out"]) / "round_00" / "state_before.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(START_STATE))
    payload["layout"]["palette"] = {"primary": palette}
    payload["layout"]["nodes"][0]["label"] = name
    target.write_text(json.dumps({"state_hash": f"raw-{name}", "state": payload}), encoding="utf-8")


def leave_restructured_start_state(options) -> None:
    """Leave a start state whose boxes the other arm did not lay out."""

    target = Path(options["out"]) / "round_00" / "state_before.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(START_STATE))
    payload["layout"]["panels"].append({"id": "p2", "bbox_percent": {"x": 0.05, "y": 0.55, "w": 0.9, "h": 0.4}})
    target.write_text(json.dumps({"state_hash": "restructured", "state": payload}), encoding="utf-8")


class PairAuditTests(unittest.TestCase):
    """The audit, at the level the runner answers it."""

    def test_a_missing_start_hash_fails_the_audit_even_with_a_matching_configuration(self):
        cases = [ValidationCase("a", "ai_method"), ValidationCase("b", "ai_method"), ValidationCase("c", "system", "holdout")]

        def runner(case, enabled, experience):
            return {"overall_score": 0.83 if enabled else 0.8, "blocking_issue_count": 0, "paired_config_hash": "cfg", "initial_state_hash": "", "candidate_policy_hash": "candidates", "evaluator_hash": "judge"}

        result = validate_experience_cases({"experience_id": "e1", "applicability": "common"}, cases, runner)

        self.assertFalse(result["all_pairs_isolated"])
        self.assertFalse(result["all_cases_measured"])
        self.assertFalse(any(item["measured"] for item in result["results"]))
        self.assertFalse(any(item["passed"] for item in result["results"]))
        self.assertEqual(result["ladder"]["stopped_because"], "pair_not_measured")
        self.assertEqual(result["summary"]["measured_cases"], 0)
        self.assertIsNone(result["summary"]["mean_target_delta"])


class UnmeasuredPairWorkflowTests(unittest.TestCase):
    """The same refusal through the cross-paper harness, where the snapshot comes off disk."""

    def _run(self, root, runner):
        library = root / "library"
        ExperienceStore(library).append({**EXPERIENCE})
        suite = root / "suite.json"
        suite.write_text(json.dumps(four_group_suite()), encoding="utf-8")
        return run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner, reselect_on_failure=False)

    def test_a_pair_the_memory_arm_never_showed_a_start_for_is_unmeasured(self):
        def runner(**kwargs):
            enabled = kwargs.get("seed_experience") is not None
            if not enabled:
                leave_start_state(kwargs)
            return {"ok": True, "judgement": {"overall_score": 0.9 if enabled else 0.8, "blocking_issue_count": 0}}

        with tempfile.TemporaryDirectory() as temp:
            report = self._run(Path(temp), runner)

        self.assertNotEqual(report["status"], "promoted")
        self.assertEqual(report["ladder"]["stopped_because"], "pair_not_measured")
        self.assertFalse(report["all_cases_measured"])
        self.assertFalse(any(item["passed"] for item in report["results"]))
        for item in report["results"]:
            self.assertFalse(item["measured"])
            self.assertFalse(item["pair_audit"]["ok"])
            self.assertFalse(item["pair_audit"]["checks"]["initial_state_hash"])
            self.assertEqual(item["pair_stop_reason"], "initial_state_not_measured")

    def test_the_same_suite_is_promoted_once_both_arms_show_the_same_start(self):
        def runner(**kwargs):
            leave_start_state(kwargs)
            enabled = kwargs.get("seed_experience") is not None
            return {"ok": True, "judgement": {"overall_score": 0.9 if enabled else 0.8, "blocking_issue_count": 0}}

        with tempfile.TemporaryDirectory() as temp:
            report = self._run(Path(temp), runner)

        self.assertEqual(report["status"], "promoted")
        self.assertTrue(report["all_cases_measured"])
        self.assertTrue(report["all_pairs_isolated"])
        self.assertTrue(all(item["pair_audit"]["checks"]["initial_state_hash"] for item in report["results"]))


class ForkedPairTests(unittest.TestCase):
    """P1: the pair is compared on the skeleton both arms forked from, not on their bytes."""

    def _run(self, root, runner):
        library = root / "library"
        ExperienceStore(library).append({**EXPERIENCE})
        suite = root / "suite.json"
        suite.write_text(json.dumps(four_group_suite()), encoding="utf-8")
        return run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner, reselect_on_failure=False)

    def test_two_arms_that_only_painted_differently_are_still_one_pair(self):
        def runner(**kwargs):
            enabled = kwargs.get("seed_experience") is not None
            leave_painted_start_state(kwargs, palette="#48BB78" if enabled else "#A9C7C7", name="Card B" if enabled else "Card A")
            return {"ok": True, "judgement": {"overall_score": 0.9 if enabled else 0.8, "blocking_issue_count": 0}}

        with tempfile.TemporaryDirectory() as temp:
            report = self._run(Path(temp), runner)

        self.assertEqual(report["status"], "promoted")
        self.assertTrue(report["all_pairs_isolated"])
        for item in report["results"]:
            self.assertNotEqual(item["off"]["raw_state_hash"], item["on"]["raw_state_hash"])
            self.assertEqual(item["off"]["initial_state_hash"], item["on"]["initial_state_hash"])
            self.assertTrue(item["pair_audit"]["checks"]["initial_state_hash"])
            # The memory arm replays the control arm and forks where the memory acts; the
            # control arm has no replay source by construction.
            self.assertEqual(item["on"]["fork_stage"], "ppt_compilation")
            self.assertTrue(str(item["on"]["fork_source"]).endswith(("off", "off\\")))
            self.assertEqual(item["off"]["fork_source"], "")
            self.assertEqual(item["off"]["paired_config_hash"], item["on"]["paired_config_hash"])

    def test_two_arms_that_laid_out_different_boxes_are_not_one_pair(self):
        def runner(**kwargs):
            enabled = kwargs.get("seed_experience") is not None
            if enabled:
                leave_restructured_start_state(kwargs)
            else:
                leave_start_state(kwargs)
            return {"ok": True, "judgement": {"overall_score": 0.9 if enabled else 0.8, "blocking_issue_count": 0}}

        with tempfile.TemporaryDirectory() as temp:
            report = self._run(Path(temp), runner)

        self.assertNotEqual(report["status"], "promoted")
        self.assertFalse(report["all_pairs_isolated"])
        self.assertEqual(report["ladder"]["stopped_because"], "pair_isolation_or_safety_failed")
        self.assertFalse(any(item["pair_audit"]["checks"]["initial_state_hash"] for item in report["results"]))


if __name__ == "__main__":
    unittest.main()
