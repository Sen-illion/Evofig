"""Sweeping Memory an older Judge measured, and earning it back under the new one."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rfs.contracts.evolution import judge_fingerprint_identity, same_judge
from rfs.contracts.judge_sweep import (
    MEMORY_SWEEP_CONTRACT,
    validate_memory_sweep_report,
)
from rfs.evolution.direct import ExperienceStore
from rfs.evolution.direct.creator_rules import read_creator_rule_manifest
from rfs.judge_training.memory_sweep import (
    SweepCandidate,
    apply_memory_sweep,
    collect_sweep_candidates,
    plan_memory_sweep,
)
from rfs.utils import write_json
from rfs.workflows.memory_revalidation import sweep_memory_with_judge


TRAINED_JUDGE = {"role": "frozen", "mode": "trained", "trained": True, "model": "qwen3-35b-a3b", "checkpoint": ""}
BOOTSTRAP_JUDGE = {"role": "frozen", "mode": "heuristic", "trained": False, "model": "", "checkpoint": ""}
CONDITION = {"metric": "cross_panel_connectors", "op": "<=", "value": 2}


def _row(identifier: str, **overrides) -> dict:
    """Build a measured Memory row in the shape the store writes."""

    row = {
        "experience_id": identifier,
        "domain": "direct_editable",
        "scope": "global",
        # A stage a pair can actually fork at: a row recorded at a stage with no shareable prefix
        # is retired as unverifiable instead, which is the subject of its own test rather than
        # something every revalidation test should trip over.
        "apply_at": "repair",
        "memory_type": "layout",
        "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
        "hypothesis": "fewer cross-panel connectors survive the gate",
        "action": {"type": "resize_nodes"},
        "condition": dict(CONDITION),
        "graph_scope": {"slot_ids": ["slot-1"]},
        "accepted": True,
        "repair_gain": 0.12,
        "validated_cases": ["a.pdf", "b.pdf"],
        "status": "cross_domain_validated",
        "applicability": "common",
        "judge_fingerprint": dict(BOOTSTRAP_JUDGE),
        "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.62},
    }
    row.update(overrides)
    return row


def _candidate(identifier: str, **overrides) -> SweepCandidate:
    values = {"identifier": identifier, "kind": "experience", "apply_at": "global_planning", "status": "cross_domain_validated", "fingerprint": dict(BOOTSTRAP_JUDGE)}
    values.update(overrides)
    return SweepCandidate(**values)


def _graph_record(paper_id: str, domain: str, **extra) -> dict:
    return {"paper_id": paper_id, "domain": domain, "features": {"node_count": 4, "edge_count": 3, "motif_histogram": {"linear": 2}}, **extra}


def _suite(path: Path) -> Path:
    write_json(
        path,
        {
            "query_graph_record": _graph_record("query", "ai_method"),
            "selector": {"quota": {"same_domain_same_structure": 2, "cross_domain_same_structure": 1, "boundary": 1}},
            "cases": [
                {"case_id": "c1", "paper_id": "c1", "paper": "a.pdf", "domain": "ai_method", "seed": 1, "graph_record": _graph_record("c1", "ai_method")},
                {"case_id": "c2", "paper_id": "c2", "paper": "b.pdf", "domain": "ai_method", "seed": 2, "graph_record": _graph_record("c2", "ai_method")},
                {"case_id": "c3", "paper_id": "c3", "paper": "c.pdf", "domain": "biology", "seed": 3, "graph_record": _graph_record("c3", "biology")},
                # A paper holding none of the objects the row is scoped to: the boundary the exam
                # needs before it may run at all.
                {"case_id": "c4", "paper_id": "c4", "paper": "d.pdf", "domain": "ai_method", "seed": 4, "graph_record": _graph_record("c4", "ai_method", slot_ids=["slot-other"])},
            ],
            "target_metric": "overall_score",
            "tolerance": 0.01,
        },
    )
    return path


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


def _start_state(options: dict) -> None:
    """A real generation run leaves the state it started from; a stand-in has to leave one too.

    Without the snapshot the pair is correctly refused as unmeasured, which would make every
    sweep here test the refusal rather than the revalidation it means to test.
    """

    write_json(Path(options["out"]) / "round_00" / "state_before.json", {"state_hash": "start", "state": START_STATE})


def _improving_runner(**options) -> dict:
    """A stand-in for the generation run: the seeded Memory is what raises the score."""

    _start_state(options)
    return {"ok": True, "judgement": {"overall_score": 0.8 if options.get("seed_experience") else 0.5, "blocking_issue_count": 0}}


def _flat_runner(**options) -> dict:
    """A stand-in that never separates ON from OFF, so nothing can be promoted."""

    _start_state(options)
    return {"ok": True, "judgement": {"overall_score": 0.6, "blocking_issue_count": 0}}


def _same_domain_only_runner(**options) -> dict:
    """A stand-in whose Memory helps the same-domain papers and not the holdout."""

    _start_state(options)
    helped = options.get("seed_experience") and not str(options.get("paper") or "").endswith("c.pdf")
    return {"ok": True, "judgement": {"overall_score": 0.8 if helped else 0.5, "blocking_issue_count": 0}}


class JudgeFingerprintTests(unittest.TestCase):
    def test_the_same_model_and_training_state_identify_one_judge(self):
        self.assertTrue(same_judge(TRAINED_JUDGE, dict(TRAINED_JUDGE, role="online")))
        self.assertFalse(same_judge(TRAINED_JUDGE, BOOTSTRAP_JUDGE))
        self.assertFalse(same_judge(TRAINED_JUDGE, dict(TRAINED_JUDGE, model="other-vlm")))

    def test_the_checkpoint_answers_before_the_model_name_does(self):
        """Two trainings published at one endpoint name are two Judges, not one."""

        served = dict(TRAINED_JUDGE, checkpoint="judge-v1@d0-pilot-v3+abc123")
        retrained = dict(TRAINED_JUDGE, checkpoint="judge-v1@d0-pilot-v4+def456")
        self.assertEqual(judge_fingerprint_identity(served), "trained:judge-v1@d0-pilot-v3+abc123")
        self.assertFalse(same_judge(served, retrained))
        self.assertTrue(same_judge(served, dict(served, model="served-under-another-name")))
        self.assertFalse(same_judge(served, TRAINED_JUDGE))

    def test_a_row_that_named_no_model_keeps_its_own_identity(self):
        self.assertEqual(judge_fingerprint_identity({}), "bootstrap:unknown")
        self.assertEqual(judge_fingerprint_identity({"mode": "vlm", "trained": False}), "bootstrap:vlm")
        self.assertFalse(same_judge({}, TRAINED_JUDGE))


class SweepPlanningTests(unittest.TestCase):
    def test_rows_another_judge_measured_are_quarantined_and_the_rest_are_kept(self):
        plan = plan_memory_sweep(
            [_candidate("stale"), _candidate("fresh", fingerprint=dict(TRAINED_JUDGE)), _candidate("draft", status="candidate")],
            incoming=TRAINED_JUDGE,
        )

        self.assertEqual([item["id"] for item in plan["quarantine"]], ["stale"])
        self.assertEqual(plan["quarantine"][0]["reason"], "judge_fingerprint_mismatch")
        self.assertEqual([item["id"] for item in plan["kept"]], ["fresh"])
        self.assertEqual(plan["skipped"], [{"id": "draft", "kind": "experience", "apply_at": "global_planning", "status": "candidate", "reason": "status_not_retrievable", "judge_fingerprint": dict(BOOTSTRAP_JUDGE), "fingerprint_identity": "bootstrap:heuristic", "source_ids": []}])
        self.assertEqual({key: plan["counts"][key] for key in ("considered", "quarantined", "kept", "skipped", "pending", "restored")}, {"considered": 3, "quarantined": 1, "kept": 1, "skipped": 1, "pending": 0, "restored": 0})

    def test_the_bootstrap_selection_spares_what_a_trained_judge_measured(self):
        plan = plan_memory_sweep([_candidate("stale"), _candidate("trained-elsewhere", fingerprint={"trained": True, "model": "another-vlm"}, status="promoted")], incoming=TRAINED_JUDGE, selection="bootstrap")

        self.assertEqual([item["id"] for item in plan["quarantine"]], ["stale"])
        self.assertEqual(plan["quarantine"][0]["reason"], "bootstrap_judge_memory")
        self.assertEqual([item["id"] for item in plan["kept"]], ["trained-elsewhere"])

    def test_the_all_selection_quarantines_every_injectable_row(self):
        plan = plan_memory_sweep([_candidate("stale"), _candidate("fresh", fingerprint=dict(TRAINED_JUDGE), status="promoted")], incoming=TRAINED_JUDGE, selection="all")

        self.assertEqual([item["id"] for item in plan["quarantine"]], ["fresh", "stale"])
        self.assertEqual({item["reason"] for item in plan["quarantine"]}, {"selection_requested"})
        self.assertEqual(plan["kept"], [])

    def test_a_patch_is_quarantined_with_the_evidence_it_came_from(self):
        plan = plan_memory_sweep(
            [
                _candidate("stale"),
                _candidate("fresh", fingerprint=dict(TRAINED_JUDGE)),
                _candidate("patch-from-stale", kind="patch", fingerprint={}, source_ids=("stale",)),
                _candidate("patch-from-fresh", kind="patch", fingerprint={}, source_ids=("fresh",)),
            ],
            incoming=TRAINED_JUDGE,
        )

        self.assertEqual([item["id"] for item in plan["quarantine"]], ["stale", "patch-from-stale"])
        self.assertEqual(plan["quarantine"][1]["reason"], "source_memory_quarantined")
        self.assertEqual([item["id"] for item in plan["kept"]], ["fresh", "patch-from-fresh"])

    def test_a_patch_whose_evidence_is_gone_is_skipped_rather_than_condemned(self):
        plan = plan_memory_sweep([_candidate("patch-orphan", kind="patch", fingerprint={}, source_ids=("deleted",)), _candidate("patch-bare", kind="patch", fingerprint={})], incoming=TRAINED_JUDGE)

        self.assertEqual(plan["quarantine"], [])
        self.assertEqual([(item["id"], item["reason"]) for item in plan["skipped"]], [("patch-bare", "no_source_memory"), ("patch-orphan", "source_memory_unavailable")])

    def test_the_same_library_and_judge_plan_the_same_way_every_time(self):
        rows = [_candidate("b"), _candidate("a", status="promoted"), _candidate("p", kind="patch", fingerprint={}, source_ids=("a",))]

        self.assertEqual(plan_memory_sweep(rows, incoming=TRAINED_JUDGE), plan_memory_sweep(list(reversed(rows)), incoming=TRAINED_JUDGE))


class SweepReportTests(unittest.TestCase):
    def test_a_report_carries_its_own_counts_and_contract(self):
        report = apply_memory_sweep(plan_memory_sweep([_candidate("stale")], incoming=TRAINED_JUDGE), dry_run=True)

        self.assertEqual(report["contract"], MEMORY_SWEEP_CONTRACT)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["counts"]["quarantined"], 1)
        self.assertTrue(validate_memory_sweep_report(report)["ok"])

    def test_a_dry_run_may_not_claim_it_revalidated_anything(self):
        report = {"dry_run": True, "selection": "mismatched", "revalidated": [{"id": "stale", "outcome": "restored"}]}

        checked = validate_memory_sweep_report(report)

        self.assertFalse(checked["ok"])
        self.assertIn("a dry run cannot have revalidated anything", checked["errors"])

    def test_an_unknown_reason_is_rejected(self):
        report = {"selection": "mismatched", "quarantine": [{"id": "stale", "kind": "experience", "reason": "vibes"}], "skipped": [{"id": "draft", "reason": "whatever"}]}

        checked = validate_memory_sweep_report(report)

        self.assertFalse(checked["ok"])
        self.assertIn("unsupported quarantine reason: vibes", checked["errors"])
        self.assertIn("unsupported skip reason: whatever", checked["errors"])


class MemorySweepStoreTests(unittest.TestCase):
    def test_a_quarantined_row_leaves_retrieval_and_keeps_its_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore(tmp)
            self.assertTrue(store.append_result(_row("direct-exp-stale"))["ok"])
            self.assertEqual(len(store.retrieve(limit=5)), 1)

            report = sweep_memory_with_judge(experience_library=tmp, incoming=TRAINED_JUDGE, out=Path(tmp) / "sweep", dry_run=False)

            self.assertEqual(report["counts"]["quarantined"], 1)
            self.assertEqual(store.lifecycle_status("direct-exp-stale"), "retrying")
            self.assertEqual(store.retrieve(limit=5), [])
            self.assertEqual(len(store.records()), 1)
            self.assertEqual([item["event"] for item in store.events("direct-exp-stale")], ["status_changed"])

    def test_a_dry_run_reports_without_touching_the_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore(tmp)
            store.append_result(_row("direct-exp-stale"))

            report = sweep_memory_with_judge(experience_library=tmp, incoming=TRAINED_JUDGE, out=Path(tmp) / "sweep", dry_run=True)

            self.assertEqual(report["counts"]["quarantined"], 1)
            self.assertEqual(store.records_with_status()[0]["status"], "cross_domain_validated")
            self.assertEqual(len(store.retrieve(limit=5)), 1)

    def test_a_matching_judge_leaves_the_library_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore(tmp)
            store.append_result(_row("direct-exp-fresh", judge_fingerprint=dict(TRAINED_JUDGE)))

            report = sweep_memory_with_judge(experience_library=tmp, incoming=TRAINED_JUDGE, out=Path(tmp) / "sweep", dry_run=False)

            self.assertEqual(report["counts"]["quarantined"], 0)
            self.assertEqual(store.lifecycle_status("direct-exp-fresh"), "cross_domain_validated")
            self.assertEqual(len(store.retrieve(limit=5)), 1)

    def test_quarantined_rows_stop_forming_creator_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore(tmp)
            store.append_result(_row("direct-exp-a"))
            store.append_result(_row("direct-exp-b"))
            self.assertEqual(len(read_creator_rule_manifest(experiences=store.records_with_status(), patches=())["rules"]), 1)

            sweep_memory_with_judge(experience_library=tmp, incoming=TRAINED_JUDGE, out=Path(tmp) / "sweep", dry_run=False)

            self.assertEqual(read_creator_rule_manifest(experiences=store.records_with_status(), patches=())["rules"], [])
            self.assertEqual(len(read_creator_rule_manifest(experiences=store.records(), patches=())["rules"]), 1)

    def test_the_candidates_the_sweep_reads_name_the_judge_they_were_measured_under(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore(tmp)
            store.append_result(_row("direct-exp-stale"))

            candidates = collect_sweep_candidates(store=store)

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].identifier, "direct-exp-stale")
            self.assertEqual(candidates[0].status, "cross_domain_validated")
            self.assertEqual(candidates[0].fingerprint, dict(BOOTSTRAP_JUDGE))


class MemoryRevalidationTests(unittest.TestCase):
    def _seeded_library(self, tmp: str, *, status: str = "cross_domain_validated") -> ExperienceStore:
        store = ExperienceStore(tmp)
        store.append_result(_row("direct-exp-stale", status=status))
        suite = _suite(Path(tmp) / "suite.json")
        report_path = Path(tmp) / "cross_paper_validation.json"
        write_json(report_path, {"status": status, "suite": str(suite)})
        store.transition("direct-exp-stale", status, validation_report=str(report_path))
        return store

    def test_a_row_the_new_judge_still_agrees_with_comes_back_stamped(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._seeded_library(tmp)

            report = sweep_memory_with_judge(experience_library=tmp, incoming=TRAINED_JUDGE, out=Path(tmp) / "sweep", dry_run=False, revalidate=True, runner=_improving_runner)

            self.assertEqual([item["outcome"] for item in report["revalidated"]], ["restored"])
            self.assertEqual(report["revalidated"][0]["status"], "promoted")
            self.assertEqual(report["counts"]["restored"], 1)
            self.assertTrue(report["validation"]["ok"])
            self.assertEqual(store.lifecycle_status("direct-exp-stale"), "promoted")
            self.assertEqual(store.revalidation_fingerprint("direct-exp-stale"), TRAINED_JUDGE)
            self.assertEqual(len(store.retrieve(limit=5)), 1)

    def test_the_restored_fingerprint_is_what_a_later_sweep_compares_against(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._seeded_library(tmp)
            sweep_memory_with_judge(experience_library=tmp, incoming=TRAINED_JUDGE, out=Path(tmp) / "sweep", dry_run=False, revalidate=True, runner=_improving_runner)

            same = sweep_memory_with_judge(experience_library=tmp, incoming=TRAINED_JUDGE, out=Path(tmp) / "again", dry_run=True)
            newer = sweep_memory_with_judge(experience_library=tmp, incoming=dict(TRAINED_JUDGE, model="next-vlm"), out=Path(tmp) / "newer", dry_run=True)

            self.assertEqual(same["counts"]["quarantined"], 0)
            self.assertEqual(newer["counts"]["quarantined"], 1)

    def test_a_row_the_new_judge_still_injects_at_a_lower_rung_is_kept(self):
        # Requiring `promoted` retired every row the incoming Judge agreed with at a middle
        # rung, so each handover destroyed Memory that had not regressed out of the injectable
        # set at all. The rung it came back at is recorded instead of being fatal.
        with tempfile.TemporaryDirectory() as tmp:
            store = self._seeded_library(tmp)

            report = sweep_memory_with_judge(experience_library=tmp, incoming=TRAINED_JUDGE, out=Path(tmp) / "sweep", dry_run=False, revalidate=True, runner=_same_domain_only_runner)

            self.assertEqual([item["outcome"] for item in report["revalidated"]], ["restored"])
            self.assertEqual(report["revalidated"][0]["status"], "same_domain_validated")
            self.assertEqual("rung moved from cross_domain_validated to same_domain_validated", report["revalidated"][0]["detail"])
            self.assertEqual(store.lifecycle_status("direct-exp-stale"), "same_domain_validated")
            self.assertEqual(store.revalidation_fingerprint("direct-exp-stale"), TRAINED_JUDGE)
            self.assertEqual(len(store.retrieve(limit=5)), 1)

    def test_a_row_that_falls_out_of_the_injectable_set_is_retired(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._seeded_library(tmp)

            report = sweep_memory_with_judge(experience_library=tmp, incoming=TRAINED_JUDGE, out=Path(tmp) / "sweep", dry_run=False, revalidate=True, runner=_flat_runner)

            self.assertEqual([item["outcome"] for item in report["revalidated"]], ["demoted"])
            self.assertEqual(store.lifecycle_status("direct-exp-stale"), "overfit_or_failed")
            self.assertIsNone(store.revalidation_fingerprint("direct-exp-stale"))
            self.assertEqual(store.retrieve(limit=5), [])

    def test_a_row_whose_suite_is_gone_stays_quarantined_and_says_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore(tmp)
            store.append_result(_row("direct-exp-stale"))

            report = sweep_memory_with_judge(experience_library=tmp, incoming=TRAINED_JUDGE, out=Path(tmp) / "sweep", dry_run=False, revalidate=True, runner=_improving_runner)

            self.assertEqual(report["revalidated"][0]["outcome"], "unverifiable")
            self.assertEqual(report["revalidated"][0]["detail"], "no_recorded_validation_report")
            self.assertEqual(store.lifecycle_status("direct-exp-stale"), "retrying")
            self.assertTrue(report["validation"]["ok"])

    def test_revalidating_without_applying_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                sweep_memory_with_judge(experience_library=tmp, incoming=TRAINED_JUDGE, out=Path(tmp) / "sweep", dry_run=True, revalidate=True)


if __name__ == "__main__":
    unittest.main()
