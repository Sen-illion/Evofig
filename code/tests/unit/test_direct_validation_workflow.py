"""Integration-style tests for cross-paper direct experience validation."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import ExperienceStore
from rfs.workflows.direct_experience_validation import (
    UNDRAWABLE_PAPERS_FILE,
    _note_undrawable_paper,
    _read_undrawable_papers,
    run_direct_experience_validation,
)


PAPER_WITH_NOTHING_TO_DRAW = "Paper preparation failed: ['figure_specification.modules is empty']"


MEASURED = {
    "repair_gain": 0.25,
    "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.75},
}

LINEAR = {"linear": 2}
CYCLIC = {"cyclic": 2}


def graph_record(paper_id, domain, motif=LINEAR, **extra):
    return {"paper_id": paper_id, "domain": domain, "features": {"node_count": 4, "edge_count": 3, "motif_histogram": motif}, **extra}


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


def start_state(options):
    """A real run leaves the state it started from; a stand-in has to leave one too."""

    (Path(options["out"]) / "round_00").mkdir(parents=True, exist_ok=True)
    (Path(options["out"]) / "round_00" / "state_before.json").write_text(json.dumps({"state_hash": "start", "state": START_STATE}), encoding="utf-8")


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
    **MEASURED,
}


class DirectValidationWorkflowTests(unittest.TestCase):
    def test_suite_without_a_graph_index_does_not_run_handwritten_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            store.append({**EXPERIENCE})
            suite = root / "suite.json"
            suite.write_text(json.dumps({"cases": [{"case_id": "a", "paper": "a.pdf", "domain": "ai_method"}, {"case_id": "b", "paper": "b.pdf", "domain": "ai_method"}, {"case_id": "c", "paper": "c.pdf", "domain": "system-platform", "split": "holdout"}]}), encoding="utf-8")

            def runner(**kwargs):
                self.fail("a suite without a graph index must not fall back to handwritten 2+1 cases")

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)
            self.assertFalse(report["case_selection"]["used"])
            self.assertEqual(report["results"], [])
            self.assertNotEqual(report["status"], "promoted")
            self.assertIn("graph_index", report["case_selection"]["reason"])

    def test_suite_promotes_experience_and_records_transition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            store.append({**EXPERIENCE})
            suite = root / "suite.json"
            suite.write_text(json.dumps(four_group_suite()), encoding="utf-8")

            def runner(**kwargs):
                enabled = kwargs.get("seed_experience") is not None
                if enabled:
                    self.assertTrue(kwargs["evolution_enabled"])
                    self.assertGreaterEqual(kwargs["max_rounds"], 2)
                self.assertTrue(kwargs["fixed_rounds"])
                start_state(kwargs)
                return {"ok": True, "judgement": {"overall_score": 0.84 if enabled else 0.8, "blocking_issue_count": 0}}

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)
            self.assertEqual(report["status"], "promoted")
            self.assertTrue(report["all_pairs_isolated"])
            self.assertTrue(all(item["pair_audit"]["ok"] for item in report["results"]))
            self.assertEqual(store.lifecycle_status("e1"), "promoted")

    def test_suite_can_select_cases_from_subgraph_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            self.assertTrue(store.append({"experience_id": "e1", "applicability": "common", "status": "current_case_validated", "domain": "direct_editable", "scope": "slot", "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
 "hypothesis": "targeted repair", "action": {"type": "regen_assets"}, "graph_scope": {"slot_ids": ["icon_n"]}, **MEASURED}))
            def record(paper_id, domain, tags):
                return {"paper_id": paper_id, "domain": domain, "content_tags": tags, "features": {"node_count": 3, "edge_count": 2, "motif_histogram": {"linear": 1}}}
            cases = [
                {"case_id": "a", "paper_id": "a", "paper": "a.pdf", "domain": "ai_method", "graph_record": record("a", "ai_method", ["vision"])},
                {"case_id": "b", "paper_id": "b", "paper": "b.pdf", "domain": "ai_method", "graph_record": record("b", "ai_method", ["language"])},
                {"case_id": "c", "paper_id": "c", "paper": "c.pdf", "domain": "biology", "split": "holdout", "graph_record": record("c", "biology", ["cell"])},
            ]
            suite = root / "suite.json"
            suite.write_text(json.dumps({"query_graph_record": record("source", "ai_method", ["vision"]), "cases": cases}), encoding="utf-8")

            def runner(**kwargs):
                enabled = kwargs.get("seed_experience") is not None
                return {"ok": True, "judgement": {"overall_score": 0.84 if enabled else 0.8, "blocking_issue_count": 0}}

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)
            self.assertTrue(report["subgraph_retrieval"]["used"])
            self.assertEqual(3, len(report["subgraph_retrieval"]["selected"]))
            self.assertEqual("suite", report["subgraph_retrieval"]["query_from"])

    def test_suite_without_a_named_basis_selects_against_its_own_source_paper(self):
        """One suite, several experiences: each set is drawn for the paper its experience came from."""

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            source_paper = str((root / "papers" / "source" / "inputs" / "paper.pdf").resolve())
            measured = {**MEASURED, "provenance": {**MEASURED["provenance"], "run": source_paper}}
            self.assertTrue(store.append({"experience_id": "e1", "applicability": "common", "status": "current_case_validated", "domain": "direct_editable", "scope": "slot", "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
 "hypothesis": "targeted repair", "action": {"type": "regen_assets"}, "graph_scope": {"slot_ids": ["icon_n"]}, **measured}))

            def record(paper_id, domain):
                return {"record_id": f"rec-{paper_id}", "paper_id": paper_id, "domain": domain, "features": {"node_count": 3, "edge_count": 2, "motif_histogram": {"linear": 2}}}

            cases = [
                {"case_id": "source", "paper_id": "src", "paper": source_paper, "domain": "ai_method", "graph_record": record("src", "ai_method")},
                {"case_id": "a", "paper_id": "a", "paper": "a.pdf", "domain": "ai_method", "graph_record": record("a", "ai_method")},
                {"case_id": "b", "paper_id": "b", "paper": "b.pdf", "domain": "ai_method", "graph_record": record("b", "ai_method")},
                {"case_id": "c", "paper_id": "c", "paper": "c.pdf", "domain": "biology", "graph_record": record("c", "biology")},
                {"case_id": "d", "paper_id": "d", "paper": "d.pdf", "domain": "ai_method", "graph_record": {**record("d", "ai_method"), "slot_ids": ["slot_other"]}},
            ]
            suite = root / "suite.json"
            suite.write_text(json.dumps({"selector": {"quota": {"same_domain_same_structure": 2, "cross_domain_same_structure": 1, "boundary": 1}}, "cases": cases}), encoding="utf-8")

            def runner(**kwargs):
                enabled = kwargs.get("seed_experience") is not None
                return {"ok": True, "judgement": {"overall_score": 0.84 if enabled else 0.8, "blocking_issue_count": 0}}

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)
            self.assertEqual("case_graph_record", report["subgraph_retrieval"]["query_from"])
            selected = [str(item["paper_id"]) for item in report["subgraph_retrieval"]["selected"]]
            self.assertTrue(report["subgraph_retrieval"]["used"])
            self.assertNotIn("src", selected)
            self.assertEqual(sorted(["a", "b", "c", "d"]), sorted(selected))
            self.assertNotIn("source", [str(item["case_id"]) for item in report["results"]])

    def test_a_suite_uses_the_index_record_of_its_source_paper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            source_paper = str((root / "papers" / "source" / "inputs" / "paper.pdf").resolve())
            measured = {**MEASURED, "provenance": {**MEASURED["provenance"], "run": source_paper}}
            self.assertTrue(store.append({"experience_id": "e1", "applicability": "domain", "applicability_domain": "biology", "status": "current_case_validated", "domain": "direct_editable", "scope": "slot", "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
 "hypothesis": "targeted repair", "action": {"type": "regen_assets"}, "graph_scope": {"slot_ids": ["icon_n"]}, **measured}))

            def record(paper_id, domain):
                return {"record_id": f"rec-{paper_id}", "paper_id": paper_id, "domain": domain, "features": {"node_count": 3, "edge_count": 2, "motif_histogram": {"linear": 2}}}

            index = root / "graph_index.jsonl"
            index_records = [record(paper_id, domain) for paper_id, domain in (("src", "biology"), ("a", "biology"), ("b", "biology"), ("c", "cv"))]
            # A case holding none of the objects the experience is scoped to: the boundary this
            # exam needs before it is allowed to run at all.
            index_records.append({**record("d", "biology"), "slot_ids": ["slot_other"]})
            index.write_text("\n".join(json.dumps(item) for item in index_records) + "\n", encoding="utf-8")
            cases = [
                {"case_id": "source", "paper_id": "src", "paper": source_paper, "domain": "biology"},
                {"case_id": "a", "paper_id": "a", "paper": "a.pdf", "domain": "biology"},
                {"case_id": "b", "paper_id": "b", "paper": "b.pdf", "domain": "biology"},
                {"case_id": "c", "paper_id": "c", "paper": "c.pdf", "domain": "cv"},
                {"case_id": "d", "paper_id": "d", "paper": "d.pdf", "domain": "biology"},
            ]
            suite = root / "suite.json"
            suite.write_text(
                json.dumps({"graph_index": str(index), "selector": {"quota": {"same_domain_same_structure": 2, "cross_domain_same_structure": 0, "same_domain_cross_structure": 1, "boundary": 1}}, "cases": cases}),
                encoding="utf-8",
            )

            def runner(**kwargs):
                enabled = kwargs.get("seed_experience") is not None
                return {"ok": True, "judgement": {"overall_score": 0.84 if enabled else 0.8, "blocking_issue_count": 0}}

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)
            self.assertEqual("graph_index", report["subgraph_retrieval"]["query_from"])
            selected = [str(item["paper_id"]) for item in report["subgraph_retrieval"]["selected"]]
            self.assertNotIn("src", selected)
            self.assertNotIn("c", selected)
            self.assertTrue(selected)

    def test_named_basis_in_the_suite_keeps_its_own_case_set(self):
        """An explicit basis still wins, so a suite that judges one experience keeps its own set."""

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            source_paper = str((root / "papers" / "source" / "inputs" / "paper.pdf").resolve())
            measured = {**MEASURED, "provenance": {**MEASURED["provenance"], "run": source_paper}}
            self.assertTrue(store.append({"experience_id": "e1", "applicability": "common", "status": "current_case_validated", "domain": "direct_editable", "scope": "slot", "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
 "hypothesis": "targeted repair", "action": {"type": "regen_assets"}, "graph_scope": {"slot_ids": ["icon_n"]}, **measured}))

            def record(paper_id, domain):
                return {"record_id": f"rec-{paper_id}", "paper_id": paper_id, "domain": domain, "features": {"node_count": 3, "edge_count": 2, "motif_histogram": {"linear": 2}}}

            cases = [
                {"case_id": "source", "paper_id": "src", "paper": source_paper, "domain": "ai_method", "graph_record": record("src", "ai_method")},
                {"case_id": "a", "paper_id": "a", "paper": "a.pdf", "domain": "ai_method", "graph_record": record("a", "ai_method")},
                {"case_id": "c", "paper_id": "c", "paper": "c.pdf", "domain": "biology", "graph_record": record("c", "biology")},
                {"case_id": "d", "paper_id": "d", "paper": "d.pdf", "domain": "ai_method", "graph_record": {**record("d", "ai_method"), "slot_ids": ["slot_other"]}},
            ]
            suite = root / "suite.json"
            suite.write_text(json.dumps({"query_graph_record": record("q", "ai_method"), "selector": {"quota": {"same_domain_same_structure": 2, "cross_domain_same_structure": 1, "boundary": 1}}, "cases": cases}), encoding="utf-8")

            def runner(**kwargs):
                enabled = kwargs.get("seed_experience") is not None
                return {"ok": True, "judgement": {"overall_score": 0.84 if enabled else 0.8, "blocking_issue_count": 0}}

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)
            self.assertEqual("suite", report["subgraph_retrieval"]["query_from"])
            selected = [str(item["paper_id"]) for item in report["subgraph_retrieval"]["selected"]]
            self.assertIn("src", selected)

    def test_a_domain_experience_is_selected_and_promoted_inside_its_domain(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            store.append({**EXPERIENCE, "applicability": "domain", "applicability_domain": "ai_method"})
            suite = root / "suite.json"
            suite.write_text(json.dumps(four_group_suite()), encoding="utf-8")

            def runner(**kwargs):
                enabled = kwargs.get("seed_experience") is not None
                start_state(kwargs)
                return {"ok": True, "judgement": {"overall_score": 0.84 if enabled else 0.8, "blocking_issue_count": 0}}

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)
            selected = [str(item.get("paper_id") or "") for item in report["case_selection"]["selected"]]
            self.assertEqual(report["case_selection"]["applicability"], "domain")
            self.assertNotIn("c", selected)
            self.assertEqual(report["case_selection"]["groups"]["cross_domain_same_structure"], [])
            self.assertEqual(report["status"], "domain_validated")
            self.assertEqual(store.lifecycle_status("e1"), "domain_validated")

    def test_a_suite_that_asks_for_none_of_a_required_group_is_refused_before_any_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            store.append({**EXPERIENCE})
            suite = root / "suite.json"
            suite.write_text(
                json.dumps({**four_group_suite(), "selector": {"quota": {"same_domain_same_structure": 2, "cross_domain_same_structure": 1, "boundary": 0}}}),
                encoding="utf-8",
            )

            def runner(**kwargs):
                self.fail("a case set that cannot reach the ladder must not be run")

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)
            on_disk = json.loads((root / "out" / "case_set_qualification.json").read_text(encoding="utf-8"))
            events = [item["event"] for item in store.events("e1")]
            final_status = store.lifecycle_status("e1")

        self.assertEqual(report["results"], [])
        # An exam that cannot be set is a dead end, not a plain untried candidate: it is named and
        # retired here so the loop stops handing it back on every call.
        self.assertEqual(report["status"], "unverifiable")
        self.assertEqual(report["reason"], "case_set_quota_incomplete")
        self.assertEqual(report["blocked_because"], "case_set_quota_incomplete")
        self.assertEqual(report["case_set_qualification"]["missing"], [{"group": "boundary", "reason": "configured_zero"}])
        self.assertEqual(on_disk["missing"], report["case_set_qualification"]["missing"])
        self.assertEqual(report["ladder"]["stopped_because"], "case_set_quota_incomplete")
        self.assertIn("case_set_refused", events)
        self.assertEqual(final_status, "unverifiable")

    def test_an_unlabeled_experience_is_not_given_a_case_set(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            unlabeled = {key: value for key, value in EXPERIENCE.items() if key != "applicability"}
            store.append(unlabeled)
            suite = root / "suite.json"
            suite.write_text(json.dumps(four_group_suite()), encoding="utf-8")

            def runner(**kwargs):
                self.fail("an experience with no applicability must not draw cases")

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)
            events = [item["event"] for item in store.events("e1")]
            final_status = store.lifecycle_status("e1")

        self.assertEqual(report["case_selection"]["selected"], [])
        self.assertEqual(report["status"], "unverifiable")
        self.assertEqual(report["blocked_because"], "applicability_unspecified")
        self.assertEqual(report["ladder"]["stopped_because"], "applicability_unspecified")
        self.assertIn("row_unverifiable", events)
        self.assertEqual(final_status, "unverifiable")

    def test_a_row_the_pair_cannot_hold_equal_is_retired_before_any_generation(self):
        """A stage with no shareable prefix cannot be paired, so the exam is refused up front.

        ``global_planning`` is the case: freezing the design would freeze away the very move
        being measured, and letting both arms plan their own figure would compare two different
        starting points. Paying for two full runs to discover that is the waste this pins.
        """

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            store.append({**EXPERIENCE, "apply_at": "global_planning"})
            suite = root / "suite.json"
            suite.write_text(json.dumps(four_group_suite()), encoding="utf-8")

            def runner(**kwargs):
                self.fail("a row whose pair cannot be held equal must not generate anything")

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)
            events = [item["event"] for item in store.events("e1")]
            final_status = store.lifecycle_status("e1")

        self.assertEqual(report["status"], "unverifiable")
        self.assertEqual(report["blocked_because"], "stage_not_forkable")
        self.assertEqual(report["summary"]["cases"], 0)
        self.assertEqual(report["results"], [])
        self.assertIn("row_unverifiable", events)
        self.assertEqual(final_status, "unverifiable")

    def test_on_off_contrast_sees_the_budget_not_only_the_action_name(self):
        from rfs.utils import write_json
        from rfs.workflows.direct_experience_validation import _executed_actions

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "evolution_rounds.json", {"rounds": [
                {"round": 0, "repair_action": {"type": "initial"}, "bounded_change": {"action": {"type": "initial"}}},
                {"round": 1, "repair_action": {"type": "relayout_global"}, "bounded_change": {"action": {"type": "relayout_global", "dx": 0.016}, "applied_budget": {"factor": 1.02, "dx": 0.016, "dy": 0.0}}},
            ]})
            self.assertEqual(
                ["initial", "relayout_global|factor=1.02|dx=0.016|dy=0.0"],
                _executed_actions(root),
            )


    def test_an_arm_that_already_ran_is_not_paid_for_again(self):
        """A validation is two arms per case, and one arm is a full generation.

        The set only becomes resumable at arm granularity: if a call dies after the control arm
        of a case finished but before its memory arm did, restarting must keep that arm's
        measurement rather than generate the same figure a second time.
        """

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            store.append({**EXPERIENCE})
            suite = root / "suite.json"
            suite.write_text(json.dumps(four_group_suite()), encoding="utf-8")
            arms: list[tuple[str, bool]] = []

            def runner(**kwargs):
                enabled = kwargs.get("seed_experience") is not None
                arms.append((Path(kwargs["out"]).parent.name, enabled))
                start_state(kwargs)
                return {"ok": True, "judgement": {"overall_score": 0.84 if enabled else 0.8, "blocking_issue_count": 0}}

            first = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)
            first_arms = list(arms)

            # A crash that landed before any case was decided leaves the arms on disk but no
            # decided case: the restart must reuse those arms instead of generating them again.
            cases = root / "out" / "cases"
            for path in sorted(cases.glob("*.json")):
                path.unlink()
            arms.clear()
            restarted = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)

        self.assertEqual("promoted", first["status"])
        self.assertEqual([], arms)
        self.assertEqual("promoted", restarted["status"])
        self.assertEqual(first["summary"], restarted["summary"])
        self.assertEqual(10, len(first_arms))

    def test_a_paper_that_cannot_be_drawn_is_unmeasured_rather_than_fatal(self):
        """A target paper with nothing to draw is a fact about the case, not about the row.

        The old reading let that refusal out of the pair runner, so one undrawable case ended the
        whole measurement: an hour of wall clock and one of the row's few retries were spent on a
        paper that was never drawn. It is reported as an unmeasured case here, which is a reason to
        draw another set, and the rung the row holds is left standing.
        """

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            store.append({**EXPERIENCE})
            suite = root / "suite.json"
            suite.write_text(json.dumps(four_group_suite()), encoding="utf-8")
            calls = []

            def runner(**kwargs):
                calls.append(str(kwargs["paper"]))
                if str(kwargs["paper"]) == "a.pdf":
                    raise ValueError(PAPER_WITH_NOTHING_TO_DRAW)
                enabled = kwargs.get("seed_experience") is not None
                start_state(kwargs)
                return {"ok": True, "judgement": {"overall_score": 0.84 if enabled else 0.8, "blocking_issue_count": 0}}

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)

            self.assertNotEqual("failed", report["status"])
            self.assertEqual("pair_not_measured", report["ladder"]["stopped_because"])
            self.assertEqual(["a"], [item["case_id"] for item in report["cases_without_a_figure"]])
            unusable = next(item for item in report["results"] if item["case_id"] == "a")
            self.assertFalse(unusable["measured"])
            self.assertEqual("paper_cannot_be_prepared", unusable["off"]["pair_stop_reason"])
            # The row is not spent on a set that never measured it.
            self.assertEqual("current_case_validated", store.lifecycle_status("e1"))
            # The paper is remembered, and its memory arm is never paid for.
            remembered = json.loads((root / UNDRAWABLE_PAPERS_FILE).read_text(encoding="utf-8"))
            self.assertEqual(["a"], remembered["paper_ids"])
            self.assertEqual(1, calls.count("a.pdf"))

    def test_a_paper_that_carries_no_figure_is_not_drawn_again(self):
        """A set drawn before is read back, unless it names a paper already found undrawable."""

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            store.append({**EXPERIENCE})
            suite = root / "suite.json"
            suite.write_text(json.dumps(four_group_suite()), encoding="utf-8")
            (root / UNDRAWABLE_PAPERS_FILE).write_text(json.dumps({"paper_ids": ["c"]}), encoding="utf-8")
            selection = root / "out" / "validation_case_selection.json"
            selection.parent.mkdir(parents=True, exist_ok=True)
            selection.write_text(
                json.dumps({"used": True, "selected": [{"paper_id": "c", "group": "cross_domain_same_structure", "split": "holdout"}], "groups": {}}),
                encoding="utf-8",
            )

            def runner(**kwargs):
                enabled = kwargs.get("seed_experience") is not None
                start_state(kwargs)
                return {"ok": True, "judgement": {"overall_score": 0.84 if enabled else 0.8, "blocking_issue_count": 0}}

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)

        self.assertEqual(["c"], report["papers_without_a_figure"])
        self.assertFalse(report["case_selection"].get("resumed"))
        chosen = {str(item.get("paper_id") or item.get("case_id") or "") for item in (report["case_selection"].get("selected") or [])}
        self.assertNotIn("c", chosen)

    def test_the_papers_without_a_figure_list_keeps_one_entry_per_paper(self):
        """The list lives beside the attempt and names each paper once, however often it is seen."""

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "wave_04"
            root.mkdir(parents=True)

            _note_undrawable_paper(root, "a", "first")
            _note_undrawable_paper(root, "a", "again")
            _note_undrawable_paper(root, "b", "second")

            self.assertEqual(["a", "b"], _read_undrawable_papers(root))
            self.assertTrue((Path(temp) / UNDRAWABLE_PAPERS_FILE).is_file())


if __name__ == "__main__":
    unittest.main()
