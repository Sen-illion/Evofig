"""Tests for the four-group case selection, its de-leak rules and the reselection fallback."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.analysis.graph import (
    CASE_GROUPS,
    DEFAULT_CASE_QUOTA,
    case_set_qualification,
    outside_condition,
    paper_family,
    select_validation_cases,
    split_for_group,
)
from rfs.evolution.direct import ExperienceStore, ValidationCase, validate_experience_cases
from rfs.workflows.direct_experience_validation import run_direct_experience_validation


LINEAR = {"linear": 2}
CYCLIC = {"cyclic": 2}

#: The fields a library row needs before it is a measurement rather than a claim.
MEASURED = {
    "repair_gain": 0.25,
    "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.75},
}


def record(paper_id, domain, motif, **extra):
    return {
        "paper_id": paper_id,
        "domain": domain,
        "features": {"node_count": 4, "edge_count": 3, "motif_histogram": motif},
        **extra,
    }


def query(domain="ai_method"):
    return {"paper_id": "query", "domain": domain, "features": {"node_count": 4, "edge_count": 3, "motif_histogram": LINEAR}}


def by_paper(selection):
    return {item["paper_id"]: item for item in selection["selected"]}


class CaseSelectionTests(unittest.TestCase):
    def test_every_group_is_filled_from_the_pool(self):
        pool = [
            record("same_a", "ai_method", LINEAR),
            record("same_b", "ai_method", LINEAR),
            record("cross", "biology", LINEAR),
            record("other", "ai_method", CYCLIC),
            record("edge", "ai_method", LINEAR, topology="reference_guided_multiframe"),
        ]
        selection = select_validation_cases(
            query(),
            pool,
            excluded_conditions=[{"dimension": "topology", "value": "reference_guided_multiframe"}],
        )
        chosen = by_paper(selection)
        self.assertEqual(sorted(chosen), ["cross", "edge", "other", "same_a", "same_b"])
        self.assertEqual(chosen["same_a"]["group"], "same_domain_same_structure")
        self.assertEqual(chosen["cross"]["group"], "cross_domain_same_structure")
        self.assertEqual(chosen["other"]["group"], "same_domain_cross_structure")
        self.assertEqual(chosen["edge"]["group"], "boundary")
        self.assertIn("excluded condition", chosen["edge"]["reason"])
        self.assertEqual(selection["uncovered"], [])
        self.assertEqual(selection["recall"], "structural_only")
        self.assertEqual(selection["quota"], DEFAULT_CASE_QUOTA)
        for name in CASE_GROUPS:
            self.assertEqual(split_for_group(name) in {"same_domain", "holdout", "boundary"}, True, name)

    def test_a_group_nobody_can_supply_is_reported_not_faked(self):
        pool = [record("same_a", "ai_method", LINEAR), record("same_b", "ai_method", LINEAR), record("cross", "biology", LINEAR)]
        selection = select_validation_cases(query(), pool)
        uncovered = {item["group"]: item["reason"] for item in selection["uncovered"]}
        self.assertIn("same_domain_cross_structure", uncovered)
        self.assertIn("boundary", uncovered)
        self.assertTrue(all(reason for reason in uncovered.values()))

    def test_a_quota_of_two_is_two_papers_not_two_records_of_one(self):
        # A paper can hold several graph records, and both of them rank. The quota counts papers,
        # so the second record of a paper already in the set is reported, not admitted, and the
        # set still fills from the papers behind it.
        pool = [
            record("twin", "ai_method", LINEAR, record_id="rec_a"),
            record("twin", "ai_method", LINEAR, record_id="rec_b"),
            record("neighbour", "ai_method", LINEAR),
            record("cross", "biology", LINEAR),
        ]
        selection = select_validation_cases(query(), pool)
        chosen = by_paper(selection)
        same_group = selection["groups"]["same_domain_same_structure"]
        self.assertEqual(len(same_group), len(set(same_group)))
        self.assertEqual(len(same_group), 2)
        self.assertEqual(selection["duplicate_paper_candidates"], 1)
        self.assertIn("neighbour", chosen)
        self.assertIn("twin", chosen)
        reasons = [item["reason"] for item in selection["excluded"]]
        self.assertTrue(any("the same paper is already a case" in reason for reason in reasons))

    def test_a_paper_already_used_in_one_group_cannot_enter_another(self):
        # Dedup is across the whole set, not per group: a paper kept as the same-structure case
        # must not reappear as the boundary case either.
        pool = [
            record("shared", "ai_method", LINEAR, record_id="rec_a"),
            record("shared", "ai_method", LINEAR, record_id="rec_b", topology="reference_guided_multiframe"),
            record("cross", "biology", LINEAR),
        ]
        selection = select_validation_cases(
            query(),
            pool,
            excluded_conditions=[{"dimension": "topology", "value": "reference_guided_multiframe"}],
        )
        selected_papers = [item["paper_id"] for item in selection["selected"]]
        self.assertEqual(len(selected_papers), len(set(selected_papers)))
        self.assertEqual(selection["duplicate_paper_candidates"], 1)

    def test_the_query_paper_and_the_source_paper_are_never_cases(self):
        pool = [record("query", "ai_method", LINEAR), record("source_of_it", "ai_method", LINEAR), record("fine", "ai_method", LINEAR), record("cross", "biology", LINEAR)]
        selection = select_validation_cases(query(), pool, exclude_paper_ids=["source_of_it"])
        excluded = {item["paper_id"]: item["reason"] for item in selection["excluded"]}
        self.assertIn("query", excluded)
        self.assertIn("source_of_it", excluded)
        self.assertNotIn("query", by_paper(selection))
        self.assertNotIn("source_of_it", by_paper(selection))
        self.assertIn("fine", by_paper(selection))

    def test_a_sibling_paper_of_the_query_is_excluded_by_family(self):
        pool = [
            record("sibling", "ai_method", LINEAR, doi="10.1609/aaai.v40i1.37043"),
            record("other_series", "ai_method", LINEAR, doi="10.1016/j.joule.2026.102522"),
        ]
        selection = select_validation_cases(
            {**query(), "doi": "10.1609/aaai.v40i1.99999"},
            pool,
        )
        excluded = {item["paper_id"]: item["reason"] for item in selection["excluded"]}
        self.assertIn("sibling", excluded)
        self.assertIn("family", excluded["sibling"])
        self.assertIn("other_series", by_paper(selection))

    def test_the_quota_bounds_how_many_of_a_group_are_taken(self):
        pool = [record(f"same_{index}", "ai_method", LINEAR) for index in range(5)]
        selection = select_validation_cases(query(), pool)
        self.assertEqual(len(by_paper(selection)), DEFAULT_CASE_QUOTA["same_domain_same_structure"])
        reasons = [item["reason"] for item in selection["excluded"]]
        self.assertTrue(any("quota is already filled" in reason for reason in reasons))

    def test_a_candidate_that_holds_none_of_the_scoped_objects_is_a_boundary_case(self):
        pool = [record("outside", "ai_method", LINEAR, slot_ids=["slot_x"]), record("inside", "ai_method", LINEAR, slot_ids=["slot_1"])]
        selection = select_validation_cases(query(), pool, graph_scope={"slot_ids": ["slot_1"]})
        chosen = by_paper(selection)
        self.assertEqual(chosen["outside"]["group"], "boundary")
        self.assertIn("graph scope", chosen["outside"]["reason"])
        self.assertEqual(chosen["inside"]["group"], "same_domain_same_structure")

    def test_a_domain_experience_does_not_draw_another_domain(self):
        pool = [
            record("same_a", "ai_method", LINEAR),
            record("same_b", "ai_method", LINEAR),
            record("cross", "biology", LINEAR),
            record("other", "ai_method", CYCLIC),
            record("edge", "ai_method", LINEAR, topology="reference_guided_multiframe"),
        ]
        selection = select_validation_cases(
            query(),
            pool,
            applicability="domain",
            excluded_conditions=[{"dimension": "topology", "value": "reference_guided_multiframe"}],
        )
        chosen = by_paper(selection)
        self.assertNotIn("cross", chosen)
        self.assertEqual(chosen["same_a"]["group"], "same_domain_same_structure")
        self.assertEqual(chosen["other"]["group"], "same_domain_cross_structure")
        self.assertEqual(chosen["edge"]["group"], "boundary")
        self.assertEqual(selection["groups"]["cross_domain_same_structure"], [])
        excluded = {item["paper_id"]: item["reason"] for item in selection["excluded"]}
        self.assertIn("another domain", excluded["cross"])

    def test_a_paper_without_a_domain_cannot_enter_a_same_domain_group(self):
        pool = [
            record("blank", "", LINEAR),
            record("unknown_domain", "unknown", LINEAR),
            record("also_blank", "missing", CYCLIC),
            record("named", "ai_method", LINEAR),
        ]
        selection = select_validation_cases(query(), pool, applicability="common")
        chosen = by_paper(selection)
        self.assertIn("named", chosen)
        self.assertNotIn("blank", chosen)
        self.assertNotIn("unknown_domain", chosen)
        both_missing = select_validation_cases(query(""), [record("blank", "", LINEAR), record("also_blank", "", LINEAR)])
        self.assertEqual(both_missing["selected"], [])
        self.assertTrue(all("without a domain" in item["reason"] for item in both_missing["excluded"]))

    def test_an_unlabeled_experience_selects_nothing(self):
        selection = select_validation_cases(query(), [record("same_a", "ai_method", LINEAR)], applicability="")
        self.assertEqual(selection["selected"], [])
        self.assertEqual(selection["recall"], "not_run")
        self.assertTrue(any("applicability was not written" in item["reason"] for item in selection["uncovered"]))


class BoundaryFromTheConditionTests(unittest.TestCase):
    """The negative control an exam needs is the figure the experience's condition does not cover.

    A suite drawn from a paper-level index holds no candidate that shares an object id with the
    experience, and an accepted row excludes nothing, so neither of the two older boundary routes
    can ever fire. Without this route the boundary group stays empty for every row, every exam is
    refused at ``case_set_qualification``, and no experience can climb the ladder at all.
    """

    def covered(self, paper_id, domain, motif, **extra):
        return {
            "paper_id": paper_id,
            "domain": domain,
            "features": {"node_count": 12, "edge_count": 12, "motif_histogram": motif},
            **extra,
        }

    def test_a_figure_the_condition_does_not_cover_is_the_boundary_case(self):
        condition = {"node_count_bucket": "high", "defect_patterns": ["cross_panel"]}
        pool = [
            self.covered("same_a", "ai_method", LINEAR),
            self.covered("same_b", "ai_method", LINEAR),
            self.covered("cross", "biology", LINEAR),
            self.covered("other", "ai_method", CYCLIC),
            self.covered("defect_free", "ai_method", LINEAR, defect_patterns=[]),
        ]

        selection = select_validation_cases(query(), pool, condition=condition)

        chosen = by_paper(selection)
        self.assertEqual(chosen["defect_free"]["group"], "boundary")
        self.assertIn("does not cover", chosen["defect_free"]["reason"])
        self.assertEqual(selection["candidate_counts"]["boundary"], 1)
        self.assertGreaterEqual(selection["candidate_counts"]["same_domain_same_structure"], 2)

    def test_the_exam_can_now_be_qualified_instead_of_refused(self):
        pool = [
            self.covered("same_a", "ai_method", LINEAR),
            self.covered("same_b", "ai_method", LINEAR),
            self.covered("cross", "biology", LINEAR),
            self.covered("other", "ai_method", CYCLIC),
            {"paper_id": "small", "domain": "ai_method", "features": {"node_count": 3, "edge_count": 2, "motif_histogram": LINEAR}},
        ]

        selection = select_validation_cases(query(), pool, condition={"node_count_bucket": "high"})
        qualification = case_set_qualification("common", quota=DEFAULT_CASE_QUOTA, pool_counts=selection["candidate_counts"])

        self.assertTrue(qualification["ok"], qualification)
        self.assertEqual([], qualification["missing"])
        self.assertEqual(by_paper(selection)["small"]["group"], "boundary")

    def test_a_record_that_answers_no_dimension_is_not_made_a_boundary(self):
        # It carries no counts and no defect list, so nothing in the condition can be checked
        # against it. Guessing there would drop a usable same-domain case for no defensible reason.
        blind = {"paper_id": "blind", "domain": "ai_method", "features": {"node_count": 0, "edge_count": 0, "motif_histogram": LINEAR}}
        pool = [self.covered("same_a", "ai_method", LINEAR), self.covered("same_b", "ai_method", LINEAR), blind]

        selection = select_validation_cases(query(), pool, condition={"node_count_bucket": "high", "defect_patterns": ["cross_panel"]})

        self.assertNotEqual("boundary", by_paper(selection).get("blind", {}).get("group"))

    def test_without_a_condition_the_older_routes_are_unchanged(self):
        pool = [self.covered("a", "ai_method", LINEAR)]

        with_condition = select_validation_cases(query(), pool, condition={"node_count_bucket": "low"})
        without = select_validation_cases(query(), pool)

        self.assertEqual(by_paper(with_condition)["a"]["group"], "boundary")
        self.assertEqual(by_paper(without)["a"]["group"], "same_domain_same_structure")


class OutsideConditionTests(unittest.TestCase):
    def _row(self):
        return {"domain": "ai_method", "features": {"node_count": 12, "edge_count": 12, "motif_histogram": LINEAR}}

    def test_it_names_the_dimension_that_does_not_match(self):
        self.assertIn("node_count_bucket=high not low", outside_condition(self._row(), {"node_count_bucket": "low"}))

    def test_it_says_nothing_when_the_record_satisfies_the_condition(self):
        self.assertEqual("", outside_condition(self._row(), {"node_count_bucket": "high"}))

    def test_it_says_nothing_when_the_record_cannot_answer_the_condition(self):
        self.assertEqual("", outside_condition({"domain": "ai_method"}, {"node_count_bucket": "low"}))

    def test_it_says_nothing_without_a_condition(self):
        self.assertEqual("", outside_condition(self._row(), {}))
        self.assertEqual("", outside_condition(self._row(), None))

    def test_a_named_defect_the_figure_does_not_have_is_a_mismatch(self):
        row = {"domain": "ai_method", "defect_patterns": [], "features": {"node_count": 12, "edge_count": 12, "motif_histogram": LINEAR}}

        self.assertIn("defect_patterns lacks cross_panel", outside_condition(row, {"defect_patterns": ["cross_panel"]}))


class PaperFamilyTests(unittest.TestCase):
    def test_arxiv_versions_are_one_family(self):
        self.assertEqual(paper_family({"arxiv_id": "2401.12345v2"}), paper_family({"arxiv_id": "2401.12345"}))

    def test_doi_siblings_share_a_family_and_another_series_does_not(self):
        self.assertEqual(paper_family({"doi": "10.1609/aaai.v40i1.1"}), paper_family({"doi": "10.1609/aaai.v40i1.2"}))
        self.assertNotEqual(paper_family({"doi": "10.1609/aaai.v40i1.1"}), paper_family({"doi": "10.1016/j.joule.2026.102522"}))

    def test_an_unnamed_paper_has_no_family(self):
        self.assertEqual(paper_family({"paper_id": "x"}), "")


class BoundaryRungTests(unittest.TestCase):
    def _report(self, *, boundary_on=0.75, boundary_off=0.75, boundary_blockers=0, groups=True):
        audit = {"paired_config_hash": "h", "initial_state_hash": "h", "candidate_policy_hash": "h", "evaluator_hash": "h"}
        named = {
            "a": ("same_domain", "same_domain_same_structure"),
            "b": ("same_domain", "same_domain_same_structure"),
            "c": ("holdout", "cross_domain_same_structure"),
            "d": ("boundary", "boundary"),
        }
        cases = [
            ValidationCase(case_id, "biology" if case_id == "c" else "ai_method", split, group if groups else "")
            for case_id, (split, group) in named.items()
        ]

        def runner(case, enabled, record):
            if case.case_id == "d":
                return {**audit, "overall_score": boundary_on if enabled else boundary_off, "blocking_issue_count": boundary_blockers if enabled else 0}
            return {**audit, "overall_score": 0.9 if enabled else 0.8, "blocking_issue_count": 0}

        return validate_experience_cases(
            {"experience_id": "e1", "applicability": "common"},
            cases,
            runner,
            expected_groups=list(CASE_GROUPS),
        )

    def test_a_boundary_case_that_does_not_improve_still_promotes(self):
        report = self._report()
        self.assertEqual(report["status"], "promoted")
        self.assertEqual(report["ladder"]["boundary_cases"], 1)
        self.assertEqual(report["ladder"]["total_cases"], 3)
        self.assertEqual(report["ladder"]["group_coverage"]["boundary"], 1)
        # The group this set never asked for is reported, and reported only: it does not block.
        self.assertEqual([item["group"] for item in report["ladder"]["group_uncovered"]], ["same_domain_cross_structure"])

    def test_a_boundary_case_that_leans_against_the_experience_is_not_safe(self):
        report = self._report(boundary_on=0.70, boundary_off=0.75)
        self.assertEqual(report["status"], "overfit_or_failed")
        # The pair was isolated and measured; what failed is the safety gate. Naming that an
        # isolation failure asserts a check that ran and passed.
        self.assertEqual(report["ladder"]["stopped_because"], "new_defect_not_safe")

    def test_a_boundary_case_that_adds_a_blocker_stops_the_ladder(self):
        report = self._report(boundary_blockers=1)
        self.assertEqual(report["status"], "overfit_or_failed")
        self.assertEqual(report["ladder"]["stopped_because"], "new_defect_not_safe")

    def test_a_set_that_drew_no_case_is_not_reported_as_an_isolation_failure(self):
        """No case drawn is the absence of an exam, not a check that came back unfavourable."""

        report = validate_experience_cases(
            {"experience_id": "e1", "applicability": "common"},
            [],
            lambda case, enabled, record: {},
            expected_groups=list(CASE_GROUPS),
        )
        self.assertEqual(report["status"], "overfit_or_failed")
        self.assertEqual(report["ladder"]["stopped_because"], "pair_not_measured")
        # ``all`` over an empty set of pairs says every pair was isolated, which reads as the
        # isolation check having passed when it never ran.
        self.assertFalse(report["all_pairs_isolated"])

    def test_a_group_nobody_supplied_is_reported_but_does_not_block(self):
        audit = {"paired_config_hash": "h", "initial_state_hash": "h", "candidate_policy_hash": "h", "evaluator_hash": "h"}
        cases = [
            ValidationCase("a", "ai_method", "same_domain", "same_domain_same_structure"),
            ValidationCase("b", "ai_method", "same_domain", "same_domain_same_structure"),
            ValidationCase("c", "biology", "holdout", "cross_domain_same_structure"),
        ]

        def runner(case, enabled, record):
            return {**audit, "overall_score": 0.9 if enabled else 0.8, "blocking_issue_count": 0}

        report = validate_experience_cases({"experience_id": "e1", "applicability": "common"}, cases, runner, expected_groups=list(CASE_GROUPS))
        self.assertEqual(report["status"], "cross_domain_validated")
        self.assertEqual(report["ladder"]["stopped_because"], "boundary_uncovered")
        uncovered = {item["group"] for item in report["ladder"]["group_uncovered"]}
        self.assertEqual(uncovered, {"boundary", "same_domain_cross_structure"})
        self.assertEqual(report["ladder"]["group_coverage"]["same_domain_same_structure"], 2)


class CaseSetQualificationTests(unittest.TestCase):
    def test_a_common_exam_that_asks_for_no_cross_domain_case_is_refused(self):
        result = case_set_qualification(
            "common",
            quota={"same_domain_same_structure": 2, "cross_domain_same_structure": 0, "boundary": 1},
            pool_counts={"cross_domain_same_structure": 3, "boundary": 1},
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "case_set_quota_incomplete")
        self.assertEqual(result["missing"], [{"group": "cross_domain_same_structure", "reason": "configured_zero"}])

    def test_a_group_the_pool_cannot_supply_is_refused_for_its_own_reason(self):
        result = case_set_qualification(
            "common",
            quota={"same_domain_same_structure": 2, "cross_domain_same_structure": 1, "boundary": 1},
            pool_counts={"cross_domain_same_structure": 3, "boundary": 0},
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["missing"], [{"group": "boundary", "reason": "no_candidate_in_pool"}])

    def test_a_domain_exam_is_not_asked_to_travel(self):
        result = case_set_qualification(
            "domain",
            quota={"same_domain_same_structure": 2, "boundary": 1},
            pool_counts={"same_domain_same_structure": 4, "boundary": 1},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["required_groups"], ["boundary"])
        self.assertEqual(result["quota"]["cross_domain_same_structure"], 0)

    def test_an_unlabeled_experience_cannot_be_qualified(self):
        result = case_set_qualification("", quota={}, pool_counts={})
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "applicability_unspecified")


class ReselectionTests(unittest.TestCase):
    def _pool(self):
        rows = []
        for index in range(4):
            rows.append({"case_id": f"s{index}", "paper_id": f"s{index}", "paper": f"s{index}.pdf", "domain": "ai_method", "graph_record": record(f"s{index}", "ai_method", LINEAR)})
        for index in range(2):
            rows.append({"case_id": f"x{index}", "paper_id": f"x{index}", "paper": f"x{index}.pdf", "domain": "biology", "graph_record": record(f"x{index}", "biology", LINEAR)})
        # A case holding none of the objects the experience is scoped to: the boundary the exam
        # needs before the ladder can mark the row usable.
        rows.append({"case_id": "b0", "paper_id": "b0", "paper": "b0.pdf", "domain": "ai_method", "graph_record": record("b0", "ai_method", LINEAR, slot_ids=["slot_other"])})
        return rows

    def test_an_unmeasurable_set_is_replaced_once_and_both_conclusions_are_kept(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            self.assertTrue(store.append({
                "experience_id": "e1",
                "applicability": "common",
                "status": "current_case_validated",
                "domain": "direct_editable",
                "scope": "slot",
                "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
                "hypothesis": "a targeted repair",
                "action": {"type": "regen_assets"},
                "graph_scope": {"slot_ids": ["icon_n"]},
                **MEASURED,
            }))
            suite = root / "suite.json"
            suite.write_text(json.dumps({"query_graph_record": record("source", "ai_method", LINEAR), "cases": self._pool()}), encoding="utf-8")

            def runner(**kwargs):
                where = str(kwargs["out"]).replace("\\", "/")
                if "reselected" not in where:
                    # The first set measured nothing: no score, no pair audit.
                    return {"ok": True, "judgement": {}}
                enabled = kwargs.get("seed_experience") is not None
                return {"ok": True, "judgement": {"overall_score": 0.9 if enabled else 0.8, "blocking_issue_count": 0}, "paired_config_hash": "h", "initial_state_hash": "h", "candidate_policy_hash": "h", "evaluator_hash": "h"}

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner)
            events = [item["event"] for item in store.events("e1")]
            final_status = store.lifecycle_status("e1")
            on_disk = json.loads((root / "out" / "cross_paper_validation.json").read_text(encoding="utf-8"))

        self.assertEqual(report["reselection"]["first_status"], "overfit_or_failed")
        self.assertTrue(report["reselection"]["attempted"])
        self.assertEqual(report["reselection"]["status"], report["status"])
        self.assertTrue(report["reselection"]["first_case_ids"])
        self.assertTrue(report["reselection"]["second_case_ids"])
        self.assertNotEqual(set(report["reselection"]["first_case_ids"]), set(report["reselection"]["second_case_ids"]))
        self.assertTrue(report["case_selection"]["used"])
        self.assertIn("another set was drawn", report["reselection"]["reason"])
        self.assertIn("suite_reselected", events)
        self.assertEqual(on_disk["reselection"], report["reselection"])
        self.assertEqual(final_status, report["status"])

    def test_the_switch_keeps_the_first_conclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            store = ExperienceStore(library)
            self.assertTrue(store.append({"experience_id": "e1", "applicability": "common", "status": "current_case_validated", "domain": "direct_editable", "scope": "slot", "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
 "hypothesis": "h", "action": {"type": "regen_assets"}, "graph_scope": {"slot_ids": ["icon_n"]}, **MEASURED}))
            suite = root / "suite.json"
            suite.write_text(json.dumps({"query_graph_record": record("source", "ai_method", LINEAR), "cases": self._pool()}), encoding="utf-8")

            def runner(**kwargs):
                return {"ok": True, "judgement": {}}

            report = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner, reselect_on_failure=False)
            events = [item["event"] for item in store.events("e1")]

        self.assertFalse(report["reselection"]["attempted"])
        self.assertEqual(report["status"], "overfit_or_failed")
        self.assertNotIn("suite_reselected", events)


class IndexLookupTests(unittest.TestCase):
    """The paper a run evolved is named by the run's own path; the exam is keyed by paper id.

    A run is handed ``.../runs/<slug>/attempt_01/inputs/paper``, and a suite's cases name the same
    paper by an archive path of their own, so the two ends share nothing but the id the run
    directory spells. Without reading that id the set cannot be drawn at all and the exam quietly
    measures nothing, which is what this pins.
    """

    SOURCE_ID = "10.1038/s41467-025-66153-5"
    RUN_PATH = "/corpus/runs/10.1038_s41467-025-66153-5/attempt_01/inputs/paper"

    def _suite(self, root: Path, *, index: bool) -> Path:
        rows = [
            record(self.SOURCE_ID, "biology", LINEAR),
            record("10.1/aaa", "biology", LINEAR),
            record("10.1/bbb", "biology", LINEAR),
            # Holds none of the objects the experience is scoped to: the boundary rung.
            record("10.1/ccc", "biology", LINEAR, slot_ids=["slot_other"]),
        ]
        spec: dict = {
            "cases": [
                {"case_id": item["paper_id"], "paper_id": item["paper_id"], "paper": f"/archive/{item['paper_id'].replace('/', '_')}", "domain": "biology"}
                for item in rows
            ]
        }
        if index:
            # An index row without a record_id is not an index row: the reader drops it.
            path = root / "graph_index.jsonl"
            path.write_text("\n".join(json.dumps({**item, "record_id": f"idx.{item['paper_id']}"}) for item in rows) + "\n", encoding="utf-8")
            spec["graph_index"] = str(path)
        suite = root / "suite.json"
        suite.write_text(json.dumps(spec), encoding="utf-8")
        return suite

    def _experience(self, library: Path) -> None:
        store = ExperienceStore(library)
        self.assertTrue(
            store.append(
                {
                    "experience_id": "e1",
                    "applicability": "domain",
                    "applicability_domain": "biology",
                    "status": "current_case_validated",
                    "domain": "direct_editable",
                    "scope": "slot",
                    "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
                    "hypothesis": "a targeted repair",
                    "action": {"type": "regen_assets"},
                    "graph_scope": {"slot_ids": ["icon_n"]},
                    "provenance": {"source": "direct_evolution_process", "run": self.RUN_PATH, "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.75},
                    **{key: value for key, value in MEASURED.items() if key != "provenance"},
                }
            )
        )

    def test_the_paper_the_run_evolved_is_found_by_the_id_its_path_spells(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            self._experience(library)

            def runner(**kwargs):
                enabled = kwargs.get("seed_experience") is not None
                return {"ok": True, "judgement": {"overall_score": 0.9 if enabled else 0.8, "blocking_issue_count": 0}, "paired_config_hash": "h", "initial_state_hash": "h", "candidate_policy_hash": "h", "evaluator_hash": "h"}

            report = run_direct_experience_validation(self._suite(root, index=True), library, "e1", root / "out", runner=runner)

        self.assertTrue(report["case_selection"]["used"])
        self.assertEqual("graph_index", report["subgraph_retrieval"]["query_from"])
        selected = [item["paper_id"] for item in report["case_selection"]["selected"]]
        self.assertTrue(selected)
        # The paper the experience came from reproduces that paper; it is never its own case.
        self.assertNotIn(self.SOURCE_ID, selected)

    def test_a_suite_with_no_index_says_which_half_is_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            self._experience(library)
            report = run_direct_experience_validation(self._suite(root, index=False), library, "e1", root / "out", runner=lambda **kwargs: {"ok": True, "judgement": {}})

        self.assertFalse(report["case_selection"]["used"])
        self.assertIn("no graph_index", report["case_selection"]["reason"])


class ARowThatNamedNoPaperTests(unittest.TestCase):
    """A row that never named the paper it came from is aimed by its own condition and shapes.

    Such a row used to be refused outright: nothing named the paper its set had to be drawn
    against, so no exam was ever set and the row sat unmeasured for good while the report read as
    a row that had been tried. What the row does say about itself -- the condition it was learned
    under and the shapes its own figure held -- is enough to find a figure in the index to draw
    the set against.
    """

    #: A run path that names no paper this index holds: the row the harness used to refuse.
    RUN_PATH = "/archive/papers/2505.22457/arxiv.pdf"

    @staticmethod
    def _record(paper_id: str, domain: str, *, nodes: int, edges: int, motif: dict) -> dict:
        return record(paper_id, domain, motif, features={"node_count": nodes, "edge_count": edges, "motif_histogram": motif})

    def _pool(self) -> list[dict]:
        return [
            self._record("10.1002/aaa", "ai_method", nodes=12, edges=10, motif={"linear": 3, "cross_panel": 2}),
            self._record("10.1002/bbb", "ai_method", nodes=12, edges=10, motif={"linear": 3, "cross_panel": 2}),
            # Another domain holding the same shapes: the case the quick profile asks for.
            self._record("10.1002/ccc", "biology", nodes=12, edges=10, motif={"linear": 3, "cross_panel": 2}),
            # Below the size the row's own condition asks for, so it is a boundary, not a case.
            self._record("10.1002/ddd", "ai_method", nodes=3, edges=2, motif={"cyclic": 2}),
        ]

    def _suite(self, root: Path, pool: list[dict]) -> Path:
        index = root / "graph_index.jsonl"
        index.write_text("\n".join(json.dumps({**item, "record_id": f"idx.{item['paper_id']}"}) for item in pool) + "\n", encoding="utf-8")
        spec = {
            "graph_index": str(index),
            "cases": [
                {"case_id": item["paper_id"], "paper_id": item["paper_id"], "paper": f"/archive/{item['paper_id'].replace('/', '_')}", "domain": item["domain"]}
                for item in pool
            ],
        }
        suite = root / "suite.json"
        suite.write_text(json.dumps(spec), encoding="utf-8")
        return suite

    def _experience(self, library: Path, *, shapes: bool) -> None:
        self.assertTrue(
            ExperienceStore(library).append(
                {
                    "experience_id": "e1",
                    "applicability": "common",
                    "status": "candidate",
                    "domain": "direct_editable",
                    "scope": "global",
                    "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
                    "hypothesis": "a targeted repair",
                    "action": {"type": "layout_constraint"},
                    "apply_at": "ppt_compilation",
                    "trigger": {"motifs": ["linear", "cross_panel"]} if shapes else {},
                    "condition": {"node_count_bucket": "high", "connector_count_bucket": "high"} if shapes else {},
                    "graph_scope": {"slot_ids": ["slot_a"]},
                    "provenance": {"source": "direct_evolution_process", "run": self.RUN_PATH, "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.75},
                    **{key: value for key, value in MEASURED.items() if key != "provenance"},
                }
            )
        )

    @staticmethod
    def _runner(**kwargs):
        enabled = kwargs.get("seed_experience") is not None
        return {
            "ok": True,
            "judgement": {"overall_score": 0.9 if enabled else 0.8, "blocking_issue_count": 0},
            "paired_config_hash": "h",
            "initial_state_hash": "h",
            "candidate_policy_hash": "h",
            "evaluator_hash": "h",
        }

    def test_the_set_is_drawn_against_a_figure_the_rows_own_condition_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            self._experience(library, shapes=True)
            report = run_direct_experience_validation(self._suite(root, self._pool()), library, "e1", root / "out", runner=self._runner)

        self.assertEqual("row_condition", report["subgraph_retrieval"]["query_from"])
        basis = report["subgraph_retrieval"]["basis"]
        self.assertEqual(["cross_panel", "linear"], basis["shapes_the_row_names"])
        self.assertEqual(2, basis["shapes_shared"])
        self.assertEqual(3, basis["records_the_condition_covers"])
        # The basis is a real index record, and it is not a case of the set it draws.
        self.assertEqual("10.1002/aaa", basis["paper_id"])
        chosen = {item["paper_id"]: item["group"] for item in report["case_selection"]["selected"]}
        # The figure it is drawn against is never one of its own cases, and the set still reaches
        # the group the ladder needs: another domain holding the same structure.
        self.assertNotIn("10.1002/aaa", chosen)
        self.assertEqual("cross_domain_same_structure", chosen.get("10.1002/ccc"))
        # Three cases are drawn where a row nobody could aim at drew none, and the holdout rung --
        # another domain holding the same structure -- is one of them.
        self.assertEqual(3, report["summary"]["cases"])
        self.assertEqual(1, report["ladder"]["holdout_cases"])
        self.assertEqual(1, report["ladder"]["boundary_cases"])

    def test_a_row_with_nothing_to_be_aimed_by_is_refused_without_being_spent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            self._experience(library, shapes=False)
            report = run_direct_experience_validation(self._suite(root, self._pool()), library, "e1", root / "out", runner=self._runner)
            # A set nobody could draw is not a verdict about the row: the attempt is recorded in
            # full and the row keeps the rung it already held -- it is not retired for a paper it
            # never named and cannot be aimed by.
            store = ExperienceStore(library)
            status = store.get("e1")["status"]
            events = [str(item.get("event") or "") for item in store.events("e1")]

        self.assertFalse(report["case_selection"]["used"])
        self.assertIn("no shape or condition of the row is answered", report["case_selection"]["reason"])
        self.assertIsNone(report["subgraph_retrieval"].get("basis"))
        self.assertEqual("candidate", status)
        self.assertIn("validation_unmeasured", events)


if __name__ == "__main__":
    unittest.main()
