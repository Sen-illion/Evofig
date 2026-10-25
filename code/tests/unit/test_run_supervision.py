import json
import tempfile
import unittest
from pathlib import Path

from rfs.evaluation.run_supervision import judge_attempt, load_attempt_reports


def accepted_run() -> dict:
    """The reports of a figure that was accepted by hand.

    It carries every signal the pipeline emits all the time: touching cards, two
    links with a poor route, thin panels and a visual review asking for another
    pass.  None of that is breakage, so none of it may buy a retry.
    """

    return {
        "paper_evolve_editable_report.json": {
            "generation_status": "completed",
            "quality_status": "needs_repair",
            "delivery_status": "available",
            "node_count": 22,
            "connector_count": 26,
            "visual_review_mode": "vlm",
            "visual_review_status": "needs_replan",
            "blocking_generation_errors": [],
        },
        "geometry_quality_report.json": {
            "status": "blocked",
            "node_overlap_count": 2,
            "node_overlaps": [
                {"node_a": "a", "node_b": "b", "overlap_area": 0.000147},
                {"node_a": "b", "node_b": "c", "overlap_area": 0.000879},
            ],
            "icon_text_overlap_count": 0,
            "hierarchy_containment_count": 0,
            "connector_label_overlap_count": 0,
            "text_pass_count": 0,
            "node_boundary_violation_count": 0,
        },
        "connector_visibility_report.json": {"status": "needs_reroute", "issue_count": 2, "connector_count": 26},
        "density_quality_report.json": {"status": "blocked", "sparse_panel_count": 3, "panel_count": 3, "empty_region_ratio": 0.59},
        "text_readability_report.json": {"status": "pass", "failure_count": 0, "minimum_font_size_pt": 8.0},
        "rendered_density_report.json": {"status": "pass", "issues": []},
        "graph_alignment_report.json": {"ok": True, "edge_coverage": 1.0, "node_coverage": 1.0},
        "final_quality_gate.json": {"status": "needs_repair", "provenance": "initial_or_local_repair"},
        "color_quality_report.json": {"status": "pass"},
    }


class RunSupervisionTests(unittest.TestCase):
    def test_an_accepted_figure_is_kept(self):
        verdict = judge_attempt(accepted_run(), paper_id="p")
        self.assertEqual(verdict["status"], "clean")
        self.assertEqual(verdict["action"], "keep")
        self.assertEqual(verdict["retry_budget"], 0)

    def test_ordinary_defects_are_still_recorded(self):
        verdict = judge_attempt(accepted_run(), paper_id="p")
        codes = {item["code"] for item in verdict["advisories"]}
        self.assertIn("connector_route", codes)
        self.assertIn("sparse_panel", codes)
        self.assertIn("node_touching", codes)
        self.assertIn("visual_review", codes)
        self.assertEqual(verdict["metrics"]["connector_issue_count"], 2)

    def test_a_correctly_nested_card_does_not_buy_a_retry(self):
        """The geometry gate counts a child inside its container as a success.

        It records that count *instead of* a collision and still passes, so
        reading it as "a child left its parent" paid for a second attempt on
        every nested figure whose links happened to route cleanly.
        """

        reports = accepted_run()
        reports["geometry_quality_report.json"] = {
            **reports["geometry_quality_report.json"],
            "status": "pass",
            "node_overlap_count": 0,
            "node_overlaps": [],
            "hierarchy_containment_count": 3,
        }
        reports["connector_visibility_report.json"] = {"status": "pass", "issue_count": 0, "connector_count": 26}
        reports["density_quality_report.json"] = {"status": "pass", "sparse_panel_count": 0, "panel_count": 3, "empty_region_ratio": 0.12}
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["status"], "clean")
        self.assertEqual(verdict["action"], "keep")
        self.assertEqual(verdict["retry_budget"], 0)
        self.assertEqual(verdict["metrics"]["nested_children_count"], 3)

    def test_a_raised_attempt_is_retried(self):
        verdict = judge_attempt({}, error="RuntimeError: provider closed the connection", paper_id="p")
        self.assertEqual(verdict["category"], "failure")
        self.assertEqual(verdict["retry_budget"], 2)

    def test_a_salvaged_contract_is_recorded_without_a_retry(self):
        """The figure is kept and what was read differently stays visible."""

        reports = accepted_run()
        reports["method_extraction_report.json"] = {
            "ok": True,
            "repairs": ["method object[4] (adapter) reads detail_level 'submodule' as 'module'"],
            "evidence_pending_ids": ["adapter"],
        }
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["status"], "clean")
        self.assertEqual(verdict["retry_budget"], 0)
        self.assertIn("method_contract_salvaged", {item["code"] for item in verdict["advisories"]})

    def test_unsourced_paper_objects_are_recorded_without_a_retry(self):
        reports = accepted_run()
        reports["paper_evolve_editable_report.json"] = {
            **reports["paper_evolve_editable_report.json"],
            "paper_contract_gap_count": 4,
        }
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["status"], "clean")
        self.assertEqual(verdict["retry_budget"], 0)
        self.assertIn("paper_contract_gaps", {item["code"] for item in verdict["advisories"]})
        self.assertEqual(4, verdict["metrics"]["paper_contract_gap_count"])

    def test_a_child_poking_out_of_its_container_is_an_advisory(self):
        """Nesting is not a collision, however much the child overhangs.

        The geometry gate now classifies a declared parent and child as
        containment even when the container was not grown to its payload, so the
        overhang has to be said out loud instead of read as two cards colliding.
        """

        reports = accepted_run()
        reports["geometry_quality_report.json"] = {
            **reports["geometry_quality_report.json"],
            "status": "blocked",
            "node_overlap_count": 0,
            "node_overlaps": [],
            "hierarchy_containment_count": 1,
            "hierarchy_containments": [
                {"parent_node_id": "group", "child_node_id": "card", "contained_area": 0.0123, "outside_area": 0.0041},
            ],
        }
        reports["connector_visibility_report.json"] = {"status": "pass", "issue_count": 0, "connector_count": 26}
        reports["density_quality_report.json"] = {"status": "pass", "sparse_panel_count": 0, "panel_count": 3, "empty_region_ratio": 0.12}
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["status"], "clean")
        self.assertEqual(verdict["retry_budget"], 0)
        self.assertIn("child_overhangs_container", {item["code"] for item in verdict["advisories"]})

    def test_a_child_that_sits_inside_its_container_says_nothing(self):
        reports = accepted_run()
        reports["geometry_quality_report.json"] = {
            **reports["geometry_quality_report.json"],
            "status": "pass",
            "node_overlap_count": 0,
            "node_overlaps": [],
            "hierarchy_containment_count": 1,
            "hierarchy_containments": [
                {"parent_node_id": "group", "child_node_id": "card", "contained_area": 0.02, "outside_area": 0.0},
            ],
        }
        reports["connector_visibility_report.json"] = {"status": "pass", "issue_count": 0, "connector_count": 26}
        reports["density_quality_report.json"] = {"status": "pass", "sparse_panel_count": 0, "panel_count": 3, "empty_region_ratio": 0.12}
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["status"], "clean")
        self.assertNotIn("child_overhangs_container", {item["code"] for item in verdict["advisories"]})

    def test_a_rejected_colour_plan_that_fell_back_is_degraded(self):
        reports = accepted_run()
        reports["color_theme_validation.json"] = {
            "ok": True,
            "mode": "vlm_then_safe_fallback",
            "fallback_used": True,
            "failure_reason": "connectors missing feedback",
        }
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["category"], "degraded")
        self.assertEqual(verdict["retry_budget"], 1)
        self.assertTrue(verdict["metrics"]["color_theme_fallback_used"])

    def test_a_preparation_failure_that_names_missing_evidence_is_not_paid_for(self):
        # The contract stage aborts the paper when the plan stays ungrounded even
        # after the repair round. Re-running reaches the same wall, so the paper
        # is recorded instead of bought a second time.
        verdict = judge_attempt({}, error="ValueError: Paper preparation failed: [\'module m2 has no evidence_ids\']", paper_id="p")
        self.assertEqual(verdict["category"], "upstream")
        self.assertEqual(verdict["status"], "record_only")
        self.assertEqual(verdict["retry_budget"], 0)

    def test_a_missing_run_report_is_retried_once(self):
        verdict = judge_attempt({}, paper_id="p")
        self.assertEqual(verdict["reasons"][0]["code"], "missing_run_report")
        self.assertEqual(verdict["status"], "retry")

    def test_a_contract_blocker_is_recorded_not_paid_for(self):
        reports = accepted_run()
        reports["paper_evolve_editable_report.json"].update(
            {
                "generation_status": "failed",
                "delivery_status": "unavailable",
                "blocking_generation_errors": ["method_contract_invalid"],
            }
        )
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["status"], "record_only")
        self.assertEqual(verdict["category"], "upstream")
        self.assertEqual(verdict["retry_budget"], 0)

    def test_a_missing_deliverable_is_retried(self):
        reports = accepted_run()
        reports["paper_evolve_editable_report.json"].update(
            {
                "generation_status": "failed",
                "delivery_status": "unavailable",
                "blocking_generation_errors": ["editable_pptx_missing_or_empty"],
            }
        )
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["category"], "failure")
        self.assertEqual(verdict["retry_budget"], 2)

    def test_a_real_card_collision_is_retried(self):
        reports = accepted_run()
        reports["geometry_quality_report.json"]["node_overlaps"] = [
            {"node_a": "a", "node_b": "b", "overlap_area": 0.031}
        ]
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["category"], "mechanical")
        self.assertIn("node_collision", {item["code"] for item in verdict["reasons"]})

    def test_a_card_off_the_canvas_is_retried(self):
        reports = accepted_run()
        reports["geometry_quality_report.json"]["node_boundary_violation_count"] = 1
        verdict = judge_attempt(reports, paper_id="p")
        self.assertIn("node_boundary_violation", {item["code"] for item in verdict["reasons"]})

    def test_unreadable_text_is_retried(self):
        reports = accepted_run()
        reports["text_readability_report.json"].update({"status": "blocked", "failure_count": 2})
        verdict = judge_attempt(reports, paper_id="p")
        self.assertIn("text_readability", {item["code"] for item in verdict["reasons"]})

    def test_missing_required_links_are_retried(self):
        reports = accepted_run()
        reports["graph_alignment_report.json"].update({"ok": False, "required_edge_count": 4, "missing_required_edge_ids": ["e1", "e2"]})
        verdict = judge_attempt(reports, paper_id="p")
        self.assertIn("graph_alignment", {item["code"] for item in verdict["reasons"]})

    def test_one_missing_link_of_twelve_is_recorded_not_retried(self):
        # iclr:2026:10006770 drew 11 of its 12 required links and one of its 13
        # required modules; a whole second figure is not the price of the twelfth.
        reports = accepted_run()
        reports["graph_alignment_report.json"].update({"ok": False, "required_edge_count": 12, "required_node_count": 13, "missing_required_edge_ids": ["e1"], "missing_required_node_ids": ["n1"]})
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["status"], "clean")
        self.assertIn("graph_alignment_partial", {item["code"] for item in verdict["advisories"]})
        self.assertEqual(verdict["metrics"]["missing_required_share"], 0.0833)

    def test_a_required_node_without_evidence_does_not_buy_a_second_figure(self):
        # 2025.naacl-long.447 compiled a complete figure and failed alignment only
        # because two required nodes carried no evidence id.  The reason claimed
        # required links were missing while printing that zero of them were.
        reports = accepted_run()
        reports["graph_alignment_report.json"].update({
            "ok": False,
            "required_edge_count": 13,
            "required_node_count": 23,
            "missing_required_edge_ids": [],
            "missing_required_node_ids": [],
            "errors": ["required scientific node has no evidence: snode.b8", "required scientific node has no evidence: snode.2f"],
        })
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["status"], "clean")
        reasons = {item["code"] for item in verdict["advisories"]}
        self.assertIn("graph_alignment_evidence", reasons)
        self.assertNotIn("graph_alignment", {item["code"] for item in verdict["reasons"]})

    def test_a_few_misrouted_links_are_not_enough_to_retry(self):
        reports = accepted_run()
        reports["connector_visibility_report.json"] = {"status": "needs_reroute", "issue_count": 2, "connector_count": 20}
        self.assertEqual(judge_attempt(reports, paper_id="p")["status"], "clean")

    def test_a_quarter_of_the_links_misrouted_is_enough_to_retry(self):
        reports = accepted_run()
        reports["connector_visibility_report.json"] = {"status": "needs_reroute", "issue_count": 5, "connector_count": 20}
        verdict = judge_attempt(reports, paper_id="p")
        self.assertIn("connector_misrouted", {item["code"] for item in verdict["reasons"]})

    def test_one_link_across_text_is_recorded_but_not_retried(self):
        reports = accepted_run()
        reports["geometry_quality_report.json"]["text_pass_count"] = 1
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["status"], "clean")
        self.assertIn("connector_text_pass", {item["code"] for item in verdict["advisories"]})

    def test_two_links_across_text_are_retried(self):
        reports = accepted_run()
        reports["geometry_quality_report.json"]["text_pass_count"] = 2
        self.assertIn("connector_text_pass", {item["code"] for item in judge_attempt(reports, paper_id="p")["reasons"]})

    def test_a_transport_fallback_is_named_apart_from_a_failed_check(self):
        transport = accepted_run()
        transport["method_extraction_report.json"] = {"ok": True, "fallback_used": True, "fallback_reason": "504 Server Error: Gateway Timeout"}
        content = accepted_run()
        content["method_extraction_report.json"] = {"ok": True, "fallback_used": True, "fallback_reason": "model returned an empty module list"}
        transport_codes = {item["code"] for item in judge_attempt(transport, paper_id="p")["reasons"]}
        content_codes = {item["code"] for item in judge_attempt(content, paper_id="p")["reasons"]}
        self.assertIn("method_contract_fallback_transient", transport_codes)
        self.assertIn("method_contract_fallback_content", content_codes)

    def test_a_fallback_spends_one_retry_and_no_more(self):
        reports = accepted_run()
        reports["scientific_hierarchy_validation.json"] = {"fallback_used": True, "fallback_reason": "invalid scientific hierarchy: section[0] requires evidence_ids"}
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["category"], "degraded")
        self.assertEqual(verdict["retry_budget"], 1)

    def test_a_reference_figure_that_cannot_be_read_is_recorded(self):
        reports = accepted_run()
        reports["reference_image_report.json"] = {"status": "failed", "source": "framework.pdf", "error": "PDFiumError: cannot open"}
        verdict = judge_attempt(reports, paper_id="p")
        codes = {item["code"] for item in verdict["reasons"]}
        self.assertTrue(any(code.startswith("reference_image_unreadable") for code in codes), codes)
        self.assertEqual(verdict["category"], "degraded")

    def test_a_reference_figure_that_was_read_is_not_a_reason(self):
        reports = accepted_run()
        reports["reference_image_report.json"] = {"status": "ok", "source": "framework.pdf", "rendered": True}
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["status"], "clean")

    def test_an_invalid_method_contract_outranks_everything_else(self):
        reports = accepted_run()
        reports["method_extraction_report.json"] = {"ok": False, "errors": ["4 objects claim should_render without evidence_ids"]}
        reports["geometry_quality_report.json"]["node_boundary_violation_count"] = 3
        verdict = judge_attempt(reports, paper_id="p")
        self.assertEqual(verdict["category"], "upstream")
        self.assertEqual(verdict["retry_budget"], 0)

    def test_load_attempt_reports_reads_only_what_the_attempt_wrote(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "paper_evolve_editable_report.json").write_text(json.dumps({"generation_status": "completed"}), encoding="utf-8")
            (root / "geometry_quality_report.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
            reports = load_attempt_reports(root)
            self.assertEqual(sorted(reports), ["geometry_quality_report.json", "paper_evolve_editable_report.json"])

    def test_load_attempt_reports_survives_a_half_written_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "geometry_quality_report.json").write_text("{ this is not json", encoding="utf-8")
            self.assertEqual(load_attempt_reports(root), {})


if __name__ == "__main__":
    unittest.main()
