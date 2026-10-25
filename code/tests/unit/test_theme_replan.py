"""The theme re-plan: a colour problem the loop can act on, and the evidence it leaves.

Three things are checked here, in the order the plan asks for them. The action is registered
in every place an action has to be known; a theme re-run really re-plans the palette and says
whether the figure changed; and the measured colour problem reaches both the theme planner and
the advisory role while staying evidence in the Memory row rather than a gain.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.contracts.evolution import REPAIR_ACTIONS
from rfs.evaluation.direct_judges import DirectJudgeInput
from rfs.evaluation.judge_prompt import online_judge_row
from rfs.evolution.direct.actions import SUPPORTED_ACTIONS, apply_program_action
from rfs.evolution.direct.controller import ACTION_PRIORITY, aliased_action_name
from rfs.evolution.direct.experience import ACTION_MEMORY_PROFILE, build_direct_experience_result
from rfs.judge_training.prompts import PROMPT_VERSION
from rfs.judge_training.preference_pairs import PROCESS_ACTION_SCOPE
from rfs.workflows import paper_evolve_editable
from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable
from rfs.workflows.upstream_retry import run_upstream_retry, upstream_retry_kind


PARAGRAPH = "We propose a frozen encoder that turns a paper into a symbolic figure plan."


def _theme(name: str, fill: str) -> dict:
    return {
        "theme_name": name,
        "canvas_background": "#F8FAFC",
        "panels": {"p1": {"fill": fill, "border": "#477FA8", "header": "#477FA8", "header_text": "#FFFFFF", "body_text": "#193247"}},
    }


class ReplanThemeRegistrationTests(unittest.TestCase):
    """A new action has to be known to every reader of the action vocabulary."""

    def test_the_action_is_registered_in_its_five_places(self):
        self.assertIn("replan_theme", REPAIR_ACTIONS)
        self.assertIn("replan_theme", SUPPORTED_ACTIONS)
        self.assertEqual(8, ACTION_PRIORITY["replan_theme"])
        self.assertEqual(("theme", "theme_planning"), ACTION_MEMORY_PROFILE["replan_theme"])
        self.assertEqual("global", PROCESS_ACTION_SCOPE["replan_theme"])

    def test_it_asks_for_a_theme_re_run_and_survives_the_alias_reader(self):
        self.assertEqual("theme", upstream_retry_kind({"type": "replan_theme"}))
        self.assertEqual(("replan_theme", "replan_theme"), aliased_action_name({"type": "replan_theme"}))

    def test_the_served_prompt_version_is_pinned_to_the_shipped_prompt(self):
        # v007 handed over the sample's reference-figure block; v008 adds the bounded-action
        # catalog to the advisory prompt so a named repair is the move the executor runs; v010
        # makes the reflection fill in the rule a later paper is drawn by; v011 lets the advisory
        # role be asked about a pair, so it answers the question the Judge was trained on; v013 lets
        # it name the stage a defect entered at, checked against the round's stage trace; v014 adds
        # the two mechanism-level rules to the menu, with the parameters each of them reads; v015
        # hands the reflection the compile's own report of what the rule did, and asks it to say
        # whether the rule was wrong or was carried out badly before it is thrown away; v016
        # widens the fields a fault attribution may cite, so the seat can point at the evidence
        # block it was actually shown instead of only at the stage trace's own metric keys; v017
        # stops the reflection presupposing a repair request, so round 0 states what it taught;
        # v018 offers follow_up in the policy menu and says what a move on the figure is answered
        # against, so the checks a follow-up needs have something to rest on; v020 writes down when
        # each action is the answer and renders the menu grouped by the layer that answers the fault
        # (v019 is the comparison-verdict wording, which this branch does not carry); v021 hands the
        # advisory role the symptom-to-stage table and asks which requirement the stage it names
        # failed to write, so the answer lands on a sentence a planning prompt could carry.
            self.assertEqual("judge-online-v031", PROMPT_VERSION)

    def test_the_request_pass_reports_a_request_without_recolouring_anything(self):
        state = {"layout": {"nodes": [{"id": "n1", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.1}}]}}
        updated, report = apply_program_action(state, {"type": "replan_theme", "problem": "the canvas is muddy"})
        # The request is a record, so it belongs in the report and not in the state the round
        # compiles: writing it into the state made this pass claim it had applied a repair.
        self.assertFalse(report["applied"])
        self.assertEqual("theme_request", report["program_changes"][0]["object_type"])
        self.assertEqual("the canvas is muddy", report["program_changes"][0]["after"]["problem"])
        self.assertEqual(state, updated)


class RunUpstreamRetryThemeTests(unittest.TestCase):
    def test_a_theme_re_run_re_plans_the_palette_and_reports_the_change(self):
        report = run_upstream_retry(
            {"type": "replan_theme"},
            planner_mode="vlm",
            feedback="the panels are muddy",
            theme=_theme("initial", "#EAF1F8"),
            design={"panels": [{"id": "p1"}], "nodes": []},
            color_feedback={"repair_required": True, "feedback": [{"target_role": "panel.p1.fill"}]},
            theme_planner=lambda feedback: (_theme("replanned", "#334455"), {"ok": True, "received": bool(feedback.get("feedback"))}),
            theme_applier=lambda theme: {"figure_layout_plan": {"panels": [{"id": "p1"}], "nodes": [], "connectors": []}, "panel_style_plan": {"panels": [{"panel_id": "p1", "fill_color_hex": theme["panels"]["p1"]["fill"]}]}},
        )
        self.assertTrue(report["attempted"])
        self.assertTrue(report["applied"])
        self.assertTrue(report["changed"])
        self.assertEqual("theme", report["kind"])
        self.assertEqual("replanned", report["theme"]["theme_name"])
        self.assertTrue(report["theme_report"]["received"])
        self.assertNotEqual(report["before_hash"], report["after_hash"])

    def test_a_re_run_that_returns_the_same_palette_changed_nothing(self):
        same = _theme("initial", "#EAF1F8")
        report = run_upstream_retry(
            {"type": "replan_theme"},
            planner_mode="vlm",
            theme=same,
            design={"panels": [{"id": "p1"}], "nodes": []},
            theme_planner=lambda feedback: (dict(same), {}),
            theme_applier=lambda theme: {"figure_layout_plan": {"panels": [{"id": "p1"}], "nodes": [], "connectors": []}},
        )
        self.assertTrue(report["applied"])
        self.assertFalse(report["changed"])

    def test_a_non_vlm_colour_mode_declines_so_nothing_is_quietly_skipped(self):
        report = run_upstream_retry(
            {"type": "replan_theme"},
            planner_mode="fallback",
            theme=_theme("initial", "#EAF1F8"),
            theme_planner=lambda feedback: (_theme("other", "#111111"), {}),
            theme_applier=lambda theme: {"figure_layout_plan": {"panels": [{"id": "p1"}], "nodes": []}},
        )
        self.assertFalse(report["attempted"])
        self.assertEqual("planner_mode_fallback", report["declined_reason"])

    def test_a_bookkeeping_stamp_on_the_theme_is_not_a_re_colour(self):
        same = _theme("initial", "#EAF1F8")
        stamped = {**same, "provenance": {"source": "vlm", "fallback_used": False}}
        report = run_upstream_retry(
            {"type": "replan_theme"},
            planner_mode="vlm",
            theme=stamped,
            design={"panels": [{"id": "p1"}], "nodes": []},
            theme_planner=lambda feedback: (dict(same), {}),
            theme_applier=lambda theme: {"figure_layout_plan": {"panels": [{"id": "p1"}], "nodes": [], "connectors": []}},
        )
        self.assertTrue(report["applied"])
        self.assertFalse(report["changed"])

    def test_the_switch_declines_a_theme_re_run_like_any_other(self):
        report = run_upstream_retry(
            {"type": "replan_theme"},
            enabled=False,
            planner_mode="vlm",
            theme=_theme("initial", "#EAF1F8"),
            theme_planner=lambda feedback: (_theme("other", "#111111"), {}),
            theme_applier=lambda theme: {"figure_layout_plan": {"panels": [], "nodes": []}},
        )
        self.assertEqual("theme", report["kind"])
        self.assertIn("switched off", report["declined_reason"])

    def test_a_missing_applier_is_recorded_rather_than_reported_as_a_change(self):
        report = run_upstream_retry(
            {"type": "replan_theme"},
            planner_mode="vlm",
            theme=_theme("initial", "#EAF1F8"),
            theme_planner=lambda feedback: (_theme("other", "#111111"), {}),
        )
        self.assertFalse(report["attempted"])
        self.assertIn("theme planner", report["declined_reason"])

    def test_a_failing_theme_provider_is_recorded_rather_than_raised(self):
        def boom(_feedback):
            raise RuntimeError("429 rate limited")

        report = run_upstream_retry(
            {"type": "replan_theme"},
            planner_mode="vlm",
            theme=_theme("initial", "#EAF1F8"),
            theme_planner=boom,
            theme_applier=lambda theme: {"figure_layout_plan": {"panels": [], "nodes": []}},
        )
        self.assertTrue(report["attempted"])
        self.assertEqual("provider_error", report["declined_reason"])
        self.assertIn("429 rate limited", report["provider_error"])


class ColorReportPromptTests(unittest.TestCase):
    def _input(self, **overrides) -> DirectJudgeInput:
        payload = {
            "method": {"figure_goal": "show it", "modules": []},
            "layout": {},
            "icons": {},
            "program": {},
        }
        payload.update(overrides)
        return DirectJudgeInput(**payload)

    def test_the_measured_colour_problem_is_only_carried_when_there_is_one(self):
        without = online_judge_row(self._input(), role="online_bootstrap")
        self.assertNotIn("color_report", without["process_context"])
        with_report = online_judge_row(
            self._input(color_report={"repair_required": True, "feedback": [{"target_role": "panel.p1.fill"}]}),
            role="online_bootstrap",
        )
        self.assertTrue(with_report["process_context"]["color_report"]["repair_required"])

    def test_an_empty_colour_report_does_not_change_the_prompt(self):
        empty = online_judge_row(self._input(color_report={}), role="online_bootstrap")
        self.assertNotIn("color_report", empty["process_context"])


class ColorMetricsExperienceTests(unittest.TestCase):
    def test_the_measured_colour_metrics_are_evidence_on_the_row(self):
        result = build_direct_experience_result(
            round_index=1,
            previous={"overall_score": 0.4},
            current={"overall_score": 0.6, "blocking_issue_count": 0},
            layout={"topology": "pipeline", "nodes": [{"id": "n1"}], "connectors": []},
            action={"type": "replan_theme", "scope_type": "global"},
            source_case="paper.pdf",
            color_metrics={"before": {"minimum_text_contrast": 2.1}, "after": {"minimum_text_contrast": 4.9}},
        )
        self.assertTrue(result["ok"])
        observed = result["experience"]["observed_effect"]
        self.assertEqual(4.9, observed["color_metrics"]["after"]["minimum_text_contrast"])
        self.assertEqual("theme", result["experience"]["memory_type"])
        self.assertEqual("theme_planning", result["experience"]["apply_at"])


def write_paper(root: Path) -> Path:
    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Colour Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(target)
    document.close()
    return target


class ThemeReplanWiringTests(unittest.TestCase):
    """A measured colour problem really becomes next round's theme re-plan."""

    def _run(self, root: Path) -> dict:
        """Run two rounds with the provider seams replaced, so a colour problem can be forced."""

        calls = {"plan": 0, "theme": 0, "feedback": None}
        real_theme = paper_evolve_editable.plan_color_theme_vlm
        scores = {"n": 0}
        advisory_inputs: list[DirectJudgeInput] = []

        class _FakeJudge:
            def __init__(self, role: str):
                self.role = role
                self.model = "fake"
                self.trained = False

            def evaluate(self, judge_input: DirectJudgeInput) -> dict:
                advisory_inputs.append(judge_input)
                if self.role.startswith("online"):
                    return {
                        "contract": "rfs.judge_result",
                        "judge_role": self.role,
                        "overall_score": None,
                        "score_measured": False,
                        "status": "fault",
                        "dimension_scores": {},
                        "stage_feedback": [],
                        "blocking_issues": [],
                        "blocking_issue_count": 0,
                        "confidence": 0.5,
                        "mode": "bootstrap",
                        "trained": False,
                    }
                scores["n"] += 1
                return {
                    "contract": "rfs.judge_result",
                    "judge_role": self.role,
                    # The second measurement is lower, so the theme round is rolled back and the
                    # row it leaves is the refuted one the loop already knows how to write.
                    "overall_score": 0.5 if scores["n"] == 1 else 0.3,
                    "score_measured": True,
                    "status": "fault",
                    "dimension_scores": {},
                    "stage_feedback": [],
                    "blocking_issues": [],
                    "blocking_issue_count": 0,
                    "confidence": 0.5,
                    "mode": "bootstrap",
                    "trained": False,
                }

            def reflect(self, outcome: dict) -> dict:
                reflection = paper_evolve_editable.default_reflection(
                    outcome["decision"], outcome["action"], retry_index=int(outcome["action"].get("retry_index") or 0)
                )
                # This seat answers the round instead of hiding behind the canned fallback: the
                # measured score fell and it stands by that reading, so it does not also call the
                # reading noise. A round that declares its own measurement doubtful is recorded as
                # unmeasured now, which would be a different test from this one.
                reflection.update({"randomness_risk": False, "measurement_noise_risk": False})
                return reflection

        def fake_theme(method, design, text_plan, reference_profile=None, exemplar_style_brief=None, **kwargs):
            theme, report = real_theme(method, design, text_plan, reference_profile, exemplar_style_brief, mode="fallback")
            if kwargs.get("color_feedback"):
                calls["theme"] += 1
                calls["feedback"] = kwargs["color_feedback"]
                theme = dict(theme)
                theme["theme_name"] = "replanned_palette"
                theme["panels"] = {pid: {**(style or {}), "fill": "#8899AA"} for pid, style in (theme.get("panels") or {}).items()}
            else:
                calls["plan"] += 1
            return theme, {**report, "mode": "vlm", "fallback_used": False, "ok": True}

        def fake_color_review(program, preview, deterministic_report, **kwargs):
            return (
                {"status": "needs_repair", "repair_required": True, "feedback": [{"target_role": "panel.01.fill", "severity": "major", "reason": "muddy canvas", "suggested_direction": "cool it down"}]},
                {"ok": True, "mode": "vlm", "errors": [], "feedback_count": 1},
            )

        with patch.object(paper_evolve_editable, "build_progress_reporter", lambda *a, **k: (lambda event: None)), \
                patch.object(paper_evolve_editable, "plan_scientific_hierarchy", lambda *a, **k: ({}, {"mode": "heuristic"})), \
                patch.object(paper_evolve_editable, "plan_color_theme_vlm", side_effect=fake_theme), \
                patch.object(paper_evolve_editable, "review_color_draft", side_effect=fake_color_review), \
                patch.object(paper_evolve_editable, "plan_decomposition_action", lambda *a, **k: {"required": False, "action": "", "violations": []}), \
                patch.object(paper_evolve_editable, "round_proposal_order", lambda **kwargs: [{"type": "replan_theme", "scope_type": "global", "problem": "the measured colour problem"}]), \
                patch.object(paper_evolve_editable, "build_direct_judges", lambda **kwargs: (_FakeJudge("online_bootstrap"), _FakeJudge("frozen_bootstrap"))):
            out = root / "run"
            run_paper_evolve_editable(
                paper=root / "paper.pdf",
                out=out,
                planner_mode="heuristic",
                method_mode="heuristic",
                structure_mode="heuristic",
                text_plan_mode="heuristic",
                ppt_plan_mode="heuristic",
                color_theme_mode="vlm",
                icon_mode="placeholder",
                visual_review_mode="heuristic",
                judge_mode="heuristic",
                max_rounds=2,
                candidates_per_round=1,
                fixed_rounds=True,
                experience_library=root / "library",
            )
        return {"out": out, "calls": calls, "inputs": advisory_inputs}

    def test_a_colour_problem_becomes_a_theme_re_plan_with_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            result = self._run(root)
            out = result["out"]
            retry = json.loads((out / "round_01" / "upstream_retry.json").read_text(encoding="utf-8"))
            execution = json.loads((out / "round_01" / "action_execution.json").read_text(encoding="utf-8"))
            figure_diff = json.loads((out / "round_01" / "candidate_00" / "figure_object_diff.json").read_text(encoding="utf-8"))
            feedback = json.loads((out / "round_00" / "color_feedback.json").read_text(encoding="utf-8"))
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            written = [path.name for path in (out / "round_01").iterdir()]

        self.assertEqual(1, result["calls"]["theme"])
        self.assertEqual("replan_theme", rounds[1]["repair_action"]["type"])
        self.assertEqual("theme", retry["kind"])
        self.assertTrue(retry["applied"])
        self.assertEqual("replanned_palette", retry["theme"]["theme_name"])
        self.assertTrue(retry["changed"])
        # The colour problem the accepted round measured is what the planner was handed, and
        # the advisory role was shown it too before it asked for the re-plan.
        self.assertTrue(result["calls"]["feedback"]["repair_required"])
        self.assertEqual("muddy canvas", result["calls"]["feedback"]["feedback"][0]["reason"])
        self.assertTrue(feedback["feedback"])
        self.assertIn("before", feedback["metrics"])
        self.assertTrue(any(item.color_report and item.color_report.get("repair_required") for item in result["inputs"]))
        # The execution report, the object diff and the figure diff tell one story.
        self.assertTrue(execution["applied"])
        self.assertIn("theme", execution["object_diff"]["changed_fields"])
        self.assertTrue(figure_diff["changed"])
        self.assertTrue(written)

    def test_the_round_that_produces_a_theme_row_also_records_its_patch_decision(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            result = self._run(root)
            out = result["out"]
            rows = list((out / "round_01").glob("experience_*.json")) + list((out / "round_02").glob("experience_*.json"))
            patch_records = list((out / "round_01").glob("prompt_patch_*.json")) + list((out / "round_02").glob("prompt_patch_*.json"))
            experiences = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in rows
                if path.name in {"experience_candidate.json", "experience_refuted.json"}
            ]
        self.assertTrue(experiences, "a theme round that produced no Memory row cannot be audited")
        row = experiences[0]
        self.assertIn("before", row["observed_effect"]["color_metrics"])
        self.assertIn("after", row["observed_effect"]["color_metrics"])
        self.assertIn("decomposability_violations", row["observed_effect"])
        self.assertTrue(patch_records, "a drawn Memory row records either a patch or why it became none")


if __name__ == "__main__":
    unittest.main()
