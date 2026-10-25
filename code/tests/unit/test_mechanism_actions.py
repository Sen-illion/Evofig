"""Mechanism-level actions: rules about how a figure is drawn, not moves on one figure.

A bounded repair names objects in the figure in front of it, so what it learns cannot act on the
next paper. These two actions carry a rule instead -- one about routing, one about layout -- and
the compile that draws a later figure honours it. The tests below check that the actions are
registered everywhere a new action has to be known, that the rule is stamped where the compiler
reads it while a bounded repair keeps meaning a move on this figure, that the compiler applies
the rule, and that the same rule changes two figures which share nothing but the rule.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.composition import CURATED_LAYOUT_RULES, apply_layout_constraints, solve_layout_constraints, style_and_route_arrows
from rfs.composition.arrow_routing import ROUTER_CONSTRAINT_VERSION
from rfs.contracts.evolution import EXPERIENCE_ACTIONS, MECHANISM_ACTIONS, REPAIR_ACTIONS
from rfs.contracts.repair_actions import ACTION_SEMANTICS, validate_action_catalog
from rfs.contracts.memory_parameters import (
    injection_blocked_reason,
    layout_rules_from,
    router_rules_from,
    routing_constraints_from,
    stamp_executable_parameters,
)
from rfs.evolution.direct.actions import EXECUTABLE_ACTIONS, SUPPORTED_ACTIONS, apply_program_action
from rfs.evolution.direct.experience import ACTION_MEMORY_PROFILE
from rfs.evolution.direct.snapshots import diff_figure_programs
from rfs.utils import write_json
from rfs.workflows.direct_repair_execution import write_mechanism_rule_report
from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable


PARAGRAPH = "We propose a frozen encoder that turns a paper into a symbolic figure plan."


def _router_figure(tag: str, *, arrows: list[dict] | None = None, nodes: int = 2) -> dict:
    """A figure whose ids all carry ``tag``, so two of these share nothing but their shape."""

    ids = [f"node_{tag}_{index}" for index in range(nodes)]
    members = [
        {"id": node_id, "panel_id": f"panel_{tag}", "bbox_percent": {"x": round(0.06 + 0.12 * index, 4), "y": 0.24, "w": 0.09, "h": 0.09}}
        for index, node_id in enumerate(ids)
    ]
    if arrows is None:
        arrows = [{
            "id": f"conn_{tag}_1",
            "source_id": ids[0],
            "target_id": ids[-1],
            "path_percent": [[0.105, 0.285], [0.3, 0.285], [0.3, 0.5], [0.3, 0.66]],
        }]
    return {
        "canvas": {"width_in": 10, "height_in": 5},
        "style": {"color_tokens": [{"token_id": "arrow_001", "hex": "#111111"}]},
        "panels": [{"id": f"panel_{tag}", "bbox_percent": {"x": 0.05, "y": 0.2, "w": 0.5, "h": 0.6}}],
        "slots": [],
        "semantic_nodes": members,
        "arrows": arrows,
    }


def _layout_figure(tag: str, *, members: list[dict] | None = None) -> dict:
    if members is None:
        members = [
            {"id": f"node_{tag}_a", "panel_id": f"panel_{tag}", "bbox_percent": {"x": 0.081, "y": 0.253, "w": 0.093, "h": 0.077}},
            {"id": f"node_{tag}_b", "panel_id": f"panel_{tag}", "bbox_percent": {"x": 0.294, "y": 0.621, "w": 0.086, "h": 0.081}},
        ]
    return {
        "canvas": {"width_in": 10, "height_in": 5},
        "style": {"color_tokens": []},
        "panels": [{"id": f"panel_{tag}", "bbox_percent": {"x": 0.05, "y": 0.2, "w": 0.44, "h": 0.62}}],
        "semantic_nodes": members,
        "slots": [],
        "arrows": [],
    }


def _snapped(value: float, grid: float) -> bool:
    return abs(float(value) - round(float(value) / grid) * grid) < 0.0005


class MechanismActionRegistrationTests(unittest.TestCase):
    def test_the_two_actions_are_legal_experience_actions(self):
        for name in ("router_constraint", "layout_constraint"):
            self.assertIn(name, MECHANISM_ACTIONS)
            self.assertIn(name, EXPERIENCE_ACTIONS)
            self.assertIn(name, EXECUTABLE_ACTIONS)
            self.assertIn(name, ACTION_MEMORY_PROFILE)
            self.assertNotIn(name, SUPPORTED_ACTIONS)

    def test_the_menu_offers_the_two_rules_beside_the_ten_repairs(self):
        # The menu is the whitelist rendered into the advisory prompt, and a rule the Judge
        # cannot name is a channel with no way in: both actions stayed dead while the executor,
        # the stamp and the compiler already read them. What must not move is what a repair
        # means, so the set the Judge-training data reads for ``action_is_repair`` stays at ten.
        for name in ("router_constraint", "layout_constraint"):
            self.assertIn(name, REPAIR_ACTIONS)
            self.assertIn(name, ACTION_SEMANTICS)
            self.assertNotIn(name, SUPPORTED_ACTIONS)
        self.assertEqual(12, len(REPAIR_ACTIONS))
        self.assertEqual(10, len(SUPPORTED_ACTIONS))
        self.assertEqual([], validate_action_catalog())

    def test_a_rule_is_not_a_name_without_its_parameters(self):
        # A rule carries parameters where a repair carries object ids, so the name alone is not
        # a prescription: running it would spend a round on a rule the compiler cannot read.
        from rfs.evolution.direct.controller import select_bounded_action

        bare = select_bounded_action({"stage_feedback": [{"problem": "gaps", "requested_actions": [{"action": "layout_constraint"}]}]})
        self.assertIsNone(bare)
        chosen = select_bounded_action({"stage_feedback": [{"problem": "gaps", "requested_actions": [{"action": "layout_constraint", "params": {"gutter_min": 0.02}}]}]})
        self.assertEqual("layout_constraint", chosen["type"])
        self.assertEqual({"gutter_min": 0.02}, chosen["params"])
        routed = select_bounded_action({"stage_feedback": [{"problem": "crossing", "requested_actions": [{"action": "router_constraint", "constraints": {"forbid_crossing": True}}]}]})
        self.assertEqual("router_constraint", routed["type"])
        self.assertEqual({"forbid_crossing": True}, routed["constraints"])

    def test_they_are_read_where_the_figure_is_drawn(self):
        self.assertEqual(("connector", "ppt_compilation"), ACTION_MEMORY_PROFILE["router_constraint"])
        self.assertEqual(("layout", "ppt_compilation"), ACTION_MEMORY_PROFILE["layout_constraint"])

    def test_the_executor_records_the_rule_and_moves_nothing(self):
        state = {"layout": {"nodes": [{"id": "n1", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.1}}]}}
        updated, report = apply_program_action(
            state, {"type": "router_constraint", "constraints": {"forbid_crossing": True}}
        )
        # The rule is about the compile that follows, so this pass must not claim a state change.
        self.assertFalse(report["applied"])
        self.assertEqual(state, updated)
        self.assertEqual("mechanism_rule", report["program_changes"][0]["object_type"])
        self.assertEqual("router_constraint", report["program_changes"][0]["after"]["rule_type"])
        self.assertEqual({"forbid_crossing": True}, report["program_changes"][0]["after"]["rule"])

    def test_a_layout_rule_is_recorded_the_same_way(self):
        updated, report = apply_program_action({"layout": {}}, {"type": "layout_constraint", "params": {"align_grid": 0.02}})
        self.assertFalse(report["applied"])
        self.assertEqual({"layout": {}}, updated)
        self.assertEqual({"align_grid": 0.02}, report["program_changes"][0]["after"]["rule"])


class MechanismRuleStampingTests(unittest.TestCase):
    def test_the_rule_is_stamped_where_the_compiler_reads_it(self):
        row = {"type": "router_constraint", "apply_at": "ppt_compilation", "constraints": {"forbid_crossing": True, "max_segments": 3}}
        stamped = stamp_executable_parameters(row)
        self.assertEqual({"router": {"forbid_crossing": True, "max_segments": 3}}, stamped["routing_constraints"])
        self.assertEqual({"forbid_crossing": True, "max_segments": 3}, router_rules_from(stamped["routing_constraints"]))

    def test_a_layout_rule_is_stamped_under_its_own_key(self):
        stamped = stamp_executable_parameters({"type": "layout_constraint", "params": {"align_grid": 0.02, "gutter_min": 0.01}})
        self.assertEqual({"params": {"align_grid": 0.02, "gutter_min": 0.01}}, stamped["layout_constraints"])
        self.assertEqual({"align_grid": 0.02, "gutter_min": 0.01}, layout_rules_from(stamped["layout_constraints"]))

    def test_a_row_that_states_no_rule_cannot_migrate(self):
        self.assertTrue(injection_blocked_reason({"type": "router_constraint"}))
        self.assertTrue(injection_blocked_reason({"type": "layout_constraint"}))
        self.assertEqual("", injection_blocked_reason({"type": "router_constraint", "constraints": {"forbid_crossing": True}}))

    def test_a_route_replay_of_ids_alone_carries_no_rule(self):
        self.assertEqual({}, router_rules_from({"selections": [{"arrow_id": "rel_12"}]}))
        self.assertEqual({"router": {"forbid_crossing": True}}, routing_constraints_from({"router": {"forbid_crossing": True}}))

    def test_a_bad_rule_value_is_not_read_as_a_rule(self):
        self.assertEqual({}, router_rules_from({"constraints": {"forbid_crossing": "yes", "max_segments": 0}}))
        self.assertEqual({}, router_rules_from({"constraints": {"min_bend": True}}))


class RouterConstraintTests(unittest.TestCase):
    def _profile(self, tmp: str) -> dict:
        return json.loads(Path(tmp, "arrow_style_profile.json").read_text(encoding="utf-8"))

    def test_a_segment_cap_redraws_the_path_that_was_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            program = _router_figure("a")
            result = style_and_route_arrows(program, tmp, mode="reference", constraints={"max_segments": 1})
            arrow = result["arrows"][0]
            self.assertEqual(1, len(arrow["path_percent"]) - 1)
            self.assertEqual(ROUTER_CONSTRAINT_VERSION, arrow["router_constraint_version"])
            self.assertEqual({"max_segments": 1}, arrow["router_constraints"])
            profile = self._profile(tmp)
            self.assertEqual(ROUTER_CONSTRAINT_VERSION, profile["router_constraint_version"])
            self.assertTrue(any(item["rule"] == "max_segments" for item in profile["router_constraint_applied"]))

    def test_no_rule_leaves_no_constraint_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = style_and_route_arrows(_router_figure("a"), tmp, mode="reference")
            self.assertNotIn("router_constraints", result["arrows"][0])
            self.assertEqual("", self._profile(tmp)["router_constraint_version"])

    def test_connectors_leaving_one_node_share_one_vertical_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrows = [
                {"id": "conn_a_1", "source_id": "node_a_0", "target_id": "node_a_1", "path_percent": [[0.105, 0.285], [0.30, 0.285], [0.30, 0.55], [0.33, 0.66]]},
                {"id": "conn_a_2", "source_id": "node_a_0", "target_id": "node_a_1", "path_percent": [[0.105, 0.285], [0.40, 0.285], [0.40, 0.60], [0.36, 0.70]]},
            ]
            result = style_and_route_arrows(_router_figure("a", arrows=arrows), tmp, mode="reference", constraints={"shared_vertical_channel": True})
            lanes = []
            for arrow in result["arrows"]:
                for start, end in zip(arrow["path_percent"], arrow["path_percent"][1:]):
                    if abs(float(start[0]) - float(end[0])) < 0.001:
                        lanes.append(round(float(start[0]), 4))
            self.assertTrue(lanes)
            self.assertEqual(1, len(set(lanes)), lanes)
            profile = self._profile(tmp)
            # Only the connector that actually moved is recorded; the one already on the
            # channel is compliant rather than changed.
            shared = [item for item in profile["router_constraint_applied"] if item["rule"] == "shared_vertical_channel"]
            self.assertGreaterEqual(len(shared), 1)

    def test_a_crossing_is_either_redrawn_or_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrows = [
                {"id": "conn_a_1", "source_id": "node_a_0", "target_id": "node_a_1", "path_percent": [[0.20, 0.30], [0.20, 0.80]]},
                {"id": "conn_a_2", "source_id": "node_a_0", "target_id": "node_a_1", "path_percent": [[0.10, 0.55], [0.45, 0.55]]},
            ]
            result = style_and_route_arrows(_router_figure("a", arrows=arrows), tmp, mode="reference", constraints={"forbid_crossing": True})
            profile = self._profile(tmp)
            accounted = [
                item
                for item in list(profile["router_constraint_applied"]) + list(profile["router_constraint_unmet"])
                if item["rule"] == "forbid_crossing"
            ]
            self.assertTrue(accounted, "the rule was neither honoured nor reported")
            for item in profile["router_constraint_applied"]:
                if item["rule"] != "forbid_crossing":
                    continue
                arrow = next(entry for entry in result["arrows"] if entry["id"] == item["arrow_id"])
                self.assertGreaterEqual(len(arrow["path_percent"]), 2)

    def test_the_same_rule_changes_a_figure_that_shares_none_of_its_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = _router_figure("a")
            second = _router_figure("b", nodes=3, arrows=[{
                "id": "conn_b_1",
                "source_id": "node_b_0",
                "target_id": "node_b_2",
                "path_percent": [[0.105, 0.285], [0.22, 0.285], [0.22, 0.47], [0.34, 0.47], [0.34, 0.66]],
            }])
            for program in (first, second):
                result = style_and_route_arrows(program, tmp, mode="reference", constraints={"max_segments": 1})
                arrow = result["arrows"][0]
                self.assertEqual(1, len(arrow["path_percent"]) - 1)
                self.assertEqual(ROUTER_CONSTRAINT_VERSION, arrow["router_constraint_version"])


class MechanismRuleEvidenceTests(unittest.TestCase):
    """The round's record that a rule reached the compile, and what the compile did with it.

    A mechanism rule moves nothing where it is executed, so without this file "this figure was
    drawn by the rule" is not readable from the run at all -- which is how both actions stayed
    invisible while every part of them was already written.
    """

    def _compiled_pair(self, rule: dict) -> tuple[dict, dict, dict]:
        """One figure compiled without the rule and with it, plus the compile's own report."""

        program = _layout_figure("a")
        without, _ = apply_layout_constraints(json.loads(json.dumps(program)), {})
        with_rule, report = apply_layout_constraints(json.loads(json.dumps(program)), rule)
        return without, with_rule, report

    def test_the_report_reads_the_compiles_record_and_the_figure_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp, "round")
            candidate = Path(tmp, "candidate")
            round_dir.mkdir()
            candidate.mkdir()
            without, with_rule, compile_report = self._compiled_pair({"align_grid": 0.05})
            write_json(candidate / "layout_constraint_report.json", compile_report)
            moved = diff_figure_programs(without, with_rule)
            # The rule is not a statement about this figure, but it is drawn onto it: the boxes
            # the compiler snapped really are somewhere else.
            self.assertTrue(moved["changed"])
            record = write_mechanism_rule_report(
                round_dir,
                round_index=3,
                action={"type": "layout_constraint", "params": {"align_grid": 0.05}, "source": "reflection_rule"},
                candidate_dir=candidate,
                figure_diff=moved,
                round_cost={"figure_generations": 0, "stage_reruns": 0},
            )
            self.assertTrue(record["took_effect"])
            self.assertEqual("applied", str(compile_report["status"]))
            self.assertTrue(record["compile_applied"])
            self.assertTrue(record["changed_object_ids"])
            # A round that ran a rule spent no generation call and no stage re-run.
            self.assertEqual(0, record["generation_calls"])
            self.assertEqual(0, record["stage_reruns"])
            self.assertEqual(
                record,
                json.loads((round_dir / "mechanism_rule_report.json").read_text(encoding="utf-8")),
            )

    def test_a_rule_the_compiler_applied_to_an_identical_figure_did_not_take_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp, "round")
            candidate = Path(tmp, "candidate")
            round_dir.mkdir()
            candidate.mkdir()
            write_json(candidate / "layout_constraint_report.json", {"status": "applied", "applied": [{"rule": "align_grid", "grid": 0.05, "box_count": 2}], "unmet": []})
            record = write_mechanism_rule_report(
                round_dir,
                round_index=4,
                action={"type": "layout_constraint", "params": {"align_grid": 0.05}},
                candidate_dir=candidate,
                figure_diff={"changed": False, "changed_object_ids": []},
                round_cost={"figure_generations": 0, "stage_reruns": 0},
            )
            # Both halves are recorded, because either alone reads as success.
            self.assertTrue(record["compile_applied"])
            self.assertFalse(record["figure_changed"])
            self.assertFalse(record["took_effect"])

    def test_a_router_rule_is_read_from_its_own_compile_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp, "round")
            round_dir.mkdir()
            program = _router_figure("a")
            before = json.loads(json.dumps(program))
            routed = style_and_route_arrows(program, tmp, mode="reference", constraints={"max_segments": 1})
            record = write_mechanism_rule_report(
                round_dir,
                round_index=5,
                action={"type": "router_constraint", "constraints": {"max_segments": 1}},
                candidate_dir=tmp,
                figure_diff=diff_figure_programs(before, routed),
                round_cost={"figure_generations": 0, "stage_reruns": 0},
            )
            self.assertEqual({"max_segments": 1}, record["rule"])
            self.assertTrue(record["compile_report_found"])
            self.assertTrue(any(item["rule"] == "max_segments" for item in record["compile_applied"]))
            self.assertTrue(record["changed_object_ids"])
            self.assertTrue(record["took_effect"])


class LayoutConstraintTests(unittest.TestCase):
    def test_a_grid_rule_snaps_boxes_that_were_not_on_it(self):
        program = _layout_figure("a")
        before = dict(program["semantic_nodes"][0]["bbox_percent"])
        updated, report = apply_layout_constraints(program, {"align_grid": 0.05})
        box = updated["semantic_nodes"][0]["bbox_percent"]
        for key in ("x", "y", "w", "h"):
            self.assertTrue(_snapped(box[key], 0.05), (key, box[key]))
        self.assertNotEqual(before, box)
        self.assertEqual("applied", report["status"])
        self.assertEqual({"align_grid": 0.05}, report["rules"])

    def test_a_gutter_rule_pushes_overlapping_neighbours_apart(self):
        members = [
            {"id": "node_a_a", "panel_id": "panel_a", "bbox_percent": {"x": 0.10, "y": 0.30, "w": 0.10, "h": 0.10}},
            {"id": "node_a_b", "panel_id": "panel_a", "bbox_percent": {"x": 0.18, "y": 0.30, "w": 0.10, "h": 0.10}},
        ]
        updated, report = apply_layout_constraints(_layout_figure("a", members=members), {"gutter_min": 0.02})
        left, right = (item["bbox_percent"] for item in updated["semantic_nodes"])
        gap = float(right["x"]) - (float(left["x"]) + float(left["w"]))
        self.assertGreaterEqual(round(gap, 4), 0.0199)
        self.assertTrue(any(item["rule"] == "gutter_min" for item in report["applied"]))

    def test_a_rule_it_cannot_reach_is_reported_instead_of_claimed(self):
        members = [
            {"id": "node_a_a", "panel_id": "panel_a", "bbox_percent": {"x": 0.10, "y": 0.30, "w": 0.04, "h": 0.04}},
            {"id": "node_a_b", "panel_id": "panel_a", "bbox_percent": {"x": 0.30, "y": 0.60, "w": 0.04, "h": 0.04}},
        ]
        updated, report = apply_layout_constraints(_layout_figure("a", members=members), {"panel_occupancy_min": 0.6})
        record = report["applied"][0]
        self.assertEqual("panel_occupancy_min", record["rule"])
        self.assertTrue(record["capped"])
        self.assertTrue(report["unmet"], "a rule that fell short has to say so")
        self.assertLess(record["after"], record["target"])

    def test_growth_stops_where_the_panel_stops_holding_the_boxes_apart(self):
        """A panel too small for the full growth keeps the largest growth it holds apart.

        The two boxes below start far enough apart at the size the model drew them, and the
        growth the rule asks for puts one on top of the other once the panel edge stops the
        separation. The pass has to give that growth back rather than hand back a figure with
        two boxes on top of each other.
        """

        members = [
            {"id": "node_a_left", "panel_id": "panel_a", "bbox_percent": {"x": 0.05, "y": 0.10, "w": 0.09, "h": 0.09}},
            {"id": "node_a_right", "panel_id": "panel_a", "bbox_percent": {"x": 0.16, "y": 0.10, "w": 0.09, "h": 0.09}},
        ]
        figure = _layout_figure("a", members=members)
        figure["panels"][0]["bbox_percent"] = {"x": 0.0, "y": 0.0, "w": 0.30, "h": 0.30}

        updated, report = apply_layout_constraints(figure, {"panel_occupancy_min": 0.45})

        record = report["applied"][0]
        self.assertTrue(record["backed_off_for_overlap"], "growth that crowds a panel has to be given back")
        self.assertLess(record["scale"], 1.6)
        self.assertEqual(0, record["crowded_pairs"])
        left, right = (item["bbox_percent"] for item in updated["semantic_nodes"])
        gap = float(right["x"]) - (float(left["x"]) + float(left["w"]))
        self.assertGreaterEqual(round(gap, 4), 0.0119, "the grown boxes were left on top of each other")
        self.assertGreater(record["after"], record["before"])

    def test_the_same_rule_changes_a_figure_that_shares_none_of_its_ids(self):
        rule = {"align_grid": 0.05}
        for tag in ("a", "b"):
            updated, report = apply_layout_constraints(_layout_figure(tag), rule)
            for item in updated["semantic_nodes"]:
                self.assertTrue(_snapped(item["bbox_percent"]["x"], 0.05))
            self.assertEqual("applied", report["status"])

    def test_the_hand_written_standard_is_a_rule_the_compile_can_apply(self):
        self.assertEqual({"panel_occupancy_min", "gutter_min"}, set(CURATED_LAYOUT_RULES))
        self.assertGreaterEqual(CURATED_LAYOUT_RULES["panel_occupancy_min"], 0.45)

        updated, report = apply_layout_constraints(_layout_figure("a"), CURATED_LAYOUT_RULES)

        self.assertEqual("applied", report["status"])
        occupancy = [item for item in report["applied"] if item["rule"] == "panel_occupancy_min"]
        self.assertTrue(occupancy, "the standard's own occupancy rule was not applied")
        self.assertGreater(occupancy[0]["after"], occupancy[0]["before"])
        self.assertTrue(any(item["rule"] == "gutter_min" for item in report["applied"]))

    def test_the_solver_honours_the_rules_it_is_handed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, report = solve_layout_constraints(_layout_figure("a"), tmp, reroute=False, params={"align_grid": 0.05})
            self.assertEqual({"align_grid": 0.05}, report["layout_constraints"])
            self.assertEqual("applied", report["layout_constraint_report"]["status"])


class MechanismRuleWiringTests(unittest.TestCase):
    def test_a_seeded_rule_reaches_the_figure_the_run_compiles(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paper = root / "paper.pdf"
            document = fitz.open()
            page = document.new_page(width=600, height=800)
            page.insert_text((50, 60), "A Rule-Carrying Study", fontsize=14)
            page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
            page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
            page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
            document.save(paper)
            document.close()
            seed = {
                "experience_id": "direct-exp-router",
                "apply_at": "ppt_compilation",
                "action": {"type": "router_constraint", "scope_type": "connector", "constraints": {"max_segments": 1}},
            }
            out = root / "run"
            with patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None)):
                run_paper_evolve_editable(
                    paper=paper,
                    out=out,
                    planner_mode="heuristic",
                    method_mode="heuristic",
                    structure_mode="heuristic",
                    text_plan_mode="heuristic",
                    ppt_plan_mode="heuristic",
                    color_theme_mode="fallback",
                    icon_mode="placeholder",
                    visual_review_mode="heuristic",
                    judge_mode="heuristic",
                    max_rounds=2,
                    candidates_per_round=1,
                    seed_experience=seed,
                )
            profiles = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(out.rglob("arrow_style_profile.json"))]
            self.assertTrue(profiles, "the run drew no figure at all")
            carried = [profile for profile in profiles if profile["router_constraint_version"]]
            self.assertTrue(carried, "the seeded rule never reached a compile")
            self.assertEqual({"max_segments": 1}, carried[0]["router_constraints"])


    def test_a_run_with_no_library_still_applies_the_standard(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paper = root / "paper.pdf"
            document = fitz.open()
            page = document.new_page(width=600, height=800)
            page.insert_text((50, 60), "A Standard-Carrying Study", fontsize=14)
            page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
            page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
            page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
            document.save(paper)
            document.close()
            out = root / "run"
            with patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None)):
                run_paper_evolve_editable(
                    paper=paper,
                    out=out,
                    planner_mode="heuristic",
                    method_mode="heuristic",
                    structure_mode="heuristic",
                    text_plan_mode="heuristic",
                    ppt_plan_mode="heuristic",
                    color_theme_mode="fallback",
                    icon_mode="placeholder",
                    visual_review_mode="heuristic",
                    judge_mode="heuristic",
                    max_rounds=1,
                    candidates_per_round=1,
                )
            reports = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(out.rglob("layout_constraint_report.json"))
            ]

        self.assertTrue(reports, "the run wrote no layout constraint report at all")
        self.assertEqual("curated_standard", reports[0]["source"])
        self.assertIn("panel_occupancy_min", reports[0]["rules"])


if __name__ == "__main__":
    unittest.main()