"""A round runs the mechanism's move and the coach's suggestion, and measures both.

A round used to run exactly one move: the mechanism's own policy and the coach's suggestion
competed for the same slot, so whichever the order put first ate the round and the other was
never measured. These tests hold the round to a list, to the three ways a second move is
refused, and to the rule that a deferred move comes back first rather than being lost.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.evolution.direct import round_proposal_order, select_next_round_proposal
from rfs.workflows.round_moves import (
    guard_reason,
    is_mechanism_move,
    move_conflict_reason,
    move_source,
    move_unexecutable_reason,
    plan_round_moves,
    round_moves_record,
)
from rfs.workflows.run_cost import move_cost_record, round_cost_record
from rfs.workflows.paper_evolve_editable import (
    _opening_repair,
    _round_runs_a_move,
    run_paper_evolve_editable,
)

MECHANISM = frozenset({"decomposition_policy", "color_feedback"})

PARAGRAPH = "We propose a frozen encoder that turns a paper into a symbolic figure plan."


def _geometry(action_type: str, factor: float, *, source: str = "", dx: float = 0.0) -> dict:
    spec = {
        "type": action_type,
        "scope_type": "global",
        "factor": factor,
        "dx": dx,
        "dy": 0.0,
        "problem": "measured occupancy is low",
        "expected_effect": "separate the boxes that are drawn on top of one another",
    }
    if source:
        spec["source"] = source
    return spec


class PlanRoundMovesTests(unittest.TestCase):
    def test_a_mechanism_move_is_followed_by_the_suggestion_it_displaced(self):
        mechanism = _geometry("relayout_global", 1.02, source="decomposition_policy")
        model = _geometry("relayout_global", 1.05)
        moves = plan_round_moves([mechanism, model], mechanism, mechanism_sources=MECHANISM)

        self.assertEqual(2, len(moves))
        self.assertTrue(is_mechanism_move(moves[0], mechanism_sources=MECHANISM))
        self.assertFalse(is_mechanism_move(moves[1], mechanism_sources=MECHANISM))

    def test_a_round_whose_action_came_from_the_coach_runs_both_moves(self):
        model = _geometry("resize_nodes", 1.05)
        mechanism = _geometry("relayout_global", 1.02, source="decomposition_policy")
        moves = plan_round_moves([model, mechanism], model, mechanism_sources=MECHANISM)

        # Both halves were proposed, so both halves run: the mechanism's cheap move first and the
        # seat's suggestion on top of it. Which of the two the selector preferred is not what
        # decides this -- the pairing "this suggestion -> how it turned out" is the datum the loop
        # exists to collect, and a round that runs only the preferred half never collects it.
        self.assertEqual(2, len(moves))
        self.assertTrue(is_mechanism_move(moves[0], mechanism_sources=MECHANISM))
        self.assertEqual("relayout_global", moves[0]["type"])
        self.assertFalse(is_mechanism_move(moves[1], mechanism_sources=MECHANISM))
        self.assertEqual("resize_nodes", moves[1]["type"])

    def test_a_round_proposing_only_the_seat_move_runs_one_move(self):
        model = _geometry("resize_nodes", 1.05)
        moves = plan_round_moves([model], model, mechanism_sources=MECHANISM)

        # Nothing is invented: a half the caller never proposed is not run.
        self.assertEqual(1, len(moves))
        self.assertEqual("resize_nodes", moves[0]["type"])

    def test_a_round_proposing_only_the_mechanism_move_runs_one_move(self):
        mechanism = _geometry("relayout_global", 1.02, source="decomposition_policy")
        moves = plan_round_moves([mechanism], mechanism, mechanism_sources=MECHANISM)

        self.assertEqual(1, len(moves))
        self.assertTrue(is_mechanism_move(moves[0], mechanism_sources=MECHANISM))

    def test_the_same_move_is_not_run_twice(self):
        model = _geometry("relayout_global", 1.05)
        moves = plan_round_moves([dict(model), model], model, mechanism_sources=MECHANISM)
        self.assertEqual(1, len(moves))

    def test_the_source_of_a_move_is_read_off_the_action(self):
        self.assertEqual("decomposition_policy", move_source(_geometry("relayout_global", 1.02, source="decomposition_policy"), mechanism_sources=MECHANISM))
        self.assertEqual("model", move_source(_geometry("relayout_global", 1.02), mechanism_sources=MECHANISM))
        self.assertEqual("", move_source({}, mechanism_sources=MECHANISM))


class OpeningRepairTests(unittest.TestCase):
    """Round 0 of an inherited iteration is a move, not a redraw.

    The change the review of the inherited figure asked for is executed against that figure. The
    two ways it must not be: a run that planned its own design or icons, whose state is a different
    figure, and a stage re-run, which plans those stages anyway.
    """

    def _geometry_with_source(self) -> dict:
        return _geometry("relayout_global", 1.05)

    def test_a_bounded_repair_opens_an_inherited_iteration(self):
        opening = _opening_repair(self._geometry_with_source(), {"design_seed": {"panels": []}}, {"replanned_stages": ["method"]})

        self.assertEqual(opening["type"], "relayout_global")
        self.assertEqual(1.05, opening["factor"])
        self.assertEqual("iteration_review", opening["source"], "the record has to say who proposed the move")

    def test_a_run_that_planned_its_own_figure_keeps_its_initial_first_round(self):
        for replanned in (["design_seed"], ["icons"], ["design_seed", "icons"]):
            with self.subTest(replanned=replanned):
                self.assertEqual(
                    {},
                    _opening_repair(self._geometry_with_source(), {"design_seed": {}}, {"replanned_stages": replanned}),
                )

    def test_nothing_was_inherited_so_nothing_opens_the_run(self):
        self.assertEqual({}, _opening_repair(self._geometry_with_source(), None, {}))
        self.assertEqual({}, _opening_repair(self._geometry_with_source(), {}, {}))

    def test_a_stage_rerun_a_bare_name_and_the_initial_marker_are_not_opening_repairs(self):
        self.assertEqual({}, _opening_repair({"type": "replan_structure"}, {"design_seed": {}}, {}))
        self.assertEqual({}, _opening_repair({"type": "resize_nodes"}, {"design_seed": {}}, {}))
        self.assertEqual({}, _opening_repair({"type": "initial", "scope_type": "global"}, {"design_seed": {}}, {}))
        self.assertEqual({}, _opening_repair(None, {"design_seed": {}}, {}))

    def test_only_the_initial_marker_is_a_round_that_runs_no_move(self):
        self.assertFalse(_round_runs_a_move(0, {"type": "initial"}))
        self.assertFalse(_round_runs_a_move(0, {}))
        self.assertTrue(_round_runs_a_move(0, _geometry("relayout_global", 1.05)))
        self.assertTrue(_round_runs_a_move(1, {"type": "initial"}))


class MoveConflictTests(unittest.TestCase):
    def test_the_same_action_under_the_same_budget_is_one_move(self):
        action = _geometry("relayout_global", 1.05)
        self.assertEqual("duplicate_move", move_conflict_reason(action, dict(action)))

    def test_a_plan_rewrite_wins_over_the_move_that_was_chosen_against_the_old_plan(self):
        rewrite = {"type": "resegment", "scope_type": "global"}
        edit = _geometry("resize_nodes", 1.05)
        self.assertEqual("plan_rewritten", move_conflict_reason(rewrite, edit))

    def test_two_ordinary_moves_are_not_in_conflict(self):
        self.assertEqual("", move_conflict_reason(_geometry("relayout_global", 1.02), _geometry("relayout_global", 1.05)))

    def test_a_move_whose_target_an_earlier_move_deleted_is_unexecutable(self):
        action = {"type": "resize_nodes", "scope_type": "slot", "object_ids": ["gone"], "factor": 1.05, "dx": 0.0, "dy": 0.0}
        state = {"layout": {"nodes": [{"id": "kept"}], "panels": [], "connectors": []}, "icons": {"icons": []}}
        self.assertEqual("target_missing_after_earlier_move", move_unexecutable_reason(action, state))

    def test_a_move_whose_target_survived_is_executable(self):
        action = {"type": "resize_nodes", "scope_type": "slot", "object_ids": ["kept"], "factor": 1.05, "dx": 0.0, "dy": 0.0}
        state = {"layout": {"nodes": [{"id": "kept"}], "panels": [], "connectors": []}, "icons": {"icons": []}}
        self.assertEqual("", move_unexecutable_reason(action, state))


class MoveGuardTests(unittest.TestCase):
    def test_a_stop_file_left_mid_round_defers_the_second_move(self):
        with tempfile.TemporaryDirectory() as name:
            stop = Path(name) / "STOP"
            self.assertEqual("", guard_reason(stop_file=stop))
            stop.write_text("stop", encoding="utf-8")
            self.assertEqual("stop_requested_between_moves", guard_reason(stop_file=stop))

    def test_a_disk_under_the_floor_defers_the_second_move(self):
        with tempfile.TemporaryDirectory() as name:
            self.assertEqual("disk_floor_between_moves", guard_reason(disk_path=name, disk_floor_gb=10 ** 9))

    def test_a_round_with_no_guard_configured_never_defers(self):
        self.assertEqual("", guard_reason())

    def test_a_volume_under_its_share_floor_defers_the_second_move(self):
        with tempfile.TemporaryDirectory() as name:
            self.assertEqual(
                "disk_floor_between_moves",
                guard_reason(disk_path=name, disk_floor_ratio=0.999999),
            )
            self.assertEqual("", guard_reason(disk_path=name, disk_floor_ratio=1e-9))

    def test_an_absolute_floor_wins_over_the_share_the_run_did_not_mean(self):
        # The loop reads a positive disk_floor_gb *instead of* the share, so a round must
        # not refuse on a share the run already overrode: 40 GB was the instruction, not ten
        # percent of whatever volume happens to hold it.
        with tempfile.TemporaryDirectory() as name:
            self.assertEqual(
                "",
                guard_reason(disk_path=name, disk_floor_gb=1e-9, disk_floor_ratio=0.999999),
            )

    def test_a_spent_budget_defers_the_second_move(self):
        self.assertEqual(
            "budget_spent_between_moves",
            guard_reason(budget_deadline=100.0, clock=lambda: 100.0),
        )
        self.assertEqual("", guard_reason(budget_deadline=100.0, clock=lambda: 99.0))
        self.assertEqual("", guard_reason(budget_deadline=0.0, clock=lambda: 10 ** 9))

    def test_the_stop_file_is_still_the_first_thing_asked(self):
        with tempfile.TemporaryDirectory() as name:
            stop = Path(name) / "STOP"
            stop.write_text("stop", encoding="utf-8")
            self.assertEqual(
                "stop_requested_between_moves",
                guard_reason(
                    stop_file=stop,
                    disk_path=name,
                    disk_floor_gb=1e-9,
                    budget_deadline=100.0,
                    clock=lambda: 100.0,
                ),
            )


class MoveRecordTests(unittest.TestCase):
    def test_the_round_record_names_both_moves(self):
        moves = [
            {"index": 0, "type": "relayout_global", "source": "decomposition_policy", "is_mechanism": True},
            {"index": 1, "type": "resize_nodes", "source": "model", "is_mechanism": False},
        ]
        record = round_moves_record(2, moves, source="planned")
        self.assertEqual(2, record["count"])
        self.assertEqual("decomposition_policy", record["mechanism_move"])
        self.assertEqual(["0", "1"], record["order"])

    def test_each_move_is_billed_on_its_own_line(self):
        line = move_cost_record(index=0, action={"type": "relayout_global"}, action_class="compile", source="decomposition_policy", compile_calls=1, render_calls=1, judge_calls=2, measured=True)
        self.assertEqual("rfs.move_cost", line["contract"])
        self.assertTrue(line["measured"])
        round_record = round_cost_record(round_index=3, action={"type": "relayout_global"}, moves=[line])
        self.assertEqual(1, round_record["move_count"])


class ProposalOrderTests(unittest.TestCase):
    def test_the_ordered_proposals_keep_the_loops_preference_and_drop_duplicates(self):
        mechanism = _geometry("relayout_global", 1.02, source="decomposition_policy")
        model = _geometry("resize_nodes", 1.05)
        order = round_proposal_order(
            revised_action=model,
            decomposition_action={"required": True, "action": "relayout_global"},
            decomposition_proposal=mechanism,
        )
        self.assertEqual("resize_nodes", order[0]["type"])
        self.assertIn("relayout_global", [item["type"] for item in order])
        self.assertEqual(order[0], select_next_round_proposal(
            revised_action=model,
            decomposition_action={"required": True, "action": "relayout_global"},
            decomposition_proposal=mechanism,
        ))


PARAGRAPH_TEXT = PARAGRAPH


def _write_paper(root: Path) -> Path:
    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Move Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH_TEXT, fontsize=10)
    page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(target)
    document.close()
    return target


def _choice_answer(record: dict) -> str:
    return json.dumps({"wrong_figure": "A", "defect": "overlap"})


class TwoMoveRoundTests(unittest.TestCase):
    def _run(self, root: Path, out: Path, *, rounds: int, guard=None) -> None:
        mechanism = _geometry("relayout_global", 1.02, source="decomposition_policy")
        model = _geometry("relayout_global", 1.05, dx=0.02)

        def _order(**kwargs):
            return [dict(mechanism), dict(model)]

        def _next(retrieved, proposal, layout, icons):
            return dict(mechanism), {}, {}

        patches = [
            patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda e: None)),
            patch("rfs.workflows.paper_evolve_editable.round_proposal_order", _order),
            patch("rfs.workflows.paper_evolve_editable._next_action_with_memory", _next),
        ]
        if guard is not None:
            patches.append(patch("rfs.workflows.paper_evolve_editable.guard_reason", guard))
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        run_paper_evolve_editable(
            paper=str(root / "paper.pdf"),
            out=str(out),
            title="Moves",
            planner_mode="heuristic",
            method_mode="heuristic",
            structure_mode="heuristic",
            text_plan_mode="heuristic",
            ppt_plan_mode="heuristic",
            color_theme_mode="fallback",
            icon_mode="placeholder",
            visual_review_mode="heuristic",
            judge_mode="heuristic",
            max_rounds=rounds,
            candidates_per_round=2,
            curated_standard="",
            acceptance_generate=_choice_answer,
        )

    def test_a_round_runs_both_moves_and_each_carries_its_own_score(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _write_paper(root)
            out = root / "both"
            self._run(root, out, rounds=2)
            moves = json.loads((out / "round_01" / "round_moves.json").read_text(encoding="utf-8"))
            cost = json.loads((out / "round_01" / "round_cost.json").read_text(encoding="utf-8"))

        self.assertEqual(2, moves["count"])
        self.assertEqual(["decomposition_policy", "model"], [item["source"] for item in moves["moves"]])
        scores = [item["measurement"]["score"] for item in moves["moves"]]
        self.assertTrue(all(score is not None for score in scores), moves["moves"])
        self.assertEqual(2, cost["move_count"])
        self.assertEqual(2, len(cost["moves"]))
        # The round the run kept holds both edits, so its execution record names both moves.
        self.assertEqual(2, cost["moves"][-1]["compile_calls"])

    def test_a_guarded_round_defers_the_second_move_into_the_next_rounds_first_slot(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _write_paper(root)
            out = root / "deferred"
            self._run(root, out, rounds=3, guard=lambda **kwargs: "stop_requested_between_moves")
            first = json.loads((out / "round_01" / "round_moves.json").read_text(encoding="utf-8"))
            second = json.loads((out / "round_02" / "round_moves.json").read_text(encoding="utf-8"))
            events = [json.loads(line).get("event") for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(1, first["count"], "the second move is deferred rather than run")
        self.assertIn("move_deferred", events)
        self.assertEqual(1, second["count"])
        # The move that was deferred runs first, so a suggestion cannot be starved twice.
        self.assertEqual("model", second["moves"][0]["source"])


if __name__ == "__main__":
    unittest.main()
