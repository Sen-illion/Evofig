"""Unit tests for scripts/summarize_evolution_metrics.py.

The script's whole job is to read numbers that are already on disk, so these tests build the two
files it reads -- a run state and the trend ledger -- and assert the two numbers the direction
change is judged on: what each layer cost, and whether the paper's versions were beating its own
recent best. A layer that was settled but never moved, and a round nobody compared, are both here
because they are the two ways this report could quietly read as progress.
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_summary_module():
    spec = importlib.util.spec_from_file_location("summarize_evolution_metrics", ROOT / "scripts" / "summarize_evolution_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = load_summary_module()


def write_run(root: pathlib.Path, *, papers: dict) -> pathlib.Path:
    (root / "papers").mkdir(parents=True, exist_ok=True)
    (root / "continuous_state.json").write_text(json.dumps({"waves": 4, "papers": papers}), encoding="utf-8")
    return root


class LayerWalkSummaryTests(unittest.TestCase):
    def test_each_layer_reports_what_it_cost_and_the_total_appears_once_all_settle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_run(
                pathlib.Path(temp),
                papers={
                    "paper_one": {
                        "iteration": 5,
                        "work_face": "ppt_compilation",
                        "upstream_requests": 3,
                        "upstream_passed": 1,
                        "last_claims": {"rate": 0.5, "settled": 2, "open": 2},
                        "layers": {
                            "paper_reading": {"first_move_at": 1, "settled_at": 1, "settled_by": "model"},
                            "content_planning": {"first_move_at": 2, "settled_at": 3, "settled_by": "human", "reopen_count": 1},
                            "global_planning": {"first_move_at": 4, "settled_at": 5, "settled_by": "model"},
                        },
                    }
                },
            )
            report = summary.summarize(root)
            paper = report["papers"][0]
            walk = paper["layer_walk"]
            self.assertEqual(1, walk["layers"]["paper_reading"]["rounds_to_settle"])
            self.assertEqual(2, walk["layers"]["content_planning"]["rounds_to_settle"])
            self.assertEqual(3, paper["upstream"]["requested"])
            self.assertEqual(1, paper["upstream"]["passed"])
            self.assertEqual(1, walk["reopen_events"])
            # Two layers are still open, so there is no "everything settled" round to report.
            self.assertFalse(walk["all_settled"])
            self.assertIsNone(walk["rounds_to_settle_all"])
            self.assertEqual(0.5, paper["claims"]["rate"])

    def test_a_layer_nobody_moved_has_no_cost_to_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_run(
                pathlib.Path(temp),
                papers={
                    "paper_one": {
                        "iteration": 2,
                        "layers": {"theme_planning": {"settled_at": 2, "settled_by": "model", "first_move_at": 0}},
                    }
                },
            )
            row = summary.summarize(root)["papers"][0]["layer_walk"]["layers"]["theme_planning"]
            self.assertEqual(2, row["settled_at"])
            self.assertIsNone(row["rounds_to_settle"], "a layer nobody moved did not take no rounds")

    def test_the_newest_iterations_ledger_is_the_one_that_is_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_run(
                pathlib.Path(temp),
                papers={"paper_one": {"iteration": 99, "layers": {"structure": {"settled_at": 0, "first_move_at": 0}}}},
            )
            for number, settled in ((99, 1), (100, 5)):
                iter_dir = root / "papers" / "paper_one" / f"iter_{number:02d}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                (iter_dir / "layer_ledger.json").write_text(
                    json.dumps({"contract": "rfs.layer_ledger", "iteration": number, "walk": {"settled": settled, "total": 8}}),
                    encoding="utf-8",
                )
            walk = summary.summarize(root)["papers"][0]["layer_walk"]
            # Sorting the directory names as text would pick iter_99, and the walk would read as
            # the older record for the rest of the run.
            self.assertEqual(5, walk["settled"])

    def test_a_layer_opened_after_its_last_answer_is_not_settled(self):
        stages = ["paper_reading", "content_planning", "global_planning", "segmentation"]
        with tempfile.TemporaryDirectory() as temp:
            root = write_run(
                pathlib.Path(temp),
                papers={
                    "paper_one": {
                        "iteration": 4,
                        "layer_settle_events": [{"stage": stage, "iteration": 1, "by": "model"} for stage in stages],
                        "layers": {stage: {"first_move_at": 1, "settled_at": 1} for stage in stages},
                    }
                },
            )
            iter_dir = root / "papers" / "paper_one" / "iter_04"
            iter_dir.mkdir(parents=True, exist_ok=True)
            (iter_dir / "layer_ledger.json").write_text(
                json.dumps({"iteration": 4, "reopen_events": [{"stage": "segmentation", "iteration": 3, "by": "model"}]}),
                encoding="utf-8",
            )
            walk = summary.summarize(root)["papers"][0]["layer_walk"]
            # Three of the four layers still stand; the one opened again after its answer does not,
            # however many times it was called finished before the reopen.
            self.assertEqual(0, walk["layers"]["segmentation"]["settled_at"])
            self.assertEqual(1, walk["layers"]["content_planning"]["settled_at"])


class TrendSummaryTests(unittest.TestCase):
    def test_the_ledger_is_read_as_wins_and_losses_rather_than_as_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_run(pathlib.Path(temp), papers={"paper_one": {"iteration": 3, "layers": {}}})
            ledger = [
                {"paper": "paper_one", "iteration": 1, "progress": {"cross_round": {"wins": 1, "losses": 0, "ties": 0, "undecided": 0}}},
                {"paper": "paper_one", "iteration": 2, "progress": {"cross_round": {"wins": 1, "losses": 0, "ties": 0, "undecided": 0}}},
                {"paper": "paper_one", "iteration": 3, "progress": {"cross_round": {"wins": 0, "losses": 1, "ties": 0, "undecided": 1}}},
                {"paper": "paper_two", "iteration": 1, "progress": {"cross_round": {"wins": 1, "losses": 0, "ties": 0, "undecided": 0}}},
            ]
            (root / "evolution_metrics.jsonl").write_text("\n".join(json.dumps(row) for row in ledger) + "\n", encoding="utf-8")

            report = summary.summarize(root)
            paper = report["papers"][0]["trend"]
            self.assertEqual({"wins": 2, "losses": 1}, {"wins": paper["wins"], "losses": paper["losses"]})
            self.assertEqual(1, paper["undecided"], "a round nobody compared is not a tie")
            self.assertEqual(2, paper["longest_win_streak"])
            # The whole run is counted too, so the three papers can be read side by side.
            self.assertEqual(3, report["trend"]["wins"])
            self.assertEqual(4, report["ledger_rows"])

    def test_every_iteration_metrics_file_is_counted_for_the_paper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_run(pathlib.Path(temp), papers={"paper_one": {"iteration": 2, "layers": {}}})
            for number in (1, 2):
                iter_dir = root / "papers" / "paper_one" / f"iter_{number:02d}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                (iter_dir / "evolution_metrics.json").write_text(
                    json.dumps({"progress": {"contract": "rfs.evolution_round_metrics", "iteration": number}}), encoding="utf-8"
                )
            self.assertEqual(2, summary.summarize(root)["papers"][0]["iterations_recorded"])

    def test_a_runs_own_metrics_line_is_not_counted_as_a_round(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_run(pathlib.Path(temp), papers={"paper_one": {"iteration": 1, "layers": {}}})
            ledger = [
                {"paper": "paper_one", "iteration": 1, "progress": {"cross_round": {"wins": 1, "losses": 0, "ties": 0, "undecided": 0}}},
                {"paper": "paper_one", "run": "iter_01", "headline": {"location_accuracy": 1.0}},
            ]
            (root / "evolution_metrics.jsonl").write_text("\n".join(json.dumps(row) for row in ledger) + "\n", encoding="utf-8")
            report = summary.summarize(root)
            self.assertEqual(1, report["papers"][0]["trend"]["rounds"])
            self.assertEqual(2, report["ledger_rows"], "both lines are still counted as ledger rows")

    def test_the_table_names_the_layers_and_the_trend(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_run(
                pathlib.Path(temp),
                papers={"paper_one": {"iteration": 1, "work_face": "segmentation", "layers": {"paper_reading": {"first_move_at": 1, "settled_at": 1, "settled_by": "model"}}}},
            )
            text = summary.markdown(summary.summarize(root))
            self.assertIn("paper_reading", text)
            self.assertIn("first moved at 1", text)
            self.assertIn("segmentation", text)


if __name__ == "__main__":
    unittest.main()
