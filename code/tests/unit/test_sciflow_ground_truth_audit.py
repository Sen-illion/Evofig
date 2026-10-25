"""Pin the arithmetic behind the SciFlow-Bench ground-truth audit.

The audit itself needs the SciFlow-Bench checkout, its dataset and an embedding
model, none of which live in this repository. These tests cover the parts that
decide what gets compared and how the rows are aggregated, so a change to the
split logic cannot silently move the published error rate.
"""

import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_sciflow_ground_truth",
        ROOT
        / "benchmarks"
        / "editable-figure-bench"
        / "review"
        / "audit_sciflow_ground_truth.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_module()


def _row(domain="cv", human_nodes=10, machine_nodes=12, human_edges=10, machine_edges=14, edge_r=0.8):
    return {
        "paper": "x",
        "domain": domain,
        "machine_file": "canvas_full.json",
        "human_nodes": human_nodes,
        "machine_nodes": machine_nodes,
        "human_edges": human_edges,
        "machine_edges": machine_edges,
        "node_p": 0.9,
        "node_r": 1.0,
        "node_f1": 0.9474,
        "edge_p": 0.7,
        "edge_r": edge_r,
        "edge_f1": 0.75,
        "stored_node_f1": None,
        "stored_edge_f1": None,
    }


class MachineGraphChoiceTests(unittest.TestCase):
    def test_the_first_existing_machine_graph_wins_in_the_declared_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper = pathlib.Path(tmp)
            (paper / "canvas_full_step10.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                pathlib.Path(audit._pick_machine_graph(str(paper))).name,
                "canvas_full_step10.json",
            )
            (paper / "canvas_full_merge.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                pathlib.Path(audit._pick_machine_graph(str(paper))).name,
                "canvas_full_merge.json",
            )
            (paper / "canvas_full.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                pathlib.Path(audit._pick_machine_graph(str(paper))).name,
                "canvas_full.json",
            )

    def test_a_paper_without_any_machine_graph_is_skipped_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(audit._pick_machine_graph(tmp))


class DomainTests(unittest.TestCase):
    def test_the_domain_comes_from_the_dataset_directory_name(self):
        path = "/data/nlp_2025_autobench/2505.00980v1/canvas_human.json"
        self.assertEqual(audit._domain_of(path), "nlp")

    def test_an_unrecognised_path_is_reported_as_unknown(self):
        self.assertEqual(audit._domain_of("/elsewhere/paper/canvas_human.json"), "unknown")


class StatisticsTests(unittest.TestCase):
    def test_percentiles_interpolate_between_neighbours(self):
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(audit._percentile(values, 0.0), 0.0)
        self.assertAlmostEqual(audit._percentile(values, 50.0), 2.0)
        self.assertAlmostEqual(audit._percentile(values, 100.0), 4.0)

    def test_a_single_value_has_no_spread(self):
        self.assertAlmostEqual(audit._percentile([0.5], 10.0), 0.5)

    def test_no_values_yields_no_statistics(self):
        self.assertEqual(audit._stats([]), {})


class SummaryTests(unittest.TestCase):
    def test_over_generation_is_counted_per_paper_and_in_net_units(self):
        rows = [_row(machine_nodes=12), _row(machine_nodes=10), _row(machine_nodes=9, machine_edges=8)]
        summary = audit._summarize(rows)
        self.assertEqual(summary["papers_scored"], 3)
        self.assertEqual(summary["machine_over_generation"]["papers_with_extra_nodes"], 1)
        self.assertEqual(summary["machine_over_generation"]["net_extra_nodes"], 1)
        self.assertEqual(summary["machine_over_generation"]["papers_missing_a_human_edge"], 3)

    def test_each_domain_is_reported_separately(self):
        summary = audit._summarize([_row(domain="cv"), _row(domain="ml"), _row(domain="ml")])
        self.assertEqual(sorted(summary["by_domain"]), ["cv", "ml"])
        self.assertEqual(summary["by_domain"]["ml"]["n"], 2)


class StoredMetricsTests(unittest.TestCase):
    def test_the_shipped_metrics_field_is_reported_when_it_disagrees(self):
        rows = [_row(), _row()]
        rows[1]["stored_node_f1"] = 0.1
        rows[1]["stored_edge_f1"] = 0.1
        agreement = audit._agreement_with_stored_metrics(rows)
        self.assertEqual(agreement["with_stored"], 1)
        self.assertEqual(agreement["matching_within_tolerance"], 0)

    def test_a_row_without_a_stored_score_is_not_counted(self):
        self.assertEqual(
            audit._agreement_with_stored_metrics([_row()]),
            {"with_stored": 0, "matching_within_tolerance": 0},
        )


if __name__ == "__main__":
    unittest.main()
