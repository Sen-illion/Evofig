"""Tests for the deterministic crosscheck of two lines' figure specifications."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from rfs.cli import _dispatch, build_parser, main
from rfs.cli_commands.categories import MAINTENANCE_COMMANDS
from rfs.evaluation.contract_crosscheck import (
    compare_figure_specifications,
    crosscheck_contract_directories,
    crosscheck_contract_pairs,
    run_contract_crosscheck,
)
from rfs.utils import read_json, write_json


def specification(**overrides: object) -> dict:
    """One small figure specification that each test overrides in one place."""
    base = {
        "topology": "pipeline",
        "inputs": [{"id": "i1", "visible_label": "Raw paper", "must_show": True}],
        "modules": [
            {"id": "m1", "visible_label": "Encoder", "must_show": True},
            {"id": "m2", "visible_label": "Decoder", "must_show": False},
        ],
        "outputs": [{"id": "o1", "visible_label": "Figure plan", "must_show": True}],
        "innovations": [],
        "relations": [
            {"source": "i1", "target": "m1", "type": "feeds"},
            {"source": "m1", "target": "m2", "type": "feeds"},
            {"source": "m2", "target": "o1", "type": "emits"},
        ],
    }
    base.update(overrides)
    return base


def write_specification(root: Path, name: str, payload: dict, *, stored_name: str = "paper_method_inventory.json") -> Path:
    """Write one paper directory carrying the given specification."""
    directory = root / name
    write_json(directory / stored_name, payload)
    return directory


class CompareSpecificationsTests(unittest.TestCase):
    def test_identical_specifications_agree(self):
        report = compare_figure_specifications(specification(), specification())
        self.assertTrue(report["agree"])
        self.assertEqual(0, report["differences"])
        self.assertEqual(["Decoder", "Encoder"], sorted(report["entities"]["modules"]["shared"]))
        self.assertEqual(3, len(report["relations"]["shared"]))
        self.assertTrue(report["topology"]["agree"])

    def test_the_same_concept_under_different_ids_is_one_entity(self):
        right = specification(
            inputs=[{"instance_id": "in_a", "name": "Raw  Paper", "must_show": True}],
            modules=[
                {"instance_id": "module_a", "name": "encoder", "must_show": True},
                {"instance_id": "module_b", "name": "Decoder", "must_show": False},
            ],
            outputs=[{"instance_id": "out_a", "name": "figure plan", "must_show": True}],
            relations=[
                {"source_id": "in_a", "target_id": "module_a", "relation_type": "feeds"},
                {"source_id": "module_a", "target_id": "module_b", "relation_type": "feeds"},
                {"source_id": "module_b", "target_id": "out_a", "relation_type": "emits"},
            ],
        )
        report = compare_figure_specifications(specification(), right)
        self.assertTrue(report["agree"])
        self.assertEqual(0, report["differences"])
        self.assertEqual([], report["relations"]["only_left"])
        self.assertEqual([], report["relations"]["type_mismatches"])

    def test_a_relation_that_changed_its_type_is_a_type_mismatch(self):
        right = specification(
            relations=[
                {"source": "i1", "target": "m1", "type": "feeds"},
                {"source": "m1", "target": "m2", "type": "emits"},
                {"source": "m2", "target": "o1", "type": "emits"},
            ]
        )
        report = compare_figure_specifications(specification(), right)
        self.assertFalse(report["agree"])
        self.assertEqual(1, report["differences"])
        self.assertEqual(3, len(report["relations"]["shared"]))
        self.assertEqual(
            [{"relation": "Encoder -> Decoder", "left": "feeds", "right": "emits"}],
            report["relations"]["type_mismatches"],
        )

    def test_a_relation_only_one_side_planned_is_reported_per_endpoint_label(self):
        right = specification(
            relations=[{"source": "i1", "target": "m1", "type": "feeds"}]
        )
        report = compare_figure_specifications(specification(), right)
        self.assertFalse(report["agree"])
        self.assertEqual(
            ["Decoder -> Figure plan", "Encoder -> Decoder"],
            report["relations"]["only_left"],
        )
        self.assertEqual([], report["relations"]["only_right"])

    def test_the_same_concept_in_another_role_is_not_a_match(self):
        right = specification(
            modules=[{"id": "m1", "visible_label": "Encoder", "must_show": True}],
            outputs=[
                {"id": "o1", "visible_label": "Figure plan", "must_show": True},
                {"id": "o2", "visible_label": "Decoder", "must_show": False},
            ],
        )
        report = compare_figure_specifications(specification(), right)
        self.assertEqual(["Decoder"], report["entities"]["modules"]["only_left"])
        self.assertEqual(["Decoder"], report["entities"]["outputs"]["only_right"])
        # The relations move with the entity: the right side renamed it o2, so
        # both of its relations are reported as differences of their own.
        self.assertEqual(6, report["differences"])
        self.assertFalse(report["agree"])

    def test_a_relation_endpoint_a_side_never_defined_is_reported_as_its_id(self):
        right = specification(relations=[{"source": "i1", "target": "ghost", "type": "feeds"}])
        report = compare_figure_specifications(specification(), right)
        self.assertIn("Raw paper -> ghost", report["relations"]["only_right"])
        self.assertEqual([], report["relations"]["shared"])
        self.assertEqual(
            ["Decoder -> Figure plan", "Encoder -> Decoder", "Raw paper -> Encoder"],
            report["relations"]["only_left"],
        )

    def test_a_must_show_flag_that_changed_is_reported_without_counting_the_entity(self):
        right = specification(
            modules=[
                {"id": "m1", "visible_label": "Encoder", "must_show": True},
                {"id": "m2", "visible_label": "Decoder", "must_show": True},
            ]
        )
        report = compare_figure_specifications(specification(), right)
        self.assertEqual(1.0, report["entities"]["modules"]["agreement"])
        self.assertEqual(["Decoder"], report["entities"]["modules"]["must_show_mismatches"])
        self.assertEqual(1, report["differences"])
        self.assertFalse(report["agree"])

    def test_a_different_topology_alone_is_a_difference(self):
        report = compare_figure_specifications(specification(), specification(topology="hub"))
        self.assertFalse(report["topology"]["agree"])
        self.assertEqual(1, report["differences"])
        self.assertFalse(report["agree"])
        self.assertEqual("pipeline", report["left"]["topology"])
        self.assertEqual("hub", report["right"]["topology"])

    def test_a_label_shared_by_two_entries_counts_once(self):
        right = specification(
            modules=[
                {"id": "m1", "visible_label": "Encoder", "must_show": True},
                {"id": "m2", "visible_label": "Encoder", "must_show": True},
                {"id": "m3", "visible_label": "Decoder", "must_show": False},
            ]
        )
        report = compare_figure_specifications(specification(), right)
        self.assertEqual(2, report["right"]["entity_counts"]["modules"])
        self.assertEqual(["Decoder", "Encoder"], sorted(report["entities"]["modules"]["shared"]))

    def test_an_empty_specification_shares_nothing(self):
        report = compare_figure_specifications({}, specification())
        self.assertFalse(report["agree"])
        self.assertEqual(0.0, report["entities"]["modules"]["agreement"])
        self.assertEqual(["Decoder", "Encoder"], report["entities"]["modules"]["only_right"])
        self.assertEqual([], report["entities"]["modules"]["only_left"])
        self.assertEqual(0, report["left"]["relation_count"])


class CrosscheckDirectoriesTests(unittest.TestCase):
    def test_a_directory_without_a_specification_names_the_side(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = write_specification(root, "left", specification())
            right = root / "right"
            right.mkdir()
            with self.assertRaises(FileNotFoundError) as caught:
                crosscheck_contract_directories(left, right)
            self.assertIn("right", str(caught.exception))

    def test_a_directory_pair_reports_both_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = write_specification(root, "left", specification())
            right = write_specification(root, "right", specification())
            report = crosscheck_contract_directories(left, right)
            self.assertTrue(report["agree"])
            self.assertEqual(str((left / "paper_method_inventory.json").resolve()), report["left"]["path"])
            self.assertEqual(str((right / "paper_method_inventory.json").resolve()), report["right"]["path"])

    def test_a_directory_written_before_the_rename_is_still_read(self):
        # The middleware renamed the file; a pair that stored it under the earlier
        # name is read through that name rather than refused.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = write_specification(root, "left", specification(), stored_name="figure_specification.json")
            right = write_specification(root, "right", specification())
            report = crosscheck_contract_directories(left, right)

        self.assertTrue(report["agree"])
        self.assertEqual("figure_specification.json", Path(report["left"]["path"]).name)
        self.assertEqual("paper_method_inventory.json", Path(report["right"]["path"]).name)

    def test_pairs_summarise_how_far_the_two_lines_agree(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            same_left = write_specification(root, "same-left", specification())
            same_right = write_specification(root, "same-right", specification())
            other_left = write_specification(root, "other-left", specification())
            other_right = write_specification(root, "other-right", specification(topology="hub"))
            report = crosscheck_contract_pairs([(same_left, same_right), (other_left, other_right)])
            self.assertEqual(2, report["pair_count"])
            self.assertEqual(1, report["agreeing_pairs"])
            self.assertEqual(0.5, report["agreeing_ratio"])
            self.assertEqual(1, report["total_differences"])
            self.assertEqual(
                ["innovations", "inputs", "modules", "outputs"],
                sorted(report["pairs"][0]["entity_agreement"]),
            )

    def test_an_empty_pair_list_has_nothing_to_disagree_with(self):
        report = crosscheck_contract_pairs([])
        self.assertEqual(0, report["pair_count"])
        self.assertEqual(1.0, report["agreeing_ratio"])
        self.assertEqual(0, report["total_differences"])


class RunContractCrosscheckTests(unittest.TestCase):
    def test_the_two_sides_are_required(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = write_specification(Path(raw), "left", specification())
            with self.assertRaises(ValueError):
                run_contract_crosscheck([], [directory])
            with self.assertRaises(ValueError):
                run_contract_crosscheck([directory], [])

    def test_unbalanced_sides_are_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = write_specification(root, "left", specification())
            right = write_specification(root, "right", specification())
            with self.assertRaises(ValueError) as caught:
                run_contract_crosscheck([left, right], [right])
            self.assertIn("pairs directories by position", str(caught.exception))

    def test_a_difference_is_still_a_passing_run(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = write_specification(root, "left", specification())
            right = write_specification(root, "right", specification(topology="hub"))
            out = root / "report"
            report = run_contract_crosscheck([left], [right], out=out)
            self.assertTrue(report["ok"])
            self.assertEqual(1, report["total_differences"])
            self.assertEqual(str(out), report["out_dir"])
            written = read_json(out / "contract_crosscheck.json")
            self.assertEqual(1, written["total_differences"])
            self.assertEqual(0.0, written["agreeing_ratio"])


class ContractCrosscheckCommandTests(unittest.TestCase):
    def test_the_command_is_registered_as_maintenance(self):
        registered = set(build_parser()._subparsers._group_actions[0].choices)
        self.assertIn("contract-crosscheck", registered)
        self.assertIn("contract-crosscheck", MAINTENANCE_COMMANDS)

    def test_dispatch_returns_the_measurement(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = write_specification(root, "left", specification())
            right = write_specification(root, "right", specification())
            args = build_parser().parse_args(
                ["contract-crosscheck", "--left", str(left), "--right", str(right)]
            )
            result = _dispatch(args)
            self.assertTrue(result["ok"])
            self.assertEqual(1, result["pair_count"])
            self.assertTrue(result["pairs"][0]["agree"])

    def test_an_unbalanced_command_line_fails_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = write_specification(root, "left", specification())
            right = write_specification(root, "right", specification())
            with redirect_stdout(io.StringIO()) as captured:
                exit_code = main(
                    ["contract-crosscheck", "--left", str(left), "--left", str(right), "--right", str(left)]
                )
            self.assertEqual(1, exit_code)
            self.assertIn("error:", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
