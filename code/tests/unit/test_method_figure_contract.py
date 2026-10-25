"""Tests for evidence-aware hierarchical method figure contracts."""

import unittest

from rfs.contracts.method_figure import normalize_method_figure_content, validate_method_figure_content
from rfs.contracts.vlm_context import compact_method


class MethodFigureContractTests(unittest.TestCase):
    def test_a_relation_keeps_its_own_label_and_never_borrows_its_id(self):
        """A relation id is a coordinate, not a caption.

        The reader leaves the words off most panel edges, and the normalizer used to fill
        that gap with the id -- which is how "panelrel_i_04" was drawn on the figure.
        """

        content = normalize_method_figure_content({
            "modules": [{"id": "a", "label": "A", "evidence_ids": ["E1"]}, {"id": "b", "label": "B", "evidence_ids": ["E1"]}],
            "relations": [
                {"id": "panelrel_i_04", "source": "a", "target": "b", "type": "data_flow", "evidence_ids": ["E1"]},
                {"id": "rel_1", "source": "b", "target": "a", "type": "data_flow", "label": "co-evolve", "evidence_ids": ["E1"]},
            ],
        })
        self.assertEqual(content["relations"][0]["label"], "")
        self.assertEqual(content["relations"][1]["label"], "co-evolve")
        self.assertTrue(validate_method_figure_content(content)["ok"])

    def test_an_object_still_falls_back_to_its_id_for_a_label(self):
        """A node id names something; only relations are coordinates."""

        content = normalize_method_figure_content({"modules": [{"id": "input_docs", "evidence_ids": ["E1"]}]})
        self.assertEqual(content["modules"][0]["label"], "input_docs")

    def test_normalizes_hierarchical_renderability_fields(self):
        content = normalize_method_figure_content({
            "modules": [
                {"id": "stage", "label": "Stage", "detail_level": "stage", "must_show": True, "should_render": True, "evidence_ids": ["E1"]},
                {"id": "mechanism", "label": "Mechanism", "parent_id": "stage", "detail_level": "mechanism", "visualization_value": 0.9, "evidence_ids": ["E2"]},
            ],
            "optional_details": [{"id": "impl", "label": "Implementation", "should_render": False, "detail_level": "artifact", "evidence_ids": ["E3"]}],
        })
        self.assertEqual(content["version"], 2)
        self.assertEqual(content["modules"][1]["parent_id"], "stage")
        self.assertEqual(content["modules"][1]["detail_level"], "mechanism")
        self.assertFalse(content["optional_details"][0]["should_render"])
        self.assertTrue(validate_method_figure_content(content)["ok"])

    def test_renderable_object_without_evidence_is_drawn_and_listed(self):
        """An unsourced object is thin, not broken: it still draws.

        The bulk corpus cannot afford to lose a paper the reader already
        understood because one card has no citation, so the object is listed as
        unsourced instead of voiding the contract.
        """

        content = normalize_method_figure_content({"modules": [{"id": "m", "should_render": True}]})
        report = validate_method_figure_content(content)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(["m"], report["evidence_pending_ids"])
        self.assertTrue(any("without an evidence id" in repair for repair in report["repairs"]))

    def test_an_off_contract_detail_level_is_read_as_the_closest_level(self):
        content = normalize_method_figure_content({
            "modules": [
                {"id": "s", "detail_level": "submodule", "evidence_ids": ["E1"]},
                {"id": "t", "detail_level": "training procedure", "evidence_ids": ["E1"]},
                {"id": "u", "detail_level": "diagnostic dataset", "evidence_ids": ["E1"]},
                {"id": "v", "detail_level": "wibble", "evidence_ids": ["E1"]},
            ]
        })
        report = validate_method_figure_content(content)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(
            ["module", "mechanism", "artifact", "module"],
            [item["detail_level"] for item in content["modules"]],
        )
        self.assertEqual(4, len(report["repairs"]))

    def test_content_without_any_node_still_fails(self):
        report = validate_method_figure_content({"modules": [], "inputs": [], "outputs": []})
        self.assertFalse(report["ok"])

    def test_common_vlm_detail_aliases_normalize_to_contract_levels(self):
        content = normalize_method_figure_content({"modules": [{"id": "m", "evidence_ids": ["E1"]}], "innovations": [{"id": "whole", "detail_level": "system", "should_render": False}]})
        self.assertEqual(content["innovations"][0]["detail_level"], "stage")
        self.assertTrue(validate_method_figure_content(content)["ok"])

    def test_compact_method_preserves_hierarchy_and_optional_details(self):
        compact = compact_method(normalize_method_figure_content({
            "stages": [{"id": "s", "label": "Stage", "detail_level": "stage", "evidence_ids": ["E0"]}],
            "modules": [{"id": "m", "label": "Core", "parent_id": "s", "detail_level": "mechanism", "visualization_value": 0.9, "should_render": True, "evidence_ids": ["E1"]}],
            "optional_details": [{"id": "d", "label": "Detail", "detail_level": "artifact", "should_render": False, "evidence_ids": ["E2"]}],
        }))
        self.assertEqual(compact["modules"][0]["parent_id"], "s")
        self.assertEqual(compact["modules"][0]["detail_level"], "mechanism")
        self.assertEqual(compact["optional_details"][0]["should_render"], False)


    def test_relation_ids_used_as_object_ports_resolve_to_the_connected_objects(self):
        content = normalize_method_figure_content({
            "inputs": [{"id": "x", "label": "Covariates", "evidence_ids": ["E1"], "output_ids": ["rel_x_to_gen"]}],
            "modules": [{"id": "gen", "label": "Generator", "evidence_ids": ["E2"], "input_ids": ["rel_x_to_gen"], "output_ids": ["rel_gen_to_w"]}],
            "outputs": [{"id": "w", "label": "Decision", "evidence_ids": ["E3"], "input_ids": ["rel_gen_to_w"]}],
            "relations": [
                {"id": "rel_x_to_gen", "source": "x", "target": "gen", "type": "flow"},
                {"id": "rel_gen_to_w", "source": "gen", "target": "w", "type": "flow"},
            ],
        })
        self.assertEqual(content["inputs"][0]["output_ids"], ["gen"])
        self.assertEqual(content["modules"][0]["input_ids"], ["x"])
        self.assertEqual(content["modules"][0]["output_ids"], ["w"])
        self.assertEqual(content["outputs"][0]["input_ids"], ["gen"])
        self.assertTrue(validate_method_figure_content(content)["ok"])

    def test_an_undeclared_port_is_dropped_and_reported(self):
        """A port nothing declares cannot be drawn, so it is dropped, not fatal.

        The deck of a paper whose only fault was this pointer was thrown away
        whole; the reference is removed instead, and the repair is named.
        """

        content = normalize_method_figure_content({
            "modules": [{"id": "gen", "evidence_ids": ["E1"], "input_ids": ["rel_missing"]}],
            "relations": [{"id": "rel_x_to_gen", "source": "gen", "target": "gen", "type": "flow"}],
        })
        report = validate_method_figure_content(content)
        self.assertEqual(content["modules"][0]["input_ids"], [])
        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(any("undeclared input_ids" in repair and "rel_missing" in repair for repair in report["repairs"]))

    def test_a_handoff_named_in_a_port_is_kept_on_the_arrow_that_carries_it(self):
        """A handoff the figure draws as an arrow keeps its name on that arrow.

        The reader wrote ``ash.outputs = solver_assignment`` and
        ``solver.inputs = solver_assignment`` while the arrow ash -> solver is
        labelled "dynamic assignment"; the name belongs to the arrow, and 180
        decks were thrown away whole over a name like it.
        """

        content = normalize_method_figure_content({
            "modules": [
                {"id": "ash", "label": "ASH", "evidence_ids": ["E1"], "output_ids": ["solver_assignment"]},
                {"id": "solver", "label": "Solver", "evidence_ids": ["E2"], "input_ids": ["solver_assignment"]},
            ],
            "relations": [{"id": "c3", "source": "ash", "target": "solver", "type": "flow", "label": "dynamic assignment"}],
        })
        self.assertEqual(content["modules"][0]["output_ids"], [])
        self.assertEqual(content["modules"][1]["input_ids"], [])
        self.assertEqual([{key: entry[key] for key in ("name", "declared_by", "field", "match")} for entry in content["relations"][0]["product_aliases"]],
                         [{"name": "solver_assignment", "declared_by": "ash", "field": "output_ids", "match": "sole_arrow_at_this_node"}])
        # The second port is the same handoff seen from the other end, so it is
        # kept on the same arrow rather than counted as a second reading.
        bound = content["provenance"]["normalization_repairs"]["bound_product_aliases"]
        self.assertEqual([{name: record[name] for name in ("id", "product", "match")} for record in bound],
                         [{"id": "ash", "product": "solver_assignment", "match": "sole_arrow_at_this_node"},
                          {"id": "solver", "product": "solver_assignment", "match": "same_name_as_another_port"}])
        report = validate_method_figure_content(content)
        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(any("solver_assignment" in repair and "c3" in repair for repair in report["repairs"]))
        self.assertFalse(any("undeclared" in repair for repair in report["repairs"]))

    def test_a_handoff_the_arrow_itself_names_is_bound_by_that_label(self):
        """When the arrow's own label says the same thing, the match is the reader's."""

        content = normalize_method_figure_content({
            "modules": [
                {"id": "refiner", "label": "Refiner", "evidence_ids": ["E1"], "output_ids": ["refined_query"]},
                {"id": "solver", "label": "Solver", "evidence_ids": ["E2"]},
            ],
            "relations": [{"id": "c2", "source": "refiner", "target": "solver", "type": "flow", "label": "refined query"}],
        })
        self.assertEqual(content["modules"][0]["output_ids"], [])
        self.assertEqual(content["relations"][0]["product_aliases"][0]["declared_by"], "refiner")
        self.assertEqual(content["relations"][0]["product_aliases"][0]["match"], "arrow_label")

    def test_a_handoff_several_arrows_could_carry_is_still_dropped(self):
        content = normalize_method_figure_content({
            "modules": [
                {"id": "router", "label": "Router", "evidence_ids": ["E1"], "output_ids": ["mystery_state"]},
                {"id": "a", "label": "A", "evidence_ids": ["E2"]},
                {"id": "b", "label": "B", "evidence_ids": ["E3"]},
            ],
            "relations": [
                {"id": "r1", "source": "router", "target": "a", "type": "flow", "label": "left"},
                {"id": "r2", "source": "router", "target": "b", "type": "flow", "label": "right"},
            ],
        })
        self.assertEqual(content["modules"][0]["output_ids"], [])
        report = validate_method_figure_content(content)
        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(any("undeclared output_ids" in repair and "mystery_state" in repair for repair in report["repairs"]))

    def test_a_node_stated_under_two_fields_is_merged_into_one(self):
        content = normalize_method_figure_content({
            "modules": [{"id": "s", "label": "Training Stage", "detail_level": "module", "evidence_ids": ["E1"], "should_render": True}],
            "stages": [{"id": "s", "label": "Training", "detail_level": "stage", "evidence_ids": ["E2"]}],
        })
        self.assertEqual([item["id"] for item in content["modules"]], ["s"])
        self.assertEqual(content["stages"], [])
        self.assertEqual(content["modules"][0]["evidence_ids"], ["E1", "E2"])
        self.assertEqual(content["modules"][0]["detail_level"], "stage")
        self.assertTrue(content["modules"][0]["should_render"])
        report = validate_method_figure_content(content)
        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(any("stated again under stages" in repair for repair in report["repairs"]))

    def test_a_dangling_parent_pointer_is_dropped_and_reported(self):
        content = normalize_method_figure_content({"modules": [{"id": "m", "label": "Core", "parent_id": "ghost", "evidence_ids": ["E1"]}]})
        self.assertIsNone(content["modules"][0]["parent_id"])
        report = validate_method_figure_content(content)
        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(any("undeclared parent ghost" in repair for repair in report["repairs"]))

    def test_two_unrelated_objects_sharing_an_id_are_both_kept(self):
        content = normalize_method_figure_content({
            "modules": [
                {"id": "dup", "label": "Encoder", "evidence_ids": ["E1"]},
                {"id": "dup", "label": "Loss", "evidence_ids": ["E2"]},
            ],
        })
        self.assertEqual([item["id"] for item in content["modules"]], ["dup", "dup_2"])
        report = validate_method_figure_content(content)
        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(any("drawn as dup_2" in repair for repair in report["repairs"]))

    def test_a_relation_with_an_undeclared_endpoint_still_fails(self):
        report = validate_method_figure_content({
            "modules": [{"id": "a", "evidence_ids": ["E1"]}],
            "relations": [{"id": "r", "source": "a", "target": "b"}],
        })
        self.assertFalse(report["ok"])
        self.assertTrue(any("endpoint is not declared" in error for error in report["errors"]))

    def test_resolved_ports_drop_duplicates_and_self_references(self):
        content = normalize_method_figure_content({
            "inputs": [{"id": "a", "evidence_ids": ["E2"]}],
            "modules": [{"id": "gen", "evidence_ids": ["E1"], "input_ids": ["rel_a_to_gen", "rel_a_to_gen"], "output_ids": ["rel_gen_to_gen"]}],
            "relations": [
                {"id": "rel_a_to_gen", "source": "a", "target": "gen", "type": "flow"},
                {"id": "rel_gen_to_gen", "source": "gen", "target": "gen", "type": "loop"},
            ],
        })
        self.assertEqual(content["modules"][0]["input_ids"], ["a"])
        self.assertEqual(content["modules"][0]["output_ids"], [])

    def test_a_priority_written_as_a_word_is_read_instead_of_read_as_a_number(self):
        """The reader writes "critical" where the contract states a 0-1 priority.

        A bare ``float()`` turned that word into a ``ValueError``, and the boundary review
        that reads this contract died on it -- four iterations in a row -- before it could
        name one thing to change about the figure. The word is read through the one table
        the contract holds, and the reading is left on the field.
        """

        content = normalize_method_figure_content({
            "inputs": [{"id": "papers", "label": "Papers", "importance": "critical", "evidence_ids": ["E1"]}],
            "modules": [{"id": "reader", "label": "Reader", "importance": "high", "evidence_ids": ["E1"]}],
        })
        self.assertEqual(content["inputs"][0]["scientific_importance"], 0.95)
        self.assertEqual(content["modules"][0]["scientific_importance"], 0.8)
        self.assertTrue(validate_method_figure_content(content)["ok"])


if __name__ == "__main__":
    unittest.main()
