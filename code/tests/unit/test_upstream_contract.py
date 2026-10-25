"""Tests for adopting the middle layer's frozen contract in the generation line."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.analysis.paper.document import parse_paper
from rfs.analysis.paper.preparation import prepare_paper_figure_contract
from rfs.cli import build_parser
from rfs.workflows.paper_evolve_editable import _upstream_method_reuse, run_paper_evolve_editable
from rfs.contracts.semantic import (
    UPSTREAM_GROUND_TRUTH_COPY,
    FrozenContractUnusable,
    adopt_upstream_ground_truth,
    load_upstream_contract,
    merge_upstream_content_plan,
    merge_upstream_inventory,
    merge_upstream_review,
    reground_frozen_contract,
)
from rfs.utils import write_json


class StopRun(Exception):
    """Sentinel that ends a workflow run once the call under test has been made."""

PARAGRAPH = "We propose a frozen encoder that transforms a paper into a symbolic figure plan."


def write_paper(root: Path) -> Path:
    """Write one page paper whose second block is a quotable paragraph."""
    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Frozen Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(target)
    document.close()
    return target


def evidence_block(parsed: dict) -> dict:
    """Return the parsed block that holds the paragraph a frozen quote would carry."""
    normalized = PARAGRAPH.casefold().replace(" ", "")
    for item in parsed["evidence"]:
        if str(item.get("text") or "").casefold().replace(" ", "").startswith(normalized[:60]):
            return item
    raise AssertionError("the fixture paragraph was not parsed as an evidence block")


def frozen_pair(block: dict, *, evidence_id: str = "E_FROZEN_1") -> tuple[dict, dict]:
    """Return a review and a specification shaped the way the middle layer writes them."""
    quote = "A Frozen Study " + str(block.get("text") or "") + " Figure 1: Overview of the proposed framework."
    reference = {"evidence_id": evidence_id, "page": block["page"], "quote": quote, "section": "Introduction"}
    review = {
        "schema_version": "paper-review-v1",
        "paper_identity": {"paper_id": "p-test", "title": "A Frozen Study", "domain": "cv"},
        "central_claims": [
            {"id": "c1", "statement": "A frozen encoder makes the figure plan reproducible.", "evidence_ids": [evidence_id], "evidence_refs": [reference]},
        ],
        "inputs": [
            {"id": "in1", "visible_label": "paper", "evidence_ids": [evidence_id], "evidence_refs": [reference]},
        ],
        "modules": [
            {
                "id": "m1",
                "visible_label": "frozen encoder",
                "must_appear_in_figure": True,
                "evidence_ids": [evidence_id],
                "evidence_refs": [reference],
            },
        ],
        "outputs": [
            {"id": "out1", "visible_label": "figure plan", "evidence_ids": [evidence_id], "evidence_refs": [reference]},
        ],
        "innovations": [
            {"id": "i1", "visible_label": "frozen encoder", "evidence_ids": [evidence_id], "evidence_refs": [reference]},
        ],
        "relations": [
            {"id": "r1", "source_id": "in1", "target_id": "m1", "relation_type": "feeds", "statement": "feeds", "evidence_ids": [evidence_id], "evidence_refs": [reference]},
            {"id": "r2", "source_id": "m1", "target_id": "out1", "relation_type": "produces", "statement": "produces", "evidence_ids": [evidence_id], "evidence_refs": [reference]},
        ],
        "workflows": {"inference": [{"id": "w1", "visible_label": "frozen encoder", "evidence_ids": [evidence_id]}]},
        "forbidden_inventions": [],
    }
    specification = {
        "topology": "linear",
        "figure_goal": "Explain the frozen pipeline.",
        "required_labels": ["paper", "frozen encoder", "figure plan"],
        "inputs": [{"id": "in1", "label": "paper", "role": "input", "detail_level": "input", "must_show": True, "should_render": True, "render_priority": 1, "evidence_ids": [evidence_id]}],
        "modules": [{"id": "m1", "label": "frozen encoder", "role": "module", "detail_level": "module", "must_show": True, "should_render": True, "render_priority": 1, "evidence_ids": [evidence_id]}],
        "outputs": [{"id": "out1", "label": "figure plan", "role": "output", "detail_level": "output", "must_show": True, "should_render": True, "render_priority": 1, "evidence_ids": [evidence_id]}],
        "innovations": [{"id": "i1", "label": "frozen encoder", "role": "innovation", "detail_level": "innovation", "must_show": True, "should_render": True, "render_priority": 1, "evidence_ids": [evidence_id]}],
        "relations": [
            {"id": "r1", "source": "in1", "target": "m1", "type": "feeds", "label": "feeds", "evidence_ids": [evidence_id]},
            {"id": "r2", "source": "m1", "target": "out1", "type": "produces", "label": "produces", "evidence_ids": [evidence_id]},
        ],
        "forbidden_inventions": [],
        "uncertainties": [],
        "overview_panels": [],
        "optional_details": [],
        "seed": {"specification_version": "frozen-figure-specification-v1", "source": "pilot-review-seed", "paper_id": "p-test"},
    }
    return review, specification


def frozen_method(*, evidence_id: str = "E_FROZEN_1", labelled: bool = True) -> dict:
    """Return a method contract shaped the way the middle layer's contract step writes it."""
    return {
        "contract": "rfs.method_figure_content",
        "version": "1",
        "figure_goal": "Explain the frozen pipeline.",
        "inputs": [{"id": "in1", "label": "paper", "role": "input", "must_show": True, "evidence_ids": [evidence_id]}],
        "modules": [{"id": "m1", "label": "frozen encoder", "role": "module", "must_show": True, "evidence_ids": [evidence_id]}],
        "outputs": [{"id": "out1", "label": "figure plan", "role": "output", "must_show": True, "evidence_ids": [evidence_id]}],
        "innovations": [],
        "relations": [
            {
                "id": "r1",
                "source": "in1",
                "target": "m1",
                "type": "feeds",
                "label": "the paper feeds the frozen encoder" if labelled else "",
                "evidence_ids": [evidence_id],
            },
        ],
        "forbidden_inventions": [],
        "uncertainties": [],
        "provenance": {"paper_id": "p-test"},
    }


def middle_layer_directory(
    root: Path,
    review: dict,
    specification: dict,
    *,
    method: dict | None = None,
    method_report: dict | None = None,
    ground_truth: dict | None = None,
) -> Path:
    """Write the per paper files the middle layer ships, as it ships them."""
    directory = root / "graphs" / "p-test"
    write_json(directory / "paper_review.json", review)
    write_json(directory / "figure_specification.json", specification)
    if ground_truth is not None:
        write_json(directory / "scientific_ground_truth.json", ground_truth)
    if method is not None:
        write_json(directory / "method_figure_content.json", method)
        write_json(directory / "method_extraction_report.json", method_report if method_report is not None else {"ok": True, "mode": "vlm", "fallback_used": False})
    return directory


class UpstreamMethodContractTests(unittest.TestCase):
    """The middleware extracts the method contract, so the generation line does not."""

    def test_a_method_contract_is_loaded_with_the_pair(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            review, specification = frozen_pair({"page": 1, "text": PARAGRAPH})
            directory = middle_layer_directory(root, review, specification, method=frozen_method())
            loaded = load_upstream_contract(directory)

        self.assertEqual(loaded["status"], "loaded")
        self.assertEqual(loaded["method_status"], "loaded")
        self.assertEqual(loaded["method_figure_content"]["modules"][0]["label"], "frozen encoder")
        self.assertFalse(loaded["method_extraction_report"]["fallback_used"])

    def test_a_source_without_a_method_contract_is_still_loaded(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            review, specification = frozen_pair({"page": 1, "text": PARAGRAPH})
            directory = middle_layer_directory(root, review, specification)
            loaded = load_upstream_contract(directory)

        self.assertEqual(loaded["status"], "loaded")
        self.assertEqual(loaded["method_status"], "absent")
        self.assertEqual(loaded["method_figure_content"], {})

    def test_a_method_contract_the_middle_layer_flagged_is_not_handed_over(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            review, specification = frozen_pair({"page": 1, "text": PARAGRAPH})
            directory = middle_layer_directory(
                root,
                review,
                specification,
                method=frozen_method(),
                method_report={"ok": False, "mode": "heuristic", "fallback_used": True},
            )
            loaded = load_upstream_contract(directory)

        self.assertEqual(loaded["status"], "loaded")
        self.assertEqual(loaded["method_status"], "unusable")
        self.assertEqual(loaded["method_figure_content"], {})
        self.assertIn("not grounded", loaded["method_declined_reason"])


class UpstreamContractLoaderTests(unittest.TestCase):
    """The generation line reads the middle layer's contract, or says why it cannot."""

    def test_no_source_is_off_rather_than_declined(self):
        self.assertEqual(load_upstream_contract(None)["status"], "off")
        self.assertEqual(load_upstream_contract("")["status"], "off")

    def test_a_paper_directory_is_loaded_with_its_review(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            review, specification = frozen_pair({"page": 1, "text": PARAGRAPH})
            directory = middle_layer_directory(root, review, specification)
            loaded = load_upstream_contract(directory)

        self.assertEqual(loaded["status"], "loaded")
        self.assertEqual(loaded["declined_reason"], "")
        self.assertEqual(loaded["figure_specification"], specification)
        self.assertEqual(loaded["paper_review"], review)
        self.assertTrue(loaded["source"].endswith("p-test"))

    def test_a_missing_source_is_declined_not_fatal(self):
        declined = load_upstream_contract("no-such-directory-anywhere")
        self.assertEqual(declined["status"], "declined")
        self.assertIn("does not exist", declined["declined_reason"])
        self.assertEqual(declined["figure_specification"], {})

    def test_a_file_is_declined_rather_than_half_read(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            review, specification = frozen_pair({"page": 1, "text": PARAGRAPH})
            directory = middle_layer_directory(root, review, specification)
            declined = load_upstream_contract(directory / "figure_specification.json")

        self.assertEqual(declined["status"], "declined")
        self.assertIn("not a directory", declined["declined_reason"])

    def test_a_specification_without_its_review_is_declined(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _review, specification = frozen_pair({"page": 1, "text": PARAGRAPH})
            directory = root / "graphs" / "p-test"
            write_json(directory / "figure_specification.json", specification)
            declined = load_upstream_contract(directory)

        self.assertEqual(declined["status"], "declined")
        self.assertIn("paper_review.json", declined["declined_reason"])


class FrozenContractRegroundingTests(unittest.TestCase):
    """A frozen contract's evidence ids belong to its snapshot, not to this parse."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parsed = parse_paper(write_paper(self.root))
        self.block = evidence_block(self.parsed)
        self.review, self.specification = frozen_pair(self.block)

    def test_every_frozen_quote_lands_on_this_parse(self):
        contract, report = reground_frozen_contract(
            {"source": "/middle-layer", "paper_review": self.review, "figure_specification": self.specification},
            self.parsed,
        )

        self.assertTrue(report["fully_regrounded"], report)
        self.assertEqual(report["regrounded_evidence_ids"], report["referenced_evidence_ids"])
        self.assertEqual(contract["paper_review"]["modules"][0]["evidence_ids"], [self.block["id"]])
        self.assertEqual(contract["figure_specification"]["modules"][0]["evidence_ids"], [self.block["id"]])
        self.assertEqual(
            contract["figure_specification"]["relations"][0]["evidence_ids"],
            [self.block["id"]],
        )
        self.assertEqual(contract["paper_review"]["modules"][0]["evidence_refs"][0]["evidence_id"], self.block["id"])
        self.assertEqual(contract["paper_review"]["forbidden_inventions"], self.review["forbidden_inventions"])

    def test_the_specification_the_caller_handed_in_is_left_alone(self):
        frozen = {"source": "x", "paper_review": self.review, "figure_specification": self.specification}
        reground_frozen_contract(frozen, self.parsed)

        self.assertEqual(self.specification["modules"][0]["evidence_ids"], ["E_FROZEN_1"])
        self.assertEqual(self.review["modules"][0]["evidence_ids"], ["E_FROZEN_1"])

    def test_a_contract_for_another_paper_is_refused(self):
        foreign = {
            "paper_identity": {"paper_id": "p-other", "title": "A Different Study"},
            "modules": [
                {
                    "id": "m1",
                    "evidence_ids": ["E_OTHER_1"],
                    "evidence_refs": [{"evidence_id": "E_OTHER_1", "page": 7, "quote": "A sentence from a different paper entirely."}],
                },
            ],
        }
        foreign_specification = {"modules": [{"id": "m1", "label": "other", "evidence_ids": ["E_OTHER_1"]}]}
        with self.assertRaises(FrozenContractUnusable) as caught:
            reground_frozen_contract(
                {"source": "x", "paper_review": foreign, "figure_specification": foreign_specification},
                self.parsed,
            )

        self.assertIn("does not describe this paper", str(caught.exception))

    def test_a_partial_rebind_is_refused_at_the_default_and_kept_when_the_floor_is_lowered(self):
        review = json.loads(json.dumps(self.review))
        review["modules"][0]["evidence_refs"].extend(
            [
                {"evidence_id": "E_FROZEN_2", "page": 1, "quote": "A sentence that this paper never wrote down anywhere."},
                {"evidence_id": "E_FROZEN_3", "page": 2, "quote": "Another sentence that this paper never wrote down."},
            ]
        )
        review["modules"][0]["evidence_ids"].extend(["E_FROZEN_2", "E_FROZEN_3"])
        frozen = {"source": "x", "paper_review": review, "figure_specification": self.specification}

        with self.assertRaises(FrozenContractUnusable):
            reground_frozen_contract(frozen, self.parsed)

        _contract, report = reground_frozen_contract(frozen, self.parsed, min_ratio=0)

        self.assertEqual(report["reground_min_ratio"], 0)
        self.assertLess(report["regrounded_ratio"], 0.5)
        self.assertGreater(report["regrounded_evidence_ids"], 0)

    def test_an_unresolved_quote_is_reported_on_an_otherwise_good_contract(self):
        review = json.loads(json.dumps(self.review))
        review["modules"][0]["evidence_refs"].append(
            {"evidence_id": "E_FROZEN_2", "page": 1, "quote": "A sentence that this paper never wrote down anywhere."},
        )
        review["modules"][0]["evidence_ids"].append("E_FROZEN_2")
        _contract, report = reground_frozen_contract(
            {"source": "x", "paper_review": review, "figure_specification": self.specification},
            self.parsed,
        )

        self.assertFalse(report["fully_regrounded"])
        self.assertEqual(report["unresolved_evidence_ids"], ["E_FROZEN_2"])
        self.assertEqual(report["unresolved_evidence_count"], 1)

    def test_the_same_sentence_cited_twice_is_still_one_evidence_id(self):
        review = json.loads(json.dumps(self.review))
        review["modules"][0]["evidence_ids"] = ["E_FROZEN_1", "E_FROZEN_1"]
        contract, report = reground_frozen_contract(
            {"source": "x", "paper_review": review, "figure_specification": self.specification},
            self.parsed,
        )

        self.assertTrue(report["fully_regrounded"])
        self.assertEqual(contract["paper_review"]["modules"][0]["evidence_ids"], [self.block["id"]])

    def test_the_method_contract_rides_the_same_rebinding(self):
        method = frozen_method()
        contract, report = reground_frozen_contract(
            {
                "source": "/middle-layer",
                "paper_review": self.review,
                "figure_specification": self.specification,
                "method_figure_content": method,
                "method_extraction_report": {"ok": True},
            },
            self.parsed,
        )

        self.assertEqual(report["method_status"], "regrounded")
        self.assertEqual(report["method_referenced_evidence_ids"], 1)
        self.assertEqual(report["method_unresolved_evidence_count"], 0)
        self.assertEqual(contract["method_figure_content"]["modules"][0]["evidence_ids"], [self.block["id"]])
        self.assertEqual(contract["method_figure_content"]["relations"][0]["evidence_ids"], [self.block["id"]])
        self.assertEqual(contract["method_extraction_report"], {"ok": True})
        # The contract the caller handed in is never rewritten in place.
        self.assertEqual(method["modules"][0]["evidence_ids"], ["E_FROZEN_1"])

    def test_a_method_contract_that_cites_unknown_evidence_is_dropped_on_its_own(self):
        method = frozen_method(evidence_id="E_UNKNOWN_1")
        method["modules"][0]["evidence_ids"] = ["E_UNKNOWN_1", "E_UNKNOWN_2"]
        contract, report = reground_frozen_contract(
            {
                "source": "/middle-layer",
                "paper_review": self.review,
                "figure_specification": self.specification,
                "method_figure_content": method,
            },
            self.parsed,
        )

        self.assertEqual(report["method_status"], "dropped")
        self.assertIn("does not have", report["method_dropped_reason"])
        self.assertEqual(contract["method_figure_content"], {})
        # The review and the specification are still adopted.
        self.assertTrue(report["fully_regrounded"], report)
        self.assertEqual(contract["paper_review"]["modules"][0]["evidence_ids"], [self.block["id"]])


class UpstreamMethodReuseTests(unittest.TestCase):
    """A run that adopted a method contract does not extract it a second time."""

    def test_an_adopted_method_is_handed_to_the_method_stage(self):
        method, report = _upstream_method_reuse({
            "frozen_method": frozen_method(),
            "frozen_method_report": {"ok": True, "mode": "vlm"},
            "frozen_contract": {"source": "/middle-layer/graphs/p-test"},
        })

        self.assertEqual(method["modules"][0]["label"], "frozen encoder")
        self.assertEqual(report["mode"], "frozen_upstream")
        self.assertTrue(report["ok"])
        self.assertFalse(report["fallback_used"])
        self.assertEqual(report["node_count"], 3)
        self.assertEqual(report["relation_count"], 1)
        self.assertEqual(report["source"], "/middle-layer/graphs/p-test")

    def test_a_run_without_an_adopted_method_reuses_nothing(self):
        self.assertEqual(_upstream_method_reuse({}), ({}, {}))
        self.assertEqual(_upstream_method_reuse({"frozen_method": {}}), ({}, {}))
        self.assertEqual(_upstream_method_reuse({"frozen_method": None}), ({}, {}))

    def test_a_method_the_middle_layer_flagged_is_not_reused(self):
        method, report = _upstream_method_reuse({
            "frozen_method": frozen_method(),
            "frozen_method_report": {"ok": False, "fallback_used": True},
        })

        self.assertEqual((method, report), ({}, {}))

class UpstreamInventoryMergeTests(unittest.TestCase):
    """The middle layer's inventory deepens a run's own contract instead of replacing it."""

    def own_spec(self) -> dict:
        return {
            "topology": "linear",
            "required_labels": ["frozen encoder"],
            "modules": [{"id": "m1", "name": "frozen encoder", "role": "module", "evidence_ids": ["E1"]}],
            "panel_graphs": [
                {"id": "panel_i", "label": "build", "nodes": [{"instance_id": "n1", "name": "frozen encoder"}]},
            ],
            "relations": [],
        }

    def test_the_objects_only_the_middle_layer_found_are_added(self):
        merged, report = merge_upstream_inventory(
            self.own_spec(),
            {
                "modules": [
                    {"id": "m1", "label": "frozen encoder", "must_show": True, "evidence_ids": ["E1", "E2"]},
                    {"id": "m2", "label": "three-layer memory", "must_show": True, "evidence_ids": ["E3"]},
                    {"id": "m3", "label": "conflict detector", "must_show": True, "evidence_ids": ["E4"]},
                ],
            },
        )

        self.assertEqual([module["id"] for module in merged["modules"]], ["m1", "m2", "m3"])
        self.assertEqual(sorted(report["added_ids"]), ["m2", "m3"])
        self.assertEqual(report["folded_ids"], ["m1"])
        self.assertEqual(report["fields"]["modules"], {"added": 2, "folded": 1})
        self.assertEqual(report["object_count"], 3)

    def test_an_object_both_state_is_kept_once_with_both_evidence_lists(self):
        merged, _report = merge_upstream_inventory(
            self.own_spec(),
            {"modules": [{"id": "m1", "label": "frozen encoder", "must_show": True, "evidence_ids": ["E2"]}]},
        )

        module = merged["modules"][0]
        self.assertEqual(module["evidence_ids"], ["E1", "E2"])
        # What the run read is still what the contract says, and what only the middle
        # layer stated -- that this object is required -- survives the fold.
        self.assertEqual(module["name"], "frozen encoder")
        self.assertEqual(module["label"], "frozen encoder")
        self.assertTrue(module["must_show"])

    def test_the_run_keeps_the_panels_it_read_off_the_paper(self):
        merged, report = merge_upstream_inventory(
            self.own_spec(),
            {"panel_graphs": [{"id": "panel_mid", "label": "middle", "nodes": [{"instance_id": "x", "name": "other"}]}]},
        )

        self.assertEqual(report["panel_source"], "run")
        self.assertEqual([panel["id"] for panel in merged["panel_graphs"]], ["panel_i"])

    def test_a_run_that_read_no_panels_takes_the_middle_layers(self):
        own = self.own_spec()
        own["panel_graphs"] = []
        merged, report = merge_upstream_inventory(
            own,
            {"panel_graphs": [{"id": "panel_mid", "label": "middle", "nodes": [{"instance_id": "x", "name": "other"}]}]},
        )

        self.assertEqual(report["panel_source"], "middle_layer")
        self.assertEqual([panel["id"] for panel in merged["panel_graphs"]], ["panel_mid"])

    def test_a_relation_only_the_middle_layer_states_is_added(self):
        merged, report = merge_upstream_inventory(
            self.own_spec(),
            {
                "modules": [
                    {"id": "m1", "label": "frozen encoder", "evidence_ids": ["E2"]},
                    {"id": "m2", "label": "memory", "evidence_ids": ["E3"]},
                ],
                "relations": [
                    {"id": "r9", "source": "m1", "target": "m2", "type": "feeds", "evidence_ids": ["E2"]},
                ],
            },
        )

        self.assertEqual(report["added_relation_count"], 1)
        self.assertEqual(
            merged["relations"],
            [{"id": "r9", "source": "m1", "target": "m2", "type": "feeds", "evidence_ids": ["E2"]}],
        )

    def test_nothing_to_merge_leaves_the_run_own_contract_alone(self):
        own = self.own_spec()
        merged, report = merge_upstream_inventory(own, None)

        self.assertEqual(merged["modules"], own["modules"])
        self.assertEqual(report["added_ids"], [])
        self.assertEqual(report["panel_source"], "run")
        self.assertEqual(report["object_count"], 1)


class UpstreamContentPlanMergeTests(unittest.TestCase):
    """The middle layer's figure content plan deepens a run's own plan instead of replacing it."""

    def own_plan(self) -> dict:
        return {
            "contract": "rfs.method_figure_content",
            "version": 2,
            "figure_goal": "Show the figure the paper printed.",
            "inputs": [],
            "outputs": [],
            "modules": [{"id": "m1", "label": "frozen encoder", "detail_level": "module", "evidence_ids": ["E1"]}],
            "relations": [],
            "stages": [],
            "innovations": [],
            "optional_details": [],
            "forbidden_inventions": [],
            "uncertainties": [],
            "provenance": {"paper": "/papers/p-test.pdf"},
        }

    def test_the_objects_only_the_middle_layer_found_are_added(self):
        merged, report = merge_upstream_content_plan(
            self.own_plan(),
            {
                "modules": [
                    {"id": "m1", "label": "frozen encoder", "must_show": True, "evidence_ids": ["E1", "E2"]},
                    {"id": "m2", "label": "three-layer memory", "must_show": True, "evidence_ids": ["E3"]},
                ],
            },
        )

        self.assertEqual([module["id"] for module in merged["modules"]], ["m1", "m2"])
        self.assertEqual(report["added_ids"], ["m2"])
        self.assertEqual(report["folded_ids"], ["m1"])
        self.assertEqual(report["fields"]["modules"], {"added": 1, "folded": 1})
        self.assertEqual(report["node_count"], 2)

    def test_the_run_keeps_its_own_aim_and_what_the_middle_layer_marked_survives(self):
        merged, _report = merge_upstream_content_plan(
            self.own_plan(),
            {
                "figure_goal": "Explain the frozen pipeline.",
                "modules": [{"id": "m1", "label": "frozen encoder", "must_show": True, "evidence_ids": ["E2"]}],
            },
        )

        # The goal and the provenance are the run's: what it planned is aimed at the figure
        # the paper printed, and the middleware's reading does not get to move that aim.
        self.assertEqual(merged["figure_goal"], "Show the figure the paper printed.")
        self.assertEqual(merged["provenance"], {"paper": "/papers/p-test.pdf"})
        module = merged["modules"][0]
        self.assertEqual(module["evidence_ids"], ["E1", "E2"])
        self.assertTrue(module["must_show"])

    def test_a_relation_lands_on_the_object_whichever_plan_named_it(self):
        merged, report = merge_upstream_content_plan(
            self.own_plan(),
            {
                "modules": [{"id": "mid-encoder", "label": "frozen encoder", "evidence_ids": ["E2"]}],
                "outputs": [{"id": "out1", "label": "figure plan", "evidence_ids": ["E4"]}],
                "relations": [{"id": "r9", "source": "mid-encoder", "target": "out1", "type": "produces"}],
            },
        )

        self.assertEqual(report["added_relation_count"], 1)
        # The middleware named the encoder by its own id; the relation has to name the object
        # this run kept, or it points at a node the plan does not hold.
        self.assertEqual(merged["relations"][0]["source"], "m1")
        self.assertEqual(merged["relations"][0]["target"], "out1")

    def test_nothing_to_merge_leaves_the_run_own_plan_alone(self):
        own = self.own_plan()
        merged, report = merge_upstream_content_plan(own, None)

        self.assertEqual(merged["modules"], own["modules"])
        self.assertEqual(merged["figure_goal"], own["figure_goal"])
        self.assertEqual(report["added_ids"], [])
        self.assertEqual(report["node_count"], 1)


class UpstreamReviewMergeTests(unittest.TestCase):
    """The run's own reading of a paper is folded into the middle layer's review."""

    def own_review(self) -> dict:
        return {
            "schema_version": "2.0-fast",
            "paper_identity": {"title": "A Frozen Study", "paper_type": "unknown"},
            "domain_profile": "ai-ml-method",
            "central_claims": [{"id": "c1", "statement": "frozen encoder claim", "evidence_ids": ["E_run"]}],
            "modules": [{"id": "m1", "visible_label": "frozen encoder", "evidence_ids": ["E_run"]}],
            "outputs": [],
            "innovations": [],
            "relations": [],
            "forbidden_inventions": [],
            "unknowns": [],
        }

    def upstream_review(self) -> dict:
        return {
            "schema_version": "2.0",
            "paper_identity": {"paper_id": "p-test", "title": "A Frozen Study", "domain": "cv"},
            "domain_profile": "general",
            "central_claims": [
                {"id": "c1", "statement": "the middleware's claim", "must_appear_in_figure": True, "evidence_ids": ["E_mid"]},
            ],
            "modules": [
                {"id": "m1", "visible_label": "frozen encoder", "statement": "the middleware's encoder", "evidence_ids": ["E_mid"]},
            ],
            "outputs": [{"id": "out1", "visible_label": "figure plan", "evidence_ids": ["E_mid"]}],
            "innovations": [],
            "relations": [],
            "forbidden_inventions": [],
            "unknowns": [],
        }

    def test_a_fact_both_readings_state_is_kept_as_the_middle_layer_stated_it(self):
        merged, report = merge_upstream_review(self.own_review(), self.upstream_review())

        self.assertEqual(report["folded_ids"], ["c1", "m1"])
        self.assertEqual(report["added_ids"], [])
        claim = merged["central_claims"][0]
        # The middleware read the paper in more depth, so what it stated stands; the run's
        # own evidence on the same fact is added to it rather than replacing it.
        self.assertEqual(claim["statement"], "the middleware's claim")
        self.assertEqual(claim["evidence_ids"], ["E_mid", "E_run"])
        self.assertEqual(merged["modules"][0]["statement"], "the middleware's encoder")
        self.assertEqual(merged["outputs"], [{"id": "out1", "visible_label": "figure plan", "evidence_ids": ["E_mid"]}])

    def test_a_fact_only_the_run_read_is_added(self):
        own = self.own_review()
        own["contributions"] = [{"id": "contrib_1", "statement": "the run's own contribution", "evidence_ids": ["E_run"]}]

        merged, report = merge_upstream_review(own, self.upstream_review())

        self.assertEqual(report["added_ids"], ["contrib_1"])
        self.assertEqual([item["id"] for item in merged["contributions"]], ["contrib_1"])

    def test_a_relation_lands_on_the_fact_that_was_kept(self):
        own = self.own_review()
        own["modules"] = [{"id": "mid-encoder", "visible_label": "frozen encoder", "evidence_ids": ["E_run"]}]
        own["relations"] = [
            {"id": "relation_0", "source_id": "mid-encoder", "target_id": "out1", "relation_type": "produces", "evidence_ids": ["E_run"]},
        ]

        merged, report = merge_upstream_review(own, self.upstream_review())

        self.assertEqual(report["added_relation_count"], 1)
        # The run named the encoder by its own id; the relation has to name the fact that
        # was kept, or it points at an object the review does not hold.
        self.assertEqual(merged["relations"][0]["source_id"], "m1")
        self.assertEqual(merged["relations"][0]["target_id"], "out1")

    def test_a_grouped_step_only_the_run_read_is_added_to_the_group(self):
        own = self.own_review()
        own["workflows"] = {"inference": [{"id": "step_1", "statement": "the run's own step", "evidence_ids": ["E_run"]}]}
        upstream = self.upstream_review()
        upstream["workflows"] = {"training": [{"id": "train_1", "statement": "the middleware's step", "evidence_ids": ["E_mid"]}]}

        merged, report = merge_upstream_review(own, upstream)

        self.assertEqual([item["id"] for item in merged["workflows"]["training"]], ["train_1"])
        self.assertEqual([item["id"] for item in merged["workflows"]["inference"]], ["step_1"])
        self.assertEqual(report["fields"]["workflows.inference"], {"added": 1, "folded": 0})
        for name in ("offline", "online", "feedback"):
            self.assertEqual(merged["workflows"][name], [])

    def test_a_fact_that_states_nothing_and_cites_nothing_is_not_a_reading(self):
        own = self.own_review()
        own["research_questions"] = [{"id": "research_problem", "statement": "unknown", "evidence_ids": []}]

        merged, report = merge_upstream_review(own, self.upstream_review())

        self.assertEqual(report["added_ids"], [])
        self.assertNotIn("research_questions", merged)

    def test_the_runs_own_domain_stands_over_the_one_the_middle_layer_filed(self):
        merged, _report = merge_upstream_review(self.own_review(), self.upstream_review())

        self.assertEqual(merged["domain_profile"], "ai-ml-method")
        self.assertEqual(merged["paper_identity"]["paper_id"], "p-test")
        self.assertEqual(merged["paper_identity"]["title"], "A Frozen Study")

    def test_nothing_to_merge_leaves_the_middle_layers_review_alone(self):
        upstream = self.upstream_review()
        merged, report = merge_upstream_review(None, upstream)

        self.assertEqual(merged["modules"], upstream["modules"])
        self.assertEqual(report["added_ids"], [])
        self.assertEqual(report["folded_ids"], [])


class GenerationLineAdoptionTests(unittest.TestCase):
    """A run that is handed the middle layer's contract plans nothing itself."""

    def paid_call(self, *args, **kwargs):
        raise AssertionError("the generation line paid for a contract it was handed")

    def test_the_frozen_contract_is_adopted_and_nothing_is_paid_for(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paper = write_paper(root)
            parsed = parse_paper(paper)
            block = evidence_block(parsed)
            review, specification = frozen_pair(block)
            directory = middle_layer_directory(root, review, specification)
            upstream = load_upstream_contract(directory)
            out = root / "run"
            with patch("rfs.analysis.paper.preparation.plan_fast_paper_contract", self.paid_call), \
                 patch("rfs.analysis.paper.preparation.plan_paper_image", self.paid_call), \
                 patch("rfs.analysis.paper.review.call_vlm_json", self.paid_call), \
                 patch("rfs.analysis.paper.preparation.repair_plan_evidence", self.paid_call):
                prepared = prepare_paper_figure_contract(
                    paper=paper, out=out, planner_mode="vlm", ocr_engine="off", read_mode="fast",
                    frozen_contract=upstream,
                )
            grounding = json.loads((out / "upstream_contract_grounding.json").read_text(encoding="utf-8"))
            report = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
            merge = json.loads((out / "upstream_inventory_merge.json").read_text(encoding="utf-8"))
            review_merge = json.loads((out / "upstream_review_merge.json").read_text(encoding="utf-8"))

        self.assertTrue(prepared["ok"], prepared.get("planning_validation"))
        self.assertEqual(prepared["contract_source"], "frozen_upstream_contract")
        self.assertEqual(prepared["paper_review_mode"], "frozen_upstream_merged")
        self.assertEqual(prepared["planner_mode"], "frozen_upstream_merged")
        self.assertTrue(prepared["production_ready"])
        self.assertTrue(grounding["fully_regrounded"], grounding)
        self.assertEqual(report["frozen_contract"], grounding)
        self.assertIn("upstream_contract_grounding.json", report["artifacts"])
        self.assertIn("upstream_review_merge.json", report["artifacts"])
        self.assertIn("upstream_inventory_merge.json", report["artifacts"])
        self.assertEqual(merge["source"], str(upstream["source"]))
        # The review's own objects were folded too: the run compiled them from the
        # same review the middle layer's specification came from.
        self.assertIn("m1", merge["folded_ids"])
        self.assertEqual(merge["panel_source"], "none")
        modules = prepared["plan"]["figure_specification"]["modules"]
        self.assertEqual([module["label"] for module in modules], ["frozen encoder"])
        # The run's own reading of the paper is folded into the middleware's rather than
        # standing in for it: the facts both readings state are kept once, and the run's
        # own reading invents nothing the middleware did not state.
        self.assertEqual(review_merge["source"], str(upstream["source"]))
        self.assertEqual(sorted(review_merge["folded_ids"]), ["c1", "i1", "in1", "m1", "out1"])
        self.assertEqual(review_merge["added_ids"], [])

    def test_without_a_source_the_run_plans_its_own_contract(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paper = write_paper(root)
            out = root / "run"
            with patch("rfs.analysis.paper.preparation.plan_fast_paper_contract", self.paid_call):
                with self.assertRaises(AssertionError):
                    prepare_paper_figure_contract(
                        paper=paper, out=out, planner_mode="vlm", ocr_engine="off", read_mode="fast",
                    )
            report = json.loads((out / "run_report.json").read_text(encoding="utf-8")) if (out / "run_report.json").is_file() else None

        self.assertIsNone(report)
        self.assertFalse((out / "upstream_contract_grounding.json").is_file())

    def test_the_adopted_method_reaches_the_run_directory(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paper = write_paper(root)
            block = evidence_block(parse_paper(paper))
            review, specification = frozen_pair(block)
            directory = middle_layer_directory(root, review, specification, method=frozen_method())
            upstream = load_upstream_contract(directory)
            out = root / "run"
            with patch("rfs.analysis.paper.preparation.plan_fast_paper_contract", self.paid_call), \
                 patch("rfs.analysis.paper.preparation.plan_paper_image", self.paid_call), \
                 patch("rfs.analysis.paper.preparation.build_paper_review", self.paid_call), \
                 patch("rfs.analysis.paper.preparation.repair_plan_evidence", self.paid_call):
                prepared = prepare_paper_figure_contract(
                    paper=paper, out=out, planner_mode="vlm", ocr_engine="off", read_mode="fast",
                    frozen_contract=upstream,
                )
            report = json.loads((out / "run_report.json").read_text(encoding="utf-8"))

        self.assertEqual(prepared["method_source"], "middle_layer")
        self.assertEqual(report["method_source"], "middle_layer")
        self.assertEqual(prepared["frozen_method"]["modules"][0]["evidence_ids"], [block["id"]])
        self.assertTrue(prepared["production_ready"], prepared.get("planning_validation"))


class AdoptedGroundTruthTests(unittest.TestCase):
    """A run files the contract's own ground truth, or says it has none to file."""

    def test_the_contract_is_filed_verbatim_beside_the_run(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "export" / "scientific_ground_truth.json"
            source.parent.mkdir(parents=True)
            source.write_bytes(b'{"contract": "rfs.scientific_ground_truth"}\n')

            record = adopt_upstream_ground_truth(root / "run", source)

            copy = root / "run" / UPSTREAM_GROUND_TRUTH_COPY
            self.assertTrue(record["adopted"])
            self.assertEqual(record["file"], UPSTREAM_GROUND_TRUTH_COPY)
            self.assertEqual(record["source"], str(source.resolve()))
            self.assertEqual(record["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(copy.read_bytes(), source.read_bytes())

    def test_a_source_without_a_ground_truth_files_nothing(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)

            record = adopt_upstream_ground_truth(root / "run", None)

            self.assertEqual(record, {"adopted": False, "file": "", "sha256": "", "source": ""})
            self.assertFalse((root / "run" / UPSTREAM_GROUND_TRUTH_COPY).exists())

    def test_a_loaded_source_hands_over_the_path_of_its_ground_truth(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            review, specification = frozen_pair({"page": 1, "text": PARAGRAPH})
            directory = middle_layer_directory(root, review, specification, ground_truth={"paper_id": "p-test"})
            loaded = load_upstream_contract(directory)

            self.assertEqual(loaded["ground_truth_path"], str((directory / "scientific_ground_truth.json").resolve()))

    def test_a_source_without_one_is_loaded_with_an_empty_path(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            review, specification = frozen_pair({"page": 1, "text": PARAGRAPH})
            directory = middle_layer_directory(root, review, specification)

            self.assertEqual(load_upstream_contract(directory)["ground_truth_path"], "")


class UpstreamGroundTruthWiringTests(unittest.TestCase):
    """The workflow only files the copy for a contract it actually adopted."""

    def test_a_declined_contract_leaves_no_adopted_copy(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            review, specification = frozen_pair({"page": 1, "text": PARAGRAPH})
            directory = middle_layer_directory(root, review, specification, ground_truth={"paper_id": "p-test"})
            calls = []

            def refuse(*args, **kwargs):
                calls.append(kwargs.get("frozen_contract"))
                raise FrozenContractUnusable("the frozen contract's evidence does not describe this paper")

            with self.assertRaises(FrozenContractUnusable):
                with patch("rfs.workflows.paper_evolve_editable.prepare_paper_figure_contract", side_effect=refuse):
                    run_paper_evolve_editable(
                        paper=root / "paper.pdf",
                        out=root / "run",
                        upstream_from=directory,
                        planner_mode="heuristic",
                    )
            report = json.loads((root / "run" / "upstream_contract_report.json").read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 1)
        self.assertEqual(report["status"], "declined")
        self.assertFalse(report["scientific_ground_truth"]["adopted"])
        self.assertFalse((root / "run" / UPSTREAM_GROUND_TRUTH_COPY).exists())


class UpstreamCommandSurfaceTests(unittest.TestCase):
    """The command takes the middle layer directory and hands it to the workflow."""

    def run_workflow(self, root: Path, upstream, side_effect):
        """Run the workflow until the preparation stand-in stops it."""
        with patch("rfs.workflows.paper_evolve_editable.prepare_paper_figure_contract", side_effect=side_effect):
            with self.assertRaises(StopRun):
                run_paper_evolve_editable(
                    paper=root / "paper.pdf",
                    out=root / "run",
                    upstream_from=upstream,
                    planner_mode="heuristic",
                )
        return json.loads((root / "run" / "upstream_contract_report.json").read_text(encoding="utf-8"))

    def test_the_workflow_hands_the_loaded_contract_to_the_paper_stage(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            review, specification = frozen_pair({"page": 1, "text": PARAGRAPH})
            directory = middle_layer_directory(root, review, specification)
            seen = {}

            def capture(*args, **kwargs):
                seen.update(kwargs)
                raise StopRun

            report = self.run_workflow(root, directory, capture)

        self.assertEqual(seen["frozen_contract"]["status"], "loaded")
        self.assertEqual(report["status"], "loaded")
        self.assertTrue(report["used"])
        self.assertEqual(report["specification_source"], "middle_layer_merged")
        # The review and the plan are no longer taken whole, so no call is reported as
        # one this run skipped on their account.
        self.assertEqual(report["provider_calls_avoided"], {})
        self.assertEqual(report["paper_identity"]["paper_id"], "p-test")

    def test_a_contract_that_is_not_this_paper_is_dropped_and_reported(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            review, specification = frozen_pair({"page": 1, "text": PARAGRAPH})
            directory = middle_layer_directory(root, review, specification)
            calls = []

            def refuse(*args, **kwargs):
                calls.append(kwargs.get("frozen_contract"))
                raise FrozenContractUnusable("the frozen contract's evidence does not describe this paper")

            with self.assertRaises(FrozenContractUnusable):
                with patch("rfs.workflows.paper_evolve_editable.prepare_paper_figure_contract", side_effect=refuse):
                    run_paper_evolve_editable(
                        paper=root / "paper.pdf",
                        out=root / "run",
                        upstream_from=directory,
                        planner_mode="heuristic",
                    )
            report = json.loads((root / "run" / "upstream_contract_report.json").read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["status"], "loaded")
        self.assertEqual(report["status"], "declined")
        self.assertFalse(report["used"])
        self.assertIn("does not describe this paper", report["declined_reason"])
        self.assertEqual(report["specification_source"], "declined")

    def test_a_source_that_cannot_be_read_still_runs_without_one(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            seen = {}

            def capture(*args, **kwargs):
                seen.update(kwargs)
                raise StopRun

            report = self.run_workflow(root, root / "absent", capture)

        self.assertIsNone(seen["frozen_contract"])
        self.assertEqual(report["status"], "declined")
        self.assertIn("does not exist", report["declined_reason"])

    def test_the_command_accepts_an_upstream_directory(self):
        args = build_parser().parse_args([
            "paper-evolve-editable", "--paper", "paper.pdf", "--out", "out",
            "--upstream-from", "middle-layer/p-test",
        ])

        self.assertEqual(args.upstream_from, "middle-layer/p-test")

    def test_the_command_defaults_to_planning_its_own_contract(self):
        args = build_parser().parse_args(["paper-evolve-editable", "--paper", "paper.pdf", "--out", "out"])

        self.assertIsNone(args.upstream_from)


if __name__ == "__main__":
    unittest.main()
