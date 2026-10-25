"""Tests for repairing plan objects that were left without evidence citations."""

from __future__ import annotations

import unittest
from unittest import mock

from rfs.planning import paper_contract
from rfs.planning.paper_contract import apply_evidence_fixes, collect_evidence_slots, repair_plan_evidence, ungrounded_evidence_slots


def _spec() -> dict:
    return {
        "modules": [
            {"id": "m1", "visible_label": "Image Encoder", "evidence_ids": ["E0001"]},
            {"id": "m2", "visible_label": "Recycler", "evidence_ids": []},
        ],
        "inputs": [{"id": "in1", "visible_label": "Sequence", "evidence_ids": ["E0002"]}],
        "outputs": [{"id": "out1", "visible_label": "Structure", "evidence_ids": []}],
        "relations": [{"source": "m1", "target": "m2", "evidence_ids": ["E0003"]}],
        "panel_graphs": [
            {
                "id": "panel_a",
                "nodes": [{"instance_id": "panel_a_m1", "visible_label": "Image Encoder", "evidence_ids": []}],
                "relations": [{"source": "panel_a_m1", "target": "panel_a_m1", "evidence_ids": []}],
            }
        ],
    }


def _parsed() -> dict:
    return {"evidence": [{"id": "E0001", "text": "one"}, {"id": "E0002", "text": "two"}, {"id": "E0003", "text": "three"}, {"id": "E0009", "text": "nine"}]}


class EvidenceSlotTests(unittest.TestCase):
    def test_every_evidence_bearing_object_gets_a_stable_key(self) -> None:
        slots = collect_evidence_slots(_spec())
        self.assertIn("module:m2", slots)
        self.assertIn("outputs:out1", slots)
        self.assertIn("panel_instance:panel_a_m1", slots)
        self.assertIn("panel_relation:panel_a:panel_a_m1->panel_a_m1", slots)
        self.assertIn("relation:m1->m2", slots)
        self.assertEqual(slots["module:m1"]["evidence_ids"], ["E0001"])

    def test_only_the_objects_that_owe_a_citation_are_listed(self) -> None:
        missing = [slot["key"] for slot in ungrounded_evidence_slots(_spec())]
        self.assertEqual(sorted(missing), ["module:m2", "outputs:out1", "panel_instance:panel_a_m1", "panel_relation:panel_a:panel_a_m1->panel_a_m1"])


class ApplyEvidenceFixTests(unittest.TestCase):
    def test_fills_empty_citations_and_drops_unknown_ids_and_keys(self) -> None:
        spec = _spec()
        outcome = apply_evidence_fixes(
            spec,
            [
                {"key": "module:m2", "evidence_ids": ["E0009", "E9999", "E0009"]},
                {"key": "panel_instance:panel_a_m1", "evidence_ids": ["E0001"]},
                {"key": "not_a_slot", "evidence_ids": ["E0001"]},
                {"key": "module:m1", "evidence_ids": ["E0009"]},
            ],
            valid_evidence={"E0001", "E0002", "E0003", "E0009"},
        )
        applied = {item["key"]: item["evidence_ids"] for item in outcome["applied"]}
        self.assertEqual(applied["module:m2"], ["E0009"])
        self.assertEqual(applied["panel_instance:panel_a_m1"], ["E0001"])
        self.assertEqual(spec["modules"][1]["evidence_ids"], ["E0009"])
        self.assertEqual(spec["panel_graphs"][0]["nodes"][0]["evidence_ids"], ["E0001"])
        self.assertEqual(spec["modules"][0]["evidence_ids"], ["E0001"])
        reasons = {item["key"]: item["reason"] for item in outcome["rejected"]}
        self.assertEqual(reasons["not_a_slot"], "unknown slot")
        self.assertEqual(reasons["module:m1"], "already cited")

    def test_a_fix_with_no_known_evidence_changes_nothing(self) -> None:
        spec = _spec()
        outcome = apply_evidence_fixes(spec, [{"key": "module:m2", "evidence_ids": ["E9999"]}], valid_evidence={"E0001"})
        self.assertEqual(outcome["applied"], [])
        self.assertEqual(spec["modules"][1]["evidence_ids"], [])


class RepairPlanEvidenceTests(unittest.TestCase):
    def test_repair_cites_the_objects_the_plan_left_ungrounded(self) -> None:
        plan = {"figure_specification": _spec()}
        answer = {"evidence_fixes": [{"key": "module:m2", "evidence_ids": ["E0009"]}, {"key": "outputs:out1", "evidence_ids": ["E0003"]}]}
        with mock.patch.object(paper_contract, "vlm_credentials_available", return_value=True), mock.patch.object(paper_contract, "call_vlm_json", return_value=answer) as call:
            repaired, metadata = repair_plan_evidence(plan, _parsed(), ["module m2 has no evidence_ids"], model="test-model")
        self.assertEqual(metadata["mode"], "vlm")
        self.assertEqual(len(metadata["applied"]), 2)
        self.assertEqual(repaired["figure_specification"]["modules"][1]["evidence_ids"], ["E0009"])
        self.assertEqual(repaired["figure_specification"]["outputs"][0]["evidence_ids"], ["E0003"])
        self.assertIn("module:m2", call.call_args.args[0])

    def test_repair_keeps_the_plan_when_the_call_fails(self) -> None:
        plan = {"figure_specification": _spec()}
        with mock.patch.object(paper_contract, "vlm_credentials_available", return_value=True), mock.patch.object(paper_contract, "call_vlm_json", side_effect=RuntimeError("timeout")):
            repaired, metadata = repair_plan_evidence(plan, _parsed(), ["module m2 has no evidence_ids"])
        self.assertEqual(metadata["mode"], "failed")
        self.assertIn("timeout", metadata["warning"])
        self.assertEqual(repaired["figure_specification"]["modules"][1]["evidence_ids"], [])

    def test_a_grounded_plan_is_not_sent_to_the_model(self) -> None:
        spec = _spec()
        for slot in list(spec["modules"]) + list(spec["outputs"]):
            slot["evidence_ids"] = ["E0001"]
        spec["panel_graphs"][0]["nodes"][0]["evidence_ids"] = ["E0001"]
        spec["panel_graphs"][0]["relations"][0]["evidence_ids"] = ["E0001"]
        with mock.patch.object(paper_contract, "call_vlm_json") as call:
            _, metadata = repair_plan_evidence({"figure_specification": spec}, _parsed(), [])
        self.assertEqual(metadata["mode"], "skipped")
        call.assert_not_called()

class FastPlanNamedEntityTests(unittest.TestCase):
    """The fast path must not ship an entity or panel node without a visible name."""

    def _plan(self, specification: dict) -> dict:
        return paper_contract._compile_fast_plan(
            {"figure_specification": specification},
            {"aspect_ratio": "16:9"},
        )

    def test_unnamed_entities_are_dropped_with_their_edges(self):
        plan = self._plan({
            "modules": [
                {"id": "m1", "name": "Global Memory"},
                {"id": "m2", "name": "   "},
                {"id": "m3", "role": "module"},
            ],
            "relations": [
                {"source": "m1", "target": "m2"},
                {"source": "m1", "target": "m3"},
            ],
            "panel_graphs": [{
                "id": "panel_a",
                "panel_label": "a",
                "nodes": [{"instance_id": "n1", "name": "Encoder"}, {"instance_id": "n2", "name": ""}],
                "relations": [{"source": "n1", "target": "n2"}, {"source": "n1", "target": "n1"}],
            }],
            "uncertainties": [],
        })
        spec = plan["figure_specification"]

        self.assertEqual([item["id"] for item in spec["modules"]], ["m1"])
        self.assertEqual([(item["source"], item["target"]) for item in spec["relations"]], [])
        self.assertEqual([node["instance_id"] for node in spec["panel_graphs"][0]["nodes"]], ["n1"])
        self.assertEqual(
            [(item["source"], item["target"]) for item in spec["panel_graphs"][0]["relations"]],
            [("n1", "n1")],
        )
        self.assertEqual(len(spec["uncertainties"]), 1)
        self.assertIn("m2", spec["uncertainties"][0])
        self.assertIn("n2", spec["uncertainties"][0])

    def test_named_entities_are_left_alone(self):
        plan = self._plan({
            "modules": [{"id": "m1", "name": "Global Memory"}],
            "relations": [{"source": "m1", "target": "m1"}],
        })
        spec = plan["figure_specification"]

        self.assertEqual([item["id"] for item in spec["modules"]], ["m1"])
        self.assertFalse(spec.get("uncertainties"))
