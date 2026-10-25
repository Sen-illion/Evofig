"""Tests for the method inventory a Judge sample is allowed to read."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.judge_training.method_inventory import (
    FIGURE_CONTENT_PLAN_FILE,
    LEGACY_INVENTORY_FILE,
    LEGACY_PLAN_FILE,
    METHOD_INVENTORY_FILE,
    SELECTION_KEYS,
    inventory_path,
    method_inventory_view,
    plan_path,
    read_figure_content_plan,
    read_method_inventory,
)


def _write(directory: Path, name: str, payload: object) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _specification() -> dict:
    """One stored inventory: concepts the paper states, and the model's own selection."""

    return {
        "inputs": [{"id": "i1", "label": "raw frames", "role": "input", "detail_level": "artifact",
                    "scientific_importance": 1.0, "visualization_value": 1.0, "must_show": True,
                    "should_render": True, "render_priority": 1, "evidence_ids": ["E1"]}],
        "modules": [{"id": "m1", "label": "encoder", "role": "module", "detail_level": "module",
                     "scientific_importance": 0.5, "visualization_value": 0.5, "must_show": False,
                     "should_render": False, "render_priority": 2, "evidence_ids": ["E2"],
                     "parent_id": "m0", "input_ids": ["i1"], "output_ids": [],
                     "seed_statements": ["We encode frames with a shared encoder."]}],
        "relations": [{"id": "r1", "source": "i1", "target": "m1", "type": "data_flow",
                       "label": "frames feed the encoder", "evidence_ids": ["E1"],
                       "must_show": True, "should_render": True, "render_priority": 1,
                       "visualization_value": 1.0}],
        "figure_goal": "Show the frozen pipeline end to end.",
        "forbidden_inventions": ["a decoder this paper never trains"],
        "uncertainties": [],
        "overview_panels": [{"id": "p1", "must_show": True}],
        "seed": {"paper_id": "p0001", "specification_version": "frozen-figure-specification-v1"},
    }


def _selection_keys_in(value: object) -> list[str]:
    """Return every selection key reachable anywhere inside a sample block."""

    found: list[str] = []
    if isinstance(value, dict):
        found.extend(key for key in value if key in SELECTION_KEYS)
        for item in value.values():
            found.extend(_selection_keys_in(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_selection_keys_in(item))
    return found


class MethodInventoryViewTests(unittest.TestCase):
    def test_the_view_keeps_what_the_paper_states(self):
        view = method_inventory_view(_specification())

        self.assertEqual("Show the frozen pipeline end to end.", view["figure_goal"])
        self.assertEqual(["raw frames"], [item["label"] for item in view["inputs"]])
        self.assertEqual(["E2"], view["modules"][0]["evidence_ids"])
        self.assertEqual(["We encode frames with a shared encoder."], view["modules"][0]["seed_statements"])
        self.assertEqual(["frames feed the encoder"], [item["label"] for item in view["relations"]])
        self.assertEqual(["a decoder this paper never trains"], view["forbidden_inventions"])

    def test_the_view_drops_every_selection_the_figure_was_planned_with(self):
        view = method_inventory_view(_specification())

        self.assertEqual([], _selection_keys_in(view))
        self.assertNotIn("scientific_importance", view["modules"][0])
        self.assertNotIn("parent_id", view["modules"][0])
        self.assertNotIn("seed", view)
        self.assertNotIn("overview_panels", view)

    def test_the_view_is_idempotent(self):
        once = method_inventory_view(_specification())

        self.assertEqual(once, method_inventory_view(once))

    def test_an_absent_document_is_an_empty_view_and_not_a_skeleton(self):
        self.assertEqual({}, method_inventory_view({}))
        self.assertEqual({}, method_inventory_view(None))

    def test_a_plan_is_viewed_the_same_way_as_an_inventory(self):
        plan = _specification()
        plan["provenance"] = {"mode": "vlm"}

        self.assertEqual(method_inventory_view(_specification()), method_inventory_view(plan))


class MethodInventoryFileTests(unittest.TestCase):
    def test_the_earlier_inventory_name_is_still_read(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _write(directory, LEGACY_INVENTORY_FILE, {"figure_goal": "earlier"})

            self.assertEqual(LEGACY_INVENTORY_FILE, inventory_path(directory).name)
            self.assertEqual("earlier", read_method_inventory(directory)["figure_goal"])

    def test_the_plan_is_read_under_both_of_its_names(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _write(directory, LEGACY_PLAN_FILE, {"figure_goal": "earlier"})
            self.assertEqual(LEGACY_PLAN_FILE, plan_path(directory).name)
            self.assertEqual("earlier", read_figure_content_plan(directory)["figure_goal"])
            _write(directory, FIGURE_CONTENT_PLAN_FILE, {"figure_goal": "current"})

            self.assertEqual(FIGURE_CONTENT_PLAN_FILE, plan_path(directory).name)
            self.assertEqual("current", read_figure_content_plan(directory)["figure_goal"])
            self.assertIsNone(plan_path(directory / "absent"))
            self.assertEqual({}, read_figure_content_plan(directory / "absent"))

    def test_the_current_name_wins_over_the_earlier_one(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _write(directory, LEGACY_INVENTORY_FILE, {"figure_goal": "earlier"})
            _write(directory, METHOD_INVENTORY_FILE, {"figure_goal": "current"})

            self.assertEqual(METHOD_INVENTORY_FILE, inventory_path(directory).name)
            self.assertEqual("current", read_method_inventory(directory)["figure_goal"])

    def test_a_directory_holding_only_a_plan_answers_with_the_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _write(directory, LEGACY_PLAN_FILE, _specification())

            self.assertEqual(LEGACY_PLAN_FILE, inventory_path(directory).name)
            viewed = method_inventory_view(read_method_inventory(directory))
            self.assertEqual([], _selection_keys_in(viewed))
            self.assertEqual("Show the frozen pipeline end to end.", viewed["figure_goal"])

    def test_the_plan_is_not_answered_when_only_an_inventory_may_be_read(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _write(directory, LEGACY_PLAN_FILE, _specification())

            self.assertIsNone(inventory_path(directory, plan_fallback=False))
            self.assertEqual({}, read_method_inventory(directory, plan_fallback=False))

    def test_a_directory_holding_neither_states_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)

            self.assertIsNone(inventory_path(directory))
            self.assertEqual({}, read_method_inventory(directory))

    def test_unreadable_or_non_object_json_states_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / METHOD_INVENTORY_FILE).write_text("{not json", encoding="utf-8")
            self.assertEqual({}, read_method_inventory(directory))
            (directory / METHOD_INVENTORY_FILE).write_text("[1, 2]", encoding="utf-8")
            self.assertEqual({}, read_method_inventory(directory))


if __name__ == "__main__":
    unittest.main()
