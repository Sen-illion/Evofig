"""One lesson, many conditions: a restated repair grows a row's tables instead of the library.

The library is read as a set of lessons rather than as a log, so the same move under the same
prohibition measured twice is one lesson with two witnesses. These tests hold the two edges of
that: the restatement is folded and traceable, and a row that won while carrying a new defect is
never folded, because that record is the one thing the ladder has to be able to see about it.
"""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import ExperienceStore


def row(experience_id: str, *, condition: dict | None = None, carried_new_defect: bool = False) -> dict:
    return {
        "experience_id": experience_id,
        "domain": "direct_editable",
        "scope": "global",
        "apply_at": "ppt_compilation",
        "memory_type": "connector",
        "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
        "hypothesis": "a shared vertical channel keeps a one-to-many fan-out readable",
        "action": {"type": "reroute", "constraints": {"shared_vertical_channel": True}},
        "condition": dict(condition or {"topology": "pipeline", "node_count_bucket": "high"}),
        "graph_scope": {"edge_ids": ["rel_12"]},
        "accepted": True,
        "repair_gain": 0.12,
        "status": "candidate",
        "applicability": "common",
        "carried_new_defect": bool(carried_new_defect),
        "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.62},
    }


class RestatedLessonTests(unittest.TestCase):
    def test_a_restated_move_is_folded_and_the_condition_stays_readable(self):
        with tempfile.TemporaryDirectory() as name:
            store = ExperienceStore(Path(name))
            first = store.append_result(row("direct-exp-first"))
            second = store.append_result(row("direct-exp-second", condition={"topology": "fan_out"}))

            self.assertTrue(first["ok"], first)
            self.assertTrue(second["ok"], second)
            self.assertEqual("merged_into_existing", second["reason"])
            self.assertEqual("direct-exp-first", second["experience_id"])
            self.assertEqual("direct-exp-second", second["merged_from"])
            self.assertEqual(1, len(store.records()))

            folded = store.merged_witnesses()["direct-exp-first"]
            self.assertEqual(1, len(folded))
            self.assertEqual({"topology": "fan_out"}, folded[0]["condition"])
            self.assertEqual("direct-exp-second", folded[0]["merged_from"])

    def test_a_different_prohibition_is_a_lesson_of_its_own(self):
        with tempfile.TemporaryDirectory() as name:
            store = ExperienceStore(Path(name))
            store.append_result(row("direct-exp-first"))
            distinct = row("direct-exp-second")
            distinct["forbidden_changes"] = ["resize_nodes"]
            kept = store.append_result(distinct)

            self.assertTrue(kept["ok"], kept)
            self.assertEqual("appended", kept["reason"])
            self.assertEqual(2, len(store.records()))

    def test_a_row_that_won_with_a_new_defect_is_never_folded(self):
        with tempfile.TemporaryDirectory() as name:
            store = ExperienceStore(Path(name))
            store.append_result(row("direct-exp-first"))
            wounded = store.append_result(row("direct-exp-wounded", carried_new_defect=True))

            self.assertTrue(wounded["ok"], wounded)
            self.assertEqual("appended", wounded["reason"])
            self.assertEqual(2, len(store.records()))
            self.assertEqual({}, store.merged_witnesses())


if __name__ == "__main__":
    unittest.main()