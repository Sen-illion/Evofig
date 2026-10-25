"""Tests for routing retrieved Memory and for injecting what it routed.

Retrieval used to be the last word: whatever it returned was handed to the Judge and never
ran. These tests pin down the layer between the two -- rows routed by the stage they apply
at, competing rows resolved by one fixed precedence, and an injected action that keeps
only the object ids the paper in front of it still has.
"""

import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import ExperienceStore, inject_experience, seed_injection
from rfs.evolution.direct.routing import (
    boundary_tier,
    injection_action,
    median_gain,
    route_by_stage,
    route_experiences,
    support,
)
from rfs.workflows.paper_evolve_editable import _next_action_with_memory


def _experience(experience_id: str, *, action_type: str = "resize_nodes", apply_at: str = "ppt_compilation", gain: float = 0.25, cases: int = 1, forbidden: tuple[str, ...] = (), object_ids: tuple[str, ...] = ("module_01",), motifs: tuple[str, ...] = ("linear",)) -> dict:
    condition = {"topology": "pipeline", "node_count_bucket": "medium", "connector_count_bucket": "low"}
    return {
        "experience_id": experience_id,
        "status": "promoted",
        "domain": "direct_editable",
        "scope": "slot",
        "memory_type": "density",
        "apply_at": apply_at,
        "problem_type": "the panel is cramped",
        "condition": dict(condition),
        "context": dict(condition),
        "required_motifs": list(motifs),
        "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
        "hypothesis": f"enlarging the cards of a pipeline figure improves density fitness ({experience_id})",
        "action": {"type": action_type, "scope_type": "slot", "object_ids": list(object_ids), "factor": 1.12, "dx": 0.01, "dy": 0.0},
        "forbidden_changes": list(forbidden),
        "graph_scope": {"object_ids": list(object_ids)},
        "validated_cases": [f"{experience_id}_case_{index}.pdf" for index in range(cases)],
        "repair_gain": gain,
        "provenance": {"source": "direct_evolution_process", "run": f"{experience_id}.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": round(0.5 + gain, 4)},
        "outcome": {"accepted": True, "after_score": round(0.5 + gain, 4), "score_delta": gain},
        "strategy": {"instruction": f"enlarge the cards ({experience_id})", "action": {"type": action_type, "scope_type": "slot"}},
    }


class RoutingTests(unittest.TestCase):
    def _rows(self, temp: str, *records: dict) -> list[dict]:
        store = ExperienceStore(Path(temp))
        for record in records:
            appended = store.append_result(record)
            self.assertTrue(appended["ok"], appended)
        return store.retrieve(limit=5, context={"topology": "pipeline", "node_count": 6, "connector_count": 3})

    def test_rows_are_routed_to_the_stage_they_apply_at(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = self._rows(
                temp,
                _experience("e1", apply_at="ppt_compilation"),
                _experience("e2", apply_at="global_planning", action_type="relayout_global"),
            )
            routed = route_experiences(rows, stage="ppt_compilation")
            self.assertEqual(["e1"], [row["evidence"]["experience_id"] for row in routed["rows"]])
            self.assertEqual({"global_planning": 1, "ppt_compilation": 1}, routed["report"]["stages"])
            self.assertEqual(["global_planning", "ppt_compilation"], list(route_by_stage(rows)))
            self.assertEqual(["e1", "e2"], sorted(row["evidence"]["experience_id"] for row in rows))

    def test_a_declared_boundary_outranks_a_larger_measured_gain(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = self._rows(
                temp,
                _experience("e1", gain=0.25, forbidden=("scientific content",)),
                _experience("e2", action_type="reroute", gain=0.6),
            )
            routed = route_experiences(rows, stage="ppt_compilation", required_forbidden=["scientific content"])
            self.assertEqual(["e1"], routed["report"]["selected"])
            self.assertEqual([2, 0], [item["boundary_tier"] for item in routed["report"]["conflicts"]])
            self.assertIn("it declares the boundary this state requires", routed["report"]["reason"])

    def test_the_median_repair_gain_decides_when_the_boundaries_tie(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = self._rows(
                temp,
                _experience("e1", gain=0.25, forbidden=("scientific content",)),
                _experience("e2", action_type="reroute", gain=0.5, forbidden=("scientific content",)),
            )
            routed = route_experiences(rows, stage="ppt_compilation", required_forbidden=["scientific content"])
            self.assertEqual(["e2"], routed["report"]["selected"])
            self.assertEqual([0.5, 0.25], [item["median_repair_gain"] for item in routed["report"]["conflicts"]])

    def test_support_decides_when_boundary_and_gain_tie(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = self._rows(
                temp,
                _experience("e1", gain=0.25, cases=1),
                _experience("e2", action_type="reroute", gain=0.25, cases=3),
            )
            routed = route_experiences(rows, stage="ppt_compilation")
            self.assertEqual(["e2", "e1"], [item["experience_id"] for item in routed["report"]["conflicts"]])
            self.assertEqual([3, 1], [item["support"] for item in routed["report"]["conflicts"]])

    def test_an_exact_tie_is_broken_by_id_so_the_choice_repeats(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = self._rows(temp, _experience("e_b"), _experience("e_a", action_type="reroute"))
            first = route_experiences(rows, stage="ppt_compilation")
            second = route_experiences(rows, stage="ppt_compilation")
            self.assertEqual(["e_a"], first["report"]["selected"])
            self.assertEqual(first["report"], second["report"])

    def test_a_stage_without_any_experience_says_so(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = self._rows(temp, _experience("e1", apply_at="icon_generation", action_type="regen_assets"))
            routed = route_experiences(rows, stage="segmentation")
            self.assertEqual([], routed["rows"])
            self.assertEqual("no experience applies at segmentation", routed["report"]["reason"])
            self.assertEqual("no stage was named, so no experience could be routed", route_experiences(rows)["report"]["reason"])

    def test_the_precedence_reads_measurements_out_of_the_retrieval_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = self._rows(temp, _experience("e1", gain=0.25, cases=2, forbidden=("scientific content",)))
            row = rows[0]
            self.assertEqual(0.25, median_gain(row))
            self.assertEqual(2, support(row))
            self.assertEqual(2, boundary_tier(row, required=["scientific content"]))
            self.assertEqual(1, boundary_tier(row, required=["the panel titles"]))
            self.assertEqual(0, boundary_tier({"evidence": {}}))


class InjectionTests(unittest.TestCase):
    def _rows(self, temp: str, *records: dict) -> list[dict]:
        store = ExperienceStore(Path(temp))
        for record in records:
            self.assertTrue(store.append_result(record)["ok"])
        return store.retrieve(limit=5, context={"topology": "pipeline"})

    def test_an_injected_action_keeps_only_the_ids_the_paper_still_has(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = self._rows(temp, _experience("e1", object_ids=("module_01", "module_99"), forbidden=("scientific content",)))
            injection = inject_experience(rows, {"type": "resize_nodes", "scope_type": "slot", "object_ids": ["panel_a"]}, known_ids={"module_01"})
            action = injection["action"]
            self.assertEqual("resize_nodes", action["type"])
            self.assertEqual(["module_01"], action["object_ids"])
            self.assertEqual(["module_99"], injection["injection"]["dropped_object_ids"])
            self.assertEqual(["scientific content"], action["forbidden_changes"])
            self.assertEqual("e1", action["experience_id"])
            self.assertIn("enlarge the cards", action["problem"])
            self.assertEqual("e1", injection["injection"]["experience_id"])
            self.assertEqual("memory_retrieval", injection["injection"]["source"])
            self.assertEqual("ppt_compilation", injection["injection"]["stage"])
            self.assertEqual("promoted", injection["injection"]["status"])

    def test_an_experience_that_applies_elsewhere_is_not_injected(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = self._rows(temp, _experience("e1", apply_at="icon_generation", action_type="regen_assets"))
            injection = inject_experience(rows, {"type": "reroute", "scope_type": "connector"})
            self.assertIsNone(injection["action"])
            self.assertEqual("", injection["injection"]["experience_id"])
            self.assertEqual("none", injection["injection"]["source"])

    def test_an_empty_library_leaves_the_proposed_action_alone(self):
        injection = inject_experience([], {"type": "resize_nodes", "scope_type": "slot"})
        self.assertIsNone(injection["action"])
        self.assertEqual("no experience applies at ppt_compilation", injection["injection"]["reason"])

    def test_an_injected_action_carries_the_bounded_parameters_it_was_measured_with(self):
        with tempfile.TemporaryDirectory() as temp:
            record = _experience("e1")
            record["action"] = {**record["action"], "factor": 1.25}
            store = ExperienceStore(Path(temp))
            self.assertTrue(store.append_result(record)["ok"])
            rows = store.retrieve(limit=5, context={"topology": "pipeline"})
            action = inject_experience(rows, {"type": "resize_nodes"})["action"]
            self.assertEqual(1.25, action["factor"])

    def test_a_row_without_an_executable_action_is_not_injected(self):
        chosen = injection_action({"evidence": {"experience_id": "e1"}, "action": {"type": "initial"}})
        self.assertIsNone(chosen["action"])
        self.assertIn("no executable action", chosen["reason"])

    def test_a_seeded_run_reports_the_experience_and_stage_it_starts_from(self):
        seeded = seed_injection({"type": "regen_assets", "scope_type": "slot"}, "direct-exp-1")
        self.assertEqual({"experience_id": "direct-exp-1", "stage": "icon_generation", "source": "seed_experience", "status": "", "reason": "the run was seeded with this experience"}, seeded)
        self.assertEqual("same_domain_validated", seed_injection(None, "direct-exp-1", status="same_domain_validated")["status"])
        self.assertEqual("", seed_injection(None, "")["experience_id"])

    def test_the_round_loop_runs_the_experience_that_fits_its_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(Path(temp))
            self.assertTrue(store.append_result(_experience("e1", object_ids=("module_01", "module_99"))))
            rows = store.retrieve(limit=5, context={"topology": "pipeline"})
            layout = {"topology": "pipeline", "nodes": [{"id": "module_01"}]}
            action, injected, report = _next_action_with_memory(rows, {"type": "resize_nodes", "scope_type": "slot", "object_ids": ["module_01"]}, layout, {})
            self.assertEqual("e1", injected["experience_id"])
            self.assertEqual("memory_retrieval", injected["source"])
            self.assertEqual("e1", action["experience_id"])
            self.assertEqual(["module_01"], action["object_ids"])
            self.assertEqual("ppt_compilation", report["stage"])
            self.assertEqual("promoted", injected["status"])

    def test_the_shipped_default_budget_is_not_injected(self):
        with tempfile.TemporaryDirectory() as temp:
            record = _experience("e1", action_type="relayout_global", apply_at="global_planning")
            record["action"] = {"type": "relayout_global", "scope_type": "global", "factor": 1.02, "dx": 0.008, "dy": 0.0}
            store = ExperienceStore(Path(temp))
            self.assertTrue(store.append_result(record)["ok"])
            rows = store.retrieve(limit=5, context={"topology": "pipeline"})
            injection = inject_experience(rows, {"type": "relayout_global", "scope_type": "global"})
            self.assertIsNone(injection["action"])
            self.assertIn("default policy", injection["injection"]["reason"])

    def test_a_geometric_row_missing_budget_is_not_injected(self):
        with tempfile.TemporaryDirectory() as temp:
            record = _experience("e1")
            record["action"] = {"type": "resize_nodes", "scope_type": "slot", "object_ids": ["module_01"]}
            store = ExperienceStore(Path(temp))
            self.assertTrue(store.append_result(record)["ok"])
            rows = store.retrieve(limit=5, context={"topology": "pipeline"})
            injection = inject_experience(rows, {"type": "resize_nodes", "scope_type": "slot"})
            self.assertIsNone(injection["action"])
            self.assertIn("missing the budget", injection["injection"]["reason"])

    def test_the_round_loop_keeps_the_proposed_action_when_memory_has_nothing(self):
        proposed = {"type": "resize_nodes", "scope_type": "slot", "object_ids": ["module_01"]}
        action, injected, report = _next_action_with_memory([], proposed, {"nodes": [{"id": "module_01"}]}, {})
        self.assertEqual(proposed, action)
        self.assertEqual("", injected["experience_id"])
        self.assertEqual("no experience applies at ppt_compilation", injected["reason"])
        self.assertEqual([], report["selected"])


if __name__ == "__main__":
    unittest.main()
