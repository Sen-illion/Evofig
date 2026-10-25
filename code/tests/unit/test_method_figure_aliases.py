"""Alias folding keeps one card per concept in the method figure contract."""

from __future__ import annotations

import unittest

from rfs.contracts.method_figure import normalize_method_figure_content, validate_method_figure_content


def _payload() -> dict:
    return {
        "figure_goal": "Explain the method.",
        "stages": [
            {"id": "stage_searching", "name": "Searching Phase", "label": "Searching Phase", "role": "stage", "detail_level": "stage", "evidence_ids": ["E0"]},
            {"id": "stage_other", "name": "Other Phase", "label": "Other Phase", "role": "stage", "detail_level": "stage", "evidence_ids": ["E0"]},
        ],
        "modules": [
            {
                "id": "mechanism_tg_mcts",
                "name": "Tool-Guided Monte Carlo Tree Search (TG-MCTS)",
                "label": "TG-MCTS",
                "role": "mechanism",
                "parent_id": "module_search",
                "detail_level": "mechanism",
                "must_show": True,
                "should_render": True,
                "evidence_ids": ["E1"],
            },
            {
                "id": "module_search",
                "name": "Search",
                "label": "Search Algorithm",
                "role": "module",
                "parent_id": "stage_searching",
                "detail_level": "module",
                "evidence_ids": ["E2"],
            },
        ],
        "innovations": [
            {
                "id": "innovation_tool_guided_monte_carlo_tree_search_tg_mcts",
                "name": "Tool-Guided Monte Carlo Tree Search (TG-MCTS)",
                "label": "TG-MCTS",
                "role": "mechanism",
                "parent_id": "module_search",
                "detail_level": "mechanism",
                "should_render": True,
                "evidence_ids": ["E1"],
            },
            {
                "id": "innovation_evidence_gated_repair",
                "name": "Evidence-gated repair",
                "label": "Evidence-gated repair",
                "role": "innovation",
                "detail_level": "mechanism",
                "should_render": True,
                "evidence_ids": ["E3"],
            },
        ],
        "relations": [
            {"id": "rel_search_to_innovation", "source": "module_search", "target": "innovation_tool_guided_monte_carlo_tree_search_tg_mcts", "type": "data_flow"},
            {"id": "rel_tg_to_repair", "source": "mechanism_tg_mcts", "target": "innovation_evidence_gated_repair", "type": "data_flow"},
        ],
    }


class AliasFoldTests(unittest.TestCase):
    def test_duplicate_innovation_is_folded_into_the_declared_node(self) -> None:
        content = normalize_method_figure_content(_payload())
        ids = [item["id"] for item in content["innovations"]]
        self.assertEqual(ids, ["innovation_evidence_gated_repair"])
        modules = {item["id"]: item for item in content["modules"]}
        self.assertEqual(modules["mechanism_tg_mcts"]["evidence_ids"], ["E1"])
        merges = content["provenance"]["alias_merges"]
        self.assertEqual([(item["alias_id"], item["merged_into"]) for item in merges], [("innovation_tool_guided_monte_carlo_tree_search_tg_mcts", "mechanism_tg_mcts")])

    def test_references_to_the_alias_are_remapped(self) -> None:
        content = normalize_method_figure_content(_payload())
        by_id = {item["id"]: item for item in content["relations"]}
        self.assertEqual(by_id["rel_search_to_innovation"]["target"], "mechanism_tg_mcts")
        self.assertEqual(by_id["rel_tg_to_repair"]["source"], "mechanism_tg_mcts")

    def test_folded_contract_stays_valid_and_keeps_every_evidence_id(self) -> None:
        content = normalize_method_figure_content(_payload())
        self.assertTrue(validate_method_figure_content(content)["ok"])
        evidence = {value for item in content["modules"] for value in item["evidence_ids"]}
        self.assertEqual(evidence, {"E1", "E2"})

    def test_distinct_concepts_are_never_merged(self) -> None:
        payload = _payload()
        payload["modules"].append({"id": "module_reviser", "name": "Reviser", "label": "Reviser", "detail_level": "module", "evidence_ids": ["E4"]})
        payload["innovations"].append({"id": "innovation_reviser", "name": "Reviser", "label": "Reviser", "parent_id": "stage_other", "detail_level": "module", "evidence_ids": ["E5"]})
        content = normalize_method_figure_content(payload)
        self.assertIn("innovation_reviser", [item["id"] for item in content["innovations"]])


if __name__ == "__main__":
    unittest.main()
