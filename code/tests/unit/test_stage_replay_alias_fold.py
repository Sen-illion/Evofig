"""A replayed design seed must match the normalized contract."""

from __future__ import annotations

import unittest

from rfs.workflows.stage_replay import _fold_seed_aliases


def _seed() -> dict:
    return {
        "nodes": [
            {"id": "module_search", "parent_id": "stage_searching"},
            {"id": "mechanism_tg_mcts", "parent_id": "module_search"},
            {"id": "innovation_tool_guided_monte_carlo_tree_search_tg_mcts", "parent_id": "module_search"},
        ],
        "panels": [
            {"id": "stage_searching", "node_ids": ["module_search", "mechanism_tg_mcts", "innovation_tool_guided_monte_carlo_tree_search_tg_mcts"], "payload_node_ids": ["innovation_tool_guided_monte_carlo_tree_search_tg_mcts"]},
        ],
        "groups": [{"id": "g", "node_ids": ["innovation_tool_guided_monte_carlo_tree_search_tg_mcts"]}],
        "connectors": [
            {"id": "rel_a", "source": "module_search", "target": "innovation_tool_guided_monte_carlo_tree_search_tg_mcts"},
            {"id": "rel_b", "source": "innovation_tool_guided_monte_carlo_tree_search_tg_mcts", "target": "module_reviser"},
        ],
    }


ALIAS = "innovation_tool_guided_monte_carlo_tree_search_tg_mcts"
REMAP = {ALIAS: "mechanism_tg_mcts"}


class SeedAliasFoldTests(unittest.TestCase):
    def test_the_alias_node_is_removed_and_references_follow(self) -> None:
        seed = _seed()
        folded = _fold_seed_aliases(seed, REMAP)
        self.assertEqual(folded, [{"node_id": ALIAS, "merged_into": "mechanism_tg_mcts"}])
        self.assertEqual([node["id"] for node in seed["nodes"]], ["module_search", "mechanism_tg_mcts"])
        self.assertEqual(seed["panels"][0]["node_ids"], ["module_search", "mechanism_tg_mcts"])
        self.assertEqual(seed["panels"][0]["payload_node_ids"], ["mechanism_tg_mcts"])
        self.assertEqual(seed["groups"][0]["node_ids"], ["mechanism_tg_mcts"])
        self.assertEqual(seed["connectors"][0]["target"], "mechanism_tg_mcts")
        self.assertEqual(seed["connectors"][1]["source"], "mechanism_tg_mcts")

    def test_a_connector_that_collapses_into_a_self_loop_is_dropped(self) -> None:
        seed = _seed()
        seed["connectors"] = [{"id": "rel_self", "source": "mechanism_tg_mcts", "target": ALIAS}]
        _fold_seed_aliases(seed, REMAP)
        self.assertEqual(seed["connectors"], [])

    def test_without_merges_the_seed_is_untouched(self) -> None:
        seed = _seed()
        self.assertEqual(_fold_seed_aliases(seed, {}), [])
        self.assertEqual(len(seed["nodes"]), 3)


if __name__ == "__main__":
    unittest.main()
