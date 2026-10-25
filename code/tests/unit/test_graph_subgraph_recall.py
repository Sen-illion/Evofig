"""Tests for subgraph level recall: vector recall plus the structural rerank."""

from __future__ import annotations

import unittest

from rfs.analysis.graph.retrieval import (
    rank_subgraph_candidates,
    vector_recall_subgraphs,
)
from rfs.providers.embeddings import resolve_embedder


ENCODER = resolve_embedder()


def locator(subgraph_id, paper_id, *, layer="scientific", motif="linear", nodes=3, edges=2, blob=""):
    return {
        "subgraph_id": subgraph_id,
        "paper_id": paper_id,
        "graph_kind": layer,
        "motif": motif,
        "node_count": nodes,
        "edge_count": edges,
        "split": "train",
        "domain": "cv",
        "embedding": ENCODER.encode([blob or f"{paper_id} {motif} {nodes} {edges}"])[0],
    }


class SubgraphRecallTests(unittest.TestCase):
    def setUp(self):
        self.query = locator("subgraph.q", "p0001", blob="encoder decoder chain")
        self.candidates = [
            locator("subgraph.a", "p0002", blob="encoder decoder chain"),
            locator("subgraph.b", "p0003", motif="merge", blob="stream fusion report"),
            locator("subgraph.c", "p0003", layer="visual", blob="panel block arrow"),
        ]

    def test_recall_never_returns_the_query_paper_or_the_query_itself(self):
        recalled = vector_recall_subgraphs(self.query, [self.query, *self.candidates], limit=10, same_layer_only=False)

        self.assertEqual(sorted(item["subgraph_id"] for item in recalled), ["subgraph.a", "subgraph.b", "subgraph.c"])
        self.assertTrue(all(item["paper_id"] != "p0001" for item in recalled))

    def test_exclude_paper_ids_removes_a_whole_paper(self):
        recalled = vector_recall_subgraphs(self.query, self.candidates, limit=10, exclude_paper_ids=("p0003",))

        self.assertEqual([item["subgraph_id"] for item in recalled], ["subgraph.a"])

    def test_same_layer_only_drops_the_other_layer(self):
        recalled = vector_recall_subgraphs(self.query, self.candidates, limit=10, same_layer_only=True)

        self.assertEqual([item["subgraph_id"] for item in recalled], ["subgraph.a", "subgraph.b"])

    def test_recall_limit_caps_the_pool(self):
        self.assertEqual(1, len(vector_recall_subgraphs(self.query, self.candidates, limit=1)))

    def test_a_missing_embedding_is_a_loud_failure(self):
        broken = locator("subgraph.d", "p0004")
        broken["embedding"] = []
        with self.assertRaises(ValueError) as caught:
            vector_recall_subgraphs(self.query, [broken], limit=5)
        self.assertIn("subgraph.d", str(caught.exception))

        query = locator("subgraph.q2", "p0001")
        query["embedding"] = []
        with self.assertRaises(ValueError) as caught:
            vector_recall_subgraphs(query, self.candidates, limit=5)
        self.assertIn("subgraph.q2", str(caught.exception))

    def test_ranking_rewards_a_matching_motif_and_size(self):
        ranked = rank_subgraph_candidates(self.query, self.candidates, top_k=3)

        self.assertEqual(ranked[0]["locator"]["subgraph_id"], "subgraph.a")
        self.assertTrue(ranked[0]["same_motif"])
        self.assertEqual(ranked[0]["selection_reason"], "similar_subgraph_structure")
        mismatch = next(item for item in ranked if not item["same_motif"])
        self.assertEqual(mismatch["selection_reason"], "vector_neighbour_with_another_structure")
        self.assertGreater(ranked[0]["score"], mismatch["score"])
        for item in ranked:
            self.assertIn("embedding_similarity", item)
            self.assertIn("layer_match", item)

    def test_ranking_reads_the_flat_locator_fields_when_there_is_no_signature(self):
        ranked = rank_subgraph_candidates(self.query, [locator("subgraph.e", "p0005", motif="linear", nodes=3, edges=2)], top_k=1)

        self.assertTrue(ranked[0]["same_motif"])
        self.assertEqual(ranked[0]["size_similarity"], 1.0)

    def test_an_unknown_motif_can_never_match(self):
        query = locator("subgraph.q3", "p0001", motif="unknown", nodes=1, edges=0)
        ranked = rank_subgraph_candidates(query, [locator("subgraph.f", "p0006", motif="unknown", nodes=1, edges=0)], top_k=1)

        self.assertFalse(ranked[0]["same_motif"])
        self.assertTrue(ranked[0]["same_layer"])
        self.assertEqual(ranked[0]["size_similarity"], 1.0)

    def test_the_graph_package_exposes_the_subgraph_recall_entry_points(self):
        import rfs.analysis.graph as graph

        self.assertIs(graph.rank_subgraph_candidates, rank_subgraph_candidates)
        self.assertIs(graph.vector_recall_subgraphs, vector_recall_subgraphs)
        for name in ("rank_subgraph_candidates", "vector_recall_subgraphs"):
            self.assertIn(name, graph.__all__)


if __name__ == "__main__":
    unittest.main()
