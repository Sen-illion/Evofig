"""Tests for vector recall in front of the structural validation-case ranking."""

import unittest

from rfs.analysis.graph import (
    build_graph_index_record,
    rank_validation_cases,
    rank_validation_cases_with_recall,
    vector_recall,
)
from rfs.providers.embeddings import resolve_embedder


ENCODER = resolve_embedder()
FAN_IN = ("Stream A", "Stream B", "Fusion", "Report")


def _graph(paper_id, labels, *, kind="scientific"):
    nodes = [{"id": f"n{index}", "kind": "module", "label": label, "role": "module", "required": True} for index, label in enumerate(labels)]
    edges = [{"id": f"e{index}", "source": f"n{index}", "target": f"n{index + 1}", "relation_type": "data_flow", "required": True} for index in range(len(nodes) - 1)]
    return {
        "graph_id": f"{kind}-{paper_id}",
        "paper_id": paper_id,
        "kind": kind,
        "nodes": nodes,
        "edges": edges,
        "required_node_ids": [item["id"] for item in nodes],
        "required_edge_ids": [item["id"] for item in edges],
    }


def _record(paper_id, text, *, domain="cv", tags=(), labels=FAN_IN, embedding=True):
    return build_graph_index_record(
        _graph(paper_id, labels),
        domain=domain,
        content_tags=tags,
        embedding=ENCODER.encode([text])[0] if embedding else None,
    )


class VectorRecallTests(unittest.TestCase):
    def setUp(self):
        self.query = _record("p0001", "stream fusion report", tags=("figure_type:framework",))
        self.same = _record("p0002", "stream fusion report", domain="biology", tags=("figure_type:framework",))
        self.other = _record("p0003", "gradient boosted trees", domain="materials", labels=("Features", "Boosted Trees", "Score"))

    def test_recall_keeps_the_most_similar_record(self):
        recalled = vector_recall(self.query, [self.other, self.same], limit=1)

        self.assertEqual([item["paper_id"] for item in recalled], ["p0002"])

    def test_recall_is_deterministic_and_bounded(self):
        first = vector_recall(self.query, [self.same, self.other], limit=2)
        second = vector_recall(self.query, [self.other, self.same], limit=2)

        self.assertEqual([item["record_id"] for item in first], [item["record_id"] for item in second])
        self.assertEqual(len(vector_recall(self.query, [self.same, self.other], limit=9)), 2)

    def test_recall_drops_the_query_paper_and_the_named_papers(self):
        recalled = vector_recall(self.query, [self.query, self.same, self.other], limit=5, exclude_paper_ids=["p0003"])

        self.assertEqual([item["paper_id"] for item in recalled], ["p0002"])

    def test_recall_refuses_a_record_without_an_embedding(self):
        with self.assertRaises(ValueError) as caught:
            vector_recall(self.query, [_record("p0003", "gradient boosted trees", embedding=False)], limit=1)

        self.assertIn("embedding", str(caught.exception))

    def test_recall_refuses_a_query_without_an_embedding(self):
        with self.assertRaises(ValueError):
            vector_recall(_record("p0001", "stream fusion report", embedding=False), [self.same], limit=1)


class RecallThenRerankTests(unittest.TestCase):
    def setUp(self):
        self.query = _record("p0001", "stream fusion report", tags=("figure_type:framework",))
        self.candidates = [
            _record("p0002", "stream fusion report", domain="biology", tags=("figure_type:framework",)),
            _record("p0003", "stream fusion report", domain="materials", tags=("figure_type:flowchart",), labels=("Stream A", "Stream B", "Fusion", "Summary")),
            _record("p0004", "gradient boosted trees", domain="cv"),
        ]

    def test_a_full_recall_reproduces_the_plain_structural_ranking(self):
        reranked = rank_validation_cases_with_recall(self.query, self.candidates, top_k=3, recall_limit=len(self.candidates))
        plain = rank_validation_cases(self.query, self.candidates, top_k=3)

        self.assertEqual([item["record"]["record_id"] for item in reranked], [item["record"]["record_id"] for item in plain])
        self.assertEqual([item["score"] for item in reranked], [item["score"] for item in plain])

    def test_a_narrow_recall_only_narrows_the_pool(self):
        recalled = vector_recall(self.query, self.candidates, limit=2)
        reranked = rank_validation_cases_with_recall(self.query, self.candidates, top_k=2, recall_limit=2)
        expected = rank_validation_cases(self.query, recalled, top_k=2)

        self.assertEqual([item["record"]["record_id"] for item in reranked], [item["record"]["record_id"] for item in expected])
        self.assertEqual([item["score"] for item in reranked], [item["score"] for item in expected])
        self.assertLessEqual(len(reranked), 2)


if __name__ == "__main__":
    unittest.main()
