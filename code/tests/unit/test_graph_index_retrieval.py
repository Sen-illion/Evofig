"""Tests for local graph indexing and diverse structural retrieval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rfs.analysis.graph import GraphIndex, build_graph_index_record, rank_validation_cases
from rfs.contracts import normalize_graph


def _record(paper_id: str, domain: str, tags: list[str], edges: list[tuple[str, str]]) -> dict:
    nodes = sorted({value for edge in edges for value in edge})
    graph = normalize_graph({
        "graph_id": f"g-{paper_id}", "paper_id": paper_id, "kind": "scientific",
        "nodes": [{"id": item} for item in nodes],
        "edges": [{"id": f"e{i}", "source": source, "target": target} for i, (source, target) in enumerate(edges)],
    })
    return build_graph_index_record(graph, domain=domain, content_tags=tags)


class GraphIndexRetrievalTests(unittest.TestCase):
    def test_jsonl_index_upsert_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            index = GraphIndex(Path(temp) / "graphs.jsonl")
            first = _record("p1", "cv", ["vision"], [("a", "b"), ("b", "c")])
            index.upsert(first)
            index.upsert(first)
            self.assertEqual([first], index.records())

    def test_retrieval_prefers_same_logic_with_cross_domain_content(self) -> None:
        query = _record("query", "cv", ["vision", "encoder"], [("a", "b"), ("b", "c")])
        cross_domain = _record("bio", "biology", ["cell", "assay"], [("x", "y"), ("y", "z")])
        same_content = _record("cv2", "cv", ["vision", "encoder"], [("x", "y"), ("y", "z")])
        wrong_structure = _record("branch", "biology", ["cell"], [("x", "y"), ("x", "z"), ("x", "w")])
        ranked = rank_validation_cases(query, [wrong_structure, same_content, cross_domain], top_k=3)
        self.assertEqual("bio", ranked[0]["record"]["paper_id"])
        self.assertTrue(ranked[0]["cross_domain"])
        self.assertGreater(ranked[0]["structure_similarity"], ranked[-1]["structure_similarity"])


if __name__ == "__main__":
    unittest.main()
