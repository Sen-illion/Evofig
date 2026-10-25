"""Tests for ranking and routing-driven retrieval over the Memory store.

Retrieval used to score by counting equal condition keys, which returned a record without
saying why. These tests pin down the replacement: ranking by stage and structural
similarity, the components behind every hit, and an explicit report whenever the vector
part of the ranking could not be produced.
"""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.analysis.graph import cosine_similarity, histogram_similarity, jaccard_similarity
from rfs.evolution.direct import ExperienceStore, count_bucket


def _experience(experience_id: str, *, topology: str, motifs: tuple[str, ...], apply_at: str = "ppt_compilation", status: str = "promoted", nodes: int = 6, connectors: int = 4, gain: float = 0.25, action_type: str = "resize_nodes", guidance: str = "") -> dict:
    # Each fixture is a lesson of its own, so each says its lesson in its own words: two rows that
    # share an action *and* a sentence are the same lesson filed twice and the library folds them,
    # which is what these rows must not look like while the ranking is what is under test.
    guidance = guidance or f"Keep {experience_id} on the same reading path as the rest of the figure so the flow stays single-direction."
    condition = {"topology": topology, "node_count_bucket": count_bucket(nodes), "connector_count_bucket": count_bucket(connectors)}
    record = {
        "experience_id": experience_id,
        "status": status,
        "domain": "direct_editable",
        "scope": "slot",
        "memory_type": "density",
        "apply_at": apply_at,
        "problem_type": "the panel is cramped",
        "condition": dict(condition),
        "context": dict(condition),
        "required_motifs": list(motifs),
        "guidance": guidance,
        "hypothesis": f"enlarging the cards of a {topology} figure improves density fitness",
        "action": {"type": action_type, "scope_type": "slot"},
        "graph_scope": {"slot_ids": ["module_01"]},
        "validated_cases": [f"{experience_id}.pdf"],
        "repair_gain": gain,
        "provenance": {"source": "direct_evolution_process", "run": f"{experience_id}.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": round(0.5 + gain, 4)},
        "outcome": {"accepted": True, "after_score": round(0.5 + gain, 4), "score_delta": gain},
        "strategy": {"instruction": "enlarge the cards", "action": {"type": action_type, "scope_type": "slot"}},
    }
    if gain <= 0:
        record["excluded_conditions"] = [{**condition, "reason": "repair_gain_not_positive", "repair_gain": gain}]
    return record


class StoreRetrievalTests(unittest.TestCase):
    def _store(self, temp: str, *records: dict) -> ExperienceStore:
        store = ExperienceStore(Path(temp))
        for record in records:
            appended = store.append_result(record)
            self.assertTrue(appended["ok"], appended)
        return store

    def test_an_experience_with_the_same_topology_and_motifs_is_recalled(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(
                temp,
                _experience("e1", topology="pipeline", motifs=("linear", "feedback")),
                _experience("e2", topology="branching", motifs=("branch", "cross_panel"), gain=0.5),
            )
            result = store.retrieve_with_report(limit=5, context={"topology": "pipeline", "motifs": ["linear", "feedback"], "node_count": 6, "connector_count": 4})
            self.assertEqual(["e1", "e2"], [row["evidence"]["experience_id"] for row in result["rows"]])
            top = result["rows"][0]["evidence"]
            self.assertEqual(1.0, top["components"]["motif"])
            self.assertEqual(1.0, top["components"]["structure"])
            self.assertEqual(1.0, top["components"]["condition"])
            self.assertIn("ranked by", top["hit_reason"])
            self.assertEqual("ppt_compilation", result["rows"][0]["apply_at"])
            self.assertEqual(1, top["support"])
            self.assertEqual(0.25, top["repair_gain"])
            self.assertEqual("direct_evolution_process", top["provenance_source"])

    def test_an_experience_that_applies_at_another_stage_is_not_returned(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(
                temp,
                _experience("e1", topology="pipeline", motifs=("linear",), apply_at="ppt_compilation"),
                _experience("e2", topology="pipeline", motifs=("linear",), apply_at="global_planning"),
            )
            self.assertEqual(["e1"], [row["evidence"]["experience_id"] for row in store.retrieve(context={"topology": "pipeline"}, apply_at="ppt_compilation")])
            self.assertEqual(["e2"], [row["evidence"]["experience_id"] for row in store.retrieve(context={"topology": "pipeline"}, apply_at="global_planning")])
            self.assertEqual(2, len(store.retrieve(context={"topology": "pipeline"})))

    def test_a_size_bucket_neighbour_still_scores_but_a_foreign_state_does_not(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(
                temp,
                _experience("same", topology="pipeline", motifs=("linear",), nodes=4, connectors=4),
                _experience("neighbour", topology="pipeline", motifs=("linear",), nodes=6, connectors=6),
                _experience("far", topology="pipeline", motifs=("linear",), nodes=14, connectors=14),
            )
            rows = store.retrieve(context={"topology": "pipeline", "motifs": ["linear"], "node_count": 4, "connector_count": 4})
            self.assertEqual(["same", "neighbour", "far"], [row["evidence"]["experience_id"] for row in rows])
            self.assertEqual([1.0, 0.5, 0.0], [row["evidence"]["components"]["structure"] for row in rows])

    def test_the_default_set_is_the_three_rungs_a_pair_walked(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(
                temp,
                _experience("same", topology="pipeline", motifs=("linear",), status="same_domain_validated"),
                _experience("cross", topology="pipeline", motifs=("linear",), status="cross_domain_validated"),
                _experience("promoted", topology="pipeline", motifs=("linear",)),
                _experience("own_case", topology="pipeline", motifs=("linear",), status="current_case_validated"),
            )
            recalled = store.retrieve(context={"topology": "pipeline"})
            self.assertEqual({"same", "cross", "promoted"}, {row["evidence"]["experience_id"] for row in recalled})
            self.assertEqual(["cross_domain_validated", "domain_validated", "promoted", "same_domain_validated"], store.retrieve_with_report(context={"topology": "pipeline"})["report"]["statuses"])
            # A caller that names one rung still gets only that rung.
            self.assertEqual(["promoted"], [row["evidence"]["experience_id"] for row in store.retrieve(context={"topology": "pipeline"}, statuses={"promoted"})])

    def test_a_refuted_experience_is_never_injected_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(
                temp,
                _experience("good", topology="pipeline", motifs=("linear",)),
                _experience("bad", topology="pipeline", motifs=("linear",), status="refuted", gain=-0.25),
            )
            self.assertEqual(["good"], [row["evidence"]["experience_id"] for row in store.retrieve(context={"topology": "pipeline"}, statuses={"promoted"})])
            refuted = store.retrieve(context={"topology": "pipeline"}, statuses={"refuted"})
            # The refuted row is asked for, and it still does not come back: it names the
            # very condition this query is in, so it is excluded here rather than reused.
            # The report says so instead of dropping it silently.
            self.assertEqual([], [row["evidence"]["experience_id"] for row in refuted])
            report = store.retrieve_with_report(context={"topology": "pipeline"}, statuses={"refuted"})["report"]
            self.assertEqual(["bad"], [row["experience_id"] for row in report["excluded_rows"]])
            self.assertEqual(["repair_gain_not_positive"], report["excluded_rows"][0]["reasons"])
            # A query in a state the row was not refuted under still reads it.
            elsewhere = store.retrieve(context={"topology": "branching"}, statuses={"refuted"})
            self.assertEqual(["bad"], [row["evidence"]["experience_id"] for row in elsewhere])
            self.assertEqual(-0.25, elsewhere[0]["evidence"]["repair_gain"])

    def test_a_condition_appended_by_an_event_keeps_a_promoted_row_out(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(temp, _experience("e1", topology="pipeline", motifs=("linear",)))
            store.append_event("e1", "experience_condition_excluded", condition={"topology": "pipeline", "reason": "reflection_next_policy_narrow_scope"}, reason="reflection_next_policy_narrow_scope", round=1)
            result = store.retrieve_with_report(context={"topology": "pipeline", "node_count": 6, "connector_count": 4})
            self.assertEqual([], [row["evidence"]["experience_id"] for row in result["rows"]])
            self.assertEqual(["e1"], [row["experience_id"] for row in result["report"]["excluded_rows"]])
            self.assertEqual(["reflection_next_policy_narrow_scope"], result["report"]["excluded_rows"][0]["reasons"])
            # The prompt's own row is untouched: the exclusion lives in the event log.
            self.assertEqual("promoted", store.get("e1")["status"])
            # A state the exclusion does not name still gets the row.
            other = store.retrieve(context={"topology": "branching", "node_count": 6, "connector_count": 4})
            self.assertEqual(["e1"], [row["evidence"]["experience_id"] for row in other])

    def test_a_domain_exclusion_is_honoured_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(temp, _experience("e1", topology="pipeline", motifs=("linear",)))
            store.append_event("e1", "experience_condition_excluded", condition={"dimension": "domain", "value": "biology", "reason": "reflection_domain_mismatch"})
            inside = store.retrieve_with_report(context={"topology": "pipeline", "domain": "biology"})
            self.assertEqual([], [row["evidence"]["experience_id"] for row in inside["rows"]])
            self.assertEqual(["reflection_domain_mismatch"], inside["report"]["excluded_rows"][0]["reasons"])
            outside = store.retrieve(context={"topology": "pipeline", "domain": "materials"})
            self.assertEqual(["e1"], [row["evidence"]["experience_id"] for row in outside])

    def test_the_vector_mode_names_its_embedder_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(temp, _experience("e1", topology="pipeline", motifs=("linear",)))
            report = store.retrieve_with_report(context={"topology": "pipeline", "problem": "the panel is cramped"})["report"]
            self.assertEqual("structure_and_embedding", report["mode"])
            self.assertFalse(report["degraded"])
            self.assertEqual("hashing_ngram", report["embedder"]["kind"])
            self.assertEqual(256, report["embedder"]["fingerprint"]["dimension"])
            self.assertEqual(1, report["embedded_rows"])

    def test_an_unavailable_embedder_is_reported_and_never_raises(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(temp, _experience("e1", topology="pipeline", motifs=("linear",)))
            result = store.retrieve_with_report(context={"topology": "pipeline", "problem": "cramped"}, embedder="st:C:/nowhere/absent-model")
            self.assertEqual("structure_only", result["report"]["mode"])
            self.assertTrue(result["report"]["degraded"])
            self.assertEqual("unavailable", result["report"]["embedder"]["kind"])
            self.assertIn("absent-model", result["report"]["embedder"]["reason"])
            self.assertEqual(["e1"], [row["evidence"]["experience_id"] for row in result["rows"]])
            self.assertIsNone(result["rows"][0]["evidence"]["components"]["embedding"])

    def test_a_silent_query_ranks_by_structure_and_says_so(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(temp, _experience("e1", topology="pipeline", motifs=("linear",)))
            result = store.retrieve_with_report(context={"topology": "pipeline"})
            row = result["rows"][0]["evidence"]
            self.assertIsNone(row["components"]["motif"])
            self.assertEqual(1.0, row["components"]["condition"])
            self.assertNotIn("motif", row["hit_reason"])

    def test_an_unsupported_domain_is_reported_instead_of_ranking_foreign_memory(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(temp, _experience("e1", topology="pipeline", motifs=("linear",)))
            result = store.retrieve_with_report(domain="other_domain", context={"topology": "pipeline"})
            self.assertEqual([], result["rows"])
            self.assertEqual("unsupported_domain", result["report"]["mode"])

    def test_retrieval_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(
                temp,
                _experience("e1", topology="pipeline", motifs=("linear", "feedback")),
                _experience("e2", topology="pipeline", motifs=("linear",)),
            )
            context = {"topology": "pipeline", "motifs": ["linear"], "node_count": 6}
            first = json.dumps(store.retrieve(context=context), ensure_ascii=False, sort_keys=True)
            second = json.dumps(store.retrieve(context=context), ensure_ascii=False, sort_keys=True)
            self.assertEqual(first, second)

    def test_a_domain_validated_row_is_injected_only_in_its_domain(self):
        row = _experience("scoped", topology="pipeline", motifs=("linear",), status="domain_validated")
        row["applicability"] = "domain"
        row["applicability_domain"] = "ai_method"
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(temp, row)
            matching = store.retrieve(context={"topology": "pipeline", "domain": "ai_method"})
            missing = store.retrieve(context={"topology": "pipeline", "domain": "", "domain_missing": True})
            other = store.retrieve(context={"topology": "pipeline", "domain": "biology"})
            self.assertEqual(["scoped"], [item["evidence"]["experience_id"] for item in matching])
            self.assertEqual([], missing)
            self.assertEqual([], other)
            report = store.retrieve_with_report(context={"topology": "pipeline", "domain": "biology"})
            self.assertIn("domain applicability does not match this paper", report["report"]["excluded_rows"][0]["reasons"])


class PublicSimilarityTests(unittest.TestCase):
    def test_the_similarity_measures_the_graph_index_uses_are_public(self):
        self.assertEqual(1.0, cosine_similarity([1.0, 0.0], [1.0, 0.0]))
        self.assertEqual(0.0, cosine_similarity([1.0, 0.0], [0.0, 1.0]))
        self.assertEqual(0.0, cosine_similarity([1.0], [1.0, 2.0]))
        self.assertEqual(0.5, jaccard_similarity({"linear", "feedback"}, {"linear"}))
        self.assertEqual(1.0, jaccard_similarity([], []))
        self.assertEqual(0.5, histogram_similarity({"linear": 1}, {"linear": 1, "feedback": 1}))
        self.assertEqual(0.25, histogram_similarity({"linear": 4}, {"linear": 1}))


if __name__ == "__main__":
    unittest.main()
