"""Unit tests for the deterministic cross-layer binding of one Pilot paper."""

import unittest
from pathlib import Path

from rfs.contracts import stable_graph_id, validate_graph_bindings
from rfs.judge_training.pilot_bindings import (
    BINDING_VERSION,
    FIGURE_CLAIMS_CONTRACT,
    FIGURE_CLAIMS_VERSION,
    SEMANTIC_MATCH_METHOD,
    build_layer_bindings,
    label_score,
    label_tokens,
    local_object_role,
    match_layer_edges,
    match_layer_nodes,
    normalize_label,
    object_is_unlabeled,
    reader_object_role,
    resolve_binding_embedder,
    structure_violations,
)
from rfs.providers.embeddings import EmbeddingUnavailable, LocalSentenceTransformerEmbedder


def scientific_graph(*, nodes, edges=None, required_nodes=None, required_edges=None):
    return {
        "graph_id": "scientific-1",
        "paper_id": "p0001",
        "kind": "scientific",
        "nodes": nodes,
        "edges": edges or [],
        "required_node_ids": required_nodes if required_nodes is not None else [node["id"] for node in nodes],
        "required_edge_ids": required_edges if required_edges is not None else [edge["id"] for edge in edges or []],
    }


def visual_graph(*, nodes, edges=None):
    return {
        "graph_id": "visual-1",
        "paper_id": "p0001",
        "kind": "visual",
        "nodes": nodes,
        "edges": edges or [],
    }


def sci_node(node_id, label, kind="module"):
    return {"id": node_id, "kind": kind, "label": label}


def vis_block(node_id, label, panels=("panel-1",)):
    return {"id": node_id, "kind": "module", "label": label, "panel_ids": list(panels)}


def vis_panel(node_id, label):
    return {"id": node_id, "kind": "panel", "label": label, "panel_ids": []}


class LabelTests(unittest.TestCase):
    def test_labels_fold_into_a_comparable_form(self):
        self.assertEqual(normalize_label("  Frozen  Encoder!! "), "frozen encoder")
        self.assertEqual(normalize_label(""), "")
        self.assertEqual(normalize_label(None), "")
        self.assertEqual(label_tokens("frozen encoder"), {"frozen", "encoder"})
        self.assertEqual(label_tokens("a b"), set())

    def test_scores_rank_exact_above_containment_above_overlap(self):
        self.assertEqual(label_score("Frozen encoder", "FROZEN ENCODER"), 1.0)
        self.assertEqual(label_score("Frozen encoder", "Frozen encoder block"), 0.85)
        self.assertAlmostEqual(label_score("Frozen encoder", "Frozen decoder"), 1 / 3, places=6)
        self.assertEqual(label_score("Frozen encoder", ""), 0.0)
        self.assertEqual(label_score("Encoder", "Decoder"), 0.0)


class NodeMatchTests(unittest.TestCase):
    def test_matching_prefers_the_highest_score_and_uses_every_block_once(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Frozen encoder"), sci_node("s2", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Frozen encoder")])
        matches, audit = match_layer_nodes(scientific, visual)

        self.assertEqual(matches["s1"]["visual_node_id"], "b1")
        self.assertNotIn("s2", matches)
        self.assertEqual(audit["matched_visual_blocks"], 1)
        self.assertEqual(audit["visual_blocks"], 1)

    def test_a_collision_is_reported_while_the_free_block_still_binds(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Encoder two")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Encoder")])
        matches, audit = match_layer_nodes(scientific, visual)

        self.assertEqual(matches["s1"]["visual_node_id"], "b1")
        self.assertEqual(matches["s2"]["visual_node_id"], "b2")
        self.assertEqual([item["visual_node_id"] for item in audit["skipped_duplicate_blocks"]], ["b1"])
        self.assertEqual(audit["skipped_duplicate_blocks"][0]["reason"], "block_already_bound")

    def test_one_block_never_carries_two_scientific_nodes(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder")])
        matches, audit = match_layer_nodes(scientific, visual)

        self.assertEqual(list(matches), ["s1"])
        self.assertEqual([item["scientific_node_id"] for item in audit["skipped_duplicate_blocks"]], ["s2"])

    def test_unlabeled_blocks_are_reported_and_never_matched(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Frozen encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Frozen encoder"), vis_block("b2", "")])
        matches, audit = match_layer_nodes(scientific, visual)

        self.assertEqual(matches["s1"]["visual_node_id"], "b1")
        self.assertEqual(audit["unlabeled_visual_blocks"], ["b2"])

    def test_labels_below_the_weak_threshold_do_not_bind(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Frozen encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Optimizer")])
        matches, _ = match_layer_nodes(scientific, visual)

        self.assertEqual(matches, {})

    def test_weaker_pairs_are_kept_for_the_audit(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder")])
        matches, _ = match_layer_nodes(scientific, visual)

        self.assertEqual(matches["s1"]["score"], 1.0)


class EdgeMatchTests(unittest.TestCase):
    def nodes(self):
        return scientific_graph(
            nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Decoder")],
            edges=[{"id": "e1", "source": "s1", "target": "s2", "relation_type": "data_flow"}],
        )

    def visual(self, *, source="b1", target="b2", relation="forward"):
        return visual_graph(
            nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Decoder")],
            edges=[{"id": "v1", "source": source, "target": target, "relation_type": relation}],
        )

    def matches(self, visual):
        scientific = self.nodes()
        node_matches, _ = match_layer_nodes(scientific, visual)
        return match_layer_edges(scientific, visual, node_matches)

    def test_a_forward_arrow_binds(self):
        matches, audit = self.matches(self.visual())
        self.assertEqual(matches["e1"]["visual_edge_ids"], ["v1"])
        self.assertEqual(matches["e1"]["direction"], "forward")
        self.assertEqual(audit["reversed_edges"], [])

    def test_a_backwards_arrow_is_reported_not_hidden(self):
        matches, audit = self.matches(self.visual(source="b2", target="b1"))
        self.assertEqual(matches["e1"]["direction"], "reversed")
        self.assertEqual(audit["reversed_edges"], ["e1"])

    def test_feedback_versus_forward_disagreement_is_reported(self):
        matches, audit = self.matches(self.visual(relation="feedback"))
        self.assertEqual(matches["e1"]["direction"], "forward")
        self.assertEqual(len(audit["relation_type_mismatches"]), 1)
        self.assertEqual(audit["relation_type_mismatches"][0]["scientific_edge_id"], "e1")

    def test_a_relation_with_an_unbound_endpoint_is_not_matched(self):
        scientific = scientific_graph(
            nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Not drawn")],
            edges=[{"id": "e1", "source": "s1", "target": "s2", "relation_type": "data_flow"}],
        )
        visual = self.visual()
        node_matches, _ = match_layer_nodes(scientific, visual)
        matches, _ = match_layer_edges(scientific, visual, node_matches)

        self.assertEqual(matches, {})


class BindingSetTests(unittest.TestCase):
    def build(self):
        scientific = scientific_graph(
            nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Decoder"), sci_node("s3", "Loss")],
            edges=[{"id": "e1", "source": "s1", "target": "s2", "relation_type": "data_flow"}],
        )
        visual = visual_graph(
            nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Decoder", panels=("panel-2",)), vis_block("b3", "Optimizer")],
            edges=[{"id": "v1", "source": "b1", "target": "b2", "relation_type": "forward"}],
        )
        return build_layer_bindings(scientific, visual)

    def test_bindings_follow_the_frozen_contract(self):
        bindings, report = self.build()
        validation = validate_graph_bindings(bindings, scientific_graph=scientific_graph(
            nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Decoder"), sci_node("s3", "Loss")],
            edges=[{"id": "e1", "source": "s1", "target": "s2", "relation_type": "data_flow"}],
        ))

        self.assertEqual(bindings["contract"], "rfs.graph_bindings")
        self.assertEqual(bindings["paper_id"], "p0001")
        self.assertTrue(validation["ok"], validation["errors"])
        self.assertEqual(bindings["binding_set_id"], stable_graph_id("graph_bindings", "p0001"))

    def test_status_reports_what_is_drawn_and_what_is_missing(self):
        bindings, report = self.build()
        statuses = {item["scientific_node_id"]: item["status"] for item in bindings["node_bindings"]}

        self.assertEqual(statuses["s1"], "bound")
        self.assertEqual(statuses["s2"], "bound")
        self.assertEqual(statuses["s3"], "unbound")
        self.assertEqual(report["node_statuses"], {"bound": 2, "unbound": 1})
        self.assertEqual(report["required_nodes"], 3)
        self.assertEqual(report["required_nodes_unbound"], ["s3"])
        self.assertEqual(report["diagnostic_required_node_coverage"], round(2 / 3, 4))
        self.assertEqual(report["diagnostic_required_edge_coverage"], 1.0)
        self.assertEqual(report["provider_calls"], 0)

    def test_panels_of_a_bound_block_are_carried_over(self):
        bindings, _ = self.build()
        by_node = {item["scientific_node_id"]: item for item in bindings["node_bindings"]}

        self.assertEqual(by_node["s1"]["panel_ids"], ["panel-1"])
        self.assertEqual(by_node["s2"]["panel_ids"], ["panel-2"])
        self.assertEqual(by_node["s3"]["panel_ids"], [])
        self.assertEqual(by_node["s1"]["visual_label"], "Encoder")

    def test_the_same_layers_always_produce_the_same_bindings(self):
        first, _ = self.build()
        second, _ = self.build()

        self.assertEqual(first, second)
        self.assertEqual(first["provenance"]["binding_version"], BINDING_VERSION)


class CheckLabelBindingTests(unittest.TestCase):
    """The second pass reuses the text the check stage read off the same figure."""

    def layers(self):
        """Sentence labels that share no word with the figure, as the real data does."""
        scientific = scientific_graph(
            nodes=[
                sci_node("s1", "Symbolic conversion front-end for user utterances."),
                sci_node("s2", "Persistent dialogue memory register."),
                sci_node("s3", "Amphipol control condition that never reaches the figure."),
            ],
        )
        visual = visual_graph(nodes=[vis_block("b1", "Translator"), vis_block("b2", "Gated-state")])
        return scientific, visual

    def review(self, concepts):
        return {"contract": "rfs.figure_text_review", "paper_id": "p0001", "concepts": concepts}

    def concept(self, concept_id, *, visible, image_label="", confidence=0.9):
        return {"concept_id": concept_id, "visible": visible, "image_label": image_label, "confidence": confidence}

    def test_a_sentence_label_binds_through_the_text_the_figure_carries(self):
        scientific, visual = self.layers()
        review = self.review([
            self.concept("s1", visible=True, image_label="Translator"),
            self.concept("s2", visible=True, image_label="Gated-state"),
        ])

        bindings, report = build_layer_bindings(scientific, visual, review)
        by_node = {item["scientific_node_id"]: item for item in bindings["node_bindings"]}

        self.assertEqual(by_node["s1"]["status"], "bound")
        self.assertEqual(by_node["s1"]["visual_node_ids"], ["b1"])
        self.assertEqual(by_node["s1"]["provenance"]["method"], "image_label_match")
        self.assertEqual(by_node["s1"]["provenance"]["check_confidence"], 0.9)
        self.assertEqual(by_node["s2"]["visual_node_ids"], ["b2"])
        self.assertEqual(report["image_label_bound_nodes"], 2)
        self.assertEqual(report["label_match_bound_nodes"], 0)
        self.assertEqual(report["provider_calls"], 0)

    def test_a_concept_the_figure_does_not_show_stays_unbound(self):
        scientific, visual = self.layers()
        review = self.review([self.concept("s3", visible=False, image_label="Translator")])

        bindings, report = build_layer_bindings(scientific, visual, review)
        by_node = {item["scientific_node_id"]: item for item in bindings["node_bindings"]}

        self.assertEqual(by_node["s3"]["status"], "unbound")
        self.assertEqual(by_node["s3"]["visual_node_ids"], [])
        self.assertEqual(report["image_label_bound_nodes"], 0)
        self.assertEqual(report["check_audit"]["image_label_skipped_not_visible"], 1)

    def test_a_visible_concept_without_readable_text_stays_unbound(self):
        scientific, visual = self.layers()
        review = self.review([self.concept("s1", visible=True, image_label="   ")])

        _bindings, report = build_layer_bindings(scientific, visual, review)

        self.assertEqual(report["image_label_bound_nodes"], 0)
        self.assertEqual(report["check_audit"]["image_label_skipped_no_image_label"], 1)

    def test_text_that_matches_no_block_stays_unbound(self):
        scientific, visual = self.layers()
        review = self.review([self.concept("s1", visible=True, image_label="Photon budget")])

        _bindings, report = build_layer_bindings(scientific, visual, review)

        self.assertEqual(report["image_label_bound_nodes"], 0)
        self.assertEqual(report["check_audit"]["image_label_skipped_no_match"], 1)

    def test_the_confirmed_reading_owns_the_pair_while_the_string_pass_stays_counted(self):
        """The fact layer decides first; the independent string pass keeps its own count."""
        scientific = scientific_graph(nodes=[sci_node("s1", "Translator"), sci_node("s2", "Persistent dialogue memory register.")])
        visual = visual_graph(nodes=[vis_block("b1", "Translator"), vis_block("b2", "Gated-state")])
        review = self.review([
            self.concept("s1", visible=True, image_label="Translator"),
            self.concept("s2", visible=True, image_label="Gated-state"),
        ])

        bindings, report = build_layer_bindings(scientific, visual, review)
        by_node = {item["scientific_node_id"]: item for item in bindings["node_bindings"]}

        self.assertEqual(by_node["s1"]["provenance"]["method"], "image_label_match")
        self.assertEqual(by_node["s2"]["provenance"]["method"], "image_label_match")
        self.assertEqual(by_node["s1"]["visual_node_ids"], ["b1"])
        self.assertEqual(report["fact_bound_nodes"], 2)
        self.assertEqual(report["label_pass_bound_nodes"], 1)
        self.assertEqual(report["label_match_bound_nodes"], 0)
        self.assertEqual(report["image_label_bound_nodes"], 2)
        self.assertEqual(report["diagnostic_label_match_required_coverage"], 0.0)
        self.assertEqual(report["diagnostic_required_node_coverage"], 1.0)

    def test_a_relation_binds_once_both_endpoints_come_back(self):
        scientific = scientific_graph(
            nodes=[
                sci_node("s1", "Symbolic conversion front-end for user utterances."),
                sci_node("s2", "Persistent dialogue memory register."),
            ],
            edges=[{"id": "e1", "source": "s1", "target": "s2", "relation_type": "data_flow"}],
        )
        visual = visual_graph(
            nodes=[vis_block("b1", "Translator"), vis_block("b2", "Gated-state")],
            edges=[{"id": "v1", "source": "b1", "target": "b2", "relation_type": "forward"}],
        )
        review = self.review([
            self.concept("s1", visible=True, image_label="Translator"),
            self.concept("s2", visible=True, image_label="Gated-state"),
        ])

        without = build_layer_bindings(scientific, visual)[1]
        with_review = build_layer_bindings(scientific, visual, review)[1]

        self.assertEqual(without["bound_edges"], 0)
        self.assertEqual(with_review["bound_edges"], 1)

    def test_without_a_review_the_result_is_the_one_bind_always_produced(self):
        scientific, visual = self.layers()

        missing, missing_report = build_layer_bindings(scientific, visual)
        empty, _ = build_layer_bindings(scientific, visual, {"concepts": []})

        self.assertEqual(missing_report["image_label_bound_nodes"], 0)
        self.assertFalse(missing["provenance"]["check_supplied"])
        self.assertTrue(empty["provenance"]["check_supplied"])
        self.assertEqual(
            [item["provenance"]["method"] for item in missing["node_bindings"]],
            ["label_match"] * len(missing["node_bindings"]),
        )

    def test_a_weak_label_match_gives_way_to_a_confirmed_reading(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Photon budget estimator")])
        visual = visual_graph(nodes=[vis_block("b1", "Photon budget sampler"), vis_block("b2", "Radiance cache")])
        review = self.review([self.concept("s1", visible=True, image_label="Radiance cache")])

        weak_only, weak_report = build_layer_bindings(scientific, visual)
        confirmed, confirmed_report = build_layer_bindings(scientific, visual, review)
        held = weak_only["node_bindings"][0]
        won = confirmed["node_bindings"][0]

        self.assertEqual(held["status"], "weak")
        self.assertEqual(held["visual_node_ids"], ["b1"])
        self.assertEqual(weak_report["diagnostic_required_node_coverage"], 0.0)
        self.assertEqual(won["status"], "bound")
        self.assertEqual(won["visual_node_ids"], ["b2"])
        self.assertEqual(won["provenance"]["method"], "image_label_match")
        self.assertEqual(confirmed_report["label_match_bound_nodes"], 0)
        self.assertEqual(confirmed_report["diagnostic_required_node_coverage"], 1.0)

    def test_the_second_pass_is_reproducible(self):
        scientific, visual = self.layers()
        review = self.review([
            self.concept("s1", visible=True, image_label="Translator"),
            self.concept("s2", visible=True, image_label="Gated-state"),
        ])

        first, _ = build_layer_bindings(scientific, visual, review)
        second, _ = build_layer_bindings(scientific, visual, review)

        self.assertEqual(first, second)
        validation = validate_graph_bindings(first, scientific_graph=scientific, visual_graph=visual)
        self.assertTrue(validation["ok"], validation["errors"])

    def test_an_answer_about_a_concept_the_layer_dropped_is_not_bound(self):
        """The review was asked about a concept list this layer may no longer declare.

        The check stage is paid and cannot be re-asked for free, so its answers about
        concepts that are gone have to be dropped and counted rather than crash the
        rebind or bind a concept the layer no longer states.
        """

        scientific, visual = self.layers()
        review = self.review([
            self.concept("s1", visible=True, image_label="Translator"),
            self.concept("s_gone", visible=True, image_label="Gated-state"),
        ])

        bindings, report = build_layer_bindings(scientific, visual, review)

        self.assertEqual(
            sorted(item["scientific_node_id"] for item in bindings["node_bindings"]),
            sorted(scientific["nodes"][index]["id"] for index in range(len(scientific["nodes"]))),
        )
        self.assertEqual(report["stale_fact_nodes"], 1)
        self.assertEqual(report["check_audit"]["stale_fact_concepts"], 1)
        self.assertEqual(report["image_label_bound_nodes"], 1)
        self.assertTrue(report["warnings"])


class StubEmbedder:
    """Deterministic stand-in for the paid endpoint, with a chosen vector per label.

    Like the real provider it counts cumulatively and is shared by every paper of a
    run, so a test can tell what one paper caused apart from what the process reached.
    """

    name = "stub"
    revision = "stub-v1"
    dimension = 2

    def __init__(self, vectors=None, *, fail=False):
        self.vectors = dict(vectors or {})
        self.fail = fail
        self.calls = 0
        self.texts = 0

    def encode(self, texts):
        wanted = [str(text) for text in texts]
        if self.fail:
            raise EmbeddingUnavailable("stub endpoint is down")
        self.calls += 1
        self.texts += len(wanted)
        return [list(self.vectors.get(text, [0.0, 1.0])) for text in wanted]

    def fingerprint(self):
        return {"contract": "rfs.text_embedding", "kind": "stub", "model": self.name, "revision": self.revision, "dimension": self.dimension}

    def usage(self):
        return {"calls": self.calls, "texts": self.texts, "total_tokens": self.texts * 7, "error": ""}


class FactLayerTests(unittest.TestCase):
    """The confirmed reading decides first, and the string pass stays countable."""

    def review(self, concepts):
        return {"contract": "rfs.figure_text_review", "paper_id": "p0001", "concepts": concepts}

    def concept(self, concept_id, image_label, *, visible=True, confidence=0.9):
        return {"concept_id": concept_id, "visible": visible, "image_label": image_label, "confidence": confidence}

    def test_a_confirmed_reading_overrides_the_string_pass_and_records_the_conflict(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Encoder (shared)")])
        review = self.review([self.concept("s1", "Encoder (shared)")])

        bindings, report = build_layer_bindings(scientific, visual, review)
        bound = bindings["node_bindings"][0]

        # The figure draws the encoder twice, so the concept owns both copies, and the
        # block the string pass preferred is the one the reader overrode.
        self.assertEqual(bound["visual_node_ids"], ["b2", "b1"])
        self.assertEqual(bound["provenance"]["method"], "image_label_match")
        self.assertTrue(bound["provenance"]["reader_overrode_label_match"])
        self.assertEqual(report["label_pass_bound_nodes"], 1)
        self.assertEqual(report["label_match_bound_nodes"], 0)
        self.assertEqual(report["fact_override_nodes"], 1)
        self.assertEqual(report["fact_vs_label_conflicts"], 1)
        self.assertEqual(report["fact_vs_label_conflict_examples"][0]["label_visual_node_id"], "b1")
        self.assertEqual(report["fact_vs_label_conflict_examples"][0]["fact_visual_node_id"], "b2")
        self.assertEqual(report["multi_bound_nodes"], 1)

    def test_a_copy_that_scores_weaker_than_the_margin_is_not_added(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Encoder module")])

        bindings, report = build_layer_bindings(scientific, visual)
        bound = bindings["node_bindings"][0]

        self.assertEqual(bound["visual_node_ids"], ["b1"])
        self.assertEqual(report["multi_bound_nodes"], 0)

    def test_the_semantic_layer_recovers_what_the_strings_cannot_reach(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Symbolic conversion front-end for user utterances.")])
        visual = visual_graph(nodes=[vis_block("b1", "Translator")])
        review = self.review([self.concept("s1", "translation front-end")])
        vectors = {"translation front-end": [1.0, 0.0], "Translator": [1.0, 0.0]}

        without = build_layer_bindings(scientific, visual, review)[1]
        bindings, report = build_layer_bindings(scientific, visual, review, embedder=StubEmbedder(vectors))

        self.assertEqual(without["image_label_bound_nodes"], 0)
        self.assertEqual(without["check_audit"]["image_label_skipped_no_match"], 1)
        self.assertEqual(report["image_label_bound_nodes"], 1)
        self.assertEqual(report["semantic_ranked_nodes"], 1)
        self.assertEqual(report["embedding_status"], "stub")
        self.assertEqual(report["embedding_fingerprint"]["kind"], "stub")
        self.assertEqual(report["provider_calls"], 1)
        self.assertEqual([item["status"] for item in bindings["node_bindings"]], ["bound"])

    def test_a_residual_semantic_pair_stays_weak_and_never_covers_a_required_node(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Frozen encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Optimizer")])
        vectors = {"Frozen encoder": [0.0, 1.0], "Optimizer": [0.0, 1.0]}

        bindings, report = build_layer_bindings(scientific, visual, embedder=StubEmbedder(vectors))
        bound = bindings["node_bindings"][0]

        self.assertEqual(bound["status"], "weak")
        self.assertEqual(bound["provenance"]["method"], SEMANTIC_MATCH_METHOD)
        self.assertEqual(report["semantic_weak_nodes"], 1)
        self.assertEqual(report["diagnostic_required_node_coverage"], 0.0)

    def test_a_rerun_with_the_same_embedder_is_reproducible(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Symbolic conversion front-end for user utterances.")])
        visual = visual_graph(nodes=[vis_block("b1", "Translator")])
        review = self.review([self.concept("s1", "translation front-end")])
        vectors = {"translation front-end": [1.0, 0.0], "Translator": [1.0, 0.0]}

        first = build_layer_bindings(scientific, visual, review, embedder=StubEmbedder(vectors))[0]
        second = build_layer_bindings(scientific, visual, review, embedder=StubEmbedder(vectors))[0]

        self.assertEqual(first, second)

    def test_a_shared_endpoint_reports_only_what_this_paper_caused(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Translator"), sci_node("s2", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Translator")])
        embedder = StubEmbedder({"Translator": [1.0, 0.0]})

        _first_bindings, first = build_layer_bindings(scientific, visual, embedder=embedder)
        _second_bindings, second = build_layer_bindings(scientific, visual, embedder=embedder)

        # Two papers, one process-wide endpoint: the counters the provider accumulates
        # must not make the second paper look like it cost twice as much as the first.
        self.assertEqual(embedder.calls, 2)
        self.assertEqual(first["embedding_calls"], 1)
        self.assertEqual(second["embedding_calls"], 1)
        self.assertEqual(second["embedding_texts"], 2)
        self.assertEqual(second["embedding_tokens"], first["embedding_tokens"])

    def test_the_vector_cannot_reorder_the_string_pass_it_only_ranks_the_reading(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder first"), sci_node("s2", "Encoder second")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder")])
        vectors = {"Encoder first": [0.0, 1.0], "Encoder second": [1.0, 0.0], "Encoder": [1.0, 0.0]}

        def pairs(items):
            return [(item["scientific_node_id"], item["visual_node_ids"], item["status"]) for item in items]

        without = build_layer_bindings(scientific, visual)[0]["node_bindings"]
        with_embedder = build_layer_bindings(scientific, visual, embedder=StubEmbedder(vectors))[0]["node_bindings"]

        # Both labels contain the drawn block equally well, and the vectors would rather
        # see s2 take it. The ids decide either way: switching an embedding on must not
        # move a pair the string pass has always made.
        self.assertEqual(pairs(without), [("s1", ["b1"], "bound"), ("s2", [], "unbound")])
        self.assertEqual(pairs(with_embedder), pairs(without))

    def test_without_an_embedder_nothing_is_called(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Translator")])
        visual = visual_graph(nodes=[vis_block("b1", "Translator")])

        _bindings, report = build_layer_bindings(scientific, visual)

        self.assertEqual(report["provider_calls"], 0)
        self.assertEqual(report["embedding_status"], "off")
        self.assertIsNone(report["embedding_fingerprint"])

    def test_a_failing_embedder_degrades_to_the_deterministic_layers_and_says_so(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Translator")])
        visual = visual_graph(nodes=[vis_block("b1", "Translator")])

        plain, _ = build_layer_bindings(scientific, visual)
        degraded, report = build_layer_bindings(scientific, visual, embedder=StubEmbedder(fail=True))

        # Only the provenance differs: the degraded run has to say why it fell back.
        self.assertEqual(degraded["node_bindings"], plain["node_bindings"])
        self.assertEqual(degraded["edge_bindings"], plain["edge_bindings"])
        self.assertEqual(degraded["provenance"]["embedding"]["status"], "unavailable")
        self.assertEqual(report["embedding_status"], "unavailable")
        self.assertIn("stub endpoint is down", report["embedding_error"])
        self.assertEqual(report["provider_calls"], 0)

    def test_a_local_model_that_is_missing_degrades_instead_of_raising(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Translator")])
        visual = visual_graph(nodes=[vis_block("b1", "Translator")])

        _bindings, report = build_layer_bindings(
            scientific, visual, embedder=LocalSentenceTransformerEmbedder(Path("tmp/definitely-not-here"))
        )

        self.assertEqual(report["embedding_status"], "unavailable")
        self.assertEqual(report["embedding_calls"], 0)


class StructureLayerTests(unittest.TestCase):
    """Structure lowers confidence and asks for review; it never moves a binding."""

    def test_each_rule_names_exactly_one_conflict(self):
        scientific = sci_node("s1", "Encoder")
        nodes = {"b1": vis_block("b1", "Encoder")}
        without_panel = {"b1": {"id": "b1", "kind": "module", "label": "Encoder"}}

        self.assertEqual(structure_violations(scientific, {"s1": (1, 1)}, ["b1"], nodes, {"b1": (0, 0)}, panels_present=True), ["isolated_block"])
        self.assertEqual(structure_violations(scientific, {"s1": (0, 1)}, ["b1"], nodes, {"b1": (1, 0)}, panels_present=True), ["direction_conflict"])
        self.assertEqual(structure_violations(scientific, {"s1": (2, 0)}, ["b1"], nodes, {"b1": (1, 0)}, panels_present=True), ["topology_conflict"])
        self.assertEqual(structure_violations(scientific, {"s1": (1, 0)}, ["b1"], without_panel, {"b1": (1, 0)}, panels_present=True), ["panel_missing"])

    def test_a_structure_conflict_lowers_confidence_without_moving_the_binding(self):
        scientific = scientific_graph(
            nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Decoder")],
            edges=[{"id": "e1", "source": "s2", "target": "s1", "relation_type": "data_flow"}],
        )
        visual = visual_graph(
            nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Decoder")],
            edges=[{"id": "v1", "source": "b1", "target": "b2", "relation_type": "forward"}],
        )

        bindings, report = build_layer_bindings(scientific, visual)
        by_node = {item["scientific_node_id"]: item for item in bindings["node_bindings"]}

        self.assertEqual(by_node["s1"]["status"], "bound")
        self.assertEqual(by_node["s1"]["score"], 1.0)
        self.assertEqual(by_node["s1"]["signals"]["structure"], ["direction_conflict"])
        self.assertEqual(by_node["s1"]["confidence"], 0.6)
        self.assertTrue(by_node["s1"]["needs_review"])
        self.assertEqual(report["bound_nodes"], 2)
        self.assertEqual(report["diagnostic_required_node_coverage"], 1.0)
        self.assertEqual(report["structure_conflict_nodes"], 2)
        self.assertEqual(report["needs_review_nodes"], 2)


class MultiBindTests(unittest.TestCase):
    """One concept drawn in two places owns both copies, and its arrows follow."""

    def test_a_module_drawn_twice_binds_both_copies(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder", panels=("panel-1",)), vis_block("b2", "Encoder", panels=("panel-2",))])

        bindings, report = build_layer_bindings(scientific, visual)
        bound = bindings["node_bindings"][0]

        self.assertEqual(bound["visual_node_ids"], ["b1", "b2"])
        self.assertEqual(bound["panel_ids"], ["panel-1", "panel-2"])
        self.assertEqual(bound["status"], "bound")
        self.assertEqual(report["multi_bound_nodes"], 1)

    def test_an_arrow_touching_the_second_copy_still_binds_the_relation(self):
        scientific = scientific_graph(
            nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Decoder")],
            edges=[{"id": "e1", "source": "s1", "target": "s2", "relation_type": "data_flow"}],
        )
        visual = visual_graph(
            nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Encoder"), vis_block("b3", "Decoder")],
            edges=[{"id": "v1", "source": "b2", "target": "b3", "relation_type": "forward"}],
        )

        bindings, _report = build_layer_bindings(scientific, visual)
        edges = {item["scientific_edge_id"]: item for item in bindings["edge_bindings"]}

        self.assertEqual(edges["e1"]["visual_edge_ids"], ["v1"])
        self.assertEqual(edges["e1"]["status"], "bound")
        self.assertEqual(edges["e1"]["confidence"], 1.0)
        self.assertFalse(edges["e1"]["needs_review"])


class PanelTargetTests(unittest.TestCase):
    """A row name is a label of the figure, so a concept stated as one must bind."""

    def layers(self):
        scientific = scientific_graph(
            nodes=[sci_node("s1", "Decision-focused generative learning (Gen-DFL) framework.")],
        )
        visual = visual_graph(nodes=[vis_panel("panel-1", "Gen-DFL"), vis_block("b1", "Generator")])
        return scientific, visual

    def review(self, **overrides):
        concept = {"concept_id": "s1", "visible": True, "image_label": "Gen-DFL", "confidence": 0.9}
        concept.update(overrides)
        return {"contract": "rfs.figure_text_review", "paper_id": "p0001", "concepts": [concept]}

    def test_the_string_pass_reaches_a_row_label(self):
        scientific, visual = self.layers()

        bindings, report = build_layer_bindings(scientific, visual)
        by_node = {item["scientific_node_id"]: item for item in bindings["node_bindings"]}

        self.assertEqual(by_node["s1"]["status"], "bound")
        self.assertEqual(by_node["s1"]["visual_node_ids"], ["panel-1"])
        self.assertEqual(by_node["s1"]["panel_ids"], ["panel-1"])
        self.assertEqual(by_node["s1"]["provenance"]["method"], "label_match")
        self.assertEqual(report["diagnostic_required_node_coverage"], 1.0)
        self.assertEqual(report["panel_bound_nodes"], 1)

    def test_the_reading_binds_a_row_label_to_the_panel_it_quotes(self):
        scientific, visual = self.layers()

        bindings, report = build_layer_bindings(scientific, visual, self.review())
        by_node = {item["scientific_node_id"]: item for item in bindings["node_bindings"]}

        self.assertEqual(by_node["s1"]["status"], "bound")
        self.assertEqual(by_node["s1"]["visual_node_ids"], ["panel-1"])
        self.assertEqual(by_node["s1"]["panel_ids"], ["panel-1"])
        self.assertEqual(by_node["s1"]["provenance"]["method"], "image_label_match")
        self.assertEqual(report["check_audit"]["image_label_bound_panels"], 1)
        self.assertEqual(report["panel_bound_nodes"], 1)

    def test_a_panel_the_reading_never_names_stays_free(self):
        scientific, visual = self.layers()
        review = self.review(image_label="Generator")

        bindings, report = build_layer_bindings(scientific, visual, review)
        by_node = {item["scientific_node_id"]: item for item in bindings["node_bindings"]}

        self.assertEqual(by_node["s1"]["visual_node_ids"], ["b1"])
        self.assertEqual(report["panel_bound_nodes"], 0)
        self.assertEqual(report["check_audit"]["image_label_bound_panels"], 0)

    def test_a_panel_target_is_exempt_from_the_degree_rules(self):
        scientific = sci_node("s1", "Gen-DFL framework")
        nodes = {"panel-1": vis_panel("panel-1", "Gen-DFL")}

        self.assertEqual(
            structure_violations(scientific, {"s1": (1, 0)}, ["panel-1"], nodes, {"panel-1": (0, 0)}, panels_present=True),
            [],
        )

    def test_a_panel_is_never_a_second_drawing_of_a_module(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Generator")])
        visual = visual_graph(nodes=[vis_block("b1", "Generator"), vis_panel("panel-1", "Generator")])

        bindings, report = build_layer_bindings(scientific, visual)
        by_node = {item["scientific_node_id"]: item for item in bindings["node_bindings"]}

        self.assertEqual(by_node["s1"]["visual_node_ids"], ["b1"])
        self.assertEqual(report["multi_bound_nodes"], 0)
        self.assertEqual(report["panel_bound_nodes"], 0)


class FigureFirstReportTests(unittest.TestCase):
    """The author's figure is the standard, so it is read block by block."""

    def review(self, concepts, extra_blocks=()):
        return {
            "contract": "rfs.figure_text_review",
            "paper_id": "p0001",
            "concepts": concepts,
            "extra_blocks": [{"label": label, "where": ""} for label in extra_blocks],
        }

    def concept(self, concept_id, image_label, *, visible=True):
        return {"concept_id": concept_id, "visible": visible, "image_label": image_label, "confidence": 0.9}

    def test_every_drawn_block_is_accounted_for(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Decoder")])
        visual = visual_graph(nodes=[
            vis_block("b1", "Encoder"),
            vis_block("b2", "Decoder"),
            vis_block("b3", "Step 1"),
            vis_block("b4", "Legend"),
        ])
        review = self.review(
            [self.concept("s1", "Encoder"), self.concept("s2", "Decoder")],
            extra_blocks=["Step 1"],
        )

        _, report = build_layer_bindings(scientific, visual, review)
        figure = report["figure_first"]

        self.assertEqual(figure["drawn_blocks"], 4)
        self.assertEqual(figure["claimed_blocks"], 2)
        self.assertEqual(figure["declared_outside_contract"], 1)
        # The block the reading listed as outside the contract is accounted for; the one
        # nothing names is the only thing left to look at.
        self.assertEqual(figure["unaccounted_blocks"], 1)
        self.assertEqual(figure["unaccounted_block_ids"], ["b4"])

    def test_a_concept_the_author_left_out_is_a_selection_sample(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Decoder"), sci_node("s3", "Loss")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Decoder")])
        review = self.review([
            self.concept("s1", "Encoder"),
            self.concept("s2", "Decoder"),
            self.concept("s3", "", visible=False),
        ])

        _, report = build_layer_bindings(scientific, visual, review)
        samples = report["selection_samples"]

        self.assertEqual(samples["required_concepts"], 3)
        self.assertEqual(samples["judged"], 3)
        self.assertEqual(samples["drawn"], 2)
        self.assertEqual(samples["not_drawn"], 1)
        # The omitted concept is a sample, never a bound node and never a defect.
        self.assertEqual(report["diagnostic_required_node_coverage"], round(2 / 3, 4))

    def test_without_a_review_no_selection_sample_is_claimed(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder")])

        _, report = build_layer_bindings(scientific, visual)
        samples = report["selection_samples"]

        self.assertFalse(samples["review_available"])
        self.assertIsNone(samples["drawn"])
        self.assertIsNone(samples["not_drawn"])


class FigureClaimTests(unittest.TestCase):
    """Every object the figure draws ends with exactly one conclusion about what it is."""

    def review(self, concepts, extra_blocks=()):
        return {
            "contract": "rfs.figure_text_review",
            "paper_id": "p0001",
            "concepts": concepts,
            "extra_blocks": [{"label": label, "where": ""} for label in extra_blocks],
        }

    def concept(self, concept_id, image_label, *, visible=True):
        return {"concept_id": concept_id, "visible": visible, "image_label": image_label, "confidence": 0.9}

    def claim(self, report, object_id):
        return next(item for item in report["figure_claims"]["objects"] if item["object_id"] == object_id)

    def test_a_blocks_own_label_is_the_first_evidence(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder")])

        _, report = build_layer_bindings(scientific, visual)
        block = self.claim(report, "b1")

        self.assertEqual(block["conclusion"], "bound")
        self.assertEqual(block["kind"], "block")
        self.assertEqual(block["claims"][0]["priority"], 1)
        self.assertEqual(block["claims"][0]["method"], "label_match")
        self.assertTrue(block["claims"][0]["accepted"])

    def test_the_text_the_reader_saw_claims_a_block_the_own_labels_cannot_reach(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Symbolic conversion front-end for user utterances.")])
        visual = visual_graph(nodes=[vis_block("b1", "Translator")])
        review = self.review([self.concept("s1", "Translator")])

        _, report = build_layer_bindings(scientific, visual, review)
        block = self.claim(report, "b1")

        self.assertEqual(block["conclusion"], "bound")
        self.assertEqual(block["claims"][0]["priority"], 2)
        self.assertEqual(block["claims"][0]["method"], "image_label_match")

    def test_the_objects_own_label_outranks_the_text_the_reader_saw(self):
        scientific = scientific_graph(nodes=[
            sci_node("s1", "Encoder"),
            sci_node("s2", "Symbolic conversion front-end for user utterances."),
        ])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder")])
        review = self.review([self.concept("s2", "Encoder")])

        _, report = build_layer_bindings(scientific, visual, review)
        block = self.claim(report, "b1")

        # Both readings are kept, the block's own label decides, and the disagreement
        # is flagged rather than averaged away.
        self.assertEqual([item["scientific_node_id"] for item in block["claims"]], ["s1", "s2"])
        self.assertTrue(block["claims"][0]["accepted"])
        self.assertFalse(block["claims"][1]["accepted"])
        self.assertTrue(block["needs_review"])

    def test_a_block_the_reading_declared_outside_is_accounted_for(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_block("b2", "PTO baseline")])
        review = self.review([self.concept("s1", "Encoder")], extra_blocks=["PTO baseline"])

        _, report = build_layer_bindings(scientific, visual, review)
        block = self.claim(report, "b2")

        self.assertEqual(block["conclusion"], "outside_paper_method")
        self.assertEqual(block["claims"][0]["method"], "reader_declared_outside")
        self.assertFalse(block["needs_review"])
        self.assertEqual(report["figure_claims"]["summary"]["outside_blocks"], 1)

    def test_only_the_reading_may_declare_an_object_outside_the_method(self):
        """A concept the figure leaves out is a trade-off, never an outside method."""
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Loss")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder")])
        review = self.review([self.concept("s1", "Encoder"), self.concept("s2", "", visible=False)])

        _, report = build_layer_bindings(scientific, visual, review)

        self.assertEqual(report["figure_claims"]["summary"]["outside_paper_method"], 0)
        self.assertEqual(report["figure_claims"]["summary"]["unresolved"], 0)

    def test_a_block_nothing_explains_carries_its_candidates_and_why_they_were_refused(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Encoder stack")])
        review = self.review([self.concept("s1", "Encoder")])

        _, report = build_layer_bindings(scientific, visual, review)
        block = self.claim(report, "b2")

        self.assertEqual(block["conclusion"], "unresolved")
        self.assertEqual(block["claims"], [])
        self.assertEqual(block["candidates"][0]["scientific_node_id"], "s1")
        self.assertIn("bound to another object", block["candidates"][0]["reason"])

    def test_a_row_label_is_claimed_like_a_block(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Ours")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_panel("p1", "Ours")])
        review = self.review([self.concept("s1", "Encoder"), self.concept("s2", "Ours")])

        _, report = build_layer_bindings(scientific, visual, review)
        panel = self.claim(report, "p1")

        self.assertEqual(panel["kind"], "row_label")
        self.assertEqual(panel["conclusion"], "bound")
        self.assertEqual(report["figure_claims"]["summary"]["bound_row_labels"], 1)

    def arrow_layers(self, *, scientific_edges=None, visual_edges=None, extra_blocks=()):
        scientific = scientific_graph(
            nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Decoder")],
            edges=scientific_edges if scientific_edges is not None else [
                {"id": "e1", "source": "s1", "target": "s2", "relation_type": "data_flow"},
            ],
        )
        visual = visual_graph(
            nodes=[
                vis_block("b1", "Encoder"),
                vis_block("b2", "Decoder"),
                vis_block("b3", "PTO baseline"),
            ],
            edges=visual_edges if visual_edges is not None else [
                {"id": "a1", "source": "b1", "target": "b2", "relation_type": "forward"},
            ],
        )
        review = self.review(
            [self.concept("s1", "Encoder"), self.concept("s2", "Decoder")],
            extra_blocks=extra_blocks,
        )
        return scientific, visual, review

    def test_an_arrow_between_two_claimed_blocks_carries_the_relation(self):
        scientific, visual, review = self.arrow_layers()

        _, report = build_layer_bindings(scientific, visual, review)
        arrow = self.claim(report, "a1")

        self.assertEqual(arrow["conclusion"], "bound")
        self.assertEqual(arrow["claims"][0]["scientific_edge_id"], "e1")
        self.assertEqual(arrow["claims"][0]["direction"], "forward")
        self.assertFalse(arrow["needs_review"])
        self.assertEqual(report["figure_claims"]["summary"]["bound_arrows"], 1)

    def test_an_arrow_that_runs_backwards_still_counts_and_asks_for_review(self):
        scientific, visual, review = self.arrow_layers(visual_edges=[
            {"id": "a1", "source": "b2", "target": "b1", "relation_type": "forward"},
        ])

        _, report = build_layer_bindings(scientific, visual, review)
        arrow = self.claim(report, "a1")

        self.assertEqual(arrow["conclusion"], "bound")
        self.assertEqual(arrow["claims"][0]["direction"], "reversed")
        self.assertTrue(arrow["needs_review"])

    def test_an_arrow_into_a_block_the_reading_declared_outside_is_outside_too(self):
        scientific, visual, review = self.arrow_layers(
            visual_edges=[{"id": "a1", "source": "b3", "target": "b1", "relation_type": "forward"}],
            extra_blocks=["PTO baseline"],
        )

        _, report = build_layer_bindings(scientific, visual, review)
        arrow = self.claim(report, "a1")

        self.assertEqual(arrow["conclusion"], "outside_paper_method")
        self.assertEqual(arrow["claims"][0]["method"], "endpoint_outside_paper_method")
        self.assertEqual(arrow["endpoints"][0]["conclusion"], "outside_paper_method")

    def test_an_arrow_between_unexplained_blocks_stays_unresolved_with_candidates(self):
        scientific = scientific_graph(
            nodes=[sci_node("s1", "Encoder")],
            edges=[{"id": "e1", "source": "s1", "target": "s1", "relation_type": "data_flow"}],
        )
        visual = visual_graph(
            nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Encoder stack")],
            edges=[{"id": "a1", "source": "b1", "target": "b2", "relation_type": "forward"}],
        )
        review = self.review([self.concept("s1", "Encoder")])

        _, report = build_layer_bindings(scientific, visual, review)
        arrow = self.claim(report, "a1")

        self.assertEqual(arrow["conclusion"], "unresolved")
        self.assertEqual(arrow["claims"], [])
        self.assertEqual(arrow["candidates"][0]["scientific_node_id"], "s1")
        self.assertEqual(arrow["candidates"][0]["end"], "target")

    def test_every_object_of_the_figure_gets_exactly_one_conclusion(self):
        scientific, visual, review = self.arrow_layers()

        _, report = build_layer_bindings(scientific, visual, review)
        claims = report["figure_claims"]
        summary = claims["summary"]

        self.assertEqual(len(claims["objects"]), 4)
        self.assertEqual(summary["blocks"], 3)
        self.assertEqual(summary["row_labels"], 0)
        self.assertEqual(summary["arrows"], 1)
        self.assertEqual(
            summary["bound"] + summary["outside_paper_method"] + summary["unresolved"],
            summary["objects"],
        )
        self.assertEqual(
            summary["claim_coverage"],
            round((summary["bound"] + summary["outside_paper_method"]) / summary["objects"], 4),
        )
        self.assertEqual(claims["contract"], FIGURE_CLAIMS_CONTRACT)
        self.assertEqual(claims["version"], FIGURE_CLAIMS_VERSION)
        self.assertEqual(claims["binding_version"], BINDING_VERSION)

    def test_the_stored_report_keeps_the_counts_and_not_the_objects(self):
        from rfs.judge_training.pilot_graphs import bind_paper

        scientific, visual, review = self.arrow_layers()
        row = {"paper_id": "p0001", "split": "train", "domain": "cv"}
        entry, ledger, report = bind_paper(
            row,
            {"paper_id": "p0001", "scientific_graph": scientific},
            {"paper_id": "p0001", "visual_graph": visual},
            {"paper_id": "p0001", "figure_text_review": review},
        )

        self.assertEqual(report["figure_claims"]["objects"], 4)
        self.assertEqual(entry["binding_report"]["figure_claims"], entry["figure_claims"]["summary"])
        self.assertEqual([item["object_id"] for item in entry["figure_claims"]["objects"]], ["b1", "b2", "b3", "a1"])
        self.assertIn("figure_claims.json", ledger["artifact_sha256"])


class SelectionSampleTests(unittest.TestCase):
    """The author's trade-off is data: drawn or left out, and drawn how."""

    def review(self, concepts):
        return {"contract": "rfs.figure_text_review", "paper_id": "p0001", "concepts": concepts}

    def concept(self, concept_id, image_label, *, visible=True, where=""):
        return {"concept_id": concept_id, "visible": visible, "image_label": image_label, "where": where, "confidence": 0.9}

    def test_a_concept_written_inside_another_drawing_still_counts_as_drawn(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Covariate x")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder")])
        review = self.review([
            self.concept("s1", "Encoder"),
            self.concept("s2", "", where="inside the Generator box, as a formula letter"),
        ])

        _, report = build_layer_bindings(scientific, visual, review)
        samples = report["selection_samples"]

        # Kept, but not given a block of its own: two different lessons, two fields.
        self.assertEqual(samples["judged"], 2)
        self.assertEqual(samples["drawn"], 2)
        self.assertEqual(samples["not_drawn"], 0)
        self.assertEqual(samples["as_block"], 1)
        self.assertEqual(samples["inside_other_graphics"], 1)

    def test_a_concept_the_figure_never_carries_is_an_omission(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Loss")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder")])
        review = self.review([self.concept("s1", "Encoder"), self.concept("s2", "", visible=False)])

        _, report = build_layer_bindings(scientific, visual, review)
        samples = report["selection_samples"]

        self.assertEqual(samples["not_drawn"], 1)
        self.assertEqual(samples["drawn"], 1)
        self.assertEqual(samples["inside_other_graphics"], 0)

    def test_a_row_name_is_carried_but_is_not_a_block_of_its_own(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Ours")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_panel("p1", "Ours")])
        review = self.review([self.concept("s1", "Ours")])

        _, report = build_layer_bindings(scientific, visual, review)
        samples = report["selection_samples"]

        self.assertEqual(samples["drawn"], 1)
        self.assertEqual(samples["as_block"], 0)
        self.assertEqual(samples["inside_other_graphics"], 1)

    def test_every_record_carries_the_content_and_the_expression_layer(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Loss")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder")])
        review = self.review([self.concept("s1", "Encoder"), self.concept("s2", "", visible=False)])

        _, report = build_layer_bindings(scientific, visual, review)
        records = {item["concept_id"]: item for item in report["figure_claims"]["selection_samples"]["records"]}

        self.assertTrue(records["s1"]["drawn"])
        self.assertTrue(records["s1"]["as_block"])
        self.assertFalse(records["s1"]["inside_other_graphics"])
        self.assertFalse(records["s2"]["drawn"])
        self.assertFalse(records["s2"]["as_block"])
        self.assertFalse(records["s2"]["inside_other_graphics"])

    def test_a_concept_the_review_never_answered_about_is_not_a_sample(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Loss")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder")])
        review = self.review([self.concept("s1", "Encoder")])

        _, report = build_layer_bindings(scientific, visual, review)
        samples = report["selection_samples"]

        self.assertEqual(samples["judged"], 1)
        self.assertEqual(samples["required_judged"], 1)

    def test_the_required_subset_keeps_its_own_counts(self):
        scientific = scientific_graph(
            nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Loss")],
            required_nodes=["s1"],
        )
        visual = visual_graph(nodes=[vis_block("b1", "Encoder")])
        review = self.review([self.concept("s1", "Encoder"), self.concept("s2", "", visible=True)])

        _, report = build_layer_bindings(scientific, visual, review)
        samples = report["selection_samples"]

        self.assertEqual(samples["judged"], 2)
        self.assertEqual(samples["required_judged"], 1)
        self.assertEqual(samples["required_drawn"], 1)
        self.assertEqual(samples["required_not_drawn"], 0)

    def test_the_stored_report_keeps_the_counts_and_the_payload_keeps_the_records(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder")])
        review = self.review([self.concept("s1", "Encoder")])

        _, report = build_layer_bindings(scientific, visual, review)

        self.assertNotIn("records", report["selection_samples"])
        self.assertEqual(report["figure_claims"]["selection_samples"]["records"][0]["concept_id"], "s1")


class EmbedderResolutionTests(unittest.TestCase):
    def test_an_off_spec_asks_for_the_deterministic_layers(self):
        for spec in (None, "", "off", "none"):
            embedder, status = resolve_binding_embedder(spec)
            self.assertIsNone(embedder)
            self.assertEqual(status["status"], "off")

    def test_an_api_spec_names_the_model(self):
        embedder, status = resolve_binding_embedder("api:text-embedding-3-large")

        self.assertEqual(status["status"], "api_embeddings")
        self.assertEqual(status["spec"], "api:text-embedding-3-large")
        self.assertEqual(embedder.fingerprint()["model"], "text-embedding-3-large")

    def test_the_offline_encoder_is_still_reachable_by_name(self):
        embedder, status = resolve_binding_embedder("hashing")

        self.assertEqual(status["status"], "local")
        self.assertEqual(embedder.fingerprint()["kind"], "hashing_ngram")


class LocalRoleTests(unittest.TestCase):
    """Four local rules classify the notation a v1 reading never named a role for."""

    def node(self, node_id, label, **properties):
        return {
            "id": node_id,
            "kind": "module",
            "label": label,
            "panel_ids": [],
            "properties": dict(properties),
        }

    def test_a_bare_number_or_a_single_letter_is_a_decoration(self):
        self.assertEqual(local_object_role(self.node("b1", "6.8")), "decorative")
        self.assertEqual(local_object_role(self.node("b2", "A")), "decorative")
        self.assertEqual(local_object_role(self.node("b3", "16x16")), "decorative")

    def test_a_short_token_or_a_formula_is_a_symbol(self):
        self.assertEqual(local_object_role(self.node("b1", "b16")), "symbol")
        self.assertEqual(local_object_role(self.node("b2", "U <- U - eta grad_U")), "symbol")

    def test_a_panel_or_legend_title_is_a_stage_annotation(self):
        self.assertEqual(local_object_role(self.node("b1", "Legend")), "stage_annotation")
        self.assertEqual(local_object_role(self.node("b2", "Main workflow")), "stage_annotation")

    def test_wording_that_looks_like_content_stays_unclassified(self):
        self.assertIsNone(local_object_role(self.node("b1", "Encoder stack")))
        self.assertIsNone(local_object_role(self.node("b2", "Retrieval")))

    def test_a_block_whose_label_is_its_own_id_is_unlabelled_not_a_decoration_of_its_name(self):
        self.assertTrue(object_is_unlabeled(self.node("b3", "b3", source_id="b3", text_confidence=0)))
        self.assertFalse(object_is_unlabeled(self.node("b3", "PVCNN Diffusion", source_id="b3")))

    def test_only_a_role_the_reading_chose_counts_never_the_old_graphic_tag(self):
        self.assertEqual(reader_object_role(self.node("b1", "x", role="external_method")), "external_method")
        self.assertIsNone(reader_object_role(self.node("b2", "x", visual_role="module")))

    def test_a_formula_beside_an_ordinary_word_is_content_not_notation(self):
        self.assertIsNone(local_object_role(self.node("b1", "Classification Loss $\\mathcal{L}_{cls}$")))
        self.assertIsNone(local_object_role(self.node("b2", "\\alpha M_s l_{dw} micromag descriptor")))

    def test_an_object_our_reading_never_put_text_on_is_not_called_decoration(self):
        self.assertIsNone(local_object_role(self.node("b3", "b3", source_id="b3", text_confidence=0)))
        self.assertIsNone(local_object_role(self.node("b4", "")))
        self.assertIsNone(local_object_role(self.node("vnode.abc123", "vnode.abc123")))

    def test_only_the_shape_of_the_wording_decides_and_the_vocabulary_stays_short(self):
        self.assertEqual(local_object_role(self.node("b1", "\\hat{z}^l")), "symbol")
        self.assertEqual(local_object_role(self.node("b2", "\\alpha\\left(\\begin{bmatrix}1\\\\2\\end{bmatrix}\\right)")), "symbol")

    def test_a_label_that_is_nothing_but_a_glyph_is_notation_not_content(self):
        self.assertEqual(local_object_role(self.node("b1", "+")), "symbol")
        self.assertEqual(local_object_role(self.node("b2", "...")), "symbol")
        self.assertEqual(local_object_role(self.node("b3", "\u2212")), "symbol")


class RepeatedDrawingTests(unittest.TestCase):
    """The same wording drawn twice is one object drawn twice, not a leftover."""

    def test_the_second_drawing_of_one_module_joins_the_concept_that_owns_the_first(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "PVCNN Diffusion")])
        visual = visual_graph(nodes=[vis_block("b1", "PVCNN Diffusion"), vis_block("b2", "PVCNN Diffusion")])

        bindings, report = build_layer_bindings(scientific, visual)

        self.assertEqual(bindings["node_bindings"][0]["visual_node_ids"], ["b1", "b2"])
        self.assertEqual(report["figure_claims"]["summary"]["repeat_groups"], 1)
        self.assertEqual(report["figure_claims"]["summary"]["repeat_instances"], 2)

    def test_the_second_drawing_records_the_place_it_repeats(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "PVCNN Diffusion")])
        visual = visual_graph(nodes=[vis_block("b1", "PVCNN Diffusion"), vis_block("b2", "PVCNN Diffusion")])

        _, report = build_layer_bindings(scientific, visual)
        copies = [item for item in report["figure_claims"]["objects"] if item["kind"] == "block"]

        self.assertEqual([item["repeat_group"]["index"] for item in copies], [1, 2])
        self.assertEqual([item["repeat_group"]["count"] for item in copies], [2, 2])

    def test_two_different_drawings_are_never_merged_into_one_group(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder"), sci_node("s2", "Decoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Decoder")])

        bindings, report = build_layer_bindings(scientific, visual)

        self.assertEqual(bindings["node_bindings"][0]["visual_node_ids"], ["b1"])
        self.assertEqual(bindings["node_bindings"][1]["visual_node_ids"], ["b2"])
        self.assertEqual(report["figure_claims"]["summary"]["repeat_groups"], 0)


class ClaimEvidenceTests(unittest.TestCase):
    """A formula is the figure's own notation, and the block that holds it owns it."""

    def block(self, node_id, label, *, box=None, **properties):
        node = vis_block(node_id, label)
        node["properties"] = dict(properties)
        if box is not None:
            node["properties"]["box"] = box
        return node

    def claim(self, report, object_id):
        return next(item for item in report["figure_claims"]["objects"] if item["object_id"] == object_id)

    def test_a_formula_is_a_figure_annotation_attached_to_the_block_that_holds_it(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[
            self.block("b1", "Encoder", box={"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.3}),
            self.block("b2", "U <- U - eta grad_U", box={"x": 0.15, "y": 0.15, "w": 0.2, "h": 0.05}),
        ])

        _, report = build_layer_bindings(scientific, visual)
        formula = self.claim(report, "b2")
        holder = self.claim(report, "b1")
        summary = report["figure_claims"]["summary"]

        self.assertEqual(formula["conclusion"], "figure_annotation")
        self.assertEqual(formula["role"], "symbol")
        self.assertEqual(formula["belongs_to"], "b1")
        self.assertEqual([item["glyph"] for item in holder["annotations"]], ["U <- U - eta grad_U"])
        self.assertEqual(summary["annotation_coverage"], 1.0)
        self.assertEqual(summary["annotations_attached"], 1)
        self.assertEqual(summary["annotations_unanchored"], 0)

    def test_a_formula_no_block_holds_is_kept_without_an_owner(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[
            self.block("b1", "Encoder", box={"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.3}),
            self.block("b2", "U <- U - eta grad_U", box={"x": 0.7, "y": 0.7, "w": 0.2, "h": 0.05}),
        ])

        _, report = build_layer_bindings(scientific, visual)
        summary = report["figure_claims"]["summary"]

        self.assertEqual(summary["annotations_attached"], 0)
        self.assertEqual(summary["annotations_unanchored"], 1)

    def test_a_short_token_the_reading_never_explained_is_a_symbol_not_a_module(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_block("b2", "b16")])

        _, report = build_layer_bindings(scientific, visual)
        summary = report["figure_claims"]["summary"]

        self.assertEqual(self.claim(report, "b2")["role"], "symbol")
        self.assertEqual(summary["roles"]["symbol"], 1)
        self.assertEqual(summary["unresolved"], 0)


class ClaimBucketTests(unittest.TestCase):
    """The five conclusions are the whole vocabulary, and each metric has its own denominator."""

    def review(self, concepts, extra_blocks=()):
        return {
            "contract": "rfs.figure_text_review",
            "paper_id": "p0001",
            "concepts": concepts,
            "extra_blocks": [{"label": label, "where": ""} for label in extra_blocks],
        }

    def concept(self, concept_id, image_label, *, visible=True):
        return {"concept_id": concept_id, "image_label": image_label, "visible": visible, "confidence": 0.9}

    def test_every_bucket_the_layer_can_reach_is_counted_and_they_add_up_to_the_objects(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[
            vis_block("b1", "Encoder"),
            vis_block("b2", "PTO baseline"),
            vis_block("b3", "Legend"),
            vis_block("b4", "ImageNet-21k"),
            vis_block("b5", "Unclear thing"),
        ])
        review = self.review([self.concept("s1", "Encoder")], extra_blocks=["PTO baseline"])
        provenance = lambda phrase: {"page": 4, "snippet": "the corpus", "matches": 1} if phrase == "ImageNet-21k" else None

        _, report = build_layer_bindings(scientific, visual, review, provenance=provenance)
        summary = report["figure_claims"]["summary"]
        buckets = ("bound", "outside_paper_method", "in_paper_not_in_method", "figure_annotation", "unresolved")

        self.assertEqual(summary["objects"], 5)
        self.assertEqual([summary[key] for key in buckets], [1, 1, 1, 1, 1])
        self.assertEqual(sum(summary[key] for key in buckets), summary["objects"])
        self.assertEqual(summary["provenance_searched"], 2)
        self.assertEqual(summary["provenance_found"], 1)
        self.assertEqual(summary["provenance_coverage"], 0.5)

    def test_an_object_the_paper_prints_but_the_method_list_omits_keeps_its_page(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_block("b2", "ImageNet-21k")])
        review = self.review([self.concept("s1", "Encoder")])
        provenance = lambda phrase: {"page": 4, "snippet": "trained on imagenet 21k", "matches": 2} if phrase == "ImageNet-21k" else None

        _, report = build_layer_bindings(scientific, visual, review, provenance=provenance)
        block = next(item for item in report["figure_claims"]["objects"] if item["object_id"] == "b2")

        self.assertEqual(block["conclusion"], "in_paper_not_in_method")
        self.assertEqual(block["provenance"]["page"], 4)
        self.assertEqual(block["provenance"]["matches"], 2)
        self.assertEqual(block["claims"][0]["method"], "paper_text_match")
        self.assertFalse(block["needs_review"])

    def test_an_object_no_page_carries_stays_unresolved_without_provenance(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Inker switch")])
        review = self.review([self.concept("s1", "Encoder")])

        _, report = build_layer_bindings(scientific, visual, review, provenance=lambda phrase: None)
        block = next(item for item in report["figure_claims"]["objects"] if item["object_id"] == "b2")

        self.assertEqual(block["conclusion"], "unresolved")
        self.assertEqual(block["provenance"], None)
        self.assertEqual(report["figure_claims"]["summary"]["provenance_found"], 0)

    def test_a_symbol_that_carries_a_concept_stays_counted_but_never_as_a_block(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Riemannian optimization")])
        symbol = vis_block("b1", "U <- U - eta grad_U")
        symbol["properties"] = {"role": "formula"}
        visual = visual_graph(nodes=[symbol])
        review = self.review([self.concept("s1", "U <- U - eta grad_U")])

        _, report = build_layer_bindings(scientific, visual, review)
        samples = report["figure_claims"]["selection_samples"]
        record = samples["records"][0]
        summary = report["figure_claims"]["summary"]

        self.assertEqual(report["figure_claims"]["objects"][0]["conclusion"], "bound")
        self.assertEqual(report["figure_claims"]["objects"][0]["role"], "symbol")
        self.assertTrue(report["figure_claims"]["objects"][0]["needs_review"])
        self.assertEqual(summary["meaningful_objects"], 0)
        self.assertEqual(summary["excluded_from_meaning"], 1)
        self.assertTrue(record["drawn"])
        self.assertFalse(record["as_block"])
        self.assertTrue(record["inside_other_graphics"])
        self.assertEqual(samples["as_block"], 0)
        self.assertEqual(samples["inside_other_graphics"], 1)

    def test_a_block_our_reading_never_put_text_on_is_unresolved_and_stays_counted(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Decoder")])
        visual = visual_graph(nodes=[vis_block("b1", "b1"), vis_block("b2", "Decoder")])

        _, report = build_layer_bindings(scientific, visual)
        block = next(item for item in report["figure_claims"]["objects"] if item["object_id"] == "b1")
        summary = report["figure_claims"]["summary"]

        self.assertEqual(block["conclusion"], "unresolved")
        self.assertIsNone(block["role"])
        self.assertTrue(block["needs_review"])
        self.assertEqual(block["claims"][0]["method"], "unreadable_label")
        self.assertEqual(summary["unlabeled_objects"], 1)
        self.assertEqual(summary["excluded_from_meaning"], 0)
        self.assertEqual(summary["meaningful_objects"], 2)
        self.assertEqual(summary["claim_coverage"], 0.5)

    def test_an_arrow_touching_the_figures_own_notation_is_that_notation_too(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(
            nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Legend")],
            edges=[{"id": "a1", "source": "b1", "target": "b2", "relation_type": "forward"}],
        )

        _, report = build_layer_bindings(scientific, visual)
        arrow = next(item for item in report["figure_claims"]["objects"] if item["object_id"] == "a1")
        summary = report["figure_claims"]["summary"]
        legend = next(item for item in report["figure_claims"]["objects"] if item["object_id"] == "b2")

        self.assertEqual(legend["conclusion"], "figure_annotation")
        self.assertEqual(arrow["conclusion"], "figure_annotation")
        self.assertEqual(arrow["claims"][0]["method"], "endpoint_figure_annotation")
        self.assertFalse(arrow["needs_review"])
        self.assertEqual(summary["annotation_arrows"], 1)
        self.assertEqual(summary["outside_paper_method"], 0)

    def test_an_arrow_into_an_object_nothing_explains_stays_unresolved(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(
            nodes=[vis_block("b1", "Legend"), vis_block("b2", "Unclear thing")],
            edges=[{"id": "a1", "source": "b1", "target": "b2", "relation_type": "forward"}],
        )

        _, report = build_layer_bindings(scientific, visual)
        arrow = next(item for item in report["figure_claims"]["objects"] if item["object_id"] == "a1")

        self.assertEqual(arrow["conclusion"], "unresolved")
        self.assertEqual(report["figure_claims"]["summary"]["annotation_arrows"], 0)

    def test_a_step_the_reading_named_stays_content_rather_than_becoming_notation(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        step = vis_block("b2", "Vocabulary head")
        step["properties"] = {"role": "method_object"}
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), step])
        review = self.review([self.concept("s1", "Encoder")])
        searched = []

        def provenance(phrase):
            searched.append(phrase)
            return None

        _, report = build_layer_bindings(scientific, visual, review, provenance=provenance)
        block = next(item for item in report["figure_claims"]["objects"] if item["object_id"] == "b2")
        summary = report["figure_claims"]["summary"]

        self.assertEqual(block["role"], "method_object")
        self.assertEqual(block["conclusion"], "unresolved")
        self.assertEqual(searched, ["Vocabulary head"])
        self.assertEqual(summary["figure_annotation"], 0)
        self.assertEqual(summary["meaningful_objects"], 2)
        self.assertEqual(summary["claim_coverage"], 0.5)

    def test_a_step_the_reading_named_and_the_paper_prints_keeps_its_page(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        step = vis_block("b2", "Vocabulary head")
        step["properties"] = {"role": "method_object"}
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), step])
        review = self.review([self.concept("s1", "Encoder")])
        provenance = (
            lambda phrase: {"page": 7, "snippet": "the vocabulary head", "matches": 2}
            if phrase == "Vocabulary head"
            else None
        )

        _, report = build_layer_bindings(scientific, visual, review, provenance=provenance)
        block = next(item for item in report["figure_claims"]["objects"] if item["object_id"] == "b2")

        self.assertEqual(block["conclusion"], "in_paper_not_in_method")
        self.assertEqual(block["provenance"]["page"], 7)
        self.assertEqual(block["claims"][0]["method"], "paper_text_match")

    def test_a_step_the_reading_named_but_could_not_read_is_our_gap_not_the_papers(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        step = vis_block("b2", "b2")
        step["properties"] = {"role": "method_object", "source_id": "b2", "text_confidence": 0}
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), step])
        review = self.review([self.concept("s1", "Encoder")])
        searched = []

        def provenance(phrase):
            searched.append(phrase)
            return {"page": 1, "snippet": "b2", "matches": 1}

        _, report = build_layer_bindings(scientific, visual, review, provenance=provenance)
        block = next(item for item in report["figure_claims"]["objects"] if item["object_id"] == "b2")
        summary = report["figure_claims"]["summary"]

        self.assertEqual(block["conclusion"], "unresolved")
        self.assertTrue(block["needs_review"])
        self.assertEqual(block["claims"][0]["method"], "unreadable_label")
        self.assertEqual(searched, [])
        self.assertEqual(summary["unlabeled_objects"], 1)
        self.assertEqual(summary["provenance_searched"], 0)

    def test_a_row_the_reading_calls_another_method_leaves_the_paper(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        row = vis_block("b2", "PTO baseline")
        row["properties"] = {"role": "external_method"}
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), row])
        review = self.review([self.concept("s1", "Encoder")])

        _, report = build_layer_bindings(scientific, visual, review)
        block = next(item for item in report["figure_claims"]["objects"] if item["object_id"] == "b2")

        self.assertEqual(block["conclusion"], "outside_paper_method")
        self.assertEqual(block["role"], "external_method")
        self.assertEqual(block["claims"][0]["method"], "reader_role")
        self.assertFalse(block["needs_review"])

class NotationRegionTests(unittest.TestCase):
    """A legend the review declared, and a region the reading drew, are the figure's own."""

    def review(self, *, region_annotations=(), unlisted_content=()):
        return {
            "contract": "rfs.figure_text_review",
            "paper_id": "p0001",
            "concepts": [],
            "region_annotations": [dict(item) for item in region_annotations],
            "unlisted_content": [dict(item) for item in unlisted_content],
        }

    def claim(self, report, object_id):
        return next(item for item in report["figure_claims"]["objects"] if item["object_id"] == object_id)

    def test_a_region_the_review_declared_is_the_figures_own_notation(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_block("b2", "Legend")])
        review = self.review(region_annotations=[{"label": "Legend", "kind": "legend", "explains": "arrow styles", "where": "bottom"}])

        _, report = build_layer_bindings(scientific, visual, review)
        legend = self.claim(report, "b2")
        summary = report["figure_claims"]["summary"]

        self.assertEqual(legend["conclusion"], "figure_annotation")
        self.assertEqual(legend["claims"][0]["method"], "reader_declared_annotation")
        self.assertEqual(legend["claims"][0]["explains"], "arrow styles")
        self.assertFalse(legend["needs_review"])
        self.assertEqual(summary["declared_region_labels"], 1)
        self.assertEqual(summary["outside_paper_method"], 0)
        self.assertEqual(summary["claim_coverage"], 1.0)

    def test_a_concept_that_claims_the_object_outranks_the_region_declaration(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Legend")])
        visual = visual_graph(nodes=[vis_block("b1", "Legend")])
        review = self.review(region_annotations=[{"label": "Legend", "kind": "legend", "explains": "", "where": ""}])

        _, report = build_layer_bindings(scientific, visual, review)
        legend = self.claim(report, "b1")

        self.assertEqual(legend["conclusion"], "bound")

    def test_unlisted_content_is_what_the_review_says_is_not_the_method(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_block("b2", "BERT baseline")])
        review = self.review(unlisted_content=[{"label": "BERT baseline", "where": "row 2"}])

        _, report = build_layer_bindings(scientific, visual, review)
        other = self.claim(report, "b2")

        self.assertEqual(other["conclusion"], "outside_paper_method")
        self.assertEqual(other["claims"][0]["method"], "reader_declared_outside")
        self.assertEqual(report["figure_claims"]["summary"]["declared_outside_labels"], 1)

    def test_a_region_the_reading_drew_is_a_notation_region_outside_the_headline(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[
            vis_block("b1", "Encoder"),
            {"id": "r1", "kind": "notation_region", "label": "Legend", "panel_ids": [],
             "properties": {"kind": "legend", "explains": "arrow styles", "entries": [{"glyph": "3x3", "meaning": "3x3 conv"}]}},
        ])

        _, report = build_layer_bindings(scientific, visual, None)
        region = self.claim(report, "r1")
        summary = report["figure_claims"]["summary"]

        self.assertEqual(region["kind"], "notation_region")
        self.assertEqual(region["conclusion"], "figure_annotation")
        self.assertEqual(region["role"], "stage_annotation")
        self.assertEqual(region["claims"][0]["method"], "figure_notation_region")
        self.assertEqual(summary["notation_regions"], 1)
        self.assertEqual(summary["meaningful_objects"], 1)
        self.assertEqual(summary["claim_coverage"], 1.0)

    def test_an_arrow_reaching_a_region_belongs_to_the_notation_layer(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(
            nodes=[
                vis_block("b1", "Encoder"),
                {"id": "r1", "kind": "notation_region", "label": "Legend", "panel_ids": [], "properties": {"kind": "legend"}},
            ],
            edges=[{"id": "a1", "source": "b1", "target": "r1", "relation_type": "forward"}],
        )

        _, report = build_layer_bindings(scientific, visual, None)
        arrow = self.claim(report, "a1")

        self.assertEqual(arrow["conclusion"], "figure_annotation")
        self.assertEqual(report["figure_claims"]["summary"]["annotation_arrows"], 1)

    def test_the_review_s_form_of_a_concept_is_recorded_beside_the_local_one(self):
        scientific = scientific_graph(nodes=[sci_node("s1", "Encoder")])
        visual = visual_graph(nodes=[vis_block("b1", "Encoder"), vis_block("b2", "p(c|x)")])
        review = {
            "contract": "rfs.figure_text_review",
            "paper_id": "p0001",
            "concepts": [
                {"concept_id": "s1", "image_label": "Encoder", "visible": True, "reader_how": "inside_other_graphics"},
            ],
        }

        _, report = build_layer_bindings(scientific, visual, review)
        samples = report["figure_claims"]["selection_samples"]
        record = next(item for item in samples["records"] if item["concept_id"] == "s1")

        self.assertEqual(record["reader_how"], "inside_other_graphics")
        self.assertTrue(record["as_block"])
        self.assertEqual(samples["reader_how_conflicts"], 1)


if __name__ == "__main__":
    unittest.main()
