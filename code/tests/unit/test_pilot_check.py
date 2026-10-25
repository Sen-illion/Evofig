"""Unit tests for the figure-to-text correspondence review of one Pilot paper."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from rfs.judge_training import pilot_graphs
from rfs.judge_training.pilot_check import (
    CHECK_PROMPT_VERSION,
    CHECK_STEP,
    CORRESPONDENCE_CONTRACT,
    LEDGER_SCHEMA_VERSION,
    MAX_CONCEPTS,
    best_block,
    build_check_prompt,
    check_alias_map,
    check_all,
    check_coverage,
    check_paper,
    normalize_check_payload,
    required_concepts,
    required_relations,
    visual_blocks,
    visual_targets,
)

ENCODER = "snode.0e93488250f6efdc"
DECODER = "snode.4b1cf4f8c0e2b8a1"
OUTPUT = "snode.7f2ad1c9b83e5d04"
FLOW = "sedge.212dc33d10303bf8"
PANEL = "vnode.893fffd7d1e267a3"
ENCODER_BLOCK = "vnode.1b0c9a4e7d2f6305"
DECODER_BLOCK = "vnode.5c3e8a1f0b9d7412"


def node(node_id, label, *, kind="module", required=True):
    return {"id": node_id, "kind": kind, "label": label, "required": required, "evidence_ids": ["ev-" + node_id[-4:]]}


def edge(edge_id, source, target, *, relation="data_flow", required=True):
    return {"id": edge_id, "source": source, "target": target, "relation_type": relation, "required": required, "evidence_ids": ["ev-" + edge_id[-4:]]}


def scientific_graph():
    return {
        "graph_id": "scientific-p0001",
        "paper_id": "p0001",
        "kind": "scientific",
        "nodes": [node(ENCODER, "Frozen encoder"), node(DECODER, "Decoder", required=False), node(OUTPUT, "Figure output", kind="artifact")],
        "edges": [edge(FLOW, ENCODER, DECODER)],
    }


def visual_graph(*, blocks=None):
    return {
        "graph_id": "visual-p0001",
        "paper_id": "p0001",
        "kind": "visual",
        "nodes": blocks
        if blocks is not None
        else [
            {"id": PANEL, "kind": "panel", "label": "panel one"},
            {"id": ENCODER_BLOCK, "kind": "module", "label": "Frozen encoder"},
            {"id": DECODER_BLOCK, "kind": "module", "label": "Decoder block v2"},
        ],
        "edges": [],
    }


def row(paper_id="p0001"):
    return {"paper_id": paper_id, "split": "train", "domain": "cv", "figure_relative_path": "assets/paper-p0001/figures/figure.png"}


def check_ids(graph=None):
    """Return the short check id the prompt prints for each scientific id."""
    graph = graph if graph is not None else scientific_graph()
    concepts, relations = check_alias_map(required_concepts(graph), required_relations(graph))
    return {**concepts, **relations}


def answer(**overrides):
    ids = check_ids()
    payload = {
        "image_summary": "Two boxes with one arrow.",
        "concepts": [
            {"id": ids[ENCODER], "visible": True, "image_label": "Frozen encoder", "where": "left", "confidence": 0.9},
            {"id": ids[DECODER], "visible": True, "image_label": "Decoder", "where": "right", "confidence": 0.7},
            {"id": ids[OUTPUT], "visible": False, "image_label": "", "where": "", "confidence": 0.6},
        ],
        "relations": [{"id": ids[FLOW], "arrow_present": True, "direction_matches": True, "confidence": 0.8}],
        "extra_blocks": [{"label": "Loss", "where": "bottom"}],
        "uncertainties": ["one arrow label is tiny"],
    }
    payload.update(overrides)
    return payload


def fixture(root):
    """Write a snapshot with one paper and one figure asset."""
    snapshot = root / "paper-d0-v001"
    figures = snapshot / "assets" / "paper-p0001" / "figures"
    figures.mkdir(parents=True)
    Image.new("RGB", (6, 6), (255, 255, 255)).save(figures / "figure.png")
    (snapshot / "eligible_papers.jsonl").write_text(
        json.dumps({"paper_id": "p0001", "title": "A Frozen Study", "primary_domain": "cv"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (snapshot / "asset_manifest.jsonl").write_text(
        json.dumps({
            "paper_id": "p0001",
            "asset_id": "asset-p0001",
            "relative_path": "assets/paper-p0001/figures/figure.png",
            "sha256": "",
            "suffix": ".png",
            "role": "figure",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return snapshot


def fake_provider(payload, *, tokens=(900, 300)):
    """Return a provider stand-in that answers the correspondence checklist."""

    def call(prompt, image_paths, model=None, timeout=180, retries=1, call_metadata=None, deadline_at=None):
        if call_metadata is not None:
            call_metadata.update({
                "attempts": 1, "success": True, "elapsed_seconds": 0.02, "failure_categories": [],
                "prompt_chars": len(prompt), "estimated_tokens": max(1, round(len(prompt) / 4)),
                "image_count": len(image_paths), "image_bytes": 0,
                "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens)},
                "token_source": "provider", "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens),
            })
        return payload

    return call


class CheckPromptTests(unittest.TestCase):
    def test_required_concepts_puts_required_first_and_caps(self):
        graph = {"nodes": [node("c", "C", required=False), node("a", "A"), node("b", "B")]}
        self.assertEqual([item["id"] for item in required_concepts(graph)], ["a", "b", "c"])
        capped = required_concepts({"nodes": [node(f"n{index:02d}", f"Node {index}") for index in range(MAX_CONCEPTS + 5)]})
        self.assertEqual(len(capped), MAX_CONCEPTS)

    def test_required_relations_puts_required_first(self):
        graph = {"edges": [edge("e2", "a", "b", required=False), edge("e1", "a", "b")]}
        self.assertEqual([item["id"] for item in required_relations(graph)], ["e1", "e2"])

    def test_prompt_lists_the_contract_and_never_the_stored_read(self):
        prompt = build_check_prompt(scientific_graph())
        ids = check_ids()
        self.assertIn(ids[ENCODER] + ' | "Frozen encoder" | kind=module | required=yes', prompt)
        self.assertIn(ids[FLOW] + ' | "Frozen encoder" -> "Decoder" | data_flow | required=yes', prompt)
        self.assertIn("Return exactly one JSON object", prompt)
        self.assertIn("never invent an id", prompt)
        self.assertNotIn("Decoder block v2", prompt)
        self.assertIn("(none declared)", build_check_prompt({"nodes": [], "edges": []}))

    def test_prompt_asks_with_short_ids_and_hides_the_hashes(self):
        prompt = build_check_prompt(scientific_graph())
        for scientific_id in (ENCODER, DECODER, OUTPUT, FLOW):
            self.assertNotIn(scientific_id, prompt)
        self.assertNotIn("snode", prompt)
        self.assertNotIn("sedge", prompt)

    def test_aliases_number_the_contract_in_prompt_order(self):
        ids = check_ids()
        self.assertEqual(ids[OUTPUT], "c1")
        self.assertEqual(ids[ENCODER], "c2")
        self.assertEqual(ids[DECODER], "c3")
        self.assertEqual(ids[FLOW], "r1")
        self.assertEqual(len(set(ids.values())), 4)

    def test_visual_blocks_skip_panels(self):
        blocks = visual_blocks(visual_graph())
        self.assertEqual([item["id"] for item in blocks], [ENCODER_BLOCK, DECODER_BLOCK])
        self.assertEqual(visual_blocks(None), [])

    def test_visual_targets_keep_the_panel_label(self):
        targets = visual_targets(visual_graph())

        self.assertEqual([item["id"] for item in targets], [PANEL, ENCODER_BLOCK, DECODER_BLOCK])
        self.assertEqual(visual_targets(None), [])

    def test_best_block_scores_labels_and_respects_the_threshold(self):
        blocks = visual_blocks(visual_graph())
        self.assertEqual(best_block("Frozen encoder", blocks), (ENCODER_BLOCK, 1.0))
        self.assertEqual(best_block("DECODER BLOCK", blocks)[0], DECODER_BLOCK)
        self.assertEqual(best_block("Something else entirely", blocks)[0], "")
        self.assertEqual(best_block("", blocks), ("", 0.0))

    def test_best_block_reaches_a_row_label(self):
        targets = visual_targets(visual_graph())

        self.assertEqual(best_block("PANEL ONE", targets), (PANEL, 1.0))


class CheckNormalizeTests(unittest.TestCase):
    def normalize(self, payload, *, visual=None, graph=None):
        return normalize_check_payload(
            payload,
            paper_id="p0001",
            graph_id="scientific-p0001",
            row=row(),
            scientific_graph=graph if graph is not None else scientific_graph(),
            visual_graph=visual if visual is not None else visual_graph(),
            image={"source_sha256": "abc", "png_sha256": "def", "width": 6, "height": 6},
            prompt_chars=1234,
        )

    def test_verdicts_cover_every_case(self):
        ids = check_ids()
        payload = answer(concepts=[
            {"id": ids[ENCODER], "visible": True, "image_label": "Frozen encoder", "confidence": 0.9},
            {"id": ids[DECODER], "visible": True, "image_label": "Decoder block v2", "confidence": 0.8},
            {"id": ids[OUTPUT], "visible": True, "image_label": "Figure output", "confidence": 0.7},
        ])
        review, report = self.normalize(payload)
        verdicts = {item["concept_id"]: item["verdict"] for item in review["concepts"]}
        printed = {item["concept_id"]: item["check_id"] for item in review["concepts"]}
        self.assertTrue(report["ok"])
        self.assertEqual(verdicts[ENCODER], "confirmed")
        self.assertEqual(verdicts[DECODER], "confirmed")
        self.assertEqual(verdicts[OUTPUT], "missed_by_reader")
        self.assertEqual(printed[ENCODER], ids[ENCODER])
        self.assertEqual(printed[OUTPUT], ids[OUTPUT])
        self.assertEqual(review["contract"], CORRESPONDENCE_CONTRACT)
        self.assertEqual(review["prompt_version"], CHECK_PROMPT_VERSION)
        self.assertEqual(review["image"]["png_sha256"], "def")
        self.assertEqual(review["extra_blocks"], [{"label": "Loss", "where": "bottom"}])

    def test_verdicts_report_hallucinated_not_drawn_and_unreadable(self):
        ids = check_ids()
        payload = answer(concepts=[
            {"id": ids[ENCODER], "visible": False, "confidence": 0.9},
            {"id": ids[DECODER], "visible": False, "confidence": 0.9},
        ])
        review, report = self.normalize(payload, visual={"nodes": [{"id": ENCODER_BLOCK, "kind": "module", "label": "Frozen encoder"}], "edges": []})
        verdicts = {item["concept_id"]: item["verdict"] for item in review["concepts"]}
        self.assertEqual(verdicts[ENCODER], "hallucinated_block")
        self.assertEqual(verdicts[DECODER], "not_drawn")
        self.assertEqual(verdicts[OUTPUT], "unreadable")
        self.assertTrue(any("could not be judged" in item for item in report["warnings"]))
        self.assertEqual(review["summary"]["reader_precision"], 0.0)
        self.assertEqual(review["summary"]["required_visible_rate"], 0.0)

    def test_relation_verdicts(self):
        ids = check_ids()
        confirmed, _ = self.normalize(answer())
        self.assertEqual([item["verdict"] for item in confirmed["relations"]], ["confirmed"])
        conflict, _ = self.normalize(answer(relations=[{"id": ids[FLOW], "arrow_present": True, "direction_matches": False}]))
        self.assertEqual([item["verdict"] for item in conflict["relations"]], ["direction_conflict"])
        undrawn, _ = self.normalize(answer(relations=[{"id": ids[FLOW], "arrow_present": False}]))
        self.assertEqual([item["verdict"] for item in undrawn["relations"]], ["not_drawn"])
        missing, _ = self.normalize(answer(relations=[]))
        self.assertEqual([item["verdict"] for item in missing["relations"]], ["unreadable"])
        self.assertEqual(missing["relations"][0]["source"], ENCODER)
        self.assertEqual(missing["relations"][0]["target"], DECODER)
        self.assertEqual(missing["relations"][0]["check_id"], ids[FLOW])

    def test_strays_and_summary_rates(self):
        ids = check_ids()
        payload = answer(concepts=[
            {"id": ids[ENCODER], "visible": True, "confidence": 0.9},
            {"id": "ghost", "visible": True, "confidence": 0.9},
            {"id": ids[DECODER], "visible": True, "confidence": 0.2},
        ])
        review, report = self.normalize(payload)
        summary = review["summary"]
        self.assertEqual(review["findings_outside_contract"], ["ghost"])
        self.assertTrue(any("outside the contract" in item for item in report["warnings"]))
        self.assertEqual(summary["concepts"], 3)
        self.assertEqual(summary["required_concepts"], 2)
        self.assertEqual(summary["confirmed"], 2)
        self.assertEqual(summary["missed_by_reader"], 0)
        self.assertEqual(summary["not_drawn"], 0)
        self.assertEqual(summary["unreadable"], 1)
        self.assertEqual(summary["required_unreadable"], 1)
        self.assertEqual(summary["required_visible_rate"], 1.0)
        self.assertEqual(summary["reader_precision"], 1.0)
        self.assertEqual(summary["reader_recall"], 1.0)
        self.assertEqual(summary["low_confidence_concepts"], 1)
        self.assertEqual(summary["extra_blocks"], 1)

    def test_an_answer_that_repeats_a_hash_id_is_a_stray(self):
        review, report = self.normalize(answer(concepts=[{"id": ENCODER, "visible": True, "confidence": 0.9}]))
        self.assertEqual(review["findings_outside_contract"], [ENCODER])
        self.assertEqual([item["verdict"] for item in review["concepts"]], ["unreadable", "unreadable", "unreadable"])
        self.assertTrue(any("outside the contract" in item for item in report["warnings"]))

    def test_a_contract_without_concepts_is_an_error(self):
        review, report = self.normalize(answer(), graph={"graph_id": "s", "nodes": [], "edges": []})
        self.assertFalse(report["ok"])
        self.assertEqual(report["errors"], ["the scientific graph declares no concept to audit"])


class CheckPaperTests(unittest.TestCase):
    def test_blocked_without_a_stored_layer(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot = fixture(root)
            entry, ledger, report = check_paper(snapshot, {"paper_id": "p0001"}, row(), None, visual_graph(), figure_dir=root / "figures")

        self.assertIsNone(entry)
        self.assertEqual(ledger["status"], "failed")
        self.assertEqual(ledger["step"], CHECK_STEP)
        self.assertEqual(ledger["token_source"], "not_called")
        self.assertIn("no stored scientific graph", ledger["error"])
        self.assertEqual(report["errors"], ["no stored scientific graph for this paper"])

    def test_refuses_to_render_into_a_sealed_version(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot = fixture(root)
            version = root / "graphs"
            (version / "figures").mkdir(parents=True)
            (version / "COMPLETE").write_text("{}", encoding="utf-8")
            entry, ledger, _report = check_paper(
                snapshot, {"paper_id": "p0001"}, row(), scientific_graph(), visual_graph(),
                figure_dir=version / "figures",
            )

        self.assertIsNone(entry)
        self.assertEqual(ledger["status"], "failed")
        self.assertIn("sealed version", ledger["error"])

    def test_check_paper_records_the_review_and_the_provider_usage(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot = fixture(root)
            with patch("rfs.judge_training.pilot_check.call_vlm_json", fake_provider(answer())):
                entry, ledger, report = check_paper(
                    snapshot, {"paper_id": "p0001"}, row(), scientific_graph(), visual_graph(),
                    figure_dir=root / "figures", model="fake-model",
                )
                rendered = (root / "figures" / "p0001" / "figure.png").is_file()

        self.assertIsNotNone(entry)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(ledger["status"], "ok")
        self.assertEqual(ledger["token_source"], "provider")
        self.assertEqual(ledger["total_tokens"], 1200)
        self.assertEqual(ledger["image_count"], 1)
        self.assertEqual(ledger["concepts"], 3)
        self.assertEqual(ledger["confirmed"], 2)
        self.assertTrue(ledger["artifact_sha256"]["figure_text_review.json"])
        self.assertEqual(entry["figure_text_review"]["contract"], CORRESPONDENCE_CONTRACT)
        self.assertTrue(rendered)
        self.assertEqual(entry["check_report"]["prompt_version"], CHECK_PROMPT_VERSION)

    def test_a_failing_provider_leaves_no_review(self):
        def boom(prompt, image_paths, model=None, timeout=180, retries=1, call_metadata=None, deadline_at=None):
            raise RuntimeError("provider exploded")

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot = fixture(root)
            with patch("rfs.judge_training.pilot_check.call_vlm_json", boom):
                entry, ledger, _report = check_paper(
                    snapshot, {"paper_id": "p0001"}, row(), scientific_graph(), visual_graph(),
                    figure_dir=root / "figures", model="fake-model",
                )

        self.assertIsNone(entry)
        self.assertEqual(ledger["status"], "failed")
        self.assertIn("provider exploded", ledger["error"])


class CheckAllTests(unittest.TestCase):
    def test_check_all_needs_credentials_before_calling(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot = fixture(root)
            with patch("rfs.judge_training.pilot_check.vlm_credentials_available", return_value=False):
                with self.assertRaises(RuntimeError):
                    check_all(
                        snapshot, [row()], graphs={}, visuals={}, figure_dir=root / "figures",
                        assets={}, workers=1,
                    )

    def test_check_all_reviews_every_pending_paper(self):
        graphs = {"p0001": {"paper_id": "p0001", "scientific_graph": scientific_graph()}}
        visuals = {"p0001": {"paper_id": "p0001", "visual_graph": visual_graph()}}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot = fixture(root)
            with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
                with patch("rfs.judge_training.pilot_check.call_vlm_json", fake_provider(answer())):
                    entries, ledger, reports = check_all(
                        snapshot, [row()], graphs=graphs, visuals=visuals,
                        figure_dir=root / "figures", model="fake-model", workers=1,
                    )

        self.assertEqual(sorted(entries), ["p0001"])
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["status"], "ok")
        self.assertTrue(reports["p0001"]["ok"])
        coverage = check_coverage(entries, papers=3)
        self.assertEqual(coverage["reviewed_papers"], 1)
        self.assertEqual(coverage["papers"], 3)
        self.assertEqual(coverage["unreviewed_papers"], 2)
        self.assertEqual(coverage["confirmed"], 2)
        self.assertEqual(coverage["missed_by_reader"], 0)
        self.assertEqual(coverage["not_drawn"], 1)
        self.assertEqual(coverage["extra_blocks"], 1)
        self.assertEqual(coverage["papers_with_missed_concepts"], [])
        self.assertEqual(coverage["papers_with_hallucinated_blocks"], [])
        self.assertGreater(coverage["mean_required_visible_rate"], 0)

    def test_check_coverage_counts_conflicts_and_unreadable(self):
        ids = check_ids()
        payload = answer(
            concepts=[{"id": ids[ENCODER], "visible": True, "confidence": 0.9}],
            relations=[{"id": ids[FLOW], "arrow_present": True, "direction_matches": False, "confidence": 0.9}],
        )
        review, _report = normalize_check_payload(
            payload, paper_id="p0001", graph_id="s", row=row(),
            scientific_graph=scientific_graph(), visual_graph=visual_graph(),
        )
        coverage = check_coverage({"p0001": {"paper_id": "p0001", "figure_text_review": review}}, papers=1)
        self.assertEqual(coverage["direction_conflicts"], 1)
        self.assertEqual(coverage["papers_with_direction_conflicts"], ["p0001"])
        self.assertEqual(coverage["unreadable"], 2)
        self.assertEqual(coverage["papers_with_unreadable_concepts"], ["p0001"])


class ReadingFormTests(unittest.TestCase):
    """The second review says how the figure carries a concept, not only whether."""

    def normalize(self, payload, *, visual=None):
        return normalize_check_payload(
            payload,
            paper_id="p0001",
            graph_id="scientific-p0001",
            row=row(),
            scientific_graph=scientific_graph(),
            visual_graph=visual if visual is not None else visual_graph(),
            image={"source_sha256": "abc", "png_sha256": "def", "width": 6, "height": 6},
        )

    def test_prompt_asks_how_and_splits_the_figures_own_annotations(self):
        prompt = build_check_prompt(scientific_graph())

        self.assertIn('"Visible" does not mean "drawn as a block of its own"', prompt)
        for how in ("as_block", "inside_other_graphics", "not_drawn"):
            self.assertIn(how, prompt)
        self.assertIn("region_annotations", prompt)
        self.assertIn("unlisted_content", prompt)
        self.assertNotIn("extra_blocks", prompt)

    def test_a_review_keeps_the_annotations_apart_from_unlisted_content(self):
        ids = check_ids()
        payload = answer(
            concepts=[{"id": ids[ENCODER], "visible": True, "how": "inside_other_graphics", "image_label": "p(c|x)", "confidence": 0.9}],
            region_annotations=[{"label": "Legend", "kind": "legend", "explains": "arrow styles", "where": "bottom"}],
            unlisted_content=[{"label": "BERT baseline", "where": "row 2"}],
        )
        review, _report = self.normalize(payload)

        hows = {item["concept_id"]: item["reader_how"] for item in review["concepts"]}
        self.assertEqual(hows[ENCODER], "inside_other_graphics")
        self.assertEqual(review["region_annotations"], [{"label": "Legend", "kind": "legend", "explains": "arrow styles", "where": "bottom"}])
        self.assertEqual(review["unlisted_content"], [{"label": "BERT baseline", "where": "row 2"}])
        self.assertEqual(review["extra_blocks"], review["unlisted_content"])
        self.assertEqual(review["summary"]["region_annotations"], 1)
        self.assertEqual(review["summary"]["unlisted_content"], 1)
        self.assertEqual(review["summary"]["reader_how"]["inside_other_graphics"], 1)

    def test_an_unknown_how_answer_is_recorded_as_no_answer(self):
        ids = check_ids()
        payload = answer(concepts=[{"id": ids[ENCODER], "visible": True, "how": "drawn twice maybe", "confidence": 0.9}])
        review, _report = self.normalize(payload)

        hows = {item["concept_id"]: item["reader_how"] for item in review["concepts"]}
        self.assertEqual(hows[ENCODER], "")
        self.assertEqual(review["summary"]["reader_how_answered"], 0)

    def test_an_older_review_reads_its_extra_blocks_as_unlisted_content(self):
        review, _report = self.normalize(answer())

        self.assertEqual(review["unlisted_content"], [{"label": "Loss", "where": "bottom"}])
        self.assertEqual(review["region_annotations"], [])
        self.assertEqual(review["summary"]["unlisted_content"], 1)


class CheckVersionGateTests(unittest.TestCase):
    """A stored review only counts while it answers the question this version asks."""

    def test_check_all_reads_again_when_the_stored_review_used_another_prompt(self):
        graphs = {"p0001": {"paper_id": "p0001", "scientific_graph": scientific_graph()}}
        visuals = {"p0001": {"paper_id": "p0001", "visual_graph": visual_graph()}}
        stale = {"p0001": {"paper_id": "p0001", "figure_text_review": {"prompt_version": "pilot-figure-check-v1"}, "check_report": {}}}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot = fixture(root)
            with patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
                with patch("rfs.judge_training.pilot_check.call_vlm_json", fake_provider(answer())):
                    entries, ledger, _reports = check_all(
                        snapshot, [row()], graphs=graphs, visuals=visuals,
                        figure_dir=root / "figures", model="fake-model", workers=1,
                        existing_checks=stale,
                    )

        self.assertEqual([item["status"] for item in ledger], ["ok"])
        self.assertEqual(entries["p0001"]["figure_text_review"]["prompt_version"], CHECK_PROMPT_VERSION)

    def test_check_all_keeps_a_review_of_this_prompt(self):
        graphs = {"p0001": {"paper_id": "p0001", "scientific_graph": scientific_graph()}}
        visuals = {"p0001": {"paper_id": "p0001", "visual_graph": visual_graph()}}
        current = {"p0001": {
            "paper_id": "p0001",
            "figure_text_review": {"prompt_version": CHECK_PROMPT_VERSION},
            "check_report": {"prompt_version": CHECK_PROMPT_VERSION},
        }}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot = fixture(root)
            entries, ledger, reports = check_all(
                snapshot, [row()], graphs=graphs, visuals=visuals,
                figure_dir=root / "figures", workers=1, existing_checks=current,
            )

        self.assertEqual(ledger, [])
        self.assertEqual(reports, {})
        self.assertEqual(sorted(entries), ["p0001"])


class CheckStageContractTests(unittest.TestCase):
    def test_stage_registration_matches_the_orchestration(self):
        self.assertEqual(LEDGER_SCHEMA_VERSION, pilot_graphs.SCHEMA_VERSION)
        self.assertEqual(CHECK_STEP, "check")
        self.assertIn(CHECK_STEP, pilot_graphs.IMPLEMENTED_STAGES)
        self.assertIn(CHECK_STEP, pilot_graphs.BILLABLE_STEPS)
        self.assertLess(
            pilot_graphs.STAGE_ORDER.index(CHECK_STEP),
            pilot_graphs.STAGE_ORDER.index(pilot_graphs.BIND_STEP),
            "bind reuses the labels check read off the figure, so check has to run first",
        )
        self.assertEqual(pilot_graphs.PAPER_ARTIFACTS[CHECK_STEP], ("figure_text_review.json",))


if __name__ == "__main__":
    unittest.main()
