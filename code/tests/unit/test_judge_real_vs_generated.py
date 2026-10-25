"""Tests for the reference-versus-generated Judge example builder."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rfs.cli import _run_judge_data, build_parser
from rfs.judge_training.method_inventory import SELECTION_KEYS
from rfs.judge_training.preference_pairs import build_preference_pairs
from rfs.judge_training.preference_sources import SourceUnavailable, open_generation_source, open_middleware_source
from rfs.judge_training.real_vs_generated import (
    GATE_CHECK_DEFECTS,
    build_real_vs_generated_examples,
    gate_verdict,
    shared_contract,
)
from rfs.judge_training.schemas import validate_example


PAPER_ID = "2401.00001"
FOLDER = "10.1234_example"
NODES = [
    {"id": "n1", "kind": "input", "label": "encoder", "required": True, "evidence_ids": ["E1"]},
    {"id": "n2", "kind": "module", "label": "decoder", "required": True, "evidence_ids": ["E2"]},
]
EDGES = [{"id": "e1", "source": "n1", "target": "n2", "relation_type": "causal", "required": True, "evidence_ids": ["E2"]}]
GROUND_TRUTH = {
    "contract": "rfs.scientific_ground_truth",
    "version": 1,
    "paper_id": PAPER_ID,
    "graph_id": "sg",
    "required_node_ids": ["n1", "n2"],
    "required_edge_ids": ["e1"],
    "forbidden_labels": ["do not invent equipment"],
}
PASSED_GATE = {
    "contract": "rfs.grounding_gate",
    "version": 1,
    "status": "pass",
    "ok": True,
    "checks": {"graph_contracts_valid": True, "required_nodes_grounded": True, "required_edges_grounded": True, "forbidden_labels_absent": True},
    "blocking_issues": [],
    "node_coverage": 1.0,
    "edge_coverage": 1.0,
}
BLOCKED_GATE = {
    "contract": "rfs.grounding_gate",
    "version": 1,
    "status": "blocked",
    "ok": False,
    "checks": {"graph_contracts_valid": True, "required_nodes_grounded": True, "required_edges_grounded": False, "forbidden_labels_absent": True},
    "blocking_issues": ["grounding check failed: required_edges_grounded"],
    "node_coverage": 1.0,
    "edge_coverage": 0.0,
}


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + chr(10), encoding="utf-8")
    return path


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding="utf-8")
    return path


def _middleware(root, *, figures=1, inventory=None, visual=None):
    graphs = root / "graphs" / FOLDER
    _write(graphs / "scientific_graph.json", {"contract": "rfs.graph", "graph_id": "sg", "paper_id": PAPER_ID, "summary": "encoder drives decoder", "nodes": NODES, "edges": EDGES, "required_node_ids": ["n1", "n2"], "required_edge_ids": ["e1"], "forbidden_labels": ["do not invent equipment"]})
    _write(graphs / "visual_graph.json", visual if visual is not None else {"graph_id": "vg", "paper_id": PAPER_ID})
    _write(graphs / "method_figure_content.json", {"contract": "rfs.method_figure_content", "figure_goal": "encoder feeds decoder"})
    if inventory is not None:
        _write(graphs / "figure_specification.json", inventory)
    _write(graphs / "scientific_ground_truth.json", GROUND_TRUTH)
    for index in range(figures):
        image = root / "figures" / FOLDER / f"asset-{index}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"png")
    _jsonl(root / "graph_index.jsonl", [
        {"record_id": "r1", "paper_id": PAPER_ID, "domain": "cv", "metadata": {"source_graph": f"graphs/{FOLDER}/scientific_graph.json"}},
        {"record_id": "r2", "paper_id": PAPER_ID, "domain": "cv", "metadata": {"source_graph": f"graphs/{FOLDER}/visual_graph.json"}},
    ])
    _jsonl(root / "subgraph_index.jsonl", [{"subgraph_id": "s1", "paper_id": PAPER_ID, "motif": "linear"}])
    return root


def _run(root, *, gates=None, previews=None, contract=None, paper_id=PAPER_ID):
    _write(root / "scientific_graph.json", {"graph_id": "sg", "paper_id": paper_id})
    _write(root / "evolution_rounds.json", {"rounds": [{"round": 0, "selected_variant": 0, "domain": "cv"}]})
    _jsonl(root / "evolution_events.jsonl", [{"seq": 1, "event": "candidate_accepted", "round": 0}])
    _write(root / "scientific_ground_truth.json", GROUND_TRUTH if contract is None else contract)
    _write(root / "evidence_map.json", {"summary": "s", "source_path": "paper.pdf", "evidence": [{"id": "E1", "page": 1, "kind": "caption", "text": "encoder feeds decoder"}, {"id": "E2", "page": 2, "kind": "body", "text": "the decoder follows the encoder"}]})
    gates = {0: BLOCKED_GATE} if gates is None else gates
    previews = {0: True} if previews is None else previews
    for variant, gate in gates.items():
        folder = root / "round_00" / f"candidate_{variant:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        if previews.get(variant, True):
            (folder / "draft_preview.png").write_bytes(b"png")
        if gate is not None:
            _write(folder / "grounding_gate.json", gate)
    if not (root / "round_00").is_dir():
        (root / "round_00" / "candidate_00").mkdir(parents=True)
    return root


class SharedContractTests(unittest.TestCase):
    def test_one_contract_is_required_on_both_sides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth, run = open_middleware_source(_middleware(root / "version")), open_generation_source(_run(root / "run"))
            state = shared_contract(truth, run, FOLDER)

            self.assertEqual(state["sha256"], hashlib.sha256((root / "run" / "scientific_ground_truth.json").read_bytes()).hexdigest())
            self.assertEqual(state["body"]["required_edge_ids"], ["e1"])

    def test_a_disagreeing_contract_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = open_middleware_source(_middleware(root / "version"))
            run = open_generation_source(_run(root / "run", contract={**GROUND_TRUTH, "required_edge_ids": []}))

            with self.assertRaises(SourceUnavailable) as caught:
                shared_contract(truth, run, FOLDER)
            self.assertIn("disagree", str(caught.exception))

    def test_a_run_without_a_contract_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = open_middleware_source(_middleware(root / "version"))
            run = open_generation_source(_run(root / "run"))
            (root / "run" / "scientific_ground_truth.json").unlink()

            with self.assertRaises(SourceUnavailable) as caught:
                shared_contract(truth, run, FOLDER)
            self.assertIn("published no scientific ground truth", str(caught.exception))


class GateVerdictTests(unittest.TestCase):
    def test_failed_checks_map_onto_the_canonical_defect_vocabulary(self):
        judgement, defect_key = gate_verdict(BLOCKED_GATE)

        self.assertEqual(defect_key, "missing_relation")
        self.assertEqual(GATE_CHECK_DEFECTS["required_edges_grounded"], "missing_relation")
        self.assertEqual(judgement["preferred_candidate"], "A")
        self.assertEqual(judgement["diagnosis"][0]["type"], "wrong_arrow_target")
        self.assertEqual(judgement["diagnosis"][0]["figure"], "B")
        self.assertEqual(judgement["diagnosis"][0]["target"], "global:required_edges_grounded")
        self.assertIn("edge_coverage", judgement["diagnosis"][0]["evidence"])
        self.assertEqual(sorted(judgement["quality_scores"]), ["domain_fit", "layout_quality", "logical_flow", "readability", "scientific_fidelity"])

    def test_a_structural_failure_names_no_figure_defect(self):
        gate = {**BLOCKED_GATE, "checks": {"graph_contracts_valid": False, "required_nodes_grounded": True, "required_edges_grounded": True, "forbidden_labels_absent": True}}

        self.assertEqual(gate_verdict(gate), ({}, ""))

    def test_every_mapped_check_is_reported(self):
        gate = {**BLOCKED_GATE, "checks": {"graph_contracts_valid": True, "required_nodes_grounded": False, "required_edges_grounded": False, "forbidden_labels_absent": False}}
        judgement, defect_key = gate_verdict(gate)

        self.assertEqual(defect_key, "missing_core_entity")
        self.assertEqual([entry["type"] for entry in judgement["diagnosis"]], ["missing_core_entity", "wrong_arrow_target", "invented_module"])


class BuildRealVsGeneratedTests(unittest.TestCase):
    def _build(self, root, **kwargs):
        truth = open_middleware_source(_middleware(root / "version", **kwargs.pop("middleware_kwargs", {})))
        run = open_generation_source(_run(root / "run", **kwargs.pop("run_kwargs", {})))
        return build_real_vs_generated_examples(truth, run, root / "examples.jsonl", **kwargs)

    def _rows(self, root):
        return [json.loads(line) for line in (root / "examples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_a_blocked_candidate_becomes_a_reference_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self._build(root)
            rows = self._rows(root)

            self.assertTrue(report["ok"])
            self.assertEqual(report["counts"]["candidates"], 1)
            self.assertEqual(report["counts"]["examples"], 2)
            self.assertEqual(report["defect_distribution"], {"missing_relation": 1})
            self.assertEqual(report["diagnosis_distribution"], {"wrong_arrow_target": 1})
            self.assertEqual(report["contract"]["sha256"], hashlib.sha256((root / "version" / "graphs" / FOLDER / "scientific_ground_truth.json").read_bytes()).hexdigest())
            row = rows[0]
            self.assertEqual(row["task_type"], "pairwise_preference")
            self.assertEqual(row["labels"]["preferred_candidate"], "A")
            self.assertEqual(row["provenance"]["corruption_type"], "missing_relation")
            self.assertEqual(row["provenance"]["defect_source"], "grounding_gate")
            self.assertEqual(row["provenance"]["grounding_gate"]["blocking_issues"], ["grounding check failed: required_edges_grounded"])
            self.assertEqual(report["version"], 3)
            self.assertEqual(row["dataset_version"], "judge-dataset-v005")
            self.assertEqual(row["ground_truth"]["required_entities"], ["encoder", "decoder"])
            self.assertEqual(row["ground_truth"]["required_relations"], ["encoder -> decoder (causal)"])
            self.assertEqual(row["ground_truth"]["method_summary"], "encoder feeds decoder")
            self.assertEqual([span["evidence_id"] for span in row["paper"]["evidence_spans"]], ["E1", "E2"])
            self.assertEqual(validate_example(row), [])

    def test_the_sample_carries_what_the_authors_figure_says_about_itself(self):
        visual = {
            "graph_id": "vg", "paper_id": PAPER_ID,
            "nodes": [
                {"id": "b1", "kind": "module", "label": "Encoder", "properties": {"role": "method_object"}},
                {"id": "b2", "kind": "module", "label": "Diffusion", "properties": {"role": "method_object", "repeat_group": "diffusion", "repeat_index": 1, "repeat_count": 3}},
                {"id": "b3", "kind": "module", "label": "Diffusion", "properties": {"role": "method_object", "repeat_group": "diffusion", "repeat_index": 2, "repeat_count": 3}},
                {"id": "r1", "kind": "notation_region", "label": "Legend", "properties": {"kind": "legend", "explains": "arrow styles", "entries": [{"glyph": "3x3", "meaning": "3x3 conv"}]}},
                {"id": "b4", "kind": "module", "label": "BERT baseline", "properties": {"role": "external_method"}},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self._build(root, middleware_kwargs={"visual": visual})
            row = self._rows(root)[0]
            block = row["ground_truth"]["reference_figure"]

            self.assertTrue(block["available"])
            self.assertEqual(block["objects"], 5)
            self.assertEqual(block["roles"], {"external_method": 1, "method_object": 3})
            self.assertEqual([item["group"] for item in block["repeat_drawn"]], ["diffusion", "diffusion"])
            self.assertEqual(block["repeat_drawn"][0]["count"], 3)
            self.assertEqual(block["notation_regions"], [{"kind": "legend", "explains": "arrow styles", "label": "Legend", "entries": 1}])
            self.assertEqual(report["reference_figure"]["objects"], 5)
            self.assertEqual(report["reference_figure"]["repeat_drawn"], 2)
            self.assertEqual(validate_example(row), [])

    def test_a_figure_nobody_read_is_reported_as_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self._build(root)
            block = self._rows(root)[0]["ground_truth"]["reference_figure"]

            self.assertFalse(block["available"])
            self.assertEqual(block["objects"], 0)
            self.assertEqual(block["roles"], {})
            self.assertEqual(block["repeat_drawn"], [])
            self.assertEqual(report["reference_figure"]["available"], False)

    def test_a_judge_reads_the_papers_inventory_and_never_the_figures_selection(self):
        # The middleware's plan says "encoder feeds decoder"; the inventory stored beside
        # it is what a Judge may read, and the plan's own selection keys stay out of the
        # sample, because a Judge must not be scored against that plan's taste.
        inventory = {
            "figure_goal": "the paper's own goal",
            "modules": [{
                "id": "m1", "label": "encoder", "role": "module", "detail_level": "module",
                "evidence_ids": ["E1"], "visualization_value": 1.0, "must_show": True,
                "should_render": True, "render_priority": 1,
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build(root, middleware_kwargs={"inventory": inventory})
            row = self._rows(root)[0]

            self.assertEqual("the paper's own goal", row["ground_truth"]["method_summary"])
            for key in SELECTION_KEYS:
                self.assertNotIn(f'"{key}"', json.dumps(row, ensure_ascii=False))

    def test_the_swapped_copy_mirrors_the_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build(root)
            row, swapped = self._rows(root)

            self.assertTrue(swapped["example_id"].endswith("__swap"))
            self.assertEqual(swapped["pair_group"], row["pair_group"])
            self.assertEqual(swapped["candidate_a"], row["candidate_b"])
            self.assertEqual(swapped["candidate_b"], row["candidate_a"])
            self.assertEqual(swapped["labels"]["preferred_candidate"], "B")
            self.assertEqual(swapped["labels"]["diagnosis"][0]["figure"], "A")
            self.assertEqual(validate_example(swapped), [])

    def test_swapped_copies_can_be_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self._build(root, include_swapped=False)

            self.assertEqual(report["counts"]["examples"], 1)
            self.assertEqual(report["counts"]["swapped"], 0)

    def test_a_candidate_that_passed_the_gate_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            passing = {**BLOCKED_GATE, "status": "pass", "ok": True, "checks": {name: True for name in BLOCKED_GATE["checks"]}}

            with self.assertRaises(SourceUnavailable) as caught:
                self._build(root, run_kwargs={"gates": {0: passing}})
            self.assertIn("certifies it as worse", str(caught.exception))
            self.assertFalse((root / "examples.jsonl").exists())

    def test_a_candidate_without_a_gate_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SourceUnavailable):
                self._build(root, run_kwargs={"gates": {0: None}})
            self.assertFalse((root / "examples.jsonl").exists())

    def test_an_excluded_candidate_leaves_the_others_in_the_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            passing = {**BLOCKED_GATE, "status": "pass", "ok": True, "checks": {name: True for name in BLOCKED_GATE["checks"]}}
            report = self._build(root, run_kwargs={"gates": {0: passing, 1: BLOCKED_GATE}})
            rows = self._rows(root)

            self.assertEqual(report["counts"]["candidates"], 2)
            self.assertEqual([item["reason"] for item in report["exclusions"]], ["candidate_passed_grounding_gate"])
            self.assertEqual(len(rows), 2)
            self.assertIn("candidate_01", rows[0]["candidate_b"])

    def test_a_structural_gate_failure_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structural = {**BLOCKED_GATE, "checks": {"graph_contracts_valid": False, "required_nodes_grounded": True, "required_edges_grounded": True, "forbidden_labels_absent": True}}
            report = self._build(root, run_kwargs={"gates": {0: structural, 1: BLOCKED_GATE}})

            self.assertEqual([item["reason"] for item in report["exclusions"]], ["gate_evidence_does_not_describe_the_figure"])
            self.assertEqual(report["counts"]["examples"], 2)

    def test_a_candidate_without_a_preview_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self._build(root, run_kwargs={"gates": {0: BLOCKED_GATE, 1: BLOCKED_GATE}, "previews": {0: True, 1: False}})

            self.assertEqual([item["reason"] for item in report["exclusions"]], ["candidate_has_no_rendered_preview"])
            self.assertEqual(report["counts"]["examples"], 2)

    def test_a_paper_outside_the_middleware_version_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SourceUnavailable) as caught:
                self._build(root, run_kwargs={"paper_id": "2401.99999"})
            self.assertIn("no graph folder", str(caught.exception))

    def test_a_run_that_only_records_its_source_kind_is_matched_by_its_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = open_middleware_source(_middleware(root / "version"))
            run = open_generation_source(_run(root / FOLDER, paper_id="arxiv"))
            report = build_real_vs_generated_examples(truth, run, root / "examples.jsonl")
            row = self._rows(root)[0]

            self.assertEqual(report["recorded_paper_id"], "arxiv")
            self.assertEqual(report["folder_route"], "run_directory")
            self.assertEqual(report["paper_id"], PAPER_ID)
            self.assertEqual(report["middleware_folder"], FOLDER)
            self.assertEqual(row["paper_id"], PAPER_ID)
            self.assertEqual(row["ground_truth_id"], PAPER_ID)
            self.assertEqual(row["provenance"]["recorded_paper_id"], "arxiv")
            self.assertEqual(row["provenance"]["paper_identity_route"], "run_directory")
            self.assertEqual(validate_example(row), [])
    def test_a_missing_or_ambiguous_reference_figure_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = _middleware(root / "version", figures=0)
            run = open_generation_source(_run(root / "run"))
            with self.assertRaises(SourceUnavailable):
                build_real_vs_generated_examples(open_middleware_source(version), run, root / "examples.jsonl")

            root = Path(tmp) / "two"
            version = _middleware(root / "version", figures=2)
            run = open_generation_source(_run(root / "run"))
            with self.assertRaises(SourceUnavailable) as caught:
                build_real_vs_generated_examples(open_middleware_source(version), run, root / "examples.jsonl")
            self.assertIn("exactly one rendered figure", str(caught.exception))

    def test_a_non_framework_split_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                self._build(root, split="heldout")

    def test_the_examples_feed_the_preference_pair_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build(root)
            report = build_preference_pairs(root / "examples.jsonl", root / "pairs.jsonl")
            pairs = [json.loads(line) for line in (root / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertTrue(report["ok"])
            self.assertEqual(sorted(report["by_family"]), ["misdiagnosed", "missed_defect", "wrong_pick"])
            self.assertEqual(report["gating_pairs"], len(pairs))
            self.assertEqual(pairs[0]["chosen"]["source"], "grounding_gate")
            self.assertEqual(pairs[0]["preference_basis"]["known_defects"][0]["source"], "real_vs_generated")
            self.assertEqual(pairs[0]["images"][0], str(root / "version" / "figures" / FOLDER / "asset-0.png"))
            self.assertTrue(pairs[0]["prompt"])

class AdoptedContractTests(unittest.TestCase):
    """A run that adopted the middle layer contract is compared against the contract."""

    def _ports(self, root, **kwargs):
        truth = open_middleware_source(_middleware(root / "version"))
        return truth, open_generation_source(_run(root / "run", **kwargs))

    def _adopt(self, root):
        """File the middleware's own ground truth beside the run, as the workflow does."""

        adopted = root / "run" / "upstream_scientific_ground_truth.json"
        _write(adopted, GROUND_TRUTH)
        return adopted

    def test_the_adopted_contract_is_what_gets_compared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # The run's own file is compiled from its figure content plan, so it differs
            # from the middleware's by construction and can never match byte for byte.
            truth, run = self._ports(root, contract={**GROUND_TRUTH, "graph_id": "planned", "required_edge_ids": []})
            adopted = self._adopt(root)

            state = shared_contract(truth, run, FOLDER)

            self.assertEqual(state["route"], "adopted_upstream_contract")
            self.assertEqual(state["adopted_path"], str(adopted))
            self.assertEqual(state["sha256"], hashlib.sha256((root / "version" / "graphs" / FOLDER / "scientific_ground_truth.json").read_bytes()).hexdigest())
            self.assertEqual(state["body"]["required_edge_ids"], ["e1"])

    def test_an_adopted_copy_of_another_contract_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth, run = self._ports(root)
            _write(root / "run" / "upstream_scientific_ground_truth.json", {**GROUND_TRUTH, "paper_id": "someone-else"})

            with self.assertRaises(SourceUnavailable) as caught:
                shared_contract(truth, run, FOLDER)
            self.assertIn("not this paper's", str(caught.exception))

    def test_an_adopted_run_builds_pairs_from_the_contract_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth, run = self._ports(root, contract={**GROUND_TRUTH, "required_edge_ids": []})
            self._adopt(root)

            report = build_real_vs_generated_examples(truth, run, root / "examples.jsonl")
            row = [json.loads(line) for line in (root / "examples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][0]

            self.assertTrue(report["ok"])
            self.assertEqual(report["contract"]["route"], "adopted_upstream_contract")
            self.assertEqual(set(report["counts"]), {"candidates", "examples", "swapped", "excluded", "evidence_spans"})
            self.assertEqual(row["provenance"]["contract_route"], "adopted_upstream_contract")
            self.assertEqual(row["provenance"]["label_tier"], "gate_blocked")
            self.assertTrue(row["provenance"]["certified"])
            self.assertEqual(validate_example(row), [])


class PassedCandidateTierTests(unittest.TestCase):
    """A candidate the gate passed is still ranked below the paper's own figure."""

    def _ports(self, root, gates=None):
        truth = open_middleware_source(_middleware(root / "version"))
        return truth, open_generation_source(_run(root / "run", gates={0: PASSED_GATE} if gates is None else gates))

    def _report(self, root, name, payload):
        _write(root / "run" / "round_00" / "candidate_00" / name, payload)

    def _rows(self, root):
        return [json.loads(line) for line in (root / "examples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_a_passed_candidate_is_not_used_unless_the_caller_asks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth, run = self._ports(root)

            with self.assertRaises(SourceUnavailable) as caught:
                build_real_vs_generated_examples(truth, run, root / "examples.jsonl")
            self.assertIn("include_passed", str(caught.exception))

    def test_a_flagged_candidate_is_labelled_correct_but_ugly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth, run = self._ports(root)
            self._report(root, "density_quality_report.json", {"summary": "density", "status": "blocked", "sparse_panel_count": 2})

            report = build_real_vs_generated_examples(truth, run, root / "examples.jsonl", include_passed=True)
            row = self._rows(root)[0]

            self.assertEqual(report["tier_distribution"], {"correct_but_ugly": 1})
            self.assertTrue(report["include_passed_candidates"])
            self.assertEqual(row["provenance"]["label_tier"], "correct_but_ugly")
            self.assertEqual(row["provenance"]["defect_source"], "deterministic_quality_reports")
            self.assertTrue(row["provenance"]["certified"])
            self.assertFalse(row["provenance"]["human_audited"])
            self.assertEqual(row["provenance"]["quality_reports_flagged"], [{"report": "density_quality_report.json", "status": "blocked"}])
            self.assertEqual(row["labels"]["diagnosis"][0]["type"], "correct_but_ugly")
            self.assertEqual(row["labels"]["diagnosis"][0]["severity"], "low")
            self.assertEqual(row["labels"]["preferred_candidate"], "A")
            self.assertEqual(validate_example(row), [])

    def test_a_candidate_nothing_faulted_is_an_uncertified_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth, run = self._ports(root)

            report = build_real_vs_generated_examples(truth, run, root / "examples.jsonl", include_passed=True)
            row = self._rows(root)[0]

            self.assertEqual(report["tier_distribution"], {"assumed_worse": 1})
            self.assertEqual(row["provenance"]["label_tier"], "assumed_worse")
            self.assertEqual(row["provenance"]["defect_source"], "ranking_without_evidence")
            self.assertFalse(row["provenance"]["certified"])
            self.assertEqual(row["provenance"]["quality_reports_flagged"], [])
            self.assertEqual(row["labels"]["diagnosis"][0]["type"], "assumed_worse")
            self.assertEqual(row["labels"]["confidence"], 0.5)
            self.assertEqual(validate_example(row), [])

    def test_a_report_that_passed_stays_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth, run = self._ports(root)
            self._report(root, "geometry_quality_report.json", {"status": "pass", "node_overlap_count": 0})
            self._report(root, "connector_review.json", {"status": "applied", "selections": []})

            report = build_real_vs_generated_examples(truth, run, root / "examples.jsonl", include_passed=True)

            self.assertEqual(report["tier_distribution"], {"assumed_worse": 1})

    def test_a_failed_report_without_a_status_still_flags_the_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth, run = self._ports(root)
            self._report(root, "joint_geometry_report.json", {"ok": False, "issues": ["panel overlaps title band"]})

            report = build_real_vs_generated_examples(truth, run, root / "examples.jsonl", include_passed=True)
            row = self._rows(root)[0]

            self.assertEqual(report["tier_distribution"], {"correct_but_ugly": 1})
            self.assertEqual(row["provenance"]["quality_reports_flagged"], [{"report": "joint_geometry_report.json", "status": "ok_false"}])

    def test_a_blocked_candidate_still_outranks_a_passed_one_in_the_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocked = _run(root / "run", gates={0: BLOCKED_GATE, 1: PASSED_GATE})

            truth = open_middleware_source(_middleware(root / "version"))
            report = build_real_vs_generated_examples(truth, open_generation_source(blocked), root / "examples.jsonl", include_passed=True)

            self.assertEqual(report["tier_distribution"], {"assumed_worse": 1, "gate_blocked": 1})
            self.assertEqual(report["defect_distribution"], {"missing_relation": 1, "assumed_worse": 1})


class BuildRealPairsCommandTests(unittest.TestCase):
    def test_the_parser_exposes_judge_data_build_real_pairs(self):
        args = build_parser().parse_args(["judge-data", "build-real-pairs", "--middleware", "v", "--run", "r", "--out", "o.jsonl", "--no-swapped"])

        self.assertEqual(args.judge_data_action, "build-real-pairs")
        self.assertFalse(args.include_swapped)
        self.assertEqual(args.split, "train")
        self.assertEqual(args.limit, 0)

    def test_the_parser_exposes_the_passed_candidate_switch(self):
        args = build_parser().parse_args([
            "judge-data", "build-real-pairs", "--middleware", "v", "--run", "r", "--out", "o.jsonl",
            "--count-passed-as-worse",
        ])

        self.assertTrue(args.include_passed)

    def test_the_dispatch_writes_the_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _middleware(root / "version")
            _run(root / "run")
            args = build_parser().parse_args([
                "judge-data", "build-real-pairs", "--middleware", str(root / "version"), "--run", str(root / "run"),
                "--out", str(root / "examples.jsonl"), "--split", "validation"])
            report = _run_judge_data(args)
            rows = [json.loads(line) for line in (root / "examples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertTrue(report["ok"])
            self.assertEqual(report["split"], "validation")
            self.assertEqual(len(rows), 2)
            self.assertTrue((root / "examples.report.json").exists())


if __name__ == "__main__":
    unittest.main()
