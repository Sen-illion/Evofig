"""Tests for the per-class gates, the adversarial checklist and the promotion chain."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.cli import _run_judge, build_parser
from rfs.judge_training.evaluator import build_suite_manifest
from rfs.judge_training.promotion import (
    ADVERSARIAL_CASES,
    GATE_CLASSES,
    adversarial_case_ids,
    adversarial_gates,
    build_promotion_report,
    per_class_gates,
    promotion_decision,
    shadow_gate,
)


CASE_IDS = sorted(str(case["case_id"]) for case in ADVERSARIAL_CASES)
GROUNDING = {
    "grounded_entities": ["encoder", "decoder"],
    "ungrounded_entities": ["lstm"],
    "missing_relations": [],
    "relation_directions": {"encoder->decoder": "forward"},
}
POLISHED = {"scientific_fidelity": 0.0, "logical_flow": 0.1, "layout_quality": 0.8, "readability": 0.9, "domain_fit": 0.3}
CLEAN = {"scientific_fidelity": 1.0, "logical_flow": 1.0, "layout_quality": 1.0, "readability": 1.0, "domain_fit": 1.0}


def _example(example_id, *, task_type="pairwise_preference", defect=None, severity="high", pick=None, layout=0.8,
             pair_group=None, adversarial_type=None, feedback=None, repeat_group=None, domain="cv", grounding=GROUNDING):
    diagnosis = [{"type": defect, "target": "encoder->decoder", "severity": severity, "evidence": "the arrow is reversed"}] if defect else []
    scores = dict(CLEAN if defect is None else POLISHED)
    scores["layout_quality"] = layout
    labels = {"quality_label": "bad" if defect else "good", "quality_scores": scores, "diagnosis": diagnosis, "confidence": 1.0}
    if pick:
        labels["preferred_candidate"] = pick
    if grounding is not None:
        labels["grounding"] = dict(grounding)
    if feedback is not None:
        labels["feedback_effective"] = feedback
    row = {
        "example_id": example_id,
        "dataset_version": "judge-dataset-v003",
        "ground_truth_id": "arxiv-" + example_id,
        "paper_id": example_id,
        "domain": domain,
        "split": "heldout",
        "task_type": task_type,
        "candidate_a": example_id + "-a.png",
        "candidate_b": example_id + "-b.png",
        "labels": labels,
        "ground_truth": {"method_summary": "encoder to decoder", "required_entities": ["encoder", "decoder"], "required_relations": ["encoder->decoder"]},
    }
    if pair_group:
        row["pair_group"] = pair_group
    if adversarial_type:
        row["adversarial_type"] = adversarial_type
    if repeat_group:
        row["repeat_group"] = repeat_group
    return row


def _judged(row, *, types=None, pick=None, feedback=None, grounding=GROUNDING, scores=None):
    labels = row["labels"]
    names = [entry["type"] for entry in labels["diagnosis"]] if types is None else list(types)
    output = {
        "quality_scores": dict(scores or labels["quality_scores"]),
        "diagnosis": [{"type": name, "target": "encoder->decoder", "severity": "high", "evidence": "seen in the figure"} for name in names],
        "confidence": 1.0,
    }
    if grounding is not None:
        output["grounding"] = dict(grounding)
    choice = pick if pick is not None else labels.get("preferred_candidate")
    if choice:
        output["preferred_candidate"] = choice
    if feedback is not None:
        output["feedback_effective"] = feedback
    elif "feedback_effective" in labels:
        output["feedback_effective"] = labels["feedback_effective"]
    return {"example_id": row["example_id"], "prediction": output}


ORDER = _example("order-a", defect="module_order_error", pick="A", pair_group="grp-order", adversarial_type="beautiful_but_wrong")
ORDER_SWAP = _example("order-b", defect="module_order_error", pick="B", pair_group="grp-order", adversarial_type="beautiful_but_wrong")
ARROW = _example("arrow-a", defect="wrong_arrow_target", pick="A", pair_group="grp-arrow", layout=0.9)
OVERLAP_MILD = _example("overlap-mild", defect="overlap", severity="low", pick="A", pair_group="grp-mild", layout=0.1)
OVERLAP_SEVERE = _example("overlap-severe", defect="overlap", severity="high", pick="A", pair_group="grp-severe", layout=0.1)
STYLE = _example("style-a", task_type="style_preference", pick="A", layout=1.0, grounding=None)
REPAIR = _example("repair-a", task_type="repair", feedback=True, grounding=None)
REPEAT_A = _example("repeat-a", pick="A", repeat_group="r1")
REPEAT_B = _example("repeat-b", pick="A", repeat_group="r1")
SUITE = [ORDER, ORDER_SWAP, ARROW, OVERLAP_MILD, OVERLAP_SEVERE, STYLE, REPAIR, REPEAT_A, REPEAT_B]
PREDICTIONS = [_judged(row) for row in SUITE]


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding="utf-8")
    return path


def _suite(root, rows=SUITE):
    dataset = _write_rows(root / "judge_examples.jsonl", rows)
    manifest = root / "suite_manifest.json"
    build_suite_manifest(dataset, manifest, suite="hidden_test")
    return dataset, manifest


class PerClassGateTests(unittest.TestCase):
    def test_each_defect_class_is_gated_on_its_own_recall(self):
        missed = _example("order-c", defect="module_order_error", pick="A")
        gold = {"order-a": ORDER, "order-c": missed, "arrow-a": ARROW, "overlap-severe": OVERLAP_SEVERE}
        gates = per_class_gates(gold, [_judged(ORDER), _judged(missed, types=[]), _judged(ARROW), _judged(OVERLAP_SEVERE)])

        self.assertFalse(gates["passed"])
        self.assertEqual(gates["classes"]["module_order_error"]["recall"], 0.5)
        self.assertEqual(gates["classes"]["module_order_error"]["missed_example_ids"], ["order-c"])
        self.assertEqual(gates["classes"]["wrong_arrow_target"]["recall"], 1.0)
        self.assertEqual(gates["classes"]["overlap"]["recall"], 1.0)

    def test_an_uninjected_class_cannot_pass(self):
        gold = {"overlap-severe": OVERLAP_SEVERE}
        gates = per_class_gates(gold, [_judged(OVERLAP_SEVERE)])

        self.assertFalse(gates["passed"])
        self.assertTrue(gates["coverable"])
        self.assertEqual(gates["classes"]["overlap"]["passed"], True)
        self.assertIsNone(gates["classes"]["module_order_error"]["recall"])
        self.assertEqual(gates["classes"]["module_order_error"]["unavailable_reason"], "no_injected_sample")

    def test_thresholds_are_configurable_per_class(self):
        missed = _example("order-c", defect="module_order_error", pick="A")
        gold = {"order-a": ORDER, "order-c": missed, "arrow-a": ARROW, "overlap-severe": OVERLAP_SEVERE}
        predictions = [_judged(ORDER), _judged(missed, types=[]), _judged(ARROW), _judged(OVERLAP_SEVERE)]

        self.assertFalse(per_class_gates(gold, predictions)["passed"])
        relaxed = per_class_gates(gold, predictions, {"module_order_error": 0.5, "wrong_arrow_target": 0.5, "overlap": 0.5})
        self.assertTrue(relaxed["passed"])
        self.assertEqual(relaxed["classes"]["module_order_error"]["threshold"], 0.5)


class AdversarialGateTests(unittest.TestCase):
    def test_the_full_suite_covers_and_passes_every_checklist_case(self):
        gold = {row["example_id"]: row for row in SUITE}
        gates = adversarial_gates(gold, PREDICTIONS)

        self.assertEqual(sorted(gates["cases"]), CASE_IDS)
        self.assertEqual(gates["uncovered"], [])
        self.assertTrue(gates["passed"])
        self.assertEqual(gates["cases"]["ab_swapped"]["samples"], 1)
        self.assertEqual(gates["cases"]["overlap"]["samples"], 2)

    def test_a_checklist_case_the_dataset_does_not_cover_fails_the_gate(self):
        rows = [ORDER, ORDER_SWAP, ARROW, STYLE, REPAIR]
        gold = {row["example_id"]: row for row in rows}
        gates = adversarial_gates(gold, [_judged(row) for row in rows])

        self.assertFalse(gates["passed"])
        self.assertEqual(gates["uncovered"], ["overlap"])
        self.assertTrue(gates["cases"]["beautiful_but_order_error"]["gate_passed"])
        self.assertFalse(gates["cases"]["overlap"]["gate_passed"])
        self.assertEqual(gates["cases"]["overlap"]["unavailable_reason"], "case_not_covered_by_this_dataset")

    def test_the_swap_case_needs_both_directions(self):
        gold = {row["example_id"]: row for row in SUITE}
        stale = [_judged(row) if row["example_id"] != "order-b" else _judged(row, pick="A") for row in SUITE]
        gates = adversarial_gates(gold, stale)

        self.assertFalse(gates["cases"]["ab_swapped"]["gate_passed"])
        self.assertFalse(gates["cases"]["truth_not_in_A"]["gate_passed"])
        self.assertFalse(gates["passed"])
        self.assertIn("grp-order", gates["cases"]["ab_swapped"]["failures"][0])

    def test_case_ids_follow_polish_and_both_overlap_severities_share_one_case(self):
        self.assertEqual(adversarial_case_ids(ORDER), ["beautiful_but_order_error"])
        self.assertEqual(adversarial_case_ids(OVERLAP_MILD), ["overlap"])
        self.assertEqual(adversarial_case_ids(OVERLAP_SEVERE), ["overlap"])
        self.assertIn("truth_not_in_A", adversarial_case_ids(ORDER_SWAP))
        self.assertEqual(adversarial_case_ids(_example("ugly-order", defect="module_order_error", pick="A", layout=0.2)), [])
        self.assertEqual(adversarial_case_ids(_example("clean", layout=1.0)), [])


class PromotionDecisionTests(unittest.TestCase):
    def test_a_checkpoint_holding_every_element_claims_online_and_frozen(self):
        decision = promotion_decision(hidden_test_passed=True, class_gates_passed=True, adversarial_passed=True, shadow_stable=True)

        self.assertTrue(decision["approved"])
        self.assertEqual(decision["roles"], ["experimental", "shadow", "online", "frozen"])
        self.assertEqual(decision["advice"], "promote to frozen")
        self.assertEqual(decision["reasons"], [])

    def test_an_unstable_shadow_keeps_it_online_only(self):
        decision = promotion_decision(hidden_test_passed=True, class_gates_passed=True, adversarial_passed=True, shadow_stable=False)

        self.assertTrue(decision["approved"])
        self.assertEqual(decision["roles"], ["experimental", "shadow", "online"])
        self.assertEqual(decision["shadow_stable"], False)
        self.assertEqual(decision["reasons"], [])
        self.assertIn("shadow", decision["advice"])
        self.assertIn("shadow", decision["notes"][0])

    def test_every_approval_blocker_is_named(self):
        cases = {
            "hidden_test_passed": "hidden test",
            "class_gates_passed": "defect class",
            "adversarial_passed": "adversarial",
        }
        for field, needle in cases.items():
            with self.subTest(field=field):
                flags = {"hidden_test_passed": True, "class_gates_passed": True, "adversarial_passed": True, "shadow_stable": True}
                flags[field] = False
                decision = promotion_decision(**flags)
                self.assertFalse(decision["approved"])
                self.assertEqual(decision["roles"], ["experimental"])
                self.assertEqual(len(decision["reasons"]), 1)
                self.assertIn(needle, decision["reasons"][0])

    def test_a_failed_baseline_is_reported(self):
        decision = promotion_decision(hidden_test_passed=True, class_gates_passed=True, adversarial_passed=True, shadow_stable=True, baseline_passed=False)

        self.assertFalse(decision["approved"])
        self.assertIn("baseline", decision["reasons"][0])


class ShadowGateTests(unittest.TestCase):
    def _evaluation(self, *, recall, gate_pass=True, pairwise=0.9):
        return {"gate_pass": gate_pass, "metrics": {"scientific_error_recall": {"value": recall}, "pairwise_accuracy": {"value": pairwise}}}

    def test_a_regression_against_the_bootstrap_blocks_stability(self):
        gate = shadow_gate(self._evaluation(recall=0.8), self._evaluation(recall=1.0))

        self.assertFalse(gate["stable"])
        self.assertIn("scientific_error_recall", gate["reasons"][0])

    def test_a_matching_or_better_judge_stays_stable(self):
        self.assertTrue(shadow_gate(self._evaluation(recall=1.0), self._evaluation(recall=1.0))["stable"])
        self.assertTrue(shadow_gate(self._evaluation(recall=1.0), self._evaluation(recall=0.7))["stable"])

    def test_a_failing_shadow_suite_is_not_stable(self):
        gate = shadow_gate(self._evaluation(recall=None, gate_pass=False), self._evaluation(recall=0.5))

        self.assertFalse(gate["stable"])
        self.assertEqual(len(gate["reasons"]), 1)


class BuildPromotionReportTests(unittest.TestCase):
    def _training_report(self, root, *, steps=24, margin=0.42):
        metrics = {"steps": steps, "mean_margin": margin, "pairwise_accuracy": 0.9, "gated_recall": {name: 1.0 for name in GATE_CLASSES}}
        return self._dump(root / "training_report.json", {"metrics": metrics})

    def _dump(self, path, payload):
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_the_full_chain_answers_with_a_gate_table_and_a_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _suite(root)
            predictions = _write_rows(root / "predictions.jsonl", PREDICTIONS)
            report = build_promotion_report(
                dataset=dataset, predictions=predictions, suite_manifest=manifest, checkpoint=root / "checkpoints" / "step-24",
                baseline_predictions=_write_rows(root / "baseline.jsonl", PREDICTIONS),
                bootstrap_predictions=_write_rows(root / "bootstrap.jsonl", PREDICTIONS),
                training_report=self._training_report(root), report_out=root / "report")

            self.assertTrue(report["hidden_test"]["gate_pass"])
            self.assertTrue(report["class_gates"]["passed"])
            self.assertTrue(report["adversarial"]["passed"])
            self.assertTrue(report["shadow"]["stable"])
            self.assertTrue(report["promotion"]["approved"])
            self.assertIn("frozen", report["promotion"]["roles"])
            self.assertTrue(report["ok"])
            self.assertTrue(report["gate_pass"])
            self.assertEqual([entry["step"] for entry in report["chain"]], ["baseline", "training", "hidden_test", "shadow", "online", "frozen"])
            self.assertTrue(all(entry["status"] in {"passed", "reported", "ready"} for entry in report["chain"]))
            self.assertEqual(report["hidden_test"]["breakdowns"]["domain"]["cv"]["sample_count"], len(SUITE))
            self.assertEqual(report["training"]["steps"], 24)
            self.assertTrue((root / "report" / "promotion_report.json").exists())
            markdown = (root / "report" / "promotion_report.md").read_text(encoding="utf-8")
            self.assertIn("# Judge promotion report", markdown)
            self.assertIn("| Case |", markdown)
            self.assertIn("promote to frozen", markdown)

    def test_strict_gates_mark_a_blocked_checkpoint_as_not_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _suite(root)
            weak = [_judged(row) if row["example_id"] != "order-b" else _judged(row, types=[], pick="A") for row in SUITE]
            report = build_promotion_report(dataset=dataset, predictions=_write_rows(root / "weak.jsonl", weak), suite_manifest=manifest, strict_gates=True)

            self.assertFalse(report["ok"])
            self.assertFalse(report["hidden_test"]["gate_pass"])
            self.assertFalse(report["class_gates"]["passed"])
            self.assertFalse(report["adversarial"]["passed"])
            self.assertFalse(report["promotion"]["approved"])
            self.assertEqual(sorted(report["strict_gates_failed"]), ["adversarial", "class_gates", "hidden_test"])
            self.assertIn("hold and repair", report["promotion"]["advice"])

    def test_a_blocked_checkpoint_stays_exit_zero_without_strict_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _suite(root)
            weak = [_judged(row) if row["example_id"] != "order-b" else _judged(row, types=[], pick="A") for row in SUITE]
            report = build_promotion_report(dataset=dataset, predictions=_write_rows(root / "weak.jsonl", weak), suite_manifest=manifest)

            self.assertTrue(report["ok"])
            self.assertFalse(report["gate_pass"])
            self.assertFalse(report["promotion"]["approved"])

    def test_a_shadow_regression_blocks_the_frozen_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _suite(root)
            weaker = [_judged(row) if row["example_id"] != "repeat-b" else _judged(row, pick="B") for row in SUITE]
            report = build_promotion_report(
                dataset=dataset, predictions=_write_rows(root / "weaker.jsonl", weaker), suite_manifest=manifest,
                bootstrap_predictions=_write_rows(root / "bootstrap.jsonl", PREDICTIONS))

            self.assertTrue(report["hidden_test"]["gate_pass"])
            self.assertTrue(report["promotion"]["approved"])
            self.assertFalse(report["shadow"]["stable"])
            self.assertEqual(report["promotion"]["roles"], ["experimental", "shadow", "online"])
            self.assertEqual([entry["status"] for entry in report["chain"] if entry["step"] == "shadow"], ["failed"])
            self.assertEqual([entry["status"] for entry in report["chain"] if entry["step"] == "frozen"], ["blocked"])
            self.assertEqual([entry["status"] for entry in report["chain"] if entry["step"] == "online"], ["ready"])


class PromoteCommandTests(unittest.TestCase):
    def test_the_parser_exposes_judge_promote(self):
        args = build_parser().parse_args(["judge", "promote", "--dataset", "d.jsonl", "--predictions", "p.jsonl", "--strict-gates"])

        self.assertEqual(args.judge_action, "promote")
        self.assertTrue(args.strict_gates)
        self.assertEqual(args.suite, "hidden_test")
        self.assertAlmostEqual(args.adversarial_threshold, 0.9)

    def test_the_dispatch_builds_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _suite(root)
            args = build_parser().parse_args([
                "judge", "promote", "--dataset", str(dataset), "--suite-manifest", str(manifest),
                "--predictions", str(_write_rows(root / "predictions.jsonl", PREDICTIONS)),
                "--class-threshold", "0.5", "--report-out", str(root / "report")])
            report = _run_judge(args)

            self.assertEqual(report["promotion"]["approved"], True)
            self.assertEqual(report["promotion"]["roles"], ["experimental", "shadow", "online"])
            self.assertEqual(report["class_gates"]["classes"]["overlap"]["threshold"], 0.5)
            self.assertEqual(report["hidden_test"]["gate_pass"], True)


if __name__ == "__main__":
    unittest.main()
