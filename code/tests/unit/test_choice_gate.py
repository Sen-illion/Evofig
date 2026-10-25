"""The choice gate table: hidden, per-class, adversarial, honesty, and promotion roles."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.judge_training.choice_gate import (
    CHOICE_GATE_THRESHOLDS,
    build_choice_gate_report,
    pair_table,
)


def _side(pair_id: str, kind: str, defect: str, *, correct: bool, legal: bool = True) -> dict:
    """One scored side. The answer names the slot the certified figure does not sit in."""

    certified = "A" if kind == "gold_in_A" else "B"
    wrong = "B" if certified == "A" else "A"
    return {
        "example_id": f"{pair_id}__{kind}",
        "pair_id": pair_id,
        "paper_id": f"paper-{pair_id}",
        "image_kind": kind,
        "gold_figure": wrong,
        "gold_defect": defect,
        "legal": legal,
        "pred_figure": (wrong if correct else certified) if legal else "",
        "pred_defect": (defect if correct else "no_defect") if legal else "",
    }


def _clean_rows() -> list[dict]:
    rows: list[dict] = []
    for defect in ("module_order_error", "wrong_arrow_target", "overlap"):
        for index in range(2):
            pair_id = f"{defect}-{index}"
            rows.append(_side(pair_id, "gold_in_A", defect, correct=True))
            rows.append(_side(pair_id, "gold_in_B", defect, correct=True))
    return rows


def _write(rows: list[dict], path: Path) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


class PairTableTests(unittest.TestCase):
    def test_a_pair_is_right_only_when_both_orientations_are(self):
        rows = [
            _side("p1", "gold_in_A", "overlap", correct=True),
            _side("p1", "gold_in_B", "overlap", correct=False),
            _side("p2", "gold_in_A", "overlap", correct=True),
            _side("p2", "gold_in_B", "overlap", correct=True),
        ]
        table = pair_table(rows)
        self.assertFalse(table["p1"]["figure_ok"])
        self.assertTrue(table["p2"]["figure_ok"])
        self.assertTrue(table["p2"]["defect_ok"])


class ChoiceGateTests(unittest.TestCase):
    def test_a_clean_run_promotes_to_online_and_leaves_frozen_unmeasured(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(_clean_rows(), Path(tmp) / "pred.jsonl")
            report = build_choice_gate_report(predictions=path, dataset="wide-exam2.jsonl", model_id="judge-9b-x")
        self.assertTrue(report["hidden"]["passed"])
        self.assertTrue(report["class_gates"]["passed"])
        self.assertTrue(report["adversarial"]["passed"])
        self.assertIsNotNone(report["shadow"])
        self.assertFalse(report["shadow"]["measured"])
        self.assertFalse(report["shadow"]["stable"])
        self.assertTrue(report["promotion"]["approved"])
        self.assertIn("online", report["promotion"]["roles"])
        self.assertNotIn("frozen", report["promotion"]["roles"])
        self.assertEqual(report["samples"], {"sides": 12, "pairs": 6})
        self.assertEqual(report["next_blockers"], [])

    def test_a_missed_defect_class_blocks_readiness_and_names_itself(self):
        rows = _clean_rows()
        # Every arrow pair now names a different defect, so the class-recall gate collapses.
        rows = [dict(row, pred_defect="no_defect") if row["gold_defect"] == "wrong_arrow_target" else row for row in rows]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(rows, Path(tmp) / "pred.jsonl")
            report = build_choice_gate_report(predictions=path, model_id="judge-9b-x")
        self.assertFalse(report["class_gates"]["classes"]["wrong_arrow_target"]["passed"])
        self.assertEqual(report["class_gates"]["classes"]["wrong_arrow_target"]["value"], 0.0)
        self.assertTrue(report["class_gates"]["classes"]["overlap"]["passed"])
        self.assertFalse(report["promotion"]["approved"])
        self.assertEqual(report["promotion"]["roles"], ["experimental"])
        self.assertIn("wrong_arrow_target", [entry["metric"] for entry in report["next_blockers"]])

    def test_an_illegal_answer_fails_legality_even_when_the_rest_is_right(self):
        rows = _clean_rows()
        rows = [dict(row, legal=False, pred_figure="", pred_defect="") if row["pair_id"] == "overlap-0" else row for row in rows]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(rows, Path(tmp) / "pred.jsonl")
            report = build_choice_gate_report(predictions=path)
        legal = next(entry for entry in report["hidden"]["gates"] if entry["metric"] == "legal_rate")
        self.assertEqual(legal["value"], round(10 / 12, 4))
        self.assertFalse(legal["passed"])
        self.assertFalse(report["hidden"]["passed"])

    def test_the_adversarial_slot_gates_split_the_two_orientations(self):
        rows = _clean_rows()
        # Naming the slot the certified figure never sits in passes legality but not the mirror.
        rows = [dict(row, pred_figure="A") if row["image_kind"] == "gold_in_A" else row for row in rows]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(rows, Path(tmp) / "pred.jsonl")
            report = build_choice_gate_report(predictions=path)
        self.assertFalse(report["adversarial"]["cases"]["certified_in_a"]["passed"])
        self.assertTrue(report["adversarial"]["cases"]["certified_in_b"]["passed"])

    def test_the_slot_gates_read_an_older_run_that_has_no_image_kind(self):
        # Exam JSONL built before image_kind existed leaves the field empty; the swapped
        # copy is only marked by its __swap id, and the gate must still split the slots.
        rows = _clean_rows()
        for row in rows:
            if row["image_kind"] == "gold_in_B":
                row["example_id"] = f"{row['example_id']}__swap"
            row["image_kind"] = ""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(rows, Path(tmp) / "pred.jsonl")
            report = build_choice_gate_report(predictions=path)
        self.assertEqual(report["adversarial"]["cases"]["certified_in_a"]["samples"], 6)
        self.assertEqual(report["adversarial"]["cases"]["certified_in_b"]["samples"], 6)
        self.assertTrue(report["adversarial"]["passed"])

    def test_threshold_overrides_and_reports_land_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = _write(_clean_rows(), root / "pred.jsonl")
            report = build_choice_gate_report(
                predictions=path,
                model_id="judge-9b-x",
                dataset_version="choice-v001",
                checkpoint="ckpt-010000",
                git_commit="deadbee",
                thresholds={"figure_accuracy": 1.0},
                report_out=root / "gate",
            )
            written = json.loads((root / "gate" / "choice_gate.json").read_text(encoding="utf-8"))
            markdown = (root / "gate" / "choice_gate.md").read_text(encoding="utf-8")
            manifest = json.loads((root / "gate" / "registry_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(report["thresholds"]["figure_accuracy"], 1.0)
        self.assertEqual(report["thresholds"]["overlap"], CHOICE_GATE_THRESHOLDS["overlap"])
        self.assertEqual(written["schema_version"], "judge-choice-gate-v1")
        self.assertIn("Choice Judge gate table", markdown)
        self.assertEqual(manifest["model_id"], "judge-9b-x")
        self.assertEqual(manifest["checkpoint"], "ckpt-010000")
        self.assertIn("online", manifest["roles"])


if __name__ == "__main__":
    unittest.main()
