"""The choice scorer writes predictions and the choice gate decides on the same run."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.cli import build_parser, main
from rfs.judge_training.choice_gate import build_choice_gate_report
from rfs.judge_training.choice_items import evaluate_choice_records, render_pair


def _pair(pair_id: str, defect: str) -> dict:
    return {"pair_id": pair_id, "paper_id": pair_id, "domain": "cv", "family": "corruption", "defect": defect, "good": f"/tmp/{pair_id}-good.png", "bad": f"/tmp/{pair_id}-bad.png", "title": "Title", "evidence": "A method overview."}


def _records() -> list[dict]:
    """Both orientations of three pairs per gating defect, so every class gate has samples."""

    records = []
    for defect in ("module_order_error", "wrong_arrow_target", "overlap"):
        for index in range(3):
            pair = _pair(f"{defect}-{index}", defect)
            records.append(render_pair(pair, swapped=False))
            records.append(render_pair(pair, swapped=True))
    return records


class _Answerer:
    """A backend that echoes each record's own gold answer, so every pair should be right."""

    name = "answerer"

    def generate(self, record, max_new_tokens: int = 64) -> str:
        return json.dumps({"wrong_figure": record["gold_figure"], "defect": record["gold_defect"]})


class ChoiceGateChainTests(unittest.TestCase):
    def test_the_scored_run_is_the_run_the_gate_reads(self):
        records = _records()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.jsonl"
            score = evaluate_choice_records(_Answerer(), records, predictions_out=predictions)
            report = build_choice_gate_report(
                predictions=predictions,
                dataset="wide-exam2.jsonl",
                checkpoint="ckpt-003174",
                model_id="judge-9b-aesthetic-tail",
                dataset_version="choice-v001",
                git_commit="deadbee",
                report_out=root / "gate",
            )
            written = json.loads((root / "gate" / "choice_gate.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "gate" / "registry_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(score["pairs"], report["samples"]["pairs"])
        self.assertEqual(score["figure_accuracy"], report["hidden"]["gates"][0]["value"])
        self.assertEqual(score["defect_accuracy"], report["hidden"]["gates"][1]["value"])
        self.assertEqual(score["legal_rate"], report["hidden"]["gates"][2]["value"])
        self.assertTrue(report["adversarial"]["passed"])
        self.assertEqual(written["samples"]["sides"], score["rows"])
        self.assertEqual(manifest["checkpoint"], "ckpt-003174")
        self.assertEqual(manifest["dataset_version"], "choice-v001")
        self.assertEqual(manifest["git_commit"], "deadbee")

    def test_both_choice_commands_parse_and_route(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "score-choice", "--dataset", "train.jsonl", "--out", "score.json", "--base-model", "qwen", "--predictions-out", "pred.jsonl"])
        self.assertEqual(args.judge_action, "score-choice")
        self.assertEqual(args.predictions_out, "pred.jsonl")
        args = parser.parse_args(["judge", "gate-choice", "--predictions", "pred.jsonl", "--report-out", "gate", "--strict-gates"])
        self.assertEqual(args.judge_action, "gate-choice")
        self.assertEqual(args.predictions, "pred.jsonl")
        self.assertTrue(args.strict_gates)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [render_pair(_pair(f"overlap-{index}", "overlap"), swapped=swap) for index, swap in enumerate((False, True))]
            path = root / "pred.jsonl"
            evaluate_choice_records(_Answerer(), records, predictions_out=path)
            code = main(["judge", "gate-choice", "--predictions", str(path), "--report-out", str(root / "gate"), "--json"])
            self.assertEqual(code, 0)
            self.assertTrue((root / "gate" / "choice_gate.json").is_file())
            empty = root / "empty.jsonl"
            empty.write_text("\n", encoding="utf-8")
            self.assertEqual(main(["judge", "gate-choice", "--predictions", str(empty), "--json"]), 1)


if __name__ == "__main__":
    unittest.main()
