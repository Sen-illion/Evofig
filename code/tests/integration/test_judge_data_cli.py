import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rfs.cli import build_parser, main


class JudgeDataCliTests(unittest.TestCase):
    def test_judge_data_build_and_validate_parse(self):
        parser = build_parser()
        args = parser.parse_args(["judge-data", "build", "--runs", "runs", "--out", "datasets/judge-v001"])
        self.assertEqual(args.command, "judge-data")
        self.assertEqual(args.judge_data_action, "build")
        self.assertEqual(args.dataset_version, "judge-dataset-v003")
        args = parser.parse_args(["judge-data", "validate", "--dataset", "datasets/judge-v001"])
        self.assertEqual(args.judge_data_action, "validate")
        args = parser.parse_args(["judge-data", "freeze-suite", "--dataset", "data.jsonl", "--out", "hidden.manifest.json"])
        self.assertEqual(args.judge_data_action, "freeze-suite")

    def test_trained_judge_commands_parse(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "register", "--checkpoint", "models/judge-v1", "--manifest", "manifest.json", "--registry", "registry.json"])
        self.assertEqual(args.judge_action, "register")
        args = parser.parse_args(["judge", "shadow", "--dataset", "data.jsonl", "--predictions", "trained.jsonl", "--bootstrap-predictions", "bootstrap.jsonl"])
        self.assertEqual(args.judge_action, "shadow")
        args = parser.parse_args(["judge", "evaluate", "--dataset", "data.jsonl", "--predictions", "trained.jsonl", "--suite-manifest", "hidden.json", "--report-out", "reports", "--strict-gates"])
        self.assertTrue(args.strict_gates)
        args = parser.parse_args(["judge", "baseline", "--dataset", "data", "--out", "out", "--limit", "5"])
        self.assertEqual(args.judge_action, "baseline")
        self.assertEqual(args.limit, 5)
        args = parser.parse_args(["judge", "bakeoff", "--dataset", "data", "--out", "out", "--model", "m1"])
        self.assertEqual(args.models, ["m1"])

    def test_render_figure_pages_parse(self):
        parser = build_parser()
        args = parser.parse_args(["judge-data", "render-figure-pages", "--batch-root", "batch", "--domains", "rag", "--dpi", "200", "--dry-run"])
        self.assertEqual(args.judge_data_action, "render-figure-pages")
        self.assertEqual(args.domains, ["rag"])
        self.assertEqual(args.dpi, 200)
        self.assertTrue(args.dry_run)
        args = parser.parse_args(["judge-data", "render-figure-pages", "--batch-root", "batch"])
        self.assertEqual(args.dpi, 150)
        self.assertEqual(args.limit, 0)
        self.assertFalse(args.dry_run)

    def test_judge_data_export_parse(self):
        parser = build_parser()
        args = parser.parse_args(["judge-data", "export", "--dataset", "datasets/judge-d1-v001", "--out", "training"])
        self.assertEqual(args.judge_data_action, "export")

    def test_judge_data_audit_parse(self):
        parser = build_parser()
        args = parser.parse_args(["judge-data", "audit", "--dataset", "datasets/judge-v001", "--strict"])
        self.assertEqual(args.judge_data_action, "audit")
        self.assertTrue(args.strict)
        self.assertIsNone(args.out)
        plain = parser.parse_args(["judge-data", "audit", "--dataset", "datasets/judge-v001"])
        self.assertFalse(plain.strict)

    def test_judge_data_audit_fails_only_under_strict_when_a_quota_is_unmet(self):
        row = {
            "example_id": "run-0001",
            "ground_truth_id": "direct-run-a",
            "task_type": "repair",
            "split": "",
            "labels": {"status": "pass", "stage_feedback": []},
            "observed": {"judge": {"judge_role": "online_bootstrap", "prompt_version": "judge-online-v006"}},
            "preview": "",
            "provenance": {"round": 0, "source": "direct_evolution_process"},
            "judge_metadata": {"judge_role": "online_bootstrap", "trained": False, "human_audited": False},
        }
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "ds"
            dataset.mkdir()
            (dataset / "judge_examples.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertFalse((dataset / "audit.json").exists())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["judge-data", "audit", "--dataset", str(dataset), "--strict", "--json"]), 1)
                self.assertEqual(main(["judge-data", "audit", "--dataset", str(dataset), "--json"]), 0)
            report = json.loads((dataset / "audit.json").read_text(encoding="utf-8"))
            self.assertFalse(report["clean"])
            self.assertTrue(report["ok"])
            self.assertTrue((dataset / "audit.md").is_file())

    def test_only_direct_generation_commands_are_registered(self):
        parser = build_parser()
        commands = set(parser._subparsers._group_actions[0].choices)
        self.assertIn("paper-evolve-editable", commands)
        self.assertIn("paper-evolve-validate", commands)
        removed = {"paper-to-" + "image", "paper-to-" + "editable", "make-framework", "fast-framework-prompt"}
        self.assertFalse(commands.intersection(removed))


if __name__ == "__main__":
    unittest.main()
