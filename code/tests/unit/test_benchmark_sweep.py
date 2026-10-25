"""Unit tests for scripts/benchmark_sweep.py.

The sweep driver shells out to the real chain, so these tests cover the parts
that decide *what* gets run and *what* gets recorded: command construction for
each experience mode, instance discovery, resume handling and the timeout path.
The chain itself is never executed.
"""

import importlib.util
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_sweep_module():
    spec = importlib.util.spec_from_file_location("benchmark_sweep", ROOT / "scripts" / "benchmark_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load_sweep_module()


def args_for(tmp_root, **overrides):
    values = {
        "python": "/usr/bin/python",
        "runs": str(pathlib.Path(tmp_root) / "runs"),
        "checkout": "/checkout",
        "experience_mode": "isolated",
        "timeout": 60.0,
        "workers": 4,
        "resume": False,
        "instances": str(pathlib.Path(tmp_root) / "instances"),
        "report": None,
        "offline_smoke": False,
        "git": "git",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class BuildCommandTest(unittest.TestCase):
    def test_isolated_gives_each_instance_its_own_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp)
            run_dir = pathlib.Path(args.runs) / "2401.00001"
            command = sweep.build_command(args, "2401.00001", "/pdfs/2401.00001.pdf", run_dir)
            self.assertEqual(command[:3], ["/usr/bin/python", "-m", "rfs.cli"])
            self.assertIn("paper-evolve-editable", command)
            self.assertEqual(command[command.index("--experience-library") + 1], str(run_dir / "experience_library"))
            self.assertEqual(command[-1], "--json")
            self.assertNotIn("--no-experience", command)

    def test_shared_uses_one_library_under_the_run_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp, experience_mode="shared")
            command = sweep.build_command(args, "2401.00001", "/pdfs/a.pdf", pathlib.Path(args.runs) / "2401.00001")
            self.assertEqual(command[command.index("--experience-library") + 1], str(pathlib.Path(args.runs) / "experience_library"))

    def test_off_disables_experience_entirely(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp, experience_mode="off")
            command = sweep.build_command(args, "2401.00001", "/pdfs/a.pdf", pathlib.Path(args.runs) / "2401.00001")
            self.assertIn("--no-experience", command)
            self.assertNotIn("--experience-library", command)


class DiscoveryTest(unittest.TestCase):
    def make_instances(self, tmp, names, with_contract):
        root = pathlib.Path(tmp) / "instances"
        for name in names:
            child = root / name
            child.mkdir(parents=True)
            if name in with_contract:
                (child / "contract.json").write_text("{}", encoding="utf-8")
        return root

    def test_only_contract_bearing_directories_are_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_instances(tmp, ["b", "a", "c"], {"b", "a"})
            self.assertEqual(sweep.discover_instances(root), ["a", "b"])

    def test_limit_takes_a_sorted_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_instances(tmp, ["b", "a", "c"], {"a", "b", "c"})
            self.assertEqual(sweep.discover_instances(root, limit=2), ["a", "b"])

    def test_missing_root_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                sweep.discover_instances(pathlib.Path(tmp) / "nope")


class ReportPathTest(unittest.TestCase):
    def test_default_lives_under_the_run_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp)
            self.assertEqual(sweep.report_path(args), pathlib.Path(args.runs) / "sweep_report.jsonl")

    def test_explicit_path_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp, report=str(pathlib.Path(tmp) / "records.jsonl"))
            self.assertEqual(sweep.report_path(args), pathlib.Path(tmp) / "records.jsonl")


class RunOneTest(unittest.TestCase):
    def test_resume_skips_a_delivered_instance_without_shelling_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp, resume=True)
            run_dir = pathlib.Path(args.runs) / "2401.00001"
            run_dir.mkdir(parents=True)
            (run_dir / "editable_composition.pptx").write_bytes(b"deck")
            with mock.patch.object(sweep.subprocess, "run") as run:
                record = sweep.run_one(1, "2401.00001", "/pdfs/a.pdf", args, {})
            run.assert_not_called()
            self.assertEqual(record["skipped"], "already_delivered")
            self.assertTrue(record["pptx"])

    def test_successful_chain_records_seconds_and_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp)
            run_dir = pathlib.Path(args.runs) / "2401.00001"

            def fake_run(command, **kwargs):
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "editable_composition.pptx").write_bytes(b"deck")
                (run_dir / "paper_evolve_editable_report.json").write_text("{}", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with mock.patch.object(sweep.subprocess, "run", side_effect=fake_run) as run:
                record = sweep.run_one(1, "2401.00001", "/pdfs/a.pdf", args, {})
            self.assertEqual(run.call_args.kwargs["cwd"], "/checkout")
            self.assertEqual(record["returncode"], 0)
            self.assertTrue(record["pptx"])
            self.assertTrue(record["report_json"])
            self.assertGreaterEqual(record["seconds"], 0.0)

    def test_timeout_is_recorded_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp)
            with mock.patch.object(sweep.subprocess, "run", side_effect=sweep.subprocess.TimeoutExpired("cmd", 60)):
                record = sweep.run_one(1, "2401.00001", "/pdfs/a.pdf", args, {})
            self.assertEqual(record["returncode"], -1)
            self.assertFalse(record["pptx"])
            self.assertIn("timeout", record["stderr_tail"])


class EnvTest(unittest.TestCase):
    def test_dotenv_values_are_merged_and_names_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / ".env"
            path.write_text('# comment\nAPI_KEY="secret"\n\nMODEL_VLM=some-model\n', encoding="utf-8")
            env, names = sweep.load_env(path)
            self.assertEqual(env["API_KEY"], "secret")
            self.assertEqual(env["MODEL_VLM"], "some-model")
            self.assertEqual(names, ["API_KEY", "MODEL_VLM"])

    def test_missing_dotenv_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, names = sweep.load_env(pathlib.Path(tmp) / "absent.env")
            self.assertEqual(names, [])
            self.assertIn("PATH", env)


class ChainConfigurationTest(unittest.TestCase):
    """The configuration decides what the sweep measures, so it must be explicit."""

    def test_defaults_are_the_product_settings_not_the_offline_smoke_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp)
            command = sweep.build_command(args, "2401.00001", "/pdfs/a.pdf", pathlib.Path(args.runs) / "2401.00001")
            for flag, value in (
                ("--planner-mode", "vlm"),
                ("--method-mode", "vlm"),
                ("--structure-mode", "vlm"),
                ("--text-plan-mode", "vlm"),
                ("--ppt-plan-mode", "vlm"),
                ("--color-theme-mode", "vlm"),
                ("--icon-mode", "image2"),
                ("--visual-review-mode", "vlm"),
                ("--judge-mode", "vlm"),
            ):
                self.assertEqual(command[command.index(flag) + 1], value, flag)
            self.assertNotIn("heuristic", command)
            self.assertNotIn("placeholder", command)

    def test_offline_smoke_switch_reproduces_the_readme_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp, offline_smoke=True)
            command = sweep.build_command(args, "2401.00001", "/pdfs/a.pdf", pathlib.Path(args.runs) / "2401.00001")
            self.assertEqual(sweep.chain_configuration(args), sweep.OFFLINE_SMOKE_CONFIGURATION)
            self.assertEqual(command[command.index("--planner-mode") + 1], "heuristic")
            self.assertEqual(command[command.index("--icon-mode") + 1], "placeholder")

    def test_stage_overrides_are_kept_and_recorded_per_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp, judge_mode="heuristic")
            self.assertEqual(sweep.chain_configuration(args)["judge_mode"], "heuristic")
            with mock.patch.object(sweep.subprocess, "run", side_effect=sweep.subprocess.TimeoutExpired("cmd", 1)):
                record = sweep.run_one(1, "2401.00001", "/pdfs/a.pdf", args, {})
            self.assertEqual(record["configuration"]["judge_mode"], "heuristic")
            self.assertEqual(record["configuration"]["planner_mode"], "vlm")

    def test_model_backed_stage_without_credentials_is_refused(self):
        problems = sweep.configuration_problems(sweep.PRODUCT_CONFIGURATION, {})
        self.assertTrue(problems)
        self.assertIn("API_KEY", problems[0])

    def test_credentials_or_an_offline_configuration_may_start(self):
        env = {"API_BASE": "https://example.invalid", "API_KEY": "secret"}
        self.assertEqual(sweep.configuration_problems(sweep.PRODUCT_CONFIGURATION, env), [])
        self.assertEqual(sweep.configuration_problems(sweep.OFFLINE_SMOKE_CONFIGURATION, {}), [])


class ConfigurationManifestTest(unittest.TestCase):
    def test_manifest_records_the_configuration_without_secret_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp)
            path = sweep.config_path(args)
            manifest = sweep.write_configuration_manifest(path, sweep.PRODUCT_CONFIGURATION, args, ["API_KEY"])
            self.assertTrue(path.is_file())
            self.assertEqual(manifest["configuration"], sweep.PRODUCT_CONFIGURATION)
            self.assertEqual(manifest["env_keys"], ["API_KEY"])
            self.assertNotIn("secret", path.read_text(encoding="utf-8"))
            self.assertEqual(path.name, "sweep_config.json")

    def test_unknown_checkout_commit_is_reported_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(sweep.checkout_commit(args_for(tmp, checkout=None)), "")
            with mock.patch.object(sweep.subprocess, "run", side_effect=OSError("no git")):
                self.assertEqual(sweep.checkout_commit(args_for(tmp)), "")


if __name__ == "__main__":
    unittest.main()
