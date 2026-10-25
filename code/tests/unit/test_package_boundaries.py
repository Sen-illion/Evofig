"""Structural gates for the sole direct EvoFig pipeline."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RFS = ROOT / "rfs"
FORBIDDEN_PACKAGES = {"paper_to_" + "image", "coevo" + "lution", "profes" + "sional", "re" + "build"}
FORBIDDEN_COMMANDS = {
    "paper-to-" + "image",
    "paper-to-" + "editable",
    "make-framework",
    "fast-framework-prompt",
    "re" + "build-evolve",
    "coevolve-" + "image",
    "coevo" + "lution-report",
}


class PackageBoundaryTests(unittest.TestCase):
    def test_direct_entry_points_are_importable(self):
        from rfs.analysis import parse_paper
        from rfs.composition import build_semantic_figure_program, compile_ppt, style_and_route_arrows
        from rfs.evaluation import build_direct_judges, validate_output
        from rfs.evolution.direct import ExperienceStore, validate_experience_cases
        from rfs.generation.icon_assets import generate_icon_assets
        from rfs.providers import call_vlm_json
        from rfs.workflows import run_direct_experience_validation, run_paper_evolve_editable

        for value in (
            parse_paper,
            build_semantic_figure_program,
            compile_ppt,
            style_and_route_arrows,
            build_direct_judges,
            validate_output,
            ExperienceStore,
            validate_experience_cases,
            generate_icon_assets,
            call_vlm_json,
            run_direct_experience_validation,
            run_paper_evolve_editable,
        ):
            self.assertTrue(callable(value))

    def test_removed_packages_and_compatibility_modules_are_absent(self):
        for relative in (
            "paper_to_" + "image",
            "coevo" + "lution",
            "profes" + "sional",
            "analysis/re" + "build",
            "planning/re" + "build",
            "composition/re" + "build",
            "evolution/re" + "build",
            "editable_re" + "build.py",
            "profes" + "sional_rebuild.py",
            "re" + "build_evolution.py",
        ):
            target = RFS / relative
            self.assertFalse(target.is_file() or (target.is_dir() and any(target.rglob("*.py"))), relative)

    def test_production_imports_do_not_reference_removed_packages(self):
        violations = []
        for path in RFS.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for name in names:
                    parts = set(name.split("."))
                    if parts & FORBIDDEN_PACKAGES:
                        violations.append(f"{path.relative_to(RFS)}:{node.lineno}: {name}")
        self.assertEqual([], violations)

    def test_cli_is_limited_to_direct_and_auxiliary_commands(self):
        from rfs.cli import build_parser
        from rfs.cli_commands.categories import COMMAND_CATEGORIES

        registered = set(build_parser()._subparsers._group_actions[0].choices)
        self.assertEqual(registered, set(COMMAND_CATEGORIES))
        self.assertTrue({"paper-evolve-editable", "paper-evolve-validate", "judge-data", "judge"} <= registered)
        self.assertFalse(registered & FORBIDDEN_COMMANDS)
        self.assertFalse(any(command.startswith("rebuild-") for command in registered))

    def test_documented_packages_exist(self):
        text = (ROOT / "docs" / "code-structure.md").read_text(encoding="utf-8")
        for package in ("contracts", "analysis", "planning", "generation", "composition", "evaluation", "providers", "evolution/direct", "judge_training", "workflows"):
            self.assertIn(f"`rfs/{package}`", text)
            self.assertTrue((RFS / package).exists())


if __name__ == "__main__":
    unittest.main()
