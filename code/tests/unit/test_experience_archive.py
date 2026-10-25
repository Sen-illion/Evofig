"""Archiving a library keeps the rows another paper can act on and moves only the shells.

One whole-directory move carried away fourteen rows the running loop had just measured and
improved a figure with, because a directory cannot tell a lesson from a shell. This module holds
the row-level decision shut: the row's own text decides, guidance always travels, a status never
does by itself, and a dry run writes nothing at all.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import ExperienceStore
from rfs.evolution.direct.store import untransferable_reason


def experience_row(**overrides) -> dict:
    base = {
        "experience_id": "direct-exp-library",
        "domain": "direct_editable",
        "schema_version": "3.7",
        "status": "candidate",
        "scope": "global",
        "apply_at": "ppt_compilation",
        "memory_type": "layout",
        "hypothesis": "The acceptance checkpoint preferred the revised figure.",
        "action": {"type": "relayout_slots", "slot_ids": ["slot_1"], "factor": 1.08, "dx": 0.0, "dy": 0.0},
        "condition": {"topology": "pipeline", "defect_patterns": ["sparse_panel"]},
        "required_motifs": [],
        "excluded_conditions": [],
    }
    base.update(overrides)
    return base


def write_rows(library: Path, rows: list[dict]) -> None:
    library.mkdir(parents=True, exist_ok=True)
    with (library / "judge_experiences.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TheRowItselfDecidesTests(unittest.TestCase):
    def test_a_row_with_a_rule_travels_and_a_shell_does_not(self):
        self.assertEqual(
            "",
            untransferable_reason(
                experience_row(
                    guidance="Leave at least 2% of the panel width between neighbouring boxes.",
                    action={"type": "layout_constraint", "params": {"gutter_min": 0.02}, "layout_constraints": {"params": {"gutter_min": 0.02}}},
                )
            ),
        )
        self.assertEqual("no_transferable_rule", untransferable_reason(experience_row()))
        self.assertEqual("rule_binds_to_case_objects", untransferable_reason(experience_row(hypothesis="Route rel_12 through one channel.", action={"type": "reroute", "edge_ids": ["rel_12"]})))

    def test_guidance_travels_whatever_the_action_says_and_whatever_the_status_is(self):
        # The prose is the one part a model writes and the one part that always travels, so a row
        # that carries it is a lesson even when its action names this figure's own objects.
        for status in ("candidate", "refuted", "unverifiable"):
            self.assertEqual(
                "",
                untransferable_reason(
                    experience_row(
                        status=status,
                        guidance="Keep a one-to-many fan-out on a single vertical channel.",
                        action={"type": "reroute", "edge_ids": ["rel_12"]},
                    )
                ),
            )

    def test_a_mechanism_row_without_its_parameters_is_a_shell(self):
        # The gate admits the rule on its action name alone, and the name is not the rule.
        self.assertEqual("no_transferable_rule", untransferable_reason(experience_row(action={"type": "router_constraint"})))
        self.assertEqual("", untransferable_reason(experience_row(action={"type": "router_constraint", "constraints": {"forbid_crossing": True}})))


class ArchivingKeepsWhatTravelsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.library = Path(self._tmp.name, "library")
        self.archive = Path(self._tmp.name, "archive")
        write_rows(
            self.library,
            [
                experience_row(
                    experience_id="direct-exp-good",
                    guidance="Leave at least 2% of the panel width between neighbouring boxes.",
                    action={"type": "layout_constraint", "params": {"gutter_min": 0.02}, "layout_constraints": {"params": {"gutter_min": 0.02}}},
                ),
                experience_row(experience_id="direct-exp-shell-candidate"),
                experience_row(experience_id="direct-exp-shell-refuted", status="refuted"),
                experience_row(
                    experience_id="direct-exp-old-binding",
                    hypothesis="Route rel_12 through one channel.",
                    action={"type": "reroute", "edge_ids": ["rel_12"]},
                ),
            ],
        )

    def test_the_move_leaves_the_lesson_and_the_manifest_agrees(self):
        report = ExperienceStore(self.library).archive_untransferable(self.archive)

        self.assertEqual(3, report["archived"])
        self.assertEqual(1, report["kept"])
        self.assertEqual({"no_transferable_rule": 2, "rule_binds_to_case_objects": 1}, report["counts_by_reason"])
        self.assertEqual(1, report["records_after"])
        self.assertEqual(["direct-exp-good"], [row["experience_id"] for row in read_rows(self.library / "judge_experiences.jsonl")])
        self.assertEqual(
            1,
            json.loads((self.library / "manifest.json").read_text(encoding="utf-8"))["records"],
        )
        self.assertEqual(
            ["direct-exp-shell-candidate", "direct-exp-shell-refuted", "direct-exp-old-binding"],
            [row["experience_id"] for row in read_rows(self.archive / "judge_experiences.jsonl")],
        )
        # The report holds the decision per row, so the split can be read back later.
        written = json.loads((self.archive / "archive_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["rows"], written["rows"])
        self.assertEqual("kept", written["rows"][0]["decision"])

    def test_a_dry_run_decides_every_row_and_writes_nothing(self):
        report = ExperienceStore(self.library).archive_untransferable(self.archive, dry_run=True)

        self.assertTrue(report["dry_run"])
        self.assertEqual(3, report["archived"])
        self.assertEqual(4, report["declared_records"])
        self.assertFalse(self.archive.exists())
        self.assertEqual(4, len(read_rows(self.library / "judge_experiences.jsonl")))

    def test_an_archive_that_already_holds_rows_keeps_them(self):
        write_rows(self.archive, [experience_row(experience_id="direct-exp-earlier-move")])

        ExperienceStore(self.library).archive_untransferable(self.archive)

        self.assertEqual(
            ["direct-exp-earlier-move", "direct-exp-shell-candidate", "direct-exp-shell-refuted", "direct-exp-old-binding"],
            [row["experience_id"] for row in read_rows(self.archive / "judge_experiences.jsonl")],
        )


class TheArchiveCommandTests(unittest.TestCase):
    def test_the_command_is_registered_in_the_product_category(self):
        from rfs.cli import build_parser
        from rfs.cli_commands.categories import COMMAND_CATEGORIES

        registered = set(build_parser()._subparsers._group_actions[0].choices)
        self.assertIn("experience-archive", registered)
        self.assertEqual("product", COMMAND_CATEGORIES["experience-archive"])

    def test_the_command_reports_the_split_and_moves_the_shells(self):
        from rfs.workflows import archive_untransferable_experiences

        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp, "library")
            archive = Path(tmp, "archive")
            write_rows(library, [experience_row(experience_id="direct-exp-good", guidance="Leave at least 2% between neighbouring boxes."), experience_row(experience_id="direct-exp-shell")])

            report = archive_untransferable_experiences(library, out=archive)

            self.assertTrue(report["ok"])
            self.assertEqual(1, report["archived"])
            self.assertIn("direct-exp-shell", (archive / "archive_report.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()