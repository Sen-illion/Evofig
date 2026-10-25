"""Counting a Judge dataset before anything is trained on it.

Two things a directory listing hides are checked here: a label no served model produced,
and a quota the plan requires that the directory never reaches. The tests also pin the
reading rule that makes the first one answerable at all -- an acceptance example carries no
observation block by design, so a round's prompt version is read from the row that has one
and stands for every row of that round.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.judge_training.dataset_audit import (
    AUDIT_JSON,
    AUDIT_MARKDOWN,
    audit_dataset,
    audit_examples,
    round_prompt_versions,
)


def example(
    example_id: str,
    *,
    role: str,
    task_type: str = "repair",
    ground_truth: str = "direct-run-a",
    round_index: int = 0,
    prompt_version: str = "",
    score: object = None,
    domain: str = "",
    split: str = "",
    preview: str = "",
    human_audited: bool = False,
    gate_only: bool = False,
) -> dict:
    """Build one example row shaped the way the dataset builder writes them."""

    labels: dict = {"status": "pass"}
    if score is not None:
        labels["overall_score"] = score
    if domain:
        labels["domain"] = domain
    observed: dict = {}
    if prompt_version or gate_only:
        observed["judge"] = {"judge_role": role, "prompt_version": prompt_version or None, "gate_only": gate_only}
    return {
        "example_id": example_id,
        "ground_truth_id": ground_truth,
        "task_type": task_type,
        "split": split,
        "labels": labels,
        "observed": observed,
        "preview": preview,
        "provenance": {"round": round_index, "source": "direct_evolution_process"},
        "judge_metadata": {"judge_role": role, "trained": False, "human_audited": human_audited},
    }


def write_dataset(directory: Path, examples: list[dict], pairs: list[dict] | None = None) -> Path:
    """Write one dataset directory in the shape the audit reads."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "judge_examples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8"
    )
    if pairs is not None:
        (directory / "preference_pairs.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs), encoding="utf-8"
        )
    return directory


def pair(example_id: str, *, family: str = "missed_defect", split: str = "", human_audited: bool = False) -> dict:
    return {
        "example_id": example_id,
        "task_type": "repair",
        "split": split,
        "rejected": {"family": family},
        "preference_basis": {"human_audited": human_audited},
    }


class JudgeDatasetAuditTests(unittest.TestCase):
    def test_empty_directory_reports_no_rows_and_keeps_the_quota_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = audit_dataset(tmp)
        self.assertEqual(report["examples"]["total"], 0)
        self.assertEqual(report["pairs"]["total"], 0)
        self.assertTrue(report["ok"])
        self.assertFalse(report["clean"])
        requirements = {gap["requirement"] for gap in report["gaps"]}
        self.assertIn("source_papers", requirements)
        self.assertIn("preference_pairs", requirements)

    def test_examples_without_pairs_are_reported_as_a_missing_pair_supply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_dataset(Path(tmp), [example("a", role="online_bootstrap", prompt_version="judge-online-v006")])
            report = audit_dataset(root)
        self.assertEqual(report["pairs"]["total"], 0)
        self.assertEqual([gap["requirement"] for gap in report["gaps"]], ["source_papers", "preference_pairs"])

    def test_a_round_without_a_prompt_version_is_counted_as_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                example("a", role="online_bootstrap", ground_truth="gt-a"),
                example("b", role="online_bootstrap", ground_truth="gt-b", prompt_version="judge-online-v006"),
            ]
            root = write_dataset(Path(tmp), rows)
            report = audit_dataset(root)
        self.assertEqual(report["examples"]["deterministic_only"], 1)
        self.assertEqual(report["examples"]["deterministic_only_examples"], ["a"])
        self.assertTrue(any("no served prompt produced" in item for item in report["warnings"]))

    def test_a_heuristic_role_is_counted_as_deterministic_even_with_a_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_dataset(Path(tmp), [example("a", role="online_heuristic", prompt_version="judge-online-v006")])
            report = audit_dataset(root)
        self.assertEqual(report["examples"]["deterministic_only"], 1)

    def test_an_acceptance_row_inherits_the_prompt_version_of_its_round(self):
        rows = [
            example("adv", role="online_bootstrap", prompt_version="judge-online-v006"),
            example("acc", role="frozen_bootstrap", score=0.4),
            example("other-round", role="frozen_bootstrap", round_index=1, score=0.6),
        ]
        summary = audit_examples(rows)
        self.assertEqual(round_prompt_versions(rows), {("direct-run-a", 0): "judge-online-v006"})
        self.assertEqual(summary["deterministic_only"], 1)
        self.assertEqual(summary["deterministic_only_examples"], ["other-round"])

    def test_a_row_that_recorded_gate_only_is_flagged_though_its_round_has_a_version(self):
        """The off and on arms of one paper share round keys, so a version cannot excuse a gate."""

        rows = [
            example("on-adv", role="online_bootstrap", prompt_version="judge-online-v006"),
            example("off-adv", role="online_bootstrap", gate_only=True),
            example("off-acc", role="frozen_bootstrap", score=0.4),
        ]
        summary = audit_examples(rows)
        self.assertEqual(summary["deterministic_only"], 1)
        self.assertEqual(summary["deterministic_only_examples"], ["off-adv"])

    def test_a_round_a_model_answered_flags_nothing(self):
        rows = [
            example("adv", role="online_bootstrap", prompt_version="judge-online-v006"),
            example("acc", role="frozen_bootstrap", score=0.4),
        ]
        self.assertEqual(audit_examples(rows)["deterministic_only"], 0)

    def test_acceptance_score_coverage_is_reported_separately(self):
        rows = [
            example("with", role="frozen_bootstrap", score=0.6, prompt_version="judge-online-v006"),
            example("without", role="frozen_bootstrap", prompt_version="judge-online-v006"),
            example("advisory", role="online_bootstrap", prompt_version="judge-online-v006"),
        ]
        summary = audit_examples(rows)
        self.assertEqual(summary["advisory"], 1)
        self.assertEqual(summary["acceptance"], 2)
        self.assertEqual(summary["acceptance_with_score"], 1)
        self.assertEqual(summary["acceptance_without_score"], 1)

    def test_role_family_and_domain_distributions_are_counted(self):
        rows = [
            example("a", role="online_bootstrap", domain="rag", ground_truth="gt-a"),
            example("b", role="frozen_bootstrap", domain="rag", ground_truth="gt-b"),
            example("c", role="online_bootstrap", domain="cv", ground_truth="gt-c"),
        ]
        summary = audit_examples(rows)
        self.assertEqual(summary["by_role_family"], {"acceptance": 1, "advisory": 2})
        self.assertEqual(summary["by_domain"], {"cv": 1, "rag": 2})
        self.assertEqual(summary["ground_truths"], 3)
        self.assertEqual(summary["domains_seen"], 2)

    def test_a_domain_short_of_its_quota_is_reported_per_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [example("a", role="online_bootstrap", domain="rag", prompt_version="judge-online-v006")]
            root = write_dataset(Path(tmp), rows)
            report = audit_dataset(root, quota={"source_papers": 1, "papers_per_domain": 2})
        self.assertEqual([gap["requirement"] for gap in report["gaps"]], ["papers_per_domain:rag", "preference_pairs"])
        self.assertEqual(report["gaps"][0]["have"], 1.0)
        self.assertEqual(report["gaps"][0]["shortfall"], 1.0)

    def test_a_met_quota_leaves_no_source_paper_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [example("a", role="online_bootstrap", prompt_version="judge-online-v006")]
            root = write_dataset(Path(tmp), rows, pairs=[pair("p1")])
            report = audit_dataset(root, quota={"source_papers": 1, "papers_per_domain": 1})
        self.assertEqual(report["gaps"], [])
        self.assertTrue(report["clean"])

    def test_the_report_is_written_beside_the_dataset_and_can_be_redirected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_dataset(Path(tmp) / "ds", [example("a", role="online_bootstrap", prompt_version="judge-online-v006")])
            report = audit_dataset(root)
            self.assertTrue((root / AUDIT_JSON).is_file())
            self.assertTrue((root / AUDIT_MARKDOWN).is_file())
            redirect = Path(tmp) / "reports"
            moved = audit_dataset(root, out=redirect)
            self.assertEqual(moved["report"], str(redirect / AUDIT_JSON))
            self.assertTrue((redirect / AUDIT_MARKDOWN).is_file())
            self.assertEqual(report["dataset_dir"], str(root))

    def test_markdown_lists_distributions_gaps_and_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_dataset(Path(tmp), [example("a", role="online_heuristic")], pairs=[pair("p1", family="harmful_repair")])
            audit_dataset(root)
            text = (root / AUDIT_MARKDOWN).read_text(encoding="utf-8")
        self.assertIn("## Examples", text)
        self.assertIn("## Preference pairs", text)
        self.assertIn("## Unmet requirements", text)
        self.assertIn("## Warnings", text)
        self.assertIn("harmful_repair=1", text)
        self.assertIn("source_papers", text)

    def test_a_review_queue_next_to_the_pairs_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_dataset(Path(tmp), [example("a", role="online_bootstrap", prompt_version="judge-online-v006")], pairs=[pair("p1")])
            review = root / "review"
            review.mkdir()
            (review / "review_queue.jsonl").write_text(
                json.dumps({"pair_id": "p1", "status": "pending", "human_audited": False}) + "\n"
                + json.dumps({"pair_id": "p2", "status": "reviewed", "human_audited": True}) + "\n",
                encoding="utf-8",
            )
            report = audit_dataset(root)
        self.assertEqual(report["review"], {"present": True, "total": 2, "pending": 1, "audited": 1})


if __name__ == "__main__":
    unittest.main()
