"""The one-off fold of a library's restated lessons, and the archive it leaves behind."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fold_duplicate_experiences", ROOT / "scripts" / "fold_duplicate_experiences.py")
fold_duplicate_experiences = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fold_duplicate_experiences)


def _row(experience_id: str, *, status: str, guidance: str, action: str = "reroute", support: int = 1) -> dict:
    return {
        "experience_id": experience_id,
        "status": status,
        "support": support,
        "action": {"type": action, "scope": "connector"},
        "rule": {"text": guidance},
        "condition": {"topology": "linear"},
    }


def _library(root: Path, rows: list[dict]) -> Path:
    directory = root / "library"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "judge_experiences.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (directory / "manifest.json").write_text(json.dumps({"schema_version": 1, "records": len(rows)}), encoding="utf-8")
    return directory


SENTENCE = "route connectors on shared vertical channels"


class FoldTests(unittest.TestCase):
    def test_rows_saying_the_same_thing_on_the_same_rung_become_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [_row(f"exp-{index}", status="measurement_noise", guidance=SENTENCE) for index in range(4)]
            rows.append(_row("exp-other", status="measurement_noise", guidance="use a wider gutter"))
            library = _library(root, rows)
            report = fold_duplicate_experiences.fold(library, apply=False)
            self.assertEqual(5, report["rows_before"])
            self.assertEqual(2, report["rows_after"])
            self.assertEqual(3, report["folded"])

    def test_a_validated_row_is_never_absorbed_by_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = _library(
                root,
                [
                    _row("exp-candidate", status="candidate", guidance=SENTENCE),
                    _row("exp-promoted", status="promoted", guidance=SENTENCE),
                ],
            )
            report = fold_duplicate_experiences.fold(library, apply=False)
            self.assertEqual(2, report["rows_after"])
            self.assertEqual(0, report["folded"])

    def test_applying_the_fold_leaves_every_folded_row_in_an_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [_row(f"exp-{index}", status="measurement_noise", guidance=SENTENCE, support=index + 1) for index in range(3)]
            library = _library(root, rows)
            report = fold_duplicate_experiences.fold(library, apply=True)
            kept = [json.loads(line) for line in (library / "judge_experiences.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, len(kept))
            # The survivor keeps the whole count, and names the rows that folded into it.
            self.assertEqual(1 + 2 + 3, kept[0]["support"])
            self.assertEqual(2, len(kept[0]["folded_experience_ids"]))
            archive = Path(report["archive"])
            self.assertTrue(archive.is_file())
            archived = [json.loads(line) for line in (library / "judge_experiences.folded.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(2, len(archived))
            self.assertEqual({"exp-0", "exp-1"}, {item["row"]["experience_id"] for item in archived})
            manifest = json.loads((library / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(1, manifest["records"])

    def test_a_row_that_passed_while_carrying_a_new_defect_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marked = {**_row("exp-marked", status="measurement_noise", guidance=SENTENCE), "carried_new_defect": True}
            library = _library(root, [marked, _row("exp-plain", status="measurement_noise", guidance=SENTENCE)])
            report = fold_duplicate_experiences.fold(library, apply=False)
            self.assertEqual(2, report["rows_after"])
            self.assertEqual(0, report["folded"])


if __name__ == "__main__":
    unittest.main()