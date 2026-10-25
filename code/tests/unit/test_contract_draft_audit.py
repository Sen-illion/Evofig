"""Audit the unsigned contract drafts without inventing content.

The audit is what tells a reviewer where to look. These tests pin the two rules
that matter: a damaged evidence id may only be rewritten when its stripped form
resolves, and a relation that points at a nested item is not a defect.
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_contract_drafts",
        ROOT / "benchmarks" / "editable-figure-bench" / "review" / "audit_contract_drafts.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_module = load_module()


def _write_case(root: pathlib.Path, name: str, plan: dict, pool: list[dict], paper: str) -> pathlib.Path:
    out_dir = root / "papers" / name
    out_dir.mkdir(parents=True)
    (out_dir / "figure_content_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (out_dir / "key_evidence.json").write_text(json.dumps({"evidence": pool}), encoding="utf-8")
    (out_dir / "paper.md").write_text(paper, encoding="utf-8")
    return out_dir


def _status(root: pathlib.Path, out_dirs) -> None:
    rows = [{"paper_id": path.name, "out_dir": str(path), "ok": True} for path in out_dirs]
    (root / "status.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class DamagedEvidenceTests(unittest.TestCase):
    def test_a_newline_split_id_is_reported_as_damaged_and_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            plan = {
                "modules": [{"id": "m1", "label": "Encoder", "evidence_ids": ["E_P00\n4_abc"]}],
                "relations": [],
            }
            out_dir = _write_case(root, "p1", plan, [{"id": "E_P004_abc", "text": "Encoder"}], "an Encoder layer")
            _status(root, [out_dir])

            before = audit_module.audit(audit_module.load_contracts(root))
            self.assertEqual(1, before["totals"]["damaged_evidence"])
            self.assertEqual(0, before["totals"]["unresolved_evidence"])

            report = audit_module.repair_evidence_ids(
                audit_module.load_contracts(root), log_path=root / "repair.jsonl"
            )
            self.assertEqual(1, len(report["repaired"]))
            written = json.loads((out_dir / "figure_content_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(["E_P004_abc"], written["modules"][0]["evidence_ids"])
            self.assertIn("repair.jsonl", (root / "repair.jsonl").name)
            after = audit_module.audit(audit_module.load_contracts(root))
            self.assertEqual(0, after["totals"]["damaged_evidence"])

    def test_an_id_that_resolves_after_no_change_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            plan = {"modules": [{"id": "m1", "label": "Encoder", "evidence_ids": ["E_missing"]}], "relations": []}
            out_dir = _write_case(root, "p1", plan, [{"id": "E_P004_abc", "text": "Encoder"}], "an Encoder layer")
            _status(root, [out_dir])
            report = audit_module.repair_evidence_ids(
                audit_module.load_contracts(root), log_path=root / "repair.jsonl"
            )
            self.assertEqual([], report["repaired"])
            self.assertEqual(1, report["totals"]["unresolved_evidence"])
            written = json.loads((out_dir / "figure_content_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(["E_missing"], written["modules"][0]["evidence_ids"])


class SuspectTests(unittest.TestCase):
    def test_a_relation_to_a_nested_item_is_not_a_dangling_relation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            plan = {
                "modules": [{"id": "m1", "label": "Encoder", "evidence_ids": []}],
                "optional_details": [{"id": "d1", "label": "Detail"}],
                "relations": [{"id": "r1", "source": "m1", "target": "d1"}],
            }
            out_dir = _write_case(root, "p1", plan, [], "an Encoder layer")
            _status(root, [out_dir])
            report = audit_module.audit(audit_module.load_contracts(root))
            self.assertEqual(0, report["totals"]["dangling_relations"])

    def test_a_relation_to_nowhere_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            plan = {
                "modules": [{"id": "m1", "label": "Encoder", "evidence_ids": []}],
                "relations": [{"id": "r1", "source": "m1", "target": "ghost"}],
            }
            out_dir = _write_case(root, "p1", plan, [], "an Encoder layer")
            _status(root, [out_dir])
            report = audit_module.audit(audit_module.load_contracts(root))
            self.assertEqual(1, report["totals"]["dangling_relations"])
            self.assertEqual("ghost", report["findings"][0]["dangling_relations"][0]["value"])

    def test_a_symbol_label_is_not_a_hallucination_suspect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            plan = {"modules": [{"id": "m1", "label": "f(X)", "evidence_ids": []}], "relations": []}
            out_dir = _write_case(root, "p1", plan, [], "the model computes things")
            _status(root, [out_dir])
            report = audit_module.audit(audit_module.load_contracts(root))
            self.assertEqual(0, report["totals"]["suspect_labels"])

    def test_a_label_with_no_foothold_in_the_paper_is_a_suspect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            plan = {"modules": [{"id": "m1", "label": "Quantum Annealing Module", "evidence_ids": []}], "relations": []}
            out_dir = _write_case(root, "p1", plan, [], "a paper about image retrieval benchmarks")
            _status(root, [out_dir])
            report = audit_module.audit(audit_module.load_contracts(root))
            self.assertEqual(1, report["totals"]["suspect_labels"])


if __name__ == "__main__":
    unittest.main()
