"""Unit tests for writing the middleware's contracts where the generation line reads them."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rfs.judge_training.contract_export import (
    EXPORT_MANIFEST_FILE,
    EXPORT_REPORT_FILE,
    ContractExportMismatch,
    export_paper_contract,
    export_version_contracts,
    recorded_hashes,
    resolve_snapshot,
    snapshot_paper_index,
    version_reviews,
)
from rfs.judge_training.frozen_document import build_frozen_document
from rfs.judge_training.pilot_graphs import canonical_sha256, derive_figure_specification, safe_paper_directory


SNAPSHOT_NAME = "paper-ut-v001"
SNAPSHOT_ROOT = Path("datasets") / "judge_source_snapshots"


def write_snapshot(root, *, paper_id="p0001", domain="cv"):
    """Write a minimal frozen snapshot carrying one paper."""
    snapshot = root / SNAPSHOT_ROOT / SNAPSHOT_NAME
    paper_dir = snapshot / "assets" / "paper-aaaa"
    paper_dir.mkdir(parents=True)
    (snapshot / "snapshot_manifest.json").write_text(
        json.dumps({"snapshot_id": SNAPSHOT_NAME}), encoding="utf-8"
    )
    records = [
        {"page": 1, "text": "A Frozen Study of Things\nAbstract\nWe propose a frozen pipeline."},
        {"page": 3, "text": "3 Method\nOur encoder feeds the decoder."},
    ]
    (paper_dir / "text_locations.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    (paper_dir / "full_text.txt").write_text(
        "\n\n".join(record["text"] for record in records), encoding="utf-8"
    )
    (paper_dir / "captions.json").write_text(
        json.dumps({"paper_id": paper_id, "figures": []}), encoding="utf-8"
    )
    (paper_dir / "bindings.json").write_text(
        json.dumps({"paper_id": paper_id, "bindings": []}), encoding="utf-8"
    )
    paper = {
        "paper_id": paper_id,
        "title": "A Frozen Study of Things",
        "primary_domain": domain,
        "captions_path": "assets/paper-aaaa/captions.json",
        "bindings_path": "assets/paper-aaaa/bindings.json",
    }
    (snapshot / "eligible_papers.jsonl").write_text(
        json.dumps(paper, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return snapshot, paper


def review_row(paper_id="p0001"):
    """Build one stored review row shaped like the middleware writes it."""
    return {
        "schema_version": "paper-review-v1",
        "paper_identity": {"title": "A Frozen Study of Things"},
        "inputs": [{"id": "input_a", "statement": "Raw records", "evidence_ids": ["E_P001_aaaa"]}],
        "outputs": [],
        "modules": [{"id": "module_a", "statement": "Encoder", "evidence_ids": ["E_P003_bbbb"]}],
        "innovations": [],
        "relations": [{"id": "rel_a", "source_id": "input_a", "target_id": "module_a", "statement": "feeds"}],
        "workflows": {"inference": [{"id": "wf_a", "statement": "Inference"}]},
        "central_claims": [{"statement": "Frozen pipelines matter."}],
        "forbidden_inventions": ["an invented module"],
        "graph_provenance": {"paper_id": paper_id, "review_mode": "vlm"},
    }


def method_content(paper_id="p0001"):
    """Build one method contract shaped like the middleware writes it."""
    return {
        "contract": "rfs.method_figure_content",
        "version": 1,
        "figure_goal": "Explain the frozen pipeline.",
        "modules": [{"id": "mod_a", "label": "Encoder", "role": "module", "must_show": True, "evidence_ids": ["E_P003_bbbb"]}],
        "relations": [{"id": "rel_a", "source": "mod_a", "target": "mod_a", "type": "related_to", "label": "feeds"}],
        "provenance": {"paper_id": paper_id},
    }


def build_case(
    root,
    *,
    paper_ids=("p0001",),
    snapshot_domain="cv",
    report_domain="biology",
    stored_specification=False,
    write_method=True,
    ledger=True,
    record="graph_report",
):
    """Write one snapshot and one middleware version, returning both and the reviews."""
    snapshot, first = write_snapshot(root, paper_id=paper_ids[0], domain=snapshot_domain)
    paper_index = {first["paper_id"]: first}
    for extra in paper_ids[1:]:
        extra_paper = dict(first)
        extra_paper["paper_id"] = extra
        with (snapshot / "eligible_papers.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(extra_paper, ensure_ascii=False) + "\n")
        paper_index[extra] = extra_paper
    version = root / "paper-ut-pilot-graph-v001"
    version.mkdir(parents=True)
    (version / "graph_manifest.json").write_text(
        json.dumps({"inputs": {"snapshot": snapshot.name}}), encoding="utf-8"
    )
    reviews = [review_row(paper_id) for paper_id in paper_ids]
    (version / "paper_reviews.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviews), encoding="utf-8"
    )
    ledger_rows = []
    for paper_id in paper_ids:
        directory = version / "graphs" / safe_paper_directory(paper_id)
        directory.mkdir(parents=True, exist_ok=True)
        expected = derive_figure_specification(
            build_frozen_document(snapshot, paper_index[paper_id]),
            {"paper_id": paper_id, "domain": report_domain},
            review_row(paper_id),
        )
        report: dict = {"paper_id": paper_id, "domain": report_domain}
        if record == "graph_report":
            report["method_contract"] = {"specification_sha256": canonical_sha256(expected)}
        (directory / "graph_report.json").write_text(
            json.dumps(report, ensure_ascii=False), encoding="utf-8"
        )
        if stored_specification:
            (directory / "figure_specification.json").write_text(
                json.dumps(expected, ensure_ascii=False), encoding="utf-8"
            )
            (directory / "paper_review.json").write_text(
                json.dumps(review_row(paper_id), ensure_ascii=False), encoding="utf-8"
            )
        if write_method:
            content = method_content(paper_id)
            (directory / "method_figure_content.json").write_text(
                json.dumps(content, ensure_ascii=False), encoding="utf-8"
            )
            (directory / "method_extraction_report.json").write_text(
                json.dumps({"paper_id": paper_id, "ok": True}, ensure_ascii=False), encoding="utf-8"
            )
            ledger_rows.append(
                {
                    "step": "contract",
                    "paper_id": paper_id,
                    "artifact_sha256": {"method_figure_content.json": canonical_sha256(content)},
                }
            )
    if ledger:
        (version / "graph_ledger.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ledger_rows), encoding="utf-8"
        )
    return {"version": version, "snapshot": snapshot, "papers": paper_index, "reviews": reviews}


class ContractExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # A version names its snapshot rather than carrying it, so the module
        # looks beside the working directory; the tests move that root into the
        # temporary one so no test can read or write the repository's datasets.
        patcher = patch(
            "rfs.judge_training.contract_export.DEFAULT_SNAPSHOT_ROOT", self.root / SNAPSHOT_ROOT
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def case(self, **kwargs):
        return build_case(self.root, **kwargs)

    def test_derived_specification_reproduces_the_recorded_hash(self):
        case = self.case()
        manifest = export_version_contracts(case["version"], self.root / "export")

        self.assertTrue(manifest["ok"])
        self.assertEqual(manifest["exported"], 1)
        self.assertEqual(manifest["failed"], 0)
        self.assertEqual(manifest["inventories_derived"], 1)
        self.assertEqual(manifest["inventories_verified"], 1)
        self.assertEqual(manifest["with_figure_content_plan"], 1)
        record = manifest["records"][0]
        self.assertEqual(record["specification_source"], "derived_from_stored_review")
        self.assertTrue(record["specification_matches_middleware"])
        self.assertTrue(record["method_matches_middleware"])

    def test_the_middleware_ground_truth_travels_with_the_export(self):
        case = self.case(stored_specification=True)
        ground_truth = {"contract": "rfs.scientific_ground_truth", "version": 1, "paper_id": "p0001", "required_node_ids": ["n1"]}
        source = case["version"] / "graphs" / safe_paper_directory("p0001") / "scientific_ground_truth.json"
        source.write_bytes(json.dumps(ground_truth, ensure_ascii=False).encode("utf-8"))

        manifest = export_version_contracts(case["version"], self.root / "export")
        exported = self.root / "export" / safe_paper_directory("p0001")
        record = json.loads((exported / EXPORT_REPORT_FILE).read_text(encoding="utf-8"))

        self.assertTrue(manifest["ok"])
        self.assertEqual(manifest["with_scientific_ground_truth"], 1)
        # The copy is the contract itself, so it is byte-identical to the middleware file.
        self.assertEqual((exported / "scientific_ground_truth.json").read_bytes(), source.read_bytes())
        self.assertTrue(record["with_scientific_ground_truth"])
        self.assertIn("scientific_ground_truth.json", record["files"])
        self.assertEqual(record["scientific_ground_truth"]["source"], str(source.resolve()))

    def test_a_version_without_a_ground_truth_ships_none(self):
        case = self.case(stored_specification=True)
        manifest = export_version_contracts(case["version"], self.root / "export")
        exported = self.root / "export" / safe_paper_directory("p0001")
        record = json.loads((exported / EXPORT_REPORT_FILE).read_text(encoding="utf-8"))

        self.assertTrue(manifest["ok"])
        self.assertEqual(manifest["with_scientific_ground_truth"], 0)
        self.assertFalse(record["with_scientific_ground_truth"])
        self.assertNotIn("scientific_ground_truth.json", record["files"])
        self.assertFalse((exported / "scientific_ground_truth.json").exists())

    def test_derived_specification_prefers_the_graph_report_domain(self):
        # The snapshot says cv and the report says biology; a rebuild that read the
        # snapshot's domain would hash differently and be refused.
        case = self.case(snapshot_domain="cv", report_domain="biology")
        manifest = export_version_contracts(case["version"], self.root / "export")
        written = json.loads(
            (self.root / "export" / safe_paper_directory("p0001") / "paper_method_inventory.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(manifest["inventories_verified"], 1)
        self.assertEqual(written["seed"]["domain"], "biology")

    def test_stored_specification_is_copied_instead_of_derived(self):
        case = self.case(stored_specification=True)
        record = export_paper_contract(case["version"], "p0001", self.root / "export" / "p0001")

        self.assertEqual(record["specification_source"], "version")
        self.assertEqual(record["domain_source"], "version")
        self.assertEqual(record["files"], sorted(record["files"]))
        self.assertIn("paper_review.json", record["files"])
        # The version stored it under the earlier name; the export writes the current one.
        self.assertIn("paper_method_inventory.json", record["files"])

    def test_figure_content_plan_is_copied_verbatim(self):
        case = self.case()
        export_paper_contract(case["version"], "p0001", self.root / "export" / "p0001")
        source = case["version"] / "graphs" / safe_paper_directory("p0001")

        self.assertEqual(
            (self.root / "export" / "p0001" / "figure_content_plan.json").read_bytes(),
            (source / "method_figure_content.json").read_bytes(),
        )

    def test_mismatched_specification_is_refused(self):
        case = self.case()
        directory = case["version"] / "graphs" / safe_paper_directory("p0001")
        report = json.loads((directory / "graph_report.json").read_text(encoding="utf-8"))
        report["method_contract"]["specification_sha256"] = "0" * 64
        (directory / "graph_report.json").write_text(json.dumps(report), encoding="utf-8")

        with self.assertRaises(ContractExportMismatch):
            export_paper_contract(case["version"], "p0001", self.root / "export" / "p0001")

    def test_mismatched_figure_content_plan_is_refused(self):
        case = self.case()
        ledger = case["version"] / "graph_ledger.jsonl"
        row = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
        # A version recorded the plan under the name it wrote, so the earlier key
        # is the one this ledger carries; the reader maps it to the file it names.
        row["artifact_sha256"]["method_figure_content.json"] = "1" * 64
        ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
        manifest = export_version_contracts(case["version"], self.root / "export")

        self.assertFalse(manifest["ok"])
        self.assertEqual(manifest["exported"], 0)
        self.assertEqual(manifest["failed"], 1)
        self.assertIn("figure content plan", manifest["failures"][0]["error"])

    def test_a_paper_without_a_stored_review_is_reported(self):
        case = self.case()
        with self.assertRaises(FileNotFoundError) as raised:
            export_paper_contract(case["version"], "p9999", self.root / "export" / "p9999")
        manifest = export_version_contracts(
            case["version"], self.root / "export", paper_ids=["p0001", "p9999"]
        )

        self.assertIn("no stored review", str(raised.exception))
        self.assertFalse(manifest["ok"])
        self.assertEqual(manifest["exported"], 1)
        self.assertEqual(manifest["failed"], 1)
        self.assertEqual(manifest["failures"][0]["paper_id"], "p9999")

    def test_a_missing_snapshot_is_reported(self):
        case = self.case()
        (case["version"] / "graph_manifest.json").write_text(json.dumps({"inputs": {}}), encoding="utf-8")

        with self.assertRaises(FileNotFoundError) as raised:
            resolve_snapshot(case["version"])
        with self.assertRaises(FileNotFoundError):
            resolve_snapshot(case["version"], self.root / "nowhere")

        self.assertIn("cannot find the frozen snapshot", str(raised.exception))

    def test_manifest_named_snapshot_is_found_beside_the_working_directory(self):
        case = self.case()
        self.assertEqual(resolve_snapshot(case["version"]).name, SNAPSHOT_NAME)

    def test_only_contract_ledger_rows_are_hashes(self):
        case = self.case(ledger=False)
        (case["version"] / "graph_ledger.jsonl").write_text(
            json.dumps({"step": "review", "paper_id": "p0001", "artifact_sha256": {"x": "9" * 64}}) + "\n",
            encoding="utf-8",
        )
        recorded = recorded_hashes(case["version"], "p0001")

        self.assertNotIn("x", recorded)
        self.assertIn("paper_method_inventory.json", recorded)

    def test_reviews_are_indexed_by_provenance_paper_id(self):
        case = self.case(paper_ids=("p0001", "p0002"))
        lines = (case["version"] / "paper_reviews.jsonl").read_text(encoding="utf-8").splitlines()
        first, second = json.loads(lines[0]), json.loads(lines[1])
        second["paper_id"] = "p0002"
        (case["version"] / "paper_reviews.jsonl").write_text(
            "\n".join([json.dumps(first), json.dumps(second), json.dumps({"no": "id"})]) + "\n",
            encoding="utf-8",
        )
        found = version_reviews(case["version"])

        self.assertEqual(sorted(found), ["p0001", "p0002"])

    def test_the_snapshot_papers_are_indexed_by_paper_id(self):
        case = self.case()
        self.assertEqual(sorted(snapshot_paper_index(case["snapshot"])), ["p0001"])

    def test_export_orders_and_limits_the_papers(self):
        case = self.case(paper_ids=("p0001", "p0002", "p0003"))
        manifest = export_version_contracts(case["version"], self.root / "export", limit=2)

        self.assertEqual(manifest["papers"], 2)
        self.assertEqual([record["paper_id"] for record in manifest["records"]], ["p0001", "p0002"])
        self.assertTrue((self.root / "export" / "p0001" / EXPORT_REPORT_FILE).is_file())
        self.assertTrue((self.root / "export" / EXPORT_MANIFEST_FILE).is_file())
        self.assertFalse((self.root / "export" / "p0003").exists())

    def test_a_paper_without_a_plan_still_exports(self):
        case = self.case(write_method=False)
        manifest = export_version_contracts(case["version"], self.root / "export")

        self.assertTrue(manifest["ok"])
        self.assertEqual(manifest["with_figure_content_plan"], 0)
        self.assertIsNone(manifest["records"][0]["method_matches_middleware"])
        self.assertTrue(manifest["records"][0]["specification_matches_middleware"])

    def test_an_empty_version_is_refused(self):
        case = self.case()
        (case["version"] / "paper_reviews.jsonl").unlink()

        with self.assertRaises(ValueError):
            export_version_contracts(case["version"], self.root / "export")
        with self.assertRaises(FileNotFoundError):
            export_version_contracts(self.root / "nowhere", self.root / "export")

    def test_export_reports_where_each_contract_came_from(self):
        case = self.case()
        record = export_paper_contract(case["version"], "p0001", self.root / "export" / "p0001")
        written = json.loads(
            (self.root / "export" / "p0001" / EXPORT_REPORT_FILE).read_text(encoding="utf-8")
        )

        self.assertEqual(written["paper_id"], "p0001")
        self.assertEqual(written["source_version"], record["source_version"])
        self.assertTrue(written["with_figure_content_plan"])
        self.assertEqual(written["files"], sorted(written["files"]))
        self.assertEqual(written["review_sha256"], record["review_sha256"])


if __name__ == "__main__":
    unittest.main()
