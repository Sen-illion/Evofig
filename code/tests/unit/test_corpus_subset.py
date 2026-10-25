import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rfs.workflows.corpus_subset import (
    apply_labels,
    build_subset,
    delivered_paper_directories,
    doi_token,
    index_source_pdfs,
    index_snapshot_pdfs,
    load_subset,
    load_snapshot_index,
    quotas_for,
    resolve_paper_inputs,
    select_undelivered,
    write_subset_version,
)

DOMAINS = ("agent", "biology", "cv", "materials", "rag")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_snapshot(root: Path, per_domain: int) -> None:
    """A miniature frozen snapshot: text and figures only, as the real one is."""
    papers = []
    assets = []
    for domain in DOMAINS:
        for index in range(per_domain):
            paper_id = f"{domain}-paper-{index:02d}"
            arxiv_id = f"2401.{index:05d}"
            asset_id = f"asset-{domain}-{index:02d}"
            papers.append(
                {
                    "paper_id": paper_id,
                    "primary_domain": domain,
                    "figures": [{"asset_ids": [asset_id]}],
                    "provenance": {
                        "source_record": {
                            "arxiv_id": arxiv_id,
                            "domain": domain,
                            "figure": {"caption": "Overview of the framework."},
                        }
                    },
                }
            )
            assets.append(
                {
                    "asset_id": asset_id,
                    "paper_id": paper_id,
                    "role": "figure",
                    "relative_path": f"assets/{paper_id}/figures/{asset_id}.pdf",
                    "suffix": ".pdf",
                    "sha256": "0" * 64,
                }
            )
    # A paper the split ledger keeps for the hidden test, plus one with a doi only.
    papers.append(
        {
            "paper_id": "sealed-paper",
            "primary_domain": "agent",
            "figures": [{"asset_ids": ["asset-sealed"]}],
            "provenance": {"source_record": {"arxiv_id": "2401.99999", "domain": "agent"}},
        }
    )
    papers.append(
        {
            "paper_id": "doi-paper",
            "primary_domain": "materials",
            "figures": [{"asset_ids": ["asset-doi"]}],
            "provenance": {"source_record": {"arxiv_id": None, "doi": "10.1038/s41586-025-10014-0", "domain": "materials"}},
        }
    )
    for paper_id, asset_id in (("sealed-paper", "asset-sealed"), ("doi-paper", "asset-doi")):
        assets.append(
            {
                "asset_id": asset_id,
                "paper_id": paper_id,
                "role": "figure",
                "relative_path": f"assets/{paper_id}/figures/{asset_id}.pdf",
                "suffix": ".pdf",
                "sha256": "0" * 64,
            }
        )
    figure_root = root / "assets"
    for paper in papers:
        directory = figure_root / str(paper["paper_id"]) / "figures"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{paper['figures'][0]['asset_ids'][0]}.pdf").write_bytes(b"%PDF-1.4")
        # The snapshot also archives its own copy of the paper file itself.
        digest = hashlib.sha256(str(paper["paper_id"]).encode("utf-8")).hexdigest()
        relative = f"sha256/{digest[:2]}/{digest}"
        archived = root / "archive" / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(b"%PDF-1.4")
        assets.append(
            {
                "asset_id": f"asset-pdf-{paper['paper_id']}",
                "paper_id": str(paper["paper_id"]),
                "role": "paper_pdf",
                "storage": "archive",
                "relative_path": relative,
                "suffix": ".pdf",
                "sha256": digest,
            }
        )
    write_jsonl(root / "eligible_papers.jsonl", papers)
    write_jsonl(root / "asset_manifest.jsonl", assets)

    split_rows = [
        {"paper_id": str(paper["paper_id"]), "split": "train", "primary_domain": str(paper["primary_domain"])}
        for paper in papers
    ]
    for row in split_rows:
        if row["paper_id"] == "sealed-paper":
            row["split"] = "hidden_test"
    for index, row in enumerate(split_rows):
        if row["split"] == "train" and index % 4 == 3:
            row["split"] = "validation"
    return split_rows


class CorpusSubsetTests(unittest.TestCase):
    def test_a_per_domain_total_becomes_a_train_and_validation_quota(self):
        self.assertEqual(quotas_for(["agent"], 20)["agent"], (16, 4))
        self.assertEqual(sum(quotas_for(["agent"], 20)["agent"]), 20)

    def test_the_subset_is_five_domains_of_twenty(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            split = root / "split"
            snapshot.mkdir()
            split.mkdir()
            write_jsonl(split / "split_ledger.jsonl", make_snapshot(snapshot, 30))
            subset = build_subset(snapshot, split, per_domain=20)
            self.assertEqual(len(subset["papers"]), 100)
            self.assertEqual(subset["by_domain"], {domain: 20 for domain in DOMAINS})

    def test_the_hidden_test_is_never_sampled(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            split = root / "split"
            snapshot.mkdir()
            split.mkdir()
            write_jsonl(split / "split_ledger.jsonl", make_snapshot(snapshot, 30))
            subset = build_subset(snapshot, split, per_domain=20)
            self.assertNotIn("sealed-paper", {entry["paper_id"] for entry in subset["papers"]})

    def test_the_same_seed_picks_the_same_papers(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            split = root / "split"
            snapshot.mkdir()
            split.mkdir()
            write_jsonl(split / "split_ledger.jsonl", make_snapshot(snapshot, 30))
            first = build_subset(snapshot, split, per_domain=20)
            second = build_subset(snapshot, split, per_domain=20)
            self.assertEqual(
                [entry["paper_id"] for entry in first["papers"]],
                [entry["paper_id"] for entry in second["papers"]],
            )

    def test_a_subset_version_is_written_once_and_read_back(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            split = root / "split"
            snapshot.mkdir()
            split.mkdir()
            write_jsonl(split / "split_ledger.jsonl", make_snapshot(snapshot, 30))
            subset = build_subset(snapshot, split, per_domain=20)
            out = root / "out"
            written = write_subset_version(out, subset)
            self.assertEqual(written["papers"], 100)
            self.assertEqual(load_subset(out)["subset_id"], subset["subset_id"])
            with self.assertRaises(FileExistsError):
                write_subset_version(out, subset)

    def test_labels_report_which_structures_the_sample_covers(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            split = root / "split"
            snapshot.mkdir()
            split.mkdir()
            write_jsonl(split / "split_ledger.jsonl", make_snapshot(snapshot, 30))
            subset = build_subset(snapshot, split, per_domain=20)
            labels = [
                {"paper_id": subset["papers"][0]["paper_id"], "structure": "feedback_loop", "figure_type": "framework"},
                {"paper_id": subset["papers"][1]["paper_id"], "structure": "linear", "figure_type": "pipeline"},
            ]
            apply_labels(subset, labels)
            report = subset["label_report"]
            self.assertEqual(report["labelled"], 2)
            self.assertEqual(report["unlabelled"], 98)
            self.assertEqual(report["by_structure"], {"feedback_loop": 1, "linear": 1})
            self.assertTrue(subset["papers"][0]["labelled"])

    def test_the_paper_file_is_found_under_either_name(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            collection = root / "collection"
            arxiv_dir = collection / "v001" / "assets" / "agent" / "2503.02519"
            arxiv_dir.mkdir(parents=True)
            (arxiv_dir / "arxiv.pdf").write_bytes(b"%PDF-1.4")
            official = collection / "v002" / "assets" / "materials" / "official" / "10.1038_s41586-025-10014-0"
            official.mkdir(parents=True)
            (official / "official.pdf").write_bytes(b"%PDF-1.4")
            index = index_source_pdfs([collection])
            self.assertIn(("agent", "arxiv:2503.02519"), index)
            self.assertIn(("materials", doi_token("10.1038/s41586-025-10014-0")), index)

    def test_a_paper_resolves_to_its_file_and_frozen_figure(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            split = root / "split"
            snapshot.mkdir()
            split.mkdir()
            write_jsonl(split / "split_ledger.jsonl", make_snapshot(snapshot, 30))
            subset = build_subset(snapshot, split, per_domain=20)
            entry = subset["papers"][0]
            collection = root / "collection"
            arxiv_dir = collection / "assets" / entry["domain"] / "2401.00000"
            arxiv_dir.mkdir(parents=True)
            (arxiv_dir / "arxiv.pdf").write_bytes(b"%PDF-1.4")
            resolved = resolve_paper_inputs(
                snapshot,
                entry,
                pdf_index=index_source_pdfs([collection]),
                papers=load_snapshot_index(snapshot),
            )
            self.assertEqual(resolved["missing"], [])
            self.assertTrue(resolved["pdf"].endswith("arxiv.pdf"))
            self.assertTrue(resolved["reference"].endswith(".pdf"))

    def test_a_paper_without_a_file_is_reported_not_hidden(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            split = root / "split"
            snapshot.mkdir()
            split.mkdir()
            write_jsonl(split / "split_ledger.jsonl", make_snapshot(snapshot, 30))
            subset = build_subset(snapshot, split, per_domain=20)
            resolved = resolve_paper_inputs(
                snapshot,
                subset["papers"][0],
                pdf_index={},
                papers=load_snapshot_index(snapshot),
            )
            self.assertEqual(resolved["missing"], ["paper_pdf"])
            self.assertIsNone(resolved["pdf"])

    def test_a_doi_paper_resolves_through_its_official_file(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            split = root / "split"
            snapshot.mkdir()
            split.mkdir()
            write_jsonl(split / "split_ledger.jsonl", make_snapshot(snapshot, 30))
            collection = root / "collection"
            official = collection / "assets" / "materials" / "official" / "10.1038_s41586-025-10014-0"
            official.mkdir(parents=True)
            (official / "official.pdf").write_bytes(b"%PDF-1.4")
            resolved = resolve_paper_inputs(
                snapshot,
                {"paper_id": "doi-paper", "domain": "materials"},
                pdf_index=index_source_pdfs([collection]),
                papers=load_snapshot_index(snapshot),
                figure_index={"asset-doi": {"relative_path": "assets/doi-paper/figures/asset-doi.pdf"}},
            )
            self.assertEqual(resolved["paper_source"], "official")
            self.assertEqual(resolved["missing"], [])


class CorpusSubsetArchiveTests(unittest.TestCase):
    def test_the_snapshot_archive_indexes_every_paper_original(self):
        with TemporaryDirectory() as temp:
            snapshot = Path(temp) / "snapshot"
            snapshot.mkdir()
            make_snapshot(snapshot, 3)
            index = index_snapshot_pdfs(snapshot, snapshot / "archive")
            self.assertEqual(len(index), 17)
            self.assertTrue(all(path.is_file() for path in index.values()))

    def test_a_paper_resolves_to_the_copy_the_snapshot_froze(self):
        # The collection directory the paper was collected from may be gone by
        # the time a subset runs; the snapshot's own archived copy is the file
        # it was verified against, so that is the one to run.
        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            split = root / "split"
            snapshot.mkdir()
            split.mkdir()
            write_jsonl(split / "split_ledger.jsonl", make_snapshot(snapshot, 30))
            subset = build_subset(snapshot, split, per_domain=20)
            resolved = resolve_paper_inputs(
                snapshot,
                subset["papers"][0],
                pdf_index={},
                pdf_by_paper=index_snapshot_pdfs(snapshot, snapshot / "archive"),
                papers=load_snapshot_index(snapshot),
            )
            self.assertEqual(resolved["missing"], [])
            self.assertEqual(resolved["paper_source"], "snapshot_archive")
            self.assertIn("archive", resolved["pdf"].replace("\\", "/"))

    def test_the_archived_copy_wins_over_the_collection_directory(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            split = root / "split"
            snapshot.mkdir()
            split.mkdir()
            write_jsonl(split / "split_ledger.jsonl", make_snapshot(snapshot, 30))
            entry = build_subset(snapshot, split, per_domain=20)["papers"][0]
            collection = root / "collection"
            arxiv_dir = collection / "assets" / entry["domain"] / "2401.00000"
            arxiv_dir.mkdir(parents=True)
            (arxiv_dir / "arxiv.pdf").write_bytes(b"%PDF-1.4")
            resolved = resolve_paper_inputs(
                snapshot,
                entry,
                pdf_index=index_source_pdfs([collection]),
                pdf_by_paper=index_snapshot_pdfs(snapshot, snapshot / "archive"),
                papers=load_snapshot_index(snapshot),
            )
            self.assertEqual(resolved["paper_source"], "snapshot_archive")
            self.assertIn("archive", resolved["pdf"].replace("\\", "/"))

    def test_an_archive_object_that_went_missing_is_not_indexed(self):
        with TemporaryDirectory() as temp:
            snapshot = Path(temp) / "snapshot"
            snapshot.mkdir()
            make_snapshot(snapshot, 3)
            index = index_snapshot_pdfs(snapshot, snapshot / "archive")
            victim = sorted(index)[0]
            index[victim].unlink()
            self.assertNotIn(victim, index_snapshot_pdfs(snapshot, snapshot / "archive"))

    def test_the_batch_finds_every_paper_in_the_snapshot_archive(self):
        from rfs.workflows.supervised_batch import run_corpus_batch

        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            split = root / "split"
            snapshot.mkdir()
            split.mkdir()
            write_jsonl(split / "split_ledger.jsonl", make_snapshot(snapshot, 30))
            result = run_corpus_batch(
                snapshot=snapshot,
                split_dir=split,
                out_root=root / "out",
                pdf_roots=[],
                archive_root=snapshot / "archive",
                per_domain=2,
                limit=0,
            )
            self.assertEqual(result["pdf_index_size"], 0)
            self.assertEqual(result["archive_pdf_index_size"], 152)
            self.assertEqual(result["unresolved_paper_ids"], [])


class PapersAlreadyDrawnTests(unittest.TestCase):
    """A paper that already owns a run directory must never be bought twice.

    The batch writes each paper into ``safe_paper_directory(paper_id)``, so an
    id and the directory it owns only agree after that mapping is applied.  A
    check that compared the raw id against the directory names answered "not
    drawn" for every id carrying a slash or a colon - nearly the whole corpus -
    and the same papers were paid for a second time.
    """

    def save_runs(self, root: Path, names) -> None:
        for name in names:
            attempt = root / "runs" / str(name) / "attempt_01"
            attempt.mkdir(parents=True, exist_ok=True)
            (attempt / "editable_composition.pptx").write_bytes(b"PK")

    def test_an_id_matches_the_directory_it_owns(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.save_runs(root, ["10.1021_acsnano.5c17401", "iclr_2026_10006567", "plain-paper"])
            papers = [
                {"paper_id": "10.1021/acsnano.5c17401"},
                {"paper_id": "iclr:2026:10006567"},
                {"paper_id": "plain-paper"},
                {"paper_id": "never-drawn"},
            ]
            kept, skipped = select_undelivered(papers, delivered_paper_directories([root]))
            self.assertEqual([paper["paper_id"] for paper in kept], ["never-drawn"])
            self.assertEqual(sorted(skipped), ["10.1021/acsnano.5c17401", "iclr:2026:10006567", "plain-paper"])

    def test_two_ids_that_would_share_one_directory_are_refused(self):
        with self.assertRaises(ValueError):
            select_undelivered([{"paper_id": "iclr:2026:1"}, {"paper_id": "iclr_2026_1"}], [])

    def test_a_root_without_run_directories_skips_nothing(self):
        with TemporaryDirectory() as temp:
            self.assertEqual(delivered_paper_directories([Path(temp)]), set())

    def test_the_batch_drops_what_another_root_already_delivered(self):
        from rfs.workflows.supervised_batch import run_corpus_batch

        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            split = root / "split"
            snapshot.mkdir()
            split.mkdir()
            write_jsonl(split / "split_ledger.jsonl", make_snapshot(snapshot, 4))
            run_corpus_batch(
                snapshot=snapshot,
                split_dir=split,
                out_root=root / "out",
                pdf_roots=[],
                per_domain=2,
                sample_only=True,
            )
            subset_dir = root / "out" / "subset"
            chosen = [
                json.loads(line)["paper_id"]
                for line in (subset_dir / "subset_ledger.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreater(len(chosen), 3)
            drawn = chosen[:3]
            older = root / "older"
            self.save_runs(older, drawn)

            result = run_corpus_batch(
                snapshot=snapshot,
                split_dir=split,
                out_root=root / "out",
                pdf_roots=[],
                subset=subset_dir,
                exclude_roots=[older],
                limit=0,
            )
            self.assertEqual(result["papers"], len(chosen))
            self.assertEqual(result["already_drawn"], 3)
            self.assertEqual(result["queued"], len(chosen) - 3)
            recorded = {
                json.loads(line)["paper_id"]
                for line in (root / "out" / "already_drawn.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            self.assertEqual(recorded, set(drawn))


if __name__ == "__main__":
    unittest.main()
