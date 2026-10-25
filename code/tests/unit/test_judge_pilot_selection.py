"""Unit tests for the D0 Pilot selection."""

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rfs.judge_training.pilot_selection import (
    CANDIDATE_SPLITS,
    STRUCTURAL_TYPES,
    caption_fallback_labels,
    candidate_pool,
    demote_rejected,
    gate_labels,
    label_candidates,
    load_figure_index,
    load_label_cache,
    select_pilot,
    select_pilot_verified,
    validate_pilot_version,
    verify_selected_assets,
    write_pilot_version,
    build_pilot_payloads,
)
from rfs.judge_training.split_ledger import build_split_payloads, write_split_version


def figure_png(seed, size=(160, 120)):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(6, 8), dtype=np.uint8)
    image = Image.fromarray(small, "L").resize(size, Image.NEAREST).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_snapshot(root, count=120, domains=("cv", "rag", "agent", "materials", "biology")):
    snapshot = root / "paper-d0-v001"
    (snapshot / "assets").mkdir(parents=True, exist_ok=True)
    papers = []
    assets = []
    for index in range(count):
        paper_id = "p%04d" % index
        domain = domains[index % len(domains)]
        relative = "assets/%s/figure.png" % paper_id
        payload = figure_png(index + 1)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        papers.append(
            {
                "paper_id": paper_id,
                "primary_domain": domain,
                "domain_labels": [domain],
                "title": "Paper %04d" % index,
                "figure_asset_tier": "A_source_verified",
                "identity_aliases": ["paper_id:%s" % paper_id],
                "figures": [{"figure_id": "figure-%04d" % index, "asset_ids": ["asset-%04d" % index]}],
                "provenance": {"source_record": {"figure": {"caption": "Overview of the framework."}}},
            }
        )
        assets.append(
            {
                "asset_id": "asset-%04d" % index,
                "paper_id": paper_id,
                "role": "figure",
                "relative_path": relative,
                "sha256": __import__("hashlib").sha256(payload).hexdigest(),
                "suffix": ".png",
            }
        )
    with (snapshot / "eligible_papers.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in papers:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (snapshot / "asset_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in assets:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (snapshot / "snapshot_manifest.json").write_text(
        json.dumps({"snapshot_id": "paper-d0-v001", "strict_unique_papers": count}),
        encoding="utf-8",
    )
    (snapshot / "COMPLETE").write_text(json.dumps({"manifest_sha256": "0" * 64}), encoding="utf-8")
    return snapshot


def make_split(root, snapshot, *, name="split"):
    payloads = build_split_payloads(
        snapshot, seed="seed-a", ratios="65/15/20", perceptual_hash_report={"ok": True}
    )
    out = root / name
    write_split_version(out, payloads, perceptual_hash={"ok": True})
    return out, payloads


def fake_labeller(prompt, image):
    digits = "".join(char for char in Path(image).name if char.isdigit())
    index = int(digits or 0)
    return {
        "figure_type": ("framework", "pipeline", "technical_route")[index % 3],
        "structure": ("linear", "branching", "converging", "feedback_loop", "multi_panel")[index % 5],
        "panel_count": 1 + index % 3,
        "node_count": 4 + index,
        "has_arrows": True,
        "complexity": 1 + index % 5,
        "confidence": 0.9,
    }


class PoolTests(unittest.TestCase):
    def test_a_ledger_without_figure_assets_yields_no_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, count=200)
            _, payloads = make_split(root, snapshot)
            pool = candidate_pool(payloads["rows"], [p for p in payloads["rows"]], seed="s")
            self.assertEqual(pool["candidates"], [])

    def test_pool_is_drawn_from_the_snapshot_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, count=200)
            _, payloads = make_split(root, snapshot)
            papers = [
                json.loads(line)
                for line in (snapshot / "eligible_papers.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            pool = candidate_pool(payloads["rows"], papers, seed="s", pool_multiplier=2)
            hidden = {row["paper_id"] for row in payloads["rows"] if row["split"] == "hidden_test"}
            pool_ids = {row["paper_id"] for row in pool["candidates"]}
            self.assertTrue(pool_ids)
            self.assertFalse(pool_ids & hidden)
            self.assertTrue(all(row["split"] in CANDIDATE_SPLITS for row in pool["candidates"]))


class LabelTests(unittest.TestCase):
    def test_labels_are_cached_and_a_second_run_calls_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, count=40)
            _, payloads = make_split(root, snapshot)
            papers = [
                json.loads(line)
                for line in (snapshot / "eligible_papers.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            from rfs.judge_training.pilot_selection import load_figure_index

            figure_index = load_figure_index(snapshot)
            pool = candidate_pool(payloads["rows"], papers, seed="s", pool_multiplier=3)
            cache = root / "labels.jsonl"
            calls = []

            def counting_labeller(prompt, image):
                calls.append(Path(image).name)
                return fake_labeller(prompt, image)

            first = label_candidates(
                pool["candidates"],
                snapshot,
                figure_index,
                cache_path=cache,
                work_dir=root / "work",
                labeller=counting_labeller,
                workers=2,
            )
            self.assertEqual(first["failures"], [])
            self.assertEqual(first["newly_labelled"], len(calls))
            self.assertTrue(calls)
            again = label_candidates(
                pool["candidates"],
                snapshot,
                figure_index,
                cache_path=cache,
                work_dir=root / "work",
                labeller=counting_labeller,
                workers=2,
            )
            self.assertEqual(again["newly_labelled"], 0)
            self.assertEqual(len(calls), first["newly_labelled"])
            cached = load_label_cache(cache)
            self.assertEqual(len(cached), first["newly_labelled"])

    def test_a_failing_labeller_is_reported_not_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, count=20)
            _, payloads = make_split(root, snapshot)
            papers = [
                json.loads(line)
                for line in (snapshot / "eligible_papers.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            from rfs.judge_training.pilot_selection import load_figure_index

            pool = candidate_pool(payloads["rows"], papers, seed="s", pool_multiplier=3)

            def broken(prompt, image):
                raise RuntimeError("provider down")

            report = label_candidates(
                pool["candidates"],
                snapshot,
                load_figure_index(snapshot),
                cache_path=root / "labels.jsonl",
                work_dir=root / "work",
                labeller=broken,
                workers=1,
            )
            self.assertTrue(report["failures"])
            self.assertEqual(report["newly_labelled"], 0)


class SelectionTests(unittest.TestCase):
    def make_labels(self, count=400):
        labels = []
        for index in range(count):
            labels.append(
                {
                    "paper_id": "p%04d" % index,
                    "split": "validation" if index % 7 == 0 else "train",
                    "domain": ("cv", "rag", "agent", "materials", "biology")[index % 5],
                    "figure_type": ("framework", "pipeline", "technical_route")[index % 3],
                    "structure": ("linear", "branching", "converging", "feedback_loop", "multi_panel")[
                        (index // 5) % 5
                    ],
                    "complexity": 1 + index % 5,
                    "confidence": 0.9,
                }
            )
        return labels

    def test_selection_uses_the_quotas_and_spreads_over_the_cells(self):
        quotas = {"cv": (4, 1), "rag": (4, 1), "agent": (4, 1), "materials": (3, 1), "biology": (3, 1)}
        selected = select_pilot(self.make_labels(), seed="s", quotas=quotas)
        records = selected["records"]
        self.assertEqual(selected["shortfalls"], [])
        train = [row for row in records if row["split"] == "train"]
        self.assertEqual(len(train), 4 + 4 + 4 + 3 + 3)
        cv_train = [row for row in train if row["domain"] == "cv"]
        self.assertEqual(len(cv_train), 4)
        self.assertEqual(
            len({(row["figure_type"], row["structure"]) for row in cv_train}),
            4,
        )

    def test_selection_is_reproducible(self):
        quotas = {"cv": (4, 1)}
        first = select_pilot(self.make_labels(), seed="s", quotas=quotas)["records"]
        second = select_pilot(self.make_labels(), seed="s", quotas=quotas)["records"]
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))


class VersionTests(unittest.TestCase):
    def build(self, root, count=300):
        snapshot = make_snapshot(root, count=count)
        split_dir, payloads = make_split(root, snapshot)
        papers = [
            json.loads(line)
            for line in (snapshot / "eligible_papers.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        from rfs.judge_training.pilot_selection import load_figure_index

        figure_index = load_figure_index(snapshot)
        pool = candidate_pool(payloads["rows"], papers, seed="s")["candidates"]
        labels = caption_fallback_labels(pool, figure_index)
        labelled = [labels[row["paper_id"]] for row in pool if row["paper_id"] in labels]
        pilot = build_pilot_payloads(
            split_dir,
            snapshot,
            labelled=labelled,
            pool=pool,
            figure_index=figure_index,
            pilot_id="p",
            seed="s",
            labeller_name="caption_fallback",
        )
        return snapshot, split_dir, pilot

    def test_a_fallback_pilot_is_sealed_and_revalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, payloads = self.build(root)
            written = write_pilot_version(root / "pilot", payloads)
            self.assertTrue(written["sealed"], payloads["validation"])
            check = validate_pilot_version(root / "pilot")
            self.assertTrue(check["ok"], check)
            self.assertTrue(check["sealed"])
            self.assertFalse(check["by_split"].get("hidden_test"))
            with self.assertRaises(FileExistsError):
                write_pilot_version(root / "pilot", payloads)

    def test_the_coverage_report_describes_the_labelled_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, payloads = self.build(root)
            coverage = payloads["coverage"]
            self.assertNotIn("None", " ".join(coverage["pool_cells"]))
            self.assertTrue(
                set(coverage["selected_cells"]) <= set(coverage["pool_cells"]),
                "a selected cell must exist in the pool it was drawn from",
            )
            self.assertEqual(
                sum(coverage["pool_cells"].values())
                + coverage["pool_papers_without_labels"],
                coverage["pool_size"],
            )

    def test_a_pilot_never_contains_a_hidden_test_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, _, payloads = self.build(root)
            hidden_manifest = json.loads(
                (root / "split" / "hidden_test_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(hidden_manifest["seal"] == "sealed")
            for row in payloads["records"]:
                self.assertIn(row["split"], CANDIDATE_SPLITS)

    def test_asset_verification_detects_a_changed_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, _, payloads = self.build(root)
            from rfs.judge_training.pilot_selection import load_figure_index

            figure_index = load_figure_index(snapshot)
            records = payloads["records"]
            self.assertTrue(verify_selected_assets(snapshot, records, figure_index)["ok"])
            victim = snapshot / records[0]["figure_relative_path"]
            victim.write_bytes(figure_png(999))
            check = verify_selected_assets(snapshot, records, figure_index)
            self.assertFalse(check["ok"])
            self.assertEqual(check["missing_or_changed"], [records[0]["paper_id"]])

    def test_caption_fallback_declares_itself(self):
        labels = caption_fallback_labels(
            [{"paper_id": "p1", "split": "train", "domain": "cv", "caption": "The training pipeline of X."}]
        )
        record = labels["p1"]
        self.assertEqual(record["label_source"], "caption_fallback")
        self.assertEqual(record["figure_type"], "pipeline")
        self.assertEqual(record["structure"], "unlabelled")
        self.assertEqual(record["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()

class StructureGateTests(unittest.TestCase):
    """The second pass that keeps plots and photographs out of the structural cells."""

    def make_labels(self, count=400):
        return SelectionTests.make_labels(SelectionTests(), count)

    @staticmethod
    def small_quotas():
        return {"cv": (4, 1), "rag": (4, 1), "agent": (4, 1), "materials": (3, 1), "biology": (3, 1)}

    @staticmethod
    def pilot_size():
        return 4 + 4 + 4 + 3 + 3 + 1 + 1 + 1 + 1 + 1

    def test_a_rejected_pick_is_demoted_and_refilled_from_the_same_cell(self):
        rows = self.make_labels()
        quotas = {"cv": (4, 1)}
        first = [
            row
            for row in select_pilot(rows, seed="s", quotas=quotas)["records"]
            if row["domain"] == "cv" and row["split"] == "train"
        ]
        victim = first[0]
        demoted = demote_rejected(rows, {victim["paper_id"]: {"is_diagram": False}})
        moved = [row for row in demoted if row["paper_id"] == victim["paper_id"]][0]
        self.assertEqual(moved["figure_type"], "other")
        self.assertEqual(moved["structure_gate"], "not_diagram")
        second = [
            row
            for row in select_pilot(demoted, seed="s", quotas=quotas)["records"]
            if row["domain"] == "cv" and row["split"] == "train"
        ]
        self.assertEqual(len(second), 4)
        self.assertNotIn(victim["paper_id"], [row["paper_id"] for row in second])
        self.assertEqual(
            {(row["figure_type"], row["structure"]) for row in first},
            {(row["figure_type"], row["structure"]) for row in second},
        )

    def test_the_gate_only_sees_the_picks(self):
        rows = self.make_labels()
        asked = []

        def runner(pending):
            asked.extend(row["paper_id"] for row in pending)
            return {row["paper_id"]: {"is_diagram": True} for row in pending}

        result = select_pilot_verified(
            rows, seed="s", quotas=self.small_quotas(), gate_runner=runner
        )
        self.assertEqual(len(result["selection"]["records"]), self.pilot_size())
        self.assertLess(len(asked), len(rows))
        self.assertEqual(sorted(asked), sorted(result["checked"]))

    def test_a_verified_selection_keeps_no_rejected_structural_pick(self):
        rows = self.make_labels()

        def runner(pending):
            return {
                row["paper_id"]: {"is_diagram": int(row["paper_id"][1:]) % 2 == 1}
                for row in pending
            }

        result = select_pilot_verified(
            rows, seed="s", quotas=self.small_quotas(), gate_runner=runner
        )
        records = result["selection"]["records"]
        self.assertEqual(len(records), self.pilot_size())
        for row in records:
            if row["figure_type"] in STRUCTURAL_TYPES:
                self.assertEqual(int(row["paper_id"][1:]) % 2, 1, row)

    def test_a_cached_rejection_is_applied_before_the_first_selection(self):
        """A resumed run must not hand back the picks the gate already refused."""

        rows = self.make_labels()
        without_gate = select_pilot(rows, seed="s", quotas=self.small_quotas())["records"]
        structural = [row for row in without_gate if row["figure_type"] in STRUCTURAL_TYPES]
        refused = structural[0]["paper_id"]
        asked = []

        def runner(pending):
            asked.extend(row["paper_id"] for row in pending)
            return {row["paper_id"]: {"is_diagram": True} for row in pending}

        result = select_pilot_verified(
            rows,
            seed="s",
            quotas=self.small_quotas(),
            gate={refused: {"is_diagram": False}},
            gate_runner=runner,
        )
        picked = [row["paper_id"] for row in result["selection"]["records"]]
        self.assertNotIn(refused, picked)
        self.assertEqual(len(picked), self.pilot_size())
        self.assertNotIn(refused, asked)
        demoted = [row for row in result["labelled"] if row["paper_id"] == refused][0]
        self.assertEqual(demoted["figure_type"], "other")

    def test_gate_answers_are_cached_instead_of_asked_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, count=40)
            figure_index = load_figure_index(snapshot)
            _, payloads = make_split(root, snapshot)
            papers = [
                json.loads(line)
                for line in (snapshot / "eligible_papers.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            pool = candidate_pool(payloads["rows"], papers, seed="s")["candidates"]
            calls = []

            def labeller(prompt, image):
                calls.append(str(image))
                self.assertIn("is_diagram", prompt)
                return {"is_diagram": False, "reason": "a scatter plot"}

            first = gate_labels(
                pool[:5],
                snapshot,
                figure_index,
                cache_path=root / "gate.jsonl",
                work_dir=root / "work",
                labeller=labeller,
                workers=1,
            )
            self.assertEqual(first["newly_checked"], 5)
            self.assertEqual(len(first["rejected"]), 5)
            second = gate_labels(
                pool[:5],
                snapshot,
                figure_index,
                cache_path=root / "gate.jsonl",
                work_dir=root / "work",
                labeller=labeller,
                workers=1,
            )
            self.assertEqual(second["newly_checked"], 0)
            self.assertEqual(second["checked"], 5)
            self.assertEqual(len(calls), 5)

    def test_a_rendered_page_figure_is_indexed_like_any_other_figure(self):
        # A snapshot that only had the paper's framework figure as a rendered
        # page still froze a framework figure; not indexing it here would drop
        # that paper from a subset the snapshot itself passed.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = make_snapshot(root, count=5)
            manifest = snapshot / "asset_manifest.jsonl"
            relative = "assets/p0000/rendered/framework.png"
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(figure_png(99))
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows.append(
                {
                    "asset_id": "asset-page-0000",
                    "paper_id": "p0000",
                    "role": "figure_page_render",
                    "relative_path": relative,
                    "sha256": "0" * 64,
                    "suffix": ".png",
                }
            )
            manifest.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            index = load_figure_index(snapshot)
            self.assertIn("asset-page-0000", index)
            self.assertEqual(index["asset-page-0000"]["paper_id"], "p0000")
