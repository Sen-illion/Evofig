"""The cost block prices tokens on one scale, pairs on delivery, and never hides an estimate."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "pair_with_baselines", ROOT / "scripts" / "pair_with_baselines.py"
)
PAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIR)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def our_paper(root: Path, domain: str, name: str, usd: float, seconds: float = 100.0) -> None:
    write_json(root / domain / name / "cost_report.json", {
        "generation_usd": usd,
        "text_usd": usd * 0.5,
        "image_usd": usd * 0.5,
        "elapsed_seconds": seconds,
        "text": {"prompt_tokens": 1000, "completion_tokens": 100},
        "image": {"images_billed": 4},
    })


def baseline_paper(root: Path, system: str, subdir: str, name: str, usage: dict | None,
                   ok: bool = True) -> Path:
    paper = root / system / subdir / name
    payload = {
        "system": system,
        "paper_id": name,
        "seconds": 30.0,
        "ok": ok,
    }
    if usage is not None:
        payload["usage"] = usage
    write_json(paper / "run.json", payload)
    return paper


class PricingTests(unittest.TestCase):
    def test_tokens_are_priced_at_the_snapshot_rate(self) -> None:
        priced = PAIR.price_tokens(1_000_000, 1_000_000, 10)
        self.assertAlmostEqual(priced["text_usd"], 1.75, places=9)
        self.assertAlmostEqual(priced["image_usd"], 0.075, places=9)
        self.assertAlmostEqual(priced["usd"], 1.825, places=9)

    def test_median_of_an_even_count_averages_the_middle(self) -> None:
        self.assertEqual(PAIR.median([3, 1, 2, 4]), 2.5)
        self.assertEqual(PAIR.median([5]), 5.0)
        self.assertIsNone(PAIR.median([]))

    def test_to_figure_clock_reads_the_compilation_event(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paper = Path(folder)
            (paper / "progress.jsonl").write_text(
                "\n".join([
                    json.dumps({"stage": "ppt_compilation", "status": "started", "elapsed_seconds": 1.0}),
                    json.dumps({"stage": "ppt_compilation", "status": "completed", "elapsed_seconds": 146.9}),
                    json.dumps({"stage": "online_judge", "status": "completed", "elapsed_seconds": 240.0}),
                ]) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(PAIR.to_figure_seconds(paper), 146.9)


class BlockTests(unittest.TestCase):
    def _trees(self) -> tuple[Path, Path]:
        ours_root = Path(tempfile.mkdtemp()) / "ours"
        base_root = Path(tempfile.mkdtemp()) / "efb"
        for index, name in enumerate(["paper_a", "paper_b", "paper_c"]):
            our_paper(ours_root, "agent", name, usd=0.10 * (index + 1))
        return ours_root, base_root

    def test_a_failed_paper_is_counted_and_kept_out_of_the_medians(self) -> None:
        ours_root, base_root = self._trees()
        measured = {"text_prompt_tokens": 1_000_000, "text_completion_tokens": 0,
                    "image_calls": 0, "unmeasured_calls": 0, "usage_source": "probe"}
        baseline_paper(base_root, "system", "turbo", "paper_a", measured)
        baseline_paper(base_root, "system", "turbo", "paper_b", measured, ok=False)
        rows = PAIR.collect_baseline_costs(base_root, "system", "turbo")
        block = PAIR.cost_block(PAIR.collect_our_costs([ours_root]), {"system": rows}, seed=1)
        row = block["systems"]["system"]
        self.assertEqual(row["papers_listed"], 2)
        self.assertEqual(row["papers_delivered"], 1)
        self.assertEqual(row["papers_failed"], 1)
        self.assertEqual(row["metrics"]["usd"]["n"], 1)
        self.assertAlmostEqual(row["metrics"]["usd"]["median"], 0.25, places=9)

    def test_the_paired_difference_uses_only_shared_delivered_papers(self) -> None:
        ours_root, base_root = self._trees()
        measured = {"text_prompt_tokens": 1_000_000, "text_completion_tokens": 0,
                    "image_calls": 0, "unmeasured_calls": 0, "usage_source": "probe"}
        baseline_paper(base_root, "system", "turbo", "paper_a", measured)
        baseline_paper(base_root, "system", "turbo", "paper_zzz", measured)
        rows = PAIR.collect_baseline_costs(base_root, "system", "turbo")
        block = PAIR.cost_block(PAIR.collect_our_costs([ours_root]), {"system": rows}, seed=1)
        row = block["systems"]["system"]
        self.assertEqual(row["papers_delivered"], 2)
        self.assertEqual(row["papers_paired_with_ours"], 1)
        self.assertAlmostEqual(row["paired_usd_difference"]["mean"], 0.25 - 0.10, places=9)
        self.assertAlmostEqual(row["paired_usd_median_difference"], 0.25 - 0.10, places=9)

    def test_the_paired_table_states_the_difference_it_measured(self) -> None:
        """The column is filled from the per-paper differences, not left blank."""

        ours_root, base_root = self._trees()
        measured = {"text_prompt_tokens": 1_000_000, "text_completion_tokens": 0,
                    "image_calls": 0, "unmeasured_calls": 0, "usage_source": "probe"}
        baseline_paper(base_root, "system", "turbo", "paper_a", measured)
        rows = PAIR.collect_baseline_costs(base_root, "system", "turbo")
        block = PAIR.cost_block(PAIR.collect_our_costs([ours_root]), {"system": rows}, seed=1)
        text = PAIR.cost_markdown(block)
        self.assertIn("+0.150", text)
        self.assertIn("median difference", text)

    def test_a_missed_probe_log_falls_back_to_disk_and_says_so(self) -> None:
        ours_root, base_root = self._trees()
        paper = baseline_paper(base_root, "system", "turbo", "paper_a", None)
        (paper / "method.txt").write_text("x" * 400, encoding="utf-8")
        (paper / "fig_1.png").write_bytes(b"png")
        (paper / "fig_2.png").write_bytes(b"png")
        rows = PAIR.collect_baseline_costs(base_root, "system", "turbo")
        record = rows[PAIR.key_of("paper_a")]
        self.assertEqual(record["usage_source"], "estimated")
        self.assertEqual(record["image_calls"], 2)
        self.assertEqual(record["text_prompt_tokens"], 100)
        block = PAIR.cost_block(PAIR.collect_our_costs([ours_root]), {"system": rows}, seed=1)
        self.assertEqual(block["systems"]["system"]["usage_basis"], ["estimated"])

    def test_a_paper_the_row_could_not_start_is_counted_as_a_failure(self) -> None:
        """Three of LiveFigure's papers leave a status and no run; they still count."""

        ours_root, base_root = self._trees()
        measured = {"text_prompt_tokens": 1_000_000, "text_completion_tokens": 0,
                    "image_calls": 0, "unmeasured_calls": 0, "usage_source": "probe"}
        baseline_paper(base_root, "system", "turbo", "paper_a", measured)
        stalled = base_root / "system" / "turbo" / "paper_b"
        stalled.mkdir(parents=True)
        write_json(stalled / "status.json", {"ok": False, "reason": "missing method section"})
        rows = PAIR.collect_baseline_costs(base_root, "system", "turbo")
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[PAIR.key_of("paper_b")]["delivered"])
        block = PAIR.cost_block(PAIR.collect_our_costs([ours_root]), {"system": rows}, seed=1)
        self.assertEqual(block["systems"]["system"]["papers_listed"], 2)
        self.assertEqual(block["systems"]["system"]["papers_delivered"], 1)
        self.assertEqual(block["systems"]["system"]["papers_failed"], 1)

    def test_images_embedded_in_a_delivered_deck_are_counted(self) -> None:
        ours_root, base_root = self._trees()
        paper = baseline_paper(base_root, "system", "turbo", "paper_a", None)
        with zipfile.ZipFile(paper / "delivery.pptx", "w") as archive:
            archive.writestr("ppt/media/image1.png", b"a")
            archive.writestr("ppt/media/image2.png", b"b")
        rows = PAIR.collect_baseline_costs(base_root, "system", "turbo")
        self.assertEqual(rows[PAIR.key_of("paper_a")]["image_calls"], 2)

    def test_our_columns_are_cut_to_the_sample_the_row_lists(self) -> None:
        """A run root holding every paper must not be set against a row that lists one."""

        ours_root, base_root = self._trees()
        measured = {"text_prompt_tokens": 1_000_000, "text_completion_tokens": 0,
                    "image_calls": 0, "unmeasured_calls": 0, "usage_source": "probe"}
        baseline_paper(base_root, "system", "turbo", "paper_a", measured)
        rows = PAIR.collect_baseline_costs(base_root, "system", "turbo")
        block = PAIR.cost_block(PAIR.collect_our_costs([ours_root]), {"system": rows}, seed=1)
        self.assertEqual(block["sample_papers"], 1)
        self.assertEqual(block["ours_papers_in_sample"], 1)
        self.assertEqual(block["ours_metrics"]["usd"]["n"], 1)
        self.assertAlmostEqual(block["ours_metrics"]["usd"]["median"], 0.10, places=9)
        self.assertAlmostEqual(
            block["systems"]["system"]["paired_ours_usd"]["median"], 0.10, places=9
        )

    def test_a_rerun_row_can_name_its_own_path_beside_its_scored_sibling(self) -> None:
        """`image_turbo/own` is not under `image/`, so the row has to be able to say where it is."""

        base_root = Path(tempfile.mkdtemp()) / "efb"
        measured = {"text_prompt_tokens": 1_000_000, "text_completion_tokens": 0,
                    "image_calls": 0, "unmeasured_calls": 0, "usage_source": "probe"}
        baseline_paper(base_root, "system_turbo", "own", "paper_a", measured)
        baseline_paper(base_root, "system", "own", "paper_b", measured)
        named = PAIR.collect_baseline_costs(base_root, "system", "system_turbo/own")
        self.assertEqual(list(named), [PAIR.key_of("paper_a")])
        sibling = PAIR.collect_baseline_costs(base_root, "system", "own")
        self.assertEqual(list(sibling), [PAIR.key_of("paper_b")])


class SampleTests(unittest.TestCase):
    """A 1401-paper row on disk must not become the sample for a 22-paper table."""

    def test_the_sample_file_holds_every_row_to_the_papers_it_names(self) -> None:
        ours_root = Path(tempfile.mkdtemp()) / "ours"
        base_root = Path(tempfile.mkdtemp()) / "efb"
        measured = {"text_prompt_tokens": 1_000_000, "text_completion_tokens": 0,
                    "image_calls": 0, "unmeasured_calls": 0, "usage_source": "probe"}
        for name in ("paper_a", "paper_b", "paper_c"):
            our_paper(ours_root, "agent", name, usd=0.10)
            baseline_paper(base_root, "system_turbo", "own", name, measured)
        sample = Path(tempfile.mkdtemp()) / "sample.jsonl"
        sample.write_text(json.dumps({"paper_id": "paper_a"}) + "\n", encoding="utf-8")
        out = Path(tempfile.mkdtemp()) / "out"
        code = PAIR.main([
            "--ours-runs", str(ours_root), "--baseline-root", str(base_root),
            "--cost-row", "system:system_turbo/own", "--sample", str(sample),
            "--out", str(out),
        ])
        self.assertEqual(code, 0)
        cost = json.loads((out / "cost_arms.json").read_text(encoding="utf-8"))
        self.assertEqual(cost["sample_papers"], 1)
        self.assertEqual(cost["sample_file"], str(sample))
        self.assertEqual(cost["systems"]["system"]["papers_listed"], 1)
        self.assertEqual(cost["ours_metrics"]["usd"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
