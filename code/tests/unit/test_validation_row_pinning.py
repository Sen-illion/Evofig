"""A validation pins the row it measures and continues a control arm it had to abandon.

Two failures of the same kind made this file necessary. The library a measurement was queued from
is not frozen while the measurement runs, so a resumed attempt that asked it again for its row
found nothing and died with "unknown experience" after a wave had already been paid for. And an
arm is about forty minutes of generation, so a call that stopped inside one threw the whole arm
away and paid for every stage again -- which is why no validation ever reached a single decided
case.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import ExperienceStore
from rfs.utils import write_json
from rfs.workflows.direct_experience_validation import ROW_UNDER_TEST_FILE, run_direct_experience_validation

OFFLINE = {
    "planner_mode": "heuristic",
    "method_mode": "heuristic",
    "structure_mode": "heuristic",
    "text_plan_mode": "heuristic",
    "ppt_plan_mode": "heuristic",
    "color_theme_mode": "fallback",
    "icon_mode": "placeholder",
    "visual_review_mode": "heuristic",
    "judge_mode": "heuristic",
    "candidates_per_round": 1,
    "max_rounds": 1,
}

#: The artifacts the per-stage replay insists on before it will continue from a source at all.
ARM_PREFIX = ("input_manifest.json", "document_model.json", "method_extraction_report.json")

EXPERIENCE = {
    "experience_id": "e1",
    "status": "current_case_validated",
    "domain": "direct_editable",
    "scope": "slot",
    "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
    "hypothesis": "a single reading path reads better",
    "action": {"type": "regen_assets", "slot_ids": ["icon_n"]},
    "graph_scope": {"slot_ids": ["icon_n"]},
    "applicability": "common",
    "repair_gain": 0.25,
    "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.75},
}


def _record(paper_id, domain, motif=None):
    return {"paper_id": paper_id, "domain": domain, "features": {"node_count": 4, "edge_count": 3, "motif_histogram": motif or {"linear": 2}}}


def _suite(papers: dict[str, Path]) -> dict:
    cases = []
    for case_id, domain, motif in (
        ("a", "ai_method", None),
        ("b", "ai_method", None),
        ("c", "biology", None),
        ("d", "ai_method", {"cyclic": 2}),
    ):
        record = _record(case_id, domain, motif)
        if case_id == "d":
            record["slot_ids"] = ["slot_other"]
        cases.append({"case_id": case_id, "paper_id": case_id, "paper": str(papers[case_id]), "domain": domain, "graph_record": record})
    return {
        "query_graph_record": _record("query", "ai_method"),
        "selector": {"quota": {"same_domain_same_structure": 2, "cross_domain_same_structure": 1, "same_domain_cross_structure": 0, "boundary": 1}},
        "cases": cases,
    }


class _Workspace:
    """A suite, a library and the papers they name, all under one temporary root."""

    def __init__(self, name: str) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.papers = {case_id: self.root / f"{case_id}.pdf" for case_id in ("a", "b", "c", "d")}
        for path in self.papers.values():
            path.write_bytes(b"%PDF-1.4 placeholder")
        self.library = self.root / "library"
        self.suite = self.root / "suite.json"
        self.suite.write_text(json.dumps(_suite(self.papers)), encoding="utf-8")

    def cleanup(self) -> None:
        self._tmp.cleanup()

    def store(self):
        return ExperienceStore(self.library)


class _ArmRunner:
    """A stand-in arm: it reports an arm the way a run does, and can be cut off on purpose."""

    def __init__(self, calls: list[dict], *, cut_off_after: int | None = None) -> None:
        self.calls = calls
        self.cut_off_after = cut_off_after
        self.cut_off = 0

    def __call__(self, **options):
        out = Path(options["out"])
        out.mkdir(parents=True, exist_ok=True)
        enabled = options.get("seed_experience") is not None
        self.calls.append(
            {
                "out": str(out),
                "enabled": enabled,
                "reuse_upstream_from": options.get("reuse_upstream_from"),
                "replay_upstream_from": options.get("replay_upstream_from"),
                "fork_before": options.get("fork_before"),
            }
        )
        if not enabled and self.cut_off_after is not None and self.cut_off < self.cut_off_after:
            self.cut_off += 1
            write_json(out / "input_manifest.json", {"positive_references": []})
            write_json(out / "document_model.json", {"source_sha256": ""})
            write_json(out / "method_extraction_report.json", {"ok": True})
            raise RuntimeError("the call ended inside the control arm")
        write_json(out / "round_00" / "state_before.json", {"state_hash": "raw", "state": {"layout": {"nodes": []}}})
        return {"ok": True, "acceptance_judgement": {"overall_score": 0.8 if enabled else 0.7, "blocking_issue_count": 0}}


class RowUnderTestTests(unittest.TestCase):
    def setUp(self):
        self.space = _Workspace("row")
        self.addCleanup(self.space.cleanup)
        self.assertTrue(self.space.store().append({**EXPERIENCE}))

    def test_the_row_under_test_is_pinned_into_the_measurement_directory(self):
        out = self.space.root / "clean"
        run_direct_experience_validation(self.space.suite, self.space.library, "e1", out, runner=_ArmRunner([]), run_options=OFFLINE)
        pinned = json.loads((out / ROW_UNDER_TEST_FILE).read_text(encoding="utf-8"))
        self.assertEqual("e1", pinned["experience_id"])
        self.assertEqual("e1", pinned["experience"]["experience_id"])

    def test_the_pinned_row_is_measured_after_the_library_stops_holding_it(self):
        out = self.space.root / "pinned"
        # The measurement opens while the library still holds the row and is cut off inside its
        # first arm, so the row is pinned but the set is not finished.
        with self.assertRaises(RuntimeError):
            run_direct_experience_validation(self.space.suite, self.space.library, "e1", out, runner=_ArmRunner([], cut_off_after=1), run_options=OFFLINE)
        self.assertTrue((out / ROW_UNDER_TEST_FILE).is_file())
        # The row leaves the library the way an archive sweep or a retirement would take it: the
        # measurement that already opened is still about the row it was queued for.
        (self.space.library / "judge_experiences.jsonl").unlink()
        calls: list[dict] = []
        report = run_direct_experience_validation(self.space.suite, self.space.library, "e1", out, runner=_ArmRunner(calls), run_options=OFFLINE)
        self.assertTrue(calls)
        self.assertEqual("e1", report["experience_id"])

    def test_a_library_that_still_holds_no_such_row_is_still_refused(self):
        out = self.space.root / "unknown"
        with self.assertRaises(ValueError):
            run_direct_experience_validation(self.space.suite, self.space.library, "e-missing", out, runner=_ArmRunner([]), run_options=OFFLINE)


class ControlArmResumeTests(unittest.TestCase):
    def setUp(self):
        self.space = _Workspace("arm")
        self.addCleanup(self.space.cleanup)
        self.assertTrue(self.space.store().append({**EXPERIENCE}))
        self.out = self.space.root / "out"

    def _crash_inside_the_first_control_arm(self) -> None:
        with self.assertRaises(RuntimeError):
            run_direct_experience_validation(self.space.suite, self.space.library, "e1", self.out, runner=_ArmRunner([], cut_off_after=1), run_options=OFFLINE)

    def test_an_interrupted_control_arm_continues_its_own_unfinished_run(self):
        self._crash_inside_the_first_control_arm()
        off_dir = self.out / "a" / "off"
        self.assertTrue(all((off_dir / name).is_file() for name in ARM_PREFIX))

        calls: list[dict] = []
        run_direct_experience_validation(self.space.suite, self.space.library, "e1", self.out, runner=_ArmRunner(calls), run_options=OFFLINE)

        resumed = [item for item in calls if Path(item["out"]).resolve() == off_dir.resolve()]
        self.assertEqual(1, len(resumed))
        self.assertEqual(str(off_dir.resolve()), str(Path(resumed[0]["reuse_upstream_from"]).resolve()))
        self.assertTrue((off_dir / "arm_resume.json").is_file())

    def test_the_memory_arm_keeps_replaying_the_control_arm_instead_of_itself(self):
        self._crash_inside_the_first_control_arm()
        calls: list[dict] = []
        run_direct_experience_validation(self.space.suite, self.space.library, "e1", self.out, runner=_ArmRunner(calls), run_options=OFFLINE)

        on_calls = [item for item in calls if item["enabled"]]
        self.assertTrue(on_calls)
        for item in on_calls:
            self.assertEqual("", str(item["reuse_upstream_from"] or ""))
            self.assertEqual(str(Path(item["out"]).parent / "off"), item["replay_upstream_from"])
            self.assertTrue(item["fork_before"])

    def test_a_fresh_control_arm_is_not_told_to_resume_anything(self):
        calls: list[dict] = []
        run_direct_experience_validation(self.space.suite, self.space.library, "e1", self.out, runner=_ArmRunner(calls), run_options=OFFLINE)
        off_calls = [item for item in calls if not item["enabled"]]
        self.assertTrue(off_calls)
        for item in off_calls:
            self.assertEqual("", str(item["reuse_upstream_from"] or ""))

    def test_the_resume_stays_out_of_the_pair_configuration_hash(self):
        # Two runs of the same suite, one of them continuing a control arm that was cut off. The
        # hash the pair audit reads has to be the same number, or a resumed pair would look like a
        # different experiment from the one it is continuing.
        clean = self.space.root / "clean"
        run_direct_experience_validation(self.space.suite, self.space.library, "e1", clean, runner=_ArmRunner([]), run_options=OFFLINE)
        clean_hash = json.loads((clean / "a" / "off" / "paired_validation_result.json").read_text(encoding="utf-8"))["paired_config_hash"]

        self._crash_inside_the_first_control_arm()
        resumed_calls: list[dict] = []
        run_direct_experience_validation(self.space.suite, self.space.library, "e1", self.out, runner=_ArmRunner(resumed_calls), run_options=OFFLINE)
        self.assertTrue(any(item["reuse_upstream_from"] for item in resumed_calls))
        resumed_hash = json.loads((self.out / "a" / "off" / "paired_validation_result.json").read_text(encoding="utf-8"))["paired_config_hash"]

        self.assertEqual(clean_hash, resumed_hash)


if __name__ == "__main__":
    unittest.main()
