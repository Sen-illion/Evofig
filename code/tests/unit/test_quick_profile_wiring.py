"""The quick bar is reachable from the running chain, not only from a direct call.

A profile nobody names is a profile that never runs, so these tests pin the wire: the continuous
config's word reaches the measurement, the session's validate station hands its word over, the CLI
can say it, and a measurement already opened keeps the bar it opened with.
"""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.cli import build_parser
from rfs.evolution.direct import ExperienceStore
from rfs.utils import write_json
from rfs.workflows import run_evolution_session
from rfs.workflows.continuous_evolution import CONTINUOUS_EVENTS, CONTINUOUS_STATE, run_continuous_evolution
from rfs.workflows.direct_experience_validation import run_direct_experience_validation
from tests.unit.test_continuous_validation_queue import _config as _continuous_config
from tests.unit.test_continuous_validation_queue import _evolve_runner, _wait_for
from tests.unit.test_evolution_session import FakeEvolve, FakeValidate, _Clock, _config, _experience_row, _paper, _suite


def _events(out: Path) -> list[dict]:
    path = out / CONTINUOUS_EVENTS
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ContinuousBarTests(unittest.TestCase):
    def _one_wave(self, out: Path, validation: dict | str, seen: list):
        def validate(**options):
            seen.append(dict(options))
            return {"status": "quick_validated", "ladder": {"stopped_because": ""}, "summary": {"cases": 1}, "all_pairs_isolated": True, "all_cases_measured": True}

        run_continuous_evolution(
            _continuous_config(validation=validation),
            out,
            experience_library=str(out / "library"),
            evolve_runner=_evolve_runner([]),
            review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
            validate_runner=validate,
            clock=lambda: 0.0,
            max_waves=1,
        )
        self.assertTrue(_wait_for(lambda: seen))

    def test_the_config_names_the_bar_the_measurement_runs_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            seen: list[dict] = []
            self._one_wave(out, {"suite": "/suites/four_groups.json", "validation_profile": "quick"}, seen)
            self.assertEqual("quick", seen[0]["validation_profile"])
            self.assertTrue(_wait_for(lambda: any(item["event"] == "validation_finished" for item in _events(out))))
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
            record = next(iter(state["validations"].values()))
            self.assertEqual("quick", record["validation_profile"])
            started = [item for item in _events(out) if item["event"] == "validation_started"]
            self.assertEqual("quick", started[0]["validation_profile"])

    def test_a_config_that_names_no_bar_leaves_it_to_the_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            seen: list[dict] = []
            self._one_wave(out, {"suite": "/suites/four_groups.json"}, seen)
            self.assertEqual("", seen[0]["validation_profile"])

    def test_a_suite_written_as_a_bare_string_still_measures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            seen: list[dict] = []
            self._one_wave(out, "/suites/four_groups.json", seen)
            self.assertEqual("", seen[0]["validation_profile"])


class SessionStationBarTests(unittest.TestCase):
    def test_a_validate_station_hands_its_bar_to_the_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            ExperienceStore(library).append_result(_experience_row("direct-exp-known"))
            config = _config(
                root,
                [self._evolve_station(root, "paper-a"), {"kind": "validate", "experience_id": "direct-exp-known", "suite": str(_suite(root)), "validation_profile": "quick"}],
            )
            validate = FakeValidate(status="quick_validated")
            report = run_evolution_session(
                config,
                root / "session",
                experience_library=library,
                evolve_runner=FakeEvolve(),
                validate_runner=validate,
                clock=_Clock(),
                sleeper=lambda _s: None,
            )
            self.assertEqual("quick", validate.calls[0]["validation_profile"])
            station = report["stations"][1]
            self.assertEqual("completed", station["status"])

    def test_a_station_that_named_no_bar_says_so_rather_than_guessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            ExperienceStore(library).append_result(_experience_row("direct-exp-known"))
            config = _config(root, [{"kind": "validate", "experience_id": "direct-exp-known", "suite": str(_suite(root))}])
            validate = FakeValidate()
            run_evolution_session(
                config,
                root / "session",
                experience_library=library,
                evolve_runner=FakeEvolve(),
                validate_runner=validate,
                clock=_Clock(),
                sleeper=lambda _s: None,
            )
            self.assertEqual("", validate.calls[0]["validation_profile"])

    @staticmethod
    def _evolve_station(root: Path, name: str, **overrides) -> dict:
        station = {"kind": "evolve", "paper": str(_paper(root, f"{name}.pdf")), "paper_id": name, "max_rounds": 1, "candidates_per_round": 1}
        station.update(overrides)
        return station


class QuickBarIsAPassTests(unittest.TestCase):
    def test_a_row_kept_on_the_quick_rung_counts_as_a_passed_station(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            ExperienceStore(library).append_result(_experience_row("direct-exp-known"))
            config = _config(
                root,
                [{"kind": "validate", "experience_id": "direct-exp-known", "suite": str(_suite(root)), "validation_profile": "quick"}],
            )
            report = run_evolution_session(
                config,
                root / "session",
                experience_library=library,
                evolve_runner=FakeEvolve(),
                validate_runner=FakeValidate(status="quick_validated"),
                clock=_Clock(),
                sleeper=lambda _s: None,
            )
            station = report["stations"][0]
            self.assertTrue(station["ok"])
            self.assertEqual("quick_validated", station["evidence"]["status_after"])


class CommandLineBarTests(unittest.TestCase):
    def test_the_validate_command_can_name_the_bar(self):
        args = build_parser().parse_args(["paper-evolve-validate", "--suite", "s.json", "--experience-library", "lib", "--experience-id", "e1", "--out", "out", "--validation-profile", "quick"])
        self.assertEqual("quick", args.validation_profile)

    def test_the_validate_command_defaults_to_the_suites_own_word(self):
        args = build_parser().parse_args(["paper-evolve-validate", "--suite", "s.json", "--experience-library", "lib", "--experience-id", "e1", "--out", "out"])
        self.assertEqual("", args.validation_profile)


QUICK_SUITE = {
    "query_graph_record": {"paper_id": "query", "domain": "ai_method", "features": {"node_count": 4, "edge_count": 3, "motif_histogram": {"linear": 2}}},
    "cases": [
        {"case_id": "a", "paper_id": "a", "paper": "a.pdf", "domain": "ai_method", "graph_record": {"paper_id": "a", "domain": "ai_method", "features": {"node_count": 4, "edge_count": 3, "motif_histogram": {"linear": 2}}}},
        {"case_id": "c", "paper_id": "c", "paper": "c.pdf", "domain": "biology", "graph_record": {"paper_id": "c", "domain": "biology", "features": {"node_count": 4, "edge_count": 3, "motif_histogram": {"linear": 2}}}},
    ],
}

ROW = {
    "experience_id": "e1",
    "status": "current_case_validated",
    "domain": "direct_editable",
    "scope": "slot",
    "guidance": "Keep the repaired part on the same reading path as the rest of the figure.",
    "hypothesis": "larger icon improves clarity",
    "action": {"type": "regen_assets", "slot_ids": ["icon_n"]},
    "graph_scope": {"slot_ids": ["icon_n"]},
    "applicability": "common",
    "repair_gain": 0.25,
    "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.75},
}


class PinnedBarTests(unittest.TestCase):
    def _runner(self, options, value):
        (Path(options["out"]) / "round_00").mkdir(parents=True, exist_ok=True)
        (Path(options["out"]) / "round_00" / "state_before.json").write_text(
            json.dumps({"state_hash": "start", "state": {"layout": {"panels": [{"id": "p1", "bbox_percent": {"x": 0.05, "y": 0.1, "w": 0.9, "h": 0.8}}], "nodes": [{"id": "n1", "panel_id": "p1", "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.2}}], "connectors": []}}}),
            encoding="utf-8",
        )
        return {"ok": True, "judgement": {"overall_score": value, "blocking_issue_count": 0}}

    def test_a_measurement_in_progress_keeps_the_bar_it_opened_with(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library"
            ExperienceStore(library).append_result(dict(ROW))
            suite = root / "suite.json"
            suite.write_text(json.dumps(QUICK_SUITE), encoding="utf-8")
            runner = lambda **options: self._runner(options, 0.9 if options.get("seed_experience") else 0.8)

            first = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner, validation_profile="quick")
            self.assertEqual("quick", first["validation_profile"])
            pin = json.loads((root / "out" / "row_under_test.json").read_text(encoding="utf-8"))
            self.assertEqual("quick", pin["validation_profile"])

            resumed = run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner, validation_profile="full")
            self.assertEqual("quick", resumed["validation_profile"])


class OneCaseNotFiveTests(unittest.TestCase):
    """The quick bar is one matched paper. The evidence has to be the same shape a caller reads."""

    def _run(self, root: Path, library: Path, suite: Path, profile: str) -> dict:
        def runner(**options):
            (Path(options["out"]) / "round_00").mkdir(parents=True, exist_ok=True)
            (Path(options["out"]) / "round_00" / "state_before.json").write_text(
                json.dumps({"state_hash": "start", "state": {"layout": {"panels": [{"id": "p1", "bbox_percent": {"x": 0.05, "y": 0.1, "w": 0.9, "h": 0.8}}], "nodes": [{"id": "n1", "panel_id": "p1", "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.2}}], "connectors": []}}}),
                encoding="utf-8",
            )
            return {"ok": True, "judgement": {"overall_score": 0.9 if options.get("seed_experience") else 0.8, "blocking_issue_count": 0}}

        return run_direct_experience_validation(suite, library, "e1", root / "out", runner=runner, validation_profile=profile)

    def _space(self, tmp: str):
        root = Path(tmp)
        library = root / "library"
        ExperienceStore(library).append_result(dict(ROW))
        suite = root / "suite.json"
        suite.write_text(json.dumps(QUICK_SUITE), encoding="utf-8")
        return root, library, suite

    def test_the_quick_bar_asks_one_matched_paper_and_keeps_the_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, library, suite = self._space(tmp)
            report = self._run(root, library, suite, "quick")
            self.assertEqual(1, report["summary"]["cases"])
            self.assertEqual(["cross_domain_same_structure"], [item.get("group") for item in report["results"]])
            self.assertEqual("quick_validated", report["status"])
            self.assertTrue(json.loads((root / "out" / "case_set_qualification.json").read_text(encoding="utf-8"))["ok"])

    def test_the_full_bar_still_refuses_a_pool_that_cannot_fill_its_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, library, suite = self._space(tmp)
            report = self._run(root, library, suite, "full")
            self.assertEqual(0, report["summary"]["cases"])
            self.assertEqual("unverifiable", report["status"])
            self.assertEqual("full", report["validation_profile"])

    def test_the_quick_bar_does_not_climb_the_full_ladder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, library, suite = self._space(tmp)
            report = self._run(root, library, suite, "quick")
            self.assertNotIn(report["status"], ("promoted", "same_domain_validated", "cross_domain_validated"))


if __name__ == "__main__":
    unittest.main()