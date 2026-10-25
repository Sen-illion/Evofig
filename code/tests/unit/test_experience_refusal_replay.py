"""Unit tests for scripts/replay_experience_refusals.py.

The replay re-judges refusals a run already recorded, so these tests drive it with synthetic
records that spell out each path the rule can take: a round refused over a defect it introduced,
the refusals the current rule still makes, and a record that names no reason at all. No rounds
are run and no historical data is needed.
"""

import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_replay_module():
    spec = importlib.util.spec_from_file_location("replay_experience_refusals", ROOT / "scripts" / "replay_experience_refusals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = load_replay_module()


def refusal(**overrides):
    base = {
        "stage": "build",
        "round": 2,
        "ok": False,
        "reason": "new_blocking_issues",
        "blocking_issue_count": 4,
        "new_blockers": 1,
        "new_blocking_defects": ["interface:box_pass"],
    }
    base.update(overrides)
    return base


class RejudgeRefusalTest(unittest.TestCase):
    def test_a_round_refused_over_a_defect_it_introduced_is_filed_now(self):
        answer = replay.rejudge_refusal(refusal())
        self.assertEqual(answer["decision"], "file")
        self.assertEqual(answer["carried_defects"], ["interface:box_pass"])
        self.assertEqual(answer["files_as"], "current_case_validated")

    def test_every_other_refusal_keeps_its_own_reason(self):
        answer = replay.rejudge_refusal(refusal(reason="no_after_measurement"))
        self.assertEqual(answer["decision"], "unchanged")
        self.assertEqual(answer["reason"], "no_after_measurement")

    def test_a_record_naming_no_reason_is_unresolved_rather_than_guessed(self):
        answer = replay.rejudge_refusal({"stage": "build", "round": 2})
        self.assertEqual(answer["decision"], "unresolved")

    def test_the_report_counts_what_would_enter_the_library(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "iter_02" / "round_02").mkdir(parents=True)
            (root / "iter_02" / "round_02" / "experience_rejected.json").write_text(
                json_text(refusal()), encoding="utf-8"
            )
            (root / "iter_02" / "round_03").mkdir(parents=True)
            (root / "iter_02" / "round_03" / "experience_rejected.json").write_text(
                json_text(refusal(reason="no_after_measurement")), encoding="utf-8"
            )
            paths = replay._iter_refusals(root)
            report = replay.replay_refusals(paths)
        self.assertEqual(report["refusal_count"], 2)
        self.assertEqual(report["now_filed_count"], 1)
        self.assertEqual(report["still_refused_count"], 1)
        self.assertEqual(report["still_refused_by_reason"], {"no_after_measurement": 1})
        self.assertEqual(report["carried_defects"], {"interface:box_pass": 1})
        self.assertTrue(all(check["ok"] for check in report["checks"]))


def json_text(payload):
    import json

    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
