"""The stage trace a round wrote is the same trace its trajectory record carries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.judge_training import records_from_run
from tests.unit.test_judge_training_dataset import _make_run


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class TrajectoryStageTraceTests(unittest.TestCase):
    def test_the_trajectory_record_carries_the_rounds_stage_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            trace = {
                "contract": "rfs.stage_trace",
                "version": 1,
                "round": 0,
                "changed_stages": ["ppt_compilation"],
                "metrics": {"before": {}, "after": {"crossing_count": 6}, "delta": {}},
            }
            _write(root / "round_00" / "candidate_00" / "stage_trace.json", trace)
            trajectory = next(item for item in records_from_run(root) if item["task_type"] == "trajectory")
            self.assertEqual(trajectory["labels"]["stage_trace"], trace)

    def test_a_fallback_trace_at_the_round_root_is_read_too(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            trace = {"contract": "rfs.stage_trace", "version": 1, "round": 0, "changed_stages": []}
            _write(root / "round_00" / "stage_trace.json", trace)
            trajectory = next(item for item in records_from_run(root) if item["task_type"] == "trajectory")
            self.assertEqual(trajectory["labels"]["stage_trace"], trace)

    def test_a_round_without_a_trace_carries_an_empty_one(self):
        with tempfile.TemporaryDirectory() as temp:
            root = _make_run(Path(temp) / "run")
            trajectory = next(item for item in records_from_run(root) if item["task_type"] == "trajectory")
            self.assertFalse(trajectory["labels"]["stage_trace"])


if __name__ == "__main__":
    unittest.main()
