"""A wave writes the experience library from several papers at once.

The stores were read-check-append-rewrite: two runs of a wave read the same rows, both appended,
and the manifest each of them rewrote described a file that never existed. These tests drive real
separate processes, because that is the shape of the race the loop has when it advances two base
papers in parallel, and assert the two things a lost row breaks: the row count and the numbering.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import ExperienceStore
from rfs.evolution.direct.controller import append_event

ROOT = Path(__file__).resolve().parents[2]

WORKERS = 4
ROWS_PER_WORKER = 25

#: The child process is a plain interpreter with the repository on its path. Keeping the whole
#: worker in one string makes the test independent of how the suite named this module.
WORKER = r"""
import json
import sys
from pathlib import Path

root, mode, out, index, count = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, root)

from rfs.evolution.direct import ExperienceStore
from rfs.evolution.direct.controller import append_event


def row(worker: int, number: int, *, same_id: bool = False) -> dict:
    experience_id = "direct-exp-shared" if same_id else f"direct-exp-{worker}-{number}"
    return {
        "experience_id": experience_id,
        "domain": "direct_editable",
        "scope": "global",
        "apply_at": "ppt_compilation",
        "memory_type": "connector",
        "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
        "hypothesis": "a shared vertical channel keeps a one-to-many fan-out readable",
        # The move is made distinct per row on purpose: this test is about the lock, so every row
        # has to be a lesson of its own. Two rows stating the same move under the same prohibition
        # are folded into one by the store, which is a different test's subject.
        "action": {"type": "reroute", "constraints": {"shared_vertical_channel": True, "max_segments": worker * 100 + number + 2}},
        "condition": {"topology": "pipeline", "node_count_bucket": "high"},
        "graph_scope": {"edge_ids": ["rel_12"]},
        "accepted": True,
        "repair_gain": 0.12,
        "status": "candidate",
        "applicability": "common",
        "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.62},
    }


if mode == "events":
    for number in range(count):
        append_event(out / "events.jsonl", "iteration_finished", worker=index, number=number)
elif mode == "rows":
    store = ExperienceStore(out)
    for number in range(count):
        print(json.dumps(store.append_result(row(index, number))))
elif mode == "same_row":
    store = ExperienceStore(out)
    print(json.dumps(store.append_result(row(index, 0, same_id=True))))
"""


def _run_worker(root: Path, mode: str, worker: int, count: int) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-c", WORKER, str(ROOT), mode, str(root), str(worker), str(count)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise AssertionError(f"a writer process failed: {completed.stderr.strip()[-2000:]}")
    return [line for line in completed.stdout.splitlines() if line.strip()]


class ConcurrentAppendTests(unittest.TestCase):
    def _workers(self, root: Path, mode: str, count: int) -> list[list[str]]:
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", WORKER, str(ROOT), mode, str(root), str(worker), str(count)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for worker in range(WORKERS)
        ]
        results: list[list[str]] = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=180)
            self.assertEqual(process.returncode, 0, stderr[-2000:])
            results.append([line for line in stdout.splitlines() if line.strip()])
        return results

    def test_rows_appended_by_separate_processes_are_all_kept_and_all_counted(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self._workers(root, "rows", ROWS_PER_WORKER)
            store = ExperienceStore(root)

            rows = store.records()
            self.assertEqual(len(rows), WORKERS * ROWS_PER_WORKER)
            self.assertEqual(len({str(item.get("experience_id")) for item in rows}), WORKERS * ROWS_PER_WORKER)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["records"], len(rows), "the manifest describes rows that are not there")
            self.assertEqual(store.integrity_report()["readable_records"], len(rows))

    def test_two_processes_appending_the_same_row_leave_exactly_one(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            results = self._workers(root, "same_row", 1)
            accepted = [item for lines in results for line in lines for item in [json.loads(line)] if item["ok"]]

            self.assertEqual(len(accepted), 1, results)
            self.assertEqual(len(ExperienceStore(root).records()), 1)
            self.assertEqual(json.loads((root / "manifest.json").read_text(encoding="utf-8"))["records"], 1)

    def test_events_appended_by_separate_processes_keep_a_unique_numbering(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self._workers(root, "events", ROWS_PER_WORKER)

            events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(events), WORKERS * ROWS_PER_WORKER)
            numbers = [int(item["seq"]) for item in events]
            self.assertEqual(len(set(numbers)), len(numbers), "two writers were given the same number")
            self.assertEqual(sorted(numbers), list(range(1, len(events) + 1)))

    def test_an_event_written_from_many_threads_is_not_lost(self):
        import threading

        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "events.jsonl"
            threads = [threading.Thread(target=lambda: [append_event(target, "wave_finished", wave=index) for index in range(20)]) for _ in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            events = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(events), 120)
            self.assertEqual(sorted(int(item["seq"]) for item in events), list(range(1, 121)))


if __name__ == "__main__":
    unittest.main()
