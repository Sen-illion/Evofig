"""One process keeps one copy of a Judge checkpoint, and one generation at a time.

Both local seats read the same weights. Seating them twice used to load the model twice, which on
a shared GPU is what turned a measured round into an out-of-memory round: the round could not be
scored, so nothing could be learned from it. These tests pin the sharing, the serialization and
the reported load failure, not the quality of any particular checkpoint.
"""

import json
import threading
import time
import unittest
from unittest import mock

from rfs.evaluation import direct_judges
from rfs.evaluation.direct_judges import (
    CheckpointAdvisoryJudge,
    ChoiceAcceptanceJudge,
    DirectJudgeInput,
    generate_local_judge,
    load_local_judge_backend,
)


class _FakeBackend:
    """A backend that records what it was asked and answers with a legal choice item."""

    instances = 0

    def __init__(self, config):
        type(self).instances += 1
        self.config = config
        self.records = []
        self.budgets = []

    def load(self, checkpoint):
        self.checkpoint = checkpoint

    def generate(self, record, max_new_tokens=None):
        self.records.append(record)
        self.budgets.append(max_new_tokens)
        return json.dumps({"preferred_candidate": "B", "defect": "overlap"})


class _BrokenBackend(_FakeBackend):
    """A backend whose weights cannot be read, standing in for a missing checkpoint."""

    def load(self, checkpoint):
        raise RuntimeError("no weights here")


def _stub_backend(backend_cls):
    """Swap the loader the seats import, leaving every other name in that module in place."""

    return mock.patch("rfs.judge_training.training.TransformersVlmBackend", backend_cls)


def _input():
    return DirectJudgeInput(method={"title": "paper"}, layout={"nodes": []}, icons={}, program={}, preview=None)


class SharedJudgeBackendTests(unittest.TestCase):
    def setUp(self):
        direct_judges._JUDGE_BACKEND_CACHE.clear()
        _FakeBackend.instances = 0

    def tearDown(self):
        direct_judges._JUDGE_BACKEND_CACHE.clear()

    def test_the_two_seats_naming_one_checkpoint_hold_one_model(self):
        with _stub_backend(_FakeBackend):
            acceptance = ChoiceAcceptanceJudge(checkpoint="/ckpt/step-1", base_model="/models/base")
            advisory = CheckpointAdvisoryJudge(checkpoint="/ckpt/step-1", base_model="/models/base")
            backend, error = load_local_judge_backend("/ckpt/step-1", "/models/base")

        self.assertEqual("", error)
        self.assertEqual(1, _FakeBackend.instances)
        self.assertIs(backend, acceptance._backend)
        self.assertIsNotNone(advisory._ask_json)

    def test_the_advisory_seat_answers_through_the_shared_copy(self):
        with _stub_backend(_FakeBackend):
            _acceptance = ChoiceAcceptanceJudge(checkpoint="/ckpt/step-1", base_model="/models/base")
            advisory = CheckpointAdvisoryJudge(checkpoint="/ckpt/step-1", base_model="/models/base")
            result = advisory.evaluate(_input())
            backend, _error = load_local_judge_backend("/ckpt/step-1", "/models/base")

        self.assertFalse(result["gate_only"])
        self.assertEqual(1, _FakeBackend.instances)
        self.assertEqual(1, len(backend.records))
        record = backend.records[0]
        self.assertEqual({"images", "messages"}, set(record))
        self.assertTrue(record["messages"][0]["content"][0]["text"].strip())

    def test_a_different_checkpoint_still_gets_its_own_copy(self):
        with _stub_backend(_FakeBackend):
            load_local_judge_backend("/ckpt/step-1", "/models/base")
            other, error = load_local_judge_backend("/ckpt/step-2", "/models/base")

        self.assertEqual("", error)
        self.assertEqual(2, _FakeBackend.instances)
        self.assertEqual("/ckpt/step-2", other.checkpoint)

    def test_a_seat_that_names_no_checkpoint_loads_nothing(self):
        with _stub_backend(_FakeBackend):
            backend, error = load_local_judge_backend("", "/models/base", seat="advisory")

        self.assertIsNone(backend)
        self.assertEqual("no advisory checkpoint was given", error)
        self.assertEqual(0, _FakeBackend.instances)

    def test_a_load_failure_names_the_seat_and_is_not_remembered_as_loaded(self):
        with _stub_backend(_BrokenBackend):
            backend, error = load_local_judge_backend("/ckpt/step-1", "/models/base", seat="advisory")
            again, _second_error = load_local_judge_backend("/ckpt/step-1", "/models/base", seat="advisory")

        self.assertIsNone(backend)
        self.assertIsNone(again)
        self.assertIn("no weights here", error)
        # A load that failed leaves nothing to share, so the next caller tries again rather
        # than being handed a backend that was never built.
        self.assertEqual(2, _BrokenBackend.instances)


class SerializedGenerationTests(unittest.TestCase):
    def test_generation_does_not_overlap_across_threads(self):
        class _Tracer:
            def __init__(self):
                self.active = 0
                self.peak = 0
                self.calls = 0
                self.guard = threading.Lock()

            def generate(self, record, max_new_tokens=None):
                with self.guard:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(0.02)
                with self.guard:
                    self.active -= 1
                    self.calls += 1
                return "{}"

        backend = _Tracer()
        start = threading.Barrier(4)

        def worker():
            start.wait()
            for _ in range(5):
                generate_local_judge(backend, {"messages": []})

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(1, backend.peak)
        self.assertEqual(20, backend.calls)

    def test_the_generation_budget_is_passed_through(self):
        backend = _FakeBackend({})
        generate_local_judge(backend, {"messages": []}, max_new_tokens=64)

        self.assertEqual([64], backend.budgets)


if __name__ == "__main__":
    unittest.main()
