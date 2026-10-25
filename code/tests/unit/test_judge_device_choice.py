"""The single-card Judge load picks the card that is actually free.

``device_map={"": 0}`` sent the advisory seat to GPU 0 whatever else was running there. On the
experiment server another session held 29 GiB of that card, and 8 of the 19 repair rounds in
the SciFlow probe answered with ``OutOfMemoryError`` -- a failed judge answer, which the round
escalated into a structural re-plan that redrew every icon.
"""

import unittest
from types import SimpleNamespace

from rfs.judge_training.training import freest_visible_gpu


def _torch(free_by_index, *, available=True, raises=False):
    def mem_get_info(index):
        if raises:
            raise RuntimeError("the driver will not say")
        return (free_by_index[index], 0)

    cuda = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: len(free_by_index),
        mem_get_info=mem_get_info,
    )
    return SimpleNamespace(cuda=cuda)


class FreestVisibleGpuTest(unittest.TestCase):
    def test_the_emptiest_card_wins(self):
        self.assertEqual(freest_visible_gpu(_torch({0: 1_000, 1: 40_000, 2: 5_000})), 1)

    def test_a_machine_without_cuda_stays_on_zero(self):
        self.assertEqual(freest_visible_gpu(_torch({0: 1_000}, available=False)), 0)

    def test_a_single_card_stays_on_zero(self):
        self.assertEqual(freest_visible_gpu(_torch({0: 1_000})), 0)

    def test_an_unreadable_memory_report_falls_back_to_zero(self):
        self.assertEqual(freest_visible_gpu(_torch({0: 1_000, 1: 40_000}, raises=True)), 0)

    def test_a_torch_without_cuda_falls_back_to_zero(self):
        self.assertEqual(freest_visible_gpu(SimpleNamespace()), 0)


if __name__ == "__main__":
    unittest.main()
