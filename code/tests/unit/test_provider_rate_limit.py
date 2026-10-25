"""Tests for serialized provider scheduling and retry backoff."""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from rfs.providers.rate_limit import rate_limit_snapshot, reset_rate_limit_state, serialized_request


class _Response:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class ProviderRateLimitTests(unittest.TestCase):
    def setUp(self):
        reset_rate_limit_state()

    def test_concurrent_image_requests_are_serialized(self):
        active = 0
        maximum = 0
        guard = threading.Lock()

        def send():
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with guard:
                active -= 1
            return _Response(200)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _index: serialized_request(send, retries=0, min_interval=0), range(4)))
        self.assertEqual(maximum, 1)
        self.assertEqual([attempts for _response, attempts in results], [1, 1, 1, 1])

    def test_retry_after_and_exponential_backoff_use_larger_delay(self):
        responses = iter([_Response(429, {"Retry-After": "7"}), _Response(200)])
        with patch("rfs.providers.rate_limit.time.sleep") as sleep:
            response, attempts = serialized_request(
                lambda: next(responses),
                retries=2,
                min_interval=0,
                base_delay=5,
                max_delay=60,
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(attempts, 2)
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 7.0, places=3)

    def test_exhausted_retries_return_last_response_for_caller_validation(self):
        response, attempts = serialized_request(
            lambda: _Response(503),
            retries=1,
            min_interval=0,
            base_delay=0,
            max_delay=0,
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(attempts, 2)

    def test_bounded_lane_allows_two_in_flight(self):
        active = 0
        maximum = 0
        guard = threading.Lock()

        def send():
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            return _Response(200)

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(
                pool.map(
                    lambda _index: serialized_request(
                        send,
                        lane="two-wide",
                        retries=0,
                        min_interval=0,
                        max_in_flight=2,
                    ),
                    range(6),
                )
            )
        self.assertEqual(maximum, 2)

    def test_rate_limit_reduces_effective_concurrency(self):
        serialized_request(
            lambda: _Response(429),
            lane="adaptive",
            retries=0,
            min_interval=0,
            max_in_flight=4,
        )
        state = rate_limit_snapshot("adaptive")["adaptive"]
        self.assertEqual(state["max_in_flight"], 4)
        self.assertEqual(state["effective_max"], 3)
        self.assertEqual(state["rate_limited"], 1)

    def test_configured_exception_is_retried_once(self):
        attempts = 0

        def send():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TimeoutError("slow")
            return _Response(200)

        response, count = serialized_request(
            send,
            lane="exceptions",
            retries=1,
            min_interval=0,
            base_delay=0,
            retry_exceptions=(TimeoutError,),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
