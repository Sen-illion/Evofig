"""Process-wide serialized request scheduling for rate-limited providers."""

from __future__ import annotations

import os
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Callable, TypeVar


ResponseT = TypeVar("ResponseT")


class _Lane:
    def __init__(self, max_in_flight: int) -> None:
        self.condition = threading.Condition()
        self.pace_lock = threading.Lock()
        self.max_in_flight = max(1, max_in_flight)
        self.effective_max = self.max_in_flight
        self.active = 0
        self.last_request_at = 0.0
        self.blocked_until = 0.0
        self.requests = 0
        self.rate_limited = 0
        self.server_errors = 0


_LANES: dict[str, _Lane] = {}
_LANES_LOCK = threading.Lock()


def _lane(name: str, max_in_flight: int = 1) -> _Lane:
    with _LANES_LOCK:
        return _LANES.setdefault(name, _Lane(max_in_flight))


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _retry_after_seconds(response: object) -> float | None:
    headers = getattr(response, "headers", {}) or {}
    value = str(headers.get("Retry-After") or "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value).timestamp()
            return max(0.0, retry_at - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


def serialized_request(
    send: Callable[[], ResponseT],
    *,
    lane: str = "image_generation",
    retries: int = 2,
    min_interval: float | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    max_in_flight: int = 1,
    retry_exceptions: tuple[type[BaseException], ...] = (),
) -> tuple[ResponseT, int]:
    """Run one bounded provider lane with Retry-After-aware backoff.

    A lane bounds logical in-flight requests while a smaller pacing lock only
    schedules request start times. Retry-After updates a shared cooldown and
    429 responses reduce effective concurrency for the rest of the process.
    Defaults can be tuned without code changes through
    ``RFS_IMAGE_MIN_INTERVAL_SECONDS``, ``RFS_IMAGE_RETRY_BASE_SECONDS`` and
    ``RFS_IMAGE_RETRY_MAX_SECONDS``.
    """
    interval = _float_env("RFS_IMAGE_MIN_INTERVAL_SECONDS", 3.0) if min_interval is None else max(0.0, min_interval)
    retry_base = _float_env("RFS_IMAGE_RETRY_BASE_SECONDS", 5.0) if base_delay is None else max(0.0, base_delay)
    retry_max = _float_env("RFS_IMAGE_RETRY_MAX_SECONDS", 60.0) if max_delay is None else max(0.0, max_delay)
    attempts = max(1, int(retries) + 1)
    state = _lane(lane, max_in_flight=max_in_flight)
    with state.condition:
        while state.active >= state.effective_max:
            state.condition.wait()
        state.active += 1
    try:
        response: ResponseT | None = None
        for attempt in range(1, attempts + 1):
            with state.pace_lock:
                wait_for_slot = max(
                    interval - (time.monotonic() - state.last_request_at),
                    state.blocked_until - time.monotonic(),
                )
                if wait_for_slot > 0:
                    time.sleep(wait_for_slot)
                state.last_request_at = time.monotonic()
            try:
                response = send()
            except retry_exceptions:
                with state.condition:
                    state.requests += 1
                if attempt >= attempts:
                    raise
                time.sleep(min(retry_max, retry_base * (2 ** (attempt - 1))))
                continue
            with state.condition:
                state.requests += 1
            status = int(getattr(response, "status_code", 0) or 0)
            if status == 429:
                provider_delay = _retry_after_seconds(response) or 0.0
                with state.condition:
                    state.rate_limited += 1
                    state.effective_max = max(1, state.effective_max - 1)
                    state.blocked_until = max(state.blocked_until, time.monotonic() + provider_delay)
            elif 500 <= status < 600:
                state.server_errors += 1
            if status != 429 and not 500 <= status < 600:
                return response, attempt
            if attempt < attempts:
                provider_delay = _retry_after_seconds(response)
                exponential_delay = min(retry_max, retry_base * (2 ** (attempt - 1)))
                with state.condition:
                    state.blocked_until = max(
                        state.blocked_until,
                        time.monotonic() + max(exponential_delay, provider_delay or 0.0),
                    )
        assert response is not None
        return response, attempts
    finally:
        with state.condition:
            state.active -= 1
            state.condition.notify_all()


def rate_limit_snapshot(prefix: str = "") -> dict[str, dict[str, int | float]]:
    """Return non-sensitive live lane metrics for collection dashboards."""
    with _LANES_LOCK:
        items = list(_LANES.items())
    return {
        name: {
            "max_in_flight": lane.max_in_flight,
            "effective_max": lane.effective_max,
            "active": lane.active,
            "requests": lane.requests,
            "rate_limited": lane.rate_limited,
            "server_errors": lane.server_errors,
        }
        for name, lane in items
        if not prefix or name.startswith(prefix)
    }


def reset_rate_limit_state() -> None:
    """Clear process-wide lanes for deterministic tests."""
    with _LANES_LOCK:
        _LANES.clear()


__all__ = ["rate_limit_snapshot", "reset_rate_limit_state", "serialized_request"]
