"""Cross-process locks and single-write appends for the append-only Memory stores.

A wave moves several papers at once and every paper writes the same experience library, so
"read the rows, then append" is a race: two runs read the same N rows, both append, and the
manifest they each rewrite says N+1 while the file holds N+2. Duplicate detection races the
same way, so the same id can be written twice.

The fix is one exclusive lock per library file, held across the whole read-check-append-write
sequence, plus the two small habits that make an append safe on its own: one ``write`` per row,
and a manifest that lands by rename instead of in place. The lock is a byte-range lock on a
sibling ``.lock`` file, so it also works when the library does not exist yet, and it is held by
the process rather than by the thread -- a process-local lock is taken first so two threads of
one run serialise too, and are not left depending on the operating system's view of a second
handle on the same file.
"""

from __future__ import annotations

import errno
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

DEFAULT_LOCK_TIMEOUT = 60.0
LOCK_POLL_SECONDS = 0.02

#: One lock per lock file, so two threads of one run never both hold the byte-range lock.
_thread_locks: dict[str, threading.Lock] = {}
_thread_locks_guard = threading.Lock()
#: The locks one thread already holds, so a writer may call another writer without deadlocking.
_local = threading.local()


class FileLockTimeout(RuntimeError):
    """A lock that was still held when the caller's patience ran out."""

    def __init__(self, target: str, timeout: float) -> None:
        super().__init__(f"the lock for {target} was still held after {timeout:g}s")
        self.target = target
        self.timeout = timeout


def lock_path(target: str | Path) -> Path:
    """The sibling file whose byte range stands for ``target``."""

    return Path(f"{target}.lock")


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _thread_locks_guard:
        lock = _thread_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _thread_locks[key] = lock
        return lock


class _ByteRangeLock:
    """One held byte-range lock, released by whichever path acquired it."""

    def __init__(self, handle: Any) -> None:
        self.handle = handle

    def acquire(self) -> bool:
        self.handle.seek(0)
        if msvcrt is not None:  # pragma: no cover - exercised on Windows
            if self.handle.seek(0, os.SEEK_END) == 0:
                self.handle.write(b"\0")
                self.handle.flush()
            self.handle.seek(0)
            try:
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                return False
            return True
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        return True

    def release(self) -> None:
        try:
            if msvcrt is not None:  # pragma: no cover - exercised on Windows
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                return
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        except OSError:  # a lock that already went away with its handle
            pass


@contextmanager
def exclusive_lock(target: str | Path, *, timeout: float = DEFAULT_LOCK_TIMEOUT, poll: float = LOCK_POLL_SECONDS) -> Iterator[None]:
    """Hold the exclusive lock for ``target`` for the duration of the block.

    Re-entrant for one thread: a caller that already holds this library's lock can call another
    writer that takes it again without deadlocking itself.
    """

    path = lock_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    held: set[str] = getattr(_local, "held", None) or set()
    _local.held = held
    key = str(path)
    if key in held:
        yield
        return
    thread_lock = _thread_lock(path)
    if not thread_lock.acquire(timeout=max(0.0, float(timeout))):
        raise FileLockTimeout(key, float(timeout))
    try:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with path.open("a+b") as handle:
            lock = _ByteRangeLock(handle)
            while True:
                if lock.acquire():
                    break
                if time.monotonic() >= deadline:
                    raise FileLockTimeout(key, float(timeout))
                time.sleep(max(0.001, float(poll)))
            held.add(key)
            try:
                yield
            finally:
                held.discard(key)
                lock.release()
    finally:
        thread_lock.release()


def append_jsonl(path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append one row as a single locked write, and return what was written."""

    target = Path(path)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with exclusive_lock(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
    return record


def append_jsonl_sequenced(path: str | Path, build: Callable[[dict[str, Any] | None, int], dict[str, Any]]) -> dict[str, Any]:
    """Append one row whose record is built from the row before it, under the same lock.

    Event logs number their rows, and two writers that count the file at the same time both
    write the same number. Reading the previous row inside the lock is what makes the number
    mean "the row before this one"; the readable row count comes along for a log whose last row
    carries no number of its own.
    """

    target = Path(path)
    with exclusive_lock(target):
        previous: dict[str, Any] | None = None
        count = 0
        if target.exists():
            for line in target.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    previous = value
                    count += 1
        record = build(previous, count)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
    return record


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a small JSON file that readers never see half-finished."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)


__all__ = [
    "DEFAULT_LOCK_TIMEOUT",
    "FileLockTimeout",
    "append_jsonl",
    "append_jsonl_sequenced",
    "exclusive_lock",
    "lock_path",
    "write_json_atomic",
]
