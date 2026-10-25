"""Token-usage probe for the external baseline rows.

The baseline runners record wall-clock seconds and nothing else, so their dollars cannot be
reconstructed after the fact: none of the six ever logged a usage block, and no amount of
re-reading their logs recovers one. This module is imported automatically through
``sitecustomize`` whenever its directory is on ``PYTHONPATH``, wraps the two HTTP exits the
baseline stacks use (``requests`` directly, ``httpx`` underneath the ``openai`` SDK), and
appends one JSON line per intercepted response to the file named by ``RFS_USAGE_PROBE_OUT``.

Two rules keep the probe from becoming the reason a run fails:

* it never raises and never alters a request or a response;
* a call it cannot measure is *counted* as unmeasured rather than silently dropped, so a cost
  table built on it can state its own coverage instead of implying it.

The driver points ``RFS_USAGE_PROBE_OUT`` at a fresh file per paper and folds the log into
``run.json`` through :func:`summarise` once the paper is done.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_INSTALLED: list[str] = []

#: A call to these paths is billed per image, so it carries no token block worth pricing.
_IMAGE_URL_HINTS = ("/images/generations", "/images/edits")

#: Vendors hand back an asset URL that the caller then fetches. That fetch is a file download,
#: not a billable call, so counting it as unmeasured would understate every row's coverage.
_ASSET_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".pdf", ".zip", ".mp4")


def enabled() -> bool:
    """True when a driver asked for the probe by naming an output file."""

    return bool(os.environ.get("RFS_USAGE_PROBE_OUT"))


def _as_json(value: Any) -> dict | None:
    """Best-effort JSON view of a request or response body. Never raises."""

    try:
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8", "replace")
        if isinstance(value, str):
            value = json.loads(value)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _count(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def usage_of(payload: Any) -> dict | None:
    """The token block of one response body, in either vendor spelling. ``None`` when absent."""

    if not isinstance(payload, dict):
        return None
    block = payload.get("usage")
    if not isinstance(block, dict):
        block = payload.get("usageMetadata")
    if not isinstance(block, dict):
        return None
    prompt = _count(block.get("prompt_tokens", block.get("promptTokenCount")))
    completion = _count(block.get("completion_tokens", block.get("candidatesTokenCount")))
    total = _count(block.get("total_tokens", block.get("totalTokenCount")))
    if not (prompt or completion or total):
        return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total or prompt + completion,
    }


def _model_of(request: Any, response: Any) -> str:
    for payload in (request, response):
        if isinstance(payload, dict):
            for key in ("model", "modelVersion"):
                name = payload.get(key)
                if isinstance(name, str) and name.strip():
                    return name.strip()
    return ""


def _is_image(url: str) -> bool:
    lowered = str(url or "").lower()
    return any(hint in lowered for hint in _IMAGE_URL_HINTS)


def _is_asset(url: str) -> bool:
    path = str(url or "").lower().split("?", 1)[0]
    return path.endswith(_ASSET_SUFFIXES)


def record(url: str, request_body: Any = None, response_body: Any = None,
           streamed: bool = False) -> dict:
    """Build one log line. Pure, so the classification rules are testable without HTTP."""

    request_json = _as_json(request_body)
    response_json = _as_json(response_body)
    line: dict[str, Any] = {"url": str(url or ""), "model": _model_of(request_json, response_json)}
    if _is_image(url):
        line["kind"] = "image"
        return line
    if _is_asset(url):
        line["kind"] = "other"
        return line
    if streamed:
        line["kind"] = "unmeasured"
        line["reason"] = "streamed response"
        return line
    usage = usage_of(response_json)
    if usage is None:
        line["kind"] = "unmeasured"
        line["reason"] = "no usage block in response"
        return line
    line["kind"] = "text"
    line.update(usage)
    return line


def _append(line: dict) -> None:
    path = os.environ.get("RFS_USAGE_PROBE_OUT")
    if not path:
        return
    try:
        with _LOCK:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        pass


def note(url: str, request_body: Any = None, response_body: Any = None,
         streamed: bool = False) -> None:
    """Log one intercepted response. Never raises, whatever the caller passed."""

    if not enabled():
        return
    try:
        _append(record(url, request_body, response_body, streamed))
    except Exception:
        pass


def _wrap_requests() -> bool:
    try:
        import requests
    except Exception:
        return False
    adapter = getattr(getattr(requests, "adapters", None), "HTTPAdapter", None)
    if adapter is None:
        return False
    if getattr(adapter.send, "_rfs_usage_probe", False):
        return True
    original = adapter.send

    def send(self, request, **kwargs):
        response = original(self, request, **kwargs)
        try:
            streamed = bool(kwargs.get("stream"))
            note(getattr(request, "url", ""), getattr(request, "body", None),
                 None if streamed else getattr(response, "content", None), streamed)
        except Exception:
            pass
        return response

    send._rfs_usage_probe = True
    adapter.send = send
    return True


def _wrap_httpx() -> bool:
    try:
        import httpx
    except Exception:
        return False
    client = getattr(httpx, "Client", None)
    if client is None:
        return False
    if getattr(client.send, "_rfs_usage_probe", False):
        return True
    original = client.send

    def send(self, request, **kwargs):
        response = original(self, request, **kwargs)
        try:
            streamed = bool(kwargs.get("stream"))
            note(str(getattr(request, "url", "")), getattr(request, "content", None),
                 None if streamed else getattr(response, "content", None), streamed)
        except Exception:
            pass
        return response

    send._rfs_usage_probe = True
    client.send = send
    return True


def _wrap_httpx_async() -> bool:
    """The async half of httpx, which is what Paper2Any's ``LLMClient`` actually calls."""

    try:
        import httpx
    except Exception:
        return False
    client = getattr(httpx, "AsyncClient", None)
    if client is None:
        return False
    if getattr(client.send, "_rfs_usage_probe", False):
        return True
    original = client.send

    async def send(self, request, **kwargs):
        response = await original(self, request, **kwargs)
        try:
            streamed = bool(kwargs.get("stream"))
            note(str(getattr(request, "url", "")), getattr(request, "content", None),
                 None if streamed else getattr(response, "content", None), streamed)
        except Exception:
            pass
        return response

    send._rfs_usage_probe = True
    client.send = send
    return True


def install() -> list[str]:
    """Wrap whichever HTTP libraries this interpreter has. Returns the ones wrapped so far."""

    wrappers = (
        ("requests", _wrap_requests),
        ("httpx", _wrap_httpx),
        ("httpx-async", _wrap_httpx_async),
    )
    for name, wrapper in wrappers:
        try:
            wrapped = wrapper()
        except Exception:
            wrapped = False
        if wrapped and name not in _INSTALLED:
            _INSTALLED.append(name)
    return list(_INSTALLED)


def read_log(path: Any) -> list[dict]:
    """Every well-formed line of one paper's probe log; a torn line is skipped, not fatal."""

    rows: list[dict] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            body = json.loads(line)
        except ValueError:
            continue
        if isinstance(body, dict):
            rows.append(body)
    return rows


def _kind_of(row: dict) -> str:
    """What a logged row means, re-read rather than taken on trust.

    A log written before the asset rule existed carries ``unmeasured`` for the vendor's own
    file download, because the build of the probe that wrote it could not tell a download
    from a billable call. Reading the URL back here keeps one paper's coverage from
    depending on which build of the probe happened to be running when it was written, so a
    row already paid for is not bought again just to be labelled correctly.
    """

    kind = str(row.get("kind") or "")
    if kind == "unmeasured" and _is_asset(row.get("url")):
        return "other"
    return kind


def summarise(path: Any) -> dict:
    """Fold one paper's probe log into the numbers the cost table reads.

    ``usage_source`` is ``probe`` only when every intercepted call was measured. One
    unmeasured call downgrades the paper to ``probe_partial``, and an absent or empty log to
    ``missing``, so an incomplete row can never be mistaken for a complete one.
    """

    rows = read_log(path)
    text = [row for row in rows if _kind_of(row) == "text"]
    images = [row for row in rows if _kind_of(row) == "image"]
    unmeasured = [row for row in rows if _kind_of(row) == "unmeasured"]
    if not rows:
        source = "missing"
    elif unmeasured:
        source = "probe_partial"
    else:
        source = "probe"
    return {
        "text_prompt_tokens": sum(_count(row.get("prompt_tokens")) for row in text),
        "text_completion_tokens": sum(_count(row.get("completion_tokens")) for row in text),
        "text_calls": len(text),
        "image_calls": len(images),
        "unmeasured_calls": len(unmeasured),
        "usage_source": source,
    }
