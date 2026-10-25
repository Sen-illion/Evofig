"""Canonical VLM provider client implementation."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from rfs.utils import sanitize_surrogates


DEFAULT_VLM_MODEL = "gemini-3-pro-preview-thinking"


TRAILING_COMMA = re.compile(r",(\s*[}\]])")

#: The only escapes JSON itself allows inside a string.
JSON_ESCAPES = "\"\\/bfnrtu"


def _escape_invalid_escapes(text: str) -> str:
    """Double the backslash of an escape JSON never allowed.

    A model that writes a Windows path, a LaTeX macro or a regex inside a JSON
    string produces ``"<PATH>"`` or ``"\\alpha"``, and the whole
    answer stops parsing: ``Invalid \\escape: line 55 column 51``. The backslash
    was meant literally, so making it literal is the repair. ``\\u`` counts as
    invalid unless four hex digits follow it, which is what JSON requires.
    """

    out: list[str] = []
    inside_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            inside_string = not inside_string
            out.append(char)
            index += 1
            continue
        if inside_string and char == "\\":
            following = text[index + 1] if index + 1 < len(text) else ""
            valid = following in JSON_ESCAPES
            if following == "u":
                valid = all(digit in "0123456789abcdefABCDEF" for digit in text[index + 2:index + 6]) and len(text[index + 2:index + 6]) == 4
            if valid:
                out.append(text[index:index + 2])
                index += 2
                continue
            out.append("\\\\")
            index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _drop_newlines_inside_strings(text: str) -> str:
    """Remove the line breaks a wrapping gateway leaves inside JSON strings."""

    out: list[str] = []
    inside_string = False
    escaped = False
    for char in text:
        if inside_string:
            if char in "\r\n":
                continue
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                inside_string = False
        elif char == '"':
            inside_string = True
        out.append(char)
    return "".join(out)


def _drop_all_newlines(text: str) -> str:
    """Join a wrapped answer back into one line; JSON whitespace is optional."""

    return text.replace("\r", "").replace("\n", "")


def _comma_variants(text: str) -> list[str]:
    """Return the punctuation repairs worth trying for one candidate text."""

    variants = [TRAILING_COMMA.sub(r"\1", text)]
    try:
        json.loads(text, strict=False)
        return variants
    except json.JSONDecodeError as nested:
        if "Expecting ',' delimiter" not in str(nested.msg):
            return variants
        position = max(0, min(int(nested.pos), len(text)))
        inserted = f"{text[:position]},{text[position:]}"
        variants.extend([inserted, TRAILING_COMMA.sub(r"\1", inserted)])
    return variants


def repair_json_syntax(candidate: str, error: json.JSONDecodeError) -> str | None:
    """Repair a punctuation defect, or return ``None`` and leave it alone.

    Three defects account for nearly every unparseable answer the plans produce:
    a comma left in front of a closing bracket, a comma the model forgot between
    two entries, and an answer a gateway handed back wrapped at a fixed column,
    which leaves line breaks inside identifiers and inside numbers. All three are
    punctuation rather than content, so the repair is verified by parsing it
    before it is used, and a wrong guess costs nothing.

    The fourth defect is the opposite of punctuation: a backslash the model
    meant literally, in a Windows path or a LaTeX macro, which JSON reads as an
    invalid escape and rejects. Doubling it restores the character the model
    wrote, and the same parse check keeps a real escape intact.

    The wrapped answer is the expensive one: re-sending that request reproduces
    the same wrapping, so an unread plan used to cost a whole stage. Joining the
    lines back together is exact, because a line break is either optional
    whitespace between tokens or something JSON never allowed inside a string.
    Unwrapping is tried before the punctuation repairs so that a line break in
    the middle of an identifier is read as the wrapping artefact it is; a reader
    that tolerated it would carry the break into a node id or a label.

    An answer that was cut off is deliberately *not* repaired: closing its
    brackets would silently deliver half a plan, so it stays unparseable and the
    caller asks for it again.
    """

    bases = [_drop_newlines_inside_strings(candidate), _drop_all_newlines(candidate), candidate]
    bases = bases + [_escape_invalid_escapes(base) for base in bases]
    attempts: list[str] = []
    for base in bases:
        attempts.append(base)
        attempts.extend(_comma_variants(base))
    for attempt in attempts:
        if attempt == candidate:
            continue
        try:
            json.loads(attempt, strict=False)
        except json.JSONDecodeError:
            continue
        return attempt
    return None


def repair_prompt(prompt: str, broken_answer: str, error: str) -> str:
    """Ask for the same answer again after showing the parser's complaint.

    The defect is the model's own syntax, and these calls run at temperature 0.1,
    so re-sending the identical request tends to reproduce the same broken text -
    a second paid attempt that buys nothing. Handing the answer back with the
    error instead gives the model something to act on. The images are left out:
    they carry the paper's evidence, not the JSON syntax, and repeating them is
    the expensive part of the call.
    """

    return (
        f"{prompt}\n\n"
        "YOUR PREVIOUS ANSWER WAS NOT VALID JSON.\n"
        f"PARSER ERROR: {error}\n"
        "PREVIOUS ANSWER:\n"
        f"{broken_answer[:20000]}\n\n"
        "Return the same object again as one complete JSON object and nothing else. "
        "Keep every value the previous answer already carried and finish anything that is missing."
    )


def extract_json(text: str) -> dict:
    """Parse one provider answer and clean characters UTF-8 cannot encode.

    A model occasionally writes half of an astral character (a truncated
    mathematical symbol). The parsed answer is cleaned so no later artifact
    write or hash can fail on it.
    """
    return sanitize_surrogates(_extract_json_answer(text))


def _extract_json_answer(text: str) -> dict:
    cleaned = str(text or "").strip().replace("```json", "```")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as strict_error:
            # A model occasionally writes a real newline or tab inside a
            # string.  That is a control character to a strict reader but the
            # intent is unambiguous, and rejecting the whole answer over it
            # would discard an otherwise usable plan.
            try:
                return json.loads(candidate, strict=False)
            except json.JSONDecodeError as error:
                repaired = repair_json_syntax(candidate, error)
                if repaired is None:
                    raise error from None
                return json.loads(repaired, strict=False)


def vlm_credentials_available() -> bool:
    return bool(os.getenv("API_BASE", "").strip()) and bool((os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip())


def resolve_vlm_model(*env_names: str, explicit_model: str | None = None) -> str:
    if explicit_model:
        return explicit_model
    for name in env_names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return os.getenv("MODEL_VLM", "").strip() or DEFAULT_VLM_MODEL


def resolve_vlm_timeout(explicit: int | None = None) -> int:
    """A VLM request timeout in seconds.

    The CLI passes ``None`` when the operator did not name one; the provider
    must never hand that straight to ``float``/``int``. Falls back to
    ``RFS_VLM_TIMEOUT`` and then to the historical 180 seconds.
    """
    if explicit is not None:
        return int(explicit)
    raw = os.getenv("RFS_VLM_TIMEOUT", "").strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 180


def resolve_vlm_retries(explicit: int | None = None) -> int:
    """A VLM retry count, falling back to ``RFS_VLM_RETRIES`` and then 1."""
    if explicit is not None:
        return int(explicit)
    raw = os.getenv("RFS_VLM_RETRIES", "").strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 1


#: Statuses the provider asks us to repeat. Everything else it rejected on
#: purpose, and sending the same request again only spends the budget.
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429})
#: Finish reasons that mean asking again cannot produce JSON. An empty decode and a
#: token-cap truncation used to look like ``JSONDecodeError: Expecting value``, so the
#: client re-asked until the timeout budget was gone.
NON_RETRYABLE_COMPLETIONS = frozenset({"empty_completion", "truncated_completion"})
#: The status a provider uses to say "slow down". A caller above the retry loop can tell it
#: apart from a real rejection and wait far longer than this loop would on its own.
RATE_LIMIT_STATUS = 429
RATE_LIMIT_MARKER = f"http status {RATE_LIMIT_STATUS}"


def _retryable_status(status: int) -> bool:
    return int(status) >= 500 or int(status) in RETRYABLE_STATUSES


def rate_limited_error(exc: BaseException | None) -> bool:
    """Whether a provider failure was the provider asking us to slow down."""

    if exc is None:
        return False
    text = str(exc).casefold()
    if RATE_LIMIT_MARKER in text or "429 client error" in text or "too many requests" in text:
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(status) == RATE_LIMIT_STATUS
    except (TypeError, ValueError):
        return False


def rate_limit_sleep_seconds() -> float:
    """How long to sit out a 429 before retrying; 0 keeps the built-in short backoff."""

    try:
        return max(0.0, float(os.getenv("RFS_VLM_RATE_LIMIT_SLEEP_SECONDS", "0")))
    except (TypeError, ValueError):
        return 0.0


def _error_category(exc: Exception) -> str:
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    if "empty completion" in message:
        return "empty_completion"
    if "token cap" in message:
        return "truncated_completion"
    if "timeout" in name or "timeout" in message:
        return "timeout"
    if "ssl" in name or "ssl" in message or "eof" in message:
        return "tls"
    if "json" in name or "json" in message:
        return "invalid_json"
    if "connection" in name or "connection" in message:
        return "connection"
    if "http" in name or "status" in message:
        return "http"
    return "provider"


def _usage_record(payload: Any) -> dict[str, Any] | None:
    """Return provider-reported token usage when the response carries it."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return None
    record: dict[str, Any] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            record[name] = int(usage[name])
        except (KeyError, TypeError, ValueError):
            record[name] = None
    if all(value is None for value in record.values()):
        return None
    return record


def _record_usage(call_metadata: dict[str, Any], usage: dict[str, Any] | None) -> None:
    """Attach token usage to the call metadata, falling back to the prompt estimate.

    The fallback counts prompt text only, so image tokens stay visible through
    ``image_count`` and ``image_bytes`` instead. Retries are reported by
    ``attempts`` because a failed attempt never returns usage.
    """
    if usage:
        call_metadata.update({"usage": usage, "token_source": "provider", "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens")})
        return
    call_metadata.update({"usage": None, "token_source": "estimated", "prompt_tokens": call_metadata.get("estimated_tokens"), "completion_tokens": None, "total_tokens": None})


def call_vlm_json(
    prompt: str,
    image_paths: list[str | Path],
    model: str | None = None,
    timeout: int = 180,
    retries: int = 1,
    call_metadata: dict[str, Any] | None = None,
    deadline_at: float | None = None,
) -> dict:
    timeout = resolve_vlm_timeout(timeout)
    retries = resolve_vlm_retries(retries)
    api_base = os.getenv("API_BASE", "").rstrip("/")
    api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_base or not api_key:
        raise RuntimeError("VLM call requires API_BASE and API_KEY/GEMINI_API_KEY environment variables")

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            continue
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    payload: dict[str, Any] = {"model": model or resolve_vlm_model(), "temperature": 0.1}
    answer_content = content
    started = time.monotonic()
    last_error: Exception | None = None
    failures: list[dict[str, Any]] = []
    attempts = max(1, int(retries) + 1)
    if call_metadata is not None:
        prompt_chars = len(prompt or "")
        image_bytes = sum(Path(path).stat().st_size for path in image_paths if Path(path).exists())
        call_metadata.update({"attempts": 0, "retries_allowed": max(0, int(retries)), "retries_used": 0, "success": False, "elapsed_seconds": 0.0, "failure_categories": [], "prompt_chars": prompt_chars, "estimated_tokens": max(1, round(prompt_chars / 4)), "image_count": len(content) - 1, "image_bytes": image_bytes})
    attempts_made = 0
    for attempt in range(attempts):
        rejected_status: int | None = None
        payload["messages"] = [{"role": "user", "content": answer_content}]
        request_timeout = max(0.1, float(timeout))
        if deadline_at is not None:
            remaining = float(deadline_at) - time.monotonic()
            if remaining <= 0.1:
                last_error = TimeoutError("VLM deadline budget exhausted")
                failures.append({"attempt": attempts_made, "category": "timeout"})
                break
            request_timeout = min(request_timeout, remaining)
        attempts_made += 1
        if call_metadata is not None:
            call_metadata["attempts"] = attempts_made
        try:
            response = requests.post(f"{api_base}/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, data=json.dumps(payload), timeout=request_timeout)
            status = int(getattr(response, "status_code", 200) or 200)
            if status >= 400:
                # Keep what the provider said and stop paying: a rejected request
                # is the caller's own, so repeating it just burns the budget.
                rejected_status = status
                detail = " ".join(str(getattr(response, "text", "") or "").split())[:400]
                raise RuntimeError(f"provider rejected the request with HTTP status {rejected_status}: {detail}")
            body = response.json()
            usage = _usage_record(body)
            choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
            message = choice.get("message") if isinstance(choice, dict) else {}
            answer = str((message or {}).get("content") or "")
            finish_reason = str((choice or {}).get("finish_reason") or "stop")
            if not answer.strip() or finish_reason == "empty":
                raise RuntimeError("judge returned an empty completion")
            try:
                result = extract_json(answer)
            except json.JSONDecodeError as exc:
                if finish_reason == "length":
                    raise RuntimeError(f"judge hit the token cap with unparseable text: {exc}") from exc
                # A malformed answer is the model's own syntax, so the retry
                # hands that answer back with the parser's complaint instead of
                # re-sending the request that produced it.
                answer_content = [{"type": "text", "text": repair_prompt(prompt, answer, str(exc))}]
                raise
            if call_metadata is not None:
                call_metadata.update({"success": True, "retries_used": attempt, "elapsed_seconds": round(time.monotonic() - started, 3), "failure_categories": list(dict.fromkeys(item["category"] for item in failures))})
                _record_usage(call_metadata, usage)
            return result
        except Exception as exc:
            last_error = exc
            category = _error_category(exc)
            failures.append({"attempt": attempts_made, "category": category})
            if category in NON_RETRYABLE_COMPLETIONS:
                break
            if rejected_status is not None and not _retryable_status(rejected_status):
                break
            if attempt < attempts - 1:
                delay = min(2.0, 0.75 * (2 ** attempt))
                if rejected_status == RATE_LIMIT_STATUS:
                    # A shared quota answers every lane at once, so the short built-in backoff
                    # only spends the retries; the caller can ask us to sit out longer instead.
                    delay = max(delay, rate_limit_sleep_seconds())
                if deadline_at is not None and float(deadline_at) - time.monotonic() <= delay + 0.1:
                    break
                time.sleep(delay)
    if call_metadata is not None:
        call_metadata.update({"success": False, "retries_used": max(0, attempts_made - 1), "elapsed_seconds": round(time.monotonic() - started, 3), "failure_categories": list(dict.fromkeys(item["category"] for item in failures)), "deadline_reached": bool(deadline_at is not None and time.monotonic() >= float(deadline_at))})
        _record_usage(call_metadata, None)
    raise last_error or RuntimeError("VLM call failed")
