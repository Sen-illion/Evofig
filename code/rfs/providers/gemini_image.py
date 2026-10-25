"""Gemini native image generation for local academic icons.

The OpenAI-compatible ``/images/generations`` route stays in the icon module.
This module is the separate ``generateContent`` call used when the icon model
is a Gemini image model, such as ``gemini-2.5-flash-image``.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


def is_gemini_image_model(model: str) -> bool:
    """True when ``model`` is a Gemini model that returns an image."""

    name = str(model or "").strip().casefold()
    return name.startswith("gemini-") and "image" in name


def gemini_image_url(model: str) -> str:
    """Build the generateContent URL for ``model`` from the configured host."""

    configured = os.getenv("GEMINI_GEN_IMG_URL", "").strip() or os.getenv("API_BASE", "").strip()
    parsed = urlparse(configured)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError("Gemini image generation requires GEMINI_GEN_IMG_URL or API_BASE")
    return f"{parsed.scheme}://{parsed.netloc}/v1beta/models/{model}:generateContent"


def _usage(payload: dict[str, Any]) -> dict[str, int]:
    meta = payload.get("usageMetadata") if isinstance(payload.get("usageMetadata"), dict) else {}
    prompt = int(meta.get("promptTokenCount") or 0)
    completion = int(meta.get("candidatesTokenCount") or 0)
    total = int(meta.get("totalTokenCount") or (prompt + completion))
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total, "image_count": 1}


def _write_image(payload: dict[str, Any], path: Path) -> None:
    for candidate in payload.get("candidates") or []:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        for part in (content or {}).get("parts") or []:
            if not isinstance(part, dict):
                continue
            blob = part.get("inlineData") or part.get("inline_data") or {}
            data = blob.get("data") if isinstance(blob, dict) else None
            if data:
                path.write_bytes(base64.b64decode(data))
                return
    raise RuntimeError("Gemini image response did not contain inline image data")


def generate_gemini_image(path: Path, prompt: str, model: str) -> dict[str, Any]:
    """Write one icon to ``path`` and return the model plus provider usage."""

    key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise RuntimeError("Gemini image generation requires GEMINI_API_KEY or API_KEY")
    timeout = max(30, int(os.getenv("RFS_IMAGE2_TIMEOUT", "180")))
    max_attempts = max(1, int(os.getenv("RFS_IMAGE2_RETRIES", os.getenv("RFS_IMAGE2_RATE_LIMIT_RETRIES", "5"))) + 1)
    retryable = {408, 425, 429, 500, 502, 503, 504}
    response = None
    attempts = 0
    last_error: Exception | None = None
    for attempts in range(1, max_attempts + 1):
        try:
            response = requests.post(
                gemini_image_url(model),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"responseModalities": ["IMAGE"]},
                },
                timeout=timeout,
            )
            if response.status_code not in retryable or attempts >= max_attempts:
                break
            delay = min(30.0, 2.0 ** (attempts - 1))
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = exc
            if attempts >= max_attempts:
                raise
            delay = min(30.0, 2.0 ** (attempts - 1))
        time.sleep(max(0.0, delay))
    if response is None:
        raise last_error or RuntimeError("Gemini image request failed without a response")
    if response.status_code == 400:
        detail = " ".join(str(response.text or "").split())[:300]
        from ..generation.icon_assets import IconProviderRefusal

        raise IconProviderRefusal(f"image provider refused this prompt (HTTP 400): {detail}")
    response.raise_for_status()
    payload = response.json()
    _write_image(payload, path)
    return {"model": model, "api_requests_attempted": attempts, "usage": _usage(payload)}
