"""Reference-conditioned image editing provider used by explicit workflows."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from PIL import Image

from .rate_limit import serialized_request


def _endpoint() -> str:
    explicit = os.getenv("RFS_IMAGE_EDIT_URL", "").strip()
    if explicit:
        return explicit
    base = os.getenv("API_BASE", "").rstrip("/")
    if not base:
        raise RuntimeError("Image editing requires API_BASE or RFS_IMAGE_EDIT_URL")
    return f"{base}/images/edits"


def _safe_endpoint(value: str) -> str:
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _model(explicit: str | None) -> str:
    value = explicit or os.getenv("RFS_IMAGE_MODEL") or os.getenv("IMAGE_MODEL") or "image-2"
    return "gpt-image-2" if value == "image-2" else value


def _size(source: Path) -> str:
    with Image.open(source) as image:
        ratio = image.width / max(1, image.height)
    if ratio >= 1.25:
        return "1536x1024"
    if ratio <= 0.8:
        return "1024x1536"
    return "1024x1024"


def _write_response(payload: dict[str, Any], output: Path) -> None:
    rows = payload.get("data") if isinstance(payload.get("data"), list) else []
    if not rows:
        raise RuntimeError("Image edit response did not contain data")
    row = rows[0] if isinstance(rows[0], dict) else {}
    output.parent.mkdir(parents=True, exist_ok=True)
    if row.get("b64_json"):
        output.write_bytes(base64.b64decode(row["b64_json"]))
    elif row.get("url"):
        response = requests.get(str(row["url"]), timeout=180)
        response.raise_for_status()
        output.write_bytes(response.content)
    else:
        raise RuntimeError("Image edit response did not contain b64_json or url")
    with Image.open(output) as image:
        image.verify()


def edit_reference_image(
    source: str | Path,
    prompt: str,
    output: str | Path,
    *,
    model: str | None = None,
    retries: int = 2,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Edit one source image and fail explicitly when the provider returns no valid image."""
    source_path = Path(source)
    output_path = Path(output)
    api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Image editing requires API_KEY/GEMINI_API_KEY")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    endpoint = _endpoint()
    resolved_model = _model(model)
    request_timeout = max(30, int(timeout or os.getenv("RFS_IMAGE2_TIMEOUT", "180")))
    attempts = max(0, min(5, int(retries)))
    try:
        def send():
            mime = "image/png" if source_path.suffix.casefold() == ".png" else "image/jpeg"
            with source_path.open("rb") as handle:
                return requests.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data={"model": resolved_model, "prompt": prompt, "n": "1", "size": _size(source_path)},
                    files={"image": (source_path.name, handle, mime)},
                    timeout=request_timeout,
                )

        response, used_attempts = serialized_request(send, retries=attempts)
        response.raise_for_status()
        _write_response(response.json(), output_path)
        return {"model": resolved_model, "endpoint": _safe_endpoint(endpoint), "attempts": used_attempts}
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Image edit failed after {attempts + 1} attempts: {exc}") from exc
