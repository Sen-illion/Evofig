from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_surrogates(value: Any) -> Any:
    """Replace lone UTF-16 surrogates so text can always be encoded as UTF-8.

    A model answer can carry half of an astral character (for example a
    mathematical symbol written as a high surrogate without its pair). Such a
    string survives ``json.loads`` but raises on every utf-8 encode, which used
    to abort a whole paid stage the moment an artifact was written or hashed.
    Only lone surrogates change; every other character - including complete
    astral characters - passes through untouched. The function walks dicts,
    lists and tuples so a parsed provider answer can be cleaned in one call.
    """
    if isinstance(value, str):
        if not any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            return value
        return "".join("\ufffd" if 0xD800 <= ord(char) <= 0xDFFF else char for char in value)
    if isinstance(value, list):
        return [sanitize_surrogates(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_surrogates(item) for item in value)
    if isinstance(value, dict):
        return {sanitize_surrogates(key): sanitize_surrogates(item) for key, item in value.items()}
    return value


def write_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(sanitize_surrogates(data), indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


def pct_to_inches(bbox: dict[str, float], width_in: float, height_in: float) -> tuple[float, float, float, float]:
    return (
        float(bbox["x"]) * width_in,
        float(bbox["y"]) * height_in,
        float(bbox["w"]) * width_in,
        float(bbox["h"]) * height_in,
    )


def ratio_string(width: float, height: float) -> str:
    if height <= 0:
        return "1.000:1.000"
    value = float(width) / float(height)
    return f"{value:.3f}:1.000"


def short_path(path: str | Path) -> str:
    return str(Path(path))


def env_present(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "***"
    return value[:4] + "***" + value[-4:]
