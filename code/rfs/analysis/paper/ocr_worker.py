from __future__ import annotations

import argparse
import time
from pathlib import Path

from ...utils import write_json
from .document import _ocr_page


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated, deadline-safe PDF OCR page worker.")
    parser.add_argument("--paper", required=True)
    parser.add_argument("--page", required=True, type=int)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--lang", default="en_ch")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import fitz

    started = time.monotonic()
    document = fitz.open(str(Path(args.paper).resolve()))
    try:
        blocks, engine, error, diagnostics = _ocr_page(
            document[args.page - 1],
            args.page,
            args.engine,
            args.lang,
            None,
            rapidocr_threads=max(1, int(args.threads)),
        )
    finally:
        document.close()
    write_json(Path(args.out), {
        "blocks": blocks,
        "engine": engine,
        "error": error,
        "diagnostics": diagnostics,
        "elapsed_seconds": round(time.monotonic() - started, 4),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
