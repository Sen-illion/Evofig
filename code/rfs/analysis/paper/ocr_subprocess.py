"""Isolated OCR backend worker for reference and generated images."""
from __future__ import annotations

import argparse
from pathlib import Path

from ...utils import write_json
from .ocr import run_easyocr, run_paddle_ocr, run_rapidocr


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one image OCR backend in an isolated process.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--engine", choices=("rapidocr", "easyocr", "paddle"), required=True)
    parser.add_argument("--lang", default="en_ch")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        if args.engine == "rapidocr":
            records = run_rapidocr(Path(args.image), args.lang, threads=1)
        elif args.engine == "easyocr":
            records = run_easyocr(Path(args.image), args.lang)
        else:
            records = run_paddle_ocr(Path(args.image), args.lang)
        payload = {"engine": args.engine, "records": records, "error": None}
    except Exception as exc:
        payload = {"engine": args.engine, "records": [], "error": str(exc)}
    write_json(Path(args.out), payload)
    return 0 if not payload["error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
