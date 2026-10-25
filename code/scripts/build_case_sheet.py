#!/usr/bin/env python3
"""Lay one case study out as a sheet: every system on the same paper, each cell labelled.

A case study figure invites the reader to compare pictures, so the sheet has to say what each
picture is. Every cell carries the system, the ruler it was produced under, and a one-line
note; a cell whose output came from a different ruler than the rest says so in the cell, not
in a footnote a reader has to hunt for.

The spec is JSON so the layout can be reviewed without re-running anything:

    {"paper_id": ..., "columns": 4, "cells": [{"label": ..., "ruler": ..., "note": ...,
                                               "path": ..., "crop": [x0, y0, x1, y1]}]}

``crop`` is recorded because the author's figure is a human crop off a rendered page: the box
is the claim, and leaving it out of the spec would make it unverifiable.

Text is wrapped by measured pixel width, not by character count, so a note cannot spill into
the neighbouring cell.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PAD = 14
TITLE_FONT_SIZE = 25
LABEL_FONT_SIZE = 20
RULER_FONT_SIZE = 15
NOTE_FONT_SIZE = 16


def font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap_px(draw, text, face, limit):
    """Greedy wrap that respects what the font actually measures, not a character budget."""
    lines, line = [], ""
    for word in text.split():
        candidate = (line + " " + word).strip()
        if draw.textlength(candidate, font=face) <= limit or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--cell-width", type=int)
    parser.add_argument("--cell-height", type=int)
    parser.add_argument("--max-note-lines", type=int, default=5)
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    cells = spec["cells"]
    columns = args.columns or spec.get("columns") or 3
    cell_w = args.cell_width or spec.get("cell_width") or 640
    cell_h = args.cell_height or spec.get("cell_height") or 430
    rows = max(1, math.ceil(len(cells) / columns))
    inner = cell_w - 2 * PAD

    title_face = font(TITLE_FONT_SIZE)
    label_face = font(LABEL_FONT_SIZE)
    ruler_face = font(RULER_FONT_SIZE)
    note_face = font(NOTE_FONT_SIZE)

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    title_lines = wrap_px(probe, spec.get("title", spec["paper_id"]), title_face, columns * cell_w - 2 * PAD)
    subtitle_lines = wrap_px(probe, spec.get("subtitle", ""), note_face, columns * cell_w - 2 * PAD)
    title_bar = 14 + len(title_lines) * (TITLE_FONT_SIZE + 6) + len(subtitle_lines) * (NOTE_FONT_SIZE + 4) + 12

    notes = [
        wrap_px(probe, cell.get("note", ""), note_face, inner - 4)[: args.max_note_lines]
        for cell in cells
    ]
    note_bar = max(1, max(len(lines) for lines in notes)) * (NOTE_FONT_SIZE + 3) + 8
    label_bar = LABEL_FONT_SIZE + RULER_FONT_SIZE + 14
    row_h = label_bar + cell_h + note_bar + PAD
    height = title_bar + rows * row_h

    sheet = Image.new("RGB", (columns * cell_w, height), "white")
    draw = ImageDraw.Draw(sheet)
    y = 14
    for line in title_lines:
        draw.text((PAD, y), line, fill="black", font=title_face)
        y += TITLE_FONT_SIZE + 6
    for line in subtitle_lines:
        draw.text((PAD, y), line, fill="#444444", font=note_face)
        y += NOTE_FONT_SIZE + 4
    draw.line([(0, title_bar - 2), (sheet.width, title_bar - 2)], fill="#999999")

    for index, cell in enumerate(cells):
        col, row = index % columns, index // columns
        x0 = col * cell_w
        y0 = title_bar + row * row_h
        draw.text((x0 + PAD, y0 + 6), cell["label"], fill="black", font=label_face)
        if cell.get("ruler"):
            draw.text((x0 + PAD, y0 + 6 + LABEL_FONT_SIZE + 2), cell["ruler"], fill="#8a3b00", font=ruler_face)
        image = Image.open(cell["path"]).convert("RGB")
        if cell.get("crop"):
            image = image.crop(tuple(cell["crop"]))
        image.thumbnail((inner, cell_h - 6))
        top = y0 + label_bar + 3 + max(0, (cell_h - 6 - image.height) // 2)
        sheet.paste(image, (x0 + PAD + (inner - image.width) // 2, top))
        draw.rectangle([x0 + PAD, y0 + label_bar, x0 + PAD + inner, y0 + label_bar + cell_h], outline="#dddddd")
        note_y = y0 + label_bar + cell_h + 6
        for line in notes[index]:
            draw.text((x0 + PAD, note_y), line, fill="#333333", font=note_face)
            note_y += NOTE_FONT_SIZE + 3
        draw.line([(x0, y0), (x0, y0 + row_h - 2)], fill="#e8e8e8")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print("wrote " + str(out) + " (" + str(sheet.width) + "x" + str(sheet.height) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())