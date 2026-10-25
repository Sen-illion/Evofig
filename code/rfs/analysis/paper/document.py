from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable


MOJIBAKE_MARKERS = ("\ufffd", "Ã", "Â", "â€", "锟", "鈥", "銆", "鏅")
SECTION_ALIASES = {
    "abstract": ("abstract", "resumen", "résumé", "zusammenfassung", "resumo", "摘要", "概要", "초록"),
    "introduction": ("introduction", "background", "introducción", "introduccion", "introdução", "introducao", "einleitung", "contexte", "contexto", "antecedentes", "引言", "介绍", "背景", "はじめに", "序論", "서론", "배경"),
    "method": ("method", "methods", "methodology", "approach", "architecture", "system overview", "framework", "método", "metodo", "métodos", "metodos", "metodología", "metodologia", "méthode", "methode", "méthodes", "methodes", "méthodologie", "methodologie", "methoden", "methodik", "enfoque", "approche", "ansatz", "abordagem", "方法", "方法论", "架构", "系统概述", "框架", "手法", "方法論", "アーキテクチャ", "방법", "방법론", "아키텍처"),
    "experiments": ("experiment", "experiments", "evaluation", "results", "experimento", "experimentos", "evaluación", "evaluacion", "resultados", "expérience", "experience", "expériences", "experiences", "évaluation", "experimente", "auswertung", "ergebnisse", "avaliação", "avaliacao", "实验", "评估", "结果", "実験", "評価", "結果", "실험", "평가", "결과"),
    "conclusion": ("conclusion", "conclusions", "discussion", "conclusión", "conclusiones", "discusión", "discusion", "conclusão", "conclusao", "conclusões", "conclusoes", "discussão", "discussao", "schlussfolgerung", "schlussfolgerungen", "fazit", "diskussion", "结论", "讨论", "总结", "結論", "考察", "まとめ", "결론", "논의", "요약"),
}

OCR_PROTECTED_WORDS = {
    "acknowledgements", "architecture", "autoregressive", "bidirectional", "classification", "configuration",
    "convolutional", "deterministic", "differentiable", "downstream", "embedding", "embeddings", "evaluation",
    "experimental", "experiments", "finetuning", "implementation", "information", "initialization", "introduction",
    "methodology", "multiframe", "multimodal", "optimization", "performance", "positional", "pretraining",
    "probabilities", "representation", "representations", "retrieval", "scientific", "segmentation", "transformer",
    "visualization",
}
OCR_SPLIT_ANCHORS = OCR_PROTECTED_WORDS | {
    "and", "decoder", "encoder", "evidence", "for", "from", "generation", "into", "method", "model", "of",
    "output", "retriever", "the", "to", "with",
}
OCR_RENDER_DPI = 84
LATIN_CAPTION_PREFIX = re.compile(
    r"^(?P<prefix>figure|fig\.?|table|figura|abbildung|tabla|tableau|tabelle)"
    r"(?=\s|\d)\s*(?P<identifier>[a-z]?\d+(?:[.\-]\d+)*|[ivxlcdm]+)\b",
    re.IGNORECASE,
)


def _caption_prefix_kind(text: str) -> str | None:
    value = _clean(text)
    match = LATIN_CAPTION_PREFIX.match(value)
    if match:
        prefix = match.group("prefix").casefold()
        return "table" if prefix in {"table", "tabla", "tableau", "tabelle"} else "caption"
    if re.match(r"^(图|圖|表)\s*[a-z0-9一二三四五六七八九十]+", value, re.IGNORECASE):
        return "caption" if value.startswith(("图", "圖")) else "table"
    return None


def _clean(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).replace("\u00ad", "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    previous = None
    while previous != value:
        previous = value
        value = re.sub(r"\b([A-Z])\s+([A-Z]{2,})\b", r"\1\2", value)
    value = re.sub(r"\bANIMAGE\b", "AN IMAGE", value)
    value = re.sub(r"\(\s*([A-Z])\s+([A-Z])\s+([A-Z])\s*\)", r"(\1\2\3)", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _repair_ocr_spacing(text: str, splitter: Callable[[str], list[str]] | None = None) -> tuple[str, int]:
    value = _clean(text)
    repairs = 0
    value, count = re.subn(r"\b(figure|fig\.|table|figura|abbildung|tabla|tableau|tabelle)(?=\d)", r"\1 ", value, flags=re.IGNORECASE)
    repairs += count
    section_terms = (
        "abstract|introduction|background|method|methods|approach|experiments|results|conclusion|discussion|references|appendix|"
        "resumen|introducción|introduccion|método|metodo|métodos|metodos|metodología|metodologia|enfoque|"
        "experimento|experimentos|evaluación|evaluacion|resultados|conclusión|conclusiones|discusión|discusion|"
        "résumé|méthode|méthodes|expérience|expériences|évaluation|résultats|"
        "zusammenfassung|einleitung|methoden|methodik|experimente|ergebnisse|schlussfolgerung|fazit|"
        "resumo|introdução|introducao|avaliação|avaliacao|conclusão|conclusao"
    )
    value, count = re.subn(rf"(?<!\w)(\d+(?:\.\d+)*)(?=(?:{section_terms})\b)", r"\1 ", value, flags=re.IGNORECASE)
    repairs += count
    cjk_section_terms = "\u7cfb\u7edf\u6982\u8ff0|\u65b9\u6cd5\u8bba|\u53c2\u8003\u6587\u732e|\u5b9e\u9a8c|\u7ed3\u8bba|\u8ba8\u8bba|\u5f15\u8a00|\u80cc\u666f|\u65b9\u6cd5|\u67b6\u6784|\u6846\u67b6|\u8bc4\u4f30|\u7ed3\u679c|\u9644\u5f55"
    value, count = re.subn(rf"^(\d+(?:\.\d+)*)(?=(?:{cjk_section_terms})(?:\s|$))", r"\1 ", value)
    repairs += count
    value, count = re.subn(r"(?<=[.,:;])(?=[A-ZÁÉÍÓÚÜÑÀÂÇÈÊËÎÏÔÙÛÜ])", " ", value)
    repairs += count
    if splitter is None:
        try:
            import wordninja

            splitter = wordninja.split
        except Exception:
            splitter = None
    if splitter is None:
        return value, repairs

    def split_token(match: re.Match[str]) -> str:
        nonlocal repairs
        token = match.group(0)
        normalized = token.casefold()
        if normalized in OCR_PROTECTED_WORDS or token.isupper() or (token[1:] != token[1:].lower() and token[1:] != token[1:].upper()):
            return token
        pieces = [str(piece) for piece in splitter(token) if str(piece)]
        lowered = [piece.casefold() for piece in pieces]
        anchor_count = sum(piece in OCR_SPLIT_ANCHORS for piece in lowered)
        if len(pieces) < 2 or any(len(piece) < 2 for piece in pieces) or anchor_count < 2:
            return token
        repairs += len(pieces) - 1
        return " ".join(pieces)

    value = re.sub(r"[A-Za-z]{8,160}", split_token, value)
    return _clean(value), repairs


def _normalized_compare_text(text: str) -> str:
    return "".join(char for char in _clean(text).casefold() if char.isalnum())


def _lexical_units(text: str) -> list[str]:
    clean = _clean(text).casefold()
    units = [f"latin:{value}" for value in re.findall(r"[a-z]{3,}", clean)]
    cjk = "".join(re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", clean))
    if len(cjk) == 1:
        units.append(f"cjk:{cjk}")
    else:
        units.extend(f"cjk:{cjk[index:index + 2]}" for index in range(len(cjk) - 1))
    return units


def _cjk_character_count(text: str) -> int:
    return sum(
        "\u3400" <= char <= "\u9fff"
        or "\u3040" <= char <= "\u30ff"
        or "\uac00" <= char <= "\ud7af"
        for char in str(text or "")
    )


def _token_overlap_agreement(left: str, right: str) -> float | None:
    left_tokens = _lexical_units(left)
    right_tokens = _lexical_units(right)
    if not left_tokens or not right_tokens:
        return None
    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    return round(overlap / max(1, len(left_tokens)), 4)


def _replacement_rate(text: str) -> float:
    value = str(text or "")
    return round(value.count("\ufffd") / max(1, len(value)), 6)


def _mojibake_rate(text: str) -> float:
    value = str(text or "")
    return round(sum(value.count(marker) for marker in MOJIBAKE_MARKERS) / max(1, len(value)), 6)


def _looks_like_typographic_heading(text: str) -> bool:
    value = _clean(text)
    words = re.findall(r"[A-Za-z][A-Za-z0-9+&/-]*", value)
    first_alpha = next((char for char in value if char.isalpha()), "")
    first_char = value[0] if value else ""
    if not 2 <= len(value) <= 100 or not 1 <= len(words) <= 10:
        return False
    if not first_char.isalpha() or (first_char.isascii() and not re.search(r"[a-z]", value)):
        return False
    if re.match(r"^(?:figure|fig\.|table)\b", value, re.IGNORECASE) or re.search(r"[=<>]{1,2}|\bdoi\b|https?://", value, re.IGNORECASE):
        return False
    if value.endswith(":") and len(words) <= 2:
        return False
    if value.endswith((",", ";", "?", "!")) or (value.endswith(".") and len(words) > 5):
        return False
    if first_alpha and first_alpha.lower() != first_alpha.upper() and not first_alpha.isupper():
        return False
    return True


def _block_kind(text: str, width_ratio: float = 0.0) -> str:
    value = _clean(text)
    caption_kind = _caption_prefix_kind(value)
    if caption_kind:
        return caption_kind
    section_aliases = tuple(alias.casefold() for aliases in SECTION_ALIASES.values() for alias in aliases)
    if len(value) <= 100 and any(value.casefold() == alias or value.casefold().startswith((f"{alias} ", f"{alias}:", f"{alias}：")) for alias in section_aliases):
        return "heading"
    if re.match(r"^(abstract|\d+(?:\.\d+)*\s+|[ivx]+[.)]\s+)(.{2,100})$", value, re.IGNORECASE) and len(value) <= 140:
        return "heading"
    if len(value) <= 180 and width_ratio >= 0.55 and not value.endswith((".", ",", ";")):
        return "title"
    if re.search(r"(?:[A-Za-z][A-Za-z0-9_{}^\-]*\s*=\s*[^=\n]{3,}|∑|∫|≤|≥|≈)", value):
        return "formula"
    return "paragraph"


def _column_groups(items: list[dict[str, Any]], page_width: float, max_columns: int = 3) -> list[list[dict[str, Any]]]:
    if len(items) < 4:
        return [items]
    positioned = sorted(items, key=lambda item: float(item["bbox"][0]))
    gaps = [
        (float(positioned[index + 1]["bbox"][0]) - float(positioned[index]["bbox"][0]), index)
        for index in range(len(positioned) - 1)
    ]
    qualified_gaps = [(gap, index) for gap, index in sorted(gaps, reverse=True) if gap >= page_width * 0.12]
    if not qualified_gaps:
        return [items]

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0

    def validated_groups(split_count: int) -> list[list[dict[str, Any]]] | None:
        split_indexes = sorted(index for _, index in qualified_gaps[:split_count])
        groups: list[list[dict[str, Any]]] = []
        start = 0
        for split in split_indexes:
            groups.append(positioned[start:split + 1])
            start = split + 1
        groups.append(positioned[start:])
        if len(groups) > max_columns or any(len(group) < 2 for group in groups):
            return None
        ordered_groups = sorted(groups, key=lambda group: median([float(item["bbox"][0]) for item in group]))
        minimum_vertical_span = page_width * 0.12
        minimum_text = 40
        for group in ordered_groups:
            vertical_span = max(float(item["bbox"][3]) for item in group) - min(float(item["bbox"][1]) for item in group)
            text_count = sum(len(_normalized_compare_text(str(item.get("text") or ""))) for item in group)
            if vertical_span < minimum_vertical_span or text_count < minimum_text:
                return None
        for left, right in zip(ordered_groups, ordered_groups[1:]):
            left_right = median([float(item["bbox"][2]) for item in left])
            right_left = median([float(item["bbox"][0]) for item in right])
            if left_right > right_left + page_width * 0.04:
                return None
        return ordered_groups

    for split_count in range(min(max_columns - 1, len(qualified_gaps)), 0, -1):
        groups = validated_groups(split_count)
        if groups:
            return groups
    return [items]


def _reading_order(blocks: list[dict[str, Any]], page_width: float) -> tuple[list[dict[str, Any]], float]:
    if not blocks:
        return [], 0.0
    usable = [item for item in blocks if _clean(item.get("text", ""))]
    if not usable:
        return [], 0.0
    spanning = [item for item in usable if float(item["bbox"][2]) - float(item["bbox"][0]) >= page_width * 0.56]
    non_spanning = [item for item in usable if item not in spanning]
    detection_items = [
        item
        for item in non_spanning
        if float(item["bbox"][2]) - float(item["bbox"][0]) >= page_width * 0.20
        and len(_normalized_compare_text(str(item.get("text") or ""))) >= 20
    ]
    detected_columns = _column_groups(detection_items, page_width)
    if len(detected_columns) == 1:
        ordered = sorted(usable, key=lambda item: (round(float(item["bbox"][1]), 1), float(item["bbox"][0])))
        confidence = 0.96
        for item in ordered:
            item["column"] = 0
    else:
        def median_x0(column: list[dict[str, Any]]) -> float:
            values = sorted(float(item["bbox"][0]) for item in column)
            middle = len(values) // 2
            return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0

        anchors = [median_x0(column) for column in detected_columns]
        columns = [[] for _ in anchors]
        for item in non_spanning:
            column_index = min(range(len(anchors)), key=lambda index: abs(float(item["bbox"][0]) - anchors[index]))
            columns[column_index].append(item)
        for column_index, column in enumerate(columns):
            for item in column:
                item["column"] = column_index
        for item in spanning:
            item["column"] = -1
        boundaries = sorted(spanning, key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])))
        ordered = []
        cursor = float("-inf")
        for boundary in boundaries + [None]:
            limit = float(boundary["bbox"][1]) if boundary else float("inf")
            segment = [item for item in usable if item not in spanning and cursor <= float(item["bbox"][1]) < limit]
            for column in columns:
                ordered.extend(sorted((item for item in segment if item in column), key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0]))))
            if boundary:
                ordered.append(boundary)
                cursor = float(boundary["bbox"][3])
        seen: set[int] = set()
        ordered = [item for item in ordered if not (id(item) in seen or seen.add(id(item)))]
        confidence = (0.88 if spanning else 0.82) if len(columns) == 2 else (0.84 if spanning else 0.78)
    for index, item in enumerate(ordered, 1):
        item["reading_order"] = index
    return ordered, confidence


def _rotate_blocks_to_page(blocks: list[dict[str, Any]], page: Any) -> list[dict[str, Any]]:
    if not int(getattr(page, "rotation", 0) or 0):
        return blocks
    try:
        import fitz

        matrix = page.rotation_matrix
        rotated = []
        for item in blocks:
            value = dict(item)
            rect = fitz.Rect(*item["bbox"]) * matrix
            value["bbox"] = [round(float(rect.x0), 3), round(float(rect.y0), 3), round(float(rect.x1), 3), round(float(rect.y1), 3)]
            rotated.append(value)
        return rotated
    except Exception:
        return blocks


def _pymupdf_line_blocks(page: Any) -> list[dict[str, Any]]:
    records = []
    payload = page.get_text("dict")
    for parent_index, block in enumerate(payload.get("blocks", []), 1):
        if int(block.get("type", 0)) != 0:
            continue
        for line in block.get("lines", []):
            spans = [span for span in line.get("spans", []) if _clean(span.get("text", ""))]
            if not spans:
                continue
            spans.sort(key=lambda span: float(span.get("bbox", [0, 0, 0, 0])[0]))
            text = _clean(" ".join(str(span.get("text") or "") for span in spans))
            bbox_values = [span.get("bbox", [0, 0, 0, 0]) for span in spans]
            bbox = [
                round(min(float(value[0]) for value in bbox_values), 3),
                round(min(float(value[1]) for value in bbox_values), 3),
                round(max(float(value[2]) for value in bbox_values), 3),
                round(max(float(value[3]) for value in bbox_values), 3),
            ]
            sizes = [float(span.get("size") or 0.0) for span in spans if float(span.get("size") or 0.0) > 0]
            span_bold = [
                bool(int(span.get("flags") or 0) & 16) or bool(re.search(r"(?:bold|demi|semi|medi)", str(span.get("font") or ""), re.IGNORECASE))
                for span in spans
            ]
            span_lengths = [max(1, len(_clean(span.get("text", "")))) for span in spans]
            bold_characters = sum(length for length, bold in zip(span_lengths, span_bold) if bold)
            total_characters = sum(span_lengths)
            font_bold_ratio = round(bold_characters / max(1, total_characters), 4)
            font_bold = font_bold_ratio >= 0.8
            records.append({
                "bbox": bbox,
                "text": text,
                "source": "pymupdf",
                "confidence": 1.0,
                "parent_block": parent_index,
                "font_size": round(max(sizes), 3) if sizes else None,
                "font_bold": font_bold,
                "font_bold_ratio": font_bold_ratio,
            })
    return records


def _merge_native_hyphenated_lines(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    merged: list[dict[str, Any]] = []
    repairs = 0
    for record in records:
        current = dict(record)
        if not merged:
            merged.append(current)
            continue
        previous = merged[-1]
        previous_text = str(previous.get("text") or "")
        current_text = str(current.get("text") or "")
        match_left = re.search(r"([A-Za-z]{2,})-$", previous_text)
        match_right = re.match(r"([A-Za-z]{2,})", current_text)
        same_parent = previous.get("parent_block") is not None and previous.get("parent_block") == current.get("parent_block")
        if not same_parent or not match_left or not match_right:
            merged.append(current)
            continue
        combined_word = f"{match_left.group(1)}{match_right.group(1)}".casefold()
        separator = "" if combined_word in OCR_PROTECTED_WORDS or len(match_left.group(1)) <= 3 else "-"
        joined_text = f"{previous_text[:-1]}{separator}{current_text.lstrip()}"
        previous_bbox = previous.get("bbox") or [0, 0, 0, 0]
        current_bbox = current.get("bbox") or previous_bbox
        previous.update({
            "text": _clean(joined_text),
            "bbox": [
                round(min(float(previous_bbox[0]), float(current_bbox[0])), 3),
                round(min(float(previous_bbox[1]), float(current_bbox[1])), 3),
                round(max(float(previous_bbox[2]), float(current_bbox[2])), 3),
                round(max(float(previous_bbox[3]), float(current_bbox[3])), 3),
            ],
            "confidence": min(float(previous.get("confidence") or 1.0), float(current.get("confidence") or 1.0)),
            "font_size": max(float(previous.get("font_size") or 0.0), float(current.get("font_size") or 0.0)) or None,
            "font_bold": bool(previous.get("font_bold")) and bool(current.get("font_bold")),
            "native_hyphenation_repairs": int(previous.get("native_hyphenation_repairs") or 0) + 1,
        })
        repairs += 1
    return merged, repairs


def _fragmented_word_page(records: list[dict[str, Any]]) -> bool:
    """A publisher PDF that stored each word as its own line.

    Column detection ignores those short pieces and then reads straight across
    both columns. A page whose lines are already sentences is left alone.
    """
    if len(records) < 12:
        return False
    short = sum(1 for item in records if len(_normalized_compare_text(str(item.get("text") or ""))) < 15)
    return short >= len(records) * 0.55


def _assemble_fragmented_lines(records: list[dict[str, Any]], page_width: float) -> tuple[list[dict[str, Any]], bool]:
    """Join word fragments that share a baseline and only a word space.

    A column gutter is wider than a word space and stays unjoined, so the two
    columns remain two lines. Pages that already have real lines come back
    unchanged.
    """
    if not _fragmented_word_page(records):
        return records, False

    def box(item: dict[str, Any]) -> list[float]:
        value = item.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]

    def height_of(item: dict[str, Any]) -> float:
        value = box(item)
        return max(1.0, value[3] - value[1])

    heights = sorted(height_of(item) for item in records)
    median_height = heights[len(heights) // 2]
    ordered = sorted(records, key=lambda item: ((box(item)[1] + box(item)[3]) / 2.0, box(item)[0]))
    lines: list[dict[str, Any]] = []
    for record in ordered:
        bbox = box(record)
        center_y = (bbox[1] + bbox[3]) / 2.0
        height = height_of(record)
        match = None
        for line in reversed(lines[-8:]):
            if abs(center_y - float(line["center_y"])) > max(height, float(line["height"]), median_height) * 0.45:
                continue
            gap = bbox[0] - float(line["bbox"][2])
            allowance = max(height, float(line["height"]), median_height) * 1.8
            if gap < -max(height, float(line["height"])) * 0.35 or gap > allowance:
                continue
            midpoint = page_width / 2.0
            crosses = float(line["bbox"][0]) < midpoint < bbox[2]
            either_crosses = float(line["bbox"][0]) < midpoint < float(line["bbox"][2]) or bbox[0] < midpoint < bbox[2]
            if crosses and not either_crosses:
                continue
            match = line
            break
        if match is None:
            lines.append({"words": [record], "bbox": list(bbox), "center_y": center_y, "height": height})
            continue
        match["words"].append(record)
        match["bbox"] = [
            min(float(match["bbox"][0]), bbox[0]),
            min(float(match["bbox"][1]), bbox[1]),
            max(float(match["bbox"][2]), bbox[2]),
            max(float(match["bbox"][3]), bbox[3]),
        ]
        match["center_y"] = sum((box(item)[1] + box(item)[3]) / 2.0 for item in match["words"]) / len(match["words"])
        match["height"] = sum(height_of(item) for item in match["words"]) / len(match["words"])

    assembled = []
    for index, line in enumerate(lines, 1):
        words = sorted(line["words"], key=lambda item: box(item)[0])
        sizes = [float(item.get("font_size") or 0.0) for item in words if float(item.get("font_size") or 0.0) > 0]
        bold_characters = sum(len(str(item.get("text") or "")) for item in words if item.get("font_bold"))
        total_characters = sum(len(str(item.get("text") or "")) for item in words) or 1
        assembled.append({
            "bbox": [round(float(value), 3) for value in line["bbox"]],
            "text": _clean(" ".join(str(item.get("text") or "") for item in words)),
            "source": words[0].get("source") or "pymupdf",
            "confidence": min(float(item.get("confidence") or 1.0) for item in words),
            "parent_block": index,
            "font_size": round(max(sizes), 3) if sizes else None,
            "font_bold": bold_characters / total_characters >= 0.8,
            "font_bold_ratio": round(bold_characters / total_characters, 4),
        })
    return assembled, True


def _join_column_paragraphs(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Join stacked lines of one column into a paragraph.

    A contract quote is a sentence. One visual line is shorter than the quote
    opening the rebind looks for, and it usually begins with the previous
    sentence's last words, so the line itself is not the quote either.
    """
    if len(records) < 2:
        return records, 0
    heights = sorted(
        max(1.0, float(item["bbox"][3]) - float(item["bbox"][1]))
        for item in records
        if isinstance(item.get("bbox"), (list, tuple)) and len(item.get("bbox")) == 4
    )
    median_height = heights[len(heights) // 2] if heights else 12.0
    joined: list[dict[str, Any]] = []
    repairs = 0
    for record in records:
        current = dict(record)
        bbox = current.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        if not joined or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            joined.append(current)
            continue
        previous = joined[-1]
        previous_bbox = previous.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        if not isinstance(previous_bbox, (list, tuple)) or len(previous_bbox) != 4:
            joined.append(current)
            continue
        same_column = previous.get("column") == current.get("column")
        gap = float(bbox[1]) - float(previous_bbox[3])
        local_height = max(
            median_height,
            float(bbox[3]) - float(bbox[1]),
            float(previous_bbox[3]) - float(previous_bbox[1]),
        )
        text = str(current.get("text") or "")
        previous_text_now = str(previous.get("text") or "")
        starts_new = bool(
            re.match(r"^(?:abstract|\d+(?:\.\d+)*\s+)\b", text, re.IGNORECASE)
            or _caption_prefix_kind(text)
        )
        previous_size = float(previous.get("font_size") or 0.0)
        current_size = float(current.get("font_size") or 0.0)
        larger_type = bool(previous_size and current_size >= previous_size * 1.35 and len(_normalized_compare_text(text)) <= 80)
        bold_heading = bool(current.get("font_bold") and not previous.get("font_bold") and len(_normalized_compare_text(text)) <= 80)
        # A short bold line is the heading itself. The body that follows starts
        # its own paragraph instead of swallowing the heading.
        previous_is_heading = bool(previous.get("font_bold")) and len(_normalized_compare_text(previous_text_now)) <= 80
        if not same_column or gap > local_height * 1.25 or gap < -local_height * 0.55 or starts_new or larger_type or bold_heading or previous_is_heading:
            joined.append(current)
            continue
        previous_text = str(previous.get("text") or "")
        match_left = re.search(r"([A-Za-z]{2,})-$", previous_text)
        match_right = re.match(r"([A-Za-z]{2,})", text)
        if match_left and match_right:
            combined_word = f"{match_left.group(1)}{match_right.group(1)}".casefold()
            separator = "" if combined_word in OCR_PROTECTED_WORDS or len(match_left.group(1)) <= 3 else "-"
            previous_text = f"{previous_text[:-1]}{separator}{text.lstrip()}"
            repairs += 1
        else:
            previous_text = f"{previous_text} {text}"
        previous["text"] = _clean(previous_text)
        previous["bbox"] = [
            round(min(float(previous_bbox[0]), float(bbox[0])), 3),
            round(min(float(previous_bbox[1]), float(bbox[1])), 3),
            round(max(float(previous_bbox[2]), float(bbox[2])), 3),
            round(max(float(previous_bbox[3]), float(bbox[3])), 3),
        ]
        previous["font_bold"] = bool(previous.get("font_bold")) and bool(current.get("font_bold"))
        if previous_size and current_size:
            previous["font_size"] = round(max(previous_size, current_size), 3)
        previous["native_hyphenation_repairs"] = int(previous.get("native_hyphenation_repairs") or 0) + (1 if match_left and match_right else 0)
    for index, item in enumerate(joined, 1):
        item["reading_order"] = index
    return joined, repairs


def _prepare_native_blocks(records: list[dict[str, Any]], page_width: float) -> tuple[list[dict[str, Any]], float, int]:
    """Turn raw text objects into reading-ordered blocks.

    Word-sized objects are assembled into lines and then into paragraphs before
    anyone tries to match a contract quote. A page of ordinary lines keeps the
    order it already had.
    """
    lines, shredded = _assemble_fragmented_lines(records, page_width)
    merged, repairs = _merge_native_hyphenated_lines(lines)
    ordered, confidence = _reading_order(merged, page_width)
    if shredded:
        ordered, extra = _join_column_paragraphs(ordered)
        repairs += extra
    return ordered, confidence, repairs


def _poppler_pages(path: Path, timeout: int = 30) -> tuple[list[str], str | None]:
    executable = shutil.which("pdftotext")
    if not executable:
        return [], "pdftotext unavailable"
    try:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "paper.txt"
            subprocess.run([executable, "-layout", str(path), str(target)], check=True, timeout=timeout, capture_output=True)
            return target.read_text(encoding="utf-8", errors="replace").split("\f"), None
    except Exception as exc:
        return [], str(exc)


def _pdfplumber_blocks(path: Path, page_number: int) -> list[dict[str, Any]]:
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as document:
            page = document.pages[page_number - 1]
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False) or []
        rows: list[list[dict[str, Any]]] = []
        for word in sorted(words, key=lambda item: (float(item.get("top", 0.0)), float(item.get("x0", 0.0)))):
            row = next((items for items in rows if abs(float(items[0].get("top", 0.0)) - float(word.get("top", 0.0))) <= 3.0), None)
            if row is None:
                row = []
                rows.append(row)
            row.append(word)
        blocks = []
        for row in rows:
            row.sort(key=lambda item: float(item.get("x0", 0.0)))
            text = _clean(" ".join(str(item.get("text") or "") for item in row))
            if text:
                blocks.append({
                    "bbox": [min(float(item["x0"]) for item in row), min(float(item["top"]) for item in row), max(float(item["x1"]) for item in row), max(float(item["bottom"]) for item in row)],
                    "text": text,
                    "source": "pdfplumber",
                    "confidence": 0.92,
                })
        return blocks
    except Exception:
        return []


def _pdf_metadata(path: Path) -> dict[str, Any]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception:
                unlocked = 0
            if not unlocked:
                raise ValueError("PDF is encrypted and cannot be opened without a password")
        return {"page_count": len(reader.pages), "encrypted": bool(reader.is_encrypted), "metadata": {str(key): str(value) for key, value in (reader.metadata or {}).items()}}
    except ImportError:
        return {"page_count": None, "encrypted": None, "metadata": {}, "warning": "pypdf unavailable"}


def _validate_pdf_container(path: Path) -> None:
    size = path.stat().st_size
    if size < 8:
        raise ValueError("PDF is empty or too small to contain a valid document")
    with path.open("rb") as handle:
        header = handle.read(min(1024, size))
        handle.seek(max(0, size - 4096))
        tail = handle.read()
    if b"%PDF-" not in header:
        raise ValueError("File does not contain a valid PDF header")
    if b"%%EOF" not in tail:
        raise ValueError("PDF appears truncated because the EOF marker is missing")


def _has_pdf_header(path: Path) -> bool:
    """True when the file starts with the PDF magic, whatever it is named.

    Snapshot archive objects are addressed by their sha256 and carry no suffix
    at all.  Choosing the reader by name alone read such a 7.8 MB PDF as text:
    the paper came back as one page of mojibake, every section check failed, and
    the run died in paper preparation while the reader still reported a
    successful parse.
    """

    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def _ocr_records(image_path: Path, engine: str, lang: str, adapter: Callable | None, rapidocr_threads: int = 1) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    if adapter:
        return list(adapter(image_path, lang) or []), "adapter", {}
    from .ocr import run_easyocr, run_paddle_ocr, run_rapidocr_detailed

    requested = engine
    if requested == "auto":
        requested = "rapidocr"
        try:
            import rapidocr_onnxruntime  # noqa: F401
        except Exception:
            requested = "easyocr"
            try:
                import easyocr  # noqa: F401
            except Exception:
                requested = "paddle"
    if requested == "rapidocr":
        records, diagnostics = run_rapidocr_detailed(image_path, lang, threads=rapidocr_threads)
        return records, "rapidocr", diagnostics
    if requested == "easyocr":
        return run_easyocr(image_path, lang), "easyocr", {}
    if requested == "paddle":
        return run_paddle_ocr(image_path, lang), "paddle", {}
    return [], "off", {}


def _group_ocr_words(records: list[dict[str, Any]], image_width: float) -> list[dict[str, Any]]:
    words = []
    for record in records:
        quad = record.get("quad") or []
        if len(quad) < 4 or not _clean(record.get("text", "")):
            continue
        xs = [float(point[0]) for point in quad]
        ys = [float(point[1]) for point in quad]
        words.append({
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "text": _clean(record.get("text", "")),
            "confidence": float(record.get("confidence") or 0.0),
        })
    words.sort(key=lambda item: ((item["bbox"][1] + item["bbox"][3]) / 2.0, item["bbox"][0]))
    lines: list[dict[str, Any]] = []
    for word in words:
        bbox = word["bbox"]
        center_y = (bbox[1] + bbox[3]) / 2.0
        height = max(1.0, bbox[3] - bbox[1])
        line = next((
            item for item in reversed(lines[-12:])
            if abs(center_y - item["center_y"]) <= max(height, item["mean_height"]) * 0.62
            and bbox[0] >= item["bbox"][0] - image_width * 0.08
        ), None)
        if line is None:
            lines.append({"words": [word], "bbox": list(bbox), "center_y": center_y, "mean_height": height})
            continue
        line["words"].append(word)
        line["bbox"] = [min(line["bbox"][0], bbox[0]), min(line["bbox"][1], bbox[1]), max(line["bbox"][2], bbox[2]), max(line["bbox"][3], bbox[3])]
        line["center_y"] = sum((item["bbox"][1] + item["bbox"][3]) / 2.0 for item in line["words"]) / len(line["words"])
        line["mean_height"] = sum(max(1.0, item["bbox"][3] - item["bbox"][1]) for item in line["words"]) / len(line["words"])

    grouped = []
    for line in lines:
        line["words"].sort(key=lambda item: item["bbox"][0])
        grouped.append({
            "bbox": line["bbox"],
            "text": _clean(" ".join(item["text"] for item in line["words"])),
            "confidence": round(sum(item["confidence"] for item in line["words"]) / max(1, len(line["words"])), 4),
        })
    grouped.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    heights = sorted(max(1.0, item["bbox"][3] - item["bbox"][1]) for item in grouped)
    median_height = heights[len(heights) // 2] if heights else 12.0
    parent = 0
    previous_by_column: dict[int, dict[str, Any]] = {}
    for item in grouped:
        center_x = (item["bbox"][0] + item["bbox"][2]) / 2.0
        width = item["bbox"][2] - item["bbox"][0]
        column = 2 if width >= image_width * 0.62 else 0 if center_x <= image_width / 2.0 else 1
        previous = previous_by_column.get(column)
        gap = item["bbox"][1] - previous["bbox"][3] if previous else float("inf")
        starts_new = bool(
            re.match(r"^(?:abstract|\d+(?:\.\d+)*\s+)\b", item["text"], re.IGNORECASE)
            or _caption_prefix_kind(item["text"])
        )
        if previous is None or gap > median_height * 1.8 or starts_new:
            parent += 1
        item["parent_block"] = parent
        previous_by_column[column] = item
    return grouped


def _assign_ocr_parent_blocks(lines: list[dict[str, Any]], image_width: float) -> list[dict[str, Any]]:
    ordered = sorted(lines, key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])))
    heights = sorted(max(1.0, float(item["bbox"][3]) - float(item["bbox"][1])) for item in ordered)
    median_height = heights[len(heights) // 2] if heights else 12.0
    parent = 0
    previous_by_column: dict[int, dict[str, Any]] = {}
    last_item: dict[str, Any] | None = None
    for item in ordered:
        center_x = (float(item["bbox"][0]) + float(item["bbox"][2])) / 2.0
        width = float(item["bbox"][2]) - float(item["bbox"][0])
        column = 2 if width >= image_width * 0.62 else 0 if center_x <= image_width / 2.0 else 1
        previous = previous_by_column.get(column)
        if last_item and last_item.get("_caption_group"):
            last_gap = float(item["bbox"][1]) - float(last_item["bbox"][3])
            last_height = max(
                median_height,
                float(item["bbox"][3]) - float(item["bbox"][1]),
                float(last_item["bbox"][3]) - float(last_item["bbox"][1]),
            )
            if last_gap <= last_height * 1.15:
                previous = last_item
        gap = float(item["bbox"][1]) - float(previous["bbox"][3]) if previous else float("inf")
        local_height = max(
            median_height,
            float(item["bbox"][3]) - float(item["bbox"][1]),
            (float(previous["bbox"][3]) - float(previous["bbox"][1])) if previous else 0.0,
        )
        text = str(item.get("text") or "")
        starts_new = bool(re.match(r"^(?:abstract|\d+(?:\.\d+)*\s+)\b", text, re.IGNORECASE) or _caption_prefix_kind(text))
        starts_caption = bool(_caption_prefix_kind(text))
        continuing_caption = bool(previous and previous.get("_caption_group") and gap <= local_height * 1.15 and not starts_new)
        paragraph_break = bool(previous and not continuing_caption and str(previous.get("text") or "").rstrip().endswith((".", ":")) and gap > local_height * 0.45)
        if previous is None or gap > local_height * 1.15 or starts_new or paragraph_break:
            parent += 1
        item["parent_block"] = parent
        item["_caption_group"] = starts_caption or continuing_caption
        previous_by_column[column] = item
        last_item = item
    return ordered


def _filter_ocr_margin_noise(records: list[dict[str, Any]], image_width: float) -> tuple[list[dict[str, Any]], int]:
    anchors = [
        item
        for item in records
        if len(_normalized_compare_text(str(item.get("text") or ""))) >= 20
        and float(item["bbox"][2]) - float(item["bbox"][0]) >= image_width * 0.30
    ]
    if len(anchors) < 3:
        return records, 0
    left = sorted(float(item["bbox"][0]) for item in anchors)[max(0, len(anchors) // 10 - 1)]
    right_values = sorted(float(item["bbox"][2]) for item in anchors)
    right = right_values[min(len(right_values) - 1, len(right_values) - max(1, len(anchors) // 10))]
    margin = image_width * 0.035
    kept = []
    removed = 0
    for item in records:
        bbox = item.get("bbox") or [0, 0, 0, 0]
        outside = float(bbox[2]) < left - margin or float(bbox[0]) > right + margin
        width = max(1.0, float(bbox[2]) - float(bbox[0]))
        height = max(1.0, float(bbox[3]) - float(bbox[1]))
        short = len(_normalized_compare_text(str(item.get("text") or ""))) < 12
        vertical = height > width * 1.15
        outer_band = float(bbox[2]) < image_width * 0.12 or float(bbox[0]) > image_width * 0.88
        compact_margin_fragment = outer_band and short and (vertical or width < image_width * 0.08)
        if (outside and (short or vertical)) or compact_margin_fragment:
            removed += 1
            continue
        kept.append(item)
    return kept, removed


def _repeated_margin_signature(text: str) -> str:
    value = _clean(text).casefold()
    value = re.sub(r"\d+", "#", value)
    value = re.sub(r"\s+", " ", value).strip(" |.-_")
    return value


def _semantic_margin_region(page: dict[str, Any], bbox: list[float] | tuple[float, ...]) -> str | None:
    width = float(page.get("width") or 0.0)
    height = float(page.get("height") or 0.0)
    if width <= 0 or height <= 0:
        return None
    rotation = int(page.get("rotation") or 0) % 360
    if rotation == 90:
        if float(bbox[0]) >= width * 0.88:
            return "header"
        if float(bbox[2]) <= width * 0.12:
            return "footer"
    elif rotation == 180:
        if float(bbox[1]) >= height * 0.88:
            return "header"
        if float(bbox[3]) <= height * 0.12:
            return "footer"
    elif rotation == 270:
        if float(bbox[2]) <= width * 0.12:
            return "header"
        if float(bbox[0]) >= width * 0.88:
            return "footer"
    else:
        if float(bbox[3]) <= height * 0.12:
            return "header"
        if float(bbox[1]) >= height * 0.88:
            return "footer"
    return None


def _remove_repeated_margin_noise(pages: list[dict[str, Any]]) -> int:
    if len(pages) < 3:
        return 0
    occurrences: dict[tuple[str, str], set[int]] = {}
    candidates: dict[tuple[int, str], tuple[str, str]] = {}
    for page in pages:
        page_number = int(page.get("page") or 0)
        for block in page.get("blocks", []):
            bbox = block.get("bbox")
            text = str(block.get("text") or "").strip()
            if str(block.get("kind") or "") == "heading" or bool(block.get("font_bold")):
                continue
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not text or len(text) > 140 or len(_lexical_units(text)) > 12 or _cjk_character_count(text) > 36:
                continue
            region = _semantic_margin_region(page, bbox)
            if not region:
                continue
            signature = _repeated_margin_signature(text)
            if not signature:
                continue
            key = (region, signature)
            occurrences.setdefault(key, set()).add(page_number)
            candidates[(page_number, str(block.get("id") or ""))] = key

    threshold = max(3, (len(pages) + 3) // 4)
    repeated = {key for key, page_numbers in occurrences.items() if len(page_numbers) >= threshold}
    if not repeated:
        return 0

    removed = 0
    for page in pages:
        page_number = int(page.get("page") or 0)
        kept = []
        page_removed = 0
        for block in page.get("blocks", []):
            key = candidates.get((page_number, str(block.get("id") or "")))
            if key in repeated:
                page_removed += 1
                continue
            kept.append(block)
        if not page_removed:
            continue
        for order, block in enumerate(kept, 1):
            block["reading_order"] = order
        page_text = "\n\n".join(str(item.get("text") or "") for item in kept)
        page.update({
            "blocks": kept,
            "text": page_text,
            "char_count": len(page_text),
            "replacement_character_rate": _replacement_rate(page_text),
            "mojibake_rate": _mojibake_rate(page_text),
            "column_count": max((int(item.get("column", 0)) for item in kept), default=0) + 1,
            "repeated_margin_noise_removed": page_removed,
        })
        removed += page_removed
    return removed


def _ocr_page(page: Any, page_number: int, engine: str, lang: str, adapter: Callable | None, rapidocr_threads: int = 1) -> tuple[list[dict[str, Any]], str, str | None, dict[str, Any]]:
    try:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / f"page_{page_number:03d}.png"
            dpi = OCR_RENDER_DPI if adapter is None and engine in {"auto", "easyocr", "rapidocr"} else 160
            render_started = time.monotonic()
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            pixmap.save(str(target))
            render_seconds = time.monotonic() - render_started
            records, used_engine, engine_diagnostics = _ocr_records(target, engine, lang, adapter, rapidocr_threads=rapidocr_threads)
            postprocess_started = time.monotonic()
            spacing_repairs = 0
            for record in records:
                repaired, count = _repair_ocr_spacing(record.get("text", ""))
                record["text"] = repaired
                spacing_repairs += count
            if used_engine != "rapidocr":
                records = _group_ocr_words(records, float(pixmap.width))
            else:
                normalized_records = []
                for record in records:
                    quad = record.get("quad") or []
                    if len(quad) < 4:
                        continue
                    xs = [float(point[0]) for point in quad]
                    ys = [float(point[1]) for point in quad]
                    normalized_records.append({"bbox": [min(xs), min(ys), max(xs), max(ys)], "text": _clean(record.get("text", "")), "confidence": float(record.get("confidence") or 0.0)})
                records = _assign_ocr_parent_blocks(normalized_records, float(pixmap.width))
            records, margin_noise_removed = _filter_ocr_margin_noise(records, float(pixmap.width))
            blocks = []
            scale_x = float(page.rect.width) / max(1, pixmap.width)
            scale_y = float(page.rect.height) / max(1, pixmap.height)
            for record in records:
                raw_bbox = record.get("bbox")
                if not raw_bbox:
                    continue
                blocks.append({
                    "bbox": [float(raw_bbox[0]) * scale_x, float(raw_bbox[1]) * scale_y, float(raw_bbox[2]) * scale_x, float(raw_bbox[3]) * scale_y],
                    "text": _clean(record.get("text", "")),
                    "source": used_engine,
                    "confidence": round(float(record.get("confidence") or 0.0), 4),
                    "parent_block": record.get("parent_block"),
                })
            return blocks, used_engine, None, {
                "margin_noise_removed": margin_noise_removed,
                "spacing_repairs": spacing_repairs,
                "render_seconds": round(render_seconds, 4),
                "render_dpi": dpi,
                "postprocess_seconds": round(time.monotonic() - postprocess_started, 4),
                **engine_diagnostics,
            }
    except Exception as exc:
        return [], engine, str(exc), {"margin_noise_removed": 0, "spacing_repairs": 0}


def _rapidocr_worker_count(engine: str, adapter: Callable | None, page_count: int) -> int:
    if adapter is not None or page_count < 2 or engine not in {"auto", "rapidocr"}:
        return 1
    if engine == "auto":
        try:
            import rapidocr_onnxruntime  # noqa: F401
        except Exception:
            return 1
    cpu_count = int(os.cpu_count() or 1)
    default_workers = 4 if cpu_count >= 8 else 2 if cpu_count >= 4 else 1
    try:
        configured = int(os.getenv("RFS_OCR_WORKERS") or default_workers)
    except ValueError:
        configured = default_workers
    return max(1, min(int(page_count), configured, cpu_count))


def _deadline_ocr_wave(
    path: Path,
    page_numbers: list[int],
    engine: str,
    lang: str,
    deadline_at: float,
) -> list[tuple[int, tuple[list[dict[str, Any]], str, str | None, dict[str, Any]], float]]:
    """Run OCR pages in killable child processes and preserve completed results."""
    cutoff = max(time.monotonic(), float(deadline_at) - 20.0)
    worker_threads = 2 if len(page_numbers) == 1 and engine in {"auto", "rapidocr"} else 1
    creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
    with tempfile.TemporaryDirectory() as temp:
        processes: dict[int, dict[str, Any]] = {}
        for page_number in page_numbers:
            output = Path(temp) / f"page_{page_number:03d}.json"
            command = [
                sys.executable,
                "-m",
                "rfs.analysis.paper.ocr_worker",
                "--paper",
                str(path),
                "--page",
                str(page_number),
                "--engine",
                engine,
                "--lang",
                lang,
                "--threads",
                str(worker_threads),
                "--out",
                str(output),
            ]
            started = time.monotonic()
            try:
                # Windows OCR workers may load OpenCV/ONNXRuntime alongside
                # another package that bundles Intel OpenMP (notably
                # EasyOCR/PyTorch).  Without an isolated single-threaded
                # runtime, libiomp5md.dll aborts the worker with OMP Error #15
                # and leaves no Python traceback for the parent to report.
                worker_env = os.environ.copy()
                if os.name == "nt":
                    worker_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
                    worker_env.setdefault("OMP_NUM_THREADS", "1")
                    worker_env.setdefault("MKL_NUM_THREADS", "1")
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=creation_flags,
                    env=worker_env,
                )
                processes[page_number] = {"process": process, "output": output, "started": started, "launch_error": None}
            except Exception as exc:
                processes[page_number] = {"process": None, "output": output, "started": started, "launch_error": str(exc)}

        pending = {page for page, item in processes.items() if item["process"] is not None}
        while pending and time.monotonic() < cutoff:
            finished = {page for page in pending if processes[page]["process"].poll() is not None}
            pending.difference_update(finished)
            if pending:
                time.sleep(0.02)

        timed_out = set(pending)
        for page_number in timed_out:
            process = processes[page_number]["process"]
            try:
                process.terminate()
            except Exception:
                pass
        for page_number in timed_out:
            process = processes[page_number]["process"]
            try:
                process.wait(timeout=1.0)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=1.0)
                except Exception:
                    pass

        results = []
        for page_number in page_numbers:
            item = processes[page_number]
            process = item["process"]
            elapsed = time.monotonic() - float(item["started"])
            stderr = ""
            if process is not None:
                try:
                    _stdout, stderr = process.communicate(timeout=0.2)
                except Exception:
                    stderr = ""
            if item["launch_error"]:
                result = ([], engine, f"OCR worker launch failed: {item['launch_error']}", {"worker_process": True})
            elif page_number in timed_out:
                result = ([], engine, "OCR exceeded extraction deadline", {"worker_process": True, "timed_out": True})
            elif item["output"].exists():
                try:
                    payload = json.loads(item["output"].read_text(encoding="utf-8"))
                    result = (
                        list(payload.get("blocks") or []),
                        str(payload.get("engine") or engine),
                        str(payload.get("error")) if payload.get("error") else None,
                        {"worker_process": True, **dict(payload.get("diagnostics") or {})},
                    )
                    elapsed = float(payload.get("elapsed_seconds") or elapsed)
                except Exception as exc:
                    result = ([], engine, f"OCR worker returned invalid output: {exc}", {"worker_process": True})
            else:
                detail = stderr.strip() or f"worker exited with code {getattr(process, 'returncode', None)}"
                result = ([], engine, f"OCR worker failed: {detail}", {"worker_process": True})
            results.append((page_number, result, elapsed))
        return results


def _prioritize_ocr_candidates(
    pages: list[dict[str, Any]],
    candidates: list[int],
    max_pages: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    if max_pages <= 0 or not candidates:
        return [], []
    page_count = max(1, len(pages))
    signals = {
        "overview_figure": re.compile(r"\b(?:figure|fig\.)\s*[12]\b.*\b(?:overview|framework|architecture|pipeline|system|model)\b", re.IGNORECASE | re.DOTALL),
        "method": re.compile(r"\b(?:methods?|methodology|approach|architecture|system overview|framework)\b", re.IGNORECASE),
        "abstract": re.compile(r"\babstract\b", re.IGNORECASE),
        "conclusion": re.compile(r"\b(?:conclusions?|discussion)\b", re.IGNORECASE),
        "caption": re.compile(r"\b(?:figure|fig\.|table)\s*\d", re.IGNORECASE),
    }
    scored: list[tuple[int, int, list[str]]] = []
    for page_number in candidates:
        page = pages[page_number - 1] if 0 < page_number <= len(pages) else {}
        text = str(page.get("text") or "")
        reasons: list[str] = []
        score = 0
        for name, weight in (("overview_figure", 14), ("method", 10), ("abstract", 9), ("conclusion", 8), ("caption", 5)):
            if signals[name].search(text):
                score += weight
                reasons.append(name)
        if page_number == 1:
            score += 7
            reasons.append("first_page")
        elif page_number <= 3:
            score += 3
            reasons.append("early_page")
        scored.append((score, page_number, reasons))

    selected: list[int] = []
    reason_map: dict[int, list[str]] = {}
    for score, page_number, reasons in sorted(scored, key=lambda item: (-item[0], item[1])):
        if score <= 0 or len(selected) >= max_pages:
            break
        selected.append(page_number)
        reason_map[page_number] = reasons

    anchors = [
        1,
        min(2, page_count),
        min(3, page_count),
        min(4, page_count),
        max(1, round(page_count * 0.85)),
        max(1, round(page_count * 0.50)),
    ]
    for anchor in anchors:
        if len(selected) >= max_pages:
            break
        if anchor in selected:
            continue
        available = [value for value in candidates if value not in selected]
        if not available:
            break
        page_number = min(available, key=lambda value: (abs(value - anchor), value))
        selected.append(page_number)
        reason_map.setdefault(page_number, []).append(f"coverage_anchor_{anchor}")

    for page_number in candidates:
        if len(selected) >= max_pages:
            break
        if page_number not in selected:
            selected.append(page_number)
            reason_map.setdefault(page_number, []).append("remaining_candidate")

    details = [
        {"page": page_number, "rank": rank, "reasons": reason_map.get(page_number, [])}
        for rank, page_number in enumerate(selected, 1)
    ]
    return selected, details


def _read_pdf_pages(
    path: Path,
    max_chars: int,
    deadline_at: float | None,
    ocr_engine: str,
    ocr_lang: str,
    ocr_adapter: Callable | None,
    max_ocr_pages: int,
    ocr_rescue_min_remaining: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError(f"PyMuPDF is required for PDF extraction: {exc}") from exc

    metadata = _pdf_metadata(path)
    started = time.monotonic()
    document = fitz.open(str(path))
    poppler_text, poppler_warning = _poppler_pages(path)
    pages: list[dict[str, Any]] = []
    warnings: list[str] = [poppler_warning] if poppler_warning else []
    used_chars = 0
    ocr_candidates: list[int] = []
    try:
        for index, page in enumerate(document, 1):
            native_width = float(page.mediabox.width) if int(page.rotation or 0) else float(page.rect.width)
            ordered, order_confidence, hyphenation_repairs = _prepare_native_blocks(_pymupdf_line_blocks(page), native_width)
            if len(_normalized_compare_text(" ".join(item["text"] for item in ordered))) < 80:
                fallback_ordered, fallback_confidence, fallback_hyphenation_repairs = _prepare_native_blocks(_pdfplumber_blocks(path, index), native_width)
                if len(_normalized_compare_text(" ".join(item["text"] for item in fallback_ordered))) > len(_normalized_compare_text(" ".join(item["text"] for item in ordered))):
                    ordered, order_confidence = fallback_ordered, min(fallback_confidence, 0.9)
                    hyphenation_repairs = fallback_hyphenation_repairs
            ordered = _rotate_blocks_to_page(ordered, page)
            for block_index, block in enumerate(ordered, 1):
                block["id"] = f"P{index:03d}_B{block_index:03d}"
                block["kind"] = _block_kind(block["text"], (block["bbox"][2] - block["bbox"][0]) / max(1.0, float(page.rect.width)))
            page_text = "\n\n".join(item["text"] for item in ordered)
            poppler_page = poppler_text[index - 1] if index - 1 < len(poppler_text) else ""
            native_norm = _normalized_compare_text(page_text)
            poppler_norm = _normalized_compare_text(poppler_page)
            order_agreement = round(SequenceMatcher(None, native_norm[:20000], poppler_norm[:20000]).ratio(), 4) if native_norm and poppler_norm else None
            agreement = _token_overlap_agreement(page_text, poppler_page)
            abnormal = len(native_norm) < 80 or _replacement_rate(page_text) > 0.005 or _mojibake_rate(page_text) > 0.005 or (agreement is not None and agreement < 0.65)
            if abnormal:
                ocr_candidates.append(index)
            pages.append({
                "page": index,
                "width": round(float(page.rect.width), 3),
                "height": round(float(page.rect.height), 3),
                "rotation": int(page.rotation or 0),
                "column_count": max((int(item.get("column", 0)) for item in ordered), default=0) + 1,
                "blocks": ordered,
                "text": page_text,
                "char_count": len(page_text),
                "replacement_character_rate": _replacement_rate(page_text),
                "mojibake_rate": _mojibake_rate(page_text),
                "poppler_agreement": agreement,
                "poppler_order_agreement": order_agreement,
                "reading_order_confidence": order_confidence,
                "native_hyphenation_repair_count": hyphenation_repairs,
                "used_ocr": False,
            })
            used_chars += len(page_text)
            if used_chars >= max_chars and "document text exceeds evidence budget; evidence will be sampled across all pages" not in warnings:
                warnings.append("document text exceeds evidence budget; evidence will be sampled across all pages")
    finally:
        pass

    ocr_pages: list[int] = []
    ocr_page_durations: list[dict[str, Any]] = []
    prioritized: list[int] = []
    ocr_priority: list[dict[str, Any]] = []
    ocr_worker_count = 1
    if ocr_engine != "off":
        prioritized, ocr_priority = _prioritize_ocr_candidates(pages, ocr_candidates, max_ocr_pages)
        ocr_worker_count = _rapidocr_worker_count(ocr_engine, ocr_adapter, len(prioritized))

        def consume_ocr_result(page_number: int, result: tuple[list[dict[str, Any]], str, str | None, dict[str, Any]], ocr_elapsed: float) -> None:
            blocks, used_engine, error, ocr_diagnostics = result
            ocr_page_durations.append({"page": page_number, "engine": used_engine, "elapsed_seconds": round(ocr_elapsed, 3), "success": not bool(error), **ocr_diagnostics})
            if error:
                warnings.append(f"page {page_number} OCR unavailable: {error}")
                return
            current = pages[page_number - 1]
            ordered, order_confidence = _reading_order(blocks, float(current["width"]))
            for block_index, block in enumerate(ordered, 1):
                block["id"] = f"P{page_number:03d}_B{block_index:03d}"
                block["kind"] = _block_kind(block["text"], (block["bbox"][2] - block["bbox"][0]) / max(1.0, float(current["width"])))
            ocr_text = "\n\n".join(item["text"] for item in ordered)
            if len(_normalized_compare_text(ocr_text)) > len(_normalized_compare_text(current["text"])):
                current.update({
                    "blocks": ordered,
                    "text": ocr_text,
                    "char_count": len(ocr_text),
                    "replacement_character_rate": _replacement_rate(ocr_text),
                    "mojibake_rate": _mojibake_rate(ocr_text),
                    "reading_order_confidence": order_confidence,
                    "column_count": max((int(item.get("column", 0)) for item in ordered), default=0) + 1,
                    "used_ocr": True,
                    "ocr_engine": used_engine,
                    "ocr_margin_noise_removed": int(ocr_diagnostics.get("margin_noise_removed") or 0),
                    "native_hyphenation_repair_count": 0,
                })
                ocr_pages.append(page_number)

        def deadline_allows_wave(rescue_page: bool = False) -> bool:
            if deadline_at is not None:
                remaining = deadline_at - time.monotonic()
                if rescue_page:
                    if remaining < max(25.0, float(ocr_rescue_min_remaining)):
                        warnings.append("OCR stopped to preserve deadline validation budget")
                        return False
                    return True
                completed_times = [float(item["elapsed_seconds"]) for item in ocr_page_durations if isinstance(item.get("elapsed_seconds"), (int, float))]
                estimated = (max(completed_times) * 1.25) if completed_times else 45.0
                if remaining < estimated + 20:
                    warnings.append("OCR stopped to preserve deadline validation budget")
                    return False
            return True

        if deadline_at is not None and ocr_adapter is None:
            start = 0
            while start < len(prioritized):
                rescue_page = start > 0
                if not deadline_allows_wave(rescue_page=rescue_page):
                    break
                wave_size = 1 if rescue_page else ocr_worker_count
                batch = prioritized[start:start + wave_size]
                for page_number, result, ocr_elapsed in _deadline_ocr_wave(path, batch, ocr_engine, ocr_lang, deadline_at):
                    consume_ocr_result(page_number, result, ocr_elapsed)
                start += wave_size
        elif ocr_worker_count > 1:
            def isolated_ocr(page_number: int) -> tuple[tuple[list[dict[str, Any]], str, str | None, dict[str, Any]], float]:
                ocr_started = time.monotonic()
                try:
                    local_document = fitz.open(str(path))
                    try:
                        result = _ocr_page(local_document[page_number - 1], page_number, ocr_engine, ocr_lang, ocr_adapter, rapidocr_threads=1)
                        return result, time.monotonic() - ocr_started
                    finally:
                        local_document.close()
                except Exception as exc:
                    return ([], ocr_engine, str(exc), {"margin_noise_removed": 0}), time.monotonic() - ocr_started

            with ThreadPoolExecutor(max_workers=ocr_worker_count) as pool:
                for start in range(0, len(prioritized), ocr_worker_count):
                    if not deadline_allows_wave():
                        break
                    batch = prioritized[start:start + ocr_worker_count]
                    futures = {page_number: pool.submit(isolated_ocr, page_number) for page_number in batch}
                    for page_number in batch:
                        result, ocr_elapsed = futures[page_number].result()
                        consume_ocr_result(page_number, result, ocr_elapsed)
        else:
            rapidocr_threads = 2 if ocr_engine in {"auto", "rapidocr"} and ocr_adapter is None else 1
            for page_number in prioritized:
                if not deadline_allows_wave():
                    break
                ocr_started = time.monotonic()
                result = _ocr_page(document[page_number - 1], page_number, ocr_engine, ocr_lang, ocr_adapter, rapidocr_threads=rapidocr_threads)
                consume_ocr_result(page_number, result, time.monotonic() - ocr_started)
    repeated_margin_noise_removed = _remove_repeated_margin_noise(pages)
    document.close()
    elapsed = round(time.monotonic() - started, 3)
    attempted_pages = [int(item["page"]) for item in ocr_page_durations if item.get("page")]
    successful_attempt_pages = [int(item["page"]) for item in ocr_page_durations if item.get("page") and item.get("success")]
    schedule_complete = set(prioritized).issubset(attempted_pages)
    run_complete = schedule_complete and set(attempted_pages).issubset(successful_attempt_pages)
    if not schedule_complete and "OCR schedule was not completed before the extraction deadline" not in warnings:
        warnings.append("OCR schedule was not completed before the extraction deadline")
    return pages, {
        "metadata": metadata,
        "ocr_candidate_pages": ocr_candidates,
        "ocr_priority_pages": prioritized,
        "ocr_priority": ocr_priority,
        "ocr_pages": ocr_pages,
        "ocr_page_durations": ocr_page_durations,
        "ocr_attempted_pages": attempted_pages,
        "ocr_successful_attempt_pages": successful_attempt_pages,
        "ocr_schedule_complete": schedule_complete,
        "ocr_run_complete": run_complete,
        "ocr_worker_count": ocr_worker_count,
        "ocr_rescue_min_remaining_seconds": float(ocr_rescue_min_remaining),
        "repeated_margin_noise_removed_count": repeated_margin_noise_removed,
        "native_hyphenation_repair_count": sum(int(page.get("native_hyphenation_repair_count") or 0) for page in pages),
        "warnings": warnings,
        "poppler_available": bool(poppler_text),
        "elapsed_seconds": elapsed,
    }


def _read_non_pdf_pages(path: Path, max_chars: int) -> tuple[list[dict[str, Any]], str]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        loader = "python-docx"
        try:
            import docx

            document = docx.Document(str(path))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)[:max_chars]
        except Exception as exc:
            raise RuntimeError(f"DOCX extraction failed: {exc}") from exc
    else:
        loader = "plain"
        text = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    clean = _clean(text)
    paragraphs = [_clean(value) for value in re.split(r"\n\s*\n", clean) if _clean(value)]
    chunks = []
    for paragraph in paragraphs:
        lines = paragraph.splitlines()
        if len(lines) > 1 and re.match(r"^(?:abstract|\d+(?:\.\d+)*\s+|[ivx]+[.)]\s+)", lines[0], re.IGNORECASE):
            chunks.extend([_clean(lines[0]), _clean("\n".join(lines[1:]))])
        else:
            chunks.append(paragraph)
    blocks = []
    for index, chunk in enumerate(chunks or [clean], 1):
        blocks.append({"id": f"P001_B{index:03d}", "bbox": None, "text": chunk, "kind": _block_kind(chunk), "source": loader, "confidence": 1.0, "reading_order": index})
    return [{"page": 1, "width": None, "height": None, "blocks": blocks, "text": clean, "char_count": len(clean), "replacement_character_rate": _replacement_rate(clean), "mojibake_rate": _mojibake_rate(clean), "poppler_agreement": None, "reading_order_confidence": 1.0, "used_ocr": False}], loader


def _heading_candidates(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headings = []
    seen = set()
    body_started = False
    body_anchor: tuple[int, float] | None = None
    for page in pages:
        font_sizes = sorted(float(block.get("font_size")) for block in page.get("blocks", []) if isinstance(block.get("font_size"), (int, float)) and len(_normalized_compare_text(str(block.get("text") or ""))) >= 20)
        middle = len(font_sizes) // 2
        body_font_size = (font_sizes[middle] if len(font_sizes) % 2 else (font_sizes[middle - 1] + font_sizes[middle]) / 2.0) if font_sizes else None
        for block in page.get("blocks", []):
            lines = [_clean(value) for value in str(block.get("text", "")).splitlines() if _clean(value)]
            for line_index, line in enumerate(lines):
                if not 2 <= len(line) <= 140:
                    continue
                explicit = re.match(r"^(abstract|references|acknowledg(?:e)?ments?|appendix)$", line, re.IGNORECASE)
                numbered = re.match(r"^\s*((?:\d+(?:\.\d+)*)|(?:[ivx]+))[.)]?\s+(.{2,120})$", line, re.IGNORECASE)
                compact_line = _normalized_compare_text(line)
                block_bold = bool(block.get("font_bold"))
                font_size = float(block.get("font_size") or 0.0)
                first_line_alpha = next((char for char in line if char.isalpha()), "")
                heading_case = not first_line_alpha or not first_line_alpha.isascii() or first_line_alpha.isupper()
                known = len(line) <= 60 and not line.endswith((".", ",", ";")) and any(
                    compact_alias and heading_case and (compact_line == compact_alias or (block_bold and compact_line.startswith(compact_alias)))
                    for aliases in SECTION_ALIASES.values()
                    for alias in aliases
                    for compact_alias in [_normalized_compare_text(alias)]
                )
                block_y = float((block.get("bbox") or [0, 0, 0, 0])[1])
                after_anchor = body_anchor is not None and (int(page["page"]) > body_anchor[0] or block_y >= body_anchor[1])
                position_ok = after_anchor if body_anchor is not None else body_started
                size_ok = body_font_size is None or font_size >= body_font_size * 0.95
                typographic = position_ok and block_bold and size_ok and _looks_like_typographic_heading(line)
                if numbered:
                    title_candidate = numbered.group(2).strip()
                    first_alpha = next((char for char in title_candidate if char.isalpha()), "")
                    numbered_style = block.get("source") != "pymupdf" or body_font_size is None or block_bold or font_size >= body_font_size * 1.05
                    if not numbered_style:
                        first_alpha = ""
                    if not title_candidate or not title_candidate[0].isalpha():
                        first_alpha = ""
                    has_case = bool(first_alpha and first_alpha.lower() != first_alpha.upper())
                    if not first_alpha or (has_case and not first_alpha.isupper()) or re.search(r"[=±∑∫]", title_candidate) or title_candidate.endswith(".") or len(title_candidate) > 80 or len(title_candidate.split()) > 10:
                        numbered = None
                if not (explicit or numbered or known or typographic):
                    continue
                normalized = numbered.group(2).strip() if numbered else line.strip().rstrip(".:")
                key = (normalized.casefold(), int(page.get("page") or 0), str(block.get("id") or ""), line_index)
                if normalized and key not in seen:
                    seen.add(key)
                    headings.append({
                        "id": f"section_{len(headings) + 1:03d}",
                        "title": normalized,
                        "page": page["page"],
                        "block_id": block["id"],
                        "line_index": line_index,
                        "_candidate_type": "explicit" if explicit else "numbered" if numbered else "known" if known else "typographic",
                        "_bbox": block.get("bbox"),
                        "_font_size": font_size,
                        "_reading_order": int(block.get("reading_order") or 0),
                    })
                if body_anchor is None and (explicit or numbered or known):
                    body_anchor = (int(page["page"]), block_y)
                body_started = body_started or bool(explicit or numbered or known)
    merged: list[dict[str, Any]] = []
    for heading in headings:
        previous = merged[-1] if merged else None
        previous_bbox = previous.get("_bbox") if previous else None
        current_bbox = heading.get("_bbox")
        same_multiline_heading = bool(
            previous
            and previous.get("_candidate_type") == "typographic"
            and heading.get("_candidate_type") == "typographic"
            and int(previous.get("page") or 0) == int(heading.get("page") or 0)
            and int(heading.get("_reading_order") or 0) == int(previous.get("_reading_order") or 0) + 1
            and previous_bbox
            and current_bbox
            and float(current_bbox[1]) - float(previous_bbox[3]) <= max(float(previous.get("_font_size") or 0), float(heading.get("_font_size") or 0), 1.0) * 0.8
            and len(f"{previous.get('title', '')} {heading.get('title', '')}") <= 140
        )
        if same_multiline_heading:
            previous["title"] = _clean(f"{previous.get('title', '')} {heading.get('title', '')}").rstrip(".:")
            previous.setdefault("continuation_block_ids", []).append(str(heading.get("block_id") or ""))
            previous["_bbox"] = [
                min(float(previous_bbox[0]), float(current_bbox[0])),
                min(float(previous_bbox[1]), float(current_bbox[1])),
                max(float(previous_bbox[2]), float(current_bbox[2])),
                max(float(previous_bbox[3]), float(current_bbox[3])),
            ]
            previous["_reading_order"] = int(heading.get("_reading_order") or 0)
            continue
        merged.append(heading)
    for index, heading in enumerate(merged, 1):
        heading["id"] = f"section_{index:03d}"
    return [{key: value for key, value in heading.items() if not key.startswith("_")} for heading in merged[:80]]


def _document_index(pages: list[dict[str, Any]], sections: list[dict[str, Any]]) -> dict[str, Any]:
    figures = []
    tables = []
    formulas = []
    for page in pages:
        page_blocks = page.get("blocks", [])
        for block_index, block in enumerate(page_blocks):
            record = {"page": page["page"], "block_id": block["id"], "bbox": block.get("bbox"), "caption": block.get("text", "")}
            if block.get("kind") == "caption":
                parent = block.get("parent_block")
                continuation_blocks = []
                for following in page_blocks[block_index + 1:]:
                    if parent is None or following.get("parent_block") != parent or _caption_prefix_kind(str(following.get("text") or "")):
                        break
                    continuation_blocks.append(following)

                # A caption may continue in the next column at the same vertical
                # position while PyMuPDF assigns a different parent block. Recover
                # those lines by geometry and font, then keep column-major order.
                start_bbox = block.get("bbox") if isinstance(block.get("bbox"), list) else [0, 0, 0, 0]
                start_y = float(start_bbox[1] or 0)
                start_font = float(block.get("font_size") or 0)
                page_height = max(1.0, float(page.get("height") or 1))
                max_y = start_y + max(42.0, page_height * 0.09)
                included_ids = {str(block.get("id") or ""), *(str(item.get("id") or "") for item in continuation_blocks)}
                for candidate in page_blocks:
                    candidate_id = str(candidate.get("id") or "")
                    if not candidate_id or candidate_id in included_ids:
                        continue
                    if int(candidate.get("column") if candidate.get("column") is not None else -99) == int(block.get("column") if block.get("column") is not None else -98):
                        continue
                    bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), list) else [0, 0, 0, 0]
                    candidate_y = float(bbox[1] or 0)
                    candidate_font = float(candidate.get("font_size") or 0)
                    if not (start_y - 2.0 <= candidate_y <= max_y):
                        continue
                    if start_font and candidate_font and abs(candidate_font - start_font) > 0.8:
                        continue
                    if _caption_prefix_kind(str(candidate.get("text") or "")):
                        continue
                    continuation_blocks.append(candidate)
                    included_ids.add(candidate_id)
                if continuation_blocks:
                    continuation_blocks.sort(key=lambda item: (int(item.get("column") or 0), float((item.get("bbox") or [0, 0, 0, 0])[1]), float((item.get("bbox") or [0, 0, 0, 0])[0])))
                    record["caption"] = _clean(record["caption"] + " " + " ".join(str(item.get("text") or "") for item in continuation_blocks))
                figures.append({"id": f"figure_{len(figures) + 1:03d}", **record})
            elif block.get("kind") == "table":
                tables.append({"id": f"table_{len(tables) + 1:03d}", **record})
            elif block.get("kind") == "formula":
                formulas.append({"id": f"formula_{len(formulas) + 1:03d}", "page": page["page"], "block_id": block["id"], "bbox": block.get("bbox"), "text": block.get("text", "")})
    page_map = {int(page["page"]): page for page in pages}
    for table in tables:
        page = page_map.get(int(table.get("page") or 0))
        if not page:
            continue
        blocks = page.get("blocks", [])
        caption_index = next((index for index, block in enumerate(blocks) if str(block.get("id") or "") == str(table.get("block_id") or "")), None)
        if caption_index is None:
            continue
        caption_bbox = table.get("bbox") or [0, 0, 0, 0]
        candidates = []
        for block in blocks[caption_index + 1:]:
            bbox = block.get("bbox")
            text = _clean(block.get("text", ""))
            if not bbox or not text:
                continue
            if str(block.get("kind") or "") in {"heading", "caption", "table", "formula"}:
                break
            if float(bbox[1]) > float(caption_bbox[3]) + min(300.0, float(page.get("height") or 800.0) * 0.35):
                break
            if len(text) > 60:
                break
            candidates.append(block)
        if len(candidates) < 4:
            continue
        heights = sorted(max(1.0, float(block["bbox"][3]) - float(block["bbox"][1])) for block in candidates)
        tolerance = max(5.0, heights[len(heights) // 2] * 0.75)
        row_groups: list[list[dict[str, Any]]] = []
        for block in sorted(candidates, key=lambda item: ((float(item["bbox"][1]) + float(item["bbox"][3])) / 2.0, float(item["bbox"][0]))):
            center = (float(block["bbox"][1]) + float(block["bbox"][3])) / 2.0
            if not row_groups:
                row_groups.append([block])
                continue
            previous_center = sum((float(item["bbox"][1]) + float(item["bbox"][3])) / 2.0 for item in row_groups[-1]) / len(row_groups[-1])
            if abs(center - previous_center) <= tolerance:
                row_groups[-1].append(block)
            else:
                row_groups.append([block])
        if len(row_groups) < 2 or len(row_groups[0]) < 2:
            continue
        header = sorted(row_groups[0], key=lambda item: float(item["bbox"][0]))
        anchors = [(float(item["bbox"][0]) + float(item["bbox"][2])) / 2.0 for item in header]
        columns = [_clean(item.get("text", "")) for item in header]
        rows = []
        cell_records = []
        for row_index, group in enumerate(row_groups):
            values: list[str | None] = [None] * len(columns)
            block_ids: list[str | None] = [None] * len(columns)
            for block in sorted(group, key=lambda item: float(item["bbox"][0])):
                center = (float(block["bbox"][0]) + float(block["bbox"][2])) / 2.0
                column_index = min(range(len(anchors)), key=lambda index: abs(center - anchors[index]))
                value = _clean(block.get("text", ""))
                values[column_index] = f"{values[column_index]} {value}".strip() if values[column_index] else value
                block_ids[column_index] = str(block.get("id") or "")
                cell_records.append({"row": row_index, "column": column_index, "text": value, "block_id": block.get("id"), "bbox": block.get("bbox")})
            if row_index:
                rows.append({"values": values, "block_ids": block_ids})
        header_signal = any(
            _cjk_character_count(value) > 0
            or (value and value[0].isalpha() and value[0].isupper())
            or value.isupper()
            or bool(re.search(r"(?:[A-Za-z].*\d|\d.*[A-Za-z])", value))
            for value in columns
        )
        if len(rows) < 2 or not header_signal:
            continue
        cell_block_ids = {str(record.get("block_id") or "") for record in cell_records}
        for block in candidates:
            if str(block.get("id") or "") in cell_block_ids:
                block["kind"] = "table_cell"
        all_boxes = [caption_bbox] + [block.get("bbox") for block in candidates if block.get("bbox")]
        table.update({
            "columns": columns,
            "rows": rows,
            "cells": cell_records,
            "bbox": [
                min(float(box[0]) for box in all_boxes),
                min(float(box[1]) for box in all_boxes),
                max(float(box[2]) for box in all_boxes),
                max(float(box[3]) for box in all_boxes),
            ],
        })
    return {"summary": "Page-aware section, formula, table-caption, and figure-caption index.", "sections": sections, "formulas": formulas[:120], "tables": tables[:80], "figures": figures[:80]}


def _section_coverage(pages: list[dict[str, Any]], sections: list[dict[str, Any]]) -> dict[str, bool]:
    candidates = ("\n".join(item["title"] for item in sections) + "\n" + "\n".join(page.get("text", "") for page in pages)).casefold()
    compact = _normalized_compare_text(candidates)
    coverage = {
        name: any(re.search(rf"\b{re.escape(alias)}\b", candidates) or (_normalized_compare_text(alias) and _normalized_compare_text(alias) in compact) for alias in aliases)
        for name, aliases in SECTION_ALIASES.items()
    }
    # Nature-style articles and several journals print a lead abstract without
    # an explicit "Abstract" heading. Treat a substantial first-page research
    # summary as abstract coverage when it contains an authorial contribution
    # statement, instead of emitting a false extraction warning.
    first_page = str(pages[0].get("text") or "") if pages else ""
    first_page_compact = _normalized_compare_text(first_page)
    implicit_abstract = bool(
        len(first_page_compact) >= 700
        and re.search(
            r"\b(?:here|in this (?:work|paper|study))\s+(?:we\s+)?(?:present|introduce|propose|develop|show|demonstrate)\b"
            r"|\bwe\s+(?:present|introduce|propose|develop|show|demonstrate)\b",
            first_page,
            re.IGNORECASE,
        )
    )
    coverage["abstract"] = bool(coverage.get("abstract") or implicit_abstract)
    return coverage


def _ocr_latin_lexical_quality(pages: list[dict[str, Any]]) -> dict[str, Any]:
    text = " ".join(str(page.get("text") or "") for page in pages if page.get("used_ocr"))
    anchors = re.findall(r"\b(?:abstract|introduction|method|methods|experiments|results|conclusion|discussion)\b", text, re.IGNORECASE)
    tokens = [token for token in re.findall(r"[A-Za-z]{3,}", text) if not token.isupper()]
    if len(tokens) < 20 or len(anchors) < 2:
        return {"applied": False, "token_count": len(tokens), "known_word_ratio": None, "english_anchor_count": len(anchors)}
    try:
        import wordninja

        known_words = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost
    except Exception:
        return {"applied": False, "token_count": len(tokens), "known_word_ratio": None, "english_anchor_count": len(anchors)}
    known = sum(token.casefold() in known_words or token.casefold() in OCR_PROTECTED_WORDS for token in tokens)
    return {
        "applied": True,
        "token_count": len(tokens),
        "known_word_ratio": round(known / max(1, len(tokens)), 4),
        "english_anchor_count": len(anchors),
    }


def evidence_id(page: int, text: str, bbox: Any, kind: str) -> str:
    payload = json.dumps({"page": page, "text": _clean(text), "bbox": bbox, "kind": kind}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"E_P{int(page):03d}_{hashlib.sha1(payload).hexdigest()[:12]}"


def _build_evidence(pages: list[dict[str, Any]], sections: list[dict[str, Any]], document_index: dict[str, Any], max_chars: int) -> list[dict[str, Any]]:
    section_positions = sorted(sections, key=lambda item: (item["page"], item.get("block_id", "")))
    evidence: list[dict[str, Any]] = []
    used = 0
    for figure in document_index.get("figures", []):
        text = _clean(figure.get("caption", ""))
        if not text or used + len(text) > max_chars:
            continue
        evidence.append({
            "id": evidence_id(int(figure.get("page") or 0), text, figure.get("bbox"), "caption"),
            "page": figure.get("page"),
            "block_id": figure.get("block_id"),
            "bbox": figure.get("bbox"),
            "section_hint": "Figure Captions",
            "kind": "caption",
            "source": "pymupdf",
            "confidence": 1.0,
            "text": text,
            "char_count": len(text),
        })
        used += len(text)

    for table in document_index.get("tables", []):
        lines = [_clean(table.get("caption", ""))]
        columns = [str(value or "") for value in table.get("columns", [])]
        if columns:
            lines.append("Columns: " + " | ".join(columns))
        for row in table.get("rows", []):
            values = [str(value or "") for value in row.get("values", [])]
            lines.append("Row: " + " | ".join(values))
        text = _clean("\n".join(value for value in lines if value))
        if not text or used + len(text) > max_chars:
            continue
        evidence.append({
            "id": evidence_id(int(table.get("page") or 0), text, table.get("bbox"), "table"),
            "page": table.get("page"),
            "block_id": table.get("block_id"),
            "bbox": table.get("bbox"),
            "section_hint": "Table Captions",
            "kind": "table",
            "source": "structured-table",
            "confidence": 1.0,
            "text": text,
            "char_count": len(text),
        })
        used += len(text)

    block_candidates: list[dict[str, Any]] = []
    for page in pages:
        for block in page.get("blocks", []):
            text = _clean(block.get("text", ""))
            if not text or block.get("kind") in {"caption", "table", "table_cell"}:
                continue
            block_id = str(block.get("id") or "")
            section = next(
                (
                    item["title"]
                    for item in reversed(section_positions)
                    if (int(item["page"]), str(item.get("block_id") or "")) <= (int(page["page"]), block_id)
                ),
                None,
            )
            block_candidates.append({
                "id": evidence_id(int(page["page"]), text, block.get("bbox"), str(block.get("kind") or "paragraph")),
                "page": page["page"],
                "block_id": block_id,
                "bbox": block.get("bbox"),
                "section_hint": section,
                "kind": block.get("kind"),
                "source": block.get("source"),
                "confidence": block.get("confidence", 1.0),
                "text": text,
                "char_count": len(text),
            })

    selected_ids: set[str] = set()
    page_groups: dict[int, list[dict[str, Any]]] = {}
    for item in block_candidates:
        page_groups.setdefault(int(item["page"]), []).append(item)

    def add(item: dict[str, Any], max_length: int | None = None) -> bool:
        nonlocal used
        item_id = str(item["id"])
        if item_id in selected_ids or used >= max_chars:
            return False
        remaining = max_chars - used
        if remaining <= 0:
            return False
        value = dict(item)
        if max_length is not None and len(value["text"]) > max_length:
            value["text"] = value["text"][:max_length].rstrip()
            value["char_count"] = len(value["text"])
            value["id"] = evidence_id(int(value["page"]), value["text"], value.get("bbox"), str(value.get("kind") or "paragraph"))
        if len(value["text"]) > remaining:
            if remaining < 40:
                return False
            value["text"] = value["text"][:remaining].rstrip()
            value["char_count"] = len(value["text"])
            value["id"] = evidence_id(int(value["page"]), value["text"], value.get("bbox"), str(value.get("kind") or "paragraph"))
            item_id = str(value["id"])
        evidence.append(value)
        selected_ids.add(str(item["id"]))
        selected_ids.add(item_id)
        used += len(value["text"])
        return True

    # Reserve a representative block for every page before spending the budget
    # on long early sections. This keeps conclusions and appendices discoverable.
    representative_quota = max(80, min(600, max(1, max_chars - used) // max(1, len(page_groups))))
    for page_number in sorted(page_groups):
        candidates = page_groups[page_number]
        representative = max(
            candidates,
            key=lambda item: (
                1 if item.get("kind") in {"paragraph", "title"} else 0,
                min(len(str(item.get("text") or "")), 600),
            ),
        )
        add(representative, max_length=representative_quota)

    section_priority = {
        "abstract": 0,
        "conclusion": 1,
        "conclusions": 1,
        "discussion": 1,
        "method": 2,
        "methods": 2,
        "methodology": 2,
        "approach": 2,
        "architecture": 2,
        "framework": 2,
        "system overview": 2,
        "introduction": 3,
        "background": 3,
        "experiment": 4,
        "experiments": 4,
        "evaluation": 4,
        "results": 4,
    }

    def priority(item: dict[str, Any]) -> tuple[int, int, str]:
        section = str(item.get("section_hint") or "").casefold()
        text = str(item.get("text") or "")
        topology_definition = bool(re.search(
            r"\b(?:has|have|consists? of|comprises?|contains?)\b.{0,80}\b(?:stages?|components?|modules?|steps?)\b"
            r"|\b(?:assisted[ -]manual|semi[ -]automatic|fully automatic)\b"
            r"|\b(?:input|image|text|query|prompt)\b.{0,100}\b(?:encoder|decoder|retriever|generator)\b"
            r"|\b(?:encoder|decoder|retriever|generator)\b.{0,100}\b(?:output|prediction|representation|embedding)\b",
            text,
            re.IGNORECASE,
        ))
        rank = min((value for name, value in section_priority.items() if name in section), default=5)
        if topology_definition:
            rank = -1
        if item.get("kind") == "heading":
            rank = min(rank, 2)
        return rank, int(item.get("page") or 0), str(item.get("block_id") or "")

    for item in sorted(block_candidates, key=priority):
        add(item)

    document_order = {
        str(item["id"]): index
        for index, item in enumerate(block_candidates)
    }
    caption_count = sum(1 for item in evidence if item.get("kind") == "caption")
    caption_items = evidence[:caption_count]
    body_items = sorted(evidence[caption_count:], key=lambda item: (int(item.get("page") or 0), document_order.get(str(item.get("id")), 10**9), str(item.get("block_id") or "")))
    evidence = caption_items + body_items
    for index, item in enumerate(evidence, 1):
        item["legacy_id"] = f"E{index:04d}"
    return evidence


def _extraction_report(source: Path, pages: list[dict[str, Any]], index: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    readable = [page for page in pages if len(_normalized_compare_text(page.get("text", ""))) >= 80]
    readable_ratio = round(len(readable) / max(1, len(pages)), 4)
    ocr_count = sum(1 for page in pages if page.get("used_ocr"))
    candidate_count = len(details.get("ocr_candidate_pages", []))
    fully_scanned_source = bool(pages and candidate_count == len(pages))
    partial_scan = bool(fully_scanned_source and 0 < ocr_count < len(pages))
    pdf_type = "scanned" if fully_scanned_source else "mixed" if ocr_count and ocr_count < len(pages) else "scanned" if ocr_count else "born_digital"
    coverage = _section_coverage(pages, index.get("sections", []))
    bold_block_ids = {(int(page["page"]), str(block.get("id") or "")) for page in pages for block in page.get("blocks", []) if block.get("font_bold")}
    warnings = list(details.get("warnings", []))
    ocr_blocks = [block for page in pages if page.get("used_ocr") for block in page.get("blocks", []) if isinstance(block, dict)]
    ocr_confidences = [float(block.get("confidence") or 0.0) for block in ocr_blocks]
    mean_ocr_confidence = round(sum(ocr_confidences) / max(1, len(ocr_confidences)), 4) if ocr_blocks else None
    short_ocr_blocks = [block for block in ocr_blocks if len(_normalized_compare_text(str(block.get("text") or ""))) < 3]
    short_ocr_block_ratio = round(len(short_ocr_blocks) / max(1, len(ocr_blocks)), 4) if ocr_blocks else None
    lexical_quality = _ocr_latin_lexical_quality(pages)
    sampled_scan_ready = bool(
        partial_scan
        and len(readable) >= min(3, len(pages))
        and coverage.get("abstract")
        and coverage.get("method")
        and mean_ocr_confidence is not None
        and mean_ocr_confidence >= 0.75
    )
    agreements = [page["poppler_agreement"] for page in pages if isinstance(page.get("poppler_agreement"), (int, float))]
    extractor_agreement = round(sum(agreements) / len(agreements), 4) if agreements else None
    extracted_chars = sum(len(_normalized_compare_text(page.get("text", ""))) for page in pages)
    # A paper whose appendix is a run of full-page figures carries almost no text
    # on those pages, and counting them made a readable paper look unreadable:
    # ICCV2025:tab transformer attention bottlenecks has 21 such pages, and the
    # ratio read 0.52 although every section was located, 66k characters were
    # extracted, and both extractors agreed on 98% of it. Four papers of the
    # measured wave were dropped this way. The gate exists to catch documents that
    # could not be read; a document whose method, experiments and conclusion
    # were located, carrying substantial text under agreeing extractors, has
    # been read. A Nature article can print the lead summary without the word
    # Abstract. When the pages that carry no text are a trailing figure
    # appendix, that missing heading does not mean the body was missed.
    readable_pages = {int(page.get("page") or 0) for page in readable}
    quiet_pages = [int(page.get("page") or 0) for page in pages if int(page.get("page") or 0) not in readable_pages]
    trailing_figure_appendix = bool(readable_pages and quiet_pages and min(quiet_pages) > max(readable_pages))
    body_fully_located = bool(
        len(readable) >= 3
        and extracted_chars >= 20000
        and coverage.get("method")
        and coverage.get("experiments")
        and coverage.get("conclusion")
        and (coverage.get("abstract") or trailing_figure_appendix)
        and (extractor_agreement is None or extractor_agreement >= 0.9)
    )
    if readable_ratio < 0.6 and not sampled_scan_ready and not body_fully_located:
        warnings.append("fewer than 60% of pages contain reliable text")
    if readable_ratio < 0.6 and body_fully_located:
        warnings.append(
            f"{len(pages) - len(readable)} of {len(pages)} pages carry no text, which is the paper's "
            "figure appendix; the body was located in full"
        )
    if sampled_scan_ready:
        warnings.append("fully scanned document was only partially OCRed; semantic scope is limited to sampled pages")
    if mean_ocr_confidence is not None and mean_ocr_confidence < 0.5:
        warnings.append("mean OCR confidence is below 0.50")
    if short_ocr_block_ratio is not None and short_ocr_block_ratio > 0.3:
        warnings.append("OCR output is highly fragmented")
    if lexical_quality.get("applied") and float(lexical_quality.get("known_word_ratio") or 0.0) < 0.72:
        warnings.append("OCR English lexical plausibility is below 0.72")
    if not coverage.get("abstract") or not coverage.get("method"):
        warnings.append("abstract or method-like section was not confidently located")
    lexical_failure = bool(lexical_quality.get("applied") and float(lexical_quality.get("known_word_ratio") or 0.0) < 0.5)
    report_status = "fail" if (readable_ratio < 0.6 and not sampled_scan_ready and not body_fully_located) or (mean_ocr_confidence is not None and mean_ocr_confidence < 0.35) or lexical_failure else "warning" if warnings else "pass"
    return {
        "summary": "PDF extraction quality and fallback report.",
        "status": report_status,
        "source_path": str(source),
        "pdf_type": pdf_type,
        "page_count": len(pages),
        "readable_page_count": len(readable),
        "readable_page_ratio": readable_ratio,
        "body_fully_located": body_fully_located,
        "semantic_scope": "sampled_pages_only" if partial_scan else "full_document",
        "scientific_scope_complete": not partial_scan,
        "sampled_scan_ready": sampled_scan_ready,
        "empty_or_low_text_pages": [page["page"] for page in pages if page not in readable],
        "ocr_candidate_pages": details.get("ocr_candidate_pages", []),
        "ocr_priority_pages": details.get("ocr_priority_pages", []),
        "ocr_priority": details.get("ocr_priority", []),
        "ocr_pages": details.get("ocr_pages", []),
        "ocr_page_durations": details.get("ocr_page_durations", []),
        "ocr_attempted_pages": details.get("ocr_attempted_pages", []),
        "ocr_successful_attempt_pages": details.get("ocr_successful_attempt_pages", []),
        "ocr_schedule_complete": bool(details.get("ocr_schedule_complete", True)),
        "ocr_run_complete": bool(details.get("ocr_run_complete", True)),
        "ocr_worker_count": int(details.get("ocr_worker_count") or 1),
        "ocr_rescue_min_remaining_seconds": float(details.get("ocr_rescue_min_remaining_seconds") or 45.0),
        "repeated_margin_noise_removed_count": int(details.get("repeated_margin_noise_removed_count") or 0),
        "native_hyphenation_repair_count": int(details.get("native_hyphenation_repair_count") or 0),
        "ocr_margin_noise_removed_count": sum(int(item.get("margin_noise_removed") or 0) for item in details.get("ocr_page_durations", [])),
        "ocr_spacing_repair_count": sum(int(item.get("spacing_repairs") or 0) for item in details.get("ocr_page_durations", [])),
        "mean_ocr_confidence": mean_ocr_confidence,
        "short_ocr_block_ratio": short_ocr_block_ratio,
        "ocr_latin_lexical_gate_applied": bool(lexical_quality.get("applied")),
        "ocr_latin_token_count": int(lexical_quality.get("token_count") or 0),
        "ocr_latin_known_word_ratio": lexical_quality.get("known_word_ratio"),
        "ocr_english_anchor_count": int(lexical_quality.get("english_anchor_count") or 0),
        "replacement_character_rate": round(sum(page.get("replacement_character_rate", 0.0) for page in pages) / max(1, len(pages)), 6),
        "mojibake_rate": round(sum(page.get("mojibake_rate", 0.0) for page in pages) / max(1, len(pages)), 6),
        "cross_extractor_agreement": extractor_agreement,
        "reading_order_confidence": round(sum(page.get("reading_order_confidence", 0.0) for page in pages) / max(1, len(pages)), 4),
        "max_column_count": max((int(page.get("column_count") or 1) for page in pages), default=1),
        "multi_column_page_count": sum(int(page.get("column_count") or 1) > 1 for page in pages),
        "rotated_pages": [int(page["page"]) for page in pages if int(page.get("rotation") or 0) % 360 != 0],
        "section_coverage": coverage,
        "figure_caption_count": len(index.get("figures", [])),
        "table_caption_count": len(index.get("tables", [])),
        "formula_count": len(index.get("formulas", [])),
        "section_count": len(index.get("sections", [])),
        "typographic_heading_count": sum((int(item.get("page") or 0), str(item.get("block_id") or "")) in bold_block_ids for item in index.get("sections", [])),
        "merged_heading_line_count": sum(len(item.get("continuation_block_ids", [])) for item in index.get("sections", [])),
        "poppler_available": details.get("poppler_available", False),
        "warnings": warnings,
        "elapsed_seconds": details.get("elapsed_seconds", 0.0),
    }


def parse_paper(
    path: str | Path,
    max_chars: int = 90000,
    deadline_at: float | None = None,
    ocr_engine: str = "off",
    ocr_lang: str = "en_ch",
    ocr_adapter: Callable | None = None,
    max_ocr_pages: int = 6,
    ocr_rescue_min_remaining: float = 45.0,
) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Paper does not exist: {source}")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source.suffix.lower() == ".pdf" or _has_pdf_header(source):
        _validate_pdf_container(source)
        pages, details = _read_pdf_pages(source, max_chars, deadline_at, ocr_engine, ocr_lang, ocr_adapter, max_ocr_pages, ocr_rescue_min_remaining)
        loader = "structured-pdf"
    else:
        pages, loader = _read_non_pdf_pages(source, max_chars)
        details = {"warnings": [], "ocr_candidate_pages": [], "ocr_pages": [], "poppler_available": False, "elapsed_seconds": 0.0}
    sections = _heading_candidates(pages)
    section_blocks = {
        (int(item.get("page") or 0), block_id)
        for item in sections
        for block_id in [str(item.get("block_id") or ""), *(str(value) for value in item.get("continuation_block_ids", []))]
        if block_id
    }
    for page in pages:
        for block in page.get("blocks", []):
            if (int(page["page"]), str(block.get("id") or "")) in section_blocks:
                block["kind"] = "heading"
    document_index = _document_index(pages, sections)
    evidence = _build_evidence(pages, sections, document_index, max_chars)
    all_text = "\n\n".join(page.get("text", "") for page in pages)
    extraction_report = _extraction_report(source, pages, document_index, details)
    evidence_pages = {int(item.get("page") or 0) for item in evidence if item.get("page")}
    extraction_report["evidence_char_count"] = sum(int(item.get("char_count") or 0) for item in evidence)
    extraction_report["evidence_page_count"] = len(evidence_pages)
    extraction_report["evidence_page_coverage_ratio"] = round(len(evidence_pages) / max(1, len(pages)), 4)
    return {
        "summary": "Paper parsed into a structured, page-aware evidence model.",
        "source_path": str(source),
        "source_name": source.name,
        "source_type": source.suffix.lower(),
        "source_sha256": source_hash,
        "loader": loader,
        "page_count": len(pages),
        "char_count": len(all_text),
        "headings": [item["title"] for item in sections],
        "sections": sections,
        "pages": pages,
        "document_index": document_index,
        "evidence": evidence,
        "extraction_report": extraction_report,
    }


def paper_markdown(parsed: dict[str, Any]) -> str:
    parts = []
    for page in parsed.get("pages", []):
        parts.append(f"<!-- page {page['page']} -->\n\n{page.get('text', '')}")
    return "\n\n".join(parts).strip() + "\n"


def _overview_figure_evidence_priority(parsed: dict[str, Any], limit: int = 2) -> dict[str, tuple[int, int]]:
    """Prioritize labels spatially inside likely overview figures.

    Figure-local labels are often short text blocks with no section heading, so
    a section-first evidence sort can push them past the fast planner's character
    budget.  Recover the visual region around the strongest overview captions
    and rank those blocks immediately after their caption.
    """
    positive = re.compile(r"\b(overview|framework|architecture|pipeline|procedure|approach|method|model|system|workflow)\b", re.IGNORECASE)
    negative = re.compile(r"\b(comparison|performance|result|ablation|visualization|qualitative|attention map)\b", re.IGNORECASE)
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, figure in enumerate(parsed.get("document_index", {}).get("figures", [])):
        if not isinstance(figure, dict):
            continue
        caption = str(figure.get("caption") or "").strip()
        if not caption or not positive.search(caption):
            continue
        score = 8
        if re.match(r"^(?:figure|fig\.)\s*[12]\b", caption, re.IGNORECASE):
            score += 5
        if 160 <= len(caption) <= 1600:
            score += 4
        if negative.search(caption):
            score -= 7
        ranked.append((score, -index, figure))
    candidates = [item for _, _, item in sorted(ranked, reverse=True)[: max(1, int(limit))]]
    if not candidates:
        return {}

    pages = {
        int(page.get("page") or 0): page
        for page in parsed.get("pages", [])
        if isinstance(page, dict) and page.get("page")
    }
    evidence = [item for item in parsed.get("evidence", []) if isinstance(item, dict)]
    priority: dict[str, tuple[int, int]] = {}
    for candidate_index, figure in enumerate(candidates):
        page_number = int(figure.get("page") or 0)
        page = pages.get(page_number, {})
        page_height = float(page.get("height") or 0.0)
        caption = str(figure.get("caption") or "")
        caption_record = next(
            (
                item
                for item in evidence
                if int(item.get("page") or 0) == page_number
                and str(item.get("kind") or "").casefold() == "caption"
                and str(item.get("text") or "").startswith(caption[:60])
            ),
            None,
        )
        if not caption_record:
            continue
        caption_id = str(caption_record.get("id") or "")
        if caption_id:
            priority[caption_id] = (candidate_index * 2, -1)
        caption_bbox = caption_record.get("bbox")
        if not (isinstance(caption_bbox, list) and len(caption_bbox) == 4 and page_height > 0):
            continue
        caption_top = float(caption_bbox[1])
        caption_bottom = float(caption_bbox[3])
        caption_center_ratio = ((caption_top + caption_bottom) / 2.0) / page_height
        for item in evidence:
            if int(item.get("page") or 0) != page_number:
                continue
            item_id = str(item.get("id") or "")
            item_bbox = item.get("bbox")
            if not item_id or not (isinstance(item_bbox, list) and len(item_bbox) == 4):
                continue
            item_top = float(item_bbox[1])
            item_bottom = float(item_bbox[3])
            if caption_center_ratio >= 0.38:
                in_visual_region = item_bottom <= caption_bottom + 6.0 and item_top >= max(0.0, caption_top - 0.62 * page_height)
            elif caption_center_ratio <= 0.25:
                in_visual_region = item_top >= caption_top - 6.0 and item_bottom <= min(page_height, caption_bottom + 0.62 * page_height)
            else:
                in_visual_region = item_bottom >= caption_top - 0.42 * page_height and item_top <= caption_bottom + 0.42 * page_height
            if not in_visual_region:
                continue
            block_id = str(item.get("block_id") or "")
            reading_order = next(
                (
                    int(block.get("reading_order") or 0)
                    for block in page.get("blocks", [])
                    if isinstance(block, dict) and str(block.get("id") or "") == block_id
                ),
                0,
            )
            value = (candidate_index * 2 + 1, reading_order)
            if item_id not in priority or value < priority[item_id]:
                priority[item_id] = value
    return priority


def overview_figure_spatial_excerpt(parsed: dict[str, Any], max_items: int = 180) -> str:
    """Return compact bbox-aware evidence for likely overview figure regions."""
    priority = _overview_figure_evidence_priority(parsed)
    if not priority:
        return "[]"
    pages = {
        int(page.get("page") or 0): page
        for page in parsed.get("pages", [])
        if isinstance(page, dict) and page.get("page")
    }
    records = []
    for item in parsed.get("evidence", []):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "")
        if item_id not in priority:
            continue
        page_number = int(item.get("page") or 0)
        page = pages.get(page_number, {})
        width = max(1.0, float(page.get("width") or 1.0))
        height = max(1.0, float(page.get("height") or 1.0))
        bbox = item.get("bbox") if isinstance(item.get("bbox"), list) and len(item.get("bbox")) == 4 else None
        bbox_percent = None
        if bbox:
            bbox_percent = [
                round(float(bbox[0]) / width, 4),
                round(float(bbox[1]) / height, 4),
                round(float(bbox[2]) / width, 4),
                round(float(bbox[3]) / height, 4),
            ]
        records.append({
            "id": item_id,
            "page": page_number,
            "block_id": str(item.get("block_id") or ""),
            "kind": str(item.get("kind") or "paragraph"),
            "reading_order": priority[item_id][1],
            "bbox_percent": bbox_percent,
            "text": str(item.get("text") or ""),
            "_priority": priority[item_id],
        })
    records.sort(key=lambda item: (item["_priority"], item["page"], item["block_id"]))
    for item in records:
        item.pop("_priority", None)
    return json.dumps(records[: max(1, int(max_items))], ensure_ascii=False, separators=(",", ":"))


_PANEL_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|[.;])\s*([a-h])\s*[,.:]\s*", re.IGNORECASE),
    re.compile(r"\(([a-h])\)\s*[,.:)]?\s*"),
    re.compile(r"\((i|ii|iii|iv|v|vi|vii|viii)\)\s*[,.:)]?\s*"),
)

_EXAMPLE_TRIPLE_PATTERN = re.compile(r"^\([^()]{0,60}(?:,[^()]{0,60}){1,4}\)$")


def _panel_marker_matches(caption: str) -> list[tuple[int, int, str]]:
    """Return ordered, non-overlapping panel markers declared in a figure caption.

    Captions declare panels in several conventions: ``a, ... b, ...``,
    ``(a) ... (b) ...``, and ``(i) ... (ii) ...``.  Only the first convention used
    to be recognised, so every paper that bracketed its panel letters produced an
    empty panel inventory even though the figure itself was extracted intact.
    Parenthesised forms stay case-sensitive so that symbol parenthesis inside a
    caption, such as ``Global Memory (M)``, never registers as a panel marker.
    """
    found: list[tuple[int, int, str]] = []
    for pattern in _PANEL_MARKER_PATTERNS:
        for match in pattern.finditer(caption):
            start, end = match.span(1)
            marker_end = match.end()
            label = match.group(1).casefold()
            if any(start < other_end and other_start < end for other_start, other_end, _ in found):
                continue
            if any(label == existing for _, _, existing in found):
                continue
            found.append((start, marker_end, label))
    found.sort()
    return found


def _figure_label_like(text_value: str) -> bool:
    """Return True when a figure-local block reads like a visible label.

    Example triples such as ``(Newton, Birthyear, 1645)`` and ellipsis runs are
    illustration content rather than node labels, so they are kept out of the
    required-candidate checklist.
    """
    value = text_value.strip()
    if not value or _EXAMPLE_TRIPLE_PATTERN.match(value):
        return False
    if not re.search(r"[A-Za-z]", value):
        return False
    if len(value) > 72 or len(value.split()) > 10:
        return False
    return True


def _match_description_anchor(description: str, records: list[dict[str, Any]], min_tokens: int = 2) -> dict[str, Any] | None:
    """Find the figure block that repeats a declared panel's caption description."""
    desc_tokens = re.findall(r"[a-z0-9]+", description.casefold())
    if len(desc_tokens) < min_tokens:
        return None
    best: tuple[tuple[float, int, int], dict[str, Any]] | None = None
    for item in records:
        tokens = re.findall(r"[a-z0-9]+", str(item.get("text") or "").casefold())
        if len(tokens) < min_tokens or len(tokens) > 8:
            continue
        common = 0
        for left, right in zip(tokens, desc_tokens):
            if left != right:
                break
            common += 1
        if common < min_tokens or common / len(tokens) < 0.6:
            continue
        key = (common / len(tokens), common, -abs(len(tokens) - len(desc_tokens)))
        if best is None or key > best[0]:
            best = (key, item)
    return best[1] if best else None


def _axis_cluster_cuts(centers: Iterable[float], count: int, iterations: int = 12) -> list[float]:
    """Cut one axis into ``count`` deterministic bands with 1-D k-means.

    Panel columns of a wide figure overlap on the horizontal axis, so a
    largest-gap split lands inside a column.  Clustering the block centers keeps
    each column on one side of the cut even when the gap between them is small.
    """
    values = sorted(float(value) for value in centers)
    if count < 2 or len(values) < count:
        return []
    seeds = [values[0]]
    while len(seeds) < count:
        seeds.append(max(values, key=lambda value: min(abs(value - seed) for seed in seeds)))
    seeds.sort()
    for _ in range(max(1, int(iterations))):
        groups = [[] for _ in seeds]
        for value in values:
            index = min(range(len(seeds)), key=lambda slot: abs(value - seeds[slot]))
            groups[index].append(value)
        centers_next = [
            (sum(group) / len(group)) if group else seeds[index]
            for index, group in enumerate(groups)
        ]
        if centers_next == seeds:
            break
        seeds = centers_next
    cuts = [(seeds[index] + seeds[index + 1]) / 2.0 for index in range(len(seeds) - 1)]
    return sorted(cuts)


def _panel_bands(
    visual_records: list[dict[str, Any]],
    cuts: list[float],
    orientation: str,
    region: list[float],
) -> list[list[float]]:
    """Split a figure region into panel bands around the given cut coordinates."""
    if orientation == "vertical":
        start_index, end_index, region_start, region_end = 1, 3, region[1], region[3]
    else:
        start_index, end_index, region_start, region_end = 0, 2, region[0], region[2]
    edges = [region_start] + sorted(cuts) + [region_end]
    bands: list[list[float]] = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        if high - low <= 0:
            continue
        if orientation == "vertical":
            bands.append([region[0], low, region[2], high])
        else:
            bands.append([low, region[1], high, region[3]])
    return bands


def _panel_inventory_for_figure(
    parsed: dict[str, Any],
    caption_record: dict[str, Any],
    region_records: list[dict[str, Any]],
    max_items_per_panel: int,
) -> dict[str, Any] | None:
    """Build the per-panel label checklist for one candidate overview figure."""
    def role_hint(text_value: str) -> str | None:
        low = text_value.casefold().strip(" .:")
        words = text_value.split()
        if len(text_value) > 72 or len(words) > 8 or low.startswith(("we ", "with ", "the vector ", "each node ")):
            return None
        if re.search(r"\b(loss|objective|optimization)\b", low):
            return "objective"
        if re.search(r"\b(predictions?|predicted .*distribution|outputs?|results?)\b", low) or re.fullmatch(r"y\s*test", low):
            return "output"
        # Figure authors often use a determiner-led dataset phrase as a heading
        # for a whole visual region (for example, "A synthetic dataset"), not
        # as a separate tensor/table node in the data-flow graph.  Keep explicit
        # operational labels such as "Input dataset" and X_train/Y_test as
        # inputs, while treating descriptive noun phrases as group containers.
        if re.fullmatch(r"(?:a|an|the)\s+(?:[\w-]+\s+){1,6}datasets?", low):
            return "group"
        if re.search(r"\b(dataset|input data|input dataset|source data)\b", low) or re.fullmatch(r"[xy]\s*(?:train|test)", low):
            return "input"
        if re.search(r"\b(layer|block|stack)\b", low) and re.search(r"\b\d+\s*[\u00d7x]\b|\(\d+", low):
            return "group"
        if len(words) <= 5 and re.search(r"\b(model|network|encoder|decoder|attention|mlp|transformer|retriever|generator|classifier|backbone|module|tabpfn)\b", low):
            return "module"
        if re.fullmatch(r"[A-Z][A-Z0-9-]{2,}", text_value.strip()):
            return "module"
        return None

    caption = str(caption_record.get("text") or "")
    matches = _panel_marker_matches(caption)
    panel_labels = [label for _, _, label in matches]
    if len(panel_labels) < 2:
        return None
    descriptions: dict[str, str] = {}
    for index, (_, match_end, label) in enumerate(matches):
        end = matches[index + 1][0] if index + 1 < len(matches) else len(caption)
        descriptions[label] = re.sub(r"\s+", " ", caption[match_end:end]).strip(" .")

    page_number = int(caption_record.get("page") or 0)
    page = next((item for item in parsed.get("pages", []) if isinstance(item, dict) and int(item.get("page") or 0) == page_number), {})
    width = max(1.0, float(page.get("width") or 1.0))
    height = max(1.0, float(page.get("height") or 1.0))
    caption_bbox = caption_record.get("bbox") if isinstance(caption_record.get("bbox"), list) and len(caption_record.get("bbox")) == 4 else [0.0, height, width, height]
    visual_records = [
        item for item in region_records
        if item is not caption_record
        and isinstance(item.get("bbox"), list)
        and len(item.get("bbox")) == 4
        and float(item["bbox"][3]) <= float(caption_bbox[3]) + 6.0
    ]
    if len(visual_records) < 2:
        return None

    anchors: list[tuple[str, dict[str, Any]]] = []
    for panel_label in panel_labels:
        candidates = [item for item in visual_records if _normalized_compare_text(str(item.get("text") or "")) == panel_label]
        if candidates:
            anchors.append((panel_label, min(candidates, key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])))))
            continue
        # Vector figures usually do not expose the panel glyph as its own text
        # block, but the panel's own title is printed inside the region and the
        # caption repeats it.  Match that title so the panel still gets an anchor.
        title = _match_description_anchor(descriptions.get(panel_label, ""), visual_records)
        if title is not None:
            anchors.append((panel_label, title))
    region = [
        min(float(item["bbox"][0]) for item in visual_records),
        min(float(item["bbox"][1]) for item in visual_records),
        max(float(item["bbox"][2]) for item in visual_records),
        min(float(caption_bbox[1]), max(float(item["bbox"][3]) for item in visual_records)),
    ]
    if len(anchors) >= 2:
        x_centers = [(float(item[1]["bbox"][0]) + float(item[1]["bbox"][2])) / 2.0 for item in anchors]
        y_centers = [(float(item[1]["bbox"][1]) + float(item[1]["bbox"][3])) / 2.0 for item in anchors]
        orientation = "vertical" if (max(y_centers) - min(y_centers)) >= (max(x_centers) - min(x_centers)) else "horizontal"
        # The anchors are the panels themselves, so the cuts belong between them.
        if orientation == "vertical":
            anchors = [item for _, item in sorted(zip(y_centers, anchors), key=lambda pair: pair[0])]
            centers = sorted(y_centers)
        else:
            anchors = [item for _, item in sorted(zip(x_centers, anchors), key=lambda pair: pair[0])]
            centers = sorted(x_centers)
        cuts = [(centers[index] + centers[index + 1]) / 2.0 for index in range(len(centers) - 1)]
        anchor_kind = "declared_marker"
    else:
        # The caption declares panels but the glyphs were not recovered as separate
        # text blocks (very common for vector figures).  Split the figure region on
        # its own layout gaps so a declared multi-panel figure still yields a
        # per-panel checklist instead of nothing.
        axis = 0 if (region[2] - region[0]) >= (region[3] - region[1]) else 1
        orientation = "horizontal" if axis == 0 else "vertical"
        panel_count = max(2, min(len(panel_labels), 4))
        centers = [
            (float(item["bbox"][axis]) + float(item["bbox"][axis + 2])) / 2.0
            for item in visual_records
        ]
        cuts = _axis_cluster_cuts(centers, panel_count)
        anchors = [(label, None) for label in panel_labels[: len(cuts) + 1]]
        anchor_kind = "layout_cluster"
    if len(anchors) >= 2:
        ordered_labels = [label for label, _ in anchors]
    else:
        ordered_labels = list(panel_labels)
    bands = _panel_bands(visual_records, cuts, orientation, region)
    if not bands:
        return None

    panels = []
    for index, band in enumerate(bands):
        panel_label = ordered_labels[index] if index < len(ordered_labels) else ""
        labels = []
        for item in visual_records:
            text_value = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
            if not text_value or text_value.casefold() in panel_labels or len(text_value) > 140:
                continue
            if sum(char.isalpha() for char in text_value) < 2:
                continue
            bbox = item["bbox"]
            center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
            center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
            if not (band[0] <= center_x <= band[2] and band[1] <= center_y <= band[3]):
                continue
            hint = role_hint(text_value)
            label_like = _figure_label_like(text_value)
            labels.append({
                "evidence_id": str(item.get("id") or ""),
                "text": text_value,
                "role_hint": hint if label_like else None,
                "required_candidate": bool(label_like),
                "bbox_percent": [
                    round(float(bbox[0]) / width, 4),
                    round(float(bbox[1]) / height, 4),
                    round(float(bbox[2]) / width, 4),
                    round(float(bbox[3]) / height, 4),
                ],
            })
        generic_module_labels = {"model", "network", "neuralnetwork", "encoder", "decoder"}
        for item in labels:
            if item.get("role_hint") != "module" or _normalized_compare_text(str(item.get("text") or "")) not in generic_module_labels:
                continue
            bbox = item.get("bbox_percent") or [0.0, 0.0, 0.0, 0.0]
            for other in labels:
                if other is item or other.get("role_hint") != "module":
                    continue
                other_key = _normalized_compare_text(str(other.get("text") or ""))
                if not other_key or other_key in generic_module_labels:
                    continue
                other_bbox = other.get("bbox_percent") or [0.0, 0.0, 0.0, 0.0]
                overlap = max(0.0, min(float(bbox[2]), float(other_bbox[2])) - max(float(bbox[0]), float(other_bbox[0])))
                min_width = max(1e-6, min(float(bbox[2]) - float(bbox[0]), float(other_bbox[2]) - float(other_bbox[0])))
                vertical_gap = float(bbox[1]) - float(other_bbox[3])
                if overlap / min_width >= 0.6 and -0.004 <= vertical_gap <= 0.018:
                    item["role_hint"] = "supporting"
                    item["required_candidate"] = False
                    break
        labels.sort(key=lambda item: (
            0 if item["required_candidate"] else 1,
            item["bbox_percent"][1],
            item["bbox_percent"][0],
        ))
        panels.append({
            "panel_label": panel_label,
            "description": descriptions.get(panel_label, ""),
            "page": page_number,
            "anchor": anchor_kind,
            "bbox_percent": [
                round(band[0] / width, 4),
                round(band[1] / height, 4),
                round(band[2] / width, 4),
                round(band[3] / height, 4),
            ],
            "label_inventory": labels[: max(1, int(max_items_per_panel))],
        })
    return {
        "page": page_number,
        "orientation": orientation,
        "anchor": anchor_kind,
        "caption": re.sub(r"\s+", " ", caption)[:240],
        "panels": panels,
    }


def overview_figure_panel_excerpt(parsed: dict[str, Any], max_items_per_panel: int = 48, max_figures: int = 3) -> str:
    """Group concise target-figure labels by explicit panel markers.

    Every candidate overview figure is inventoried, not just the single highest
    ranked one, because the planner is free to select a different caption; the
    inventory is therefore ready for whichever figure it picks.
    """
    priority = _overview_figure_evidence_priority(parsed, limit=max(1, int(max_figures)))
    if not priority:
        return "{}"
    by_id = {
        str(item.get("id") or ""): item
        for item in parsed.get("evidence", [])
        if isinstance(item, dict)
    }
    groups: dict[int, dict[str, Any]] = {}
    for item_id, (rank, _reading_order) in priority.items():
        record = by_id.get(item_id)
        if record is None:
            continue
        index = rank // 2
        entry = groups.setdefault(index, {"caption": None, "region": []})
        if rank % 2 == 0:
            entry["caption"] = record
        else:
            entry["region"].append(record)
    figures = []
    for index in sorted(groups):
        entry = groups[index]
        caption_record = entry["caption"]
        if not isinstance(caption_record, dict):
            continue
        figure = _panel_inventory_for_figure(parsed, caption_record, entry["region"], max_items_per_panel)
        if figure:
            figures.append(figure)
    if not figures:
        return "{}"
    panels = [panel for figure in figures for panel in figure["panels"]]
    return json.dumps({
        "summary": "Concise panel-local label inventory recovered from explicit figure panel markers.",
        "page": figures[0]["page"],
        "orientation": figures[0]["orientation"],
        "panels": panels,
        "figures": figures,
    }, ensure_ascii=False, separators=(",", ":"))


def evidence_excerpt(parsed: dict, max_chars: int = 58000) -> str:
    overview_priority = _overview_figure_evidence_priority(parsed)

    def priority_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
        item_id = str(item.get("id") or "")
        if item_id in overview_priority:
            local_tier, reading_order = overview_priority[item_id]
            return local_tier, reading_order, int(item.get("page") or 0), str(item.get("block_id") or "")
        tier = (
            10 if item.get("kind") == "caption" and re.match(r"^(figure|fig\.)\s*1\b", str(item.get("text") or ""), re.IGNORECASE) and re.search(r"\b(components?|overview|framework|architecture|pipeline)\b", str(item.get("text") or ""), re.IGNORECASE) else
            11 if item.get("kind") == "caption" and re.search(r"\b(overview|framework|architecture|pipeline)\b", str(item.get("text") or ""), re.IGNORECASE) else
            12 if item.get("kind") == "caption" and re.match(r"^(figure|fig\.)\s*1\b", str(item.get("text") or ""), re.IGNORECASE) else
            13 if item.get("kind") == "caption" else
            14 if any(term in str(item.get("section_hint") or "").casefold() for term in ("abstract", "introduction", "method", "approach", "architecture", "framework", "conclusion")) else 15
        )
        return tier, 0, int(item.get("page") or 0), str(item.get("block_id") or "")

    prioritized = sorted(
        parsed.get("evidence", []),
        key=priority_key,
    )
    lines: list[str] = []
    used = 0
    for item in prioritized:
        block = f"[{item['id']} | page {item['page']} | {item.get('section_hint') or 'unknown section'} | {item.get('block_id') or 'no block'}]\n{item['text']}"
        if used + len(block) > max_chars:
            continue
        lines.append(block)
        used += len(block)
    return "\n\n".join(lines)
