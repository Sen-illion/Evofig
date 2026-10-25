from __future__ import annotations

import io
import math
import time
from pathlib import Path
from typing import Any, Callable

from ..analysis.paper.document import parse_paper
from ..utils import ensure_dir, write_json
from .scanned_pdf import rasterize_pdf_as_scan


def _save_document(document: Any, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(target))
    document.close()
    return target


def _native_two_column_fixture(target: Path) -> Path:
    import fitz

    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((45, 48), "Abstract", fontname="hebo", fontsize=13)
    page.insert_textbox((45, 68, 555, 125), "A deterministic two-column extraction benchmark with enough scientific text for parser quality gates.", fontsize=10)
    page.insert_text((45, 155), "2 Method", fontname="hebo", fontsize=13)
    page.insert_textbox((45, 185, 275, 260), "LEFT_TOP encoder evidence begins the method flow and establishes the input representation.", fontsize=10)
    page.insert_textbox((45, 300, 275, 375), "LEFT_BOTTOM decoder evidence follows the encoder in the left reading column.", fontsize=10)
    page.insert_textbox((325, 185, 555, 260), "RIGHT_TOP retrieval evidence belongs after the complete left column.", fontsize=10)
    page.insert_textbox((325, 300, 555, 375), "RIGHT_BOTTOM output evidence finishes the two-column reading order.", fontsize=10)
    page.insert_textbox((45, 430, 555, 485), "Figure 1: Overview of the encoder, decoder, retrieval module, and final output.", fontsize=10)
    page.insert_text((45, 540), "3 Conclusion", fontname="hebo", fontsize=13)
    page.insert_textbox((45, 560, 555, 620), "The benchmark concludes with a complete evidence-grounded pipeline.", fontsize=10)
    return _save_document(document, target)


def _bold_heading_fixture(target: Path) -> Path:
    import fitz

    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((45, 50), "Abstract", fontname="hebo", fontsize=12)
    page.insert_textbox((45, 70, 555, 130), "A paper with unnumbered bold sections validates typography-aware section recovery.", fontsize=10)
    page.insert_text((45, 175), "Model Architecture", fontname="hebo", fontsize=11)
    page.insert_textbox((45, 195, 555, 260), "The architecture contains an image encoder, a prompt encoder, and a mask decoder.", fontsize=10)
    page.insert_text((45, 305), "Encoder and Decoder Stacks", fontname="hebo", fontsize=10)
    page.insert_textbox((45, 325, 555, 390), "The encoder representation conditions the decoder before the final prediction.", fontsize=10)
    page.insert_text((45, 435), "Conclusion", fontname="hebo", fontsize=11)
    page.insert_textbox((45, 455, 555, 520), "Typography-based boundaries preserve the scientific section context.", fontsize=10)
    return _save_document(document, target)


def _rotated_fixture(target: Path) -> Path:
    import fitz

    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((45, 50), "Abstract", fontname="hebo", fontsize=12)
    page.insert_textbox((45, 75, 555, 145), "A rotated scientific page must retain displayed coordinates and readable text.", fontsize=10)
    page.insert_text((45, 190), "2 Method", fontname="hebo", fontsize=12)
    page.insert_textbox((45, 215, 555, 290), "The rotated encoder sends its representation to a decoder and final output.", fontsize=10)
    page.insert_textbox((45, 335, 555, 390), "Figure 1: Rotated architecture overview with encoder and decoder stages.", fontsize=10)
    page.set_rotation(90)
    return _save_document(document, target)


def _mixed_scan_fixture(target: Path) -> Path:
    import fitz

    scan_source = fitz.open()
    scan_page = scan_source.new_page(width=600, height=800)
    scan_page.insert_text((50, 65), "2 Method", fontname="hebo", fontsize=18)
    scan_page.insert_textbox((50, 105, 550, 190), "The scanned encoder passes evidence to the scanned decoder and output module.", fontsize=15)
    scan_page.insert_textbox((50, 245, 550, 330), "Figure 1: Scanned architecture overview with encoder decoder and output.", fontsize=15)
    pixmap = scan_page.get_pixmap(dpi=180, alpha=False)
    scan_png = pixmap.tobytes("png")
    scan_source.close()

    document = fitz.open()
    page1 = document.new_page(width=600, height=800)
    page1.insert_text((45, 50), "Abstract", fontname="hebo", fontsize=12)
    page1.insert_textbox((45, 75, 555, 155), "A mixed PDF combines native text with one scanned method page for local OCR fallback.", fontsize=10)
    page2 = document.new_page(width=600, height=800)
    page2.insert_image(page2.rect, stream=scan_png)
    page3 = document.new_page(width=600, height=800)
    page3.insert_text((45, 50), "3 Conclusion", fontname="hebo", fontsize=12)
    page3.insert_textbox((45, 75, 555, 155), "The mixed extraction path preserves native pages and recovers the scanned method evidence.", fontsize=10)
    return _save_document(document, target)


def _repeated_margin_fixture(target: Path) -> Path:
    import fitz

    document = fitz.open()
    sections = [
        ("Abstract", "A document encoder grounds paper evidence before framework generation."),
        ("2 Method", "The document encoder sends grounded entities to a relation decoder and final output."),
        ("3 Experiments", "Experiments measure entity recall, relation recall, and extraction latency."),
        ("4 Conclusion", "The system preserves evidence citations and editable scientific labels."),
    ]
    for page_number, (heading, body) in enumerate(sections, 1):
        page = document.new_page(width=600, height=800)
        page.insert_text((45, 25), "Proceedings of the Example Conference 2026", fontsize=8)
        page.insert_text((45, 785), f"Anonymous Paper 1234 | Page {page_number}", fontsize=8)
        page.insert_text((45, 70), heading, fontname="hebo", fontsize=12)
        page.insert_textbox((45, 95, 555, 220), body, fontsize=10)
    return _save_document(document, target)


def _native_chinese_fixture(target: Path) -> Path:
    import fitz

    document = fitz.open()
    page = document.new_page(width=600, height=800)
    font = fitz.Font(fontname="china-s")
    page.insert_font(fontname="CJK", fontbuffer=font.buffer)
    page.insert_text((45, 50), "\u6458\u8981", fontname="CJK", fontsize=13)
    page.insert_textbox((45, 75, 555, 145), "\u672c\u6587\u63d0\u51fa\u8bba\u6587\u7f16\u7801\u5668\u548c\u5173\u7cfb\u89e3\u7801\u5668,\u751f\u6210\u53ef\u7f16\u8f91\u6846\u67b6\u56fe\u3002", fontname="CJK", fontsize=10)
    page.insert_text((45, 185), "2 \u65b9\u6cd5", fontname="CJK", fontsize=13)
    page.insert_textbox((45, 210, 555, 300), "\u8f93\u5165\u8bba\u6587\u9996\u5148\u8fdb\u5165\u6587\u6863\u7f16\u7801\u5668\u3002\u5173\u7cfb\u89e3\u7801\u5668\u8fde\u63a5\u5b9e\u4f53,\u6700\u540e\u8f93\u51fa\u53ef\u7f16\u8f91\u7684\u6f14\u793a\u6587\u7a3f\u56fe\u3002", fontname="CJK", fontsize=10)
    page.insert_textbox((45, 340, 555, 400), "\u56fe 1:\u8bba\u6587\u7f16\u7801\u5668\u3001\u5173\u7cfb\u89e3\u7801\u5668\u4e0e\u53ef\u7f16\u8f91\u8f93\u51fa\u7684\u7cfb\u7edf\u6846\u67b6\u3002", fontname="CJK", fontsize=10)
    page.insert_text((45, 455), "3 \u5b9e\u9a8c", fontname="CJK", fontsize=13)
    page.insert_textbox((45, 480, 555, 545), "\u5b9e\u9a8c\u8bc4\u4f30\u5b9e\u4f53\u53ec\u56de\u7387\u3001\u5173\u7cfb\u53ec\u56de\u7387\u548c\u5904\u7406\u65f6\u95f4\u3002", fontname="CJK", fontsize=10)
    page.insert_text((45, 600), "4 \u7ed3\u8bba", fontname="CJK", fontsize=13)
    page.insert_textbox((45, 625, 555, 690), "\u8be5\u65b9\u6cd5\u80fd\u591f\u4fdd\u6301\u8bba\u6587\u672f\u8bed\u3001\u8bc1\u636e\u9875\u7801\u548c\u6a21\u5757\u5173\u7cfb\u3002", fontname="CJK", fontsize=10)
    return _save_document(document, target)


def _native_spanish_fixture(target: Path) -> Path:
    import fitz

    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((45, 50), "Resumen", fontname="hebo", fontsize=13)
    page.insert_textbox((45, 75, 555, 145), "Proponemos un codificador de documentos que transforma el artículo en evidencia estructurada y genera una figura editable.", fontsize=10)
    page.insert_text((45, 185), "2 Método", fontname="hebo", fontsize=13)
    page.insert_textbox((45, 210, 555, 295), "El documento de entrada pasa al codificador de documentos. El decodificador de relaciones conecta las entidades y produce una salida editable.", fontsize=10)
    page.insert_textbox((45, 335, 555, 395), "Figura 1: Codificador de documentos, decodificador de relaciones y salida editable.", fontsize=10)
    page.insert_text((45, 455), "3 Experimentos", fontname="hebo", fontsize=13)
    page.insert_textbox((45, 480, 555, 545), "Evaluamos la recuperación de entidades, las relaciones correctas y el tiempo de procesamiento.", fontsize=10)
    page.insert_text((45, 600), "4 Conclusión", fontname="hebo", fontsize=13)
    page.insert_textbox((45, 625, 555, 690), "El método conserva la terminología científica y las referencias de evidencia.", fontsize=10)
    return _save_document(document, target)


def _hyphenated_native_fixture(target: Path) -> Path:
    import fitz

    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((45, 50), "Abstract", fontname="hebo", fontsize=12)
    page.insert_textbox((45, 75, 555, 145), "We introduce a trans-\nformer encoder for evidence-grounded figure generation.", fontsize=10)
    page.insert_text((45, 185), "2 Method", fontname="hebo", fontsize=12)
    page.insert_textbox((45, 210, 555, 300), "The trans-\nformer encoder passes the input representation to a relation de-\ncoder and produces an editable output.", fontsize=10)
    return _save_document(document, target)


def _formula_table_fixture(target: Path) -> Path:
    import fitz

    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((45, 45), "Abstract", fontname="hebo", fontsize=12)
    page.insert_textbox((45, 65, 555, 115), "We introduce a calibrated relation encoder and evaluate it with a dense ablation table.", fontsize=10)
    page.insert_text((45, 145), "2 Method", fontname="hebo", fontsize=12)
    page.insert_text((80, 185), "Q = X W_q     K = X W_k     V = X W_v", fontname="cour", fontsize=10)
    page.insert_text((80, 215), "A = softmax(Q K^T / sqrt(d)) V", fontname="cour", fontsize=10)
    page.insert_textbox((45, 245, 555, 295), "The relation encoder sends the calibrated representation to the prediction head.", fontsize=10)
    page.insert_text((45, 335), "3 Experiments", fontname="hebo", fontsize=12)
    page.insert_textbox((45, 360, 555, 390), "Table 1: Ablation results for encoder depth and relation accuracy.", fontsize=10)
    rows = [("Model", "Depth", "Accuracy"), ("Base", "6", "81.2"), ("Large", "12", "84.7"), ("Ours", "18", "87.9")]
    top = 420
    for row in rows:
        for left, value in zip((65, 260, 390), row):
            page.insert_text((left, top), value, fontsize=10)
        top += 28
    page.insert_textbox((45, 555, 555, 600), "The ablation shows that calibrated relation encoding improves accuracy.", fontsize=10)
    page.insert_text((45, 650), "4 Conclusion", fontname="hebo", fontsize=12)
    page.insert_textbox((45, 675, 555, 720), "The method preserves directed evidence relations and calibrated outputs.", fontsize=10)
    return _save_document(document, target)


def _rotated_repeated_margin_fixture(target: Path) -> Path:
    import fitz

    document = fitz.open()
    for page_number in range(1, 5):
        page = document.new_page(width=600, height=800)
        page.insert_text((45, 25), "Rotated Conference Header 2026", fontsize=8)
        page.insert_text((45, 785), f"Rotated Paper 42 | Page {page_number}", fontsize=8)
        page.insert_text((45, 75), f"{page_number} Method Stage", fontname="hebo", fontsize=12)
        page.insert_textbox((45, 105, 555, 220), f"Stage {page_number} preserves rotated scientific evidence and semantic reading order.", fontsize=10)
        page.set_rotation(90)
    return _save_document(document, target)


def _skewed_scan_fixture(source: Path, target: Path, angle: float = 2.0) -> Path:
    import fitz
    from PIL import Image

    source_document = fitz.open(str(source))
    try:
        pixmap = source_document[0].get_pixmap(dpi=144, alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    finally:
        source_document.close()
    rotated = image.rotate(float(angle), resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
    payload = io.BytesIO()
    rotated.save(payload, format="PNG")
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_image(page.rect, stream=payload.getvalue())
    return _save_document(document, target)


def _degraded_scan_fixture(source: Path, target: Path) -> Path:
    import fitz
    from PIL import Image, ImageFilter

    source_document = fitz.open(str(source))
    try:
        pixmap = source_document[0].get_pixmap(dpi=120, alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("L")
    finally:
        source_document.close()
    degraded = image.resize((420, 560), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(0.3)).convert("RGB")
    payload = io.BytesIO()
    degraded.save(payload, format="JPEG", quality=45, optimize=True)
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_image(page.rect, stream=payload.getvalue())
    return _save_document(document, target)


def _fixture_ocr_adapter(image_path: Path, _lang: str) -> list[dict[str, Any]]:
    if "page_002" not in image_path.name:
        return []
    lines = [
        ("2 Method", 110),
        ("The scanned encoder passes evidence to the scanned decoder and output module.", 220),
        ("Figure 1: Scanned architecture overview with encoder decoder and output.", 420),
    ]
    return [
        {
            "text": text,
            "confidence": 0.99,
            "quad": [[90, top], [1240, top], [1240, top + 55], [90, top + 55]],
        }
        for text, top in lines
    ]


def _render_preview(source: Path, target: Path, page_index: int = 0) -> Path:
    import fitz

    document = fitz.open(str(source))
    try:
        pixmap = document[page_index].get_pixmap(dpi=110, alpha=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(str(target))
    finally:
        document.close()
    return target


def _check(name: str, passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "actual": actual, "expected": expected}


def _run_case(
    case_id: str,
    fixture: Path,
    out_dir: Path,
    checks: Callable[[dict[str, Any]], list[dict[str, Any]]],
    *,
    ocr_engine: str = "off",
    ocr_adapter: Callable | None = None,
    preview_page: int = 0,
) -> dict[str, Any]:
    started = time.monotonic()
    parsed = parse_paper(fixture, ocr_engine=ocr_engine, ocr_adapter=ocr_adapter)
    elapsed = round(time.monotonic() - started, 4)
    case_dir = ensure_dir(out_dir / case_id)
    write_json(case_dir / "document_model.json", parsed)
    write_json(case_dir / "extraction_report.json", parsed.get("extraction_report", {}))
    preview = _render_preview(fixture, case_dir / f"preview_page_{preview_page + 1}.png", preview_page)
    assertions = checks(parsed)
    ocr_page_durations = parsed.get("extraction_report", {}).get("ocr_page_durations", [])
    result = {
        "case_id": case_id,
        "ok": all(item.get("passed") for item in assertions),
        "fixture": str(fixture),
        "preview": str(preview),
        "elapsed_seconds": elapsed,
        "pdf_type": parsed.get("extraction_report", {}).get("pdf_type"),
        "page_count": parsed.get("page_count"),
        "section_count": parsed.get("extraction_report", {}).get("section_count"),
        "figure_caption_count": parsed.get("extraction_report", {}).get("figure_caption_count"),
        "ocr_pages": parsed.get("extraction_report", {}).get("ocr_pages", []),
        "ocr_spacing_repair_count": parsed.get("extraction_report", {}).get("ocr_spacing_repair_count", 0),
        "ocr_page_durations": ocr_page_durations,
        "ocr_stage_seconds": {
            name: round(sum(float(item.get(name) or 0.0) for item in ocr_page_durations), 4)
            for name in ("render_seconds", "detection_seconds", "classification_seconds", "recognition_seconds", "postprocess_seconds", "inference_seconds")
        },
        "assertions": assertions,
    }
    write_json(case_dir / "benchmark_result.json", result)
    return result


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return round(ordered[index], 4)


def run_pdf_extraction_stress_suite(out: str | Path, ocr_engine: str = "off") -> dict[str, Any]:
    root = ensure_dir(out).resolve()
    fixtures = ensure_dir(root / "fixtures")
    native = _native_two_column_fixture(fixtures / "native_two_column.pdf")
    bold = _bold_heading_fixture(fixtures / "unnumbered_bold_sections.pdf")
    rotated = _rotated_fixture(fixtures / "rotated_native_page.pdf")
    mixed = _mixed_scan_fixture(fixtures / "mixed_scan.pdf")
    repeated_margin = _repeated_margin_fixture(fixtures / "repeated_margin_noise.pdf")
    chinese = _native_chinese_fixture(fixtures / "native_chinese.pdf")
    spanish = _native_spanish_fixture(fixtures / "native_spanish.pdf")
    hyphenated = _hyphenated_native_fixture(fixtures / "hyphenated_native.pdf")
    formula_table = _formula_table_fixture(fixtures / "formula_table_dense.pdf")
    rotated_margin = _rotated_repeated_margin_fixture(fixtures / "rotated_repeated_margin.pdf")

    results = [
        _run_case(
            "native_two_column",
            native,
            root,
            lambda parsed: [
                _check("two_columns", parsed["extraction_report"].get("max_column_count") == 2, parsed["extraction_report"].get("max_column_count"), 2),
                _check("reading_order", all(parsed["pages"][0]["text"].index(left) < parsed["pages"][0]["text"].index(right) for left, right in (("LEFT_TOP", "LEFT_BOTTOM"), ("LEFT_BOTTOM", "RIGHT_TOP"), ("RIGHT_TOP", "RIGHT_BOTTOM"))), parsed["pages"][0]["text"], "LEFT_TOP < LEFT_BOTTOM < RIGHT_TOP < RIGHT_BOTTOM"),
                _check("cross_column_caption_order", parsed["pages"][0]["text"].index("RIGHT_BOTTOM") < parsed["pages"][0]["text"].index("Figure 1:") < parsed["pages"][0]["text"].index("3 Conclusion"), parsed["pages"][0]["text"], "RIGHT_BOTTOM < Figure 1 < Conclusion"),
                _check("figure_caption", parsed["extraction_report"].get("figure_caption_count", 0) >= 1, parsed["extraction_report"].get("figure_caption_count"), ">=1"),
                _check("priority_sections", all(parsed["extraction_report"].get("section_coverage", {}).get(name) for name in ("abstract", "method", "conclusion")), parsed["extraction_report"].get("section_coverage"), "abstract/method/conclusion"),
            ],
        ),
        _run_case(
            "unnumbered_bold_sections",
            bold,
            root,
            lambda parsed: [
                _check("model_architecture_heading", "Model Architecture" in parsed.get("headings", []), parsed.get("headings", []), "Model Architecture"),
                _check("stack_heading", "Encoder and Decoder Stacks" in parsed.get("headings", []), parsed.get("headings", []), "Encoder and Decoder Stacks"),
                _check("section_hint", any(item.get("section_hint") == "Model Architecture" and "image encoder" in str(item.get("text") or "").casefold() for item in parsed.get("evidence", [])), [item.get("section_hint") for item in parsed.get("evidence", []) if "image encoder" in str(item.get("text") or "").casefold()], "Model Architecture"),
            ],
        ),
        _run_case(
            "rotated_native_page",
            rotated,
            root,
            lambda parsed: [
                _check("rotation", parsed["extraction_report"].get("rotated_pages") == [1], parsed["extraction_report"].get("rotated_pages"), [1]),
                _check("display_coordinates", all(0 <= block["bbox"][0] <= block["bbox"][2] <= parsed["pages"][0]["width"] and 0 <= block["bbox"][1] <= block["bbox"][3] <= parsed["pages"][0]["height"] for block in parsed["pages"][0]["blocks"]), parsed["pages"][0]["blocks"], "all blocks inside displayed page"),
                _check("semantic_reading_order", parsed["pages"][0]["text"].index("Abstract") < parsed["pages"][0]["text"].index("2 Method") < parsed["pages"][0]["text"].index("Figure 1:"), parsed["pages"][0]["text"], "Abstract < Method < Figure 1"),
                _check("caption", parsed["extraction_report"].get("figure_caption_count", 0) >= 1, parsed["extraction_report"].get("figure_caption_count"), ">=1"),
            ],
        ),
        _run_case(
            "mixed_scan_fixture_adapter",
            mixed,
            root,
            lambda parsed: [
                _check("mixed_pdf", parsed["extraction_report"].get("pdf_type") == "mixed", parsed["extraction_report"].get("pdf_type"), "mixed"),
                _check("local_ocr", parsed["extraction_report"].get("ocr_pages") == [2], parsed["extraction_report"].get("ocr_pages"), [2]),
                _check("method_recovered", parsed["extraction_report"].get("section_coverage", {}).get("method") is True, parsed["extraction_report"].get("section_coverage"), "method=true"),
                _check("scanned_caption", any("Scanned architecture overview" in str(item.get("caption") or "") for item in parsed.get("document_index", {}).get("figures", [])), parsed.get("document_index", {}).get("figures", []), "scanned Figure 1 caption"),
            ],
            ocr_engine="easyocr",
            ocr_adapter=_fixture_ocr_adapter,
            preview_page=1,
        ),
        _run_case(
            "repeated_margin_noise",
            repeated_margin,
            root,
            lambda parsed: [
                _check("noise_removed", parsed["extraction_report"].get("repeated_margin_noise_removed_count") == 8, parsed["extraction_report"].get("repeated_margin_noise_removed_count"), 8),
                _check("header_absent", all("Proceedings of the Example Conference" not in item.get("text", "") for item in parsed.get("evidence", [])), [item.get("text") for item in parsed.get("evidence", [])], "no repeated header evidence"),
                _check("footer_absent", all("Anonymous Paper 1234" not in item.get("text", "") for item in parsed.get("evidence", [])), [item.get("text") for item in parsed.get("evidence", [])], "no repeated footer evidence"),
                _check("scientific_content_preserved", any("relation decoder" in item.get("text", "").casefold() for item in parsed.get("evidence", [])), [item.get("text") for item in parsed.get("evidence", [])], "relation decoder evidence"),
            ],
        ),
        _run_case(
            "native_chinese",
            chinese,
            root,
            lambda parsed: [
                _check("unicode_preserved", "\u6587\u6863\u7f16\u7801\u5668" in parsed["pages"][0]["text"], parsed["pages"][0]["text"], "contains document encoder in Chinese"),
                _check("priority_sections", all(parsed["extraction_report"].get("section_coverage", {}).get(name) for name in ("abstract", "method", "experiments", "conclusion")), parsed["extraction_report"].get("section_coverage"), "Chinese abstract/method/experiments/conclusion"),
                _check("figure_caption", any("\u8bba\u6587\u7f16\u7801\u5668" in str(item.get("caption") or "") for item in parsed.get("document_index", {}).get("figures", [])), parsed.get("document_index", {}).get("figures", []), "Chinese Figure 1 caption"),
                _check("no_mojibake", parsed["extraction_report"].get("mojibake_rate") == 0.0, parsed["extraction_report"].get("mojibake_rate"), 0.0),
            ],
        ),
        _run_case(
            "native_spanish",
            spanish,
            root,
            lambda parsed: [
                _check("unicode_preserved", "artículo" in parsed["pages"][0]["text"] and "Conclusión" in parsed["pages"][0]["text"], parsed["pages"][0]["text"], "Spanish accents preserved"),
                _check("priority_sections", all(parsed["extraction_report"].get("section_coverage", {}).get(name) for name in ("abstract", "method", "experiments", "conclusion")), parsed["extraction_report"].get("section_coverage"), "Spanish abstract/method/experiments/conclusion"),
                _check("figure_caption", any("codificador de documentos" in str(item.get("caption") or "").casefold() for item in parsed.get("document_index", {}).get("figures", [])), parsed.get("document_index", {}).get("figures", []), "Spanish Figura 1 caption"),
                _check("source_language_preserved", "document encoder" not in parsed["pages"][0]["text"].casefold() and "codificador de documentos" in parsed["pages"][0]["text"].casefold(), parsed["pages"][0]["text"], "Spanish terminology without English translation"),
            ],
        ),
        _run_case(
            "hyphenated_native",
            hyphenated,
            root,
            lambda parsed: [
                _check("transformer_repaired", sum("transformer encoder" in item.get("text", "").casefold() for item in parsed.get("evidence", [])) >= 2, [item.get("text") for item in parsed.get("evidence", [])], "two transformer encoder mentions"),
                _check("decoder_repaired", any("relation decoder" in item.get("text", "").casefold() for item in parsed.get("evidence", [])), [item.get("text") for item in parsed.get("evidence", [])], "relation decoder"),
                _check("repair_count", parsed["extraction_report"].get("native_hyphenation_repair_count") == 3, parsed["extraction_report"].get("native_hyphenation_repair_count"), 3),
            ],
        ),
        _run_case(
            "formula_table_dense",
            formula_table,
            root,
            lambda parsed: [
                _check("formulas", parsed["extraction_report"].get("formula_count") == 2, parsed["extraction_report"].get("formula_count"), 2),
                _check("table_caption", parsed["extraction_report"].get("table_caption_count") == 1, parsed["extraction_report"].get("table_caption_count"), 1),
                _check("table_columns", parsed["document_index"]["tables"][0].get("columns") == ["Model", "Depth", "Accuracy"], parsed["document_index"]["tables"][0].get("columns"), ["Model", "Depth", "Accuracy"]),
                _check("table_rows", any(row.get("values") == ["Large", "12", "84.7"] for row in parsed["document_index"]["tables"][0].get("rows", [])), parsed["document_index"]["tables"][0].get("rows"), ["Large", "12", "84.7"]),
                _check("structured_table_evidence", any(item.get("kind") == "table" and "Row: Large | 12 | 84.7" in item.get("text", "") for item in parsed.get("evidence", [])), [item.get("text") for item in parsed.get("evidence", []) if item.get("kind") == "table"], "structured Large row"),
                _check("no_orphan_cells", not any(item.get("text") == "84.7" for item in parsed.get("evidence", [])), [item.get("text") for item in parsed.get("evidence", [])], "no isolated table cells"),
            ],
        ),
        _run_case(
            "rotated_repeated_margin",
            rotated_margin,
            root,
            lambda parsed: [
                _check("all_pages_rotated", parsed["extraction_report"].get("rotated_pages") == [1, 2, 3, 4], parsed["extraction_report"].get("rotated_pages"), [1, 2, 3, 4]),
                _check("rotated_noise_removed", parsed["extraction_report"].get("repeated_margin_noise_removed_count") == 8, parsed["extraction_report"].get("repeated_margin_noise_removed_count"), 8),
                _check("repeated_section_boundaries", parsed["extraction_report"].get("section_count") == 4, parsed["extraction_report"].get("section_count"), 4),
                _check("header_absent", all("Rotated Conference Header" not in item.get("text", "") for item in parsed.get("evidence", [])), [item.get("text") for item in parsed.get("evidence", [])], "no rotated header evidence"),
                _check("body_preserved", all(any(f"Stage {page_number} preserves" in item.get("text", "") for item in parsed.get("evidence", [])) for page_number in range(1, 5)), [item.get("text") for item in parsed.get("evidence", [])], "all rotated stage evidence"),
            ],
        ),
    ]

    if ocr_engine != "off":
        scanned_two_column = rasterize_pdf_as_scan(native, fixtures / "scanned_two_column.pdf", dpi=144)
        skewed_two_column = _skewed_scan_fixture(native, fixtures / "skewed_two_column.pdf")
        degraded_two_column = _degraded_scan_fixture(native, fixtures / "degraded_two_column.pdf")
        scanned_formula_table = rasterize_pdf_as_scan(formula_table, fixtures / "scanned_formula_table.pdf", dpi=144)
        scanned_spanish = rasterize_pdf_as_scan(spanish, fixtures / "scanned_spanish.pdf", dpi=144)
        results.append(_run_case(
            "mixed_scan_runtime_ocr",
            mixed,
            root,
            lambda parsed: [
                _check("ocr_completed", 2 in parsed["extraction_report"].get("ocr_pages", []), parsed["extraction_report"].get("ocr_pages", []), "contains page 2"),
                _check("method_spacing", "2 method" in parsed["pages"][1].get("text", "").casefold(), parsed["pages"][1].get("text", ""), "contains '2 Method'"),
                _check("sentence_spacing", "scanned encoder passes evidence to the scanned decoder and output" in parsed["pages"][1].get("text", "").casefold(), parsed["pages"][1].get("text", ""), "spaced scientific sentence"),
                _check("caption_spacing", "scanned architecture overview with encoder decoder and output" in parsed["pages"][1].get("text", "").casefold(), parsed["pages"][1].get("text", ""), "spaced caption"),
                _check("repairs_recorded", parsed["extraction_report"].get("ocr_spacing_repair_count", 0) > 0, parsed["extraction_report"].get("ocr_spacing_repair_count", 0), ">0"),
            ],
            ocr_engine=ocr_engine,
            preview_page=1,
        ))
        results.append(_run_case(
            "scanned_two_column_runtime_ocr",
            scanned_two_column,
            root,
            lambda parsed: [
                _check("two_columns", parsed["extraction_report"].get("max_column_count") == 2, parsed["extraction_report"].get("max_column_count"), 2),
                _check("reading_order", all(parsed["pages"][0]["text"].index(left) < parsed["pages"][0]["text"].index(right) for left, right in (("LEFT_TOP", "LEFT_BOTTOM"), ("LEFT_BOTTOM", "RIGHT_TOP"), ("RIGHT_TOP", "RIGHT_BOTTOM"))), parsed["pages"][0]["text"], "LEFT_TOP < LEFT_BOTTOM < RIGHT_TOP < RIGHT_BOTTOM"),
                _check("headings", all(value in parsed.get("headings", []) for value in ("Abstract", "Method", "Conclusion")), parsed.get("headings", []), "Abstract/Method/Conclusion"),
                _check("caption", parsed["extraction_report"].get("figure_caption_count") == 1, parsed["extraction_report"].get("figure_caption_count"), 1),
            ],
            ocr_engine=ocr_engine,
        ))
        results.append(_run_case(
            "skewed_two_column_runtime_ocr",
            skewed_two_column,
            root,
            lambda parsed: [
                _check("two_columns", parsed["extraction_report"].get("max_column_count") == 2, parsed["extraction_report"].get("max_column_count"), 2),
                _check("reading_order", all(parsed["pages"][0]["text"].index(left) < parsed["pages"][0]["text"].index(right) for left, right in (("LEFT_TOP", "LEFT_BOTTOM"), ("LEFT_BOTTOM", "RIGHT_TOP"), ("RIGHT_TOP", "RIGHT_BOTTOM"))), parsed["pages"][0]["text"], "LEFT_TOP < LEFT_BOTTOM < RIGHT_TOP < RIGHT_BOTTOM"),
                _check("mean_confidence", float(parsed["extraction_report"].get("mean_ocr_confidence") or 0.0) >= 0.9, parsed["extraction_report"].get("mean_ocr_confidence"), ">=0.9"),
                _check("caption", parsed["extraction_report"].get("figure_caption_count") == 1, parsed["extraction_report"].get("figure_caption_count"), 1),
            ],
            ocr_engine=ocr_engine,
        ))
        results.append(_run_case(
            "degraded_two_column_quality_gate",
            degraded_two_column,
            root,
            lambda parsed: [
                _check("quality_gate_applied", parsed["extraction_report"].get("ocr_latin_lexical_gate_applied") is True, parsed["extraction_report"].get("ocr_latin_lexical_gate_applied"), True),
                _check("low_known_word_ratio", float(parsed["extraction_report"].get("ocr_latin_known_word_ratio") or 1.0) < 0.5, parsed["extraction_report"].get("ocr_latin_known_word_ratio"), "<0.5"),
                _check("extraction_failed", parsed["extraction_report"].get("status") == "fail", parsed["extraction_report"].get("status"), "fail"),
                _check("no_false_cache_scope", parsed["extraction_report"].get("sampled_scan_ready") is False, parsed["extraction_report"].get("sampled_scan_ready"), False),
            ],
            ocr_engine=ocr_engine,
        ))
        results.append(_run_case(
            "scanned_formula_table_runtime_ocr",
            scanned_formula_table,
            root,
            lambda parsed: [
                _check("formula_recovered", parsed["extraction_report"].get("formula_count", 0) >= 1, parsed["extraction_report"].get("formula_count"), ">=1"),
                _check("table_columns", parsed["document_index"]["tables"][0].get("columns") == ["Model", "Depth", "Accuracy"], parsed["document_index"]["tables"][0].get("columns"), ["Model", "Depth", "Accuracy"]),
                _check("large_row", any(row.get("values") == ["Large", "12", "84.7"] for row in parsed["document_index"]["tables"][0].get("rows", [])), parsed["document_index"]["tables"][0].get("rows"), ["Large", "12", "84.7"]),
                _check("structured_evidence", any(item.get("kind") == "table" and "Row: Large | 12 | 84.7" in item.get("text", "") for item in parsed.get("evidence", [])), [item.get("text") for item in parsed.get("evidence", []) if item.get("kind") == "table"], "structured Large row"),
            ],
            ocr_engine=ocr_engine,
        ))
        results.append(_run_case(
            "scanned_spanish_runtime_ocr",
            scanned_spanish,
            root,
            lambda parsed: [
                _check("ocr_completed", parsed["extraction_report"].get("ocr_pages") == [1], parsed["extraction_report"].get("ocr_pages"), [1]),
                _check("priority_sections", all(parsed["extraction_report"].get("section_coverage", {}).get(name) for name in ("abstract", "method", "experiments", "conclusion")), parsed["extraction_report"].get("section_coverage"), "Spanish abstract/method/experiments/conclusion"),
                _check("figure_caption", parsed["extraction_report"].get("figure_caption_count") == 1, parsed["extraction_report"].get("figure_caption_count"), 1),
                _check("source_language_preserved", "document encoder" not in parsed["pages"][0]["text"].casefold() and "codificador de documentos" in parsed["pages"][0]["text"].casefold(), parsed["pages"][0]["text"], "Spanish terminology without English translation"),
                _check("quality_gate_passed", parsed["extraction_report"].get("status") != "fail", parsed["extraction_report"].get("status"), "pass or warning"),
            ],
            ocr_engine=ocr_engine,
        ))

    elapsed = [float(item.get("elapsed_seconds") or 0.0) for item in results]
    ocr_stage_totals = {
        name: round(sum(float(item.get("ocr_stage_seconds", {}).get(name) or 0.0) for item in results), 4)
        for name in ("render_seconds", "detection_seconds", "classification_seconds", "recognition_seconds", "postprocess_seconds", "inference_seconds")
    }
    report = {
        "summary": "Deterministic multi-layout PDF extraction stress suite completed.",
        "ok": all(item.get("ok") for item in results),
        "out_dir": str(root),
        "ocr_engine": ocr_engine,
        "aggregate": {
            "case_count": len(results),
            "passed_case_count": sum(bool(item.get("ok")) for item in results),
            "mean_elapsed_seconds": round(sum(elapsed) / max(1, len(elapsed)), 4),
            "p95_elapsed_seconds": _percentile(elapsed, 0.95),
            "ocr_case_count": sum(bool(item.get("ocr_pages")) for item in results),
            "ocr_stage_seconds": ocr_stage_totals,
        },
        "cases": results,
    }
    write_json(root / "pdf_extraction_stress_report.json", report)
    return report
