import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.analysis.paper.document import _assign_ocr_parent_blocks, _block_kind, _column_groups, _deadline_ocr_wave, _filter_ocr_margin_noise, _normalized_compare_text, _prepare_native_blocks, _prioritize_ocr_candidates, _rapidocr_worker_count, _repair_ocr_spacing, _section_coverage, _token_overlap_agreement, evidence_excerpt, overview_figure_panel_excerpt, overview_figure_spatial_excerpt, parse_paper
from rfs.contracts.semantic import _match_frozen_quote, _normalized_quote
from rfs.analysis.paper.inspection import inspect_paper
from rfs.analysis.paper.document_cache import DOCUMENT_CACHE_VERSION, write_document_cache
from rfs.analysis.paper.preparation import _semantic_cache_safe
from rfs.analysis.paper.ocr import _positive_ocr_setting, _rapidocr_detector_limit, run_rapidocr_detailed


class PaperAnalyzerTests(unittest.TestCase):
    def _pdf(self, draw) -> Path:
        root = Path(tempfile.mkdtemp())
        target = root / "paper.pdf"
        document = fitz.open()
        page = document.new_page(width=600, height=800)
        draw(page)
        document.save(target)
        document.close()
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        return target

    def _multipage_pdf(self, page_texts: list[list[str]]) -> Path:
        root = Path(tempfile.mkdtemp())
        target = root / "paper.pdf"
        document = fitz.open()
        for blocks in page_texts:
            page = document.new_page(width=600, height=800)
            top = 50
            for text in blocks:
                page.insert_textbox((45, top, 555, top + 110), text, fontsize=10)
                top += 120
        document.save(target)
        document.close()
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        return target

    def test_structured_pdf_preserves_page_blocks_and_captions(self):
        def draw(page):
            page.insert_text((50, 60), "1 Introduction", fontsize=14)
            page.insert_textbox((50, 90, 550, 180), "A readable scientific paper paragraph with enough text for extraction quality checks.", fontsize=10)
            page.insert_textbox((50, 220, 550, 280), "Figure 1: Overview of the proposed framework.", fontsize=10)

        parsed = parse_paper(self._pdf(draw))

        self.assertEqual(parsed["page_count"], 1)
        self.assertTrue(parsed["source_sha256"])
        self.assertTrue(parsed["pages"][0]["blocks"])
        self.assertEqual(parsed["document_index"]["figures"][0]["page"], 1)
        self.assertIn("Introduction", parsed["headings"])
        self.assertTrue(all(item.get("block_id") for item in parsed["evidence"]))

    def test_a_pdf_without_a_suffix_is_read_as_a_pdf(self):
        """Snapshot archive objects are sha256-addressed and carry no suffix."""

        source = self._multipage_pdf(
            [
                ["1 Introduction", "A readable scientific paper paragraph with enough text for extraction quality checks."],
                ["2 Method", "The encoder maps the input sequence into a representation used by the decoder."],
            ]
        )
        anonymous = source.with_name("paper")
        source.rename(anonymous)

        parsed = parse_paper(anonymous)

        self.assertEqual(parsed["loader"], "structured-pdf")
        self.assertEqual(parsed["page_count"], 2)
        self.assertEqual(parsed["source_name"], "paper")
        self.assertIn("Introduction", parsed["pages"][0]["text"])
        self.assertLess(parsed["extraction_report"]["mojibake_rate"], 0.01)

    def test_a_text_file_without_a_suffix_stays_on_the_plain_reader(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        target = root / "paper"
        target.write_text("A plain text note that is not a PDF at all.", encoding="utf-8")

        parsed = parse_paper(target)

        self.assertEqual(parsed["loader"], "plain")
        self.assertEqual(parsed["page_count"], 1)

    def test_bold_unnumbered_headings_create_section_boundaries(self):
        def draw(page):
            page.insert_text((50, 55), "Abstract", fontname="hebo", fontsize=12)
            page.insert_textbox((50, 75, 550, 145), "This paper introduces a reliable architecture with enough text for extraction quality checks.", fontsize=10)
            page.insert_text((50, 185), "Model Architecture", fontname="hebo", fontsize=11)
            page.insert_textbox((50, 205, 550, 285), "The encoder maps the input sequence into a representation used by the decoder and output layer.", fontsize=10)
            page.insert_text((50, 325), "Encoder and Decoder Stacks", fontname="hebo", fontsize=10)
            page.insert_textbox((50, 345, 550, 425), "Each stack contains attention and feed-forward modules with residual connections.", fontsize=10)

        parsed = parse_paper(self._pdf(draw))
        evidence_by_text = {item["text"]: item for item in parsed["evidence"]}
        architecture_text = next(text for text in evidence_by_text if text.startswith("The encoder maps"))
        stack_text = next(text for text in evidence_by_text if text.startswith("Each stack contains"))

        self.assertIn("Model Architecture", parsed["headings"])
        self.assertIn("Encoder and Decoder Stacks", parsed["headings"])
        self.assertEqual(evidence_by_text[architecture_text]["section_hint"], "Model Architecture")
        self.assertEqual(evidence_by_text[stack_text]["section_hint"], "Encoder and Decoder Stacks")
        self.assertGreaterEqual(parsed["extraction_report"]["typographic_heading_count"], 3)

    def test_multiline_figure_caption_is_combined_into_priority_evidence(self):
        def draw(page):
            page.insert_textbox((50, 80, 550, 150), "Figure 1: Three interconnected components: a promptable task,\na model, and a data engine for collecting a large dataset.", fontsize=10)
            page.insert_textbox((50, 180, 550, 250), "1 Introduction\nA sufficiently long introduction paragraph for quality validation.", fontsize=10)

        parsed = parse_paper(self._pdf(draw))

        caption = parsed["document_index"]["figures"][0]["caption"]
        self.assertIn("data engine", caption)
        self.assertTrue(any(item["kind"] == "caption" and "data engine" in item["text"] for item in parsed["evidence"]))

    def test_cross_column_caption_continuation_is_recovered(self):
        def draw(page):
            page.insert_textbox((40, 330, 290, 345), "Fig. 1 | Overview of the method. a, High-level pre-training and", fontsize=7)
            page.insert_textbox((40, 345, 290, 360), "usage. b, The model architecture is adapted from the", fontsize=7)
            page.insert_textbox((310, 330, 560, 345), "standard transformer encoder for two-dimensional tables.", fontsize=7)
            page.insert_textbox((40, 390, 290, 470), "Body text in the left column must not enter the caption.", fontsize=9)
            page.insert_textbox((310, 390, 560, 470), "Body text in the right column must not enter the caption.", fontsize=9)

        parsed = parse_paper(self._pdf(draw))
        captions = parsed["document_index"]["figures"]

        self.assertEqual(len(captions), 1)
        self.assertIn("standard transformer encoder", captions[0]["caption"])
        self.assertNotIn("Body text", captions[0]["caption"])

    def test_evidence_excerpt_prioritizes_labels_inside_overview_figure_region(self):
        caption = "Figure 1: Overview of the proposed method and model architecture."
        parsed = {
            "pages": [{
                "page": 2,
                "width": 600,
                "height": 800,
                "blocks": [
                    {"id": "P002_B001", "reading_order": 1},
                    {"id": "P002_B002", "reading_order": 2},
                    {"id": "P002_B003", "reading_order": 3},
                    {"id": "P002_B004", "reading_order": 4},
                ],
            }],
            "document_index": {"figures": [{"page": 2, "block_id": "P002_B004", "caption": caption}]},
            "evidence": [
                {"id": "E_CAP", "page": 2, "block_id": "P002_B004", "bbox": [40, 340, 560, 360], "kind": "caption", "text": caption},
                {"id": "E_FEATURE", "page": 2, "block_id": "P002_B001", "bbox": [310, 180, 410, 195], "kind": "paragraph", "text": "1D feature attention"},
                {"id": "E_SAMPLE", "page": 2, "block_id": "P002_B002", "bbox": [420, 180, 520, 195], "kind": "paragraph", "text": "1D sample attention"},
                {"id": "E_OUTPUT", "page": 2, "block_id": "P002_B003", "bbox": [500, 260, 550, 275], "kind": "paragraph", "text": "Predictions"},
                *[
                    {"id": f"E_OTHER_{index}", "page": 3 + index, "block_id": f"P{3 + index:03d}_B001", "bbox": [40, 50, 560, 120], "kind": "caption", "text": "Figure 9: Architecture visualization. " + ("irrelevant analysis " * 20)}
                    for index in range(8)
                ],
            ],
        }

        excerpt = evidence_excerpt(parsed, max_chars=700)

        self.assertIn("1D feature attention", excerpt)
        self.assertIn("1D sample attention", excerpt)
        self.assertIn("Predictions", excerpt)
        spatial = json.loads(overview_figure_spatial_excerpt(parsed))
        by_text = {item["text"]: item for item in spatial}
        self.assertEqual(by_text["1D feature attention"]["bbox_percent"], [0.5167, 0.225, 0.6833, 0.2437])

    def test_panel_excerpt_treats_descriptive_dataset_headings_as_groups(self):
        caption = "Figure 1: Overview. a, Pre-training and use. b, Model architecture."
        parsed = {
            "pages": [{"page": 1, "width": 600, "height": 800, "blocks": []}],
            "document_index": {"figures": [{"page": 1, "block_id": "P001_B009", "caption": caption}]},
            "evidence": [
                {"id": "E_A", "page": 1, "block_id": "P001_B001", "bbox": [40, 80, 48, 92], "kind": "paragraph", "text": "a"},
                {"id": "E_A_GROUP", "page": 1, "block_id": "P001_B002", "bbox": [55, 105, 180, 120], "kind": "paragraph", "text": "A synthetic dataset"},
                {"id": "E_A_INPUT", "page": 1, "block_id": "P001_B003", "bbox": [55, 135, 120, 150], "kind": "paragraph", "text": "X train"},
                {"id": "E_B", "page": 1, "block_id": "P001_B004", "bbox": [310, 80, 318, 92], "kind": "paragraph", "text": "b"},
                {"id": "E_B_GROUP", "page": 1, "block_id": "P001_B005", "bbox": [330, 105, 500, 120], "kind": "paragraph", "text": "An arbitrary real-world dataset"},
                {"id": "E_B_INPUT", "page": 1, "block_id": "P001_B006", "bbox": [330, 135, 420, 150], "kind": "paragraph", "text": "Input dataset"},
                {"id": "E_CAP", "page": 1, "block_id": "P001_B009", "bbox": [40, 380, 560, 410], "kind": "caption", "text": caption},
            ],
        }

        excerpt = json.loads(overview_figure_panel_excerpt(parsed))
        labels = {
            item["text"]: item
            for panel in excerpt["panels"]
            for item in panel["label_inventory"]
        }

        self.assertEqual(labels["A synthetic dataset"]["role_hint"], "group")
        self.assertEqual(labels["An arbitrary real-world dataset"]["role_hint"], "group")
        self.assertEqual(labels["X train"]["role_hint"], "input")
        self.assertEqual(labels["Input dataset"]["role_hint"], "input")

    def test_panel_excerpt_reads_parenthesised_panel_markers(self):
        caption = (
            "Figure 4: Overview of the pipeline with two phases: "
            "(i) Memory-Based Indexing Graph Construction, where Global Memory ( M ) co-evolves; "
            "and (ii) Memory-Guided Online Retrieval, which leverages Personalized PageRank."
        )
        parsed = {
            "pages": [{"page": 4, "width": 600, "height": 800, "blocks": []}],
            "document_index": {"figures": [{"page": 4, "block_id": "P004_B097", "caption": caption}]},
            "evidence": [
                {"id": "E_CAP", "page": 4, "block_id": "P004_B097", "bbox": [40, 380, 560, 400], "kind": "caption", "text": caption},
                {"id": "E_TITLE_I", "page": 4, "block_id": "P004_B002", "bbox": [60, 100, 220, 110], "kind": "paragraph", "text": "Memory-based Indexing Graph Construction"},
                {"id": "E_AGENTS", "page": 4, "block_id": "P004_B004", "bbox": [60, 120, 140, 130], "kind": "paragraph", "text": "Multi-Agent Group"},
                {"id": "E_ONTOLOGY", "page": 4, "block_id": "P004_B044", "bbox": [60, 140, 160, 150], "kind": "paragraph", "text": "Ontology Layer ( M ont )"},
                {"id": "E_TITLE_II", "page": 4, "block_id": "P004_B038", "bbox": [340, 100, 500, 110], "kind": "paragraph", "text": "Memory-guided Online Retrieval"},
                {"id": "E_QUESTION", "page": 4, "block_id": "P004_B043", "bbox": [340, 130, 380, 140], "kind": "paragraph", "text": "Question"},
                {"id": "E_ANSWER", "page": 4, "block_id": "P004_B064", "bbox": [340, 160, 380, 170], "kind": "paragraph", "text": "Answer"},
            ],
        }

        excerpt = json.loads(overview_figure_panel_excerpt(parsed))
        panels = {panel["panel_label"]: panel for panel in excerpt["panels"]}
        labels = {
            panel_label: {item["text"] for item in panel["label_inventory"]}
            for panel_label, panel in panels.items()
        }

        self.assertEqual(sorted(panels), ["i", "ii"])
        self.assertTrue(all(panel["anchor"] == "declared_marker" for panel in panels.values()))
        self.assertIn("Multi-Agent Group", labels["i"])
        self.assertIn("Ontology Layer ( M ont )", labels["i"])
        self.assertNotIn("Question", labels["i"])
        self.assertIn("Question", labels["ii"])
        self.assertIn("Answer", labels["ii"])
        required = {
            item["text"]
            for panel in panels.values()
            for item in panel["label_inventory"]
            if item["required_candidate"]
        }
        self.assertIn("Ontology Layer ( M ont )", required)

    def test_panel_excerpt_splits_by_layout_when_markers_have_no_text_blocks(self):
        caption = "Figure 2: Overview of the pipeline. (a) Training stage. (b) Inference stage."
        parsed = {
            "pages": [{"page": 2, "width": 600, "height": 800, "blocks": []}],
            "document_index": {"figures": [{"page": 2, "block_id": "P002_B090", "caption": caption}]},
            "evidence": [
                {"id": "E_CAP", "page": 2, "block_id": "P002_B090", "bbox": [40, 380, 560, 400], "kind": "caption", "text": caption},
                {"id": "E_ENC", "page": 2, "block_id": "P002_B001", "bbox": [60, 100, 120, 110], "kind": "paragraph", "text": "Encoder"},
                {"id": "E_LOSS", "page": 2, "block_id": "P002_B002", "bbox": [60, 130, 100, 140], "kind": "paragraph", "text": "Loss"},
                {"id": "E_DEC", "page": 2, "block_id": "P002_B003", "bbox": [340, 100, 400, 110], "kind": "paragraph", "text": "Decoder"},
                {"id": "E_PRED", "page": 2, "block_id": "P002_B004", "bbox": [340, 130, 420, 140], "kind": "paragraph", "text": "Predictions"},
            ],
        }

        excerpt = json.loads(overview_figure_panel_excerpt(parsed))
        panels = {panel["panel_label"]: panel for panel in excerpt["panels"]}

        self.assertEqual(sorted(panels), ["a", "b"])
        self.assertTrue(all(panel["anchor"] == "layout_cluster" for panel in panels.values()))
        self.assertEqual(
            {item["text"] for item in panels["a"]["label_inventory"]},
            {"Encoder", "Loss"},
        )
        self.assertEqual(
            {item["text"] for item in panels["b"]["label_inventory"]},
            {"Decoder", "Predictions"},
        )

    def test_two_column_blocks_are_sorted_left_then_right(self):
        def draw(page):
            page.insert_textbox((330, 100, 560, 180), "RIGHT COLUMN SECOND with enough words to form a complete paragraph.", fontsize=10)
            page.insert_textbox((40, 100, 270, 180), "LEFT COLUMN FIRST with enough words to form a complete paragraph.", fontsize=10)
            page.insert_textbox((40, 40, 560, 75), "2 Method", fontsize=14)
            page.insert_textbox((330, 220, 560, 300), "RIGHT COLUMN FOURTH with enough words to form a complete paragraph.", fontsize=10)
            page.insert_textbox((40, 220, 270, 300), "LEFT COLUMN THIRD with enough words to form a complete paragraph.", fontsize=10)

        parsed = parse_paper(self._pdf(draw))
        text = parsed["pages"][0]["text"]

        self.assertLess(text.index("LEFT COLUMN FIRST"), text.index("LEFT COLUMN THIRD"))
        self.assertLess(text.index("LEFT COLUMN THIRD"), text.index("RIGHT COLUMN SECOND"))
        self.assertGreaterEqual(parsed["pages"][0]["reading_order_confidence"], 0.8)
        self.assertEqual(parsed["pages"][0]["column_count"], 2)

    def test_word_fragments_join_inside_one_column_and_rebind(self):
        def word(text: str, x: float, y: float) -> dict:
            return {
                "bbox": [x, y, x + 42, y + 10],
                "text": text,
                "source": "pymupdf",
                "confidence": 1.0,
                "parent_block": int(x + y),
                "font_size": 10,
                "font_bold": False,
            }

        left = "An adaptive fusion module is then applied to obtain a unified embedding which is used for downstream tasks such as spatial domain identification and later clustering of the learned representations".split()
        right = "Other column keeps its own sentence about denoising far from the fusion module and never enters that paragraph at all even after several more lines of text here".split()
        records = []
        for index, text in enumerate(left):
            row, column = divmod(index, 4)
            records.append(word(text, 40 + column * 48, 100 + row * 14))
        for index, text in enumerate(right):
            row, column = divmod(index, 4)
            records.append(word(text, 340 + column * 48, 100 + row * 14))

        ordered, confidence, _repairs = _prepare_native_blocks(records, 450)
        holders = [
            item["text"]
            for item in ordered
            if "adaptive fusion module" in item["text"] and "unified embedding" in item["text"]
        ]

        self.assertEqual(len(holders), 1)
        self.assertNotIn("Other column", holders[0])
        self.assertLess(
            " ".join(item["text"] for item in ordered).index("adaptive fusion module"),
            " ".join(item["text"] for item in ordered).index("Other column"),
        )
        self.assertGreaterEqual(confidence, 0.8)
        quote = _normalized_quote(" ".join(left))
        indexed = [(f"b{index}", _normalized_quote(item["text"])) for index, item in enumerate(ordered)]
        self.assertTrue(_match_frozen_quote(quote, 1, {1: indexed}, indexed))

    def test_ordinary_lines_are_not_folded_into_one_paragraph(self):
        records = [
            {"bbox": [40, 100, 260, 112], "text": "Left column first sentence stays on its own line.", "source": "pymupdf", "confidence": 1.0, "parent_block": 1, "font_size": 10},
            {"bbox": [40, 118, 260, 130], "text": "Left column second sentence stays on its own line.", "source": "pymupdf", "confidence": 1.0, "parent_block": 1, "font_size": 10},
            {"bbox": [330, 100, 560, 112], "text": "Right column first sentence stays on its own line.", "source": "pymupdf", "confidence": 1.0, "parent_block": 2, "font_size": 10},
            {"bbox": [330, 118, 560, 130], "text": "Right column second sentence stays on its own line.", "source": "pymupdf", "confidence": 1.0, "parent_block": 2, "font_size": 10},
        ]

        ordered, _confidence, _repairs = _prepare_native_blocks(records, 600)

        self.assertEqual(sorted(item["text"] for item in ordered), sorted(item["text"] for item in records))

    def test_cross_extractor_token_agreement_ignores_column_order(self):
        native = "left column first passage left column second passage right column first passage"
        poppler = "left column first passage right column first passage left column second passage"

        self.assertEqual(_token_overlap_agreement(native, poppler), 1.0)

    def test_cross_extractor_token_agreement_ignores_equation_variable_layout(self):
        native = "The projected point gives the origin and direction. ax x z ay y z az bz."
        poppler = "The projected point gives the origin and direction. a x x/z a y y/z a z + b z."

        self.assertEqual(_token_overlap_agreement(native, poppler), 1.0)

    def test_cross_extractor_token_agreement_allows_poppler_hidden_text(self):
        native = "The network architecture contains an encoder and decoder."
        poppler = native + " hidden latex accessibility expansion with many additional formula tokens"

        self.assertEqual(_token_overlap_agreement(native, poppler), 1.0)

    def test_unicode_quality_gates_and_cjk_parser_agreement(self):
        native = "摘要 本文提出一个多模态框架，用于图像编码与文本生成。"
        poppler = "摘 要 本 文 提 出 一 个 多 模 态 框 架，用 于 图 像 编 码 与 文 本 生 成。"

        self.assertGreater(len(_normalized_compare_text(native)), 20)
        self.assertEqual(_token_overlap_agreement(native, poppler), 1.0)

    def test_cjk_section_coverage_and_caption_kinds(self):
        pages = [{"page": 1, "text": "摘要\n1 引言\n2 方法\n3 实验\n4 结论"}]
        coverage = _section_coverage(pages, [])

        self.assertTrue(all(coverage.values()))
        self.assertEqual(_block_kind("图 1 系统框架概览"), "caption")
        self.assertEqual(_block_kind("表 2 实验结果"), "table")
        self.assertEqual(_block_kind("2 方法"), "heading")

    def test_implicit_first_page_abstract_is_recognized(self):
        lead = (
            "Tabular data are used across scientific fields. "
            "Here we present a foundation model that learns an efficient prediction algorithm. "
            + "This lead summary reports the problem, method, evidence and implications. " * 14
        )
        coverage = _section_coverage([{"page": 1, "text": lead}], [{"title": "Methods"}])

        self.assertTrue(coverage["abstract"])
        self.assertTrue(coverage["method"])

    def test_localized_caption_prefix_requires_a_real_identifier(self):
        self.assertEqual(_block_kind("Figura 1: Arquitectura del sistema"), "caption")
        self.assertEqual(_block_kind("Tabelle A2: Ablation results"), "table")
        self.assertEqual(_block_kind("figura editable"), "paragraph")
        self.assertEqual(_block_kind("tableau comparatif"), "paragraph")
        self.assertEqual(_block_kind("Figures 1 and 2 outline our approach"), "paragraph")

    def test_three_column_blocks_are_sorted_column_major(self):
        def draw(page):
            page.insert_textbox((400, 100, 560, 170), "COLUMN THREE TOP complete scientific paragraph.", fontsize=9)
            page.insert_textbox((220, 100, 380, 170), "COLUMN TWO TOP complete scientific paragraph.", fontsize=9)
            page.insert_textbox((40, 100, 200, 170), "COLUMN ONE TOP complete scientific paragraph.", fontsize=9)
            page.insert_textbox((400, 220, 560, 290), "COLUMN THREE BOTTOM complete scientific paragraph.", fontsize=9)
            page.insert_textbox((220, 220, 380, 290), "COLUMN TWO BOTTOM complete scientific paragraph.", fontsize=9)
            page.insert_textbox((40, 220, 200, 290), "COLUMN ONE BOTTOM complete scientific paragraph.", fontsize=9)
            page.insert_textbox((40, 40, 560, 75), "3 Method Architecture and System Overview Across Three Columns", fontsize=13)

        parsed = parse_paper(self._pdf(draw))
        text = parsed["pages"][0]["text"]

        self.assertLess(text.index("COLUMN ONE TOP"), text.index("COLUMN ONE BOTTOM"))
        self.assertLess(text.index("COLUMN ONE BOTTOM"), text.index("COLUMN TWO TOP"))
        self.assertLess(text.index("COLUMN TWO BOTTOM"), text.index("COLUMN THREE TOP"))
        self.assertEqual(parsed["pages"][0]["column_count"], 3)
        self.assertEqual(parsed["extraction_report"]["max_column_count"], 3)

    def test_invalid_three_column_split_falls_back_to_two_columns(self):
        items = []
        for index in range(6):
            items.append({"bbox": [95, 300 + index * 30, 395, 322 + index * 30], "text": f"Complete left column scientific sentence number {index}."})
            items.append({"bbox": [412, 300 + index * 30, 713, 322 + index * 30], "text": f"Complete right column scientific sentence number {index}."})
        items.extend([
            {"bbox": [157, 96, 650, 120], "text": "BERT Pre-training of Deep Bidirectional Transformers"},
            {"bbox": [294, 125, 512, 148], "text": "Language Understanding"},
        ])

        groups = _column_groups(items, 804)

        self.assertEqual(len(groups), 2)

    def test_rotated_page_uses_display_coordinate_space(self):
        def draw(page):
            page.insert_textbox((40, 40, 560, 85), "Rotated Paper Architecture Overview", fontsize=14)
            page.insert_textbox((40, 120, 270, 220), "First scientific block with enough evidence for extraction.", fontsize=10)
            page.insert_textbox((330, 120, 560, 220), "Second scientific block with enough evidence for extraction.", fontsize=10)
            page.set_rotation(90)

        parsed = parse_paper(self._pdf(draw))
        page = parsed["pages"][0]

        self.assertEqual(page["rotation"], 90)
        self.assertEqual((page["width"], page["height"]), (800.0, 600.0))
        self.assertTrue(all(0 <= block["bbox"][0] <= block["bbox"][2] <= page["width"] for block in page["blocks"]))
        self.assertTrue(all(0 <= block["bbox"][1] <= block["bbox"][3] <= page["height"] for block in page["blocks"]))
        self.assertEqual(parsed["extraction_report"]["rotated_pages"], [1])

    def test_unicode_normalization_keeps_math_symbols(self):
        def draw(page):
            page.insert_text((50, 60), "3 Method", fontsize=14)
            page.insert_textbox((50, 100, 550, 180), "Model accuracy is approximately 95 percent and beta parameters are preserved.", fontsize=10)

        parsed = parse_paper(self._pdf(draw))

        self.assertEqual(parsed["extraction_report"]["replacement_character_rate"], 0.0)
        self.assertEqual(parsed["extraction_report"]["mojibake_rate"], 0.0)

    def test_ocr_spacing_repair_restores_scientific_english_boundaries(self):
        mapping = {
            "Thescannedencoderpassesevidencetothescanned": ["The", "scanned", "encoder", "passes", "evidence", "to", "the", "scanned"],
            "architectureoverviewwithencoder": ["architecture", "overview", "with", "encoder"],
            "decoderandoutput": ["decoder", "and", "output"],
        }

        repaired, count = _repair_ocr_spacing(
            "2Method Thescannedencoderpassesevidencetothescanned decoderandoutput. Figure1:Scanned architectureoverviewwithencoder.",
            splitter=lambda token: mapping.get(token, [token]),
        )

        self.assertIn("2 Method", repaired)
        self.assertIn("The scanned encoder passes evidence to the scanned decoder and output", repaired)
        self.assertIn("Figure 1: Scanned architecture overview with encoder", repaired)
        self.assertGreaterEqual(count, 10)

    def test_ocr_spacing_repair_preserves_cjk_acronyms_and_known_compounds(self):
        source = "多模态 ImageBind BIDIRECTIONALTRANSFORMER multimodal representation"
        repaired, _ = _repair_ocr_spacing(source, splitter=lambda token: ["multi", "modal"])

        self.assertEqual(repaired, source)

    def test_ocr_spacing_repair_restores_numbered_chinese_section_heading(self):
        repaired, count = _repair_ocr_spacing("2\u65b9\u6cd5", splitter=lambda token: [token])

        self.assertEqual(repaired, "2 \u65b9\u6cd5")
        self.assertEqual(count, 1)

    def test_ocr_spacing_repair_restores_numbered_spanish_section_headings(self):
        repaired, count = _repair_ocr_spacing("2Metodo 3Experimentos 4Conclusion Figura1", splitter=lambda token: [token])

        self.assertEqual(repaired, "2 Metodo 3 Experimentos 4 Conclusion Figura 1")
        self.assertEqual(count, 4)

    def test_ocr_spacing_repair_does_not_split_spanish_word_on_single_english_anchor(self):
        repaired, count = _repair_ocr_spacing("procesamiento", splitter=lambda token: ["proc", "esa", "mien", "to"])

        self.assertEqual(repaired, "procesamiento")
        self.assertEqual(count, 0)

    def test_ocr_spacing_repair_restores_sentence_boundary_before_accented_latin_text(self):
        repaired, count = _repair_ocr_spacing("documentos.El módulo continúa.Éste termina.", splitter=lambda token: [token])

        self.assertEqual(repaired, "documentos. El módulo continúa. Éste termina.")
        self.assertEqual(count, 2)

    def test_scanned_chinese_numbered_sections_form_real_section_boundaries(self):
        path = self._multipage_pdf([[]])

        def adapter(_image_path, _lang):
            lines = [
                ("\u6458\u8981", 30),
                ("\u672c\u6587\u63d0\u51fa\u6587\u6863\u7f16\u7801\u5668\u548c\u5173\u7cfb\u89e3\u7801\u5668,\u751f\u6210\u53ef\u7f16\u8f91\u6846\u67b6\u56fe\u3002", 90),
                ("2\u65b9\u6cd5", 190),
                ("\u8f93\u5165\u8bba\u6587\u8fdb\u5165\u6587\u6863\u7f16\u7801\u5668,\u7136\u540e\u7531\u5173\u7cfb\u89e3\u7801\u5668\u751f\u6210\u8f93\u51fa\u3002", 250),
                ("3\u5b9e\u9a8c", 390),
                ("\u5b9e\u9a8c\u8bc4\u4f30\u5b9e\u4f53\u53ec\u56de\u7387\u548c\u5173\u7cfb\u53ec\u56de\u7387\u3002", 450),
                ("4\u7ed3\u8bba", 590),
                ("\u8be5\u65b9\u6cd5\u4fdd\u6301\u8bc1\u636e\u9875\u7801\u548c\u6a21\u5757\u5173\u7cfb\u3002", 650),
            ]
            return [
                {"text": text, "confidence": 0.99, "quad": [[40, top], [900, top], [900, top + 45], [40, top + 45]]}
                for text, top in lines
            ]

        parsed = parse_paper(path, ocr_engine="easyocr", ocr_adapter=adapter)

        self.assertEqual(parsed["headings"], ["\u6458\u8981", "\u65b9\u6cd5", "\u5b9e\u9a8c", "\u7ed3\u8bba"])
        self.assertEqual(parsed["extraction_report"]["section_count"], 4)
        self.assertGreaterEqual(parsed["extraction_report"]["ocr_spacing_repair_count"], 3)

    def test_truncated_pdf_fails_before_parser_noise(self):
        path = self._multipage_pdf([["Abstract", "A valid document before truncation."]])
        payload = path.read_bytes()
        path.write_bytes(payload[: max(16, len(payload) // 2)])

        with self.assertRaisesRegex(ValueError, "EOF marker is missing"):
            parse_paper(path, ocr_engine="off")

    def test_non_pdf_payload_with_pdf_extension_fails_preflight(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        path = root / "not_really.pdf"
        path.write_bytes(b"this is not a PDF document")

        with self.assertRaisesRegex(ValueError, "valid PDF header"):
            parse_paper(path, ocr_engine="off")

    def test_encrypted_pdf_returns_explicit_password_error(self):
        from pypdf import PdfReader, PdfWriter

        source = self._multipage_pdf([["Abstract", "Encrypted scientific content."]])
        target = source.parent / "encrypted.pdf"
        reader = PdfReader(str(source))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.encrypt("secret")
        with target.open("wb") as handle:
            writer.write(handle)

        with self.assertRaisesRegex(ValueError, "encrypted"):
            parse_paper(target, ocr_engine="off")

    def test_high_confidence_garbled_english_ocr_fails_lexical_gate(self):
        path = self._multipage_pdf([[]])

        def adapter(_image_path, _lang):
            lines = [
                "Abstract",
                "2 Method",
                "LEFT TOP encoder evidenoe begins themetiod ecndevidn letang clumn",
                "Figur oveview TOerevl edne beong complene lett comn",
                "RIGHT BOTTOM outputvid finihes rnading order clumn evidn",
                "3 Conclusion",
            ]
            return [
                {"text": text, "confidence": 0.92, "quad": [[40, 30 + index * 80], [900, 30 + index * 80], [900, 75 + index * 80], [40, 75 + index * 80]]}
                for index, text in enumerate(lines)
            ]

        parsed = parse_paper(path, ocr_engine="easyocr", ocr_adapter=adapter)
        report = parsed["extraction_report"]

        self.assertTrue(report["ocr_latin_lexical_gate_applied"])
        self.assertLess(report["ocr_latin_known_word_ratio"], 0.5)
        self.assertEqual(report["status"], "fail")

    def test_low_text_page_uses_local_ocr_adapter(self):
        path = self._pdf(lambda _page: None)

        parsed = parse_paper(
            path,
            ocr_engine="easyocr",
            ocr_adapter=lambda _image, _lang: [{
                "text": "Abstract Method Image Encoder produces the final representation with reliable evidence.",
                "confidence": 0.93,
                "quad": [[10, 10], [900, 10], [900, 80], [10, 80]],
            }],
        )

        self.assertEqual(parsed["extraction_report"]["ocr_pages"], [1])
        self.assertTrue(parsed["pages"][0]["used_ocr"])
        self.assertIn("Image Encoder", parsed["pages"][0]["text"])

    def test_cross_extractor_content_disagreement_still_triggers_ocr(self):
        path = self._pdf(lambda page: page.insert_textbox(
            (40, 80, 560, 220),
            "Abstract Method The native parser returns a sufficiently long but unsupported passage about an encoder and decoder.",
            fontsize=10,
        ))
        ocr_text = "Abstract Method The verified page contains an image encoder, transformer decoder, training objective, inference output, and complete scientific evidence for the recovered framework."
        with patch("rfs.analysis.paper.document._poppler_pages", return_value=(["Completely unrelated alternate parser content about chemistry molecules and laboratory measurements."], None)):
            parsed = parse_paper(
                path,
                ocr_engine="easyocr",
                ocr_adapter=lambda _image, _lang: [{
                    "text": ocr_text,
                    "confidence": 0.96,
                    "quad": [[20, 20], [950, 20], [950, 120], [20, 120]],
                }],
            )

        self.assertEqual(parsed["extraction_report"]["ocr_candidate_pages"], [1])
        self.assertEqual(parsed["extraction_report"]["ocr_pages"], [1])
        self.assertIn("verified page", parsed["pages"][0]["text"])

    def test_empty_ocr_result_returns_failed_document_model_instead_of_raising(self):
        path = self._pdf(lambda _page: None)

        parsed = parse_paper(path, ocr_engine="easyocr", ocr_adapter=lambda _image, _lang: [])

        self.assertEqual(parsed["extraction_report"]["status"], "fail")
        self.assertEqual(parsed["evidence"], [])
        self.assertEqual(parsed["extraction_report"]["pdf_type"], "scanned")

    def test_ocr_lines_are_grouped_into_complete_figure_caption(self):
        path = self._pdf(lambda _page: None)
        records = [
            {"text": "Figure 1: Overview of the proposed framework.", "confidence": 0.95, "quad": [[40, 100], [560, 100], [560, 125], [40, 125]]},
            {"text": "The image encoder feeds a transformer decoder.", "confidence": 0.94, "quad": [[40, 130], [560, 130], [560, 155], [40, 155]]},
            {"text": "1 Introduction", "confidence": 0.98, "quad": [[40, 210], [250, 210], [250, 240], [40, 240]]},
            {"text": "A sufficiently long scientific paragraph provides reliable evidence for the method.", "confidence": 0.96, "quad": [[40, 250], [560, 250], [560, 275], [40, 275]]},
        ]

        parsed = parse_paper(path, ocr_engine="easyocr", ocr_adapter=lambda _image, _lang: records)

        self.assertEqual(parsed["document_index"]["figures"][0]["caption"], "Figure 1: Overview of the proposed framework. The image encoder feeds a transformer decoder.")
        self.assertGreater(parsed["extraction_report"]["mean_ocr_confidence"], 0.9)

    def test_rapidocr_caption_group_survives_sentence_boundaries(self):
        lines = [
            {"bbox": [40, 100, 560, 120], "text": "Fig. 1: Overview of the model.", "confidence": 0.98},
            {"bbox": [40, 124, 560, 144], "text": "The encoder produces features.", "confidence": 0.98},
            {"bbox": [40, 148, 560, 168], "text": "A decoder predicts the output.", "confidence": 0.98},
            {"bbox": [40, 210, 560, 230], "text": "We next describe the training objective.", "confidence": 0.98},
        ]

        grouped = _assign_ocr_parent_blocks(lines, 600)

        self.assertEqual({item["parent_block"] for item in grouped[:3]}, {1})
        self.assertNotEqual(grouped[2]["parent_block"], grouped[3]["parent_block"])

    def test_ocr_margin_filter_removes_vertical_watermark_fragments(self):
        records = [
            {"bbox": [130, 100 + index * 30, 730, 124 + index * 30], "text": f"Complete scientific body line number {index} with reliable method evidence."}
            for index in range(6)
        ]
        records.extend([
            {"bbox": [35, 120, 70, 190], "text": "May"},
            {"bbox": [38, 210, 68, 245], "text": "28"},
            {"bbox": [90, 400, 180, 425], "text": "CNN"},
        ])

        filtered, removed = _filter_ocr_margin_noise(records, 842)

        self.assertEqual(removed, 2)
        self.assertIn("CNN", [item["text"] for item in filtered])

    def test_ocr_margin_filter_removes_fragments_even_when_body_starts_left(self):
        records = [
            {"bbox": [70, 300 + index * 30, 770, 324 + index * 30], "text": f"Long body line {index} establishes the left boundary near the page margin."}
            for index in range(6)
        ]
        records.extend([
            {"bbox": [35, 120, 70, 190], "text": "May"},
            {"bbox": [40, 220, 66, 246], "text": "S"},
            {"bbox": [63, 40, 790, 75], "text": "End-to-End Object Detection with Transformers"},
        ])

        filtered, removed = _filter_ocr_margin_noise(records, 842)

        self.assertEqual(removed, 2)
        self.assertIn("End-to-End Object Detection with Transformers", [item["text"] for item in filtered])

    def test_inspection_reuses_matching_document_cache(self):
        path = self._pdf(lambda page: page.insert_textbox((40, 80, 560, 180), "Abstract Method A sufficiently long scientific paragraph for deterministic PDF inspection.", fontsize=10))
        with tempfile.TemporaryDirectory() as temp:
            first = inspect_paper(path, temp, ocr_engine="off")
            second = inspect_paper(path, temp, ocr_engine="off")

            self.assertTrue(first["document_model"])
            self.assertEqual(second["status"], "cached")

    def test_inspection_invalidates_stale_local_document_model(self):
        path = self._pdf(lambda page: page.insert_textbox((40, 80, 560, 180), "Abstract Method A sufficiently long scientific paragraph for deterministic PDF inspection.", fontsize=10))
        with tempfile.TemporaryDirectory() as temp:
            first = inspect_paper(path, temp, ocr_engine="off")
            model_path = Path(first["document_model"])
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["document_cache_version"] = DOCUMENT_CACHE_VERSION - 1
            model_path.write_text(json.dumps(model), encoding="utf-8")

            second = inspect_paper(path, temp, ocr_engine="off")

            self.assertNotEqual(second["status"], "cached")
            refreshed = json.loads(model_path.read_text(encoding="utf-8"))
            self.assertEqual(refreshed["document_cache_version"], DOCUMENT_CACHE_VERSION)

    def test_global_document_cache_reuses_parse_across_output_directories(self):
        path = self._pdf(lambda page: page.insert_textbox((40, 80, 560, 220), "Abstract Method A sufficiently long scientific paragraph for global cache validation. The document contains repeated evidence, page-aware terminology, a complete method description, and enough characters to satisfy the readable-page quality gate.", fontsize=10))
        with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as first_out, tempfile.TemporaryDirectory() as second_out, patch.dict("os.environ", {"RFS_CACHE_DIR": cache}, clear=False):
            first = inspect_paper(path, first_out, ocr_engine="off")
            second = inspect_paper(path, second_out, ocr_engine="off")

            self.assertFalse(first["document_cache_hit"])
            self.assertTrue(second["document_cache_hit"])

    def test_evidence_ids_are_stable_across_repeated_parses(self):
        path = self._pdf(lambda page: page.insert_textbox((40, 80, 560, 180), "1 Method\nA stable block of scientific evidence for repeated parsing.", fontsize=10))

        first = parse_paper(path)
        second = parse_paper(path)

        self.assertEqual([item["id"] for item in first["evidence"]], [item["id"] for item in second["evidence"]])
        self.assertTrue(all(item["id"].startswith("E_P") for item in first["evidence"]))

    def test_evidence_budget_preserves_late_document_coverage(self):
        path = self._multipage_pdf([
            ["Abstract", "Early context " * 90],
            ["1 Introduction", "Our data engine has three stages:", "assisted-manual, semi-automatic, and fully automatic.", "Introduction evidence " * 24],
            ["2 Method", "The encoder transforms inputs into representations. " * 18],
            ["3 Experiments", "Experimental settings and evaluation metrics. " * 18],
            ["4 Analysis", "Ablation and error analysis evidence. " * 18],
            ["5 Conclusion", "The proposed method improves the final prediction while preserving scientific evidence. " * 12],
        ])

        parsed = parse_paper(path, max_chars=1800)
        evidence_pages = {item["page"] for item in parsed["evidence"]}

        self.assertEqual(evidence_pages, {1, 2, 3, 4, 5, 6})
        self.assertTrue(any(item["page"] == 6 and "Conclusion" in item["text"] for item in parsed["evidence"]))
        self.assertTrue(any("three stages" in item["text"] for item in parsed["evidence"]))
        self.assertTrue(any("assisted-manual" in item["text"] for item in parsed["evidence"]))
        self.assertEqual(parsed["extraction_report"]["evidence_page_coverage_ratio"], 1.0)
        self.assertLessEqual(parsed["extraction_report"]["evidence_char_count"], 1800)

    def test_ocr_priority_spreads_fully_scanned_long_document(self):
        pages = [{"page": index, "text": ""} for index in range(1, 13)]

        selected, details = _prioritize_ocr_candidates(pages, list(range(1, 13)), 6)

        self.assertEqual(selected, [1, 2, 3, 4, 10, 6])
        self.assertEqual([item["rank"] for item in details], [1, 2, 3, 4, 5, 6])

    def test_rapidocr_worker_count_is_bounded_and_adapter_safe(self):
        with patch.dict("os.environ", {"RFS_OCR_WORKERS": "3"}, clear=False):
            self.assertEqual(_rapidocr_worker_count("rapidocr", None, 6), min(3, __import__("os").cpu_count() or 1))
            self.assertEqual(_rapidocr_worker_count("rapidocr", lambda *_args: [], 6), 1)
            self.assertEqual(_rapidocr_worker_count("easyocr", None, 6), 1)

        with patch.dict("os.environ", {"RFS_OCR_WORKERS": ""}, clear=False), patch("rfs.analysis.paper.document.os.cpu_count", return_value=12):
            self.assertEqual(_rapidocr_worker_count("rapidocr", None, 6), 4)
        with patch.dict("os.environ", {"RFS_OCR_WORKERS": ""}, clear=False), patch("rfs.analysis.paper.document.os.cpu_count", return_value=4):
            self.assertEqual(_rapidocr_worker_count("rapidocr", None, 6), 2)

    def test_invalid_rapidocr_environment_setting_falls_back(self):
        with patch.dict("os.environ", {"RFS_RAPIDOCR_THREADS": "invalid"}, clear=False):
            self.assertEqual(_positive_ocr_setting(None, "RFS_RAPIDOCR_THREADS", 1), 1)

    def test_rapidocr_detector_limit_is_bounded_and_overridable(self):
        with patch.dict("os.environ", {"RFS_RAPIDOCR_DET_LIMIT": "384"}, clear=False):
            self.assertEqual(_rapidocr_detector_limit(), 384)
            self.assertEqual(_rapidocr_detector_limit(128), 256)
            self.assertEqual(_rapidocr_detector_limit(4096), 2048)

    def test_rapidocr_detailed_exposes_stage_timings(self):
        fake_engine = lambda _path: ([([[0, 0], [10, 0], [10, 10], [0, 10]], "Method", 0.95)], [0.2, 0.0, 0.3])
        local = __import__("rfs.analysis.paper.ocr", fromlist=["_RAPIDOCR_LOCAL"])._RAPIDOCR_LOCAL
        prior = getattr(local, "engines", None)
        local.engines = {(1, 6, 384): fake_engine}
        try:
            records, diagnostics = run_rapidocr_detailed("unused.png", "en", detector_limit=384)
        finally:
            if prior is None:
                delattr(local, "engines")
            else:
                local.engines = prior

        self.assertEqual(records[0]["text"], "Method")
        self.assertEqual(diagnostics["detector_limit"], 384)
        self.assertEqual(diagnostics["detection_seconds"], 0.2)
        self.assertEqual(diagnostics["recognition_seconds"], 0.3)
        self.assertEqual(diagnostics["inference_seconds"], 0.5)

    def test_ocr_priority_prefers_semantic_pages_before_coverage_anchors(self):
        pages = [{"page": index, "text": ""} for index in range(1, 11)]
        pages[3]["text"] = "Figure 1: Overview of the proposed architecture and system pipeline."
        pages[7]["text"] = "8 Conclusion We summarize the method and its limitations."

        selected, details = _prioritize_ocr_candidates(pages, [1, 4, 6, 8, 10], 3)

        self.assertEqual(selected[:2], [4, 8])
        self.assertIn("overview_figure", details[0]["reasons"])
        self.assertIn("conclusion", details[1]["reasons"])

    def test_scanned_pdf_report_records_adaptive_ocr_schedule(self):
        path = self._multipage_pdf([[] for _ in range(12)])

        def adapter(image_path, _lang):
            page_number = int(Path(image_path).stem.rsplit("_", 1)[-1])
            return [{
                "text": f"Page {page_number} Abstract Method architecture evidence with enough reliable text for adaptive OCR scheduling and document recovery.",
                "confidence": 0.97,
                "quad": [[20, 20], [560, 20], [560, 70], [20, 70]],
            }]

        parsed = parse_paper(path, ocr_engine="easyocr", ocr_adapter=adapter, max_ocr_pages=6)

        self.assertEqual(parsed["extraction_report"]["ocr_priority_pages"], [1, 2, 3, 4, 10, 6])
        self.assertEqual(parsed["extraction_report"]["ocr_pages"], [1, 2, 3, 4, 10, 6])
        self.assertEqual([item["page"] for item in parsed["extraction_report"]["ocr_priority"]], [1, 2, 3, 4, 10, 6])
        self.assertEqual(parsed["extraction_report"]["status"], "warning")
        self.assertEqual(parsed["extraction_report"]["pdf_type"], "scanned")
        self.assertEqual(parsed["extraction_report"]["semantic_scope"], "sampled_pages_only")
        self.assertFalse(parsed["extraction_report"]["scientific_scope_complete"])
        self.assertTrue(parsed["extraction_report"]["ocr_schedule_complete"])
        self.assertTrue(parsed["extraction_report"]["ocr_run_complete"])

    def test_three_high_confidence_scan_pages_are_usable_partial_evidence(self):
        path = self._multipage_pdf([[] for _ in range(12)])

        def adapter(image_path, _lang):
            page_number = int(Path(image_path).stem.rsplit("_", 1)[-1])
            return [{
                "text": f"Page {page_number} Abstract Method architecture evidence with enough reliable text for a partial engineering contract.",
                "confidence": 0.97,
                "quad": [[20, 20], [560, 20], [560, 70], [20, 70]],
            }]

        parsed = parse_paper(path, ocr_engine="easyocr", ocr_adapter=adapter, max_ocr_pages=3)
        report = parsed["extraction_report"]

        self.assertEqual(report["ocr_pages"], [1, 2, 3])
        self.assertTrue(report["sampled_scan_ready"])
        self.assertEqual(report["status"], "warning")
        self.assertEqual(report["semantic_scope"], "sampled_pages_only")
        self.assertFalse(report["scientific_scope_complete"])

    def test_deadline_limited_ocr_is_not_cached_as_complete(self):
        path = self._multipage_pdf([[] for _ in range(3)])

        def adapter(_image_path, _lang):
            return [{
                "text": "Abstract Method architecture evidence recovered by OCR.",
                "confidence": 0.97,
                "quad": [[20, 20], [560, 20], [560, 70], [20, 70]],
            }]

        parsed = parse_paper(path, deadline_at=time.monotonic() - 1, ocr_engine="easyocr", ocr_adapter=adapter, max_ocr_pages=3)
        report = parsed["extraction_report"]

        self.assertFalse(report["ocr_schedule_complete"])
        self.assertFalse(report["ocr_run_complete"])
        self.assertEqual(report["ocr_attempted_pages"], [])
        self.assertTrue(any("schedule was not completed" in warning for warning in report["warnings"]))
        with tempfile.TemporaryDirectory() as cache, patch.dict("os.environ", {"RFS_CACHE_DIR": cache}, clear=False):
            self.assertIsNone(write_document_cache(path, parsed, ocr_engine="easyocr", ocr_lang="en_ch"))
        self.assertFalse(_semantic_cache_safe(parsed))

    def test_deadline_ocr_wave_preserves_completed_pages_and_terminates_slow_workers(self):
        class FakeProcess:
            def __init__(self, command, **_kwargs):
                self.page = int(command[command.index("--page") + 1])
                self.threads = int(command[command.index("--threads") + 1])
                self.output = Path(command[command.index("--out") + 1])
                self.returncode = None
                self.terminated = False
                if self.page == 1:
                    self.output.write_text(json.dumps({
                        "blocks": [{"text": "Method", "bbox": [0, 0, 10, 10], "confidence": 0.9}],
                        "engine": "rapidocr",
                        "error": None,
                        "diagnostics": {"recognition_seconds": 0.1},
                        "elapsed_seconds": 0.2,
                    }), encoding="utf-8")
                    self.returncode = 0

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                del timeout
                return self.returncode

            def kill(self):
                self.returncode = -9

            def communicate(self, timeout=None):
                del timeout
                return "", ""

        created = []

        def fake_popen(command, **kwargs):
            process = FakeProcess(command, **kwargs)
            created.append(process)
            return process

        with patch("rfs.analysis.paper.document.subprocess.Popen", side_effect=fake_popen):
            results = _deadline_ocr_wave(Path("paper.pdf"), [1, 2], "rapidocr", "en", time.monotonic() + 20.05)

        by_page = {page: result for page, result, _elapsed in results}
        self.assertIsNone(by_page[1][2])
        self.assertEqual(by_page[1][0][0]["text"], "Method")
        self.assertTrue(by_page[1][3]["worker_process"])
        self.assertEqual(by_page[2][2], "OCR exceeded extraction deadline")
        self.assertTrue(by_page[2][3]["timed_out"])
        self.assertTrue(next(item for item in created if item.page == 2).terminated)

        created.clear()
        with patch("rfs.analysis.paper.document.subprocess.Popen", side_effect=fake_popen):
            single = _deadline_ocr_wave(Path("paper.pdf"), [1], "rapidocr", "en", time.monotonic() + 20.05)
        self.assertIsNone(single[0][1][2])
        self.assertEqual(created[0].page, 1)
        self.assertEqual(created[0].threads, 2)


if __name__ == "__main__":
    unittest.main()
