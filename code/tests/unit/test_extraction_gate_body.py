"""The reading gate must judge a paper by its body, not by its figure appendix."""

import tempfile
import unittest
from pathlib import Path

from rfs.analysis.paper.document import _extraction_report


def _page(number, text, agreement=0.98):
    return {"page": number, "text": text, "blocks": [], "column_count": 1, "poppler_agreement": agreement, "used_ocr": False}


def _long_page(number, sentence, agreement=0.98, repeats=240):
    return _page(number, " ".join(((sentence + " ") * repeats).split()), agreement=agreement)


def _report(pages, *, details=None):
    index = {"sections": [], "figures": [], "tables": [], "formulas": []}
    merged = {"warnings": [], "ocr_candidate_pages": [], "ocr_pages": [], "poppler_available": True, "elapsed_seconds": 0.0}
    merged.update(details or {})
    with tempfile.TemporaryDirectory() as temp:
        return _extraction_report(Path(temp) / "paper.pdf", pages, index, merged)


def _body_with_a_figure_appendix(agreement=0.98):
    pages = [
        _long_page(1, "Abstract we propose a method for editing attention.", agreement),
        _long_page(2, "Introduction attention bottlenecks are interpretable.", agreement),
        _long_page(3, "Method we replace multi head self attention.", agreement),
    ]
    for number in range(4, 14):
        pages.append(_long_page(number, "Method and experiments report the results on three datasets.", agreement))
    pages.append(_long_page(14, "Conclusion the method enables user intervention.", agreement))
    for number in range(15, 30):
        pages.append(_page(number, "", agreement))
    return pages


class ReadingGateTests(unittest.TestCase):
    def test_a_paper_whose_appendix_is_figures_is_read_not_thrown_away(self):
        report = _report(_body_with_a_figure_appendix())

        self.assertLess(report["readable_page_ratio"], 0.6)
        self.assertTrue(report["body_fully_located"])
        self.assertEqual(report["status"], "warning")
        self.assertNotIn("fewer than 60% of pages contain reliable text", report["warnings"])
        self.assertTrue(any("figure appendix" in warning for warning in report["warnings"]))

    def test_the_same_paper_is_still_refused_when_the_extractors_disagree(self):
        report = _report(_body_with_a_figure_appendix(agreement=0.4))

        self.assertFalse(report["body_fully_located"])
        self.assertEqual(report["status"], "fail")
        self.assertIn("fewer than 60% of pages contain reliable text", report["warnings"])

    def test_a_paper_with_too_little_text_is_still_refused(self):
        pages = [_long_page(number, "Abstract method experiments conclusion.", repeats=4) for number in range(1, 6)]
        for number in range(6, 16):
            pages.append(_page(number, ""))
        report = _report(pages)

        self.assertLess(report["readable_page_ratio"], 0.6)
        self.assertFalse(report["body_fully_located"])
        self.assertEqual(report["status"], "fail")

    def test_a_paper_with_no_located_sections_is_still_refused(self):
        pages = [_long_page(number, "Lorem ipsum dolor sit amet consectetur.") for number in range(1, 16)]
        for number in range(16, 31):
            pages.append(_page(number, ""))
        report = _report(pages)

        self.assertLess(report["readable_page_ratio"], 0.6)
        self.assertFalse(report["body_fully_located"])
        self.assertEqual(report["status"], "fail")

    def test_a_trailing_figure_appendix_is_read_when_the_abstract_heading_is_absent(self):
        pages = [
            _long_page(1, "This article studies a battery electrode."),
            _long_page(2, "Introduction the cell is assembled from two layers."),
            _long_page(3, "Method we measure the capacity of the electrode."),
        ]
        for number in range(4, 8):
            pages.append(_long_page(number, "Experiments report the capacity on three cells."))
        pages.append(_long_page(8, "Conclusion the electrode retains its capacity."))
        for number in range(9, 18):
            pages.append(_page(number, ""))
        report = _report(pages)

        self.assertFalse(report["section_coverage"]["abstract"])
        self.assertLess(report["readable_page_ratio"], 0.6)
        self.assertTrue(report["body_fully_located"])
        self.assertNotEqual(report["status"], "fail")

    def test_a_fully_readable_paper_still_passes(self):
        pages = [_long_page(number, "Abstract method experiments conclusion report results.") for number in range(1, 13)]
        report = _report(pages)

        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["body_fully_located"])


if __name__ == "__main__":
    unittest.main()
