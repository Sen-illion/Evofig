"""Tests for the arXiv-derived domain strata of EditableFigureBench."""

import tempfile
import unittest
from pathlib import Path

from rfs.evaluation import editable_figure_domains as domains


CSV_HEADER = "arxiv_id,primary_category,categories\n"


class ClassifyDomainTests(unittest.TestCase):
    def test_a_specific_primary_category_decides_on_its_own(self):
        self.assertEqual("language", domains.classify_domain("cs.CL", "cs.CL; cs.AI"))
        self.assertEqual("agent", domains.classify_domain("cs.MA", "cs.MA; cs.LG"))
        self.assertEqual("vision", domains.classify_domain("cs.CV", "cs.CV"))
        self.assertEqual("general", domains.classify_domain("cs.LG", "cs.LG; cs.CL"))

    def test_the_broad_cs_ai_bucket_is_refined_by_the_remaining_categories(self):
        self.assertEqual("agent", domains.classify_domain("cs.AI", "cs.AI; cs.MA"))
        self.assertEqual("language", domains.classify_domain("cs.AI", "cs.AI; cs.CL"))
        self.assertEqual("vision", domains.classify_domain("cs.AI", "cs.AI; cs.CV"))

    def test_a_broad_primary_without_a_signal_falls_back_to_general(self):
        self.assertEqual("general", domains.classify_domain("cs.AI", "cs.AI; cs.LG"))
        self.assertEqual("general", domains.classify_domain("cs.AI", "cs.AI"))

    def test_out_of_scope_primary_categories_land_in_the_cross_domain_bucket(self):
        self.assertEqual("other", domains.classify_domain("cs.CR", "cs.CR; cs.CL"))
        self.assertEqual("other", domains.classify_domain("q-bio.BM", "q-bio.BM"))
        self.assertEqual("other", domains.classify_domain("cond-mat.mtrl-sci", "cond-mat.mtrl-sci"))

    def test_empty_metadata_never_invents_a_stratum(self):
        self.assertEqual("other", domains.classify_domain("", ""))
        self.assertEqual("other", domains.classify_domain(None, None))
        self.assertEqual("agent", domains.classify_domain("", ["cs.MA"]))

    def test_categories_accept_a_semicolon_string_or_a_sequence(self):
        self.assertEqual(("cs.CL", "cs.LG"), domains.normalize_categories("cs.CL; cs.LG"))
        self.assertEqual(("cs.CL", "cs.LG"), domains.normalize_categories([" cs.CL ", "cs.LG", "cs.CL"]))
        self.assertEqual((), domains.normalize_categories(" ; "))

    def test_every_stratum_has_a_documented_label(self):
        self.assertEqual(sorted(domains.DOMAIN_ORDER), sorted(domains.DOMAIN_LABELS))
        self.assertEqual(len(domains.DOMAIN_ORDER), len(set(domains.DOMAIN_ORDER)))


class CategoryIndexTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, text: str) -> Path:
        path = self.root / "papers.csv"
        path.write_text(text, encoding="utf-8-sig")
        return path

    def test_a_missing_file_yields_an_empty_index(self):
        self.assertEqual({}, domains.load_arxiv_categories(self.root / "absent.csv"))

    def test_the_index_reads_a_bom_csv_and_classifies_every_row(self):
        path = self._write(
            CSV_HEADER
            + "2301.02083,cs.MA,cs.MA; cs.AI; cs.LG\n"
            + "2302.00001,cs.AI,cs.AI; cs.CL\n"
            + "2303.00001,,cs.CV; cs.LG\n"
        )
        index = domains.load_arxiv_categories(path)
        self.assertEqual(["2301.02083", "2302.00001", "2303.00001"], sorted(index))
        self.assertEqual("agent", index["2301.02083"]["domain"])
        self.assertEqual("language", index["2302.00001"]["domain"])
        self.assertEqual("vision", index["2303.00001"]["domain"])
        self.assertEqual("cs.CV", index["2303.00001"]["arxiv_primary_category"])
        self.assertEqual(["cs.CV", "cs.LG"], index["2303.00001"]["arxiv_categories"])

    def test_rows_without_a_paper_id_are_skipped(self):
        path = self._write(CSV_HEADER + ",cs.CL,cs.CL\n" + "2301.02083,cs.CL,cs.CL\n")
        self.assertEqual(["2301.02083"], sorted(domains.load_arxiv_categories(path)))

    def test_domain_fields_mark_an_unknown_paper(self):
        index = {"p1": {"domain": "agent", "arxiv_primary_category": "cs.MA", "arxiv_categories": ["cs.MA"]}}
        known = domains.domain_fields("p1", index)
        self.assertEqual("agent", known["domain"])
        self.assertFalse(known["domain_unlabeled"])
        self.assertEqual(domains.DOMAIN_SCHEMA_VERSION, known["domain_schema_version"])
        unknown = domains.domain_fields("p2", index)
        self.assertIsNone(unknown["domain"])
        self.assertTrue(unknown["domain_unlabeled"])
        self.assertEqual([], unknown["arxiv_categories"])

    def test_summary_keeps_the_canonical_order_and_counts_unlabeled(self):
        summary = domains.summarize_domains(["general", "language", "general", None, ""])
        self.assertEqual(["language", "general", "unlabeled"], list(summary))
        self.assertEqual({"language": 1, "general": 2, "unlabeled": 2}, summary)
        self.assertEqual({}, domains.summarize_domains([]))


if __name__ == "__main__":
    unittest.main()
