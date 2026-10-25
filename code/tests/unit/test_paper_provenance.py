"""Unit tests for the paper-text provenance lookup of the middleware bind layer."""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.judge_training.paper_provenance import (
    PaperProvenanceLookup,
    PaperText,
    is_searchable_phrase,
    normalize_phrase,
)


class PhraseTests(unittest.TestCase):
    """A phrase is looked up only when it is specific enough to mean something."""

    def test_a_short_or_letterless_phrase_is_never_searched(self):
        self.assertFalse(is_searchable_phrase("b16"))
        self.assertFalse(is_searchable_phrase("6.8"))
        self.assertFalse(is_searchable_phrase("(1) - (2)"))
        self.assertTrue(is_searchable_phrase("Riemannian optimization"))

    def test_a_print_hyphen_split_across_lines_folds_into_one_word(self):
        self.assertEqual(normalize_phrase("distilla-\ntion"), "distillation")
        self.assertEqual(normalize_phrase("  Gated\tEncoder  "), "gated encoder")


class PaperTextTests(unittest.TestCase):
    """Only contiguous text matches; a bag of words never does."""

    def test_a_phrase_the_pages_carry_as_one_run_is_found(self):
        text = PaperText([(7, "We train the model with a distillation loss and report accuracy.")])
        hit = text.find("distillation loss")
        self.assertEqual(hit["page"], 7)
        self.assertIn("distillation loss", hit["snippet"])

    def test_words_that_only_occur_far_apart_never_match(self):
        text = PaperText([(7, "The model is trained on ImageNet. Accuracy is reported in Table 2.")])
        self.assertIsNone(text.find("model accuracy"))

    def test_a_phrase_the_snapshot_split_with_a_hyphen_still_matches(self):
        text = PaperText([(3, "the loss uses distilla-\ntion from the teacher")])
        hit = text.find("distillation")
        self.assertEqual(hit["page"], 3)

    def test_symbols_and_letterless_tokens_are_never_searched(self):
        text = PaperText([(1, "b16 appears here and so do 6.8 and (c)")])
        self.assertIsNone(text.find("b16"))
        self.assertIsNone(text.find("6.8"))
        self.assertIsNone(text.find(""))


class LookupTests(unittest.TestCase):
    """The lookup resolves each paper's directory once and degrades to nothing."""

    def paper(self, paper_id="p0001", pointer="assets/p0001/captions.json"):
        return {"paper_id": paper_id, "captions_path": pointer}

    def test_the_lookup_finds_a_phrase_in_one_papers_own_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_dir = root / "assets" / "p0001"
            paper_dir.mkdir(parents=True)
            (paper_dir / "text_locations.jsonl").write_text(
                json.dumps({"page": 5, "text": "we introduce a gated encoder layer"}) + "\n",
                encoding="utf-8",
            )
            lookup = PaperProvenanceLookup(root, [self.paper()])

            hit = lookup.finder("p0001")("gated encoder")

            self.assertEqual(hit["page"], 5)
            self.assertEqual(lookup.usage_report()["papers"], 1)

    def test_a_paper_the_snapshot_never_froze_finds_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            lookup = PaperProvenanceLookup(tmp, [self.paper()])
            self.assertIsNone(lookup.finder("p0001")("gated encoder"))

    def test_a_paper_record_without_a_pointer_is_skipped_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            lookup = PaperProvenanceLookup(tmp, [{"paper_id": "p0001"}])
            self.assertIsNone(lookup.finder("p0001")("gated encoder"))


if __name__ == "__main__":
    unittest.main()
