"""The figure must never print an internal id where the paper printed no words."""

import unittest

from rfs.planning.figure_text_planner import _drop_identifier_labels, _identifier_label


class IdentifierLabelTests(unittest.TestCase):
    def test_an_internal_id_reads_as_an_identifier(self):
        for value in ("panelrel_i_04", "connector_paper_006", "rel_1", "sedge.d6bde954182bc0e5", "panel_i"):
            self.assertTrue(_identifier_label(value), value)

    def test_printed_words_do_not(self):
        for value in ("Extraction", "Multi-Agent Group (A)", "Three-Layer Global Memory", "co-evolve", "", "Type: Person"):
            self.assertFalse(_identifier_label(value), value)
        # Two words are always words, even when the first one looks like our prefix.
        self.assertFalse(_identifier_label("Panel Overview"))

    def test_drops_items_whose_text_is_an_id_and_keeps_the_rest(self):
        plan = {"items": [
            {"id": "a", "text": "panelrel_i_04"},
            {"id": "b", "text": "Extraction"},
            {"id": "c", "text": "connector_paper_006"},
            {"id": "d", "text": "input_docs"},
        ]}
        design = {"nodes": [{"id": "input_docs", "label": "Unstructured"}], "connectors": [{"id": "c1"}]}
        cleaned, dropped = _drop_identifier_labels(plan, design)
        self.assertEqual([item["text"] for item in cleaned["items"]], ["Extraction"])
        self.assertEqual(dropped, ["connector_paper_006", "input_docs", "panelrel_i_04"])
        self.assertEqual(cleaned["provenance"]["identifier_labels_dropped"], dropped)

    def test_a_clean_plan_is_returned_untouched(self):
        plan = {"items": [{"id": "a", "text": "Extraction"}]}
        cleaned, dropped = _drop_identifier_labels(plan, {"nodes": [], "connectors": []})
        self.assertIs(cleaned, plan)
        self.assertEqual(dropped, [])


if __name__ == "__main__":
    unittest.main()
