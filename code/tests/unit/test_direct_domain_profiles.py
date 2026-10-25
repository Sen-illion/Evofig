"""Tests for domain-aware direct evolution profiles."""

import unittest

from rfs.evolution.direct import resolve_domain_profile


class DirectDomainProfileTests(unittest.TestCase):
    def test_alias_resolves_to_stable_domain(self):
        profile = resolve_domain_profile("ai-ml-method")
        self.assertEqual(profile["domain"], "ai_method")

    def test_auto_without_a_paper_label_stays_missing(self):
        profile = resolve_domain_profile("auto", {})
        self.assertEqual(profile["domain"], "")
        self.assertTrue(profile["domain_missing"])
        self.assertEqual(profile["icon_style"], "")

    def test_an_explicit_biology_request_keeps_the_index_domain(self):
        profile = resolve_domain_profile("biology", {})
        self.assertEqual(profile["domain"], "biology")
        self.assertFalse(profile["domain_missing"])
        profile = resolve_domain_profile("general", {})
        self.assertEqual(profile["domain"], "general")
        self.assertFalse(profile["domain_missing"])


if __name__ == "__main__":
    unittest.main()
