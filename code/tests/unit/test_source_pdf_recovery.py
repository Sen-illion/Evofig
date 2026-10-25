"""Official original recovery must verify identity and preserve existing data."""

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from rfs.judge_training.literature_guard import LiteratureBusyError
from rfs.providers.literature import ProviderRequestError
from rfs.judge_training.source_pdf_recovery import (
    COPY_CHANNELS,
    _ArticleNotServed,
    _download_pmc_cloud,
    _pmc_package_key,
    _pmc_pdf_url,
    _pow_challenge,
    _pow_nonce,
    _request_spec,
    recover_official_source_pdfs,
)
from rfs.judge_training.source_snapshot_inventory import _input_files
from rfs.judge_training.source_snapshot_io import write_jsonl


class _FakeResponse:
    """Stand for one HTTP answer, carrying only what the transport reads."""

    def __init__(self, url, content, status_code=200):
        self.url = url
        self.content = content
        self.status_code = status_code


class _FakeCloudClient:
    """Answer from a fixed address map and record every address asked for, in order."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def request(self, url, *, params=None):
        self.calls.append(url)
        if url not in self.answers:
            raise ProviderRequestError("official_source_recovery", "not_found",
                                       "provider returned HTTP 404")
        return self.answers[url]


class OfficialSourceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.batch = Path(self.temp.name) / "batch"
        self.folder = self.batch / "assets/materials/official/paper"
        self.text = self.folder / "text/full.txt"
        self.text.parent.mkdir(parents=True)
        self.text.write_text("Original HTML-derived full text for 10.1038/s41467-026-00001-x\n"
                             "Verified scientific framework", encoding="utf-8")
        self.row = {"paper_id": "10.1038/s41467-026-00001-x", "doi": "10.1038/s41467-026-00001-x",
                    "title": "Verified scientific framework", "domain": "materials",
                    "figure_asset_tier": "B_official_standalone",
                    "full_text_path": "assets/materials/official/paper/text/full.txt",
                    "fulltext_source": "https://www.nature.com/articles/s41467-026-00001-x",
                    "publisher_landing_evidence": {"resolved_url": "https://www.nature.com/articles/s41467-026-00001-x",
                                                   "title_found": True, "venue_found": True}}
        self.strict = self.batch / "verified/strict_training_papers.jsonl"
        write_jsonl(self.strict, [self.row])
        self.target = self.folder / "official.pdf"
        guard = patch("rfs.judge_training.source_pdf_recovery.active_legacy_writers", return_value=[])
        guard.start()
        self.addCleanup(guard.stop)

    def pdf(self, title=None, doi=None):
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((30, 30), (title or self.row["title"]) + "\n" + (doi or self.row["doi"]))
            return document.tobytes()

    def run_recovery(self, content):
        with patch("rfs.judge_training.source_pdf_recovery._download",
                   return_value=(content, {"resolved_url": self.row["fulltext_source"] + ".pdf", "http_status": 200})):
            return recover_official_source_pdfs(self.batch)

    def test_recovery_preserves_input_and_resumes_without_network(self):
        before = self.strict.read_bytes(), self.text.read_bytes()
        content = self.pdf()
        result = self.run_recovery(content)
        self.assertTrue(result["ok"], result)
        self.assertEqual(content, self.target.read_bytes())
        self.assertEqual(before, (self.strict.read_bytes(), self.text.read_bytes()))
        self.assertTrue(result["papers"][0]["identity_check"]["doi_found"])
        self.assertIn(self.batch / "manifests/source_pdf_recovery.json", _input_files(self.batch))
        stamp = self.target.stat().st_mtime_ns
        with patch("rfs.judge_training.source_pdf_recovery._download") as download:
            self.assertTrue(recover_official_source_pdfs(self.batch)["ok"])
            download.assert_not_called()
        self.assertEqual(stamp, self.target.stat().st_mtime_ns)

    def test_a_wrong_doi_or_a_non_pdf_never_publishes(self):
        for content in (self.pdf(doi="10.1038/wrong"), b"<html>Login</html>", b"PK\x03\x04zip"):
            with self.subTest(content=content[:12]):
                self.assertFalse(self.run_recovery(content)["ok"])
                self.assertFalse(self.target.exists())

    def test_a_title_that_extracts_badly_is_reported_without_blocking_the_doi(self):
        result = self.run_recovery(self.pdf(title="Micro plit: semantic unmixing"))
        self.assertTrue(result["ok"], result)
        check = result["papers"][0]["identity_check"]
        self.assertTrue(check["doi_found"])
        self.assertFalse(check["title_found"], "a typographic title must be recorded, not silently claimed")
        self.assertTrue(self.target.exists())

    def test_verified_pdf_evidence_can_replace_failed_html_landing(self):
        self.row["publisher_landing_evidence"] = {"error": "HTML request failed"}
        self.row["official_pdf_evidence"] = {"method": "official_pdf_text",
            "resolved_url": self.row["fulltext_source"] + ".pdf", "title_found": True, "venue_found": True}
        write_jsonl(self.strict, [self.row])
        self.assertTrue(self.run_recovery(self.pdf())["ok"])

    def test_a_publisher_that_refuses_its_file_falls_back_to_the_page_text_copy(self):
        refused = _ArticleNotServed("Official PDF redirected away from declared article")
        with patch("rfs.judge_training.source_pdf_recovery._download_nature", side_effect=refused):
            result = recover_official_source_pdfs(self.batch, channels=COPY_CHANNELS)
        self.assertTrue(result["ok"], result)
        paper = result["papers"][0]
        self.assertEqual("official_html_text", paper["channel"])
        self.assertEqual("publisher_page_text_rendered", paper["transport"])
        self.assertEqual(self.row["fulltext_source"], paper["resolved_url"])
        self.assertEqual(hashlib.sha256(self.text.read_bytes()).hexdigest(), paper["source_text_sha256"])
        with pymupdf.open(self.target) as document:
            first_page = document[0].get_text()
        self.assertIn(self.row["doi"], first_page)
        self.assertIn(self.row["fulltext_source"], first_page)
        self.assertIn("not the publisher's PDF", first_page,
                      "a generated copy has to say that it is one")

    def test_a_transport_failure_never_becomes_a_page_copy(self):
        failure = ProviderRequestError("official_source_recovery", "failed", "connection reset")
        with patch("rfs.judge_training.source_pdf_recovery._download_nature", side_effect=failure):
            result = recover_official_source_pdfs(self.batch, channels=COPY_CHANNELS)
        self.assertFalse(result["ok"])
        self.assertEqual(1, result["failed"])
        self.assertFalse(self.target.exists(),
                         "a file that was never reached has to stay a failure to retry")

    def test_the_page_copy_is_refused_when_no_address_can_be_named(self):
        self.row["doi"] = "10.1371/journal.ppat.1014164"
        self.row["paper_id"] = self.row["doi"]
        self.row["fulltext_source"] = None
        self.row["publisher_landing_evidence"] = {}
        write_jsonl(self.strict, [self.row])
        with patch("rfs.judge_training.source_pdf_recovery._download",
                   side_effect=_ArticleNotServed("no file channel serves this record")):
            result = recover_official_source_pdfs(self.batch, channels=COPY_CHANNELS)
        self.assertFalse(result["ok"],
                         "a copy has to be a copy of an address this record can answer for")
        self.assertFalse(self.target.exists())

    def test_a_recorded_text_source_can_supply_the_copy(self):
        self.row["doi"] = "10.7554/elife.106452"
        self.row["paper_id"] = self.row["doi"]
        self.row["fulltext_source"] = "https://elifesciences.org/articles/106452"
        self.row["publisher_landing_evidence"] = {}
        self.text.write_text("Original full text for 10.7554/eLife.106452\nVerified framework",
                             encoding="utf-8")
        write_jsonl(self.strict, [self.row])
        refused = _ArticleNotServed("no file channel serves this record")
        with patch("rfs.judge_training.source_pdf_recovery._download_nature", side_effect=refused):
            result = recover_official_source_pdfs(self.batch, channels=COPY_CHANNELS)
        self.assertTrue(result["ok"], result)
        paper = result["papers"][0]
        self.assertEqual("official_html_text", paper["channel"])
        self.assertEqual("recorded_full_text_source", paper["address_source"])
        with pymupdf.open(self.target) as document:
            first_page = document[0].get_text()
        self.assertIn("the address this record recorded for its own text", first_page)
        self.assertIn("https://elifesciences.org/articles/106452", first_page)

    def test_a_nature_doi_names_the_page_a_copy_is_made_from(self):
        self.row["fulltext_source"] = None
        self.row["publisher_landing_evidence"] = {}
        write_jsonl(self.strict, [self.row])
        refused = _ArticleNotServed("Official PDF redirected away from declared article")
        with patch("rfs.judge_training.source_pdf_recovery._download_nature", side_effect=refused):
            result = recover_official_source_pdfs(self.batch, channels=COPY_CHANNELS)
        self.assertTrue(result["ok"], result)
        paper = result["papers"][0]
        self.assertEqual("official_html_text", paper["channel"])
        self.assertEqual("https://www.nature.com/articles/s41467-026-00001-x", paper["resolved_url"])
        self.assertEqual("doi_derived_article_url", paper["address_source"])
        with pymupdf.open(self.target) as document:
            first_page = document[0].get_text()
        self.assertIn("the DOI of this article", first_page)
        self.assertNotIn("captured from", first_page,
                         "a record that never recorded where its text came from must not claim one")

    def test_bytes_no_reader_can_open_yield_to_the_next_channel(self):
        broken = b"%PDF-1.7\ntruncated body that no reader can open"
        with patch("rfs.judge_training.source_pdf_recovery._download_nature",
                   return_value=(broken, {"resolved_url": self.row["fulltext_source"] + ".pdf",
                                          "http_status": 200})):
            result = recover_official_source_pdfs(self.batch, channels=COPY_CHANNELS)
        self.assertTrue(result["ok"], result)
        paper = result["papers"][0]
        self.assertEqual("official_html_text", paper["channel"])
        self.assertEqual("publisher_page_text_rendered", paper["transport"])
        self.assertTrue(paper["refused_channels"][0].startswith("nature_article:"),
                        "the channel that could not serve this paper has to stay on the record")
        with pymupdf.open(self.target) as document:
            self.assertIn(self.row["doi"], document[0].get_text())

    def test_an_available_pdf_is_never_replaced_by_a_page_copy(self):
        content = self.pdf()
        response = {"resolved_url": self.row["fulltext_source"] + ".pdf", "http_status": 200}
        with patch("rfs.judge_training.source_pdf_recovery._download_nature",
                   return_value=(content, response)):
            result = recover_official_source_pdfs(self.batch, channels=COPY_CHANNELS)
        self.assertTrue(result["ok"], result)
        self.assertEqual("nature_article", result["papers"][0]["channel"])
        self.assertEqual(content, self.target.read_bytes())

    def test_the_copy_channel_stays_off_until_it_is_asked_for(self):
        refused = _ArticleNotServed("Official PDF redirected away from declared article")
        with patch("rfs.judge_training.source_pdf_recovery._download_nature", side_effect=refused):
            result = recover_official_source_pdfs(self.batch)
        self.assertFalse(result["ok"])
        self.assertFalse(self.target.exists())

    def test_interrupted_checkpoint_recovers_only_identical_pdf(self):
        content = self.pdf()
        self.target.write_bytes(content)
        self.assertTrue(self.run_recovery(content)["ok"])
        self.target.write_bytes(b"different existing bytes")
        self.assertFalse(self.run_recovery(content)["ok"])
        self.assertEqual(b"different existing bytes", self.target.read_bytes())

    def test_an_unverified_url_is_never_used_and_a_derived_address_must_prove_itself(self):
        self.row["fulltext_source"] = "https://unverified.example/paper"
        write_jsonl(self.strict, [self.row])
        asked = {}

        def answer(spec):
            asked.update({"url": spec["url"], "address_source": spec["address_source"]})
            return self.pdf(doi="10.1038/someone-else"), {"resolved_url": spec["url"], "http_status": 200}

        with patch("rfs.judge_training.source_pdf_recovery._download", side_effect=answer):
            self.assertFalse(recover_official_source_pdfs(self.batch)["ok"])
        self.assertEqual("https://www.nature.com/articles/s41467-026-00001-x.pdf", asked["url"])
        self.assertEqual("doi_derived_article_url", asked["address_source"])
        self.assertFalse(self.target.exists(),
                         "a derived address must still prove identity inside the bytes it returns")
        with patch("rfs.judge_training.source_pdf_recovery.active_legacy_writers", return_value=[{"pid": 123}]):
            with self.assertRaises(LiteratureBusyError):
                recover_official_source_pdfs(self.batch)

    def test_a_nature_doi_names_its_article_without_any_recorded_url(self):
        self.row.pop("fulltext_source")
        self.row["publisher_landing_evidence"] = {"error": "HTML request failed"}
        write_jsonl(self.strict, [self.row])
        content = self.pdf()
        with patch("rfs.judge_training.source_pdf_recovery._download",
                   return_value=(content, {"resolved_url": "https://www.nature.com/articles/s41467-026-00001-x.pdf",
                                          "http_status": 200})):
            result = recover_official_source_pdfs(self.batch)
        self.assertTrue(result["ok"], result)
        self.assertEqual("doi_derived_article_url", result["papers"][0]["address_source"])
        self.assertEqual(content, self.target.read_bytes())

    def pmc_row(self, pmcid="PMC13345934"):
        folder = self.batch / "assets/biology/official/pmc"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "text").mkdir(exist_ok=True)
        (folder / "text/full.txt").write_text("PMC-derived full text", encoding="utf-8")
        return {"paper_id": "10.1038/s41592-026-03128-4", "doi": "10.1038/s41592-026-03128-4",
                "title": self.row["title"], "domain": "biology",
                "figure_asset_tier": "B_official_standalone",
                "full_text_path": "assets/biology/official/pmc/text/full.txt",
                "paper_type_evidence": {"pmcid": pmcid, "source": "europe_pmc"}}

    def test_pmc_record_recovers_through_the_verified_identifier(self):
        row = self.pmc_row()
        write_jsonl(self.strict, [row])
        content = self.pdf(doi=row["doi"])
        response = {"resolved_url": "https://pmc-oa-opendata.s3.amazonaws.com/PMC13345934.1/PMC13345934.1.pdf",
                    "http_status": 200, "transport": "pmc_cloud_package", "proof_of_work": None}
        with patch("rfs.judge_training.source_pdf_recovery._download", return_value=(content, response)):
            result = recover_official_source_pdfs(self.batch)
        self.assertTrue(result["ok"], result)
        self.assertEqual("pmc_oa", result["papers"][0]["channel"])
        self.assertEqual("PMC13345934", result["papers"][0]["pmcid"])
        target = self.batch / "assets/biology/official/pmc/official.pdf"
        self.assertEqual(content, target.read_bytes())

    def test_a_record_with_neither_a_pmc_identifier_nor_a_nature_doi_is_ineligible(self):
        row = self.pmc_row()
        row["paper_type_evidence"] = {"source": "europe_pmc"}
        row["doi"] = row["paper_id"] = "10.1109/tmi.2026.0001"
        write_jsonl(self.strict, [row])
        with patch("rfs.judge_training.source_pdf_recovery._download") as download:
            report = recover_official_source_pdfs(self.batch)
            download.assert_not_called()
        self.assertEqual((1, 0), (report["ineligible"], report["failed"]))

    def test_a_record_without_a_pmc_identifier_falls_back_to_its_nature_doi(self):
        row = self.pmc_row()
        row["paper_type_evidence"] = {"source": "europe_pmc"}
        write_jsonl(self.strict, [row])
        asked = {}

        def answer(spec):
            asked.update({"url": spec["url"], "address_source": spec["address_source"]})
            content = self.pdf(doi=row["doi"])
            asked["content"] = content
            return content, {"resolved_url": spec["url"], "http_status": 200}

        with patch("rfs.judge_training.source_pdf_recovery._download", side_effect=answer):
            report = recover_official_source_pdfs(self.batch)
        self.assertTrue(report["ok"], report)
        self.assertEqual("https://www.nature.com/articles/s41592-026-03128-4.pdf", asked["url"])
        self.assertEqual("doi_derived_article_url", asked["address_source"])
        self.assertEqual(asked["content"],
                         (self.batch / "assets/biology/official/pmc/official.pdf").read_bytes())

    def test_a_verified_pmc_identifier_outranks_a_derived_nature_address(self):
        row = self.pmc_row()
        write_jsonl(self.strict, [row])
        spec = _request_spec(self.batch, row)
        self.assertEqual("pmc_oa", spec["channel"])
        self.assertEqual("PMC13345934", spec["pmcid"])

    def test_a_ledger_path_written_on_windows_still_resolves_elsewhere(self):
        row = self.pmc_row()
        row["full_text_path"] = "assets\\biology\\official\\pmc\\text\\full.txt"
        write_jsonl(self.strict, [row])
        content = self.pdf(doi=row["doi"])
        response = {"resolved_url": "https://pmc-oa-opendata.s3.amazonaws.com/PMC13345934.1/PMC13345934.1.pdf",
                    "http_status": 200, "transport": "pmc_cloud_package", "proof_of_work": None}
        with patch("rfs.judge_training.source_pdf_recovery._download", return_value=(content, response)):
            result = recover_official_source_pdfs(self.batch)
        self.assertTrue(result["ok"], result)
        self.assertEqual(content, (self.batch / "assets/biology/official/pmc/official.pdf").read_bytes())
        self.assertEqual("assets\\biology\\official\\pmc\\text\\full.txt",
                         json.loads(self.strict.read_text(encoding="utf-8").splitlines()[0])["full_text_path"])

    def test_pmc_package_prefers_the_article_pdf_over_figures_and_older_versions(self):
        listing = (b'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult>'
                   b"<Contents><Key>PMC13345934.1/41592_2026_3128_Fig1_HTML.jpg</Key></Contents>"
                   b"<Contents><Key>PMC13345934.1/41592_2026_3128_MOESM1_ESM.pdf</Key></Contents>"
                   b"<Contents><Key>PMC13345934.1/PMC13345934.1.pdf</Key></Contents>"
                   b"<Contents><Key>PMC13345934.2/PMC13345934.2.pdf</Key></Contents>"
                   b"<Contents><Key>PMC13345935.1/PMC13345935.1.pdf</Key></Contents>"
                   b"</ListBucketResult>")
        self.assertEqual("PMC13345934.2/PMC13345934.2.pdf", _pmc_package_key(listing, "PMC13345934"))
        with self.assertRaises(ValueError):
            _pmc_package_key(listing, "PMC99999999")
        with self.assertRaises(ValueError):
            _pmc_package_key(b"<html>not a listing</html>", "PMC13345934")

    def test_pmc_address_guard_rejects_foreign_hosts_and_rewritten_paths(self):
        good = "https://pmc.ncbi.nlm.nih.gov/articles/PMC13345934/pdf/41592_2026_3128.pdf"
        self.assertEqual(good, _pmc_pdf_url(good, "PMC13345934"))
        for bad in (good.replace("pmc.ncbi.nlm.nih.gov", "evil.example"), good + "?x=1",
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC13345934/pdf/",
                    good.replace("PMC13345934", "PMC99999999")):
            with self.subTest(url=bad):
                with self.assertRaises(ValueError):
                    _pmc_pdf_url(bad, "PMC13345934")

    def test_proof_of_work_reproduces_the_page_rule_and_rejects_a_wild_difficulty(self):
        page = (b'const POW_CHALLENGE = "abc"\nconst POW_DIFFICULTY = "3"\n'
                b'const POW_COOKIE_NAME = "cloudpmc-viewer-pow"')
        self.assertEqual(("abc", 3, "cloudpmc-viewer-pow"), _pow_challenge(page))
        challenge, difficulty, _ = _pow_challenge(page)
        nonce = _pow_nonce(challenge, difficulty)
        digest = hashlib.sha256((challenge + str(nonce)).encode()).hexdigest()
        self.assertTrue(digest.startswith("0" * difficulty))
        earlier = [value for value in range(nonce)
                   if hashlib.sha256((challenge + str(value)).encode()).hexdigest().startswith("0" * difficulty)]
        self.assertEqual([], earlier, "the first matching nonce must be the one returned")
        with self.assertRaises(ValueError):
            _pow_challenge(b'const POW_CHALLENGE = "abc"\nconst POW_DIFFICULTY = "9"')
        with self.assertRaises(ValueError):
            _pow_challenge(b"<html>no challenge here</html>")

    def test_the_cloud_transport_takes_the_direct_address_without_listing(self):
        direct = "https://pmc-oa-opendata.s3.amazonaws.com/PMC13345934.1/PMC13345934.1.pdf"
        client = _FakeCloudClient({direct: _FakeResponse(direct, b"%PDF-direct%")})
        content, response = _download_pmc_cloud(client, {"pmcid": "PMC13345934"})
        self.assertEqual(b"%PDF-direct%", content)
        self.assertEqual([direct], client.calls,
                         "the bucket must not be listed when the address answers")
        self.assertEqual(direct, response["resolved_url"])

    def test_the_cloud_transport_lists_only_when_the_direct_address_is_absent(self):
        direct = "https://pmc-oa-opendata.s3.amazonaws.com/PMC13345934.1/PMC13345934.1.pdf"
        bucket = "https://pmc-oa-opendata.s3.amazonaws.com/"
        resolved = "https://pmc-oa-opendata.s3.amazonaws.com/PMC13345934.3/PMC13345934.3.pdf"
        listing = (b'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult>'
                   b"<Contents><Key>PMC13345934.1/PMC13345934.1.pdf</Key></Contents>"
                   b"<Contents><Key>PMC13345934.3/PMC13345934.3.pdf</Key></Contents>"
                   b"</ListBucketResult>")
        client = _FakeCloudClient({bucket: _FakeResponse(bucket, listing),
                                   resolved: _FakeResponse(resolved, b"%PDF-listed%")})
        content, response = _download_pmc_cloud(client, {"pmcid": "PMC13345934"})
        self.assertEqual(b"%PDF-listed%", content)
        self.assertEqual([direct, bucket, resolved], client.calls)
        self.assertEqual(resolved, response["resolved_url"])
