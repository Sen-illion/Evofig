"""Tests for true official standalone B-tier assets."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from rfs.providers.literature import ProviderRequestError
from rfs.judge_training.official_assets import (
    _bind_route_assets,
    _figure_routes,
    _official_probe_url,
    _resolution_ladder,
    bind_official_asset,
    parse_html_figures,
    parse_html_fulltext,
    parse_jats_figures,
)


# One figure of a Springer-platform paper as the PMC JATS asset route names it: the
# narrowest spelling of an object the publisher also serves at full width.
SPRINGER_GRAPHIC = (
    "https://media.springernature.com/lw685/springer-static/image"
    "/art%3A10.1186%2Fs13059-025-03917-8/MediaObjects/13059_2025_3917_Fig1_HTML.png"
)


class OfficialAssetTests(unittest.TestCase):
    def test_probe_url_prefers_verified_publisher_landing(self):
        row = {
            "doi": "10.1609/aaai.v40i38.40478",
            "official_url": "https://doi.org/10.1609/aaai.v40i38.40478",
            "publisher_landing_evidence": {
                "resolved_url": "https://ojs.aaai.org/index.php/AAAI/article/view/40478",
                "title_found": True,
                "venue_found": True,
            },
        }
        self.assertEqual(
            _official_probe_url(row),
            "https://ojs.aaai.org/index.php/AAAI/article/view/40478",
        )

    def test_probe_url_bypasses_doi_redirect_for_acm(self):
        row = {
            "doi": "10.1145/3767695.3769670",
            "official_url": "https://doi.org/10.1145/3767695.3769670",
        }
        self.assertEqual(
            _official_probe_url(row),
            "https://dl.acm.org/doi/10.1145/3767695.3769670",
        )

    def test_aaai_pdf_only_provider_is_terminal_without_network(self):
        class FailingClient:
            def request(self, _url):
                raise AssertionError("AAAI PDF-only landing must not be probed for B assets")

        row = {
            "paper_id": "10.1609/aaai.v40i38.40478",
            "doi": "10.1609/aaai.v40i38.40478",
            "title": "Paper",
            "official_source": "aaai_ojs",
            "official_url": "https://doi.org/10.1609/aaai.v40i38.40478",
            "publisher_landing_evidence": {
                "resolved_url": "https://ojs.aaai.org/index.php/AAAI/article/view/40478",
                "title_found": True,
                "venue_found": True,
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            result = bind_official_asset(Path(folder), "agent", row, client=FailingClient())
        self.assertEqual(result["status"], "no_eligible_standalone_figure")
        self.assertIn("forbids PDF", result["probe_note"])

    def test_html_figure_requires_method_caption(self):
        html = """
        <figure><img src='/framework.png'><figcaption>Overview of the proposed workflow</figcaption></figure>
        <figure><img src='/result.png'><figcaption>ROC curve result comparison</figcaption></figure>
        """
        rows = parse_html_figures(html, "https://publisher.test/paper")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["asset_source_urls"], ["https://publisher.test/framework.png"])

    def test_html_figure_keeps_the_panel_description_in_the_stored_caption(self):
        html = """
        <figure>
          <figcaption><b>Fig. 1: Schematic of the screening workflow.</b></figcaption>
          <picture><img src='/fig1.png'></picture>
          <div class='c-article-section__figure-link'><a href='/figures/1'>Full size image</a></div>
          <div data-test='bottom-caption'><p>a Cells were imaged by microscopy.</p></div>
        </figure>
        """
        rows = parse_html_figures(html, "https://www.nature.com/articles/x")
        self.assertEqual(len(rows), 1)
        # The gate reads the title; the panel text is still kept for later review.
        self.assertEqual(rows[0]["caption_title"], "Schematic of the screening workflow.")
        self.assertIn("imaged by microscopy", rows[0]["caption"])
        self.assertNotIn("Full size image", rows[0]["caption"])

    def test_html_figure_ignores_a_term_that_only_appears_in_a_panel(self):
        html = """
        <figure>
          <figcaption><b>Fig. 4: Example of a 2D deformation model.</b></figcaption>
          <picture><img src='/fig4.png'></picture>
          <div data-test='bottom-caption'><p>A schematic map view of the sill is shown in Fig. 7.</p></div>
        </figure>
        """
        self.assertEqual(parse_html_figures(html, "https://www.nature.com/articles/x"), [])

    def test_jats_binds_graphic_and_fulltext(self):
        xml = """<article xmlns:xlink='http://www.w3.org/1999/xlink'><body>
        <fig><caption><p>Experimental workflow and study design</p></caption><graphic xlink:href='fig1.jpg'/></fig>
        <p>Methods text</p></body></article>"""
        rows, fulltext = parse_jats_figures(xml, "https://pmc.test/article/")
        self.assertEqual(rows[0]["asset_source_urls"], ["https://pmc.test/article/fig1.jpg"])
        self.assertIn("Methods text", fulltext)

    def test_gate_recognises_physical_science_figure_vocabulary(self):
        from rfs.judge_training.official_assets import _caption_title, _eligible_caption

        for title in (
            "Fig. 1: Concept of the PTD-immobilized photocatalyst platform.",
            "Fig. 2: Working principle of the device.",
            "Fig. 3: Synthesis strategy for the covalent framework.",
        ):
            self.assertTrue(_eligible_caption(_caption_title(title, "")), title)
        for title in (
            "Sketch of the microbial community assembly model.",
            "Graphical Abstract illustrating the proposed mechanism.",
        ):
            self.assertTrue(_eligible_caption(_caption_title(title, "")), title)
        for title in (
            "Fig. 1: The characterizations of the catalysts.",
            "Fig. 2: Electrochemical ORR measurements.",
            "Cluster dispersal increases beta-diversity and decreases alpha-diversity.",
            "Centrally-projecting DRG axons experience a decline in regenerative capacity.",
        ):
            self.assertFalse(_eligible_caption(_caption_title(title, "")), title)

    def test_jats_gate_reads_the_caption_title_not_the_panel_text(self):
        admitted = """<article xmlns:xlink='http://www.w3.org/1999/xlink'><body>
        <fig><caption><title>Overview of the analysis pipeline.</title>
        <p>a Samples were imaged by microscopy and scored for accuracy.</p></caption>
        <graphic xlink:href='fig1.jpg'/></fig></body></article>"""
        rows, _ = parse_jats_figures(admitted, "https://pmc.test/article/")
        # A panel mentioning microscopy/accuracy no longer kills a real overview figure.
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["caption_title"], "Overview of the analysis pipeline.")
        self.assertIn("microscopy", rows[0]["caption"])

        rejected = """<article xmlns:xlink='http://www.w3.org/1999/xlink'><body>
        <fig><caption><title>Turn angle distributions for discretized motion.</title>
        <p>The schematic of the arena is shown in Fig. 1.</p></caption>
        <graphic xlink:href='fig2.jpg'/></fig></body></article>"""
        self.assertEqual(parse_jats_figures(rejected, "https://pmc.test/article/")[0], [])

    def test_html_fulltext_uses_article_content_only(self):
        body = "<nav>menu</nav><article><h1>Paper</h1><p>Methods and experiments</p></article>"
        self.assertEqual(parse_html_fulltext(body), "Paper Methods and experiments")

    def test_jats_skips_caption_equation_graphics(self):
        xml = """<article xmlns:xlink='http://www.w3.org/1999/xlink'><body>
        <fig><caption><p>Model architecture overview for
        <inline-formula><graphic xlink:href='paper.e003.jpg'/></inline-formula> inputs</p></caption>
        <graphic xlink:href='paper.g001.jpg'/></fig></body></article>"""
        rows, _ = parse_jats_figures(xml, "https://pmc.test/article/")
        self.assertEqual(rows[0]["asset_source_urls"], ["https://pmc.test/article/paper.g001.jpg"])

    def test_routes_put_publisher_doi_ahead_of_aggregator_landing(self):
        row = {
            "doi": "10.1038/s41467-025-67622-7",
            "official_url": "https://europepmc.org/article/MED/42224365",
            "paper_type_evidence": {"pmcid": "PMC13225645"},
        }
        self.assertEqual(
            [route["url"] for route in _figure_routes(row, row["official_url"])],
            [
                # No doi.org duplicate (it resolves to the nature.com page) and
                # no europepmc.org shell (it answers 403 and parses to nothing).
                "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13225645/fullTextXML",
                "https://www.nature.com/articles/s41467-025-67622-7",
            ],
        )

    def test_plos_graphics_gain_a_publisher_served_alternate(self):
        row = {
            "doi": "10.1371/journal.pcbi.1014069",
            "official_url": "https://europepmc.org/article/MED/1",
            "paper_type_evidence": {"pmcid": "PMC13446673"},
        }
        alias = _figure_routes(row, row["official_url"])[0]["asset_alias"]
        xml = """<article xmlns:xlink='http://www.w3.org/1999/xlink'><body>
        <fig><caption><p>Experimental workflow</p></caption>
        <graphic xlink:href='pcbi.1014069.g001.jpg'/></fig></body></article>"""
        rows, _ = parse_jats_figures(xml, "https://pmc.test/bin/", asset_alias=alias)
        self.assertEqual(
            rows[0]["asset_source_urls"],
            [
                "https://journals.plos.org/plosone/article/figure/image"
                "?size=large&id=10.1371/journal.pcbi.1014069.g001",
                "https://pmc.test/bin/pcbi.1014069.g001.jpg",
            ],
        )

    def test_springer_graphics_gain_a_publisher_served_alternate(self):
        row = {
            "doi": "10.1186/s13059-025-03917-8",
            "official_url": "https://europepmc.org/article/MED/1",
            "paper_type_evidence": {"pmcid": "PMC1"},
        }
        alias = _figure_routes(row, row["official_url"])[0]["asset_alias"]
        # The stored extension is unreliable; the served object is a PNG.
        self.assertEqual(
            alias("13059_2025_3917_Fig1_HTML.jpg"),
            [
                "https://media.springernature.com/lw685/springer-static/image"
                "/art%3A10.1186%2Fs13059-025-03917-8/MediaObjects/13059_2025_3917_Fig1_HTML.png"
            ],
        )
        # A non-Springer graphic name must not be rewritten onto their host.
        self.assertEqual(alias("gkaf1243fig1.jpg"), [])

    def test_a_springer_graphic_is_requested_widest_first(self):
        self.assertEqual(
            _resolution_ladder(SPRINGER_GRAPHIC),
            [
                SPRINGER_GRAPHIC.replace("/lw685/", "/full/"),
                SPRINGER_GRAPHIC.replace("/lw685/", "/original/"),
                SPRINGER_GRAPHIC.replace("/lw685/", "/lw1200/"),
                SPRINGER_GRAPHIC.replace("/lw685/", "/lw1000/"),
                SPRINGER_GRAPHIC,
            ],
        )

    def test_an_unmapped_host_keeps_its_asset_url(self):
        url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/bin/dead.jpg"
        self.assertEqual(_resolution_ladder(url), [url])

    def test_an_asset_already_at_its_widest_keeps_its_url(self):
        url = SPRINGER_GRAPHIC.replace("/lw685/", "/full/")
        self.assertEqual(_resolution_ladder(url), [url])

    def test_binder_stores_the_widest_spelling_that_answers(self):
        published = b"\x89PNG\r\n\x1a\n" + b"published-figure" * 4
        downscaled = b"\x89PNG\r\n\x1a\n" + b"downscaled-copy" * 4

        def request(url):
            payload = downscaled if url == SPRINGER_GRAPHIC else published
            return SimpleNamespace(url=url, headers={"content-type": "image/png"}, content=payload)

        figure = {"caption": "Model architecture overview", "asset_source_urls": [SPRINGER_GRAPHIC]}
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            candidates, saved, failures = _bind_route_assets(
                SimpleNamespace(request=request), batch, batch / "assets", [figure], "pmc_jats"
            )
        self.assertEqual(failures, [])
        self.assertEqual(Path(saved[0]).parts, ("assets", "figures", "figure_001.png"))
        self.assertEqual(candidates[0]["asset_source_url"], SPRINGER_GRAPHIC.replace("/lw685/", "/full/"))
        self.assertEqual(candidates[0]["asset_sha256"], hashlib.sha256(published).hexdigest())

    def test_binder_falls_back_to_the_spelling_the_route_advertised(self):
        downscaled = b"\x89PNG\r\n\x1a\n" + b"downscaled-copy" * 4

        def request(url):
            if url == SPRINGER_GRAPHIC:
                return SimpleNamespace(url=url, headers={"content-type": "image/png"}, content=downscaled)
            raise ProviderRequestError("p", "not_found", "provider returned HTTP 404")

        figure = {"caption": "Model architecture overview", "asset_source_urls": [SPRINGER_GRAPHIC]}
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            candidates, _saved, failures = _bind_route_assets(
                SimpleNamespace(request=request), batch, batch / "assets", [figure], "pmc_jats"
            )
        self.assertEqual(failures, [])
        self.assertEqual(candidates[0]["asset_source_url"], SPRINGER_GRAPHIC)

    def test_only_the_advertised_spelling_decides_the_failure_verdict(self):
        def request(_url):
            raise ProviderRequestError("p", "rate_limited", "provider returned HTTP 429")

        figure = {"caption": "Model architecture overview", "asset_source_urls": [SPRINGER_GRAPHIC]}
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            candidates, saved, failures = _bind_route_assets(
                SimpleNamespace(request=request), batch, batch / "assets", [figure], "pmc_jats"
            )
        self.assertEqual(candidates, [])
        self.assertEqual(saved, [])
        self.assertEqual(failures, ["rate_limited"])

    def test_unmapped_publisher_jats_keeps_only_the_pmc_asset(self):
        row = {"doi": "10.1093/nar/gkaf1", "official_url": "https://europepmc.org/article/MED/1",
               "paper_type_evidence": {"pmcid": "PMC1"}}
        self.assertIsNone(_figure_routes(row, row["official_url"])[0]["asset_alias"])

    def test_routes_fall_back_to_doi_redirector_for_unmapped_publisher(self):
        row = {"doi": "10.1093/bioinformatics/btaf001", "official_url": "https://europepmc.org/article/MED/1"}
        self.assertEqual(
            [route["url"] for route in _figure_routes(row, row["official_url"])][0],
            "https://doi.org/10.1093/bioinformatics/btaf001",
        )

    def test_routes_replace_the_doi_redirector_with_the_publisher(self):
        row = {"doi": "10.1038/s41467-025-66526-w", "official_url": "https://doi.org/10.1038/s41467-025-66526-w"}
        # Fetching both would hit nature.com twice, the second time off its
        # polite lane, which is what earned a host-wide 403.
        self.assertEqual(
            [route["url"] for route in _figure_routes(row, row["official_url"])],
            ["https://www.nature.com/articles/s41467-025-66526-w"],
        )

    def test_routes_keep_a_resolved_publisher_landing_ahead_of_the_doi(self):
        row = {
            "doi": "10.1016/j.actamat.2025.001",
            "official_url": "https://www.sciencedirect.com/science/article/pii/S1",
        }
        self.assertEqual(
            [route["url"] for route in _figure_routes(row, row["official_url"])],
            [
                "https://www.sciencedirect.com/science/article/pii/S1",
                "https://doi.org/10.1016/j.actamat.2025.001",
            ],
        )

    def test_binder_falls_back_to_publisher_when_pmc_assets_are_unusable(self):
        jats = (
            "<article xmlns:xlink='http://www.w3.org/1999/xlink'><body>"
            "<fig><caption><p>Model architecture overview</p></caption>"
            "<graphic xlink:href='dead.jpg'/></fig>"
            f"<p>{'Methods and experiments. ' * 400}</p></body></article>"
        )
        publisher = "<figure><img src='figure.png'><figcaption>Model architecture overview</figcaption></figure>"
        blocked = SimpleNamespace(
            text="", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/bin/dead.jpg",
            headers={"content-type": "text/html"}, content=b"<html>not an image</html>" * 4,
        )
        figure = SimpleNamespace(
            text="", url="https://www.nature.com/articles/s41467-025-67622-7/figure.png",
            headers={"content-type": "image/png"},
            content=b"\x89PNG\r\n\x1a\n" + b"valid-standalone-figure" * 2,
        )

        def request(url):
            if "fullTextXML" in url:
                return SimpleNamespace(text=jats, url=url, headers={}, content=b"")
            if url.endswith("dead.jpg"):
                return blocked
            if url.endswith("figure.png"):
                return figure
            return SimpleNamespace(
                text=publisher, url="https://www.nature.com/articles/s41467-025-67622-7",
                headers={"content-type": "text/html"}, content=b"html",
            )

        row = {
            "paper_id": "paper-1",
            "title": "Paper",
            "doi": "10.1038/s41467-025-67622-7",
            "official_url": "https://europepmc.org/article/MED/1",
            "paper_type_evidence": {"pmcid": "PMC1"},
        }
        with tempfile.TemporaryDirectory() as folder:
            result = bind_official_asset(Path(folder), "biology", row, client=SimpleNamespace(request=request))
        self.assertEqual(result["figure_asset_tier"], "B_official_standalone")
        self.assertEqual(result["figure_route"], "official_html_figure")
        self.assertEqual(result["figure_route_url"], "https://www.nature.com/articles/s41467-025-67622-7")
        # The dead PMC route still supplies the richest full text.
        self.assertEqual(result["fulltext_method"], "pmc_jats")

    def test_binder_reports_no_figure_when_every_route_is_exhausted(self):
        empty = SimpleNamespace(
            text="<html><body>no figures</body></html>", url="https://www.nature.com/articles/x",
            headers={"content-type": "text/html"}, content=b"html",
        )
        row = {
            "paper_id": "paper-2",
            "title": "Paper",
            "doi": "10.1038/x",
            "official_url": "https://europepmc.org/article/MED/2",
        }
        with tempfile.TemporaryDirectory() as folder:
            result = bind_official_asset(
                Path(folder), "biology", row, client=SimpleNamespace(request=lambda _url: empty)
            )
        self.assertEqual(result["status"], "no_eligible_standalone_figure")
        self.assertIsNone(result["figure_asset_tier"])

    def test_an_earlier_route_404_does_not_mask_a_served_article(self):
        served = SimpleNamespace(
            text="<figure><img src='/f.png'><figcaption>Results table</figcaption></figure>",
            url="https://www.nature.com/articles/x", headers={"content-type": "text/html"}, content=b"html",
        )

        def request(url):
            if "fullTextXML" in url:
                raise ProviderRequestError("p", "not_found", "provider returned HTTP 404")
            return served

        row = {"paper_id": "p", "title": "T", "doi": "10.1038/x",
               "official_url": "https://europepmc.org/article/MED/1", "paper_type_evidence": {"pmcid": "PMC1"}}
        with tempfile.TemporaryDirectory() as folder:
            result = bind_official_asset(Path(folder), "biology", row, client=SimpleNamespace(request=request))
        # The publisher page was served, so the caption gate is the real verdict.
        self.assertEqual(result["status"], "no_eligible_standalone_figure")

    def test_every_route_failing_prefers_the_recoverable_verdict(self):
        def request(url):
            if "fullTextXML" in url:
                raise ProviderRequestError("p", "not_found", "provider returned HTTP 404")
            raise ProviderRequestError("p", "rate_limited", "provider returned HTTP 429")

        row = {"paper_id": "p", "title": "T", "doi": "10.1038/x",
               "official_url": "https://europepmc.org/article/MED/1", "paper_type_evidence": {"pmcid": "PMC1"}}
        with tempfile.TemporaryDirectory() as folder:
            result = bind_official_asset(Path(folder), "biology", row, client=SimpleNamespace(request=request))
        self.assertEqual(result["status"], "retryable_rate_limited")

    def test_binder_never_labels_pdf_object_extraction_as_b(self):
        page = SimpleNamespace(
            text="<figure><img src='figure.png'><figcaption>Model architecture overview</figcaption></figure>",
            url="https://publisher.test/article",
            headers={"content-type": "text/html"},
            content=b"html",
        )
        asset = SimpleNamespace(
            text="",
            url="https://publisher.test/figure.png",
            headers={"content-type": "image/png"},
            content=b"\x89PNG\r\n\x1a\n" + b"valid-standalone-figure" * 2,
        )
        client = SimpleNamespace(request=lambda url: asset if url.endswith("figure.png") else page)
        row = {
            "paper_id": "paper-1",
            "title": "Paper",
            "official_url": "https://publisher.test/article",
        }
        with tempfile.TemporaryDirectory() as folder:
            result = bind_official_asset(Path(folder), "agent", row, client=client)
        self.assertEqual(result["figure_asset_tier"], "B_official_standalone")
        self.assertNotIn("C_pdf_object_extracted", str(result))


if __name__ == "__main__":
    unittest.main()
