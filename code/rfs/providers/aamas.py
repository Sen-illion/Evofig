"""Official AAMAS proceedings adapter for full research papers."""

from __future__ import annotations

from html import unescape
import re
from typing import Any
from urllib.parse import urljoin

from .literature import LiteratureHttpClient, ProviderPage, clean, normalized_title


class AamasProceedingsProvider:
    """Read full papers from the official AAMAS Research Paper Track."""

    name = "aamas_proceedings"

    def __init__(self, client: LiteratureHttpClient | None = None) -> None:
        self.client = client or LiteratureHttpClient(self.name, min_interval=0.25, timeout=120)

    def list_works(self, *, year: int) -> ProviderPage:
        listing = f"https://www.ifaamas.org/Proceedings/aamas{year}/forms/contents.htm"
        response = self.client.request(listing)
        text = response.text
        start = re.search(r'<a\b[^>]*(?:name|id)=["\']R["\'][^>]*>', text, re.I)
        if start is None:
            return ProviderPage(
                provider=self.name,
                status="not_yet_published",
                request={"url": listing, "year": year},
                evidence={"official_listing": listing, "missing_track": "Research Paper Track"},
            )
        tail = text[start.end() :]
        end = re.search(r'<a\b[^>]*(?:name|id)=["\'](?:E|1)["\'][^>]*>', tail, re.I)
        section = tail[: end.start()] if end else tail
        records: list[dict[str, Any]] = []
        pattern = re.compile(
            r'<a\b[^>]*href=["\']([^"\']+\.pdf)["\'][^>]*>(.*?)</a>',
            re.I | re.S,
        )
        for match in pattern.finditer(section):
            pdf_url = urljoin(listing, unescape(match.group(1)))
            title = clean(unescape(match.group(2)))
            if len(title) < 12:
                continue
            paper_id = pdf_url.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            page_match = re.search(
                r'\(Page\s+(\d+)\)',
                section[match.end() : match.end() + 500],
                re.I,
            )
            records.append(
                {
                    "title": title,
                    "title_norm": normalized_title(title),
                    "doi": "",
                    "official_paper_id": f"aamas:{year}:{paper_id}",
                    "official_url": pdf_url,
                    "official_pdf_url": pdf_url,
                    "publication_date": str(year),
                    "venue": "AAMAS",
                    "record_type": "full",
                    "provider": self.name,
                    "official_proof": True,
                    "raw_evidence": {
                        "official_listing": listing,
                        "track": "Research Paper Track",
                        "start_page": int(page_match.group(1)) if page_match else None,
                    },
                }
            )
        return ProviderPage(
            provider=self.name,
            records=records,
            request={"url": listing, "year": year},
            evidence={"official_listing": listing, "track": "Research Paper Track"},
        )


__all__ = ["AamasProceedingsProvider"]
