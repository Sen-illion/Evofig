"""Paper and reference analysis entry points for the direct pipeline."""

from .paper.document import parse_paper
from .paper.ocr import extract_reference_text

__all__ = ["extract_reference_text", "parse_paper"]
