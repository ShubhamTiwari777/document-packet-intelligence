"""Heading detection and hierarchy assignment from font/layout evidence.

The previous detector compared each block against a single page-median multiplier and could
only ever emit level 1 or 2, so it could not express real document hierarchy. This version
first collects the distinct heading font sizes across the whole document, then ranks them --
largest size becomes H1, next distinct size H2, and so on. Ranking within the document is what
makes the level meaningful: an 18pt line is only an H1 relative to the other headings in *that*
document, not against an absolute point size.
"""
from __future__ import annotations

import re

from src.domain import PageRepresentation, TextBlock

MAX_HEADING_CHARS = 120
MAX_HEADING_WORDS = 15
MAX_LEVELS = 4

_NUMBER_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+")
_SENTENCE_END = re.compile(r"[.!?]\s*$")


def looks_like_title(text: str) -> bool:
    """Shape test for a heading line.

    This replaced a character allow-list regex that had to enumerate every permitted symbol.
    That approach failed twice in practice: it demanded a leading capital LETTER (so every
    numbered heading such as "3.3 Example Measurement Table" was rejected), and it did not list
    the separator glyph a PDF font actually emitted for an em-dash, rejecting
    "Appendix - Expected Structure". Discriminative power comes from length, single-line shape
    and the font/case evidence in `is_heading`, so the shape test only checks that the line is
    short, single-line, and begins with a capital after any section number.
    """
    if "\n" in text.strip():
        return False  # headings are single-line; a multi-line block is body copy
    stripped = " ".join(text.split())
    if not stripped or len(stripped) > MAX_HEADING_CHARS or len(stripped.split()) > MAX_HEADING_WORDS:
        return False
    body = _NUMBER_PREFIX_RE.sub("", stripped, count=1).strip()
    return bool(body) and body[0].isupper()


def _block_size(block: TextBlock) -> float:
    return max(block.font_sizes, default=0.0)


def _is_bold(block: TextBlock) -> bool:
    # PyMuPDF span flag bit 4 (value 16) marks a bold face.
    return any(int(flag) & 16 for flag in block.flags)


def is_heading(block: TextBlock, page_median_font: float) -> bool:
    """Heuristic heading test for a single block."""
    text = block.text.strip()
    if not text or len(text) > MAX_HEADING_CHARS or len(text.split()) > MAX_HEADING_WORDS:
        return False
    if _SENTENCE_END.search(text):
        return False  # prose sentences are not headings even when short
    size = _block_size(block)
    letters = text.replace(" ", "")
    if not letters:
        return False
    mostly_upper = sum(char.isupper() for char in letters) >= max(3, len(letters) * 0.6)
    larger = size > page_median_font * 1.18
    title_like = looks_like_title(text)
    return (larger and title_like) or (mostly_upper and size >= page_median_font) or (_is_bold(block) and title_like and size >= page_median_font)


_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+){0,3})[.)]?\s+(\S.{0,78})$")


def is_text_heading(text: str) -> bool:
    """Heading test for sources with no font information (OCR-only input).

    Relies purely on typographic convention: headings are short, do not terminate like a
    sentence, and are either numbered, ALL-CAPS, or Title Case.
    """
    stripped = text.strip()
    lines = [line for line in stripped.splitlines() if line.strip()]
    if len(lines) != 1:
        return False
    line = lines[0].strip()
    words = line.split()
    if not line or len(line) > 90 or not (1 <= len(words) <= 12):
        return False
    if _SENTENCE_END.search(line):
        return False
    if _NUMBERED_HEADING.match(line):
        return True
    letters = [char for char in line if char.isalpha()]
    if len(letters) < 3:
        return False
    if all(char.isupper() for char in letters):
        return True
    # A lone capitalised word is a common section heading ("Introduction", "Scope", "Results").
    # The previous `capitalised >= max(2, ...)` rule was unsatisfiable for a single word, so
    # every one-word heading was missed by this fallback.
    if len(words) == 1:
        return words[0][:1].isupper() and len(words[0]) >= 3
    # Title Case: most words capitalised, and not a run-on prose fragment.
    capitalised = sum(1 for word in words if word[:1].isupper())
    return len(words) <= 8 and capitalised >= max(2, len(words) * 0.7)


def numbering_level(text: str) -> int | None:
    """Hierarchy depth from a section number ("3.3 Example" -> 2), or None when unnumbered."""
    match = _NUMBERED_HEADING.match(text.strip())
    if not match:
        return None
    return min(len(match.group(1).split(".")), MAX_LEVELS)


def text_heading_level(text: str) -> int:
    """Depth from a numeric prefix ("2.1 Scope" -> level 2), else level 1."""
    return numbering_level(text) or 1


def heading_levels(sizes: list[float]) -> dict[float, int]:
    """Map each distinct heading font size to a 1-based level, largest size first."""
    ranked = sorted({round(size, 1) for size in sizes if size > 0}, reverse=True)
    return {size: min(index, MAX_LEVELS) for index, size in enumerate(ranked, start=1)}


def collect_heading_sizes(pages: list[PageRepresentation], page_medians: dict[int, float], skip: set[str], normalizer) -> list[float]:
    """Gather font sizes of every block that looks like a heading and is not boilerplate."""
    sizes: list[float] = []
    for page in pages:
        for block in page.blocks:
            if normalizer(block.text) in skip:
                continue
            if is_heading(block, page_medians.get(page.page_number, 0.0)):
                sizes.append(_block_size(block))
    return sizes
