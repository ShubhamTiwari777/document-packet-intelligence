"""Running header/footer (boilerplate) detection.

Multi-page documents repeat a letterhead and a page-number line on every page. Left in the
structure these cause three concrete problems, all observed in the Stage 1 sample output:

* the letterhead is picked up as a heading on every page, producing duplicate sections
  ("INVOICE" x3) that fragment the document;
* the page-number block merges into the following real heading ("Page 2" + "Service Details"
  -> "Page 2Service Details"), corrupting section titles and breadcrumbs;
* the repeated text becomes its own tiny chunk, which pollutes retrieval with near-identical
  low-information candidates.

Detection is positional + frequency based: a block in the top/bottom margin whose normalized
text repeats across most pages of the *same document* is boilerplate. Scoping to the document
(not the packet) matters -- two different documents legitimately share nothing, but pages of
one document share their letterhead.
"""
from __future__ import annotations

from collections import defaultdict
import re

from src.domain import PageRepresentation, TextBlock

# Digits are normalized away so "Page 1"/"Page 2" collapse to one repeated key.
_DIGITS = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")

TOP_MARGIN = 0.18
BOTTOM_MARGIN = 0.82
MIN_PAGES = 2
REPEAT_RATIO = 0.6


def normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", _DIGITS.sub("#", text)).strip().lower()


def _in_margin(block: TextBlock, height: float) -> bool:
    if len(block.bbox) != 4 or height <= 0:
        return False
    return block.bbox[1] <= height * TOP_MARGIN or block.bbox[3] >= height * BOTTOM_MARGIN


def detect_boilerplate(pages: list[PageRepresentation]) -> set[str]:
    """Return normalized texts that behave as running headers/footers across `pages`."""
    if len(pages) < MIN_PAGES:
        return set()
    seen: dict[str, set[int]] = defaultdict(set)
    for page in pages:
        for block in page.blocks:
            if _in_margin(block, page.height):
                key = normalize(block.text)
                if key:
                    seen[key].add(page.page_number)
    threshold = max(MIN_PAGES, int(len(pages) * REPEAT_RATIO))
    return {key for key, page_numbers in seen.items() if len(page_numbers) >= threshold}


def is_boilerplate(block: TextBlock, boilerplate: set[str]) -> bool:
    return normalize(block.text) in boilerplate
