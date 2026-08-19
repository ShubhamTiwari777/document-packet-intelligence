"""Named deterministic features, including page-number and header/footer signals."""
from __future__ import annotations

import re
from src.domain import PageRepresentation
from src.features.text_features import tokenize

PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?(\d{1,4})(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.I)


def _edge(page: PageRepresentation, top: bool) -> str:
    if not page.blocks:
        return ""
    cutoff = page.height * (0.18 if top else 0.82)
    chosen = [block.text for block in page.blocks if len(block.bbox) == 4 and ((block.bbox[1] <= cutoff) if top else (block.bbox[3] >= cutoff))]
    return " ".join(chosen)


def edge_similarity(left: str, right: str) -> float:
    a, b = set(tokenize(left)), set(tokenize(right))
    return len(a & b) / len(a | b) if a or b else 0.0


def extract_page_number(page: PageRepresentation) -> int | None:
    """Printed page numbers land in either the header or the footer depending on the
    template, so check both edges rather than assuming a footer-only convention."""
    lines = [line for line in page.text.splitlines() if line.strip()]
    for line in lines[:4] + lines[-4:]:
        match = PAGE_NUMBER.match(line)
        if match:
            return int(match.group(1))
    return None


def pair_heuristics(left: PageRepresentation, right: PageRepresentation) -> dict[str, float]:
    left_no, right_no = extract_page_number(left), extract_page_number(right)
    return {
        "page_number_reset": float(left_no is not None and right_no is not None and right_no <= left_no),
        "page_number_continuation": float(left_no is not None and right_no == left_no + 1),
        "header_similarity": edge_similarity(_edge(left, True), _edge(right, True)),
        "footer_similarity": edge_similarity(_edge(left, False), _edge(right, False)),
    }
