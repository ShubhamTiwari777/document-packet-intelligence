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


# How many opening lines of a page count as its "header block" for date/opener detection.
HEAD_LINES = 5

SENTENCE_END = re.compile(r"[.!?:;,]['\")\]]?\s*$")
DATE_START = re.compile(r"^\s*(?:\d{1,2}[-/. ]\d{1,2}[-/. ]\d{2,4}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},\s*\d{4})")
DOCUMENT_OPENER = re.compile(r"^\s*(?:dear\b|geachte\b|betreft\b|subject\b|to\b|from\b|re:|memorandum\b|invoice\b|contract\b|agreement\b|report\b)", re.I)


def _lines(page: PageRepresentation) -> list[str]:
    return [line.strip() for line in page.text.splitlines() if line.strip()]


def continuation_features(left: PageRepresentation, right: PageRepresentation) -> dict[str, float]:
    """Sentence-flow signals across the page break.

    For OCR-only sources these are the strongest available evidence, because they need no layout
    metadata: prose that runs across a page break is almost never a document boundary, while a
    page opening with a salutation, a date line, or a capitalised title usually starts one. The
    existing feature set had nothing capturing this, which is why eight of fourteen features were
    inert on OpenPSS.
    """
    left_lines, right_lines = _lines(left), _lines(right)
    last = left_lines[-1] if left_lines else ""
    first = right_lines[0] if right_lines else ""
    # Opening furniture is almost never on line 1: a letter runs letterhead -> date -> address
    # block -> salutation, and a memo runs title -> TO/FROM/SUBJECT. Matching only the first line
    # left `starts_with_date` and `starts_with_opener` constant zero on entire packets that plainly
    # contained both cues, so they could carry no information. Scan the opening block instead.
    head = right_lines[:HEAD_LINES]

    # Unterminated last line + lowercase opener = the sentence flows on = same document.
    ends_open = bool(last) and not SENTENCE_END.search(last)
    starts_lower = bool(first) and first[0].islower()
    starts_upper = bool(first) and first[0].isupper()

    return {
        "ends_mid_sentence": float(ends_open),
        "starts_lowercase": float(starts_lower),
        "sentence_continues": float(ends_open and starts_lower),
        "starts_new_sentence": float((not ends_open) and starts_upper),
        "starts_with_date": float(any(DATE_START.match(line) for line in head)),
        "starts_with_opener": float(any(DOCUMENT_OPENER.match(line) for line in head)),
        "first_line_similarity": edge_similarity(first, left_lines[0] if left_lines else ""),
    }


def pair_heuristics(left: PageRepresentation, right: PageRepresentation) -> dict[str, float]:
    left_no, right_no = extract_page_number(left), extract_page_number(right)
    return {
        "page_number_reset": float(left_no is not None and right_no is not None and right_no <= left_no),
        "page_number_continuation": float(left_no is not None and right_no == left_no + 1),
        "header_similarity": edge_similarity(_edge(left, True), _edge(right, True)),
        "footer_similarity": edge_similarity(_edge(left, False), _edge(right, False)),
        **continuation_features(left, right),
    }
