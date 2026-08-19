"""Figure/table caption detection.

Captions were previously indistinguishable from body paragraphs, so "Figure 1. Conceptual
flow ..." was stored as ordinary prose. Marking them matters for retrieval: a caption
describes an adjacent visual rather than asserting document content, and downstream ranking
benefits from being able to treat it differently.

Text-based only -- no image understanding is required to recognise a caption.
"""
from __future__ import annotations

import re

CAPTION = re.compile(
    r"^\s*(?P<label>(?:figure|fig|table|chart|exhibit|diagram|image|plate)\s*\.?\s*\d+[a-z]?)\s*[.:—–-]?\s+\S",
    re.IGNORECASE,
)
MAX_CAPTION_CHARS = 400


def caption_label(text: str) -> str | None:
    """Return the caption label ("Figure 1") when `text` reads as a caption, else None."""
    stripped = " ".join(text.split())
    if not stripped or len(stripped) > MAX_CAPTION_CHARS:
        return None
    match = CAPTION.match(stripped)
    if not match:
        return None
    return " ".join(match.group("label").split())


def is_caption(text: str) -> bool:
    return caption_label(text) is not None
