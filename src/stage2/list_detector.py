"""List detection operating on lines, not whole blocks.

PyMuPDF frequently returns an entire bulleted list as ONE block, so the previous check --
`block.text.startswith(("-", "*", "•"))` -- classified a multi-item list as a single
paragraph and lost the item structure entirely. Splitting the block into lines first lets a
run of marker-prefixed lines be recovered as a genuine list element with its items preserved.
"""
from __future__ import annotations

import re

BULLET = re.compile(r"^\s*[-*•·▪●‣⁃]\s+(.*\S)\s*$")
NUMBERED = re.compile(r"^\s*(?:\(?\d{1,3}[.)]|[a-z][.)])\s+(.*\S)\s*$", re.I)
MIN_ITEMS = 2


def _item(line: str) -> str | None:
    for pattern in (BULLET, NUMBERED):
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    return None


def match_item(text: str) -> str | None:
    """Return the item body when `text` is a single list-item line.

    Extractors commonly emit each list item as its own block, so the parser needs to recognise
    an item in isolation and stitch consecutive ones back into a single list.
    """
    stripped = [line for line in text.splitlines() if line.strip()]
    if len(stripped) != 1:
        return None
    return _item(stripped[0])


def is_ordered_item(text: str) -> bool:
    """True when the line uses an enumerator ("1.", "a)") rather than a bullet glyph."""
    stripped = [line for line in text.splitlines() if line.strip()]
    if not stripped:
        return False
    return bool(NUMBERED.match(stripped[0])) and not BULLET.match(stripped[0])


def is_ordered_block(text: str) -> bool:
    """True when a multi-line block reads as an enumerated list."""
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and sum(1 for line in lines if NUMBERED.match(line)) >= max(1, len(lines) * 0.6)


def split_list_items(text: str) -> list[str]:
    """Return list items when `text` reads as a list, else an empty list."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < MIN_ITEMS:
        return []
    items = [_item(line) for line in lines]
    recognized = [item for item in items if item is not None]
    # Require most lines to carry a marker so a paragraph that merely opens with a dash
    # is not mistaken for a list.
    if len(recognized) >= MIN_ITEMS and len(recognized) >= len(lines) * 0.6:
        return recognized
    return []
