"""Text-only structural fallback for sources without layout metadata.

The layout-driven parser needs per-block bounding boxes and font sizes. Native PDFs supply
both, but OCR-only sources (scanned packets, and the OpenPSS corpus used for boundary
evaluation) supply a bare text string -- and with no blocks the parser emitted a single
undifferentiated section per document, i.e. no structure at all.

This module reconstructs pseudo-blocks from raw text: paragraphs are split on blank lines and
each is given a positional bounding box derived from where it falls down the page. The boxes
are approximations, but they are sufficient for the margin-based running-header detection, and
they let a single downstream code path serve both native and OCR inputs.
"""
from __future__ import annotations

from src.domain import PageRepresentation, TextBlock


def synthesize_blocks(page: PageRepresentation) -> list[TextBlock]:
    """Rebuild block structure from plain text using blank-line paragraph separation."""
    raw_lines = page.text.splitlines()
    if not raw_lines:
        return []

    groups: list[tuple[int, int, list[str]]] = []  # (first_line, last_line, lines)
    current: list[str] = []
    start_index = 0
    for index, line in enumerate(raw_lines):
        if line.strip():
            if not current:
                start_index = index
            current.append(line.strip())
        elif current:
            groups.append((start_index, index - 1, current))
            current = []
    if current:
        groups.append((start_index, len(raw_lines) - 1, current))

    total = max(len(raw_lines), 1)
    height = page.height or 792.0
    width = page.width or 612.0
    blocks: list[TextBlock] = []
    for first, last, lines in groups:
        top = height * (first / total)
        bottom = height * ((last + 1) / total)
        blocks.append(TextBlock(text="\n".join(lines), bbox=[0.0, top, width, bottom], font_sizes=[], font_names=[], flags=[]))
    return blocks
