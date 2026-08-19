"""Table extraction via pdfplumber, gated by a cheap text-layout heuristic.

Previously this module was never imported by the structure parser, so tables -- explicitly
called out in the Stage 2 brief -- were never extracted at all; every table's cells were
flattened into ordinary paragraph text. It is now wired into `structure_parser`.

Extraction returns table bounding boxes as well as content, because the parser must suppress
the raw text blocks that sit inside a table region; otherwise every table would appear twice,
once as a structured table and once as scrambled paragraph text.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from src.domain import Element, PageRepresentation

TABLE_KEYWORDS = ("quantity", "amount", "balance", "description", "qty", "unit price", "total", "debit", "credit")


@dataclass
class ExtractedTable:
    element: Element
    bbox: tuple[float, float, float, float]


def looks_like_table(page: PageRepresentation) -> bool:
    """Cheap gate so pdfplumber is only opened for pages plausibly containing a table.

    The original gate combined whitespace alignment with an invoice/bank keyword list
    ('amount', 'balance', 'qty', ...). A generic report table -- "Metric | Baseline | Target |
    Unit" -- matched no keyword and, because the extractor concatenated cells without spaces,
    showed no whitespace alignment either. The gate returned False, pdfplumber was never
    invoked, and the table was silently flattened to paragraph text.

    Detection is now structural: a page qualifies when several of its blocks look like grid
    rows, independent of vocabulary.
    """
    lines = [line for line in page.text.splitlines() if line.strip()]
    if any(word in page.text.lower() for word in TABLE_KEYWORDS):
        return True
    if sum(("  " in line or "\t" in line) for line in lines) >= 3:
        return True
    # Structural signal: repeated short, multi-field rows. Works for both whitespace-aligned
    # and newline-separated cell layouts.
    row_like = 0
    for block in page.blocks:
        cells = [part for part in re.split(r"\n|\s{2,}|\t", block.text) if part.strip()]
        if len(cells) >= 3 and all(len(cell) <= 40 for cell in cells):
            row_like += 1
    return row_like >= 3


def _to_markdown(rows: list[list[str]]) -> str:
    """Render as a Markdown table: retrieval and embedding models handle this far better
    than a bare pipe-joined blob, and it keeps the header row semantically distinct."""
    if not rows:
        return ""
    header, *body = rows
    width = len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in body:
        padded = (row + [""] * width)[:width]
        lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(lines)


# Ruled tables are found from their ruling lines; borderless tables need the text strategy,
# which infers columns from word alignment. Trying lines first keeps false positives low.
# The text strategy needs a stricter bar than the line strategy: ruling lines are strong
# evidence of a real table, whereas inferred columns can slice ordinary prose into a grid.
_LINES_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
_TEXT_SETTINGS = {"vertical_strategy": "text", "horizontal_strategy": "text"}
_LINES_MIN_COLUMNS = 2
_TEXT_MIN_COLUMNS = 3


def _valid(rows: list[list[str]], min_columns: int) -> bool:
    """Reject prose that a column-inference strategy mistook for a grid."""
    if len(rows) < 2 or len(rows[0]) < min_columns:
        return False
    cells = [cell for row in rows for cell in row]
    filled = [cell for cell in cells if cell.strip()]
    if len(filled) < len(cells) * 0.6:
        return False
    # Most rows should actually be populated across columns, unlike sliced paragraphs.
    well_formed = sum(1 for row in rows if sum(1 for cell in row if cell.strip()) >= min_columns - 1)
    if well_formed < len(rows) * 0.6:
        return False
    # Real table cells are short; a paragraph sliced into "columns" is not.
    return sum(len(cell) for cell in filled) / max(len(filled), 1) <= 60


_ROW_SPLIT = re.compile(r"\n|\s{2,}|\t")


def _row_cells(text: str) -> list[str]:
    return [part.strip() for part in _ROW_SPLIT.split(text) if part.strip()]


def _union(boxes: list[list[float]], pad: float = 3.0) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes) - pad, min(box[1] for box in boxes) - pad,
        max(box[2] for box in boxes) + pad, max(box[3] for box in boxes) + pad,
    )


def candidate_regions(page: PageRepresentation, min_rows: int = 3, min_cells: int = 3, max_gap: float = 25.0) -> list[tuple[float, float, float, float]]:
    """Locate probable borderless-table regions from block geometry.

    Running pdfplumber's text strategy over a whole page yields one page-sized pseudo-grid
    (prose and all), which is unusable. Cropping to a region first is what makes the strategy
    precise, so candidate regions are derived from runs of vertically adjacent, row-shaped
    blocks -- exactly the shape a borderless table has.
    """
    row_boxes = [
        block.bbox for block in page.blocks
        if len(block.bbox) == 4 and len(_row_cells(block.text)) >= min_cells
        and all(len(cell) <= 40 for cell in _row_cells(block.text))
    ]
    row_boxes.sort(key=lambda box: box[1])
    regions: list[tuple[float, float, float, float]] = []
    run: list[list[float]] = []
    for box in row_boxes:
        if run and box[1] - run[-1][3] > max_gap:
            if len(run) >= min_rows:
                regions.append(_union(run))
            run = []
        run.append(box)
    if len(run) >= min_rows:
        regions.append(_union(run))
    return regions


def _overlaps(left: tuple[float, ...], right: tuple[float, ...], threshold: float = 0.5) -> bool:
    """True when two table regions describe substantially the same area."""
    ax0, ay0, ax1, ay1 = left
    bx0, by0, bx1, by1 = right
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h
    if intersection <= 0:
        return False
    smaller = min(max((ax1 - ax0) * (ay1 - ay0), 1e-6), max((bx1 - bx0) * (by1 - by0), 1e-6))
    return intersection / smaller >= threshold


def _tables_from(finder, page_number: int, min_columns: int, settings: dict) -> list[ExtractedTable]:
    try:
        tables = finder.find_tables(table_settings=settings)
    except Exception:
        return []
    extracted: list[ExtractedTable] = []
    for table in tables:
        try:
            raw = table.extract()
        except Exception:
            continue
        rows = [[(cell or "").strip() for cell in row] for row in raw if row and any(cell for cell in row)]
        if not _valid(rows, min_columns):
            continue
        bbox = tuple(float(value) for value in table.bbox)
        extracted.append(ExtractedTable(
            element=Element(type="table", text=_to_markdown(rows), page=page_number,
                            bbox=[float(v) for v in bbox], rows=rows),
            bbox=bbox,
        ))
    return extracted


def extract_tables_for_pages(pdf_path: str, page_numbers: list[int], regions_by_page: dict[int, list[tuple[float, float, float, float]]] | None = None) -> dict[int, list[ExtractedTable]]:
    """Extract tables for several pages, opening the PDF once.

    The previous signature took a single page and reopened the whole PDF per call, which is
    O(pages) file opens for one document.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return {}
    found: dict[int, list[ExtractedTable]] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number in page_numbers:
                if page_number - 1 >= len(pdf.pages):
                    continue
                page = pdf.pages[page_number - 1]
                results: list[ExtractedTable] = []

                # Ruled tables: the line strategy is reliable across the whole page.
                for candidate in _tables_from(page, page_number, _LINES_MIN_COLUMNS, _LINES_SETTINGS):
                    if not any(_overlaps(candidate.bbox, existing.bbox) for existing in results):
                        results.append(candidate)

                # Borderless tables: only meaningful inside a cropped candidate region. A page
                # may hold both kinds, so this runs regardless of whether the line strategy hit.
                for region in (regions_by_page or {}).get(page_number, []):
                    if any(_overlaps(region, existing.bbox) for existing in results):
                        continue
                    box = (max(0.0, region[0]), max(0.0, region[1]),
                           min(float(page.width), region[2]), min(float(page.height), region[3]))
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    try:
                        cropped = page.crop(box)
                    except Exception:
                        continue
                    for candidate in _tables_from(cropped, page_number, _TEXT_MIN_COLUMNS, _TEXT_SETTINGS):
                        if not any(_overlaps(candidate.bbox, existing.bbox) for existing in results):
                            results.append(candidate)

                if results:
                    found[page_number] = sorted(results, key=lambda item: item.bbox[1])
    except Exception:
        # A malformed document must not abort structuring of the whole packet.
        return found
    return found


def extract_tables(pdf_path: str, page_number: int) -> list[ExtractedTable]:
    """Single-page convenience wrapper retained for the existing call signature."""
    return extract_tables_for_pages(pdf_path, [page_number]).get(page_number, [])


def inside(bbox: list[float], table_bbox: tuple[float, float, float, float], tolerance: float = 2.0) -> bool:
    """True when a text block's centre falls within a table region."""
    if len(bbox) != 4:
        return False
    centre_x = (bbox[0] + bbox[2]) / 2
    centre_y = (bbox[1] + bbox[3]) / 2
    left, top, right, bottom = table_bbox
    return (left - tolerance) <= centre_x <= (right + tolerance) and (top - tolerance) <= centre_y <= (bottom + tolerance)
