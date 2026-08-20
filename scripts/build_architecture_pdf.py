"""Render the system architecture diagram to PDF as vector graphics.

The Markdown source in docs/architecture.md carries Mermaid diagrams, which need a JavaScript
renderer; converting that file to PDF directly would print the diagram *source* rather than the
diagram. Drawing the flow with PyMuPDF primitives instead keeps the deliverable dependency-free
(no browser, no Node toolchain), produces true vector output that stays sharp at any zoom, and
lets the layout be designed for a printed page rather than a scrolling one.

Page 1: the processing pipeline. Page 2: model inventory and stage contracts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf

INK = (0.06, 0.13, 0.23)
MUTED = (0.35, 0.40, 0.47)
ACCENT = (0.17, 0.37, 0.66)
RULE = (0.78, 0.82, 0.88)
FILL = (0.965, 0.972, 0.98)
STAGE_FILL = (0.99, 0.99, 1.0)
WHITE = (1, 1, 1)

COLUMNS = [
    ("Ingestion", [
        ("Packet PDF", "N pages, M documents"),
        ("PDFParser · PyMuPDF", "gap-aware span join;\nblocks, bboxes, fonts"),
        ("OCR fallback", "Tesseract, off by default"),
        ("Page renderer", "150 dpi"),
    ]),
    ("Stage 1 · Packet Intelligence", [
        ("21 pairwise features", "text · visual · layout · heuristic"),
        ("Calibrated GBM", "HistGradientBoosting + isotonic\n0.5 MB"),
        ("Page-number-reset override", "domain-invariant cue"),
        ("Grouping", "expected-count from\ncalibrated probabilities"),
        ("Hybrid classifier", "TF-IDF+LR, 16 classes\n+ lexicon extension"),
        ("Abstention", "below 0.35 -> unknown"),
    ]),
    ("Stage 2 · Document Structuring", [
        ("Boilerplate detection", "margin + repetition"),
        ("Heading hierarchy", "font rank · numbering depth"),
        ("Element classification", "list · caption · paragraph"),
        ("Table extraction", "ruled + borderless (cropped)"),
        ("Section tree", "parent · breadcrumb · page refs"),
        ("Structure-aware chunker", "tables atomic;\nbreadcrumb prefix"),
    ]),
    ("Stage 3 · Evidence Retrieval", [
        ("BM25", "lexical"),
        ("Dense index", "TF-IDF+SVD default,\nbge-small optional"),
        ("Reciprocal rank fusion", "k = 60"),
        ("Feature reranker", "coverage · phrase · exactness"),
        ("MMR diversity", "lambda = 0.7"),
        ("Context assembly", "dedup · token budget · citations"),
    ]),
]

CONTRACTS = [
    ("Ingestion -> Stage 1", "list[PageRepresentation] - text, blocks, bboxes, fonts, render path"),
    ("Stage 1 -> Stage 2", "list[DocumentGroup] - pages, type, confidence, source, alternatives"),
    ("Stage 2 -> Stage 3", "list[Chunk] - text, page_refs, breadcrumb, element_types, token_count"),
    ("Stage 3 -> caller", "list[EvidenceResult] - evidence, doc_id, page_ref, confidence, scores"),
]

MODELS = [
    ("Boundary classifier", "HistGradientBoosting + isotonic calibration", "OpenPSS SHORT, 15,906 pairs", "0.5 MB"),
    ("Boundary text signal", "TF-IDF vectoriser", "same", "2.0 MB"),
    ("Document classifier", "TF-IDF + Logistic Regression", "RVL-CDIP OCR, 8,887 docs / 16 classes", "3.3 MB"),
    ("Extension classes", "Weighted lexicon (no training data exists)", "-", "< 1 KB"),
    ("Dense retrieval", "TF-IDF + TruncatedSVD", "fitted per corpus at index time", "112 MB peak RAM"),
    ("Dense retrieval (optional)", "BAAI/bge-small-en-v1.5", "pretrained", "130 MB"),
]


# Base-14 PDF fonts are Latin-1 only; characters outside it render as "?" rather than raising,
# so text is folded to safe equivalents before it is drawn.
_SUBSTITUTIONS = {"—": "-", "–": "-", "‘": "'", "’": "'",
                  "“": '"', "”": '"', "→": "->", "≥": ">=", "…": "..."}


def safe(text: str) -> str:
    for source, target in _SUBSTITUTIONS.items():
        text = text.replace(source, target)
    return text.encode("latin-1", "replace").decode("latin-1")


def _arrow(shape, x0: float, y: float, x1: float) -> None:
    shape.draw_line(pymupdf.Point(x0, y), pymupdf.Point(x1 - 5, y))
    shape.finish(color=ACCENT, width=1.1)
    shape.draw_polyline([pymupdf.Point(x1 - 5, y - 3.2), pymupdf.Point(x1, y),
                         pymupdf.Point(x1 - 5, y + 3.2)])
    shape.finish(color=ACCENT, fill=ACCENT, width=0.6)


def draw_pipeline(page: pymupdf.Page) -> None:
    width, height = page.rect.width, page.rect.height
    margin = 34.0
    # insert_text places on a baseline directly; insert_textbox silently returns a negative value
    # and draws nothing when the rect cannot fit the line, which is easy to miss.
    page.insert_text(pymupdf.Point(margin, 38),
                     safe("Document Packet Intelligence & Evidence Retrieval"),
                     fontsize=15, fontname="hebo", color=INK)
    page.insert_textbox(pymupdf.Rect(margin, 46, width - margin, 68),
                        safe("System architecture - three stages with dataclass contracts between them. "
                             "CPU-only; 5.8 MB of committed models."),
                        fontsize=8, fontname="helv", color=MUTED)

    top = 74.0
    bottom = height - 74.0
    gap = 16.0
    column_width = (width - 2 * margin - gap * (len(COLUMNS) - 1)) / len(COLUMNS)

    for index, (title, boxes) in enumerate(COLUMNS):
        left = margin + index * (column_width + gap)
        panel = pymupdf.Rect(left, top, left + column_width, bottom)
        page.draw_rect(panel, color=RULE, fill=STAGE_FILL, width=0.7, radius=0.02)
        page.insert_textbox(pymupdf.Rect(left + 6, top + 6, left + column_width - 6, top + 26),
                            safe(title), fontsize=8.4, fontname="hebo", color=ACCENT, align=1)

        inner_top = top + 26
        available = bottom - inner_top - 8
        # Cap the slot so a column with few boxes does not stretch them into tall empty panels;
        # the stack is then centred in the column.
        slot = min(available / len(boxes), 62.0)
        inner_top += (available - slot * len(boxes)) / 2
        for position, (heading, detail) in enumerate(boxes):
            box_top = inner_top + position * slot + 3
            box = pymupdf.Rect(left + 7, box_top, left + column_width - 7, box_top + slot - 7)
            page.draw_rect(box, color=RULE, fill=WHITE, width=0.6)
            page.insert_textbox(pymupdf.Rect(box.x0 + 5, box.y0 + 4, box.x1 - 5, box.y0 + 18),
                                safe(heading), fontsize=7.2, fontname="hebo", color=INK)
            page.insert_textbox(pymupdf.Rect(box.x0 + 5, box.y0 + 14, box.x1 - 5, box.y1 - 2),
                                safe(detail), fontsize=6.1, fontname="helv", color=MUTED)
            if position < len(boxes) - 1:
                shape = page.new_shape()
                shape.draw_line(pymupdf.Point(box.x0 + column_width / 2 - 7, box.y1),
                                pymupdf.Point(box.x0 + column_width / 2 - 7, box.y1 + 3.5))
                shape.finish(color=RULE, width=0.8)
                shape.commit()

        if index < len(COLUMNS) - 1:
            shape = page.new_shape()
            _arrow(shape, panel.x1 + 2, (top + bottom) / 2, panel.x1 + gap - 2)
            shape.commit()

    footer = pymupdf.Rect(margin, bottom + 8, width - margin, height - 34)
    page.draw_rect(footer, color=RULE, fill=FILL, width=0.7)
    page.insert_textbox(pymupdf.Rect(footer.x0 + 8, footer.y0 + 5, footer.x1 - 8, footer.y1 - 4),
                        safe("Output:  evidence + doc_id + page_ref + breadcrumb + confidence      |      "
                        "FastAPI:  /process  ·  /retrieve  ·  /context      |      "
                        "No GPU, no external API, no network call at inference"),
                        fontsize=7.2, fontname="helv", color=INK, align=1)


def _table(page: pymupdf.Page, x: float, y: float, width: float, headers: list[str],
           rows: list[tuple], widths: list[float], row_height: float = 15.0) -> float:
    columns = [x]
    for fraction in widths:
        columns.append(columns[-1] + width * fraction)
    header_rect = pymupdf.Rect(x, y, x + width, y + row_height)
    page.draw_rect(header_rect, color=RULE, fill=(0.93, 0.95, 0.97), width=0.6)
    for index, label in enumerate(headers):
        page.insert_text(pymupdf.Point(columns[index] + 4, y + row_height - 4.8),
                         safe(label), fontsize=6.9, fontname="hebo", color=INK)
    cursor = y + row_height
    for row in rows:
        rect = pymupdf.Rect(x, cursor, x + width, cursor + row_height)
        page.draw_rect(rect, color=RULE, fill=WHITE, width=0.4)
        for index, cell in enumerate(row):
            page.insert_text(pymupdf.Point(columns[index] + 4, cursor + row_height - 4.8),
                             safe(str(cell)), fontsize=6.6, fontname="helv", color=INK)
        cursor += row_height
    return cursor


def draw_reference(page: pymupdf.Page) -> None:
    width = page.rect.width
    margin = 34.0
    inner = width - 2 * margin
    page.insert_text(pymupdf.Point(margin, 40), safe("Model inventory"),
                     fontsize=12, fontname="hebo", color=INK)
    cursor = _table(page, margin, 50, inner, ["Component", "Model", "Trained on", "Size"],
                    MODELS, [0.22, 0.32, 0.32, 0.14])
    page.insert_textbox(pymupdf.Rect(margin, cursor + 6, width - margin, cursor + 30),
                        safe("Total committed model footprint: 5.8 MB. No component is trained on the DocSplit "
                             "benchmark; the optional transformer is a general-purpose pretrained encoder."),
                        fontsize=7, fontname="helv", color=MUTED)

    cursor += 34
    page.insert_text(pymupdf.Point(margin, cursor + 12), safe("Stage contracts"),
                     fontsize=12, fontname="hebo", color=INK)
    cursor = _table(page, margin, cursor + 24, inner, ["Boundary", "Contract"],
                    CONTRACTS, [0.26, 0.74])
    page.insert_textbox(pymupdf.Rect(margin, cursor + 6, width - margin, cursor + 40),
                        safe("Each stage consumes only the contract above it, so a stage can be replaced without "
                             "touching its neighbours. The Stage 3 dense encoder was swapped three times during "
                             "benchmarking - hashed bag-of-words, TF-IDF+SVD, then a transformer - with no change "
                             "to Stage 1 or Stage 2 code."),
                        fontsize=7, fontname="helv", color=MUTED)

    cursor += 46
    page.insert_text(pymupdf.Point(margin, cursor + 12), safe("Artifacts written per packet"),
                     fontsize=12, fontname="hebo", color=INK)
    artifacts = [
        ("pages.json", "raw page representations"),
        ("boundary_features.json", "21 features per adjacent page pair"),
        ("stage1.json", "document groups, types, confidences"),
        ("structured_documents.json", "section tree with elements and page refs"),
        ("markdown/*.md", "human-readable rendering per document"),
        ("chunks.json", "retrieval units with breadcrumb and token count"),
        ("index/", "dense vectors and fitted encoder"),
    ]
    _table(page, margin, cursor + 24, inner, ["File", "Contents"], artifacts, [0.30, 0.70])


def build(output_path: str | Path) -> int:
    document = pymupdf.open()
    landscape = pymupdf.paper_rect("a4-l")
    draw_pipeline(document.new_page(width=landscape.width, height=landscape.height))
    draw_reference(document.new_page(width=landscape.width, height=landscape.height))
    document.save(str(output_path), deflate=True)
    pages = len(document)
    document.close()
    return pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/architecture.pdf")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    count = build(args.output)
    print(f"Wrote {args.output}: {count} pages, {Path(args.output).stat().st_size/1024:.0f} KB")
