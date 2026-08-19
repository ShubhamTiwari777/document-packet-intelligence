"""Build synthetic multi-document packets from RVL-CDIP pages with real layout.

Motivation: boundary detection was trained on OpenPSS, which supplies OCR text but no geometry,
leaving 8 of 14 pairwise features constant. Ablations showed the resulting ceiling is
representational -- reconstructing pseudo-layout from text did not help, because inferred
positions are not real ones. RVL-CDIP ships word-level bounding boxes, so packets assembled from
it carry genuine block geometry and font-proxy signals.

Packets are assembled by concatenating whole documents of differing types, which is how a real
packet arises; a boundary is exactly a document change. Using RVL-CDIP directly (a general
document corpus) rather than a benchmark's prepared splits keeps this our own training data.
"""
from __future__ import annotations

from typing import Any
import random

from src.domain import PageRepresentation, TextBlock

# RVL-CDIP boxes are normalised to a 1000x1000 canvas by convention in most redistributions;
# they are rescaled to the page's own pixel size so geometry-derived features are comparable.
CANVAS = 1000.0


def _rescale(box: list[int], width: float, height: float) -> list[float]:
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    if max(x0, y0, x1, y1) <= CANVAS + 1:
        return [x0 / CANVAS * width, y0 / CANVAS * height, x1 / CANVAS * width, y1 / CANVAS * height]
    return [x0, y0, x1, y1]


def blocks_from_words(words: list[str], boxes: list[list[int]], width: float, height: float,
                      line_tolerance: float = 0.012, block_gap: float = 0.025) -> list[TextBlock]:
    """Group word boxes into lines, then lines into blocks.

    Words sharing a vertical band form a line; lines separated by more than `block_gap` of page
    height start a new block. Word height doubles as a font-size proxy, which is what makes
    `font_mean_delta` and `dominant_font_size_delta` meaningful for this corpus.
    """
    if not words or len(words) != len(boxes):
        return []
    scaled = [(_rescale(box, width, height), word) for box, word in zip(boxes, words)]
    scaled.sort(key=lambda item: (item[0][1], item[0][0]))

    lines: list[list[tuple[list[float], str]]] = []
    tolerance = height * line_tolerance
    for box, word in scaled:
        if lines and abs(box[1] - lines[-1][0][0][1]) <= tolerance:
            lines[-1].append((box, word))
        else:
            lines.append([(box, word)])
    for line in lines:
        line.sort(key=lambda item: item[0][0])

    blocks: list[TextBlock] = []
    current: list[list[tuple[list[float], str]]] = []
    gap = height * block_gap
    for line in lines:
        if current and line[0][0][1] - current[-1][0][0][3] > gap:
            blocks.append(_to_block(current))
            current = []
        current.append(line)
    if current:
        blocks.append(_to_block(current))
    return [block for block in blocks if block.text.strip()]


def _to_block(lines: list[list[tuple[list[float], str]]]) -> TextBlock:
    text = "\n".join(" ".join(word for _, word in line) for line in lines)
    boxes = [box for line in lines for box, _ in line]
    sizes = [max(1.0, round(box[3] - box[1], 1)) for box in boxes]
    return TextBlock(
        text=text,
        bbox=[min(b[0] for b in boxes), min(b[1] for b in boxes),
              max(b[2] for b in boxes), max(b[3] for b in boxes)],
        font_sizes=sizes, font_names=["ocr"] * len(sizes), flags=[0] * len(sizes),
    )


def page_from_record(record: dict[str, Any], page_number: int) -> PageRepresentation:
    width, height = float(record.get("width", 754)), float(record.get("height", 1000))
    blocks = blocks_from_words(record["words"], record["boxes"], width, height)
    return PageRepresentation(
        page_number=page_number, text="\n".join(block.text for block in blocks),
        blocks=blocks, fonts=[{"size": size} for block in blocks for size in block.font_sizes],
        width=width, height=height, image_path=None, extraction_method="rvlcdip_ocr",
    )


def build_packets(pages: list[dict[str, Any]], packet_count: int = 400,
                  min_documents: int = 2, max_documents: int = 5,
                  min_pages: int = 1, max_pages: int = 4, seed: int = 42) -> list[dict[str, Any]]:
    """Assemble packets of consecutive documents; a boundary is a document change.

    Pages are grouped by type and consumed without replacement so a document's pages are
    contiguous and no page appears in two packets.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in pages:
        by_label.setdefault(record["label"], []).append(record)
    for bucket in by_label.values():
        rng.shuffle(bucket)

    packets: list[dict[str, Any]] = []
    for index in range(packet_count):
        available = [label for label, bucket in by_label.items() if len(bucket) >= min_pages]
        if len(available) < min_documents:
            break
        document_count = rng.randint(min_documents, max_documents)
        labels = rng.sample(available, min(document_count, len(available)))
        packet_pages: list[PageRepresentation] = []
        starts_document: list[bool] = []  # one flag per page
        documents: list[dict[str, Any]] = []
        for label in labels:
            bucket = by_label[label]
            take = min(rng.randint(min_pages, max_pages), len(bucket))
            if take <= 0:
                continue
            start = len(packet_pages) + 1
            for offset in range(take):
                packet_pages.append(page_from_record(bucket.pop(), len(packet_pages) + 1))
                starts_document.append(offset == 0)
            documents.append({"label": label, "pages": list(range(start, len(packet_pages) + 1))})
        if len(documents) < min_documents or len(packet_pages) < 2:
            continue
        # One label per adjacent page pair: pair (i, i+1) is a boundary when page i+1 opens a
        # document. Derived from per-page flags so the count is N-1 by construction.
        boundary_labels = [int(flag) for flag in starts_document[1:]]
        packets.append({
            "packet_id": f"rvl_packet_{index:05d}", "pages": packet_pages,
            "boundary_labels": boundary_labels, "documents": documents,
        })
    return packets
