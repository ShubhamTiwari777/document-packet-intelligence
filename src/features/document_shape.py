"""Geometry-only descriptors of what a whole document looks like, independent of its words.

Distinct from `layout_features.py`, which produces six per-page signals consumed by the Stage 1
pairwise boundary features. These are document-level and feed the type classifier: an invoice, a
memo and a scientific paper differ in shape long before they differ in vocabulary -- column count,
where the largest text sits, how regular the left margins are, how much of the page carries ink.
A TF-IDF classifier sees none of that, and on OCR text the words themselves are unreliable.

Everything is computed from `PageRepresentation.blocks`, so the same code runs on training pages
rebuilt from word boxes and on production pages parsed by PyMuPDF. That symmetry is deliberate:
Stage 1 lost a third of its feature set to a train/serve mismatch where line-based features saw
whole-page text during training and real lines at inference.
"""
from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Sequence

SHAPE_FEATURE_NAMES: list[str] = [
    "block_count", "text_coverage", "mean_block_area", "max_block_area",
    "mean_block_width", "block_width_spread",
    "left_margin_mean", "left_margin_spread", "column_estimate",
    "density_top", "density_middle", "density_bottom",
    "font_mean", "font_spread", "font_max_ratio", "largest_font_position",
    "mean_words_per_block", "short_block_ratio",
    "gap_mean", "gap_spread", "aspect_ratio",
]

_ZERO = {name: 0.0 for name in SHAPE_FEATURE_NAMES}


def _safe(values: Sequence[float], fn) -> float:
    try:
        return float(fn(values)) if values else 0.0
    except Exception:
        return 0.0


def _column_estimate(left_edges: list[float], bins: int = 10) -> float:
    """How many distinct left margins carry real mass.

    A single-column letter puts nearly every block at one left edge; a two-column paper splits them
    into two clusters; a form scatters them. Counting populated bins is crude but stable under OCR
    jitter, which a clustering algorithm would not be.
    """
    if not left_edges:
        return 0.0
    counts = [0] * bins
    for edge in left_edges:
        counts[min(bins - 1, max(0, int(edge * bins)))] += 1
    floor = max(1.0, 0.12 * len(left_edges))
    return float(sum(1 for count in counts if count >= floor))


def page_shape_features(page: Any) -> dict[str, float]:
    """Shape descriptors for a single page."""
    blocks = [b for b in getattr(page, "blocks", []) or [] if len(getattr(b, "bbox", []) or []) == 4]
    width = float(getattr(page, "width", 0) or 0)
    height = float(getattr(page, "height", 0) or 0)
    if not blocks or width <= 0 or height <= 0:
        return dict(_ZERO)

    areas, widths, lefts, tops, gaps, word_counts = [], [], [], [], [], []
    fonts: list[float] = []
    largest_font, largest_font_y = 0.0, 0.0
    previous_bottom = None

    for block in sorted(blocks, key=lambda b: b.bbox[1]):
        x0, y0, x1, y1 = (float(v) for v in block.bbox)
        block_width = max(0.0, x1 - x0) / width
        block_height = max(0.0, y1 - y0) / height
        areas.append(block_width * block_height)
        widths.append(block_width)
        lefts.append(min(1.0, max(0.0, x0 / width)))
        tops.append(min(1.0, max(0.0, y0 / height)))
        word_counts.append(len((getattr(block, "text", "") or "").split()))
        sizes = [float(s) for s in (getattr(block, "font_sizes", []) or []) if s]
        if sizes:
            fonts.extend(sizes)
            if max(sizes) > largest_font:
                largest_font, largest_font_y = max(sizes), tops[-1]
        if previous_bottom is not None:
            gaps.append(max(0.0, (y0 - previous_bottom) / height))
        previous_bottom = y1

    thirds = [0.0, 0.0, 0.0]
    for top, area in zip(tops, areas):
        thirds[min(2, int(top * 3))] += area

    font_mean = _safe(fonts, mean)
    return {
        "block_count": float(len(blocks)),
        "text_coverage": float(min(1.5, sum(areas))),
        "mean_block_area": _safe(areas, mean),
        "max_block_area": float(max(areas)) if areas else 0.0,
        "mean_block_width": _safe(widths, mean),
        "block_width_spread": _safe(widths, pstdev),
        "left_margin_mean": _safe(lefts, mean),
        "left_margin_spread": _safe(lefts, pstdev),
        "column_estimate": _column_estimate(lefts),
        "density_top": thirds[0],
        "density_middle": thirds[1],
        "density_bottom": thirds[2],
        "font_mean": font_mean,
        "font_spread": _safe(fonts, pstdev),
        "font_max_ratio": (largest_font / font_mean) if font_mean else 0.0,
        "largest_font_position": largest_font_y,
        "mean_words_per_block": _safe(word_counts, mean),
        "short_block_ratio": (sum(1 for w in widths if w < 0.30) / len(widths)) if widths else 0.0,
        "gap_mean": _safe(gaps, mean),
        "gap_spread": _safe(gaps, pstdev),
        "aspect_ratio": width / height if height else 0.0,
    }


def document_shape_features(pages: Sequence[Any]) -> dict[str, float]:
    """Average the per-page descriptors over the pages of one document.

    Averaging rather than concatenating keeps the vector a fixed size regardless of page count, so
    a two-page invoice and a nine-page report are described in the same space.
    """
    per_page = [page_shape_features(page) for page in pages]
    per_page = [f for f in per_page if any(abs(v) > 0 for v in f.values())]
    if not per_page:
        return dict(_ZERO)
    return {name: sum(f[name] for f in per_page) / len(per_page) for name in SHAPE_FEATURE_NAMES}


def shape_vector(features: dict[str, float]) -> list[float]:
    """Fixed-order vector, so training and inference cannot silently disagree on column order."""
    return [float(features.get(name, 0.0)) for name in SHAPE_FEATURE_NAMES]
