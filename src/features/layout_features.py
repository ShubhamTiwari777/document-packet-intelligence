"""Layout signals derived solely from extracted page geometry."""
from __future__ import annotations

from collections import Counter
from src.domain import PageRepresentation


def page_layout_features(page: PageRepresentation) -> dict[str, float]:
    occupied = sum(max(0.0, b.bbox[2] - b.bbox[0]) * max(0.0, b.bbox[3] - b.bbox[1]) for b in page.blocks if len(b.bbox) == 4)
    sizes = [font.get("size", 0.0) for font in page.fonts]
    mean = sum(sizes) / len(sizes) if sizes else 0.0
    variance = sum((size - mean) ** 2 for size in sizes) / len(sizes) if sizes else 0.0
    return {
        "block_count": float(len(page.blocks)), "whitespace_ratio": max(0.0, 1 - occupied / max(page.width * page.height, 1.0)),
        "font_mean": mean, "font_std": variance ** 0.5,
        "dominant_font_size": float(Counter(sizes).most_common(1)[0][0]) if sizes else 0.0,
        "image_area_ratio": min(1.0, page.image_count * 0.1),
    }
