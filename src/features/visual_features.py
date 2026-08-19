"""Visual feature interface; CLIP remains an optional experiment, never implicit."""
from __future__ import annotations

from pathlib import Path
from src.features.text_features import cosine


def image_histogram_embedding(image_path: str | None, bins: int = 16) -> list[float]:
    """Use a tiny grayscale histogram fallback; returns zeros without a render."""
    if not image_path or not Path(image_path).exists():
        return [0.0] * bins
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return [0.0] * bins
    with Image.open(image_path).convert("L") as image:
        histogram = image.histogram()
    grouped = [sum(histogram[i * 256 // bins:(i + 1) * 256 // bins]) for i in range(bins)]
    total = max(sum(grouped), 1)
    return [item / total for item in grouped]


def visual_delta(left_path: str | None, right_path: str | None) -> float:
    return 1.0 - cosine(image_histogram_embedding(left_path), image_histogram_embedding(right_path))
