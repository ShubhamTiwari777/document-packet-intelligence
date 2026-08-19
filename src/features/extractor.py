"""Feature matrix builder and cache format for boundary-model training."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from typing import Any

from src.domain import PageRepresentation
from src.features.heuristic_features import pair_heuristics
from src.features.layout_features import page_layout_features
from src.features.text_features import cosine, hashed_embedding, text_statistics
from src.features.visual_features import visual_delta

FEATURE_NAMES = [
    "text_delta", "visual_delta", "page_number_reset", "page_number_continuation",
    "font_mean_delta", "font_std_delta", "whitespace_ratio_delta", "header_similarity",
    "footer_similarity", "text_length_ratio", "block_count_delta", "image_area_ratio_delta",
    "text_density_delta", "dominant_font_size_delta",
]


def pair_features(left: PageRepresentation, right: PageRepresentation, text_embedder: Any = None) -> dict[str, float]:
    left_layout, right_layout = page_layout_features(left), page_layout_features(right)
    left_text = text_statistics(left.text, left.width * left.height)
    right_text = text_statistics(right.text, right.width * right.height)
    text_delta = text_embedder.delta(left.text, right.text) if text_embedder is not None else 1 - cosine(hashed_embedding(left.text), hashed_embedding(right.text))
    values = {
        "text_delta": text_delta,
        "visual_delta": visual_delta(left.image_path, right.image_path),
        "font_mean_delta": abs(left_layout["font_mean"] - right_layout["font_mean"]),
        "font_std_delta": abs(left_layout["font_std"] - right_layout["font_std"]),
        "whitespace_ratio_delta": abs(left_layout["whitespace_ratio"] - right_layout["whitespace_ratio"]),
        "text_length_ratio": min(len(left.text), len(right.text)) / max(len(left.text), len(right.text), 1),
        "block_count_delta": abs(left_layout["block_count"] - right_layout["block_count"]),
        "image_area_ratio_delta": abs(left_layout["image_area_ratio"] - right_layout["image_area_ratio"]),
        "text_density_delta": abs(left_text["text_density"] - right_text["text_density"]),
        "dominant_font_size_delta": abs(left_layout["dominant_font_size"] - right_layout["dominant_font_size"]),
    }
    values.update(pair_heuristics(left, right))
    return {name: float(values[name]) for name in FEATURE_NAMES}


def packet_feature_rows(pages: list[PageRepresentation], packet_id: str, text_embedder: Any = None) -> list[dict[str, object]]:
    return [{"packet_id": packet_id, "left_page": left.page_number, "right_page": right.page_number, **pair_features(left, right, text_embedder)} for left, right in zip(pages, pages[1:])]


def save_feature_cache(rows: list[dict[str, object]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"feature_names": FEATURE_NAMES, "rows": rows}, indent=2), encoding="utf-8")


def load_feature_cache(path: str | Path) -> list[dict[str, object]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["rows"]
