"""Reciprocal-rank fusion with independent ranker scores retained for auditability."""
from __future__ import annotations

from collections import defaultdict
from src.domain import Chunk


def reciprocal_rank_fusion(rankings: list[list[tuple[Chunk, float]]], k: int = 60, weights: list[float] | None = None) -> list[tuple[Chunk, float, dict[str, float]]]:
    weights = weights or [1.0] * len(rankings)
    totals: dict[str, float] = defaultdict(float); chunks: dict[str, Chunk] = {}; sources: dict[str, dict[str, float]] = defaultdict(dict)
    for source_index, (ranking, weight) in enumerate(zip(rankings, weights)):
        for rank, (chunk, score) in enumerate(ranking, 1):
            chunks[chunk.chunk_id] = chunk; totals[chunk.chunk_id] += weight / (k + rank); sources[chunk.chunk_id][str(source_index)] = score
    return sorted(((chunks[key], score, sources[key]) for key, score in totals.items()), key=lambda item: item[1], reverse=True)
