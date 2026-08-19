"""Maximal Marginal Relevance re-ordering.

Pure relevance ranking has a failure mode that matters specifically for retrieval-augmented
generation: the top-k can be five near-paraphrases of one passage. Recall@k looks fine, the
generator sees one fact repeated five times, and any question needing two facts fails.

MMR (Carbonell & Goldstein, 1998) selects iteratively, trading relevance against dissimilarity to
what is already selected:

    score(c) = lambda * relevance(c) - (1 - lambda) * max_similarity(c, selected)

`lambda_relevance = 1.0` reduces exactly to the input ranking, so this is a no-op when disabled --
which is how it is benchmarked against the unmodified pipeline.
"""
from __future__ import annotations

from src.domain import Chunk
from src.features.text_features import tokenize


def _similarity(left: str, right: str) -> float:
    a, b = set(tokenize(left)), set(tokenize(right))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mmr_reorder(
    candidates: list[tuple[Chunk, float]],
    top_k: int,
    lambda_relevance: float = 0.7,
) -> list[tuple[Chunk, float]]:
    """Re-order candidates by MMR, preserving each candidate's original score."""
    if not candidates or lambda_relevance >= 1.0:
        return candidates[:top_k]

    scores = [score for _, score in candidates]
    lowest, highest = min(scores), max(scores)
    span = (highest - lowest) or 1.0
    remaining = list(candidates)
    selected: list[tuple[Chunk, float]] = []

    while remaining and len(selected) < top_k:
        best_index, best_value = 0, float("-inf")
        for index, (chunk, score) in enumerate(remaining):
            relevance = (score - lowest) / span
            redundancy = max((_similarity(chunk.text, chosen.text) for chosen, _ in selected), default=0.0)
            value = lambda_relevance * relevance - (1 - lambda_relevance) * redundancy
            if value > best_value:
                best_index, best_value = index, value
        selected.append(remaining.pop(best_index))
    return selected
