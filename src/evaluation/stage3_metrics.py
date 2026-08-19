"""Information retrieval metrics based on supplied relevance judgements."""
from __future__ import annotations

import math


def retrieval_metrics(rankings: list[list[str]], relevant: list[set[str]], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    if len(rankings) != len(relevant): raise ValueError("Rankings and relevance labels must align.")
    output: dict[str, float] = {}
    for k in ks:
        output[f"recall@{k}"] = sum(bool(set(ranking[:k]) & truth) for ranking, truth in zip(rankings, relevant)) / max(len(rankings), 1)
        output[f"precision@{k}"] = sum(len(set(ranking[:k]) & truth) / k for ranking, truth in zip(rankings, relevant)) / max(len(rankings), 1)
    reciprocal = [next((1 / (rank + 1) for rank, item in enumerate(ranking) if item in truth), 0.0) for ranking, truth in zip(rankings, relevant)]
    output["mrr"] = sum(reciprocal) / max(len(reciprocal), 1)
    ndcgs = []
    for ranking, truth in zip(rankings, relevant):
        dcg = sum(1 / math.log2(rank + 2) for rank, item in enumerate(ranking) if item in truth)
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(truth), len(ranking))))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    output["ndcg"] = sum(ndcgs) / max(len(ndcgs), 1)
    return output
