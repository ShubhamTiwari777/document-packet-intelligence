"""Dependency-free BM25 index for exact/entity-heavy evidence retrieval."""
from __future__ import annotations

from collections import Counter
import math
from src.domain import Chunk
from src.features.text_features import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.documents = [tokenize(chunk.text) for chunk in chunks]
        self.lengths = [len(doc) for doc in self.documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.document_frequency: Counter[str] = Counter(term for document in self.documents for term in set(document))

    def score(self, query: str) -> list[float]:
        scores = [0.0] * len(self.chunks); count = len(self.chunks)
        for term in set(tokenize(query)):
            frequency = self.document_frequency.get(term, 0)
            if not frequency: continue
            idf = math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for index, document in enumerate(self.documents):
                term_frequency = document.count(term)
                denominator = term_frequency + self.k1 * (1 - self.b + self.b * self.lengths[index] / max(self.average_length, 1))
                scores[index] += idf * term_frequency * (self.k1 + 1) / denominator if denominator else 0.0
        return scores

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        scores = self.score(query)
        indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:top_k]
        return [(self.chunks[index], scores[index]) for index in indices]
