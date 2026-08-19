"""Text features with a deterministic CPU fallback when model downloads are disabled."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable
import math
import pickle
import re


TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def hashed_embedding(text: str, dimensions: int = 384) -> list[float]:
    """A reproducible hashed bag-of-words fallback, not a pretrained embedding."""
    values = [0.0] * dimensions
    for token, count in Counter(tokenize(text)).items():
        slot = int.from_bytes(token.encode("utf-8"), "little", signed=False) % dimensions
        values[slot] += 1 + math.log(count)
    return normalize(values)


def normalize(values: Iterable[float]) -> list[float]:
    result = list(values)
    norm = math.sqrt(sum(value * value for value in result))
    return [value / norm for value in result] if norm else result


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    a, b = list(left), list(right)
    if len(a) != len(b):
        raise ValueError("Vectors need equal dimensions.")
    denominator = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
    return sum(x*y for x, y in zip(a, b)) / denominator if denominator else 0.0


def text_statistics(text: str, area: float) -> dict[str, float]:
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "character_count": float(len(text)), "word_count": float(len(tokenize(text))),
        "line_count": float(len(lines)), "text_density": len(text) / max(area, 1.0),
    }


class TfidfTextEmbedder:
    """Corpus-fitted TF-IDF cosine distance; a much sharper text_delta signal than the
    hashed bag-of-words fallback, without requiring a pretrained transformer (none of
    which have prebuilt wheels for this project's Python yet). Word 1-2 grams with IDF
    weighting suppress boilerplate (letterheads, page numbers) and emphasize the
    vocabulary shifts that actually mark a new document.
    """

    def __init__(self, vectorizer: Any):
        self.vectorizer = vectorizer

    @classmethod
    def fit(cls, texts: list[str], min_df: int = 2, max_features: int = 50_000) -> "TfidfTextEmbedder":
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=max_features, sublinear_tf=True)
        vectorizer.fit(texts)
        return cls(vectorizer)

    def delta(self, left_text: str, right_text: str) -> float:
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        vectors = self.vectorizer.transform([left_text, right_text])
        similarity = float(cosine_similarity(vectors[0], vectors[1])[0][0])
        return 1.0 - similarity

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as handle:
            pickle.dump(self.vectorizer, handle)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfTextEmbedder":
        with Path(path).open("rb") as handle:
            return cls(pickle.load(handle))
