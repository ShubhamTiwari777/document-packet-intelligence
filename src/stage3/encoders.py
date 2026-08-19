"""Pluggable text encoders for the dense half of the hybrid retriever.

The dense index previously used `hashed_embedding` -- a hashed bag-of-words whose slots are
`int.from_bytes(token) % dimensions`. Two tokens with related meaning land in unrelated slots,
so that "dense" index scored purely on lexical overlap and was largely redundant with BM25:
the hybrid was effectively lexical + lexical, and reciprocal-rank fusion had little independent
evidence to combine.

Three encoders are provided so the choice can be measured rather than asserted:

* `HashedEncoder`      -- the previous behaviour, kept as the baseline to beat.
* `SvdEncoder`         -- TF-IDF followed by truncated SVD (LSA). Captures co-occurrence
                          semantics, trains in seconds on CPU, and adds no heavy dependency.
* `TransformerEncoder` -- `BAAI/bge-small-en-v1.5` via sentence-transformers: genuine semantic
                          embeddings, at the cost of a large dependency and slower encoding.

All three expose `fit`/`encode`, so `DenseIndex` is agnostic to which is in use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
import math
import pickle

from src.features.text_features import hashed_embedding, normalize


class Encoder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> "Encoder": ...
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class HashedEncoder:
    """Baseline: hashed bag-of-words. No training, no semantics."""

    name = "hashed_bow"

    def fit(self, texts: list[str]) -> "HashedEncoder":
        return self

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [hashed_embedding(text) for text in texts]

    def save(self, directory: str | Path) -> None:
        return None


class SvdEncoder:
    """TF-IDF + truncated SVD (latent semantic analysis).

    Chosen as the lightweight semantic option: SVD projects the sparse term space onto latent
    factors, so documents sharing meaning-bearing co-occurring terms score close even when the
    exact query words are absent -- the property the hashed encoder lacks entirely.
    """

    name = "tfidf_svd"

    def __init__(self, components: int = 192, vectorizer: Any = None, svd: Any = None):
        self.components = components
        self.vectorizer = vectorizer
        self.svd = svd

    def fit(self, texts: list[str]) -> "SvdEncoder":
        from sklearn.decomposition import TruncatedSVD  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        matrix = self.vectorizer.fit_transform(texts)
        # SVD cannot produce more components than the smaller matrix dimension.
        components = max(2, min(self.components, min(matrix.shape) - 1))
        self.svd = TruncatedSVD(n_components=components, random_state=42)
        self.svd.fit(matrix)
        return self

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("SvdEncoder must be fitted before encoding.")
        reduced = self.svd.transform(self.vectorizer.transform(texts))
        return [normalize([float(value) for value in row]) for row in reduced]

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "svd_encoder.pkl").open("wb") as handle:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd}, handle)

    @classmethod
    def load(cls, directory: str | Path) -> "SvdEncoder":
        with (Path(directory) / "svd_encoder.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        return cls(vectorizer=payload["vectorizer"], svd=payload["svd"])


class TransformerEncoder:
    """Sentence-transformer embeddings (default `BAAI/bge-small-en-v1.5`).

    bge models are trained with an asymmetric retrieval objective: queries are prefixed with an
    instruction while passages are not. Encoding both sides identically measurably degrades
    recall, so the query prefix is applied here rather than left to the caller.
    """

    name = "bge_small"
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", model: Any = None):
        self.model_name = model_name
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def fit(self, texts: list[str]) -> "TransformerEncoder":
        return self  # pretrained; nothing to fit

    def encode(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        payload = [self.QUERY_PREFIX + text for text in texts] if is_query else list(texts)
        vectors = self.model.encode(payload, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        return [[float(value) for value in row] for row in vectors]

    def save(self, directory: str | Path) -> None:
        return None  # weights live in the sentence-transformers cache


def build_encoder(kind: str, model_name: str = "BAAI/bge-small-en-v1.5") -> Encoder:
    if kind in {"hashed", "hashed_bow"}:
        return HashedEncoder()
    if kind in {"svd", "tfidf_svd", "lsa"}:
        return SvdEncoder()
    if kind in {"transformer", "bge", "bge_small", "sentence_transformer"}:
        return TransformerEncoder(model_name)
    raise ValueError(f"Unknown encoder kind: {kind}")
