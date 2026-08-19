"""Dense index over chunk embeddings, agnostic to which encoder produced them."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math

from src.domain import Chunk
from src.stage3.encoders import Encoder, HashedEncoder, SvdEncoder, TransformerEncoder, build_encoder


def _dot(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


class DenseIndex:
    def __init__(self, chunks: list[Chunk], encoder: Encoder | None = None, vectors: list[list[float]] | None = None):
        self.chunks = chunks
        self.encoder = encoder or HashedEncoder()
        if vectors is not None:
            self.vectors = vectors
        else:
            texts = [chunk.text for chunk in chunks]
            if texts:
                self.encoder.fit(texts)
            self.vectors = self.encoder.encode(texts) if texts else []
        # Vectors are unit-normalised by every encoder, so cosine reduces to a dot product.
        self.norms = [math.sqrt(_dot(vector, vector)) or 1.0 for vector in self.vectors]

    def _encode_query(self, query: str) -> list[float]:
        if isinstance(self.encoder, TransformerEncoder):
            return self.encoder.encode([query], is_query=True)[0]
        return self.encoder.encode([query])[0]

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        vector = self._encode_query(query)
        query_norm = math.sqrt(_dot(vector, vector)) or 1.0
        scored = [
            (chunk, _dot(vector, item) / (query_norm * norm))
            for chunk, item, norm in zip(self.chunks, self.vectors, self.norms)
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "dense_index.json").write_text(json.dumps({
            "encoder": getattr(self.encoder, "name", "unknown"),
            "dimensions": len(self.vectors[0]) if self.vectors else 0,
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "vectors": self.vectors,
        }, indent=2), encoding="utf-8")
        save = getattr(self.encoder, "save", None)
        if callable(save):
            save(target)

    @classmethod
    def load(cls, directory: str | Path) -> "DenseIndex":
        target = Path(directory)
        data = json.loads((target / "dense_index.json").read_text(encoding="utf-8"))
        name = data.get("encoder", "hashed_bow")
        if name == "tfidf_svd" and (target / "svd_encoder.pkl").exists():
            encoder: Encoder = SvdEncoder.load(target)
        elif name == "bge_small":
            encoder = TransformerEncoder()
        else:
            encoder = build_encoder("hashed")
        return cls([Chunk(**item) for item in data["chunks"]], encoder, data["vectors"])
