"""Hybrid retriever: returns evidence only, never generated answers.

Architecture: BM25 (lexical) and a dense index (semantic) are searched independently, their
rankings are combined with reciprocal-rank fusion, and an optional feature reranker reorders the
shortlist. Every returned item carries its document id, page references and a normalised
confidence, plus the section breadcrumb Stage 2 produced.
"""
from __future__ import annotations

from src.config import RetrievalConfig
from src.domain import Chunk, EvidenceResult
from src.stage3.bm25_index import BM25Index
from src.stage3.dense_index import DenseIndex
from src.stage3.encoders import Encoder, build_encoder
from src.stage3.query_classifier import classify_query
from src.stage3.reranker import rerank
from src.stage3.rrf import reciprocal_rank_fusion


def _confidence(scores: list[float]) -> list[float]:
    """Normalise scores onto [0, 1] relative to the returned candidates.

    Deliberately relative, not absolute: RRF scores have no natural scale, so the value
    communicates "how strongly this beats the alternatives for this query", which is what a
    downstream consumer thresholds on. A single result scores 1.0.
    """
    if not scores:
        return []
    top, bottom = max(scores), min(scores)
    if top - bottom < 1e-12:
        return [1.0 for _ in scores]
    return [(score - bottom) / (top - bottom) for score in scores]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], config: RetrievalConfig, encoder: Encoder | None = None):
        self.chunks, self.config = chunks, config
        selected = encoder or build_encoder(getattr(config, "encoder", "hashed"), getattr(config, "embedding_model", "BAAI/bge-small-en-v1.5"))
        self.dense, self.bm25 = DenseIndex(chunks, selected), BM25Index(chunks)

    def retrieve(self, query: str, top_k: int | None = None, query_aware: bool = False) -> list[EvidenceResult]:
        limit = top_k or self.config.top_k
        dense = self.dense.search(query, self.config.dense_top_k)
        lexical = self.bm25.search(query, self.config.bm25_top_k)

        weights = [1.0, 1.0]
        if query_aware and classify_query(query) in {"identifier_exact", "entity_heavy"}:
            weights = [0.75, 1.5]  # identifier lookups reward the lexical ranker
        fused = reciprocal_rank_fusion([dense, lexical], self.config.rrf_k, weights)

        # Rerank a shortlist rather than the whole fusion output: the features are cheap but the
        # candidate set is what keeps them cheap.
        shortlist = [(chunk, score) for chunk, score, _ in fused[: max(limit * 4, limit)]]
        reranked = rerank(query, shortlist, self.config.rerank)[:limit]

        dense_scores = {chunk.chunk_id: score for chunk, score in dense}
        bm25_scores = {chunk.chunk_id: score for chunk, score in lexical}
        ranking_scores = [reranker if reranker is not None else fusion for _, fusion, reranker in reranked]
        confidences = _confidence(ranking_scores)

        return [
            EvidenceResult(
                evidence=chunk.text, doc_id=chunk.doc_id, page_ref=chunk.page_refs,
                rrf_score=fusion, reranker_score=reranker,
                dense_score=dense_scores.get(chunk.chunk_id), bm25_score=bm25_scores.get(chunk.chunk_id),
                confidence=round(confidence, 4), chunk_id=chunk.chunk_id, section_id=chunk.section_id,
                doc_type=chunk.doc_type, breadcrumb=list(chunk.breadcrumb), element_types=list(chunk.element_types),
            )
            for (chunk, fusion, reranker), confidence in zip(reranked, confidences)
        ]

    def save(self, directory: str) -> None:
        self.dense.save(directory)
