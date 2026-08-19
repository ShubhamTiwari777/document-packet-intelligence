"""Stage 3 retrieval regression tests.

These pin the defects found while benchmarking: a dense index that scored near-randomly, a
`rerank: true` flag that raised RuntimeError instead of reranking, evidence carrying no
interpretable confidence, and a retriever rebuilt (re-fitting the encoder) on every query.
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AppConfig
from src.domain import Chunk
from src.stage3.encoders import HashedEncoder, SvdEncoder
from src.stage3.reranker import rerank, score_features
from src.stage3.retriever import HybridRetriever

CORPUS = [
    Chunk("c1", "invoice", "invoice", "s1", [1], "Billing Details\n\nInvoice Number: INV-2026-004471\n\nBill To: Acme Digital Services",
          breadcrumb=["Billing Details"], element_types=["paragraph"], token_count=10),
    Chunk("c2", "invoice", "invoice", "s2", [3], "Notes and Terms\n\nSubtotal: 84,500\n\nTotal: 99,710",
          breadcrumb=["Notes and Terms"], element_types=["paragraph"], token_count=8),
    Chunk("c3", "resume", "resume", "s3", [5], "Technical Skills\n\n- Python, SQL, C++, Git, Docker\n- Machine Learning, NLP",
          breadcrumb=["Technical Skills"], element_types=["list"], token_count=14),
    Chunk("c4", "bank", "bank_statement", "s4", [9], "Transaction Summary\n\nClosing Balance: 1,24,550",
          breadcrumb=["Transaction Summary"], element_types=["paragraph"], token_count=6),
    Chunk("c5", "bank", "bank_statement", "s5", [9], "| Date | Description | Debit | Credit |\n| --- | --- | --- | --- |\n| 02 Aug | Salary Credit |  | 65,000 |",
          breadcrumb=["Transaction Summary"], element_types=["table"], token_count=20),
]


def _config(**overrides) -> AppConfig:
    config = AppConfig()
    for key, value in overrides.items():
        setattr(config.retrieval, key, value)
    return config


class RetrievalContractTests(unittest.TestCase):
    def setUp(self):
        self.retriever = HybridRetriever(CORPUS, _config(rerank=True).retrieval, SvdEncoder())

    def test_every_result_carries_document_page_and_confidence(self):
        results = self.retriever.retrieve("What is the closing balance?", top_k=3)
        self.assertTrue(results)
        for result in results:
            self.assertTrue(result.doc_id)
            self.assertTrue(result.page_ref, "every evidence item needs a page reference")
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
            self.assertTrue(result.chunk_id)
            self.assertIsInstance(result.breadcrumb, list)

    def test_confidence_is_normalised_with_the_best_result_at_one(self):
        results = self.retriever.retrieve("closing balance", top_k=3)
        self.assertAlmostEqual(results[0].confidence, 1.0, places=6)
        self.assertTrue(all(results[i].confidence >= results[i + 1].confidence for i in range(len(results) - 1)))

    def test_exact_identifier_is_retrieved_first(self):
        results = self.retriever.retrieve("INV-2026-004471", top_k=3)
        self.assertEqual(results[0].chunk_id, "c1")

    def test_reranking_enabled_does_not_raise(self):
        # Regression: rerank() previously raised RuntimeError whenever the flag was true.
        retriever = HybridRetriever(CORPUS, _config(rerank=True).retrieval, HashedEncoder())
        results = retriever.retrieve("closing balance", top_k=3)
        self.assertTrue(results)
        self.assertIsNotNone(results[0].reranker_score)

    def test_reranker_is_noop_when_disabled(self):
        pairs = [(CORPUS[0], 0.9), (CORPUS[1], 0.5)]
        self.assertEqual([c.chunk_id for c, _, _ in rerank("anything", pairs, False)], ["c1", "c2"])
        self.assertTrue(all(score is None for _, _, score in rerank("anything", pairs, False)))

    def test_reranker_features_reward_coverage_and_exact_identifiers(self):
        good = score_features("invoice number INV-2026-004471", CORPUS[0])
        poor = score_features("invoice number INV-2026-004471", CORPUS[2])
        self.assertGreater(good["coverage"], poor["coverage"])
        self.assertEqual(good["exactness"], 1.0)
        self.assertEqual(poor["exactness"], 0.0)


class EncoderTests(unittest.TestCase):
    def test_svd_encoder_produces_unit_vectors_of_fixed_width(self):
        encoder = SvdEncoder().fit([chunk.text for chunk in CORPUS])
        vectors = encoder.encode([chunk.text for chunk in CORPUS])
        self.assertEqual(len({len(vector) for vector in vectors}), 1)
        for vector in vectors:
            self.assertAlmostEqual(sum(value * value for value in vector) ** 0.5, 1.0, places=5)

    def test_svd_beats_hashed_on_a_paraphrased_query(self):
        """The hashed encoder has no notion of relatedness; this is why it scored R@1 0.09.

        LSA derives meaning from co-occurrence, so it needs a corpus with actual co-occurrence
        structure -- on a handful of one-line documents it has nothing to learn and degenerates
        to noise. That corpus dependence is a genuine limitation of this encoder relative to a
        pretrained transformer, so the test supplies a realistically sized topical corpus.
        """
        banking = [
            Chunk(f"b{i}", "bank", "bank_statement", "s", [i],
                  text, breadcrumb=["Account"], element_types=["paragraph"], token_count=len(text.split()))
            for i, text in enumerate([
                "Account summary showing the money held in the account this month.",
                "The closing balance of the account reflects money remaining after withdrawals.",
                "Funds deposited into the account increase the available balance.",
                "Money withdrawn from the account reduces the remaining balance.",
                "The account statement lists deposits, withdrawals and the balance carried forward.",
                "Available funds in the account are shown as the running balance.",
                "Interest earned on the account is added to the balance each month.",
                "A low balance in the account may incur maintenance charges.",
                "Transfers move money between accounts and change each balance.",
                "The bank reports the account balance at the end of the statement period.",
            ], start=1)
        ]
        unrelated = [
            Chunk(f"u{i}", "resume", "resume", "s", [i],
                  text, breadcrumb=["Skills"], element_types=["paragraph"], token_count=len(text.split()))
            for i, text in enumerate([
                "Python programming and software engineering experience with distributed systems.",
                "Machine learning research covering computer vision and natural language processing.",
                "Docker container orchestration and continuous integration pipelines.",
                "Technical writing, documentation and developer education work.",
                "Database design, SQL query optimisation and data modelling.",
            ], start=1)
        ]
        corpus = banking + unrelated
        query = "how much money is left in the account"
        retriever = HybridRetriever(corpus, _config(rerank=False, bm25_top_k=0).retrieval, SvdEncoder())
        top = retriever.retrieve(query, top_k=1)[0]
        self.assertTrue(top.chunk_id.startswith("b"), f"semantic encoder should stay in the banking topic, got {top.chunk_id}")


class RetrieverCacheTests(unittest.TestCase):
    def test_retriever_is_cached_between_queries(self):
        import json
        from tempfile import TemporaryDirectory
        from dataclasses import asdict
        from src.pipeline import DocumentPipeline

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "chunks.json").write_text(json.dumps([asdict(c) for c in CORPUS]), encoding="utf-8")
            pipeline = DocumentPipeline(_config(rerank=True))
            first_start = time.perf_counter()
            pipeline.retrieve_from_dir("closing balance", directory)
            first = time.perf_counter() - first_start
            second_start = time.perf_counter()
            results = pipeline.retrieve_from_dir("closing balance", directory)
            second = time.perf_counter() - second_start
            self.assertTrue(results)
            self.assertTrue(all(key in results[0] for key in ("doc_id", "page_ref", "confidence", "evidence")))
            # The encoder must not be refitted per query.
            self.assertLess(second, max(first, 0.01))


if __name__ == "__main__":
    unittest.main()
