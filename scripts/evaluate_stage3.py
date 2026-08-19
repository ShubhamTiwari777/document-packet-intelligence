"""Stage 3 retrieval benchmark: real IR metrics across retrieval configurations.

Builds a corpus from the annotated fixtures, optionally padded with real OpenPSS chunks as
distractors, resolves gold judgments by answer-string matching, then measures every retrieval
configuration on the same corpus and query set.

  python scripts/evaluate_stage3.py --queries data/samples/retrieval_queries.json \
      --distractors data/raw/openpss/test/manifest.json --distractor-streams 3
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.domain import Chunk, DocumentGroup, PageRepresentation
from src.evaluation.benchmark import write_results
from src.evaluation.stage3_metrics import retrieval_metrics
from src.ingestion.pdf_parser import PDFParser
from src.stage2.chunker import chunk_document
from src.stage2.structure_parser import structure_document
from src.stage3.encoders import build_encoder
from src.stage3.retriever import HybridRetriever

parser = argparse.ArgumentParser()
parser.add_argument("--queries", default="data/samples/retrieval_queries.json")
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--distractors", default=None, help="OpenPSS manifest to pad the corpus with")
parser.add_argument("--distractor-streams", type=int, default=3)
parser.add_argument("--output", default="outputs/benchmarks/stage3_results")
parser.add_argument("--error-output", default="outputs/error_analysis/stage3_errors.json")
parser.add_argument("--skip-transformer", action="store_true")
args = parser.parse_args()

config = load_config(args.config)
spec = json.loads(Path(args.queries).read_text(encoding="utf-8"))


def chunks_for(pdf_path: str) -> list[Chunk]:
    pages = PDFParser(config.ingestion, config.runtime).parse(pdf_path, None)
    group = DocumentGroup(doc_id=Path(pdf_path).stem, pages=[page.page_number for page in pages], doc_type="document")
    document = structure_document(group, pages, Path(pdf_path).name, pdf_path)
    return chunk_document(document, config.chunking)


corpus: list[Chunk] = []
for source in spec["corpora"]:
    corpus += chunks_for(source)
evaluated_chunks = len(corpus)

if args.distractors:
    manifest = json.loads(Path(args.distractors).read_text(encoding="utf-8"))
    for stream in manifest["streams"][: args.distractor_streams]:
        pages = [
            PageRepresentation(page_number=page["position"], text=page["text"], blocks=[], fonts=[],
                               width=float(page["width"]), height=float(page["height"]), image_path=page["image_path"])
            for page in stream["pages"]
        ]
        group = DocumentGroup(doc_id=stream["stream_id"], pages=[p.page_number for p in pages], doc_type="unknown")
        corpus += chunk_document(structure_document(group, pages, stream["stream_id"], None), config.chunking)

# Resolve judgments by answer-string containment so they survive re-chunking.
queries: list[dict] = []
unresolved: list[str] = []
for entry in spec["queries"]:
    gold = {
        chunk.chunk_id for chunk in corpus
        if all(needle.lower() in chunk.text.lower() for needle in entry["answer_contains"])
    }
    if gold:
        queries.append({**entry, "gold": gold})
    else:
        unresolved.append(entry["id"])

if not queries:
    raise SystemExit("No queries could be resolved against the corpus.")

CONFIGURATIONS = [
    ("bm25_only", {"encoder": "hashed", "dense_top_k": 0, "rerank": False, "query_aware": False}),
    ("dense_hashed_only", {"encoder": "hashed", "bm25_top_k": 0, "rerank": False, "query_aware": False}),
    ("dense_svd_only", {"encoder": "svd", "bm25_top_k": 0, "rerank": False, "query_aware": False}),
    ("rrf_hashed", {"encoder": "hashed", "rerank": False, "query_aware": False}),
    ("rrf_svd", {"encoder": "svd", "rerank": False, "query_aware": False}),
    ("rrf_svd_query_aware", {"encoder": "svd", "rerank": False, "query_aware": True}),
    ("rrf_svd_reranked", {"encoder": "svd", "rerank": True, "query_aware": False}),
]
if not args.skip_transformer:
    CONFIGURATIONS += [
        ("dense_bge_only", {"encoder": "transformer", "bm25_top_k": 0, "rerank": False, "query_aware": False}),
        ("rrf_bge", {"encoder": "transformer", "rerank": False, "query_aware": False}),
        ("rrf_bge_reranked", {"encoder": "transformer", "rerank": True, "query_aware": False}),
    ]

rows: list[dict] = []
errors: dict[str, list[dict]] = {}

# Warm up the heavy imports once so the first configuration measured does not absorb sklearn /
# torch import cost and report a misleadingly large indexing time.
for warm in ("svd", "transformer"):
    try:
        build_encoder(warm, config.retrieval.embedding_model).fit(["warm up"]).encode(["warm up"])
    except Exception:
        pass

for name, overrides in CONFIGURATIONS:
    retrieval = load_config(args.config).retrieval
    retrieval.rerank = overrides.get("rerank", False)
    if "dense_top_k" in overrides:
        retrieval.dense_top_k = overrides["dense_top_k"]
    if "bm25_top_k" in overrides:
        retrieval.bm25_top_k = overrides["bm25_top_k"]
    query_aware = overrides.get("query_aware", False)

    try:
        encoder = build_encoder(overrides["encoder"], retrieval.embedding_model)
        tracemalloc.start()
        index_start = time.perf_counter()
        retriever = HybridRetriever(corpus, retrieval, encoder)
        index_seconds = time.perf_counter() - index_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    except Exception as exc:  # e.g. sentence-transformers not installed
        rows.append({"experiment": name, "status": f"skipped: {type(exc).__name__}: {str(exc)[:160]}"})
        continue

    rankings: list[list[str]] = []
    relevant: list[set[str]] = []
    latencies: list[float] = []
    per_query: list[dict] = []
    for entry in queries:
        start = time.perf_counter()
        results = retriever.retrieve(entry["query"], top_k=5, query_aware=query_aware)
        latencies.append((time.perf_counter() - start) * 1000)
        ranking = [result.chunk_id for result in results]
        rankings.append(ranking)
        relevant.append(entry["gold"])
        hit_rank = next((index + 1 for index, cid in enumerate(ranking) if cid in entry["gold"]), None)
        per_query.append({
            "id": entry["id"], "query": entry["query"], "type": entry["type"], "hit_rank": hit_rank,
            "top_result": results[0].evidence[:150] if results else "",
            "top_confidence": results[0].confidence if results else None,
            "top_pages": results[0].page_ref if results else [],
        })

    metrics = retrieval_metrics(rankings, relevant)
    latencies_sorted = sorted(latencies)
    rows.append({
        "experiment": name, "encoder": overrides["encoder"],
        "reranked": retrieval.rerank, "query_aware": query_aware,
        "queries": len(queries), "corpus_chunks": len(corpus), "fixture_chunks": evaluated_chunks,
        **{key: round(value, 4) for key, value in metrics.items()},
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
        "p95_latency_ms": round(latencies_sorted[int(len(latencies_sorted) * 0.95) - 1], 2),
        "indexing_seconds": round(index_seconds, 3),
        "index_peak_memory_mb": round(peak / 1e6, 2),
        "status": "measured",
    })
    errors[name] = [record for record in per_query if record["hit_rank"] is None or record["hit_rank"] > 1]

write_results(rows, Path(args.output).parent, Path(args.output).name)
error_path = Path(args.error_output)
error_path.parent.mkdir(parents=True, exist_ok=True)
error_path.write_text(json.dumps({
    "unresolved_queries": unresolved,
    "corpus_chunks": len(corpus),
    "per_configuration_misses": errors,
}, indent=2), encoding="utf-8")

for row in rows:
    if row.get("status") == "measured":
        print(f"  {row['experiment']:<22} R@1={row['recall@1']:.3f} R@5={row['recall@5']:.3f} "
              f"MRR={row['mrr']:.3f} nDCG={row['ndcg']:.3f} lat={row['mean_latency_ms']:.1f}ms idx={row['indexing_seconds']:.2f}s")
    else:
        print(f"  {row['experiment']:<22} {row['status']}")
print(f"\nWrote {args.output}.json and {error_path}")
