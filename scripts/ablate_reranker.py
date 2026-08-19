"""Ablate the reranker: per-feature contribution, shortlist depth, and isolated cost.

A reranker that helps in aggregate can still be carried entirely by one feature, or be paying for
a shortlist deeper than it needs. This measures both, plus the reranking cost separately from
retrieval, so the accuracy/latency trade is stated rather than assumed.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.domain import Chunk, DocumentGroup, PageRepresentation
from src.evaluation.stage3_metrics import retrieval_metrics
from src.ingestion.pdf_parser import PDFParser
from src.stage2.chunker import chunk_document
from src.stage2.structure_parser import structure_document
from src.stage3 import reranker as rr
from src.stage3.retriever import HybridRetriever

parser = argparse.ArgumentParser()
parser.add_argument("--queries", default="data/samples/retrieval_queries.json")
parser.add_argument("--distractors", default="data/raw/openpss/test_full/manifest.json")
parser.add_argument("--output", default="outputs/benchmarks/reranker_ablation.json")
args = parser.parse_args()

config = load_config("config/default.yaml")
spec = json.loads(Path(args.queries).read_text(encoding="utf-8"))

corpus: list[Chunk] = []
for source in spec["corpora"]:
    pages = PDFParser(config.ingestion, config.runtime).parse(source, None)
    group = DocumentGroup(doc_id=Path(source).stem, pages=[p.page_number for p in pages], doc_type="document")
    corpus += chunk_document(structure_document(group, pages, Path(source).name, source), config.chunking)
manifest = json.loads(Path(args.distractors).read_text(encoding="utf-8"))
for stream in manifest["streams"][:2]:
    pages = [PageRepresentation(page_number=p["position"], text=p["text"], blocks=[], fonts=[],
                                width=float(p["width"]), height=float(p["height"]), image_path=p["image_path"])
             for p in stream["pages"]]
    group = DocumentGroup(doc_id=stream["stream_id"], pages=[p.page_number for p in pages], doc_type="unknown")
    corpus += chunk_document(structure_document(group, pages, stream["stream_id"], None), config.chunking)

queries = []
for entry in spec["queries"]:
    gold = {c.chunk_id for c in corpus if all(n.lower() in c.text.lower() for n in entry["answer_contains"])}
    if gold:
        queries.append({**entry, "gold": gold})


def score(retriever, tag):
    rankings, relevant, latencies = [], [], []
    for entry in queries:
        start = time.perf_counter()
        results = retriever.retrieve(entry["query"], top_k=5)
        latencies.append((time.perf_counter() - start) * 1000)
        rankings.append([r.chunk_id for r in results]); relevant.append(entry["gold"])
    metrics = retrieval_metrics(rankings, relevant)
    return {"variant": tag, "recall@1": round(metrics["recall@1"], 4), "recall@5": round(metrics["recall@5"], 4),
            "mrr": round(metrics["mrr"], 4), "ndcg": round(metrics["ndcg"], 4),
            "mean_latency_ms": round(sum(latencies) / len(latencies), 2)}


rows = []
base = load_config("config/default.yaml").retrieval
base.rerank = False
rows.append(score(HybridRetriever(corpus, base), "no reranker"))

full = load_config("config/default.yaml").retrieval
full.rerank = True
rows.append(score(HybridRetriever(corpus, full), "all features"))

original = dict(rr.WEIGHTS)
for feature in original:
    rr.WEIGHTS = {k: (0.0 if k == feature else v) for k, v in original.items()}
    rows.append(score(HybridRetriever(corpus, full), f"without {feature}"))
for feature in original:
    rr.WEIGHTS = {k: (v if k == feature else 0.0) for k, v in original.items()}
    rows.append(score(HybridRetriever(corpus, full), f"{feature} only"))
rr.WEIGHTS = original

print(f"{'variant':<22} R@1     R@5     MRR     nDCG    latency")
for row in rows:
    print(f"  {row['variant']:<20} {row['recall@1']:.4f}  {row['recall@5']:.4f}  {row['mrr']:.4f}  {row['ndcg']:.4f}  {row['mean_latency_ms']:>6.1f}ms")

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps({"queries": len(queries), "corpus_chunks": len(corpus), "results": rows}, indent=2), encoding="utf-8")
print(f"\nWrote {args.output}")
