"""Stage 2 benchmark: labeled structure metrics on the fixture + coverage stats at scale.

Usage:
  # labeled metrics against the annotated fixture
  python scripts/evaluate_stage2.py --packet data/samples/sample_packet.pdf \
      --ground-truth data/samples/ground_truth.json

  # additionally run label-free coverage over real OpenPSS pages
  python scripts/evaluate_stage2.py --packet ... --ground-truth ... \
      --openpss-manifest data/raw/openpss/test/manifest.json --openpss-streams 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.domain import DocumentGroup, PageRepresentation
from src.evaluation.benchmark import write_results
from src.evaluation.stage2_metrics import coverage_statistics, element_metrics, field_metrics, heading_metrics, page_reference_accuracy
from src.ingestion.pdf_parser import PDFParser
from src.stage2.chunker import chunk_document
from src.stage2.structure_parser import structure_document

parser = argparse.ArgumentParser()
parser.add_argument("--packet", default="data/samples/sample_packet.pdf")
parser.add_argument("--ground-truth", default="data/samples/ground_truth.json")
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--openpss-manifest", default=None)
parser.add_argument("--openpss-streams", type=int, default=20)
parser.add_argument("--output", default="outputs/benchmarks/stage2_results")
args = parser.parse_args()

config = load_config(args.config)
truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))["documents"]

# ---- Labeled run on the annotated fixture -------------------------------------------------
pages = PDFParser(config.ingestion, config.runtime).parse(args.packet, None)
page_texts = {page.page_number: page.text for page in pages}
groups = [
    DocumentGroup(doc_id=f"fixture_doc_{index:03d}", pages=entry["pages"], doc_type=entry["doc_type"])
    for index, entry in enumerate(truth, 1)
]
start = time.perf_counter()
documents = [structure_document(group, pages, Path(args.packet).name, args.packet) for group in groups]
chunks = [chunk for document in documents for chunk in chunk_document(document, config.chunking)]
elapsed = time.perf_counter() - start

headings = heading_metrics(documents, truth)
tables = element_metrics(documents, truth, "table", "tables")
lists = element_metrics(documents, truth, "list", "lists")
captions = element_metrics(documents, truth, "caption", "captions")
fields = field_metrics(documents, truth)
pages_metric = page_reference_accuracy(documents, truth)
coverage = coverage_statistics(documents, chunks, page_texts, elapsed)

titles_correct = sum(
    1 for document, expected in zip(documents, truth)
    if document.metadata.get("document_title", "").strip().lower() == expected.get("title", "").strip().lower()
)

rows: list[dict] = [{
    "experiment": "rule_based_structure_fixture",
    "evaluation_set": "data/samples/ground_truth.json (authored fixture, 4 documents / 9 pages)",
    "heading_precision": headings["precision"], "heading_recall": headings["recall"], "heading_f1": headings["f1"],
    "table_precision": tables["precision"], "table_recall": tables["recall"], "table_f1": tables["f1"],
    "list_precision": lists["precision"], "list_recall": lists["recall"], "list_f1": lists["f1"],
    "caption_precision": captions["precision"], "caption_recall": captions["recall"], "caption_f1": captions["f1"],
    "document_title_accuracy": round(titles_correct / max(len(truth), 1), 4),
    "type_field_accuracy": fields["field_accuracy"],
    "type_fields_correct": fields["fields_correct"], "type_fields_expected": fields["fields_expected"],
    **pages_metric, **coverage,
    "status": "labeled structure metrics; fixture is authored because no public page-stream dataset ships heading/table annotations",
}]

# ---- Label-free coverage over real documents ----------------------------------------------
if args.openpss_manifest:
    manifest = json.loads(Path(args.openpss_manifest).read_text(encoding="utf-8"))
    streams = manifest["streams"][: args.openpss_streams]
    real_documents = []
    real_chunks = []
    real_page_texts: dict[int, str] = {}
    offset = 0
    start = time.perf_counter()
    for stream in streams:
        stream_pages = [
            PageRepresentation(
                page_number=page["position"], text=page["text"], blocks=[], fonts=[],
                width=float(page["width"]), height=float(page["height"]), image_path=page["image_path"],
            )
            for page in stream["pages"]
        ]
        for page in stream_pages:
            real_page_texts[offset + page.page_number] = page.text
        offset += len(stream_pages)
        group = DocumentGroup(doc_id=stream["stream_id"], pages=[p.page_number for p in stream_pages], doc_type="unknown")
        document = structure_document(group, stream_pages, stream["stream_id"], None)
        real_documents.append(document)
        real_chunks += chunk_document(document, config.chunking)
    real_elapsed = time.perf_counter() - start
    real_coverage = coverage_statistics(real_documents, real_chunks, real_page_texts, real_elapsed)
    rows.append({
        "experiment": "rule_based_structure_openpss",
        "evaluation_set": f"OpenPSS SHORT test, first {len(streams)} streams (no structure labels)",
        "heading_precision": None, "heading_recall": None, "heading_f1": None,
        "table_precision": None, "table_recall": None, "table_f1": None,
        "list_precision": None, "list_recall": None, "list_f1": None,
        **real_coverage,
        "status": "label-free coverage only; OpenPSS provides no heading/table ground truth",
    })

write_results(rows, Path(args.output).parent, Path(args.output).name)
print(json.dumps(rows, indent=2))
print(f"\nWrote {args.output}.json / .csv")
