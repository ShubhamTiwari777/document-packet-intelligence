# Document Packet Intelligence & Evidence Retrieval

Splits a PDF packet containing several independent documents, converts each into a structured
representation, and retrieves supporting evidence for a query. It does not generate answers:
every result carries the retrieved text, document id, page reference and a confidence score.

Full measurements and design rationale: **[technical_report.pdf](technical_report.pdf)**.
Architecture: **[docs/architecture.pdf](docs/architecture.pdf)**.

## Pipeline

```
PDF → PDFParser (PyMuPDF) → [Stage 1] 21 pairwise features → calibrated GBM → grouping → classifier
                          → [Stage 2] boilerplate → headings → elements/tables → section tree → chunks
                          → [Stage 3] BM25 + dense → RRF → reranker → evidence + page + confidence
                          → FastAPI /process, /retrieve
```

Everything runs CPU-only. Committed models total 5.8 MB; no GPU, no external API, no network call
at inference.

## Headline results

| Stage | Measurement | Result |
|---|---|---|
| 1 — boundaries | OpenPSS SHORT test, 108 streams | F1 0.379 (lift +0.191 over trivial) |
| 1 — boundaries | DocSplit `our200` benchmark | F1 0.823 — but **trivial is 0.815**, lift +0.008 |
| 1 — classification | RVL-CDIP held-out, 16 classes | 0.807 accuracy / 0.786 macro-F1 |
| 2 — structure | annotated fixtures | heading / table / list / caption F1 1.00 |
| 3 — retrieval | 35 queries, 515-chunk corpus | R@1 0.771, MRR 0.821 (0.886 / 0.929 with bge) |

The DocSplit row is the important one and is discussed in §5.1 of the report: raw F1 flatters the
system there, and the system does not transfer to short packets. Read `lift_over_trivial`, not F1.

## Installation

Python 3.11+ (developed on 3.14).

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Optional extras, neither required for the default pipeline:

- **Tesseract** — only when `ingestion.enable_ocr: true` in the config.
- **sentence-transformers + torch** (~1 GB) — enables `retrieval.encoder: transformer`, worth
  +0.115 R@1. Measured numbers are in `requirements.txt`.

## Run it

```bash
# Generate the sample packet, then process it end to end
python scripts/generate_sample_packet.py
python scripts/run_pipeline.py --input data/samples/sample_packet.pdf --output outputs/sample

# Retrieve evidence (never an answer)
python scripts/run_pipeline.py --query "What is the closing balance?" --processed-dir outputs/sample --top-k 5
```

A processed packet directory contains `pages.json`, `boundary_features.json`, `stage1.json`,
`structured_documents.json`, `chunks.json`, `markdown/*.md`, rendered pages, and the dense index.

## API

```bash
uvicorn src.api:create_app --factory --host 0.0.0.0 --port 8000
```

| Endpoint | Purpose |
|---|---|
| `POST /process` | Upload a PDF; returns groups, structure and chunks. `?include_structure=false&include_chunks=false` trims the payload. |
| `POST /retrieve` | `{query, processed_dir, top_k}` → evidence with `doc_id`, `page_ref`, `breadcrumb`, `confidence`. |
| `GET /health` | Reports whether a trained boundary model is configured. |

Docker:

```bash
docker build -t document-packet-intelligence .
docker run -p 8000:8000 document-packet-intelligence
```

## Reproducing the models and benchmarks

Datasets download themselves through the HuggingFace datasets-server REST API — no manual
downloads, and no `datasets`/`pyarrow` dependency (neither has a Python 3.14 wheel).

```bash
# Stage 1 — boundary detection
python scripts/fetch_openpss.py --config SHORT --split train --output data/raw/openpss/train --max-rows 16000
python scripts/fetch_openpss.py --config SHORT --split test  --output data/raw/openpss/test_full --max-rows 11462
python scripts/train_openpss_boundary.py --manifest data/raw/openpss/train/manifest.json --output models/boundary_openpss --calibrate
python scripts/evaluate_openpss_boundary.py --manifest data/raw/openpss/test_full/manifest.json --model models/boundary_openpss --output outputs/benchmarks/boundary_validation_full.json

# Score against the DocSplit benchmark (evaluation only -- never trained on)
python scripts/fetch_docsplit_benchmark.py
python scripts/evaluate_openpss_boundary.py --manifest data/raw/docsplit_benchmark/manifest.json --model models/boundary_openpss --output outputs/benchmarks/boundary_validation_docsplit.json

# Stage 1 — document type classification
python scripts/fetch_rvlcdip_text.py --output data/raw/rvlcdip/train_text.json --max-rows 12000
python scripts/train_document_classifier.py --training_json data/raw/rvlcdip/train_text.json --output models/document_classifier/tfidf_lr.pkl

# Stages 2 and 3
python scripts/generate_sample_packet.py && python scripts/generate_benchmark_report.py
python scripts/evaluate_stage2.py --packet data/samples/benchmark_report.pdf --ground-truth data/samples/benchmark_ground_truth.json
python scripts/evaluate_stage3.py --distractors data/raw/openpss/test_full/manifest.json --distractor-streams 2

# Reports
python scripts/generate_stage1_report.py
python scripts/build_report_pdf.py && python scripts/build_architecture_pdf.py
```

Results are written to `outputs/benchmarks/`. Negative results reproduce via
`--no-synthetic-blocks`, `--class-weight none`, and `scripts/train_rvlcdip_boundary.py`.

## Tests

```bash
pytest tests/ -q
```

34 tests covering the pipeline contracts plus regressions for every defect found during
development — table flattening, frozen breadcrumbs, silent classifier fallback, a dense index
that scored worse than its own lexical half, and threshold selection leaking into scoring.

## Configuration

All settings live in [`config/default.yaml`](config/default.yaml). Notable ones:

| Key | Default | Effect |
|---|---|---|
| `boundary.threshold` | 0.6334 | split where pair probability ≥ this |
| `classification.min_confidence` | 0.35 | below this, type is reported `unknown` |
| `retrieval.encoder` | `svd` | `hashed`, `svd`, or `transformer` |
| `retrieval.rerank` | `true` | feature reranker; +0.171 R@1 for ~0 ms |
| `ingestion.enable_ocr` | `false` | requires Tesseract when enabled |

Do not compare experiments across different data splits, render DPIs or label schemas. Rendered
visual features are capped at `runtime.max_render_pixels` so unusually large pages cannot exhaust
RAM.

## Known limitations

- Boundary detection is tuned for long page streams and is at the trivial baseline on short
  packets (report §5.1). Retraining in the target regime is the top future-work item.
- `passport` and `bank_statement` are lexicon-scored with **no held-out measurement**; no public
  labelled corpus for them was available.
- Stage 2's 1.00 scores are on authored fixtures — they measure conformance to chosen constructs,
  not generalisation. The label-free OpenPSS coverage run sits beside them for that reason.
