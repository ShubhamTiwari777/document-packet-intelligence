# Document Packet Intelligence & Evidence Retrieval

An evidence-retrieval system for PDF packets containing multiple independent documents. It does not generate answers: every result contains the retrieved evidence, document ID, original page reference, and rank score.

## What is implemented

```mermaid
flowchart LR
  A[PDF packet] --> B[PyMuPDF ingestion]
  B --> C[Text, layout, visual and heuristic features]
  C --> D[Trained boundary classifier]
  D --> E[Document grouping]
  E --> F[Document type classification]
  F --> G[Rule-based structured JSON]
  G --> H[Section-aware chunks]
  H --> I[Dense and BM25 retrieval]
  I --> J[RRF fusion]
  J --> K[Evidence + doc ID + page reference]
```

The primary implementation is CPU-first. It uses deterministic hashed text vectors and grayscale page histograms as an immediately runnable fallback, while retaining explicit extension points for BGE/E5, CLIP, LayoutLMv3 and a cross-encoder. Optional heavyweight models are deliberately not installed or downloaded implicitly.

## Design choices

- PyMuPDF is the primary parser. OCR is triggered only for pages with an inadequate native text layer.
- The boundary model is trained by this repository on extracted adjacent-page features. The default production fallback is transparent weighted fusion; `train_boundary_model.py` writes the trained HistGradientBoosting model and its feature names/configuration.
- Structured data is JSON with page-level evidence and geometry. The structure path is deterministic; pdfplumber table extraction is gated rather than run for every page.
- Retrieval uses separate dense-like and lexical rankers with reciprocal-rank fusion. Scores remain retrieval scores; no score is represented as a probability.

## Installation

Python 3.11 is the Docker target. For a local environment:

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

Tesseract itself is additionally required only when `ingestion.enable_ocr: true`. Optional model experiments require uncommenting/installing the listed packages and updating the model adapter; they are not required for the baseline pipeline.

## Dataset setup

Download DocSplit v2 yourself, respecting its license, into `data/raw/docsplit/`. This repository does not automatically download the dataset. Normalize ground truth into adjacent-pair records such as:

```json
{"packet_id":"packet_001","left_page":1,"right_page":2,"is_boundary":0}
```

Use the training split only for fitting the boundary classifier. Keep validation/test packets distinct at the packet level to avoid adjacent-pair leakage.

## Commands

```bash
# Validate a locally downloaded dataset
python scripts/prepare_dataset.py --dataset data/raw/docsplit

# Convert downloaded DocSplit CSV labels into packet-level manifests
python scripts/prepare_docsplit_labels.py --csv "C:\hf-data\docsplit\datasets\poly_seq\small\train.csv" --output data/processed/docsplit_train_manifest.json

# Reconstruct a bounded, local packet-training set from the downloaded source PDFs
python scripts/build_docsplit_packets.py --manifest data/processed/docsplit_train_manifest.json --raw-pdfs "C:\hf-data\rvl_cdip_n_mp" --output data/processed/docsplit_packets --limit 500

# Extract features and train the custom boundary model. Add --render-pages for visual histogram features.
python scripts/train_docsplit_boundary.py --packet-index data/processed/docsplit_packets/packet_index.json --output models/boundary --render-pages

# Build held-out validation packets, then select the boundary threshold on validation data
python scripts/prepare_docsplit_labels.py --csv "C:\hf-data\docsplit\datasets\poly_seq\small\validation.csv" --output data/processed/docsplit_validation_manifest.json
python scripts/build_docsplit_packets.py --manifest data/processed/docsplit_validation_manifest.json --raw-pdfs "C:\hf-data\rvl_cdip_n_mp" --output data/processed/docsplit_validation_packets --limit 500
python scripts/evaluate_docsplit_boundary.py --packet-index data/processed/docsplit_validation_packets/packet_index.json --model models/boundary --render-pages

# Extract cached adjacent-page features
python scripts/extract_features.py --input packet.pdf --output data/processed/features/packet.json

# Train the custom boundary model (labels align with feature rows)
python scripts/train_boundary_model.py --features data/processed/features/packet.json --labels labels.json --output models/boundary

# Train document type classifier from [{"text": ..., "label": ...}]
python scripts/train_document_classifier.py --training_json docs.json --output models/document_classifier.pkl

# Alternative embedding-centroid classifier (portable hashed-embedding baseline)
python scripts/train_document_classifier.py --training_json docs.json --method embedding_centroid --output models/document_centroids.json

# Process a PDF end-to-end
python scripts/run_pipeline.py --input packet.pdf --output outputs/packet_001

# Process with trained models
python scripts/run_pipeline.py --input packet.pdf --output outputs/packet_001 --boundary-model models/boundary --document-model models/document_classifier.pkl

# Retrieve evidence (never an answer)
python scripts/run_pipeline.py --query "What is the invoice total?" --processed-dir outputs/packet_001 --top-k 5

# Build a portable dense-index artifact from processed chunks
python scripts/build_index.py --input outputs/packet_001 --output outputs/index

# Generate truth-preserving benchmark/report placeholders until labels exist
python scripts/run_benchmark.py
python scripts/error_analysis.py
python scripts/generate_report.py
```

## API and Docker

```bash
uvicorn src.api:create_app --factory --host 0.0.0.0 --port 8000
docker build -t document-packet-intelligence .
docker run -p 8000:8000 document-packet-intelligence
```

Endpoints are `POST /process` (PDF upload), `POST /index`, `POST /retrieve`, and `GET /health`. The API is intentionally thin; `src/pipeline.py` owns orchestration.

## Outputs

A processed packet contains `pages.json`, `boundary_features.json`, `stage1.json`, `structured_documents.json`, `chunks.json`, rendered pages, and an index. These artifacts preserve page and document IDs end-to-end.

`scripts/run_benchmark.py` writes `outputs/benchmarks/{stage1,stage2,stage3}_results.{csv,json}`, `summary.{csv,json}`, and `resource_report.{csv,json}`. Values are `null` with an explicit pending state until real held-out labels/judgments are supplied; no metrics are fabricated.

## Evaluation protocol

- Stage 1: boundary precision/recall/F1, pairwise grouping accuracy, class accuracy/macro-F1, latency, memory and model size.
- Stage 2: extraction coverage, heading/table precision/recall when annotations exist, page-reference correctness, chunk statistics, OCR rate, and latency.
- Stage 3: Recall@1/3/5, Precision@1/5, MRR, nDCG, index time and query latency from held-out relevance judgments.

Compare text-only, text+visual, heuristic weighted fusion and trained gradient boosting for boundaries; TF-IDF versus an embedding-based implementation for types; and dense, BM25, fixed RRF, query-aware RRF and reranked RRF for retrieval. Select based on measured validation results rather than assumed superiority.

## Limitations and next steps

The checked-in environment contains no DocSplit data, trained artifacts, or synthetic packet, so trained-model metrics and end-to-end PDF evidence examples are pending. The fallback encoder is intended for reliable smoke tests, not as a claim of semantic-model quality. The next measured experiments should add a BGE/E5 encoder, benchmark CLIP against basic visual features, validate threshold calibration, introduce dataset-specific label adaptation, and evaluate optional cross-encoder reranking.

## Reproducibility

The configuration is centralized in [`config/default.yaml`](config/default.yaml). Cache inputs, model metadata, random seed, feature order, and benchmark artifacts. Do not compare experiments across different data splits, page render DPIs, or label schemas.

Rendered visual features are capped at `runtime.max_render_pixels` (30 million by default), so unusually large PDF pages do not exhaust RAM during training. Lower this limit on memory-constrained machines.

### Using trained models with the API

After training, set `boundary.model_path: models/boundary` and (if trained) `classification.model_path: models/document_classifier.pkl` in `config/default.yaml`, then restart Uvicorn. The `/process` response reports `model_status`; do not treat a `weighted_fusion_fallback` result as a trained packet-splitting result.
