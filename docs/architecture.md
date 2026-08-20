# System Architecture

## End-to-end pipeline

```mermaid
flowchart TB
    subgraph INGEST["Ingestion"]
        PDF[/"Packet PDF<br/>(N pages, M documents)"/]
        PARSE["PDFParser · PyMuPDF<br/>gap-aware span join, line breaks<br/>blocks · bboxes · fonts"]
        OCRQ{"native text<br/>sufficient?"}
        OCR["Tesseract OCR<br/>(optional, off by default)"]
        RENDER["Page renderer<br/>150 dpi PNG"]
        PDF --> PARSE --> OCRQ
        OCRQ -- no --> OCR --> RENDER
        OCRQ -- yes --> RENDER
    end

    subgraph S1["Stage 1 · Packet Intelligence"]
        FEAT["Pairwise features (14)<br/>text · visual · layout · heuristics"]
        BM["HistGradientBoosting<br/>isotonic-calibrated · 0.5 MB"]
        OVR["page-number-reset override"]
        GRP["Grouping<br/>expected-count from calibrated p"]
        CLS["Hybrid classifier<br/>TF-IDF+LR (16 RVL-CDIP classes)<br/>+ lexicon extension"]
        ABS{"conf ≥ 0.35?"}
        FEAT --> BM --> OVR --> GRP --> CLS --> ABS
    end

    subgraph S2["Stage 2 · Document Structuring"]
        BOIL["Boilerplate detection<br/>margin + repetition"]
        LAY{"layout metadata<br/>available?"}
        TXT["Text-only fallback<br/>synthesised blocks"]
        HEAD["Heading detection<br/>font rank · numbering depth"]
        ELEM["Element classification<br/>list · caption · paragraph"]
        TAB["Table extraction<br/>ruled + borderless (cropped)"]
        TREE["Section tree<br/>parent · breadcrumb · page refs"]
        CHUNK["Structure-aware chunker<br/>tables atomic · breadcrumb prefix"]
        BOIL --> LAY
        LAY -- no --> TXT --> HEAD
        LAY -- yes --> HEAD
        LAY -- yes --> TAB
        HEAD --> ELEM --> TREE
        TAB -- "regions suppress<br/>duplicate text" --> TREE
        TREE --> CHUNK
    end

    subgraph S3["Stage 3 · Evidence Retrieval"]
        Q[/"User query"/]
        BM25["BM25 index<br/>lexical"]
        DENSE["Dense index<br/>TF-IDF+SVD (default)<br/>or bge-small"]
        RRF["Reciprocal rank fusion<br/>k=60"]
        RR["Feature reranker<br/>coverage · phrase · exactness"]
        MMR["MMR diversity<br/>lambda = 0.7"]
        CONF["Confidence + context assembly<br/>dedup · token budget · citations"]
        Q --> BM25 & DENSE --> RRF --> RR --> MMR --> CONF
    end

    OUT[/"Evidence + doc_id + page_ref<br/>+ breadcrumb + confidence"/]

    RENDER --> FEAT
    ABS -- "typed, or 'unknown'<br/>(structuring proceeds either way)" --> BOIL
    CHUNK --> BM25
    CHUNK --> DENSE
    CONF --> OUT

    API(["FastAPI · /process · /retrieve · /context"])
    API -.drives.-> PDF
    API -.serves.-> OUT

    classDef stage fill:#eef4ff,stroke:#4a6fa5,color:#1a2b45
    classDef io fill:#fff7e6,stroke:#c08a2e,color:#4a3510
    class INGEST,S1,S2,S3 stage
    class PDF,Q,OUT io
```

## Artifacts written per packet

```mermaid
flowchart LR
    P["pipeline.process()"] --> A["pages.json<br/>raw page representations"]
    P --> B["boundary_features.json<br/>14 features per page pair"]
    P --> C["stage1.json<br/>document groups + confidences"]
    P --> D["structured_documents.json<br/>section tree + elements"]
    P --> E["markdown/*.md<br/>human-readable view"]
    P --> F["chunks.json<br/>retrieval units"]
    P --> G["index/dense_index.json<br/>+ svd_encoder.pkl"]
```

## Model inventory

| Component | Model | Trained on | Size |
|---|---|---|---|
| Boundary classifier | HistGradientBoosting + isotonic calibration | OpenPSS SHORT, 15,906 page pairs | 0.5 MB |
| Boundary text signal | TF-IDF vectoriser | same | 2.0 MB |
| Document classifier | TF-IDF + Logistic Regression | RVL-CDIP OCR, 8,887 docs / 16 classes | 3.3 MB |
| Extension classes | Weighted lexicon (no training data available) | — | < 1 KB |
| Dense retrieval | TF-IDF + TruncatedSVD (fit per corpus) | corpus at index time | ~112 MB peak RAM |
| Dense retrieval (optional) | BAAI/bge-small-en-v1.5 | pretrained | ~130 MB |

**Total committed model footprint: 5.8 MB.** No component is trained on the DocSplit benchmark.

## Data flow contracts

| Boundary | Contract |
|---|---|
| Ingestion → Stage 1 | `list[PageRepresentation]` — text, blocks, bboxes, fonts, render path |
| Stage 1 → Stage 2 | `list[DocumentGroup]` — pages, type, confidence, source, alternatives |
| Stage 2 → Stage 3 | `list[Chunk]` — text, page_refs, breadcrumb, element_types, token_count |
| Stage 3 → caller | `list[EvidenceResult]` — evidence, doc_id, page_ref, confidence, scores |

Each stage consumes only the dataclass contract above it, so a stage can be replaced without
touching its neighbours — the dense encoder swap (hashed → SVD → transformer) exercises exactly
this property and required no change to Stage 1 or Stage 2.
