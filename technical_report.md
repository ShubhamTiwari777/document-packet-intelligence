# Document Packet Intelligence & Evidence Retrieval — Technical Report

All figures in this report are measured by the scripts in `scripts/` and written to
`outputs/benchmarks/`. Nothing is estimated or copied from published baselines. Where a number
could not be measured honestly it is reported as **not measured** rather than filled in.

---

## 1. Problem understanding

A packet is a single PDF holding several unrelated documents concatenated together. Three
questions must be answered in order, each depending on the previous answer being right:

1. **Where does one document end and the next begin?** A decision over the *N−1 adjacent page
   pairs*, not over pages — which makes grouping a deterministic consequence of boundary
   decisions rather than a second model that can disagree with the first.
2. **What is each document, and how sure are we?** Type prediction needs an abstention path: a
   confident wrong label is worse than an explicit `unknown`.
3. **What is inside it, and where exactly?** Retrieval must cite a document and a page, so
   structure and page provenance are the substrate the answer is built from, not presentation.

The solution is three stages with dataclass contracts between them (`PageRepresentation →
DocumentGroup → Chunk → EvidenceResult`). The contracts are the design: each stage is replaceable
without touching its neighbours, which is what let the Stage 3 dense encoder be swapped three
times with no change to Stages 1 or 2.

### Dataset decision

The brief named DocSplit v2, but that dataset is an evaluation benchmark with no public
train/validation split. Following the clarification, **OpenPSS** (`nutrientdocs/openpss-mirror`,
SHORT config) is used for boundary detection: Dutch FOIA page streams with per-page
`label = 1` marking a document start.

`nutrientdocs/doc-split-benchmark` did prove publicly readable (test split only, as the guidance
said). It is therefore **evaluated on but never trained on** — that is what a benchmark is for,
and fitting to it is what the brief prohibits. Scoring against it turned out to be the single most
informative measurement in this project (§5.1).

OpenPSS carries **no document-type labels**, so it cannot train a type classifier. I added
**RVL-CDIP OCR text** (`albertklorer/rvl_cdip_ocr`) — the standard 16-class document-type
taxonomy, distributed with OCR words already extracted, which matters because no OCR binary is
installed in the target environment. Neither dataset is DocSplit-specific and no
benchmark-trained checkpoint is used.

Both are accessed through the HuggingFace `datasets-server` REST API rather than `datasets` +
`pyarrow`, because this environment runs Python 3.14 where `pyarrow` has no wheel and fails to
build from source without a CMake toolchain.

---

## 2. System architecture

Full diagrams: **[docs/architecture.md](docs/architecture.md)**.

```
PDF → PDFParser (PyMuPDF) → [Stage 1] pairwise features → calibrated GBM → grouping → classifier
                          → [Stage 2] boilerplate → headings → elements/tables → section tree → chunks
                          → [Stage 3] BM25 + dense → RRF → reranker → evidence + page + confidence
                          → FastAPI /process, /retrieve
```

**Stage 1** computes 21 features per adjacent page pair (text delta, visual delta, page-number
reset/continuation, header/footer similarity, font and layout deltas), scores them with a
calibrated gradient-boosted tree, and splits where `p ≥ 0.633`. Grouping is deterministic from
those decisions. Types come from a hybrid classifier described in §3.2.

**Stage 2** detects running headers/footers first, then classifies each remaining block as
heading, list, caption or paragraph, extracts tables separately, and assembles a section tree
with parent links, breadcrumbs and per-element page numbers. Chunks are packed from whole
elements.

**Stage 3** searches BM25 and a dense index independently, fuses with reciprocal rank fusion,
reranks the shortlist on deterministic match features, and normalises a confidence.

---

## 3. Design decisions

### 3.1 Boundary detection as calibrated pairwise classification

Alternatives considered: a sequence model (CRF/BiLSTM) and a similarity-threshold heuristic.
Per-pair classification was chosen because it needs no sequence-labelling infrastructure, trains
in seconds on CPU, and yields the per-boundary probability the brief asks for. Isotonic
calibration was added because raw GBM probabilities on an imbalanced split compress toward zero,
making any fixed threshold unreliable.

One deliberately non-learned addition: **a printed page number resetting is a near-conclusive
boundary cue and is domain-invariant**, unlike the learned score which reflects whichever corpus
the model saw. It therefore overrides a muted learned probability. This single rule is what took
the 9-page verification packet from one merged blob to four correct documents.

### 3.2 A hybrid classifier with explicit abstention

RVL-CDIP has no `passport` or `bank_statement` class, and no public labelled corpus for them was
available. Rather than pretend, the classifier is two-tier: the trained model handles its 16
classes, and a weighted lexicon covers types with no training data. Reconciliation treats the
trained model's confidence as **not comparable** for an out-of-taxonomy type — for a passport the
model must assign probability mass to some unrelated class, so it is confidently wrong by
construction. A regression test caught this: a stub model returning `letter @ 0.90` beat a
correct `passport @ 0.82` under naive numeric comparison.

Confidence is a normalised distribution gated by absolute evidence, not "fraction of this class's
keyword list matched" — the latter cannot express ambiguity. `min_confidence` is enforced, so
weak evidence yields `unknown`: text reading only `"Total amount 500"` returns **unknown @ 0.111**
rather than **invoice @ 0.2**.

### 3.3 Structure is layout-driven, with a text-only fallback

Every structural signal — heading size, boilerplate position, table regions — derives from
bounding boxes and font sizes, which OCR-only sources lack: the engine originally produced *zero*
structure on 2,500 real OpenPSS pages. A text-only path reconstructs pseudo-blocks from blank-line
grouping and infers headings from typographic convention. Both paths converge downstream.

### 3.4 Retrieval: measure the encoder rather than assume it

The dense index originally used a hashed bag-of-words whose slots are `int.from_bytes(token) %
dims` — related words land in unrelated slots. It scored **R@1 = 0.086**, and fusing it with BM25
produced **R@1 = 0.257 against BM25-alone's 0.714**: the shipped hybrid was worse than deleting
half of it. Encoders were made pluggable and all three measured (§6.3).

---

## 4. Experimental methodology

| Stage | Evaluation set | Labels from |
|---|---|---|
| 1 — boundaries | OpenPSS SHORT test, 108 streams / 11,354 page pairs | dataset ground truth |
| 1 — boundaries (target) | DocSplit benchmark `our200`, 200 streams / 694 pairs | dataset ground truth (evaluation only) |
| 1 — classification | RVL-CDIP held-out 20%, 2,222 documents | dataset ground truth |
| 2 — structure | two authored fixtures, 12 pages | authored annotations |
| 2 — coverage | OpenPSS SHORT test, 2,500 pages | none (label-free) |
| 3 — retrieval | 35 queries over a 515-chunk corpus | authored judgments |

**Train/test discipline.** Stage 1 boundary training uses 95 OpenPSS streams; evaluation uses the
full 108-stream test split, with the decision threshold fitted on 54 tuning streams and scored on
54 held-out streams (split by stream, since pairs within a stream are not independent). The
DocSplit benchmark is scored but never trained on. The classifier holds out 20% by seeded shuffle.

**Why authored fixtures for Stage 2.** No public page-stream dataset ships heading/table/caption
annotations. I generate the fixtures programmatically (`scripts/generate_sample_packet.py`,
`scripts/generate_benchmark_report.py`) and emit ground truth alongside the PDF, so the labels are
authored rather than guessed. **This measures conformance to constructs I chose, not
generalisation** — which is precisely why the label-free OpenPSS coverage run exists beside it.

**Retrieval judgments** are resolved by answer-string containment rather than pinned chunk ids,
so they survive re-chunking. The corpus is padded with real OpenPSS chunks as distractors
(515 total, of which 38 come from the fixtures), so retrieval is not a 30-way choice.

**A metrics honesty fix.** Precision/recall/F1 now return `null` when a category has zero expected
and zero predicted instances. The sample packet contains no captions; scoring that `0.0` reported
correct behaviour as total failure.

---

## 5. Benchmark results

### 5.1 Stage 1 — boundary detection

Full OpenPSS SHORT test split: **108 streams, 11,354 adjacent pairs, 1,296 true boundaries**
(11.4% base rate). Threshold is fitted on 54 tuning streams and scored on 54 held-out streams.

| Metric | Value |
|---|---|
| Precision | 0.289 |
| Recall | 0.551 |
| **F1 (held-out threshold)** | **0.379** |
| F1 (threshold tuned on the scored data) | 0.406 — biased, shown for contrast |
| Inference | 96 ms per 8-page packet |
| Model size | 0.5 MB (+ 2.0 MB vectoriser) |

Verification packet (4 documents / 9 pages): **4/4 documents split correctly**.

#### How this number was corrected

The first version of this evaluation was wrong twice: the threshold was selected on the data it
was then scored on, and the test set held only 12 streams — splitting it in half gave F1 **0.21**
on one half and **0.44** on the other, for the same model. Enlarging to the full split and
separating threshold selection from scoring fixes both; the optimism gap collapses from 0.167 to
0.027, and the honest figure lands close to the originally reported 0.377, which was therefore
roughly right by luck rather than by method.

| Test set | Honest held-out F1 | Threshold-on-test F1 | Gap |
|---|---|---|---|
| 12 streams | 0.210 | 0.377 | 0.167 |
| 108 streams | **0.379** | 0.406 | 0.027 |

#### Four attempts to raise F1 — all negative

The training matrix showed **8 of 14 features constant** across all 15,905 rows, including
`header_similarity` and `footer_similarity`, so the model effectively learned from six. Four
hypotheses were tested against the corrected protocol:

| Variant | Live features | Precision | Recall | F1 | Lift |
|---|---|---|---|---|---|
| **Baseline (retained)** | 6 / 14 | 0.289 | 0.551 | **0.379** | **+0.191** |
| Pseudo-blocks reconstructed from OCR text | 10 / 14 | 0.287 | 0.478 | 0.359 | +0.171 |
| Natural class prior (no `balanced` weighting) | 6 / 14 | 0.301 | 0.486 | 0.371 | +0.183 |
| + 7 sentence-flow features | 13 / 21 | 0.311 | 0.458 | 0.371 | +0.183 |
| Synthetic RVL-CDIP packets *(different corpus)* | 18 / 21 | 0.468 | 0.578 | *0.517* | *+0.031* |

The first three moved **along one precision/recall curve** rather than shifting it — every variant
traded precision up against recall down and landed on the same F1. The threshold already exposes
that trade-off, so recalibration and these feature families cannot help; the limit is the model's
ability to separate boundary from non-boundary pairs at all.

The fourth is the instructive one. RVL-CDIP ships word-level bounding boxes, so synthetic packets
built from it carry real geometry and lifted live features from 6/14 to 18/21 — apparently
producing **F1 0.517 against 0.379**. It is not a gain: at that corpus's 32.1% boundary density a
trivial always-boundary classifier already scores 0.486, so the lift is +0.031 against the
baseline's +0.191. The cause is the corpus, not the model — **RVL-CDIP rows are independent
single-page documents** (5,449 pages, 5,449 distinct filenames), so grouping same-class pages into
a "document" fabricates continuity that does not exist. DocSplit avoids this by building on
RVL-CDIP-N-MP, the multi-page variant, which is public (`jordyvl/rvl_cdip_n_mp`) and is taken up
in §8. `boundary_lift` entered the metric set here.

Three latent defects were fixed while running these: models now record the feature construction
they were trained with (`synthetic_blocks`, `class_weight`) and evaluation reads it back; and a
saved model scores with the feature list it was **trained** on rather than whatever
`FEATURE_NAMES` currently holds — without which adding the seven features would have silently
broken every previously saved model.

#### Evaluation on the actual DocSplit benchmark — the system does not transfer

`nutrientdocs/doc-split-benchmark` turns out to be publicly readable (config `our200`) and, as the
guidance stated, ships **a test split only**. It is therefore used strictly for evaluation and
never for training — scoring on it is what a benchmark is for; fitting to it is what the brief
prohibits. Its schema mirrors OpenPSS with renamed fields, so it normalises into the same
manifest structure.

| Model (both trained elsewhere) | F1 | Precision | Recall | Trivial F1 | **Lift** |
|---|---|---|---|---|---|
| OpenPSS-trained (shipped) | 0.823 | 0.703 | 0.992 | 0.815 | **+0.008** |
| RVL-CDIP-trained | 0.815 | 0.688 | 1.000 | 0.815 | **+0.000** |

**F1 0.82 on the target benchmark is not a good result — it is the trivial result.** Recall 0.99
against precision ≈ the base rate means the model marks nearly every pair a boundary. Even with a
perfect oracle threshold the ceiling is 0.854, a lift of +0.039. This is the clearest possible
demonstration of why `boundary_lift` was added: reported as raw F1, 0.823 would have looked like
the strongest number in this report.

The cause is a **regime inversion**, not a tuning problem:

| Corpus | Streams | Median pages/stream | Boundary rate | Minority class |
|---|---|---|---|---|
| OpenPSS (trained on) | 108 | 21 | 11.4% | boundary |
| DocSplit (target) | 200 | 4 | 72.3% | **continuation** |

OpenPSS packets are long streams where boundaries are rare, so the model learned that a boundary
requires strong evidence. DocSplit packets are short — a median of four pages, mostly one-page
documents — so the rare, informative event is a *continuation*. The model was trained to detect
the opposite minority class, which no threshold can repair.

The remedy is a training corpus in the target regime: short packets built from multi-page
documents, so both boundaries and continuations are genuinely represented. `jordyvl/rvl_cdip_n_mp`
(the multi-page RVL-CDIP variant DocSplit itself is assembled from) is public and is the obvious
candidate; it is not served by the datasets-server auto-converter, so it needs direct download.
That work is not done here, and the honest position is that **Stage 1 is tuned for long-stream
segmentation and underperforms on short packets.**

#### Summary of Stage 1 boundary detection

| Corpus | Role | F1 | Trivial F1 | **Lift** |
|---|---|---|---|---|
| OpenPSS SHORT test (108 streams) | held-out, same regime as training | 0.379 | 0.188 | **+0.191** |
| DocSplit `our200` (200 streams) | the assignment's target task | 0.823 | 0.815 | **+0.008** |

The system segments long page streams meaningfully and short packets not at all. Both numbers are
stated because reporting only the first would overstate the system and reporting only the second
(0.823) would flatter it.

### 5.2 Stage 1 — document classification

RVL-CDIP held-out, 2,222 documents, 16 classes.

| Metric | Value |
|---|---|
| Accuracy | 0.807 |
| Macro F1 | 0.786 |
| invoice F1 | 0.885 |
| resume F1 | 0.952 |
| Model size | 3.3 MB |

The two classes most relevant to document packets score highest. A feature-size sweep found
20k features **beat** 200k (0.807 vs 0.803 accuracy, 0.786 vs 0.775 macro-F1) at **one tenth the
size** — 3.3 MB vs 33.3 MB. The larger space was mostly noisy OCR tokens.

`passport` and `bank_statement` are lexicon-scored and have **no held-out measurement** — no
labelled corpus for them was available. This is the largest unmeasured area in the system.

### 5.3 Stage 2 — structure

| Metric | 4-doc packet | Report fixture |
|---|---|---|
| Heading P / R / F1 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Table F1 | 1.00 | 1.00 |
| List F1 | 1.00 | 1.00 |
| Caption F1 | *not applicable* | 1.00 |
| Page-reference accuracy | 1.00 (6/6) | 1.00 (6/6) |
| Type-field accuracy | 1.00 (10/10) | *not applicable* |
| Throughput | 24.6 pages/s | 12.0 pages/s |

Label-free coverage on **2,500 real OpenPSS pages**, before and after the text-only fallback:

| Signal | Before | After |
|---|---|---|
| Sections | 12 | 2,575 |
| Elements | 0 | 31,039 |
| Mean chunk tokens | 256.6 (fixed window) | 141.3 (semantic) |
| Text retention | — | 97.7% |
| Throughput | — | 1,250 pages/s |

`single_page_chunk_ratio` moved from 1.00 to 0.655 here — an **improvement**, because the old
1.00 was false precision: every chunk claimed its document's first page regardless of origin.

### 5.4 Stage 3 — retrieval

35 queries, 515-chunk corpus with OpenPSS distractors.

| Configuration | R@1 | R@3 | R@5 | P@1 | MRR | nDCG | Mean lat. | p95 | Index |
|---|---|---|---|---|---|---|---|---|---|
| bm25_only | 0.714 | 0.829 | 0.857 | 0.714 | 0.774 | 0.776 | 45.5 ms | 52.0 | 1.0 s |
| dense_hashed_only | 0.086 | 0.200 | 0.229 | 0.086 | 0.134 | 0.152 | 44.9 ms | 50.9 | 1.1 s |
| **rrf_hashed** *(old default)* | **0.257** | 0.457 | 0.743 | 0.257 | 0.407 | 0.477 | 47.9 ms | 57.0 | 1.1 s |
| dense_svd_only | 0.514 | 0.800 | 0.829 | 0.514 | 0.645 | 0.673 | 57.7 ms | 66.2 | 8.4 s |
| rrf_svd | 0.600 | 0.800 | 0.857 | 0.600 | 0.713 | 0.731 | 69.5 ms | 79.4 | 8.0 s |
| rrf_svd_query_aware | 0.657 | 0.800 | 0.857 | 0.657 | 0.743 | 0.753 | 54.8 ms | 65.0 | 8.9 s |
| **rrf_svd_reranked** *(default)* | **0.771** | 0.857 | 0.886 | 0.771 | 0.821 | 0.819 | 57.9 ms | 64.8 | 7.5 s |
| dense_bge_only | 0.657 | 0.857 | 0.886 | 0.657 | 0.755 | 0.769 | 85.9 ms | 93.0 | 70.1 s |
| rrf_bge | 0.800 | 0.914 | 0.971 | 0.800 | 0.867 | 0.874 | 95.4 ms | 109.8 | 67.2 s |
| rrf_bge_reranked | **0.886** | 0.971 | **1.000** | 0.886 | 0.929 | 0.928 | 118.0 ms | 129.0 | 67.0 s |

Two findings matter more than the best row. First, **the previously shipped default was worse
than its own lexical half** (0.257 vs 0.714) — a broken component silently degraded a working one,
which no aggregate score would have revealed without ablation. Second, the reranker adds
**+0.171 R@1** on SVD for ~0 ms and no dependency, making it the best value in the table.

**Default: `svd` + reranker.** bge wins by +0.115 R@1 but needs ~1 GB of torch, which required two
install attempts and a DLL failure to get working in this environment. A submission that does not
install is worth less than 0.115 R@1. The transformer is one config line away and its numbers are
recorded above and in `requirements.txt`.

### 5.5 Resource summary

| Resource | Value |
|---|---|
| Committed model footprint | **5.8 MB** total |
| Process RSS (full pipeline) | 153 MB |
| Index peak RAM — SVD | 112 MB |
| Index peak RAM — bge | 13.6 MB |
| Stage 1 inference | 96 ms / 8-page packet |
| Stage 2 structuring | 12–24 pages/s native, 1,250 pages/s text-only |
| Stage 3 query latency | 58 ms mean, 65 ms p95 |
| Retrieval through API | 6–34 ms (cached retriever) |

Everything runs CPU-only. No GPU, no external API, no network call at inference.

---

## 6. Error analysis

### 6.1 Boundary detection

Dominant failure is **recall on visually similar adjacent documents** — two same-template forms
concatenated share fonts, margins and header text, so every layout feature reads "continuation".
Precision 0.289 means roughly two false splits per true one; the threshold favours recall
because an over-split document is recoverable downstream whereas a merged one loses a document
entirely. The larger failure is structural rather than per-pair: on short packets (§5.1) the model
is at the trivial baseline, because it was trained where boundaries are rare and the target has
them common.

### 6.2 Classification

`passport`/`bank_statement` rest on lexicon evidence with no held-out measurement. A document
using unusual vocabulary for those types would fall below `min_confidence` and return `unknown` —
the intended degradation, but still a miss.

### 6.3 Retrieval

From `outputs/error_analysis/stage3_errors.json`, SVD leaves 8 of 35 queries imperfect, bge 4.
The pattern is consistent — **low lexical overlap paraphrases**:

| Query | SVD | bge | Cause |
|---|---|---|---|
| "How can I contact the applicant by email?" | miss | rank 5 | target says `Contact: arjun.mehta@...`; the word "email" never appears |
| "What programming languages does the candidate know?" | miss | hit | target lists `Python, SQL, C++` without the word "languages" |
| "What is the total amount due on the invoice?" | rank 2 | rank 2 | competing Subtotal/GST chunk |
| "What is the passport number?" | rank 4 | rank 2 | identifier appears in several documents |

**False positives** are instructive: for the email query SVD returned a Dutch OpenPSS distractor —
LSA fitted on a mixed-language corpus produced a latent dimension grouping unrelated administrative
text. That is the concrete cost of a corpus-fitted encoder versus a pretrained one.

### 6.4 Resolved: removed dead code

`src/stage2/llm_fallback.py` defined an LLM fallback contract nothing called. Deleted rather than
wired up: the parser exposes no structure-confidence signal to trigger it, and PDF structure is
deterministic enough that an LLM would add cost, latency and nondeterminism without addressing any
measured failure.

---

## 7. Trade-offs consciously made

| Trade-off | Decision | Why |
|---|---|---|
| Accuracy vs reproducibility | SVD default over bge (−0.115 R@1) | torch install failed twice here; a submission that won't install is worth less |
| Recall vs precision on boundaries | favour recall | over-split is recoverable, merged documents are not |
| Simplicity vs coverage | rule-based structure, no LLM | PDF structure is deterministic; an LLM adds cost, latency and nondeterminism for no gain |
| Model size vs accuracy | 20k TF-IDF features | measured *better* accuracy at 1/10th size |
| Confidence honesty vs looking good | real probabilities + abstention | invoice confidence dropped 0.8 → 0.49, but 0.8 was meaningless |

---

## 8. Future improvements (in priority order)

1. **Retrain in the target regime.** The DocSplit evaluation (§5.1) shows the shipped model is at
   the trivial baseline on short packets because it was trained where boundaries are rare and the
   target has them common. Building packets from `jordyvl/rvl_cdip_n_mp` — multi-page documents,
   so continuations are real rather than fabricated — directly addresses this and is the single
   highest-value change available. It supersedes everything else on this list.
2. **A labelled corpus for `passport`/`bank_statement`.** The largest unmeasured area; every claim
   about those classes currently rests on a 4-document fixture.
3. **Boundary detection needs a different model class, not more hand-built features.** Three
   feature/calibration changes were measured and all landed on the same precision/recall curve
   (§5.1), which says the ceiling is representational. Two directions follow from that, in order
   of expected value:
   * **Sequence modelling.** Every current feature compares page *i* with page *i+1* in isolation,
     while document boundaries are a sequence-labelling problem — documents have length priors and
     boundaries do not cluster. Adding neighbouring-pair context and decoding with a length prior
     is the standard remedy and is untried here.
   * **Learned page-pair representations.** `text_delta` is TF-IDF cosine, which fires on shared
     vocabulary; consecutive pages of one report and two different reports on one topic look alike
     to it. A cross-encoder over page-pair text would model *continuation* rather than *overlap*.
     torch and sentence-transformers are already installed for Stage 3, so this is reachable.
4. **Transformer embeddings as default**, once dependency installation is reliable — worth
   +0.115 R@1 and a perfect R@5 on the current evaluation set.
5. **Grow the retrieval evaluation set.** 35 queries gives ±0.07 confidence intervals; conclusions
   about ~0.05 differences are not statistically safe.
6. **Persist the SVD index** rather than refitting per corpus (8 s today).
7. **Recover the visual signal.** `visual_delta` is a 16-bin grey histogram over 224×224
   thumbnails and was constant on the RVL-CDIP packets — a downsampled ink-density grid would
   capture layout rather than tone.

---

## Technical questionnaire

**1. Problem understanding.** See §1. Boundaries are decisions over N−1 page pairs; grouping is
deterministic from them; structure and page provenance are the substrate retrieval cites.

**2. Three most important decisions.** (a) Calibrated pairwise boundary classification with a
domain-invariant page-number-reset override — chosen over a sequence model for CPU cost and
per-boundary probabilities. (b) Hybrid classifier with explicit abstention over a single trained
model, because no training data exists for two required classes and a confident wrong label is
worse than `unknown`. (c) Pluggable retrieval encoders measured by ablation, which is the only
reason the harmful default (§5.4) was found.

**3. Technology selection.** PyMuPDF (fonts + bboxes), pdfplumber (dual ruled/borderless table
strategies), scikit-learn (HistGradientBoosting, TF-IDF, LogisticRegression, TruncatedSVD — all
CPU-only), BM25 implemented directly, FastAPI. Rejected: `datasets`/`pyarrow` (no Python 3.14
wheel), Camelot (heavier than pdfplumber for equivalent output), sentence-transformers (measured
better but ~1 GB — kept as a documented option).

**4. Evaluation methodology.** See §4. Held-out splits where dataset labels exist; authored
fixtures where no public annotations exist, clearly separated from label-free coverage on real
data; retrieval judgments anchored to answer strings; ablations rather than single aggregate
scores.

**5. Failure analysis.** See §6. The biggest limitation by far: on the DocSplit benchmark the
system scores F1 0.823 against a trivial baseline of 0.815 — a lift of +0.008, i.e. no better than
marking every pair a boundary. It was trained on long streams where boundaries are rare (11%) and
the target has them common (72%), inverting which class is informative. Secondary limitations:
`passport`/`bank_statement` have no held-out measurement, and the SVD encoder misses low-overlap
paraphrases.

**6. Resource & performance.** See §5.5 — 5.8 MB of models, 153 MB RSS, CPU-only. To shrink
further: drop SVD for BM25 + reranker (1 s index, 9 MB peak). To scale up: the dense scan is O(N)
per query, fine at 515 chunks and not at 10⁶ — it needs an ANN index.

**7. Trade-offs.** See §7.

**8. Two more weeks.** §8 item 1 first and alone if necessary: rebuild training packets from
`jordyvl/rvl_cdip_n_mp` so the model is trained in the target regime, since every other Stage 1
improvement is worth less than closing a +0.008 lift. Then labelling the extension classes, and
sequence modelling over page pairs.

**9. AI usage declaration.**
> **This section must be reviewed and edited by the submitting engineer before submission — only
> you can accurately describe your own process.**
>
> Factual record of this repository's development: an AI coding assistant (Claude, via Claude Code)
> was used interactively throughout. It was used to implement modules, diagnose defects, run
> experiments and draft this report. The engineering decisions recorded above were made in that
> dialogue and were repeatedly overridden by measurement rather than accepted as proposed —
> several assistant-proposed approaches were rejected after benchmarking (for example, the
> transformer encoder was measured as superior yet not adopted as the default, and an initial
> extension-class reconciliation rule was rewritten after a regression test disproved it).
> All numbers in this report were produced by executing the scripts in `scripts/`, not generated
> by a model.

---

## Reproducing every number

Full command sequence in [README.md](README.md). In short: fetch OpenPSS (train + full test) and
RVL-CDIP text, train the boundary and classifier models, generate the two annotated fixtures, then
run `evaluate_openpss_boundary.py` (against both the OpenPSS and DocSplit manifests),
`evaluate_stage2.py` and `evaluate_stage3.py`. Every table above is written to
`outputs/benchmarks/`. The negative results reproduce via `--no-synthetic-blocks`,
`--class-weight none`, and `train_rvlcdip_boundary.py`. `pytest tests/ -q` runs 35 tests.
