# Document Packet Intelligence & Evidence Retrieval

## What problem does this solve?

Sometimes one PDF contains several completely different documents scanned together — an invoice,
then a resume, then a passport scan, all in one file. A computer sees one 9-page PDF. A person sees
three separate documents.

This project does three things with such a file:

1. **Splits it** into the separate documents it actually contains, and says what type each one is
2. **Reads the structure** of each document — headings, tables, lists, which page everything is on
3. **Finds evidence** for a question, and tells you which document and page it came from

It does **not** write answers for you. Ask *"What is the closing balance?"* and it returns the exact
text that answers it, plus the document and page number — so you can verify it yourself. That was a
deliberate requirement, not a limitation.

## See it work

The quickest way is the browser interface:

```bash
pip install -r requirements.txt
uvicorn src.api:create_app --factory --port 8000
```

Open <http://localhost:8000>, then click **"Try the bundled 9-page sample"** — or drag in your own
PDF. You get back:

- each document as **its own downloadable PDF**, plus a zip of the whole set
- what type each one is, and how confident the model is
- a strip showing the score at every page seam, so you can see *why* it split where it did
- a search box that answers questions with the document and page the answer came from

The bundled sample is a 9-page PDF holding four documents, and the system splits it into invoice
(1–3), resume (4–5), passport (6–7) and bank statement (8–9).

If you supply ground truth — a JSON list of the real documents — it also reports **measured
accuracy, precision, recall and F1** for that packet. Without it, the interface says so plainly
rather than presenting a confidence score as if it were correctness.

Prefer the command line?

```bash
python scripts/generate_sample_packet.py
python scripts/run_pipeline.py --input data/samples/sample_packet.pdf --output outputs/sample
python scripts/run_pipeline.py --query "What is the closing balance?" --processed-dir outputs/sample --top-k 5
```

## How it works, in plain terms

```
PDF file
   ↓  read every page (text, position of each word, font sizes)
STAGE 1 — split into documents, and label each one
   ↓  for every pair of neighbouring pages, decide: same document, or new one?
STAGE 2 — understand each document
   ↓  find headings, tables, lists; remember which page everything came from
STAGE 3 — answer questions with evidence
   ↓  search two different ways, combine, re-rank, return with page citations
```

**Stage 1** looks at each pair of neighbouring pages and asks "do these belong together?" It uses 21
clues — does the text suddenly change topic, does the page number restart, do the headers differ,
does the layout change. A trained model turns those clues into a probability.

**Stage 2** turns each document into structured data. It first removes repeated headers and footers
(so a company letterhead on every page doesn't get mistaken for a real heading), then finds real
headings, tables and lists, keeping track of the page each piece came from.

**Stage 3** searches in two different ways at once — keyword matching (good for exact things like
invoice numbers) and meaning-based matching (good for questions worded differently from the
document). It merges both result lists, re-ranks them, and returns the best evidence with citations.

Everything runs on a normal CPU. No GPU, no paid API, no internet needed once set up. All the
trained models together are only **10.5 MB**.

## Results, and how to read them

| What | Measured on | Result |
|---|---|---|
| Splitting documents | TABME++ held-out **test**, 501 English packets | page grouping accuracy **0.97** |
| Identifying document type | 5,447 held-out documents, 16 types | **83.9%** accuracy |
| Finding the right evidence | 35 questions, 515 text chunks | correct answer ranked #1 **77%** of the time |
| Structure extraction | annotated test files | headings, tables, lists all correct |

**The splitter is language-specific, and that matters more here than any other single fact.**
The shipped model is trained on English. On 501 held-out English packets it scores **0.968** page
grouping and beats the lazy *"every page is its own document"* baseline by **+0.222** — the first
result in this project to clear that baseline by a margin that is not arguable.

Point the same model at Dutch and it falls to 0.606, because its vocabulary recognises barely a
quarter of the words. A Dutch-trained model is included: set `boundary.model_path` to
`models/boundary_shortpackets` and `boundary.decision` to `expected_count`, and it scores **0.761**
on the Dutch OpenPSS set where the English model gets 0.606. Neither is better in general — the
choice is a language choice.

Every number above was selected on a validation split and then measured **once** on a held-out test
split. That protocol earned its keep twice: two rules that looked like improvements on validation
did not survive the test set, and both are written up in the report rather than quietly dropped.

Full analysis — including the six experiments that failed first, and the discovery that the earlier
performance ceiling was a language artefact rather than a real limit — is in section 5.1 of the
technical report.

Full details: **[technical_report.pdf](technical_report.pdf)** ·
Diagram: **[docs/architecture.pdf](docs/architecture.pdf)**

## Testing on real PDFs

The figures above come from public corpora of **thousands** of page pairs, and they are the numbers
that matter. But those corpora store each page as OCR text — in TABME++ every page is a single line
of space-joined words — so seven of the twenty-one features are degenerate there and the pipeline
never runs end to end on an actual PDF.

Four annotated PDFs ship with the repo for that. Run all of them at once:

```bash
python scripts/evaluate_pdf_suite.py
```

| Fixture | Pages | Docs | What it is for |
|---|---|---|---|
| `sample_packet.pdf` | 9 | 4 | the original smoke test — invoice, resume, passport, bank statement |
| `document_packet_test.pdf` | 9 | 4 | the same four types with different content, to check nothing is memorised |
| `test_case_2_mixed_packet.pdf` | 10 | 5 | adds a cover letter, and repeats one boilerplate line on **every** page, which raises text similarity across real boundaries |
| `stress_packet.pdf` | 13 | 7 | built to fail — see below |

**Pooled over all four: 41 pages, 20 documents.**

| | |
|---|---|
| Documents split at **exactly** the right pages | **20 / 20** |
| Document types identified correctly | **19 / 20** |
| Mean lift over the always-split baseline | **+0.407** |

The one miss is a cover letter returned as `unknown` rather than `letter` — it landed just under the
confidence floor and abstained instead of guessing, which is the intended behaviour.

Drop any PDF plus a matching `<name>_ground_truth.json` into `data/samples` and it joins the suite
automatically.

### The stress packet

`sample_packet.pdf` is scored perfectly by every version of this system, which makes it useless for
telling whether a change helped. `stress_packet.pdf` exists to be hard:

- an **invoice and a budget sit adjacent**, sharing a letterhead and both leading with a ruled money
  table, so layout similarity argues for merging two different documents
- a **one-page letter and a one-page memo sit back to back**, making two consecutive seams both real
  boundaries, which is where short-document splitting fails
- a **three-page passport** whose pages look nothing alike — fields, then a table, then prose —
  inviting a false split *inside* one document

It earned its place. It caught both failures that the corpus benchmarks could not see, and the
progression below is measured on it because it is the only fixture with room to improve.

### What this does not prove

Four packets is a small sample, and all four are **synthetically generated** — clean digital text,
tidy `Page N of M` footers, no OCR noise or skew. They are an end-to-end correctness check, not the
evidence base: the headline accuracy, precision and recall come from the held-out corpora above,
scored on 501 packets and 5,447 documents. Real scanned documents remain untested.

## How the splitter got from 0.22 to 0.97

Eight attempts, in order. The numbers below come from two different benchmarks in two different
languages, so **they are not one continuous series** — the honest comparison within each phase is
against that corpus's own *"every page is its own document"* baseline, shown in each caption.

**Phase 1 — measured on Dutch benchmarks.** Baseline to beat: **0.854**.

| # | What changed | Page grouping |
|---|---|---|
| 1 | Fixed threshold, fitted on a corpus with 11% boundaries, applied where 72% are | 0.216 |
| 2 | **Expected-count rule** — split at the *N* best seams, *N* = sum of probabilities | 0.615 |
| 3 | **Regime-matched training** — rebuild training packets to match deployment shape | 0.701 |
| 4 | Cross-encoder over the page seam, 1,600 training pairs | 0.636 |
| 5 | Cross-encoder, same everything, **5.3× the data** (8,460 pairs) | 0.804 |

Five attempts, real gains, and **every one still below the trivial baseline.** The report called
this a ceiling and blamed the features. That conclusion was wrong.

**Phase 2 — measured on English held-out data.** Baseline to beat: **0.746**.

| # | What changed | Page grouping | Stress packet |
|---|---|---|---|
| — | *The Phase 1 model, pointed at English documents* | *0.518* | 2 of 7 |
| 6 | **Retrain on English data** — the corpora had been Dutch all along | 0.947 | 4 of 7 |
| 7 | **Decision rule chosen on validation, confirmed once on test** | **0.968** | 5 of 7 |
| 8 | Feature fix: scan the opening block, not just line one | unchanged | **7 of 7** |

The last column is the stress packet described above: how many of its 7 documents came out split at
**exactly** the right pages, with one page early or late not counting. It is harsher than page
grouping, and it is what decides whether you get the right PDF out.

Step 8 is why both columns are there. It moved the corpus metric by **zero** — every TABME++ page
is a single line of text, so a line-based feature cannot fire there — while taking a real 13-page
PDF from 5 correct documents to 7. Measured on the corpus alone it looks like nothing happened.
That gap is a train/serve mismatch: seven of the twenty-one features see whole-page text during
training and real line structure in production.

The single biggest jump was not a better model. It was noticing that every benchmark was Dutch
while every document being processed was English — the vocabulary recognised **27.6%** of the
tokens it was scoring. Retraining the *same architecture* on English data moved grouping from
0.518 to 0.947 and cleared the baseline by **+0.188**, after six attempts at better features and
better learners had not.

**Why train on Dutch at all?** OpenPSS was the labelled page-stream-segmentation corpus available —
there are very few — and the benchmark it was evaluated against came from the same source. Training
and evaluation agreed, so the model was always scored in the language it was trained in, where a
language mismatch is invisible by construction. The error was not training on Dutch; it was never
holding out an evaluation set in the language the system would actually process. Nothing was
discarded when it surfaced: the features, calibration, decision-rule machinery and evaluation
protocol all carried over unchanged, and the Dutch model still ships for Dutch documents.

### What was tried and did not work

Kept here because they cost real time and are the reason the shipped design looks like it does:

| Attempt | Outcome |
|---|---|
| Pseudo-blocks reconstructed from OCR text | Worse (0.359 vs 0.377) — inferred geometry is not geometry |
| Four gradient-boosting libraries compared | All within 0.005 — the learner was never the bottleneck |
| Training on single-page RVL-CDIP documents | Fabricates continuations that do not exist |
| Training on Dutch and English mixed | Worse than either alone; the larger corpus dominates |
| **Cosine similarity alone as the splitter** | Scores at real boundaries and inside documents **overlap completely** — no threshold can separate them |
| Hybrid decision rule (top-*N* plus an override) | Tied the simpler rule, then optimised itself into it |
| `ceil` instead of `round` for the count estimate | Led validation by +0.0165, did not transfer (+0.0025 on test) |

Two of those last three won on validation and lost on held-out test. They are listed because the
protocol that caught them — select on validation, measure once on test — is the part of this
project worth copying.

## Using it as a web service

```bash
uvicorn src.api:create_app --factory --host 0.0.0.0 --port 8000
```

The same command serves both the browser interface at `/` and the API. The interface is plain
HTML, CSS and JavaScript with no build step — nothing to install beyond `requirements.txt`.

For the UI and for any client that should not be handed filesystem paths:

| Endpoint | What it does |
|---|---|
| `POST /api/process` | Upload a PDF (and optionally ground truth), get documents, scores and download links |
| `POST /api/process-sample` | Run the bundled sample, scored against its shipped ground truth |
| `GET /api/jobs/{id}/documents/{n}/pdf` | Download one split document as its own PDF |
| `GET /api/jobs/{id}/archive` | Download every split document as a zip |
| `POST /api/jobs/{id}/search` | Ask a question about that packet |
| `GET /api/benchmarks` | The held-out benchmark figures the UI displays |

Jobs are addressed by an opaque id and every path resolves underneath the outputs directory, so a
client cannot walk the filesystem.

The original path-based endpoints are unchanged, for scripts that already use them:

| Endpoint | What it does |
|---|---|
| `POST /process` | Send a PDF, get back the split documents and their structure |
| `POST /retrieve` | Ask a question, get evidence with document and page numbers |
| `POST /context` | Same, but packaged as one block with `[1] [2]` citations, ready to hand to an AI assistant |
| `GET /health` | Check the service is running |

With Docker:

```bash
docker build -t document-packet-intelligence .
docker run -p 8000:8000 document-packet-intelligence
```

## Tests

```bash
pytest tests/ -q
```

**43 tests.** Most exist because something was genuinely broken and got fixed — tables being
flattened into plain text, a search index that made results *worse*, a settings flag that crashed
the program when switched on. Each test stops that specific bug coming back.

## Rebuilding the models yourself

Everything downloads automatically; no manual dataset setup.

```bash
# Split-detection model (English, the shipped default)
python scripts/fetch_tabme.py --split train --max-rows 6000 --output data/raw/tabme/manifest.json
python scripts/train_openpss_boundary.py --manifest data/raw/tabme/manifest.json --output models/boundary_tabme --calibrate

# ...and to check it on held-out English packets
python scripts/fetch_tabme.py --split val --max-rows 6000 --output data/raw/tabme_val/manifest.json

# Split-detection model (Dutch, optional alternative)
python scripts/fetch_openpss.py --config SHORT --split train --output data/raw/openpss/train --max-rows 16000
python scripts/train_openpss_boundary.py --manifest data/raw/openpss/train/manifest.json --output models/boundary_openpss --calibrate

# Document-type model
python scripts/fetch_rvlcdip_text.py --output data/raw/rvlcdip/train_text.json --max-rows 12000
python scripts/train_document_classifier.py --training_json data/raw/rvlcdip/train_text.json --output models/document_classifier/tfidf_lr.pkl

# Test files and benchmarks
python scripts/generate_sample_packet.py && python scripts/generate_benchmark_report.py
python scripts/generate_stress_packet.py          # 13 pages, 7 documents, built to be hard
python scripts/evaluate_stage2.py --packet data/samples/benchmark_report.pdf --ground-truth data/samples/benchmark_ground_truth.json
python scripts/evaluate_stage3.py --distractors data/raw/openpss/test_full/manifest.json --distractor-streams 2

# How the decision rule was chosen, and why expected_count was not kept
python scripts/tune_decision_rule.py --val data/raw/tabme_val/manifest.json --test data/raw/tabme_test/manifest.json
python scripts/analyse_expected_count.py
```

Results land in `outputs/benchmarks/`. There is also an experimental AI-based splitter
(`scripts/train_cross_encoder.py`) that performs better on Dutch but is far slower — see report
section 5.1.

The stress packet is the one worth trying first. The shipped sample is scored perfectly, which
makes it useless for telling whether a change helped; the stress packet puts an invoice next to a
budget under the same letterhead, two one-page documents back to back, and a three-page passport
whose pages look nothing like each other.

## Settings

All in [`config/default.yaml`](config/default.yaml). The ones worth knowing:

| Setting | Default | What it changes |
|---|---|---|
| `boundary.decision` | `threshold` | How it decides where to split. `threshold` uses `boundary.threshold`; `expected_count` instead splits at the *N* highest-scoring seams where *N* is the sum of the probabilities. Use `expected_count` with the Dutch model |
| `boundary.threshold` | `0.185011` | Only used when `decision: threshold`. **Belongs to the shipped English model** — a threshold is a property of one model's scores, so retune it if you change the model |
| `classification.min_confidence` | `0.35` | Below this, the type is reported as `unknown` instead of guessing |
| `retrieval.encoder` | `svd` | Meaning-based search method. `transformer` is more accurate but needs ~1 GB of extra downloads |
| `retrieval.rerank` | `true` | Re-orders results. Big accuracy gain, costs ~2 milliseconds |
| `ingestion.enable_ocr` | `false` | Turn on for scanned images; needs Tesseract installed |

## What this does not do well

Stated up front rather than buried:

- **Documents in a language the splitter was not trained on.** The shipped model is English. On
  Dutch its recall for document boundaries collapses to 0.34, meaning it silently merges documents
  rather than splitting them. Retrain, or switch to the bundled Dutch model.
- **Packets of mostly single-page documents.** Two one-page documents side by side give the model
  no continuation cue to reject, so it tends to merge them. Held-out recall is 0.991, but that is on
  packets averaging five pages; short documents are harder.
- **Scanned documents.** Everything measured here is digital text with clean page furniture. OCR
  noise, skew and missing page numbers are untested, and `ingestion.enable_ocr` is off by default.
  Do not assume the 0.968 transfers to scans.
- **Passport and bank statement types** have no proper accuracy measurement — no public labelled
  data for them exists, so they use a keyword-based fallback with an honest confidence score.
- **The structure tests use files I created myself**, so a perfect score there means the code handles
  those cases, not that it handles every PDF in the world.
