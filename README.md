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

```bash
pip install -r requirements.txt
python scripts/generate_sample_packet.py
python scripts/run_pipeline.py --input data/samples/sample_packet.pdf --output outputs/sample
```

The sample file is a 9-page PDF holding four documents. The system correctly splits it into:

| Pages | Detected as |
|---|---|
| 1–3 | invoice |
| 4–5 | resume |
| 6–7 | passport |
| 8–9 | bank statement |

Then ask it something:

```bash
python scripts/run_pipeline.py --query "What is the closing balance?" --processed-dir outputs/sample --top-k 5
```

It answers with the evidence, the document, the page, and a confidence score.

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
trained models together are only **5.8 MB**.

## Results, and how to read them

| What | Measured on | Result |
|---|---|---|
| Splitting documents | TABME++ held-out, 510 English packets | page grouping accuracy **0.95** |
| Identifying document type | 2,222 held-out documents, 16 types | **80.7%** accuracy |
| Finding the right evidence | 35 questions, 515 text chunks | correct answer ranked #1 **77%** of the time |
| Structure extraction | annotated test files | headings, tables, lists all correct |

**The splitter is language-specific, and that matters more here than any other single fact.**
The shipped model is trained on English. On 510 held-out English packets it scores 0.947 page
grouping and beats the lazy *"every page is its own document"* baseline by **+0.188** — the first
result in this project to clear that baseline by a real margin.

Point the same model at Dutch and it falls to 0.606, because its vocabulary recognises barely a
quarter of the words. A Dutch-trained model is included: set `boundary.model_path` to
`models/boundary_shortpackets` and it scores **0.761** on the Dutch OpenPSS set, where the English
model gets 0.606. Neither is better in general — the choice is a language choice.

Full analysis, including the six experiments that failed first, is in section 5.1 of the technical
report.

Full details: **[technical_report.pdf](technical_report.pdf)** ·
Diagram: **[docs/architecture.pdf](docs/architecture.pdf)**

## Using it as a web service

```bash
uvicorn src.api:create_app --factory --host 0.0.0.0 --port 8000
```

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
python scripts/evaluate_stage2.py --packet data/samples/benchmark_report.pdf --ground-truth data/samples/benchmark_ground_truth.json
python scripts/evaluate_stage3.py --distractors data/raw/openpss/test_full/manifest.json --distractor-streams 2
```

Results land in `outputs/benchmarks/`. There is also an experimental AI-based splitter
(`scripts/train_cross_encoder.py`) that performs better but is far slower — see report section 5.1.

## Settings

All in [`config/default.yaml`](config/default.yaml). The ones worth knowing:

| Setting | Default | What it changes |
|---|---|---|
| `boundary.decision` | `expected_count` | How it decides where to split. This setting works across different document types; a fixed cut-off did not |
| `classification.min_confidence` | `0.35` | Below this, the type is reported as `unknown` instead of guessing |
| `retrieval.encoder` | `svd` | Meaning-based search method. `transformer` is more accurate but needs ~1 GB of extra downloads |
| `retrieval.rerank` | `true` | Re-orders results. Big accuracy gain, costs ~2 milliseconds |
| `ingestion.enable_ocr` | `false` | Turn on for scanned images; needs Tesseract installed |

## What this does not do well

Stated up front rather than buried:

- **Documents in a language the splitter was not trained on.** The shipped model is English. On
  Dutch its recall for document boundaries collapses to 0.34, meaning it silently merges documents
  rather than splitting them. Retrain, or switch to the bundled Dutch model.
- **Packets of mostly single-page documents.** Boundary recall is the weak side even in English
  (0.911 on held-out data, but lower when documents are one page each), and the decision rule is
  forced to commit to a fixed number of splits per packet, so it cannot say "I only found three
  boundaries I trust". This is the main open problem; the report explains what would fix it.
- **Passport and bank statement types** have no proper accuracy measurement — no public labelled
  data for them exists, so they use a keyword-based fallback with an honest confidence score.
- **The structure tests use files I created myself**, so a perfect score there means the code handles
  those cases, not that it handles every PDF in the world.
