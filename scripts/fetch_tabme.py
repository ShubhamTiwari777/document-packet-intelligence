"""Fetch TABME++ pages and assemble them into packets with REAL word geometry.

Why this dataset. Two earlier attempts to revive the eight constant layout features failed for
opposite reasons: pseudo-blocks inferred from OCR text gave geometry that was not real, and
RVL-CDIP gave real geometry but single-page rows, so grouping them fabricated continuations.
TABME++ is the first source with both -- `doc_id` groups pages into genuine multi-page documents,
and its OCR carries per-word bounding boxes.

Images are skipped. They dominate the row payload (~165 KB/page) and only `visual_delta` needs
them, so fetching text plus boxes makes a useful subset practical.

Boxes arrive normalised to 0-1 and are rescaled to a nominal page size so the existing
block-reconstruction and feature code consumes them unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.parse
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.openpss_dataset import _get_with_backoff

DATASET = "bevaya%2FTABMEpp"
PAGE_SIZE = 20          # rows carry embedded images, so keep requests small
PAGE_WIDTH, PAGE_HEIGHT = 1000.0, 1000.0

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train", choices=["train", "val", "test"])
parser.add_argument("--max-rows", type=int, default=6000)
parser.add_argument("--output", default="data/raw/tabme/manifest.json")
parser.add_argument("--min-words", type=int, default=8)
parser.add_argument("--min-documents", type=int, default=2)
parser.add_argument("--max-documents", type=int, default=5)
parser.add_argument("--max-document-pages", type=int, default=4)
parser.add_argument("--packets", type=int, default=4000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def parse_ocr(blob: str) -> tuple[list[str], list[list[float]]]:
    """Pull words and pixel-space boxes out of the TABME++ OCR JSON."""
    try:
        payload = json.loads(blob) if isinstance(blob, str) else (blob or {})
    except json.JSONDecodeError:
        return [], []
    entries = payload.get("words_data") or payload.get("lines_data") or []
    words: list[str] = []
    boxes: list[list[float]] = []
    for entry in entries:
        word = str(entry.get("Word") or entry.get("Text") or "").strip()
        if not word:
            continue
        try:
            x1, y1 = float(entry["X1"]), float(entry["Y1"])
            x2, y2 = float(entry["X2"]), float(entry["Y2"])
        except (KeyError, TypeError, ValueError):
            continue
        # Normalised coordinates -> pixels. Y2 is sometimes the baseline rather than the box
        # bottom, so give every word a minimum height or block grouping collapses.
        top, bottom = min(y1, y2), max(y1, y2)
        if bottom - top < 0.004:
            bottom = top + 0.012
        words.append(word)
        boxes.append([x1 * PAGE_WIDTH, top * PAGE_HEIGHT, x2 * PAGE_WIDTH, bottom * PAGE_HEIGHT])
    return words, boxes


rows_by_document: dict[str, list[dict]] = defaultdict(list)
offset = 0
while offset < args.max_rows:
    length = min(PAGE_SIZE, args.max_rows - offset)
    url = (f"https://datasets-server.huggingface.co/rows?dataset={DATASET}"
           f"&config=default&split={args.split}&offset={offset}&length={length}")
    batch = json.loads(_get_with_backoff(url, retries=8, base_delay=2.0))["rows"]
    if not batch:
        break
    for item in batch:
        row = item["row"]
        words, boxes = parse_ocr(row.get("ocr"))
        if len(words) < args.min_words:
            continue
        rows_by_document[row["doc_id"]].append(
            {"pg_id": int(row["pg_id"]), "words": words, "boxes": boxes})
    offset += len(batch)
    if offset % 500 == 0:
        print(f"  {offset} rows -> {len(rows_by_document)} documents", flush=True)
    if len(batch) < length:
        break

# Order pages inside each document, then keep only short ones so packet density matches the
# target regime (short packets, boundaries common).
documents = []
for doc_id, pages in rows_by_document.items():
    pages.sort(key=lambda p: p["pg_id"])
    if 1 <= len(pages) <= args.max_document_pages:
        documents.append(pages)

rng = random.Random(args.seed)
rng.shuffle(documents)
print(f"documents: {len(rows_by_document)} total, {len(documents)} of 1-{args.max_document_pages} pages")

streams: list[dict] = []
cursor = 0
for index in range(args.packets):
    count = rng.randint(args.min_documents, args.max_documents)
    if cursor + count > len(documents):
        break
    chosen = documents[cursor:cursor + count]
    cursor += count
    pages: list[dict] = []
    labels: list[int] = []
    for document in chosen:
        for offset_in_doc, page in enumerate(document):
            if pages:
                labels.append(1 if offset_in_doc == 0 else 0)
            pages.append({
                "position": len(pages) + 1, "text": " ".join(page["words"]),
                "words": page["words"], "boxes": page["boxes"],
                "width": PAGE_WIDTH, "height": PAGE_HEIGHT, "image_path": None,
            })
    if len(pages) < 2:
        continue
    streams.append({"stream_id": f"tabme_{index:05d}", "page_count": len(pages),
                    "pages": pages, "boundary_labels": labels})

pairs = sum(len(s["boundary_labels"]) for s in streams)
positives = sum(sum(s["boundary_labels"]) for s in streams)
manifest = {"dataset": "bevaya/TABMEpp (packets rebuilt from doc_id groups)", "config": "TABMEpp",
            "split": args.split, "stream_count": len(streams),
            "page_count": sum(s["page_count"] for s in streams),
            "positive_boundaries": positives, "streams": streams}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest), encoding="utf-8")

median = sorted(s["page_count"] for s in streams)[len(streams) // 2] if streams else 0
print(f"packets {len(streams)}  pages {manifest['page_count']}  pairs {pairs}  positives {positives}  "
      f"density {positives / max(pairs, 1):.1%}  median {median} pages/packet")
print(f"-> {output}")
