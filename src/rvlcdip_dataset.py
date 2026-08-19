"""RVL-CDIP OCR-text adapter (albertklorer/rvl_cdip_ocr).

Supplies labeled document-type training text for the Stage 1 document classifier.
OpenPSS carries boundary labels but no document-type labels, so it cannot train a type
classifier; RVL-CDIP is the standard public document-type taxonomy (16 general business
classes, including `invoice` and `resume`) and this mirror ships pre-extracted OCR words,
which means no OCR engine is required at training time.

Only `words` and `label` are consumed -- page images are never downloaded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import urllib.parse

from src.openpss_dataset import _get_with_backoff

DATASET = "albertklorer/rvl_cdip_ocr"
PAGE_SIZE = 100

# ClassLabel order as published by the dataset's features.
LABEL_NAMES = [
    "letter", "form", "email", "handwritten", "advertisement", "scientific_report",
    "scientific_publication", "specification", "file_folder", "news_article", "budget",
    "invoice", "presentation", "questionnaire", "resume", "memo",
]


def fetch_rvlcdip_text(
    output_path: str | Path,
    max_rows: int = 10_000,
    split: str = "train",
    min_words: int = 10,
    progress: bool = True,
) -> dict[str, Any]:
    """Download OCR text + type labels into a local JSON training file (resumable-safe)."""
    quoted = urllib.parse.quote(DATASET, safe="")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    offset = 0
    while offset < max_rows:
        length = min(PAGE_SIZE, max_rows - offset)
        url = (
            f"https://datasets-server.huggingface.co/rows?dataset={quoted}"
            f"&config=default&split={split}&offset={offset}&length={length}"
        )
        payload = json.loads(_get_with_backoff(url, retries=8, base_delay=3.0))
        rows = payload["rows"]
        if not rows:
            break
        for item in rows:
            row = item["row"]
            words = row.get("words") or []
            if len(words) < min_words:
                continue  # near-empty OCR carries no usable type signal
            records.append({"text": " ".join(words), "label": LABEL_NAMES[row["label"]]})
        offset += len(rows)
        if progress and offset % 1000 == 0:
            print(f"  fetched {offset} rows -> {len(records)} usable records")
        # Checkpoint so a network blip does not discard the whole download.
        if offset % 1000 == 0:
            output.write_text(json.dumps({"dataset": DATASET, "records": records}, indent=2), encoding="utf-8")
        if len(rows) < length:
            break
    manifest = {"dataset": DATASET, "split": split, "record_count": len(records), "records": records}
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
