"""Download RVL-CDIP OCR text + document-type labels for classifier training."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rvlcdip_dataset import fetch_rvlcdip_text

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, help="e.g. data/raw/rvlcdip/train_text.json")
parser.add_argument("--max-rows", type=int, default=10_000)
parser.add_argument("--split", default="train")
args = parser.parse_args()

manifest = fetch_rvlcdip_text(args.output, max_rows=args.max_rows, split=args.split)
counts = Counter(record["label"] for record in manifest["records"])
print(f"Fetched {manifest['record_count']} labeled documents -> {args.output}")
for label, count in counts.most_common():
    print(f"  {label:<26} {count}")
