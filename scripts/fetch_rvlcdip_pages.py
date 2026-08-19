"""Download RVL-CDIP pages with word-level bounding boxes for synthetic packet construction."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rvlcdip_dataset import fetch_rvlcdip_pages

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--max-rows", type=int, default=6000)
parser.add_argument("--split", default="train")
args = parser.parse_args()

manifest = fetch_rvlcdip_pages(args.output, max_rows=args.max_rows, split=args.split)
counts = Counter(page["label"] for page in manifest["pages"])
print(f"Fetched {manifest['page_count']} pages with boxes -> {args.output}")
for label, count in counts.most_common(5):
    print(f"  {label:<24} {count}")
