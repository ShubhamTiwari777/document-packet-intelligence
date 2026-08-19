"""Download a local OpenPSS manifest (images + OCR text + boundary labels) for training/eval."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.openpss_dataset import fetch_openpss_manifest

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="SHORT", choices=["SHORT", "LONG"])
parser.add_argument("--split", default="train", choices=["train", "test"])
parser.add_argument("--output", required=True)
parser.add_argument("--max-rows", type=int, default=8000)
parser.add_argument("--max-streams", type=int, default=None)
parser.add_argument("--workers", type=int, default=16)
args = parser.parse_args()

manifest = fetch_openpss_manifest(
    args.output, config=args.config, split=args.split,
    max_rows=args.max_rows, max_streams=args.max_streams, workers=args.workers,
)
print(f"Fetched {manifest['stream_count']} streams, {manifest['page_count']} pages, "
      f"{manifest['positive_boundaries']} positive boundaries -> {args.output}/manifest.json")
