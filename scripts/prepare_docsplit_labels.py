"""Normalize a downloaded DocSplit CSV into packet-level training labels."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import docsplit_csv_to_manifest
from src.domain import dump_json


parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="DocSplit train.csv or validation.csv")
parser.add_argument("--output", required=True, help="Output JSON packet-label manifest")
args = parser.parse_args()

manifest = docsplit_csv_to_manifest(args.csv)
dump_json(manifest, args.output)
print(f"Wrote {len(manifest)} packet manifests to {args.output}")
