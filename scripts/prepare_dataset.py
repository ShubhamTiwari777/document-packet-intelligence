"""Validate a local DocSplit export; downloading is deliberately explicit."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import discover_packets

parser = argparse.ArgumentParser(); parser.add_argument("--dataset", required=True); args = parser.parse_args()
packets = discover_packets(args.dataset)
print(f"Found {len(packets)} PDFs under {Path(args.dataset).resolve()}")
if not packets: raise SystemExit("No PDFs found. Download DocSplit v2 manually and point --dataset to the export.")
