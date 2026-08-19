from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.domain import Chunk
from src.stage3.retriever import HybridRetriever

parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--output", required=True); parser.add_argument("--config", default="config/default.yaml"); args = parser.parse_args()
chunks = [Chunk(**item) for item in json.loads((Path(args.input) / "chunks.json").read_text(encoding="utf-8"))]; HybridRetriever(chunks, load_config(args.config).retrieval).save(args.output); print(f"Indexed {len(chunks)} chunks")
