from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.pipeline import DocumentPipeline

parser = argparse.ArgumentParser(); parser.add_argument("--input"); parser.add_argument("--output"); parser.add_argument("--query"); parser.add_argument("--processed-dir"); parser.add_argument("--config", default="config/default.yaml"); parser.add_argument("--top-k", type=int, default=5); parser.add_argument("--query-aware", action="store_true"); parser.add_argument("--boundary-model"); parser.add_argument("--document-model"); args = parser.parse_args()
pipeline = DocumentPipeline(load_config(args.config))
if args.query:
    if not args.processed_dir: parser.error("--processed-dir is required with --query")
    print(json.dumps({"query": args.query, "results": pipeline.retrieve_from_dir(args.query, args.processed_dir, args.top_k, args.query_aware)}, indent=2))
else:
    if not args.input or not args.output: parser.error("--input and --output are required for processing")
    print(json.dumps(pipeline.process(args.input, args.output, boundary_model_dir=args.boundary_model, document_model=args.document_model), indent=2))
