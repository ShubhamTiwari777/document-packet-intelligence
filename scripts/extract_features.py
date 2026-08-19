from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.features.extractor import packet_feature_rows, save_feature_cache
from src.ingestion.pdf_parser import PDFParser

parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--output", required=True); parser.add_argument("--config", default="config/default.yaml"); args = parser.parse_args()
config = load_config(args.config); pages = PDFParser(config.ingestion, config.runtime).parse(args.input, Path(args.output).parent / "rendered_pages")
save_feature_cache(packet_feature_rows(pages, Path(args.input).stem), args.output)
