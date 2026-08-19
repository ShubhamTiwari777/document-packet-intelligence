"""Extract features from reconstructed DocSplit packets and train our boundary model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.features.extractor import packet_feature_rows, save_feature_cache
from src.ingestion.pdf_parser import PDFParser
from src.stage1.boundary_classifier import SklearnBoundaryModel


parser = argparse.ArgumentParser()
parser.add_argument("--packet-index", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--limit", type=int, default=500)
parser.add_argument("--seed", type=int, default=None, help="Seeded packet sample; defaults to config random_seed")
args = parser.parse_args()

config = load_config(args.config); index = json.loads(Path(args.packet_index).read_text(encoding="utf-8")); records = list(index["built"])
if args.limit and args.limit < len(records):
    random.Random(args.seed if args.seed is not None else config.runtime.random_seed).shuffle(records)
    records = records[:args.limit]
parser_impl = PDFParser(config.ingestion, config.runtime); rows: list[dict] = []; labels: list[int] = []; skipped: list[dict] = []
cache_dir = Path(config.paths.cache_dir) / "docsplit_train"
for position, record in enumerate(records, 1):
    try:
        render_dir = cache_dir / "rendered" / record["packet_id"]
        pages = parser_impl.parse(record["pdf_path"], render_dir)
        features = packet_feature_rows(pages, record["packet_id"])
        if len(features) != len(record["boundary_labels"]): raise ValueError("feature/label count mismatch")
        rows.extend(features); labels.extend(record["boundary_labels"])
    except Exception as exc:
        skipped.append({"packet_id": record["packet_id"], "reason": str(exc)})
    if position % 25 == 0 or position == len(records): print(f"Processed {position}/{len(records)} packets")
if not rows: raise SystemExit("No feature rows were extracted.")
save_feature_cache(rows, cache_dir / "boundary_features.json")
model = SklearnBoundaryModel.train(rows, labels, config.runtime.random_seed)
model.save(args.output, {"training_config": config.to_dict(), "packet_count": len(records) - len(skipped), "feature_row_count": len(rows), "positive_boundaries": sum(labels), "skipped": skipped})
print(f"Trained boundary model on {len(rows)} adjacent page pairs: {args.output}")