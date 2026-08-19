from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.features.extractor import load_feature_cache
from src.stage1.boundary_classifier import SklearnBoundaryModel

parser = argparse.ArgumentParser(); parser.add_argument("--features", required=True, help="Feature cache JSON"); parser.add_argument("--labels", required=True, help="JSON list of 0/1 labels aligned to rows"); parser.add_argument("--output", required=True); parser.add_argument("--config", default="config/default.yaml"); args = parser.parse_args()
config = load_config(args.config); rows = load_feature_cache(args.features); labels = json.loads(Path(args.labels).read_text(encoding="utf-8")); model = SklearnBoundaryModel.train(rows, labels, config.runtime.random_seed); model.save(args.output, {"training_config": config.to_dict(), "row_count": len(rows)})
