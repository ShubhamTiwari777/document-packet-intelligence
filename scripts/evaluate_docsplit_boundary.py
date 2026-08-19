"""Evaluate a trained boundary model on held-out reconstructed DocSplit packets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluation.stage1_metrics import boundary_metrics
from src.features.extractor import packet_feature_rows
from src.ingestion.pdf_parser import PDFParser
from src.stage1.boundary_classifier import SklearnBoundaryModel


parser = argparse.ArgumentParser()
parser.add_argument("--packet-index", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--limit", type=int, default=500)
parser.add_argument("--output", default="outputs/benchmarks/boundary_validation.json")
args = parser.parse_args()

config = load_config(args.config); records = json.loads(Path(args.packet_index).read_text(encoding="utf-8"))["built"][:args.limit]
model = SklearnBoundaryModel.load(args.model); parser_impl = PDFParser(config.ingestion, config.runtime)
probabilities: list[float] = []; labels: list[int] = []; skipped: list[dict] = []
for position, record in enumerate(records, 1):
    try:
        render_dir = Path(config.paths.cache_dir) / "docsplit_validation" / "rendered" / record["packet_id"]
        rows = packet_feature_rows(parser_impl.parse(record["pdf_path"], render_dir), record["packet_id"])
        if len(rows) != len(record["boundary_labels"]): raise ValueError("feature/label count mismatch")
        probabilities.extend(model.predict_proba(rows)); labels.extend(record["boundary_labels"])
    except Exception as exc:
        skipped.append({"packet_id": record["packet_id"], "reason": str(exc)})
    if position % 25 == 0 or position == len(records): print(f"Evaluated {position}/{len(records)} packets")
if not labels: raise SystemExit("No validation pairs were evaluated.")
thresholds = sorted({0.0, 1.0, *probabilities})
best_threshold, best_metrics = max(((threshold, boundary_metrics([int(score >= threshold) for score in probabilities], labels)) for threshold in thresholds), key=lambda item: item[1]["f1"])
default_metrics = boundary_metrics([int(score >= config.boundary.threshold) for score in probabilities], labels)
result = {"evaluated_packets": len(records) - len(skipped), "adjacent_pairs": len(labels), "positive_boundaries": sum(labels), "recommended_threshold": best_threshold, "recommended_metrics": best_metrics, "current_threshold": config.boundary.threshold, "current_threshold_metrics": default_metrics, "skipped": skipped}
output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))