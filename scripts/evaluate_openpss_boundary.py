"""Evaluate a trained boundary model on a held-out OpenPSS manifest and pick a threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.domain import PageRepresentation
from src.evaluation.stage1_metrics import boundary_metrics
from src.features.extractor import packet_feature_rows
from src.features.text_features import TfidfTextEmbedder
from src.stage1.boundary_classifier import SklearnBoundaryModel

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True, help="data/raw/openpss/test/manifest.json")
parser.add_argument("--model", required=True)
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--output", default="outputs/benchmarks/boundary_validation_openpss.json")
args = parser.parse_args()

config = load_config(args.config)
manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
model = SklearnBoundaryModel.load(args.model)
embedder_path = Path(args.model) / "text_embedder.pkl"
text_embedder = TfidfTextEmbedder.load(embedder_path) if embedder_path.exists() else None

probabilities: list[float] = []
labels: list[int] = []
skipped: list[dict] = []
for stream in manifest["streams"]:
    try:
        pages = [
            PageRepresentation(
                page_number=page["position"], text=page["text"], blocks=[], fonts=[],
                width=float(page["width"]), height=float(page["height"]), image_path=page["image_path"],
            )
            for page in stream["pages"]
        ]
        rows = packet_feature_rows(pages, stream["stream_id"], text_embedder)
        if len(rows) != len(stream["boundary_labels"]):
            raise ValueError("feature/label count mismatch")
        probabilities.extend(model.predict_proba(rows))
        labels.extend(stream["boundary_labels"])
    except Exception as exc:
        skipped.append({"stream_id": stream["stream_id"], "reason": str(exc)})

if not labels:
    raise SystemExit("No validation pairs were evaluated.")

thresholds = sorted({0.0, 1.0, *probabilities})
best_threshold, best_metrics = max(
    ((threshold, boundary_metrics([int(score >= threshold) for score in probabilities], labels)) for threshold in thresholds),
    key=lambda item: item[1]["f1"],
)
default_metrics = boundary_metrics([int(score >= config.boundary.threshold) for score in probabilities], labels)
result = {
    "dataset": manifest.get("dataset"), "config": manifest.get("config"), "split": manifest.get("split"),
    "evaluated_streams": manifest["stream_count"] - len(skipped), "adjacent_pairs": len(labels),
    "positive_boundaries": sum(labels), "recommended_threshold": best_threshold, "recommended_metrics": best_metrics,
    "current_threshold": config.boundary.threshold, "current_threshold_metrics": default_metrics, "skipped": skipped,
}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
