"""Train the Stage 1 document-type classifier and report held-out classification metrics.

Input is a JSON file of `{"records": [{"text":..., "label":...}]}` (as produced by
scripts/fetch_rvlcdip_text.py) or a bare `[{"text":..., "label":...}]` list.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluation.stage1_metrics import classification_metrics
from src.stage1.document_classifier import TfidfDocumentClassifier, EmbeddingCentroidDocumentClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--training_json", required=True, help='JSON: {"records":[{"text","label"}]} or [{"text","label"}]')
parser.add_argument("--output", required=True)
parser.add_argument("--method", choices=("tfidf", "embedding_centroid"), default="tfidf")
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--holdout", type=float, default=0.2, help="Fraction held out for metrics")
parser.add_argument("--metrics-output", default="outputs/benchmarks/classification_validation.json")
args = parser.parse_args()

config = load_config(args.config)
payload = json.loads(Path(args.training_json).read_text(encoding="utf-8"))
records = payload["records"] if isinstance(payload, dict) else payload

random.Random(config.runtime.random_seed).shuffle(records)
split = int(len(records) * (1 - args.holdout))
train_records, test_records = records[:split], records[split:]
if not train_records or not test_records:
    raise SystemExit("Not enough records to build a train/holdout split.")

train_texts = [r["text"] for r in train_records]
train_labels = [r["label"] for r in train_records]

if args.method == "tfidf":
    model = TfidfDocumentClassifier.train(train_texts, train_labels, config.runtime.random_seed)
else:
    model = EmbeddingCentroidDocumentClassifier.train(train_texts, train_labels)
model.save(args.output)

predicted = [prediction.label for prediction in model.predict([r["text"] for r in test_records])]
actual = [r["label"] for r in test_records]
metrics = classification_metrics(predicted, actual)

per_class: dict[str, dict[str, float]] = {}
for label in sorted(set(actual)):
    tp = sum(p == a == label for p, a in zip(predicted, actual))
    fp = sum(p == label and a != label for p, a in zip(predicted, actual))
    fn = sum(p != label and a == label for p, a in zip(predicted, actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    per_class[label] = {
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "support": sum(a == label for a in actual),
    }

result = {
    "method": args.method, "dataset": payload.get("dataset") if isinstance(payload, dict) else None,
    "train_documents": len(train_records), "holdout_documents": len(test_records),
    "classes": sorted(set(train_labels)), "accuracy": round(metrics["accuracy"], 4),
    "macro_f1": round(metrics["macro_f1"], 4), "per_class": per_class,
    "train_label_counts": dict(Counter(train_labels).most_common()),
}
output = Path(args.metrics_output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "per_class"}, indent=2))
print(f"Saved model -> {args.output}\nSaved metrics -> {output}")
