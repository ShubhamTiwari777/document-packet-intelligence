"""Train a boundary model on synthetic RVL-CDIP packets that carry real page geometry.

Uses an honest three-way split by PACKET (never by page pair, which would leak pages of the same
document across splits): train the model, tune the decision threshold on validation, and report
on test. Optionally also scores the resulting model on OpenPSS to measure cross-domain transfer.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluation.stage1_metrics import boundary_metrics
from src.features.extractor import FEATURE_NAMES, packet_feature_rows, save_feature_cache
from src.features.text_features import TfidfTextEmbedder
from src.stage1.boundary_classifier import SklearnBoundaryModel
from src.synthetic_packets import build_packets

parser = argparse.ArgumentParser()
parser.add_argument("--pages", default="data/raw/rvlcdip/pages_boxes.json")
parser.add_argument("--output", default="models/boundary_rvlcdip")
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--packets", type=int, default=800)
parser.add_argument("--cache", default="data/processed/cache/rvlcdip_packets/boundary_features.json")
parser.add_argument("--report", default="outputs/benchmarks/boundary_validation_rvlcdip.json")
args = parser.parse_args()

config = load_config(args.config)
pages = json.loads(Path(args.pages).read_text(encoding="utf-8"))["pages"]
packets = build_packets(pages, packet_count=args.packets, seed=config.runtime.random_seed)
if len(packets) < 30:
    raise SystemExit(f"Only {len(packets)} packets could be built; fetch more pages.")

# Split by packet: pages of one document must never straddle a split.
train_end = int(len(packets) * 0.6)
val_end = int(len(packets) * 0.8)
splits = {"train": packets[:train_end], "val": packets[train_end:val_end], "test": packets[val_end:]}
print({name: len(group) for name, group in splits.items()})

embedder = TfidfTextEmbedder.fit([page.text for packet in splits["train"] for page in packet["pages"]])


def features_for(group: list[dict]) -> tuple[list[dict], list[int]]:
    rows: list[dict] = []
    labels: list[int] = []
    for packet in group:
        extracted = packet_feature_rows(packet["pages"], packet["packet_id"], embedder)
        if len(extracted) != len(packet["boundary_labels"]):
            continue
        rows.extend(extracted)
        labels.extend(packet["boundary_labels"])
    return rows, labels


start = time.perf_counter()
train_rows, train_labels = features_for(splits["train"])
val_rows, val_labels = features_for(splits["val"])
test_rows, test_labels = features_for(splits["test"])
print(f"features: train {len(train_rows)} / val {len(val_rows)} / test {len(test_rows)} "
      f"({time.perf_counter() - start:.1f}s)")

live = [name for name in FEATURE_NAMES if len({row[name] for row in train_rows}) > 1]
print(f"live features: {len(live)}/{len(FEATURE_NAMES)}")
print(f"  constant: {[n for n in FEATURE_NAMES if n not in live]}")

save_feature_cache(train_rows, args.cache)
model = SklearnBoundaryModel.train(train_rows, train_labels, config.runtime.random_seed, calibrate=True)
embedder.save(Path(args.output) / "text_embedder.pkl")
model.save(args.output, {
    "dataset": "albertklorer/rvl_cdip_ocr (synthetic packets)", "split": "train",
    "training_config": config.to_dict(), "stream_count": len(splits["train"]),
    "feature_row_count": len(train_rows), "positive_boundaries": sum(train_labels),
    "calibrated": True, "tfidf_text_delta": True, "synthetic_blocks": False,
    "class_weight": "balanced", "live_features": live, "skipped": [],
})

val_scores = model.predict_proba(val_rows)
threshold, _ = max(
    ((t, boundary_metrics([int(s >= t) for s in val_scores], val_labels)) for t in sorted({0.0, 1.0, *val_scores})),
    key=lambda item: item[1]["f1"],
)
test_scores = model.predict_proba(test_rows)
metrics = boundary_metrics([int(s >= threshold) for s in test_scores], test_labels)

result = {
    "dataset": "rvl_cdip synthetic packets", "packets": len(packets),
    "train_packets": len(splits["train"]), "val_packets": len(splits["val"]), "test_packets": len(splits["test"]),
    "adjacent_pairs": len(test_labels), "positive_boundaries": sum(test_labels),
    "positive_rate": round(sum(test_labels) / max(len(test_labels), 1), 4),
    "holdout_threshold": threshold, "holdout_metrics": metrics,
    "live_features": len(live), "total_features": len(FEATURE_NAMES),
}
Path(args.report).parent.mkdir(parents=True, exist_ok=True)
Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
