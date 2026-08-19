"""Train the boundary classifier on a locally fetched OpenPSS manifest (see fetch_openpss.py)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.openpss_dataset import pages_from_stream
from src.features.extractor import packet_feature_rows, save_feature_cache
from src.features.text_features import TfidfTextEmbedder
from src.stage1.boundary_classifier import SklearnBoundaryModel

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True, help="data/raw/openpss/train/manifest.json")
parser.add_argument("--output", required=True, help="Model directory, e.g. models/boundary_openpss")
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--calibrate", action="store_true")
parser.add_argument("--no-tfidf", action="store_true", help="Fall back to the hashed bag-of-words text_delta")
parser.add_argument("--no-synthetic-blocks", action="store_true", help="Disable text-derived pseudo-blocks (ablation)")
parser.add_argument("--class-weight", default="balanced", choices=["balanced","none"], help="none keeps the natural boundary prior")
parser.add_argument("--cache", default=None, help="Optional path to cache the extracted feature rows")
args = parser.parse_args()

config = load_config(args.config)
manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

streams = [(stream, pages_from_stream(stream, not args.no_synthetic_blocks)) for stream in manifest["streams"]]

text_embedder = None
if not args.no_tfidf:
    corpus = [page.text for _, pages in streams for page in pages]
    text_embedder = TfidfTextEmbedder.fit(corpus)
    text_embedder.save(Path(args.output) / "text_embedder.pkl")

rows: list[dict] = []
labels: list[int] = []
skipped: list[dict] = []
for stream, pages in streams:
    try:
        features = packet_feature_rows(pages, stream["stream_id"], text_embedder)
        if len(features) != len(stream["boundary_labels"]):
            raise ValueError("feature/label count mismatch")
        rows.extend(features)
        labels.extend(stream["boundary_labels"])
    except Exception as exc:
        skipped.append({"stream_id": stream["stream_id"], "reason": str(exc)})

if not rows:
    raise SystemExit("No feature rows were extracted from the manifest.")

if args.cache:
    save_feature_cache(rows, args.cache)

model = SklearnBoundaryModel.train(rows, labels, config.runtime.random_seed, calibrate=args.calibrate, class_weight=None if args.class_weight == "none" else args.class_weight)
model.save(args.output, {
    "dataset": manifest.get("dataset"), "config": manifest.get("config"), "split": manifest.get("split"),
    "training_config": config.to_dict(), "stream_count": manifest["stream_count"] - len(skipped),
    "feature_row_count": len(rows), "positive_boundaries": sum(labels), "calibrated": args.calibrate,
    "tfidf_text_delta": text_embedder is not None,
    # Recorded so evaluation and inference reproduce the exact feature construction used in
    # training; a mismatch here silently changes 4 of the 14 features.
    "synthetic_blocks": not args.no_synthetic_blocks, "class_weight": args.class_weight,
    "skipped": skipped,
})
print(f"Trained boundary model on {len(rows)} adjacent page pairs ({sum(labels)} positive) from "
      f"{manifest['stream_count'] - len(skipped)} streams -> {args.output}")
