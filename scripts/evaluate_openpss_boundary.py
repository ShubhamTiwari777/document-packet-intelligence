"""Evaluate a trained boundary model on a held-out OpenPSS manifest and pick a threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.openpss_dataset import pages_from_stream
from src.evaluation.stage1_metrics import boundary_metrics
from src.features.extractor import packet_feature_rows
from src.features.text_features import TfidfTextEmbedder
from src.stage1.boundary_classifier import SklearnBoundaryModel

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True, help="data/raw/openpss/test/manifest.json")
parser.add_argument("--model", required=True)
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--output", default="outputs/benchmarks/boundary_validation_openpss.json")
parser.add_argument("--no-synthetic-blocks", action="store_true")
args = parser.parse_args()

config = load_config(args.config)
manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
model = SklearnBoundaryModel.load(args.model)
embedder_path = Path(args.model) / "text_embedder.pkl"
text_embedder = TfidfTextEmbedder.load(embedder_path) if embedder_path.exists() else None

# Feature construction must match training exactly. The model records whether it was trained on
# text-derived pseudo-blocks; evaluating with the other setting silently changes 4 of 14 features
# and produces a number that describes no real deployment.
metadata_path = Path(args.model) / "metadata.json"
trained_with_blocks = False
if metadata_path.exists():
    trained_with_blocks = json.loads(metadata_path.read_text(encoding="utf-8")).get("synthetic_blocks", False)
synthesize = trained_with_blocks and not args.no_synthetic_blocks
print(f"Feature mode: synthetic_blocks={synthesize} (model was trained with synthetic_blocks={trained_with_blocks})")

probabilities: list[float] = []
labels: list[int] = []
skipped: list[dict] = []
# Kept per stream so threshold tuning can be split by stream rather than by pair.
stream_probabilities: list[tuple[str, tuple[list[float], list[int]]]] = []
for stream in manifest["streams"]:
    try:
        pages = pages_from_stream(stream, synthesize)
        rows = packet_feature_rows(pages, stream["stream_id"], text_embedder)
        if len(rows) != len(stream["boundary_labels"]):
            raise ValueError("feature/label count mismatch")
        stream_scores = model.predict_proba(rows)
        probabilities.extend(stream_scores)
        labels.extend(stream["boundary_labels"])
        stream_probabilities.append((stream["stream_id"], (stream_scores, list(stream["boundary_labels"]))))
    except Exception as exc:
        skipped.append({"stream_id": stream["stream_id"], "reason": str(exc)})

if not labels:
    raise SystemExit("No validation pairs were evaluated.")


def _tune(probs: list[float], labs: list[int]) -> tuple[float, dict]:
    candidates = sorted({0.0, 1.0, *probs})
    return max(((t, boundary_metrics([int(p >= t) for p in probs], labs)) for t in candidates),
               key=lambda item: item[1]["f1"])


# Threshold selection and scoring must not share data. Selecting the F1-optimal threshold on the
# same pairs it is then scored on reports an optimistic number that no deployment can reproduce:
# a split-half check showed the tuned-and-scored figure (0.377) sitting well above what the same
# threshold achieved on unseen streams. Streams -- not pairs -- are split, because pairs within a
# stream are not independent.
stream_ids = sorted({sid for sid, _ in stream_probabilities})
holdout_ids = {sid for index, sid in enumerate(stream_ids) if index % 2 == 1}
tune_probs = [p for sid, pairs in stream_probabilities if sid not in holdout_ids for p in pairs[0]]
tune_labels = [l for sid, pairs in stream_probabilities if sid not in holdout_ids for l in pairs[1]]
test_probs = [p for sid, pairs in stream_probabilities if sid in holdout_ids for p in pairs[0]]
test_labels = [l for sid, pairs in stream_probabilities if sid in holdout_ids for l in pairs[1]]

honest_threshold, honest_metrics = (None, None)
if tune_probs and test_probs:
    honest_threshold, _ = _tune(tune_probs, tune_labels)
    honest_metrics = boundary_metrics([int(p >= honest_threshold) for p in test_probs], test_labels)

best_threshold, best_metrics = _tune(probabilities, labels)
default_metrics = boundary_metrics([int(score >= config.boundary.threshold) for score in probabilities], labels)
result = {
    "dataset": manifest.get("dataset"), "config": manifest.get("config"), "split": manifest.get("split"),
    "evaluated_streams": manifest["stream_count"] - len(skipped), "adjacent_pairs": len(labels),
    "positive_boundaries": sum(labels), "positive_rate": round(sum(labels) / max(len(labels), 1), 4),
    # The headline, honest figure: threshold fitted on tuning streams, scored on held-out streams.
    "holdout_threshold": honest_threshold, "holdout_metrics": honest_metrics,
    "holdout_streams": len(holdout_ids), "tuning_streams": len(stream_ids) - len(holdout_ids),
    # Retained for comparison; optimistically biased because tuned on the data it scores.
    "oracle_threshold": best_threshold, "oracle_metrics": best_metrics,
    "recommended_threshold": honest_threshold if honest_threshold is not None else best_threshold,
    "recommended_metrics": honest_metrics if honest_metrics is not None else best_metrics,
    "current_threshold": config.boundary.threshold, "current_threshold_metrics": default_metrics, "skipped": skipped,
}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
