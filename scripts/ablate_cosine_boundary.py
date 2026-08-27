"""Measure cosine similarity between adjacent page vectors as a standalone boundary detector.

The intuition -- embed each page, split where consecutive pages are dissimilar -- is sound enough
that it is already inside the shipped model: `text_delta` is literally 1 - cosine(left, right),
and header_similarity / footer_similarity / first_line_similarity are three more cosine features.
The open question is whether that one signal is sufficient on its own, which is a measurement,
not an opinion.

The cosine-only variants here are deliberately handed an advantage: the threshold is swept on the
evaluation set itself and the best F1 kept. That is an oracle no deployment could have, so it is
an upper bound on the idea rather than a fair estimate of it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.stage1_metrics import boundary_lift, grouping_accuracy
from src.features.extractor import packet_feature_rows
from src.features.text_features import TfidfTextEmbedder
from src.openpss_dataset import pages_from_stream
from src.stage1.boundary_classifier import SklearnBoundaryModel
from src.stage1.grouping import decide_boundaries

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="data/raw/docsplit_benchmark/manifest.json")
parser.add_argument("--model", default="models/boundary_shortpackets")
parser.add_argument("--output", default="outputs/benchmarks/cosine_boundary_ablation.json")
args = parser.parse_args()

manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
model = SklearnBoundaryModel.load(args.model)
embedder_path = Path(args.model) / "text_embedder.pkl"
embedder = TfidfTextEmbedder.load(embedder_path) if embedder_path.exists() else None
metadata = Path(args.model) / "metadata.json"
synthesize = json.loads(metadata.read_text(encoding="utf-8")).get("synthetic_blocks", False) if metadata.exists() else False

streams: list[dict] = []
for stream in manifest["streams"]:
    try:
        pages = pages_from_stream(stream, synthesize)
        rows = packet_feature_rows(pages, stream["stream_id"], embedder)
        if len(rows) != len(stream["boundary_labels"]):
            continue
        streams.append({
            "labels": list(stream["boundary_labels"]),
            "cosine_distance": [float(row["text_delta"]) for row in rows],
            "model": [float(p) for p in model.predict_proba(rows)],
        })
    except Exception:
        continue

print(f"evaluated {len(streams)} streams, {sum(len(s['labels']) for s in streams)} pairs\n")


def owners(decisions: list[int]) -> list[str]:
    """Page-level document ids implied by a list of split decisions."""
    current, out = 0, ["d0"]
    for decision in decisions:
        current += decision
        out.append(f"d{current}")
    return out


def score(name: str, per_stream_decisions: list[list[int]]) -> dict:
    predicted = [d for decisions in per_stream_decisions for d in decisions]
    actual = [label for stream in streams for label in stream["labels"]]
    metrics = boundary_lift(predicted, actual)
    grouping = sum(grouping_accuracy(owners(decisions), owners(stream["labels"]))
                   for decisions, stream in zip(per_stream_decisions, streams)) / len(streams)
    metrics["grouping_accuracy"] = round(grouping, 4)
    metrics["name"] = name
    print(f"  {name:<44} F1 {metrics['f1']:.3f}  lift {metrics['lift_over_trivial']:+.3f}  "
          f"grouping {grouping:.3f}")
    return metrics


results = []
print("=== cosine similarity between adjacent page vectors, ALONE ===")
best = None
for step in range(1, 100):
    threshold = step / 100
    decisions = [[int(value >= threshold) for value in s["cosine_distance"]] for s in streams]
    predicted = [d for row in decisions for d in row]
    actual = [label for stream in streams for label in stream["labels"]]
    f1 = boundary_lift(predicted, actual)["f1"]
    if best is None or f1 > best[0]:
        best = (f1, threshold, decisions)
results.append(score(f"cosine only, ORACLE threshold {best[1]:.2f}", best[2]))
results.append(score("cosine only, expected-count rule",
                     [decide_boundaries(s["cosine_distance"], 0.5, "expected_count") for s in streams]))

print("\n=== the shipped 21-feature model (cosine is 4 of those features) ===")
results.append(score("21-feature model, expected-count",
                     [decide_boundaries(s["model"], 0.5, "expected_count") for s in streams]))
best_model = None
for step in range(1, 100):
    threshold = step / 100
    decisions = [[int(value >= threshold) for value in s["model"]] for s in streams]
    predicted = [d for row in decisions for d in row]
    actual = [label for stream in streams for label in stream["labels"]]
    f1 = boundary_lift(predicted, actual)["f1"]
    if best_model is None or f1 > best_model[0]:
        best_model = (f1, threshold, decisions)
results.append(score(f"21-feature model, ORACLE threshold {best_model[1]:.2f}", best_model[2]))

print("\n=== baselines ===")
results.append(score("always split (trivial)", [[1] * len(s["labels"]) for s in streams]))
results.append(score("never split", [[0] * len(s["labels"]) for s in streams]))

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"manifest": args.manifest, "model": args.model,
                              "streams": len(streams), "results": results}, indent=2), encoding="utf-8")
print(f"\n-> {output}")
