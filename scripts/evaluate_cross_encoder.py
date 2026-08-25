"""Evaluate the cross-encoder boundary model against the feature model and the trivial baseline.

Reports page grouping accuracy alongside boundary F1, and both against "every page is its own
document" -- at the target's 72% boundary density that baseline scores F1 0.840 and grouping
0.854, so a bare F1 says almost nothing about whether a model is doing useful work.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluation.stage1_metrics import boundary_lift, grouping_accuracy
from src.stage1.grouping import decide_boundaries

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/boundary_cross_encoder")
parser.add_argument("--manifests", nargs="+", default=[
    "data/raw/docsplit_benchmark/manifest.json", "data/raw/openpss/test_full/manifest.json"])
parser.add_argument("--limit-streams", type=int, default=0, help="0 = all")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--output", default="outputs/benchmarks/cross_encoder.json")
args = parser.parse_args()

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

metadata = json.loads((Path(args.model) / "metadata.json").read_text(encoding="utf-8"))
window, max_length = metadata["window_words"], metadata["max_length"]
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model)
model.eval()
config = load_config("config/default.yaml")


def seam(left_text: str, right_text: str) -> tuple[str, str]:
    return (" ".join(left_text.split()[-window:]) or "[leeg]",
            " ".join(right_text.split()[:window]) or "[leeg]")


def probabilities(pairs: list[tuple[str, str]]) -> list[float]:
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), args.batch_size):
            block = pairs[start:start + args.batch_size]
            encoded = tokenizer([a for a, _ in block], [b for _, b in block],
                                truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            scores.extend(torch.softmax(model(**encoded).logits, dim=-1)[:, 1].tolist())
    return scores


def group_accuracy(decisions: list[int], labels: list[int], pages: int) -> float:
    predicted, actual, p, a = [], [], 0, 0
    for index in range(pages):
        if index > 0:
            p += decisions[index - 1]
            a += labels[index - 1]
        predicted.append(f"p{p}")
        actual.append(f"t{a}")
    return grouping_accuracy(predicted, actual)


rows = []
for manifest_path in args.manifests:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    streams = manifest["streams"][: args.limit_streams] if args.limit_streams else manifest["streams"]
    predicted_all, labels_all, accuracies, trivial_accuracies = [], [], [], []
    start = time.perf_counter()
    for stream in streams:
        pages, labels = stream["pages"], stream["boundary_labels"]
        if len(labels) != len(pages) - 1 or not labels:
            continue
        pairs = [seam(pages[i]["text"], pages[i + 1]["text"]) for i in range(len(labels))]
        scores = probabilities(pairs)
        decisions = decide_boundaries(scores, config.boundary.threshold, config.boundary.decision)
        predicted_all += decisions
        labels_all += labels
        accuracies.append(group_accuracy(decisions, labels, len(pages)))
        trivial_accuracies.append(group_accuracy([1] * len(labels), labels, len(pages)))
    elapsed = time.perf_counter() - start
    metrics = boundary_lift(predicted_all, labels_all)
    row = {
        "manifest": manifest_path, "streams": len(accuracies), "pairs": len(labels_all),
        **{k: v for k, v in metrics.items()},
        "grouping_accuracy": round(statistics.mean(accuracies), 4),
        "trivial_grouping_accuracy": round(statistics.mean(trivial_accuracies), 4),
        "grouping_lift": round(statistics.mean(accuracies) - statistics.mean(trivial_accuracies), 4),
        "seconds": round(elapsed, 1),
    }
    rows.append(row)
    name = Path(manifest_path).parent.name
    print(f"{name:<20} P {row['precision']:.4f}  R {row['recall']:.4f}  F1 {row['f1']:.4f}  "
          f"F1-lift {row['lift_over_trivial']:+.4f}")
    print(f"{'':<20} grouping {row['grouping_accuracy']:.4f}  vs trivial {row['trivial_grouping_accuracy']:.4f}  "
          f"--> LIFT {row['grouping_lift']:+.4f}   ({row['seconds']}s)")

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps({"model": args.model, "metadata": metadata, "results": rows}, indent=2), encoding="utf-8")
print(f"\nWrote {args.output}")
