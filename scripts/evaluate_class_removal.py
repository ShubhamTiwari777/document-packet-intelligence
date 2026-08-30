"""Measure what removing a class from the taxonomy actually buys.

Dropping the worst class raises macro-F1 arithmetically whatever else happens, so that number on
its own proves nothing. The question worth answering is whether the removed class was also acting
as a *sink* -- absorbing predictions that belonged to classes we keep. In RVL-CDIP, `file_folder`
images are photographs of folder tabs carrying almost no text, and the shipped model reaches
precision 0.311 on it: two of every three `file_folder` predictions are wrong, and each of those
is a document stolen from a real class.

Both models are therefore scored on the *same* documents -- the test split with the dropped class
removed -- so the comparison isolates the effect on the classes that remain rather than measuring
the absence of a hard one.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.domain import PageRepresentation
from src.stage1.document_classifier import TfidfDocumentClassifier
from src.synthetic_packets import blocks_from_words

parser = argparse.ArgumentParser()
parser.add_argument("--pages", default="data/raw/rvlcdip/pages_boxes_large.json")
parser.add_argument("--drop", default="file_folder")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--min-words", type=int, default=10)
parser.add_argument("--val", type=float, default=0.2)
parser.add_argument("--test", type=float, default=0.2)
parser.add_argument("--output", default="outputs/benchmarks/class_removal.json")
args = parser.parse_args()

records = json.loads(Path(args.pages).read_text(encoding="utf-8"))["pages"]
samples = []
for record in records:
    words, boxes = record.get("words") or [], record.get("boxes") or []
    if len(words) < args.min_words or len(words) != len(boxes):
        continue
    samples.append({"text": " ".join(words), "label": record["label"]})

# Identical split to scripts/train_layout_classifier.py, so "test" is the same documents.
random.Random(args.seed).shuffle(samples)
n = len(samples)
test_cut = int(n * (1 - args.test))
val_cut = int(test_cut * (1 - args.val / (1 - args.test)))
train, test = samples[:val_cut], samples[test_cut:]

kept_train = [s for s in train if s["label"] != args.drop]
kept_test = [s for s in test if s["label"] != args.drop]
print(f"{n} pages -> {len(train)} train / {len(test)} test")
print(f"dropping {args.drop!r}: {len(train)-len(kept_train)} training and "
      f"{len(test)-len(kept_test)} test documents removed")
print(f"both models scored on the same {len(kept_test)} documents\n")


def metrics(actual, predicted, drop):
    labels = sorted(set(actual) | set(predicted))
    per_class, f1s = {}, []
    for label in labels:
        tp = sum(p == label and a == label for p, a in zip(predicted, actual))
        fp = sum(p == label and a != label for p, a in zip(predicted, actual))
        fn = sum(p != label and a == label for p, a in zip(predicted, actual))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(a == label for a in actual)
        if support:
            f1s.append(f1)
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                            "f1": round(f1, 4), "support": support}
    return {"accuracy": round(sum(p == a for p, a in zip(predicted, actual)) / len(actual), 4),
            "macro_f1": round(sum(f1s) / max(len(f1s), 1), 4),
            "predicted_dropped_class": sum(1 for p in predicted if p == drop),
            "per_class": per_class}


actual = [s["label"] for s in kept_test]
results = {}
for name, rows in ((f"full taxonomy (16 classes)", train),
                   (f"without {args.drop} (15 classes)", kept_train)):
    model = TfidfDocumentClassifier.train([r["text"] for r in rows],
                                          [r["label"] for r in rows], args.seed)
    predicted = [p.label for p in model.predict([s["text"] for s in kept_test])]
    results[name] = metrics(actual, predicted, args.drop)
    results[name]["classes"] = len(set(r["label"] for r in rows))
    print(f"  {name:<32} accuracy {results[name]['accuracy']:.4f}   "
          f"macro-F1 {results[name]['macro_f1']:.4f}   "
          f"wrongly called {args.drop}: {results[name]['predicted_dropped_class']}")

names = list(results)
before, after = results[names[0]], results[names[1]]
print(f"\n  effect on the {len(kept_test)} documents that remain:")
print(f"    accuracy {after['accuracy']-before['accuracy']:+.4f}   "
      f"macro-F1 {after['macro_f1']-before['macro_f1']:+.4f}")
print(f"    documents no longer stolen by {args.drop}: "
      f"{before['predicted_dropped_class'] - after['predicted_dropped_class']}")

print(f"\n  per-class change (classes that remain):")
print(f"    {'class':<26}{'F1 before':>11}{'F1 after':>10}{'delta':>9}")
for label in sorted(after["per_class"], key=lambda k: after["per_class"][k]["f1"]):
    if label == args.drop:
        continue
    b = before["per_class"].get(label, {}).get("f1", 0.0)
    a = after["per_class"][label]["f1"]
    print(f"    {label:<26}{b:>11.3f}{a:>10.3f}{a-b:>+9.3f}")

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(
    {"dropped": args.drop, "test_documents": len(kept_test), "results": results}, indent=2),
    encoding="utf-8")
print(f"\n-> {args.output}")
