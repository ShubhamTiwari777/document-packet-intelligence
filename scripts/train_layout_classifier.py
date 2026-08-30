"""Train the document-type classifier with character n-grams and shape features, under a
select-on-validation protocol.

Three variants are fitted on an identical split so the contribution of each change is separable:

  word only            -- the shipped baseline: word 1-2 gram TF-IDF
  word + char          -- adds character 3-5 grams, which survive OCR misreads that put a word
                          out of vocabulary entirely
  word + char + shape  -- adds 21 geometry descriptors: column count, margin regularity, where the
                          largest text sits, density by page third

The data is split three ways, not two. Comparing variants on the same set that reports the final
number is selection on the evaluation set, which this project has already been burned by twice:
a boundary threshold that won validation by +0.068 and lost test by -0.027, and a ceil() rule that
led validation by +0.0165 and did not transfer. Variants are compared on validation, the winner is
frozen, and test is scored once. Macro-F1 is the selection metric, fixed here in advance, because
accuracy hides the rare classes that are exactly where these features should help.

Trained from a pages-with-boxes manifest rather than the text-only one: shape needs the boxes, and
deriving the text from the same records guarantees the two views describe the same document. Pages
are rebuilt into blocks with the same routine the pipeline uses at inference.
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
from src.features.document_shape import SHAPE_FEATURE_NAMES, document_shape_features, shape_vector
from src.stage1.document_classifier import LayoutTextDocumentClassifier, TfidfDocumentClassifier
from src.synthetic_packets import blocks_from_words

SELECTION_METRIC = "macro_f1"      # pre-committed, before any variant is fitted

parser = argparse.ArgumentParser()
parser.add_argument("--pages", default="data/raw/rvlcdip/pages_boxes_large.json")
parser.add_argument("--output", default="models/document_classifier/layout_text.pkl")
parser.add_argument("--val", type=float, default=0.2)
parser.add_argument("--test", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--min-words", type=int, default=10)
parser.add_argument("--metrics-output", default="outputs/benchmarks/classification_ablation.json")
args = parser.parse_args()

manifest = json.loads(Path(args.pages).read_text(encoding="utf-8"))
records = manifest["pages"] if isinstance(manifest, dict) else manifest

samples = []
for record in records:
    words, boxes = record.get("words") or [], record.get("boxes") or []
    if len(words) < args.min_words or len(words) != len(boxes):
        continue
    width = float(record.get("width") or 1000)
    height = float(record.get("height") or 1000)
    page = PageRepresentation(page_number=1, text=" ".join(words),
                              blocks=blocks_from_words(words, boxes, width, height),
                              fonts=[], width=width, height=height)
    samples.append({"text": page.text, "label": record["label"],
                    "shape": shape_vector(document_shape_features([page]))})

if len(samples) < 300:
    raise SystemExit(f"Only {len(samples)} usable pages; fetch more with scripts/fetch_rvlcdip_pages.py")

random.Random(args.seed).shuffle(samples)
n = len(samples)
test_cut = int(n * (1 - args.test))
val_cut = int(test_cut * (1 - args.val / (1 - args.test)))
train, val, test = samples[:val_cut], samples[val_cut:test_cut], samples[test_cut:]
print(f"{n} usable pages, {len(Counter(s['label'] for s in samples))} classes")
print(f"  {len(train)} train / {len(val)} validation / {len(test)} test")
print(f"  selection metric: {SELECTION_METRIC} (fixed before fitting)\n")


def metrics(actual: list[str], predicted: list[str]) -> dict:
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
            "macro_f1": round(sum(f1s) / max(len(f1s), 1), 4), "per_class": per_class}


def fit_word_only(rows):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed))])
    pipeline.fit([r["text"] for r in rows], [r["label"] for r in rows])
    return ("word_only", pipeline)


def fit_word_char(rows):
    return ("word_char", TfidfDocumentClassifier.train(
        [r["text"] for r in rows], [r["label"] for r in rows], args.seed))


def fit_word_char_shape(rows):
    return ("word_char_shape", LayoutTextDocumentClassifier.train(
        [r["text"] for r in rows], [r["shape"] for r in rows], [r["label"] for r in rows], args.seed))


def predict(kind, model, rows) -> list[str]:
    if kind == "word_only":
        return list(model.predict([r["text"] for r in rows]))
    if kind == "word_char":
        return [p.label for p in model.predict([r["text"] for r in rows])]
    return [p.label for p in model.predict([r["text"] for r in rows], [r["shape"] for r in rows])]


NAMES = {"word_only": "word only (baseline)", "word_char": "word + char",
         "word_char_shape": "word + char + shape"}

fitted = [fit_word_only(train), fit_word_char(train), fit_word_char_shape(train)]

print("=== PHASE 1  selection on VALIDATION ===")
val_actual = [r["label"] for r in val]
validation = {}
for kind, model in fitted:
    validation[kind] = metrics(val_actual, predict(kind, model, val))
    print(f"  {NAMES[kind]:<24} accuracy {validation[kind]['accuracy']:.4f}   "
          f"macro-F1 {validation[kind]['macro_f1']:.4f}")

winner = max(validation, key=lambda k: validation[k][SELECTION_METRIC])
print(f"\n  >>> SELECTED on {SELECTION_METRIC}: {NAMES[winner]}")
print("  >>> frozen; the test split has not been scored yet")

print("\n=== PHASE 2  single evaluation on TEST ===")
test_actual = [r["label"] for r in test]
predictions, testing = {}, {}
for kind, model in fitted:
    predictions[kind] = predict(kind, model, test)
    testing[kind] = metrics(test_actual, predictions[kind])
    mark = "  <-- selected" if kind == winner else ""
    print(f"  {NAMES[kind]:<24} accuracy {testing[kind]['accuracy']:.4f}   "
          f"macro-F1 {testing[kind]['macro_f1']:.4f}{mark}")

# McNemar between the two strongest: is the difference real, or a handful of documents?
ranked = sorted(testing, key=lambda k: testing[k][SELECTION_METRIC], reverse=True)[:2]
a, b = ranked
only_a = sum(1 for p, q, t in zip(predictions[a], predictions[b], test_actual) if p == t and q != t)
only_b = sum(1 for p, q, t in zip(predictions[a], predictions[b], test_actual) if p != t and q == t)
statistic = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b) if (only_a + only_b) else 0.0
try:
    from scipy.stats import chi2
    p_value = float(1 - chi2.cdf(statistic, 1))
except Exception:
    p_value = float("nan")
print(f"\n  McNemar, {NAMES[a]} vs {NAMES[b]} on test:")
print(f"    {a} right / {b} wrong: {only_a}     {b} right / {a} wrong: {only_b}")
print(f"    chi2 {statistic:.3f}, p {p_value:.4f}  "
      f"({'significant' if p_value < 0.05 else 'NOT significant'})")

base = testing["word_only"]
best = testing[winner]
print(f"\n  per-class on test, {NAMES[winner]} (worst first):")
print(f"    {'class':<26}{'prec':>7}{'rec':>7}{'F1':>7}{'n':>6}{'vs base':>9}")
for label, row in sorted(best["per_class"].items(), key=lambda kv: kv[1]["f1"]):
    delta = row["f1"] - base["per_class"].get(label, {}).get("f1", 0.0)
    print(f"    {label:<26}{row['precision']:>7.3f}{row['recall']:>7.3f}{row['f1']:>7.3f}"
          f"{row['support']:>6}{delta:>+9.3f}")

selected_model = dict(fitted)[winner]
if winner == "word_char_shape":
    selected_model.save(args.output)
    print(f"\nsaved -> {args.output}  (requires shape vectors at inference)")
elif winner == "word_char":
    selected_model.save(args.output)
    print(f"\nsaved -> {args.output}  (text only)")

Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
Path(args.metrics_output).write_text(json.dumps({
    "pages_manifest": args.pages, "usable_pages": n,
    "train": len(train), "validation": len(val), "test": len(test),
    "selection_metric": SELECTION_METRIC, "selected": winner,
    "shape_features": SHAPE_FEATURE_NAMES,
    "validation_results": validation, "test_results": testing,
    "mcnemar": {"a": a, "b": b, "only_a": only_a, "only_b": only_b,
                "chi2": round(statistic, 4), "p_value": round(p_value, 4)},
}, indent=2), encoding="utf-8")
print(f"-> {args.metrics_output}")
