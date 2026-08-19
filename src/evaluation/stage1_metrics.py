"""Metrics for boundary, grouping and classification."""
from __future__ import annotations


def _prf(predicted: list[int], actual: list[int]) -> dict[str, float]:
    if len(predicted) != len(actual): raise ValueError("Metric inputs must be aligned.")
    tp = sum(p == a == 1 for p, a in zip(predicted, actual)); fp = sum(p == 1 and a == 0 for p, a in zip(predicted, actual)); fn = sum(p == 0 and a == 1 for p, a in zip(predicted, actual))
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def boundary_metrics(predicted: list[int], actual: list[int]) -> dict[str, float]: return _prf(predicted, actual)


def trivial_baseline_f1(actual: list[int]) -> float:
    """F1 of predicting EVERY adjacent pair a boundary: precision = base rate, recall = 1."""
    base = sum(actual) / max(len(actual), 1)
    return 2 * base / (base + 1.0) if base else 0.0


def boundary_lift(predicted: list[int], actual: list[int]) -> dict[str, float]:
    """Boundary metrics plus lift over the trivial all-boundary baseline.

    Raw F1 cannot be compared across corpora with different boundary densities: a dataset where
    32% of pairs are boundaries hands a do-nothing classifier F1 0.49, while one at 11% hands it
    only 0.20. Reporting lift makes "is this model actually learning anything" answerable, and it
    reversed the apparent ranking of two models measured here.
    """
    metrics = _prf(predicted, actual)
    trivial = trivial_baseline_f1(actual)
    metrics["base_rate"] = round(sum(actual) / max(len(actual), 1), 4)
    metrics["trivial_f1"] = round(trivial, 4)
    metrics["lift_over_trivial"] = round(metrics["f1"] - trivial, 4)
    return metrics

def grouping_accuracy(predicted_doc_ids: list[str], actual_doc_ids: list[str]) -> float:
    if len(predicted_doc_ids) != len(actual_doc_ids): raise ValueError("Group IDs must be aligned by page.")
    # Pairwise agreement avoids arbitrary document-label permutation.
    pairs = [(i, j) for i in range(len(actual_doc_ids)) for j in range(i + 1, len(actual_doc_ids))]
    return sum((predicted_doc_ids[i] == predicted_doc_ids[j]) == (actual_doc_ids[i] == actual_doc_ids[j]) for i, j in pairs) / max(len(pairs), 1)


def classification_metrics(predicted: list[str], actual: list[str]) -> dict[str, float]:
    if len(predicted) != len(actual): raise ValueError("Labels must be aligned.")
    labels = sorted(set(predicted) | set(actual)); per_class = []
    for label in labels:
        p = [int(item == label) for item in predicted]; a = [int(item == label) for item in actual]; per_class.append(_prf(p, a)["f1"])
    return {"accuracy": sum(p == a for p, a in zip(predicted, actual)) / max(len(actual), 1), "macro_f1": sum(per_class) / max(len(per_class), 1)}
