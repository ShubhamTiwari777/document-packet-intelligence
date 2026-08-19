"""Metrics for boundary, grouping and classification."""
from __future__ import annotations


def _prf(predicted: list[int], actual: list[int]) -> dict[str, float]:
    if len(predicted) != len(actual): raise ValueError("Metric inputs must be aligned.")
    tp = sum(p == a == 1 for p, a in zip(predicted, actual)); fp = sum(p == 1 and a == 0 for p, a in zip(predicted, actual)); fn = sum(p == 0 and a == 1 for p, a in zip(predicted, actual))
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def boundary_metrics(predicted: list[int], actual: list[int]) -> dict[str, float]: return _prf(predicted, actual)

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
