"""Score one processed packet against its ground truth.

The existing metric helpers work on flat aligned label lists across a whole corpus. The UI needs
the same numbers for a single packet the user just uploaded, plus a per-document breakdown to
show beside each split. Without ground truth none of this is computable, which is exactly why
the UI reports model confidence and measured benchmarks separately from these.
"""
from __future__ import annotations

from typing import Any

from src.evaluation.stage1_metrics import boundary_lift, grouping_accuracy


def _page_owner(documents: list[dict[str, Any]], page_count: int, fallback_prefix: str) -> list[str]:
    """page number -> owning document id, indexed from 0."""
    owners = [""] * page_count
    for position, document in enumerate(documents):
        identifier = str(document.get("doc_id") or f"{fallback_prefix}_{position}")
        for page in document.get("pages", []):
            if 1 <= int(page) <= page_count:
                owners[int(page) - 1] = identifier
    # A page no document claimed still needs a distinct owner or it would merge with its
    # neighbour and silently inflate grouping accuracy.
    for index, owner in enumerate(owners):
        if not owner:
            owners[index] = f"{fallback_prefix}_unassigned_{index}"
    return owners


def _boundaries(owners: list[str]) -> list[int]:
    return [int(owners[i] != owners[i - 1]) for i in range(1, len(owners))]


def _macro_prf(predicted: list[str], actual: list[str]) -> dict[str, Any]:
    """Per-class and macro-averaged precision/recall/F1 over document types."""
    labels = sorted(set(predicted) | set(actual))
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(p == label and a == label for p, a in zip(predicted, actual))
        fp = sum(p == label and a != label for p, a in zip(predicted, actual))
        fn = sum(p != label and a == label for p, a in zip(predicted, actual))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                            "f1": round(f1, 4), "support": sum(a == label for a in actual)}
    scored = [v for k, v in per_class.items() if v["support"] or k in predicted]
    count = max(len(scored), 1)
    return {
        "accuracy": round(sum(p == a for p, a in zip(predicted, actual)) / max(len(actual), 1), 4),
        "macro_precision": round(sum(v["precision"] for v in scored) / count, 4),
        "macro_recall": round(sum(v["recall"] for v in scored) / count, 4),
        "macro_f1": round(sum(v["f1"] for v in scored) / count, 4),
        "per_class": per_class,
    }


def evaluate_packet(predicted: list[dict[str, Any]], truth: list[dict[str, Any]],
                    page_count: int) -> dict[str, Any]:
    """Boundary, grouping and classification metrics for a single packet."""
    predicted_owners = _page_owner(predicted, page_count, "pred")
    truth_owners = _page_owner(truth, page_count, "true")

    report: dict[str, Any] = {
        "page_count": page_count,
        "documents_predicted": len(predicted),
        "documents_actual": len(truth),
        "boundary": boundary_lift(_boundaries(predicted_owners), _boundaries(truth_owners)),
        "page_grouping_accuracy": round(grouping_accuracy(predicted_owners, truth_owners), 4),
    }

    # Align each true document to the predicted document sharing the most pages, so document
    # type is scored even when the split itself was imperfect.
    comparison: list[dict[str, Any]] = []
    for position, actual_document in enumerate(truth):
        actual_pages = {int(p) for p in actual_document.get("pages", [])}
        best, best_overlap = None, 0
        for candidate in predicted:
            overlap = len(actual_pages & {int(p) for p in candidate.get("pages", [])})
            if overlap > best_overlap:
                best, best_overlap = candidate, overlap
        predicted_type = (best or {}).get("doc_type", "unmatched")
        actual_type = actual_document.get("doc_type", "unknown")
        comparison.append({
            "index": position + 1,
            "actual_pages": sorted(actual_pages),
            "predicted_pages": sorted(int(p) for p in (best or {}).get("pages", [])),
            "actual_type": actual_type,
            "predicted_type": predicted_type,
            "confidence": round(float((best or {}).get("classification_confidence", 0.0)), 4),
            "type_correct": predicted_type == actual_type,
            "pages_exact": sorted(actual_pages) == sorted(int(p) for p in (best or {}).get("pages", [])),
        })

    if comparison:
        report["classification"] = _macro_prf([c["predicted_type"] for c in comparison],
                                              [c["actual_type"] for c in comparison])
        report["documents_exactly_split"] = sum(c["pages_exact"] for c in comparison)
    report["documents"] = comparison
    return report
