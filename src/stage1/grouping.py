"""Deterministic grouping from adjacent-boundary decisions."""
from __future__ import annotations

from src.domain import DocumentGroup


def group_pages(packet_id: str, page_numbers: list[int], boundary_probabilities: list[float], threshold: float) -> list[DocumentGroup]:
    """Start a group at page one and split before pages above the threshold."""
    if not page_numbers:
        return []
    if len(boundary_probabilities) != len(page_numbers) - 1:
        raise ValueError("Expected one boundary probability for each adjacent page pair.")
    groups: list[DocumentGroup] = []
    current = [page_numbers[0]]
    confidences: list[float] = []
    for page_number, probability in zip(page_numbers[1:], boundary_probabilities):
        if probability >= threshold:
            groups.append(DocumentGroup(doc_id=f"{packet_id}_doc_{len(groups)+1:03d}", pages=current, boundary_confidences=confidences))
            current, confidences = [page_number], []
        else:
            current.append(page_number); confidences.append(probability)
    groups.append(DocumentGroup(doc_id=f"{packet_id}_doc_{len(groups)+1:03d}", pages=current, boundary_confidences=confidences))
    return groups
