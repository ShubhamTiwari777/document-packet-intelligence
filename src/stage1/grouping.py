"""Deterministic grouping from adjacent-boundary decisions."""
from __future__ import annotations

from src.domain import DocumentGroup


def decide_boundaries(probabilities: list[float], threshold: float, mode: str = "expected_count") -> list[int]:
    """Turn per-pair probabilities into split decisions.

    `threshold` applies one global cut-off. That is only correct when the deployment corpus has
    roughly the boundary density the threshold was tuned on, and it fails badly otherwise: a
    threshold fitted on OpenPSS (long streams, ~11% of pairs are boundaries) recovers just 6% of
    boundaries on short packets where 72% of pairs are, collapsing 3.5 documents into 1.2.

    `expected_count` instead exploits the isotonic calibration already applied to the model: for
    calibrated probabilities the sum over a packet estimates the expected number of boundaries in
    it. Splitting at the highest-scoring K pairs, K = round(sum(p)), therefore adapts per packet
    with no threshold and no tuning against the target corpus. Measured on the DocSplit benchmark
    this lifts page grouping accuracy from 0.216 to 0.615, at a cost of 0.012 on OpenPSS.
    """
    if not probabilities:
        return []
    if mode == "threshold":
        return [int(probability >= threshold) for probability in probabilities]
    if mode != "expected_count":
        raise ValueError(f"Unknown boundary decision mode: {mode}")
    count = max(0, min(len(probabilities), round(sum(probabilities))))
    ranked = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)[:count]
    chosen = set(ranked)
    return [int(index in chosen) for index in range(len(probabilities))]


def group_pages(packet_id: str, page_numbers: list[int], boundary_probabilities: list[float], threshold: float, mode: str = "expected_count") -> list[DocumentGroup]:
    """Start a group at page one and split before each page marked a boundary."""
    if not page_numbers:
        return []
    if len(boundary_probabilities) != len(page_numbers) - 1:
        raise ValueError("Expected one boundary probability for each adjacent page pair.")
    decisions = decide_boundaries(boundary_probabilities, threshold, mode)
    groups: list[DocumentGroup] = []
    current = [page_numbers[0]]
    confidences: list[float] = []
    for page_number, probability, is_boundary in zip(page_numbers[1:], boundary_probabilities, decisions):
        if is_boundary:
            groups.append(DocumentGroup(doc_id=f"{packet_id}_doc_{len(groups)+1:03d}", pages=current, boundary_confidences=confidences))
            current, confidences = [page_number], []
        else:
            current.append(page_number); confidences.append(probability)
    groups.append(DocumentGroup(doc_id=f"{packet_id}_doc_{len(groups)+1:03d}", pages=current, boundary_confidences=confidences))
    return groups
