"""Context assembly: turn ranked evidence into a grounded, citation-carrying block.

This is the hand-off point to a generator in a RAG system, and it is deliberately where this
project stops. The brief states twice that the objective is *not* a chatbot and *not* generated
answers, so no model writes prose here. What the stage does provide is everything a generator
would need and everything that determines whether its output can be trusted:

* **Deduplication** — near-identical chunks waste the context window and bias a generator toward
  whatever happens to be repeated. Packets repeat boilerplate heavily, so this matters more here
  than in a typical corpus.
* **A token budget** — context windows are finite; chunks are admitted in rank order until the
  budget is spent, so the most relevant evidence survives truncation rather than the last-fetched.
* **Inline citations** — every admitted chunk carries `[n]` with its document id and page, so any
  downstream claim is traceable to a page. Retrieval-side citation is what makes a generated
  answer checkable; adding it afterwards cannot work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain import EvidenceResult
from src.features.text_features import tokenize


@dataclass
class AssembledContext:
    context: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    chunks_used: int = 0
    chunks_dropped_duplicate: int = 0
    chunks_dropped_budget: int = 0
    token_count: int = 0


def _overlap(left: str, right: str) -> float:
    """Jaccard overlap on token sets — cheap, and near-duplicates score very high."""
    a, b = set(tokenize(left)), set(tokenize(right))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def assemble_context(
    results: list[EvidenceResult],
    token_budget: int = 1500,
    duplicate_threshold: float = 0.85,
    include_citations: bool = True,
) -> AssembledContext:
    """Assemble ranked evidence into a single grounded context block."""
    selected: list[EvidenceResult] = []
    duplicates = 0
    budget_dropped = 0
    tokens = 0

    for result in results:
        if any(_overlap(result.evidence, chosen.evidence) >= duplicate_threshold for chosen in selected):
            duplicates += 1
            continue
        cost = len(tokenize(result.evidence))
        if tokens + cost > token_budget:
            # Rank order is preserved: once the budget is spent, lower-ranked evidence is dropped
            # rather than truncating a higher-ranked chunk mid-sentence.
            budget_dropped += 1
            continue
        selected.append(result)
        tokens += cost

    blocks: list[str] = []
    citations: list[dict[str, Any]] = []
    for index, result in enumerate(selected, 1):
        pages = ", ".join(str(page) for page in result.page_ref)
        header = f"[{index}] {result.doc_id}" + (f" p.{pages}" if pages else "")
        if result.breadcrumb:
            header += f" — {' > '.join(result.breadcrumb)}"
        blocks.append(f"{header}\n{result.evidence}" if include_citations else result.evidence)
        citations.append({
            "marker": f"[{index}]", "doc_id": result.doc_id, "doc_type": result.doc_type,
            "page_ref": list(result.page_ref), "chunk_id": result.chunk_id,
            "breadcrumb": list(result.breadcrumb), "confidence": result.confidence,
        })

    return AssembledContext(
        context="\n\n".join(blocks), citations=citations, chunks_used=len(selected),
        chunks_dropped_duplicate=duplicates, chunks_dropped_budget=budget_dropped,
        token_count=tokens,
    )
