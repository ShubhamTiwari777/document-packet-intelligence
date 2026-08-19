"""Stage 2 metrics.

Two complementary families, kept separate because they answer different questions:

* **Labeled structure metrics** -- heading/table/list precision, recall and F1 against an
  annotated fixture. These need ground truth, which no public page-stream dataset provides
  (OpenPSS carries boundary labels only), so they are measured on the authored fixture at
  data/samples/ground_truth.json.
* **Unlabeled coverage statistics** -- text retention, boilerplate suppression, chunk size
  distribution. These need no labels and therefore run over any corpus at scale, which is what
  makes them useful as a regression signal on real documents.
"""
from __future__ import annotations

from statistics import mean, median

from src.domain import Chunk, StructuredDocument


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float | None]:
    """Precision/recall/F1, or nulls when the class does not occur at all.

    Reporting 0.0 for "nothing expected and nothing predicted" describes correct behaviour as
    total failure -- the sample packet contains no captions, and scoring that 0.0 would drag an
    honest benchmark table down for a category the fixture never exercised.
    """
    if true_positive + false_positive + false_negative == 0:
        return {"precision": None, "recall": None, "f1": None}
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _normalize(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def heading_metrics(documents: list[StructuredDocument], truth: list[dict]) -> dict[str, float]:
    tp = fp = fn = 0
    for document, expected in zip(documents, truth):
        predicted = {_normalize(section.title) for section in document.sections if _normalize(section.title) != "document"}
        gold = {_normalize(title) for title in expected.get("headings", [])}
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    return _prf(tp, fp, fn)


def element_metrics(documents: list[StructuredDocument], truth: list[dict], element_type: str, truth_key: str) -> dict[str, float]:
    """Count-based precision/recall for tables or lists, matched per page."""
    tp = fp = fn = 0
    for document, expected in zip(documents, truth):
        predicted_pages: list[int] = []
        for section in document.sections:
            predicted_pages += [element.page for element in section.elements if element.type == element_type]
        gold_pages = [entry["page"] for entry in expected.get(truth_key, [])]
        for page in set(predicted_pages) | set(gold_pages):
            predicted_count = predicted_pages.count(page)
            gold_count = gold_pages.count(page)
            tp += min(predicted_count, gold_count)
            fp += max(0, predicted_count - gold_count)
            fn += max(0, gold_count - predicted_count)
    return _prf(tp, fp, fn)


def page_reference_accuracy(documents: list[StructuredDocument], truth: list[dict]) -> dict[str, float]:
    """Share of table/list/caption elements whose page matches the annotation."""
    correct = total = 0
    for document, expected in zip(documents, truth):
        elements = [element for section in document.sections for element in section.elements]
        for key, kind in (("tables", "table"), ("lists", "list"), ("captions", "caption")):
            gold_pages = sorted(entry["page"] for entry in expected.get(key, []))
            found_pages = sorted(element.page for element in elements if element.type == kind)
            total += len(gold_pages)
            # Positional comparison over sorted pages: counts a hit only when an element of the
            # right kind was recovered on the right page.
            remaining = list(found_pages)
            for page in gold_pages:
                if page in remaining:
                    remaining.remove(page)
                    correct += 1
    return {"page_reference_accuracy": round(correct / total, 4) if total else None,
            "page_refs_correct": correct, "page_refs_expected": total}


def field_metrics(documents: list[StructuredDocument], truth: list[dict]) -> dict[str, float]:
    """Exact-match accuracy of type-specific field extraction."""
    correct = total = 0
    for document, expected in zip(documents, truth):
        gold = expected.get("fields", {})
        extracted = document.metadata.get("type_specific_fields", {})
        for key, value in gold.items():
            total += 1
            if _normalize(str(extracted.get(key, ""))) == _normalize(str(value)):
                correct += 1
    return {"field_accuracy": round(correct / total, 4) if total else None, "fields_expected": total, "fields_correct": correct}


def coverage_statistics(documents: list[StructuredDocument], chunks: list[Chunk], page_texts: dict[int, str], elapsed_seconds: float | None = None) -> dict[str, float | None]:
    """Label-free quality signals that can be run over any corpus."""
    sections = [section for document in documents for section in document.sections]
    elements = [element for section in sections for element in section.elements]
    pages = sum(len(document.pages) for document in documents)

    structured_words = sum(len(section.content_md.split()) for section in sections)
    source_words = sum(len(text.split()) for text in page_texts.values())
    boilerplate_lines = sum(len(document.metadata.get("boilerplate_lines", [])) for document in documents)

    token_counts = [chunk.token_count for chunk in chunks] or [0]
    chunks_with_pages = sum(1 for chunk in chunks if chunk.page_refs)
    single_page_chunks = sum(1 for chunk in chunks if len(chunk.page_refs) == 1)

    return {
        "document_count": len(documents),
        "page_count": pages,
        "section_count": len(sections),
        "element_count": len(elements),
        "elements_per_page": round(len(elements) / max(pages, 1), 3),
        # Retention < 1 is expected and desirable: boilerplate is deliberately dropped.
        "text_retention_ratio": round(structured_words / source_words, 4) if source_words else None,
        "boilerplate_lines_suppressed": boilerplate_lines,
        "sections_with_page_refs_ratio": round(sum(1 for s in sections if s.page_refs) / max(len(sections), 1), 4),
        "chunk_count": len(chunks),
        "chunk_tokens_mean": round(mean(token_counts), 2),
        "chunk_tokens_median": round(median(token_counts), 2),
        "chunk_tokens_max": max(token_counts),
        "chunks_with_page_refs_ratio": round(chunks_with_pages / max(len(chunks), 1), 4),
        # High is good: a chunk citing exactly one page gives a precise Stage 3 citation.
        "single_page_chunk_ratio": round(single_page_chunks / max(len(chunks), 1), 4),
        "processing_seconds": round(elapsed_seconds, 4) if elapsed_seconds is not None else None,
        "seconds_per_page": round(elapsed_seconds / max(pages, 1), 4) if elapsed_seconds is not None else None,
    }


# Retained for backwards compatibility with the original benchmark entry point.
def structure_statistics(documents: list[StructuredDocument], elapsed_seconds: float | None = None) -> dict[str, float | None]:
    return coverage_statistics(documents, [], {}, elapsed_seconds)
