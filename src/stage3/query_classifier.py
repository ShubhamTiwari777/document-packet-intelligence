"""Cheap query-type detector used only to compare optional adaptive retrieval."""
from __future__ import annotations

import re


def classify_query(query: str) -> str:
    if re.search(r"\b[A-Z]{2,}[A-Z0-9-]*\b|\b\d{4,}\b|[\"']", query): return "identifier_exact"
    if re.search(r"\b(invoice|passport|account|total|number|date|balance|salary)\b", query, re.I): return "entity_heavy"
    return "semantic"
