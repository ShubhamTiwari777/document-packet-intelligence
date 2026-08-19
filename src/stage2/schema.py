"""Explicit Stage 2 metadata schema.

The metadata block was previously an ad-hoc dict assembled inline, which meant downstream
consumers had no contract to code against and no way to tell a missing field from an
unsupported one. This module defines the schema in one place and versions it, so a change to
the structured representation is visible to anything that stored an older version.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass
class DocumentMetadata:
    """Per-document metadata attached to every StructuredDocument."""

    schema_version: str = SCHEMA_VERSION
    source_file: str = ""
    # The document's own title (letterhead/masthead). Captured separately from section headings
    # because it typically repeats on every page and is therefore filtered out of the body as
    # boilerplate -- but it is still the single most identifying line in the document.
    document_title: str = ""
    packet_page_range: list[int] = field(default_factory=list)
    page_count: int = 0

    # Provenance of the text itself, so retrieval can discount OCR-derived content.
    extraction_method: str = ""
    ocr_used: bool = False

    # Stage 1 hand-off: what the classifier decided and how sure it was.
    doc_type: str = "unknown"
    classification_confidence: float = 0.0
    classification_source: str = "unknown"

    # Structure summary, useful both for retrieval filters and for benchmarking.
    section_count: int = 0
    element_counts: dict[str, int] = field(default_factory=dict)
    has_tables: bool = False
    boilerplate_lines: list[str] = field(default_factory=list)

    # Type-conditional extracted fields (invoice number, closing balance, ...).
    type_specific_fields: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
