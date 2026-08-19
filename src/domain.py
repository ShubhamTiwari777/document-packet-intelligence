"""Stable, JSON-serializable domain structures shared across stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class TextBlock:
    text: str
    bbox: list[float]
    font_sizes: list[float] = field(default_factory=list)
    font_names: list[str] = field(default_factory=list)
    flags: list[int] = field(default_factory=list)


@dataclass
class PageRepresentation:
    page_number: int
    text: str
    blocks: list[TextBlock]
    fonts: list[dict[str, Any]]
    width: float
    height: float
    image_path: str | None = None
    image_count: int = 0
    extraction_method: str = "pymupdf"
    ocr_used: bool = False
    extraction_confidence: float | None = None


@dataclass
class DocumentGroup:
    doc_id: str
    pages: list[int]
    doc_type: str = "unknown"
    classification_confidence: float = 0.0
    boundary_confidences: list[float] = field(default_factory=list)
    classification_source: str = "unknown"
    classification_alternatives: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Element:
    """A leaf of the document structure.

    `rows`/`headers` are set for tables, `items`/`ordered` for lists, `label` for captions.
    """
    type: str
    text: str
    page: int
    bbox: list[float]
    rows: list[list[str]] | None = None
    items: list[str] | None = None
    headers: list[str] | None = None
    ordered: bool | None = None
    label: str | None = None
    element_id: str | None = None


@dataclass
class Section:
    section_id: str
    title: str
    level: int
    content_md: str
    page_refs: list[int]
    elements: list[Element]
    parent_section_id: str | None = None
    breadcrumb: list[str] = field(default_factory=list)


@dataclass
class StructuredDocument:
    doc_id: str
    doc_type: str
    pages: list[int]
    metadata: dict[str, Any]
    sections: list[Section]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    section_id: str
    page_refs: list[int]
    text: str
    breadcrumb: list[str] = field(default_factory=list)
    element_types: list[str] = field(default_factory=list)
    token_count: int = 0


@dataclass
class EvidenceResult:
    evidence: str
    doc_id: str
    page_ref: list[int]
    rrf_score: float
    reranker_score: float | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    # Raw RRF scores sit around 0.016-0.033 and are not interpretable on their own; `confidence`
    # is normalised across the returned candidates so callers can threshold it.
    confidence: float = 0.0
    chunk_id: str = ""
    section_id: str = ""
    doc_type: str = ""
    breadcrumb: list[str] = field(default_factory=list)
    element_types: list[str] = field(default_factory=list)


def dump_json(value: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serializable = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    Path(path).write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
