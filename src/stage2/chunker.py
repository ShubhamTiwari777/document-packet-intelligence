"""Structure-aware chunking.

The previous chunker split `section.content_md` -- a flat string -- on a fixed token window.
Three consequences mattered for retrieval:

* tables were sliced mid-row, destroying the row/column relationship that makes them useful;
* every chunk inherited the *section's* whole page range, so a chunk physically on page 5 could
  claim `page_refs: [4, 5]`, weakening the page citation Stage 3 must return;
* chunks carried no heading context, so a chunk reading "1 45,000 45,000" was unretrievable.

This version packs whole elements, keeps tables intact, tracks the pages actually contributing
to each chunk, and prefixes the heading breadcrumb so a chunk carries its own context.
"""
from __future__ import annotations

from src.config import ChunkingConfig
from src.domain import Chunk, Element, StructuredDocument


def _tokens(text: str) -> int:
    return len(text.split())


def _split_long_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Sliding window used only when a single element exceeds the budget on its own."""
    words = text.split()
    if len(words) <= max_tokens:
        return [text] if words else []
    stride = max(1, max_tokens - max(0, overlap))
    parts: list[str] = []
    start = 0
    while start < len(words):
        parts.append(" ".join(words[start:start + max_tokens]))
        if start + max_tokens >= len(words):
            break
        start += stride
    return parts


def chunk_document(document: StructuredDocument, config: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in document.sections:
        elements = section.elements or [Element(type="paragraph", text=section.content_md, page=(section.page_refs or document.pages or [1])[0], bbox=[])]
        prefix = " > ".join(section.breadcrumb) if section.breadcrumb else section.title
        part = 1
        buffer: list[Element] = []

        def emit() -> None:
            nonlocal buffer, part
            if not buffer:
                return
            body = "\n\n".join(element.text for element in buffer if element.text.strip())
            if not body.strip():
                buffer = []
                return
            text = f"{prefix}\n\n{body}" if prefix else body
            chunks.append(Chunk(
                chunk_id=f"{section.section_id}_c{part:02d}", doc_id=document.doc_id,
                doc_type=document.doc_type, section_id=section.section_id,
                page_refs=sorted({element.page for element in buffer}), text=text,
                breadcrumb=list(section.breadcrumb),
                element_types=sorted({element.type for element in buffer}),
                token_count=_tokens(text),
            ))
            part += 1
            buffer = []

        for element in elements:
            if not element.text.strip():
                continue
            size = _tokens(element.text)
            # A table is an atomic unit: splitting it separates rows from their header.
            if element.type == "table":
                emit()
                buffer = [element]
                emit()
                continue
            if size > config.max_tokens:
                emit()
                for piece in _split_long_text(element.text, config.max_tokens, config.overlap_tokens):
                    buffer = [Element(type=element.type, text=piece, page=element.page, bbox=element.bbox)]
                    emit()
                continue
            if sum(_tokens(item.text) for item in buffer) + size > config.max_tokens:
                emit()
            buffer.append(element)
        emit()
    return chunks
