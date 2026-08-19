"""Markdown rendering of a StructuredDocument.

Storage format decision: JSON is the canonical representation (exact bboxes, page refs,
element types, stable schema for programmatic access), and Markdown is emitted alongside it as
the human- and model-readable view. Markdown is kept because heading levels, lists and tables
survive as literal syntax, which both reviewers and embedding models parse well -- whereas a
JSON blob fed to an embedding model wastes tokens on punctuation and key names.
"""
from __future__ import annotations

from src.domain import StructuredDocument


def render_markdown(document: StructuredDocument) -> str:
    metadata = document.metadata
    lines = [
        f"# {document.doc_type.replace('_', ' ').title()} ({document.doc_id})",
        "",
        f"- **Source**: {metadata.get('source_file', '')}",
        f"- **Pages**: {', '.join(str(page) for page in document.pages)}",
        f"- **Type confidence**: {metadata.get('classification_confidence', 0.0):.3f} "
        f"({metadata.get('classification_source', 'unknown')})",
    ]
    fields = metadata.get("type_specific_fields") or {}
    if fields:
        lines.append(f"- **Extracted fields**: " + ", ".join(f"{key}={value}" for key, value in sorted(fields.items())))
    lines.append("")

    for section in document.sections:
        heading = "#" * min(section.level + 1, 6)
        lines.append(f"{heading} {section.title}")
        page_refs = ", ".join(str(page) for page in section.page_refs)
        if page_refs:
            lines.append(f"*(p. {page_refs})*")
        lines.append("")
        for element in section.elements:
            if element.type == "table":
                lines.extend([element.text, ""])
            elif element.type == "list" and element.items:
                lines.extend([f"- {item}" for item in element.items])
                lines.append("")
            elif element.text.strip():
                lines.extend([element.text.strip(), ""])
        if not section.elements and section.content_md.strip():
            lines.extend([section.content_md.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"
