"""Layout-to-structure conversion: headings, hierarchy, lists, tables and page references.

Pipeline per document:
  1. detect running headers/footers across its pages and exclude them from the body;
  2. extract tables (pdfplumber) and record their regions;
  3. walk blocks in reading order, suppressing anything inside a table region;
  4. classify each remaining block as heading / list / paragraph;
  5. open a new section at each heading, nesting it under the closest heading of a lower level
     so the result is a real tree rather than a flat list.
"""
from __future__ import annotations

from dataclasses import replace
from statistics import median
from typing import Any

from src.domain import DocumentGroup, Element, PageRepresentation, Section, StructuredDocument
from src.stage2.boilerplate import detect_boilerplate, normalize
from src.stage2.heading_detector import (
    MAX_LEVELS, collect_heading_sizes, heading_levels, is_heading, is_text_heading,
    numbering_level, text_heading_level,
)
from src.stage2.caption_detector import caption_label
from src.stage2.list_detector import is_ordered_block, is_ordered_item, match_item, split_list_items
from src.stage2.schema import DocumentMetadata
from src.stage2.table_extractor import candidate_regions, extract_tables_for_pages, inside, looks_like_table
from src.stage2.text_structure import synthesize_blocks
from src.stage2.type_specific import extract_type_fields


def _page_medians(pages: list[PageRepresentation]) -> dict[int, float]:
    medians: dict[int, float] = {}
    for page in pages:
        sizes = [font.get("size", 0.0) for font in page.fonts]
        medians[page.page_number] = float(median(sizes)) if sizes else 0.0
    return medians


def _document_title(pages: list[PageRepresentation]) -> str:
    """Largest-font line on the document's first page -- the masthead/letterhead."""
    if not pages or not pages[0].blocks:
        return ""
    best = max(pages[0].blocks, key=lambda block: (max(block.font_sizes, default=0.0), -block.bbox[1] if len(block.bbox) == 4 else 0))
    return best.text.strip()


def _element_for(block: Any, page_number: int) -> Element:
    items = split_list_items(block.text)
    if items:
        ordered = is_ordered_block(block.text)
        marker = (lambda index: f"{index}. ") if ordered else (lambda index: "- ")
        text = "\n".join(f"{marker(index)}{item}" for index, item in enumerate(items, 1))
        return Element(type="list", text=text, page=page_number, bbox=block.bbox, items=items, ordered=ordered)
    label = caption_label(block.text)
    if label:
        return Element(type="caption", text=block.text.strip(), page=page_number, bbox=block.bbox, label=label)
    return Element(type="paragraph", text=block.text.strip(), page=page_number, bbox=block.bbox)


def structure_document(
    group: DocumentGroup,
    pages: list[PageRepresentation],
    source_file: str,
    pdf_path: str | None = None,
) -> StructuredDocument:
    selected = [page for page in pages if page.page_number in set(group.pages)]

    # OCR-only sources arrive with text but no blocks/fonts; rebuild pseudo-blocks so the same
    # downstream path applies, and switch to convention-based heading detection.
    text_only = any(page.text.strip() and not page.blocks for page in selected)
    if text_only:
        # `replace` rather than assignment: the caller's PageRepresentation objects are shared
        # across the packet and must not gain synthesized blocks as a side effect.
        selected = [page if page.blocks else replace(page, blocks=synthesize_blocks(page)) for page in selected]

    medians = _page_medians(selected)
    boilerplate = detect_boilerplate(selected)
    levels = {} if text_only else heading_levels(collect_heading_sizes(selected, medians, boilerplate, normalize))

    def heading_test(block, page_number: int) -> bool:
        return is_text_heading(block.text) if text_only else is_heading(block, medians.get(page_number, 0.0))

    # Tables first: their regions suppress duplicate paragraph text below.
    tables_by_page: dict[int, list] = {}
    if pdf_path:
        candidates = [page.page_number for page in selected if looks_like_table(page)]
        if candidates:
            regions = {page.page_number: candidate_regions(page) for page in selected if page.page_number in set(candidates)}
            tables_by_page = extract_tables_for_pages(pdf_path, candidates, regions)

    sections: list[Section] = []
    # A Section is created the moment its heading is seen, not when content arrives. A heading
    # whose body lives entirely in subsections ("Results" above "4.1 ...") previously produced no
    # Section at all, which both lost the heading and left children pointing at a
    # parent_section_id that existed nowhere in the output.
    counter = 0
    current_section: Section | None = None
    # Stack of (level, section_id, title) tracking the open heading path for breadcrumbs.
    open_path: list[tuple[int, str, str]] = []
    pending: list[Element] = []

    def open_section(section_id: str, title: str, level: int, parent: str | None, breadcrumb: list[str]) -> Section:
        nonlocal current_section
        current_section = Section(
            section_id=section_id, title=title, level=level, content_md="",
            page_refs=[], elements=[], parent_section_id=parent, breadcrumb=list(breadcrumb),
        )
        sections.append(current_section)
        return current_section
    # Consecutive single-item blocks are stitched back into one list element.
    list_run: list[tuple[str, int, list[float], bool]] = []

    def flush_list() -> None:
        nonlocal list_run
        if not list_run:
            return
        if len(list_run) >= 2:
            items = [item for item, _, _, _ in list_run]
            ordered = sum(1 for _, _, _, flag in list_run if flag) >= len(list_run) / 2
            marker = (lambda index: f"{index}. ") if ordered else (lambda index: "- ")
            pending.append(Element(
                type="list", text="\n".join(f"{marker(i)}{item}" for i, item in enumerate(items, 1)),
                page=list_run[0][1], bbox=list_run[0][2], items=items, ordered=ordered,
            ))
        else:  # a lone marker line is prose, not a list
            item, page_number, bbox, _ = list_run[0]
            pending.append(Element(type="paragraph", text=item, page=page_number, bbox=bbox))
        list_run = []

    def flush() -> None:
        nonlocal pending
        flush_list()
        if not pending:
            return
        section = current_section or open_section(f"{group.doc_id}_s00", "Document", 1, None, [])
        section.elements.extend(pending)
        for index, element in enumerate(section.elements, 1):
            element.element_id = f"{section.section_id}_e{index:02d}"
            if element.type == "table" and element.rows and element.headers is None:
                element.headers = element.rows[0]
        section.page_refs = sorted({element.page for element in section.elements})
        section.content_md = "\n\n".join(element.text for element in section.elements if element.text)
        pending = []

    for page in selected:
        page_tables = tables_by_page.get(page.page_number, [])
        # Interleave blocks and tables by vertical position. Appending tables after the page's
        # blocks attached every table to whichever section happened to be open at the end of the
        # page, so a table could be filed under a later heading than the one introducing it.
        ordered: list[tuple[float, int, Any]] = []
        for block in page.blocks:
            if not block.text.strip() or normalize(block.text) in boilerplate:
                continue
            if any(inside(block.bbox, table.bbox) for table in page_tables):
                continue  # already represented as a structured table
            ordered.append((block.bbox[1] if len(block.bbox) == 4 else 0.0, 0, block))
        for table in page_tables:
            ordered.append((table.bbox[1], 1, table))
        ordered.sort(key=lambda entry: (entry[0], entry[1]))

        for _, kind, payload in ordered:
            if kind == 1:
                flush_list()
                pending.append(payload.element)
                continue
            block = payload
            if heading_test(block, page.page_number):
                flush()  # also flushes any open list run
                title = block.text.strip()
                if text_only:
                    level = text_heading_level(title)
                else:
                    # An explicit section number is stronger evidence of depth than font size:
                    # "3.3 Example" is level 2 even when set in the same face as a level-1 heading.
                    size = round(max(block.font_sizes, default=0.0), 1)
                    level = numbering_level(title) or levels.get(size, min(len(levels) + 1, MAX_LEVELS)) or 1
                while open_path and open_path[-1][0] >= level:
                    open_path.pop()
                counter += 1
                section_id = f"{group.doc_id}_s{counter:02d}"
                open_section(section_id, title, level,
                             open_path[-1][1] if open_path else None,
                             [entry[2] for entry in open_path] + [title])
                open_path.append((level, section_id, title))
            else:
                item = match_item(block.text)
                if item is not None:
                    list_run.append((item, page.page_number, block.bbox, is_ordered_item(block.text)))
                    continue
                flush_list()
                pending.append(_element_for(block, page.page_number))
        flush_list()
    flush()

    if not sections:
        body = "\n\n".join(page.text for page in selected if page.text.strip())
        sections = [Section(section_id=f"{group.doc_id}_s01", title="Document", level=1, content_md=body, page_refs=list(group.pages), elements=[], breadcrumb=["Document"])]

    element_counts: dict[str, int] = {}
    for section in sections:
        for element in section.elements:
            element_counts[element.type] = element_counts.get(element.type, 0) + 1

    full_text = "\n".join(section.content_md for section in sections)
    metadata = DocumentMetadata(
        source_file=source_file, document_title=_document_title(selected),
        packet_page_range=list(group.pages), page_count=len(selected),
        extraction_method=",".join(sorted({page.extraction_method for page in selected})),
        ocr_used=any(page.ocr_used for page in selected), doc_type=group.doc_type,
        classification_confidence=group.classification_confidence,
        classification_source=group.classification_source,
        section_count=len(sections), element_counts=element_counts,
        has_tables=element_counts.get("table", 0) > 0,
        boilerplate_lines=sorted(boilerplate),
        type_specific_fields=extract_type_fields(group.doc_type, full_text),
    )
    return StructuredDocument(doc_id=group.doc_id, doc_type=group.doc_type, pages=list(group.pages), metadata=metadata.to_dict(), sections=sections)
