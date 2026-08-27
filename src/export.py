"""Write each detected document out as its own PDF.

Stage 1 decides where a packet splits, but until now that decision only ever existed as page
numbers in JSON. Producing real PDFs is what makes the split usable outside this repo, and it
is the artifact the UI hands back to the user.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import re
import zipfile

try:  # PyMuPDF renamed its import; support both.
    import pymupdf  # type: ignore
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore


def slugify(value: str) -> str:
    """Filesystem-safe stem. Document types come from a model, so never trust them as paths."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return cleaned[:60] or "document"


def _page_span(pages: list[int]) -> str:
    return f"p{pages[0]}" if len(pages) == 1 else f"p{pages[0]}-{pages[-1]}"


def split_packet(source_pdf: str | Path, documents: Iterable[dict[str, Any]],
                 output_dir: str | Path) -> list[dict[str, Any]]:
    """Extract each document group into its own PDF; return one manifest entry per file.

    Pages are copied one at a time rather than as a range because a group is a list of page
    numbers, not necessarily a contiguous span -- a mis-split packet can produce gaps, and
    silently emitting the wrong pages would be worse than emitting a short file.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    book = pymupdf.open(str(source_pdf))
    try:
        for index, document in enumerate(documents, start=1):
            pages = sorted(int(p) for p in document.get("pages", []))
            pages = [p for p in pages if 1 <= p <= book.page_count]
            if not pages:
                continue
            doc_type = document.get("doc_type") or "unknown"
            filename = f"{index:02d}_{slugify(doc_type)}_{_page_span(pages)}.pdf"
            destination = target_dir / filename
            part = pymupdf.open()
            try:
                for page_number in pages:
                    part.insert_pdf(book, from_page=page_number - 1, to_page=page_number - 1)
                part.save(str(destination))
            finally:
                part.close()
            manifest.append({
                "index": index,
                "doc_id": document.get("doc_id", f"doc_{index}"),
                "doc_type": doc_type,
                "pages": pages,
                "page_count": len(pages),
                "filename": filename,
                "bytes": destination.stat().st_size,
            })
    finally:
        book.close()
    return manifest


def archive(files_dir: str | Path, archive_path: str | Path) -> Path:
    """Zip every split PDF so the whole packet can be downloaded in one click."""
    source = Path(files_dir)
    destination = Path(archive_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
        for pdf in sorted(source.glob("*.pdf")):
            bundle.write(pdf, arcname=pdf.name)
    return destination
