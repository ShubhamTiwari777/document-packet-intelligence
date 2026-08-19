"""PyMuPDF-first page extraction with deliberately narrow pdfplumber fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.config import IngestionConfig, RuntimeConfig
from src.domain import PageRepresentation, TextBlock
from src.ingestion.ocr import extract_with_tesseract
from src.ingestion.page_renderer import render_page


class PDFParser:
    """Extract page text, layout, font and image metadata from a PDF packet."""

    def __init__(self, ingestion: IngestionConfig, runtime: RuntimeConfig):
        self.ingestion = ingestion
        self.runtime = runtime

    def parse(self, pdf_path: str | Path, render_dir: str | Path | None = None) -> list[PageRepresentation]:
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required. Install dependencies with: pip install -r requirements.txt") from exc
        packet = fitz.open(str(pdf_path))
        rendered = Path(render_dir) if render_dir else None
        pages: list[PageRepresentation] = []
        try:
            for index, page in enumerate(packet):
                image_path = None
                if rendered:
                    image_path = render_page(page, rendered / f"page_{index + 1:04d}.png", self.runtime.render_dpi, self.runtime.max_render_pixels)
                blocks, fonts = self._extract_blocks_and_fonts(page)
                native_text = page.get_text("text").strip()
                representation = PageRepresentation(
                    page_number=index + 1,
                    text=native_text,
                    blocks=blocks,
                    fonts=fonts,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    image_path=image_path,
                    image_count=len(page.get_images(full=True)),
                    extraction_method="pymupdf",
                )
                if self._needs_ocr(representation) and self.ingestion.enable_ocr and image_path:
                    result = extract_with_tesseract(image_path)
                    representation.text = result.text
                    representation.ocr_used = True
                    representation.extraction_method = "ocr"
                    representation.extraction_confidence = result.confidence
                pages.append(representation)
        finally:
            packet.close()
        return pages

    def _extract_blocks_and_fonts(self, page: object) -> tuple[list[TextBlock], list[dict[str, object]]]:
        """Extract blocks preserving intra-line spacing and line breaks.

        Spans were previously concatenated with `"".join(...)` across BOTH spans and lines,
        which destroyed two things downstream:
          * horizontally separated cells merged into one token ("MetricBaselineTargetUnit"),
            defeating any column-aware table heuristic;
          * lines within a block lost their newline, so list detection -- which splits on
            newlines -- saw a single run-on line and silently found no lists.
        Spans are now joined with a space when their bounding boxes show a real horizontal
        gap, and lines are joined with a newline.
        """
        raw = page.get_text("dict")
        blocks: list[TextBlock] = []
        fonts: list[dict[str, object]] = []
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            line_texts: list[str] = []
            sizes: list[float] = []
            names: list[str] = []
            flags: list[int] = []
            for line in block.get("lines", []):
                parts: list[str] = []
                previous_right: float | None = None
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    size, font, flag = float(span.get("size", 0)), str(span.get("font", "")), int(span.get("flags", 0))
                    sizes.append(size); names.append(font); flags.append(flag)
                    fonts.append({"name": font, "size": size, "flags": flag})
                    bbox = span.get("bbox") or []
                    if parts and text and previous_right is not None and len(bbox) == 4:
                        gap = float(bbox[0]) - previous_right
                        if gap > max(1.0, size * 0.15) and not parts[-1].endswith(" ") and not text.startswith(" "):
                            parts.append(" ")
                    parts.append(text)
                    if len(bbox) == 4:
                        previous_right = float(bbox[2])
                line_text = "".join(parts).strip()
                if line_text:
                    line_texts.append(line_text)
            text = "\n".join(line_texts).strip()
            if text:
                blocks.append(TextBlock(text=text, bbox=[float(x) for x in block.get("bbox", [])], font_sizes=sizes, font_names=names, flags=flags))
        return blocks, fonts

    def _needs_ocr(self, page: PageRepresentation) -> bool:
        density = len(page.text) / max(page.width * page.height, 1.0)
        return len(page.text) < self.ingestion.min_native_characters or density < self.ingestion.min_text_density


def pages_from_json(items: Iterable[dict]) -> list[PageRepresentation]:
    """Build page objects from fixtures/cached JSON without requiring PyMuPDF."""
    return [PageRepresentation(
        page_number=item["page_number"], text=item.get("text", ""),
        blocks=[TextBlock(**block) for block in item.get("blocks", [])], fonts=item.get("fonts", []),
        width=item.get("width", 612), height=item.get("height", 792), image_path=item.get("image_path"),
        image_count=item.get("image_count", 0), extraction_method=item.get("extraction_method", "fixture"),
        ocr_used=item.get("ocr_used", False), extraction_confidence=item.get("extraction_confidence"),
    ) for item in items]
