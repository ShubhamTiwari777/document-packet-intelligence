"""Page rendering with no side effects beyond the caller-selected directory."""
from __future__ import annotations

from pathlib import Path
import math


def render_page(page: object, output_path: Path, dpi: int, max_pixels: int = 30_000_000) -> str:
    """Render a page while capping pixels to avoid oversized-image memory spikes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_area = max(float(page.rect.width) * float(page.rect.height), 1.0)
    safe_dpi = min(float(dpi), 72 * math.sqrt(max_pixels / page_area))
    pixmap = page.get_pixmap(matrix=__import__("fitz").Matrix(safe_dpi / 72, safe_dpi / 72), alpha=False)
    pixmap.save(str(output_path))
    return str(output_path)
