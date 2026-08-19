"""OCR fallback isolated from PDF parsing."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OCRResult:
    text: str
    confidence: float | None
    engine: str


def extract_with_tesseract(image_path: str) -> OCRResult:
    """Run Tesseract only when explicitly enabled by configuration."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OCR requested but pytesseract and Pillow are not installed.") from exc
    data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
    words = [word for word in data["text"] if word.strip()]
    confidences = [float(item) for item in data["conf"] if str(item) not in {"-1", ""}]
    return OCRResult(" ".join(words), sum(confidences) / len(confidences) / 100 if confidences else None, "tesseract")
