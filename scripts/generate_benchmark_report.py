"""Generate the "Document Intelligence Evaluation Report" Stage 2 benchmark PDF + ground truth.

Reproduces the structure of the report used to test the API: mixed unnumbered and numbered
headings, bullet and numbered lists, one ruled and one borderless table, a figure caption, and
a running header/footer on every page.

The two table styles are deliberate: a ruled table exercises pdfplumber's line strategy and a
borderless one exercises the text strategy, which is the case that silently produced flattened
"MetricBaselineTargetUnit" output.

NOTE: the numbers inside the "Example Measurement Table" are *targets authored into the
document*, not measured results. Benchmark reporting must never confuse the two.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

TITLE_SIZE, H1, H2, BODY, FOOT = 16.0, 13.0, 11.0, 9.5, 8.0
LEFT = 72.0
DOC_TITLE = "Document Intelligence Evaluation Report"


class Writer:
    def __init__(self, doc):
        self.doc = doc
        self.page = None
        self.y = 0.0
        self.page_number = 0
        self.new_page()

    def new_page(self):
        self.page = self.doc.new_page(width=612, height=792)
        self.page_number += 1
        # Running header + footer (boilerplate on every page).
        self.page.insert_text((LEFT, 46), DOC_TITLE, fontsize=FOOT, fontname="helv")
        self.page.insert_text((500, 770), f"Page {self.page_number}", fontsize=FOOT, fontname="helv")
        self.y = 80.0

    def space(self, amount=10.0):
        self.y += amount

    def _ensure(self, needed=40.0):
        if self.y + needed > 740:
            self.new_page()

    def heading(self, text, size=H1):
        self._ensure(50)
        self.space(8)
        self.page.insert_text((LEFT, self.y), text, fontsize=size, fontname="hebo")
        self.y += size + 8

    def para(self, text, size=BODY):
        self._ensure()
        self.page.insert_text((LEFT, self.y), text, fontsize=size, fontname="helv")
        self.y += size + 5

    def bullets(self, items):
        for item in items:
            self._ensure()
            self.page.insert_text((LEFT + 10, self.y), f"• {item}", fontsize=BODY, fontname="helv")
            self.y += BODY + 5

    def numbered(self, items):
        for index, item in enumerate(items, 1):
            self._ensure()
            self.page.insert_text((LEFT + 10, self.y), f"{index}. {item}", fontsize=BODY, fontname="helv")
            self.y += BODY + 5

    def table(self, rows, widths, ruled: bool):
        """Draw a table; `ruled` draws grid lines, otherwise columns are whitespace-aligned."""
        self._ensure(30 + 18 * len(rows))
        self.space(6)
        top = self.y
        row_height = 18.0
        xs = [LEFT]
        for width in widths:
            xs.append(xs[-1] + width)
        if ruled:
            bottom = top + row_height * len(rows)
            for index in range(len(rows) + 1):
                y = top + row_height * index
                self.page.draw_line(fitz.Point(xs[0], y), fitz.Point(xs[-1], y))
            for x in xs:
                self.page.draw_line(fitz.Point(x, top), fitz.Point(x, bottom))
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                self.page.insert_text((xs[c] + 4, top + row_height * r + 12), str(cell),
                                      fontsize=9.0, fontname="hebo" if r == 0 else "helv")
        self.y = top + row_height * len(rows) + 12
        return self.page_number


MEASUREMENT_TABLE = [
    ["Metric", "Baseline", "Target", "Unit"],
    ["Text retention", "90%", ">=95%", "percent"],
    ["Heading recovery", "65%", ">=85%", "percent"],
    ["List recovery", "70%", ">=90%", "percent"],
    ["Table preservation", "60%", ">=90%", "percent"],
    ["Page-reference accuracy", "80%", ">=95%", "percent"],
]

FINDINGS_TABLE = [
    ["Category", "Documents", "Elements", "Retention"],
    ["Headings", "12", "28", "93%"],
    ["Paragraphs", "12", "86", "98%"],
    ["Bullet lists", "12", "16", "100%"],
    ["Numbered lists", "12", "12", "100%"],
    ["Tables", "12", "8", "95%"],
]

CAPTION = "Figure 1. Conceptual flow from source PDF to structured representation and retrieval-ready chunks."


def build(output_path: str | Path) -> tuple[Path, dict]:
    doc = fitz.open()
    writer = Writer(doc)
    truth: dict = {"headings": [], "tables": [], "lists": [], "captions": []}

    writer.page.insert_text((LEFT, writer.y), DOC_TITLE, fontsize=TITLE_SIZE, fontname="hebo")
    writer.y += TITLE_SIZE + 12
    writer.para("Stage 2 Structured Document Extraction Benchmark")

    def heading(text, level, size):
        writer.heading(text, size)
        truth["headings"].append({"title": text, "level": level, "page": writer.page_number})

    heading("Executive Summary", 1, H1)
    writer.para("This report evaluates structured extraction quality across a document packet.")
    heading("Key objectives", 2, H2)
    items = ["Preserve headings and their hierarchy.", "Retain tables as structured rows.", "Keep precise page provenance."]
    writer.bullets(items)
    truth["lists"].append({"page": writer.page_number, "items": len(items), "ordered": False})

    heading("Introduction", 1, H1)
    writer.para("Document packets combine unrelated documents into a single file.")
    heading("1.1 Background", 2, H2)
    writer.para("Page stream segmentation recovers original document boundaries.")
    heading("1.2 Evaluation Principles", 2, H2)
    writer.para("Metrics must be measured rather than asserted.")

    heading("Scope", 1, H1)
    writer.para("The scope covers structuring and retrieval readiness only.")

    heading("Methodology", 1, H1)
    heading("3.1 Processing Steps", 2, H2)
    steps = ["Parse pages and blocks.", "Detect boilerplate.", "Classify headings.", "Extract tables and lists."]
    writer.numbered(steps)
    truth["lists"].append({"page": writer.page_number, "items": len(steps), "ordered": True})
    heading("3.2 Extraction Signals", 2, H2)
    signals = ["Font size and weight.", "Bounding box position.", "Numbering patterns."]
    writer.bullets(signals)
    truth["lists"].append({"page": writer.page_number, "items": len(signals), "ordered": False})

    heading("3.3 Example Measurement Table", 2, H2)
    page_number = writer.table(MEASUREMENT_TABLE, [180, 80, 80, 80], ruled=False)  # borderless
    truth["tables"].append({"page": page_number, "rows": len(MEASUREMENT_TABLE), "columns": 4, "ruled": False})

    heading("Results", 1, H1)
    heading("4.1 Quantitative Findings", 2, H2)
    page_number = writer.table(FINDINGS_TABLE, [140, 100, 90, 90], ruled=True)  # ruled
    truth["tables"].append({"page": page_number, "rows": len(FINDINGS_TABLE), "columns": 4, "ruled": True})
    heading("4.2 Findings", 2, H2)
    writer.para("Structured extraction improved retrieval precision in internal testing.")
    heading("4.3 Figure Caption", 2, H2)
    writer.para(CAPTION)
    truth["captions"].append({"page": writer.page_number, "text": CAPTION})

    heading("Discussion", 1, H1)
    heading("5.1 Important Observations", 2, H2)
    writer.para("Boilerplate suppression materially reduces duplicate retrieval candidates.")
    heading("5.2 Risks", 2, H2)
    writer.para("Borderless tables remain the most fragile extraction case.")
    heading("5.3 Recommendations", 2, H2)
    writer.para("Adopt dual table strategies and validate on held-out documents.")

    heading("Conclusion", 1, H1)
    heading("6.1 Recommendations", 2, H2)
    writer.para("Continue measuring structure quality against annotated ground truth.")

    # ASCII hyphen, not an em-dash: the base-14 PDF fonts used here cannot encode U+2014 and
    # silently substitute a middle dot, which would make this fixture's ground truth disagree
    # with the text its own PDF actually contains.
    heading("Appendix - Expected Structure", 1, H1)
    writer.para("The appendix enumerates the expected structural elements.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    page_count = writer.page_number
    doc.close()

    ground_truth = {
        "packet": path.name, "page_count": page_count, "document_title": DOC_TITLE,
        "note": "Numbers inside the measurement table are TARGETS authored into the PDF, not measured results.",
        "documents": [{
            "pages": list(range(1, page_count + 1)), "doc_type": "report", "title": DOC_TITLE,
            "headings": [h["title"] for h in truth["headings"]],
            "heading_details": truth["headings"],
            "tables": truth["tables"], "lists": truth["lists"], "captions": truth["captions"],
            "fields": {},
        }],
    }
    (path.parent / "benchmark_ground_truth.json").write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    return path, ground_truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/samples/benchmark_report.pdf")
    args = parser.parse_args()
    path, truth = build(args.output)
    document = truth["documents"][0]
    print(f"Wrote {path} ({truth['page_count']} pages)")
    print(f"  headings={len(document['headings'])} tables={len(document['tables'])} "
          f"lists={len(document['lists'])} captions={len(document['captions'])}")
