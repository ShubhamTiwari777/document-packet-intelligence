"""Generate an annotated multi-document PDF packet used as the Stage 1/2 verification fixture.

Emits a 9-page packet containing four independent documents plus a `ground_truth.json`
describing, per document: page range, type, title, section headings, tables (with row/column
counts) and lists (with item counts). Structure here is authored rather than guessed, so it can
be used as labeled ground truth for Stage 2 heading/table/list metrics -- which no public
boundary dataset (OpenPSS carries page labels only) provides.

Tables are drawn with real ruling lines so that a PDF table extractor has to find them the same
way it would in a genuine document.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # PyMuPDF renamed its import; support both.
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

LETTERHEAD = 18
HEADING = 12
BODY = 10
FOOT = 8


def _text(page, x, y, value, size=BODY, bold=False):
    page.insert_text((x, y), value, fontsize=size, fontname="hebo" if bold else "helv")


def _table(page, top, rows, widths, left=72.0, row_height=20.0):
    """Draw a ruled table and return its logical rows."""
    xs = [left]
    for width in widths:
        xs.append(xs[-1] + width)
    bottom = top + row_height * len(rows)
    for index in range(len(rows) + 1):
        y = top + row_height * index
        page.draw_line(fitz.Point(xs[0], y), fitz.Point(xs[-1], y))
    for x in xs:
        page.draw_line(fitz.Point(x, top), fitz.Point(x, bottom))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            _text(page, xs[c] + 4, top + row_height * r + 14, str(cell), size=9, bold=(r == 0))
    return bottom


def _header(page, title, subtitle, page_label):
    _text(page, 72, 60, title, LETTERHEAD, bold=True)
    _text(page, 72, 78, subtitle, BODY)
    _text(page, 500, 790, page_label, FOOT)


def build_packet(output_path: str | Path) -> tuple[Path, dict]:
    doc = fitz.open()
    truth: list[dict] = []

    # ---- Document A: invoice (pages 1-3), with a ruled line-item table and two lists ----
    page = doc.new_page(width=612, height=792)
    _header(page, "INVOICE", "TechNova Solutions Pvt. Ltd.", "Page 1")
    _text(page, 72, 120, "Billing Details", HEADING, bold=True)
    _text(page, 72, 145, "Invoice Number: INV-2026-004471", BODY)
    _text(page, 72, 162, "Bill To: Acme Digital Services", BODY)
    _table(page, 190, [
        ["Description", "Qty", "Unit Price", "Amount"],
        ["Cloud Infrastructure Services", "1", "45,000", "45,000"],
        ["AI Model Development", "1", "32,500", "32,500"],
        ["Technical Support", "2", "3,500", "7,000"],
    ], widths=[220, 50, 90, 90])

    page = doc.new_page(width=612, height=792)
    _header(page, "INVOICE", "TechNova Solutions Pvt. Ltd.", "Page 2")
    _text(page, 72, 120, "Service Details", HEADING, bold=True)
    for index, line in enumerate([
        "- Cloud Infrastructure Services covering AWS compute and storage.",
        "- AI Model Development including training and evaluation.",
        "- Technical Support during business hours.",
    ]):
        _text(page, 72, 150 + index * 18, line, BODY)

    page = doc.new_page(width=612, height=792)
    _header(page, "INVOICE", "TechNova Solutions Pvt. Ltd.", "Page 3")
    _text(page, 72, 120, "Notes and Terms", HEADING, bold=True)
    for index, line in enumerate([
        "1. Services are provided according to the agreed statement of work.",
        "2. Late payments may attract interest at 1.5% per month.",
    ]):
        _text(page, 72, 150 + index * 18, line, BODY)
    _text(page, 72, 210, "Subtotal: 84,500", BODY)
    _text(page, 72, 228, "GST (18%): 15,210", BODY)
    _text(page, 72, 246, "Total: 99,710", BODY)
    truth.append({
        "pages": [1, 2, 3], "doc_type": "invoice", "title": "INVOICE",
        "headings": ["Billing Details", "Service Details", "Notes and Terms"],
        "tables": [{"page": 1, "rows": 4, "columns": 4}],
        "lists": [{"page": 2, "items": 3}, {"page": 3, "items": 2}],
        "fields": {"invoice_number": "INV-2026-004471", "total": "99,710", "subtotal": "84,500"},
    })

    # ---- Document B: resume (pages 4-5) ----
    page = doc.new_page(width=612, height=792)
    _header(page, "RESUME", "Arjun Mehta - AI / Machine Learning Engineer", "Page 1")
    _text(page, 72, 120, "Professional Summary", HEADING, bold=True)
    _text(page, 72, 145, "AI Engineer with experience building machine learning, NLP and", BODY)
    _text(page, 72, 162, "document understanding pipelines in production environments.", BODY)
    _text(page, 72, 185, "Contact: arjun.mehta@example.com", BODY)

    page = doc.new_page(width=612, height=792)
    _header(page, "RESUME", "Arjun Mehta - AI / Machine Learning Engineer", "Page 2")
    _text(page, 72, 120, "Technical Skills", HEADING, bold=True)
    for index, line in enumerate([
        "- Python, SQL, C++, Git, Docker",
        "- Machine Learning, NLP, Computer Vision",
        "- Retrieval systems and vector databases",
        "- Distributed training and model evaluation",
    ]):
        _text(page, 72, 150 + index * 18, line, BODY)
    truth.append({
        "pages": [4, 5], "doc_type": "resume", "title": "RESUME",
        "headings": ["Professional Summary", "Technical Skills"],
        "tables": [], "lists": [{"page": 5, "items": 4}],
        "fields": {"email": "arjun.mehta@example.com"},
    })

    # ---- Document C: passport / travel document (pages 6-7) ----
    page = doc.new_page(width=612, height=792)
    _header(page, "TRAVEL DOCUMENT", "Identity and Passport Information", "Page 1")
    _text(page, 72, 120, "Holder Information", HEADING, bold=True)
    for index, line in enumerate([
        "Passport No: P9876543",
        "Surname: SHARMA",
        "Given Names: ROHAN",
        "Nationality: IND",
        "Date of Birth: 14 March 1992",
        "Date of Expiry: 21/06/2032",
    ]):
        _text(page, 72, 145 + index * 17, line, BODY)

    page = doc.new_page(width=612, height=792)
    _header(page, "TRAVEL DOCUMENT", "Identity and Passport Information", "Page 2")
    _text(page, 72, 120, "Document Notes", HEADING, bold=True)
    for index, line in enumerate([
        "- Issuing authority: Regional Passport Office, Pune.",
        "- This page contains continuation information only.",
    ]):
        _text(page, 72, 150 + index * 18, line, BODY)
    truth.append({
        "pages": [6, 7], "doc_type": "passport", "title": "TRAVEL DOCUMENT",
        "headings": ["Holder Information", "Document Notes"],
        "tables": [], "lists": [{"page": 7, "items": 2}],
        "fields": {"passport_number": "P9876543", "surname": "SHARMA", "nationality": "IND"},
    })

    # ---- Document D: bank statement (pages 8-9), second ruled table ----
    page = doc.new_page(width=612, height=792)
    _header(page, "BANK STATEMENT", "SampleBank - Account Statement", "Page 1")
    _text(page, 72, 120, "Account Summary", HEADING, bold=True)
    for index, line in enumerate([
        "Account Holder: Neha Kulkarni",
        "Account Number: XXXXXX4521",
        "Statement Period: 01 August 2026 - 15 August 2026",
        "Opening Balance: 74,000",
    ]):
        _text(page, 72, 145 + index * 17, line, BODY)

    page = doc.new_page(width=612, height=792)
    _header(page, "BANK STATEMENT", "SampleBank - Account Statement", "Page 2")
    _text(page, 72, 120, "Transaction Summary", HEADING, bold=True)
    _table(page, 150, [
        ["Date", "Description", "Debit", "Credit"],
        ["02 Aug", "Salary Credit", "", "65,000"],
        ["07 Aug", "Rent Payment", "15,450", ""],
    ], widths=[70, 220, 80, 80])
    _text(page, 72, 250, "Closing Balance: 1,24,550", BODY)
    truth.append({
        "pages": [8, 9], "doc_type": "bank_statement", "title": "BANK STATEMENT",
        "headings": ["Account Summary", "Transaction Summary"],
        "tables": [{"page": 9, "rows": 3, "columns": 4}], "lists": [],
        "fields": {"closing_balance": "1,24,550", "opening_balance": "74,000", "account_number": "XXXXXX4521"},
    })

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    ground_truth = {"packet": path.name, "page_count": 9, "documents": truth}
    (path.parent / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    return path, ground_truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/samples/sample_packet.pdf")
    args = parser.parse_args()
    path, truth = build_packet(args.output)
    print(f"Wrote {path} with {len(truth['documents'])} documents")
    for document in truth["documents"]:
        print(f"  pages {document['pages']} {document['doc_type']:<15} "
              f"headings={len(document['headings'])} tables={len(document['tables'])} lists={len(document['lists'])}")
    print(f"Wrote ground truth -> {Path(args.output).parent / 'ground_truth.json'}")
